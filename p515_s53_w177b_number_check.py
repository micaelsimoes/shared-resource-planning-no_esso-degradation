"""P5.15 Addendum 76, Planner task W177b-1 -- PARTS 0 AND 1 OF THE W177(b) ORDER AT OVERLEAF 51a6fb5: (0) the W177(a)
findings against the pasted text and every remaining [CONFIRM -- W177] comment; (1) every number of the abstract, the
highlights (highlights.tex and the highlights block of main.tex), section 4.1-4.7 prose, every section 4 table note and
caption, and the Conclusions, against the frozen tables (590088fe) / the W160 export rounding or a named record.
ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the W177a script is IMPORTED (it imports W176b -> W174a -> W171a -> W164 -> ...
-> W160, each arming its guards) and its context (Ctx: frozen JSON, records, cost workbook), its token evaluator
(`evaluate`: the export rounding rules of export/README.md), its spec constructors and its CONFIRM checks
(`confirm_salvage`, `confirm_checks`, which read records only) are USED; nothing earlier is edited. Every output is a
new file opened 'x' in a new directory.

SCOPES (main.tex at 51a6fb5, sha256 61fe630c...; line ranges computed from the section markers and checked against the
constants below, pinned to that sha256):
  in scope (Addendum 76 Part 1): abstract, highlights (main.tex block + highlights.tex), section 4.1-4.7 (prose, table
  bodies, table notes, captions, the comments inside them), Conclusions;
  reported separately (section 4 but outside "4.1-4.7"): the section 4 preamble (salvage equation) and section 4.8
  (Limitations). Other scopes of main.tex and the letter are NOT carried from W176b here (W176b's results stand at
  8b76423; their re-location at 51a6fb5 is not part of this task).

DECLARATIONS. One per token, by (file, line) in token order (the W177a mechanism: the written text of every token must
equal the declaration, else the declaration is stale and the run exits 3; every in-scope token must be assigned, else
exit 3; a token claimed twice exits 3). Section 4 lines whose text is IDENTICAL (whitespace-normalized) to a line of
the frame `section4_expert_draft.tex` CARRY W177a's declaration of that frame line unchanged (the identity is checked
at run time). New kinds beside W177a's:
  derived   a number computed from frozen values (the arithmetic is recorded); holds iff the computed value, rounded
            to the written precision, equals the written value (and every consistency condition of the arithmetic
            holds); a derived number that does not hold is a MISMATCH;
  compound  a hyphenated number word ("sixty-one") split by the tokenizer: both parts are assigned and checked as one.
Statuses: match (frozen JSON / named record at the export's printed rounding) / approximate (printed coarser than that
rule, equal at its own precision) / derived / MISMATCH / no counterpart / unchecked (category) / excluded (structure,
identifier).

PART 0. For each W177a finding (P5_15_ADDENDUM73_W177A_REPORT.md: Blocked items 1-7 and the CONFIRM table C1-C9):
still present / fixed (the new text quoted) / moved / no longer applicable / holds, each decided by a text check on
main.tex at the HEAD (line numbers and quotes recorded); the W177a CONFIRM checks are re-run on the records and their
outcomes compared with W177a's committed JSON; every [CONFIRM -- W177] comment still in main.tex is re-checked.

RECORDED PREDICTIONS (expert, Addendum 76): Part 1 -- zero MISMATCH in numbers carried from the tables; the derived
numbers hold. Scored in the outputs.

OBJECTIVE CONVENTION. Q = gross_operational_cost, settlement excluded (the frozen primary); net = Q - terminal salvage
credit; value = Q(0) - Q(x); F = Q + I.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W177a chain arms; `pickle.load` / `pickle.loads` blocked for the whole run, every counter
verified at 0. git reads only (`git show` / `git rev-parse` / `git ls-files` / `git status` / `git log` /
`git hash-object` without -w), also in the manuscript clone, which is never pulled or modified.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w177b_number_check && \\
    mkdir data/SRP1/Results/P515S53/w177b_number_check/overleaf_51a6fb5 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b_number_check.py \\
        --overleaf-commit 51a6fb5 --expect main.tex=61fe630c... --expect highlights.tex=001352ee... \\
        --expect response_to_reviewers_draft.tex=334e74ea... --expect cover_letter.tex=363ad821... \\
        --expect section2_expert_draft.tex=7e347dc8... \\
        > data/SRP1/Results/P515S53/w177b_number_check/overleaf_51a6fb5/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w177b_number_check/overleaf_51a6fb5/w177b_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w177b_number_check/overleaf_51a6fb5/w177b_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b_number_check.py \\
        --overleaf-commit 51a6fb5 --post-run
(the --expect values are the full sha256 of every *.tex of the clone). Exit: 0 = written, every integrity check holds
and the guards are at 0 (MISMATCHES and Part 0 findings are FINDINGS and never change the exit code); 3 = written, an
integrity check failed (listed); 1 = precondition or guard fault (nothing written).
"""
import argparse
import collections
import difflib
import json
import math
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W177b-1 number check at Overleaf 51a6fb5 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W177b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W177b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w177a_section4_frame_check as W177A  # noqa: E402 -- arms its guard, imports W176b -> W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W176, W174, W171, W164, W160 = W177A.W176, W177A.W174, W177A.W171, W177A.W164, W177A.W160
GUARDS = W160.W157._dedupe((('w177b_number_check', GUARD),) + tuple(W177A.GUARDS))
_log, _sha, _sha_bytes, _git = W160._log, W160._sha, W160._sha_bytes, W160._git
S, U, P, W3, DEF, D = W177A.S, W177A.U, W177A.P, W177A.W3, W177A.DEF, W177A.D
NOT, SINGLE, LANG, XREF = W177A.NOT, W177A.SINGLE, W177A.LANG, W177A.XREF
_cl, _t10, _fmt, _dp, _norm, _json, _text = W177A._cl, W177A._t10, W177A._fmt, W177A._dp, W177A._norm, W177A._json, W177A._text
_arm, _retention, _same, _same_vals, _range = W177A._arm, W177A._retention, W177A._same, W177A._same_vals, W177A._range
_min_max_aged, _w173_year_dims = W177A._min_max_aged, W177A._w173_year_dims

SCRIPT_REL = os.path.basename(__file__)
S53 = W160.S53
OUT_DIR = os.path.join(S53, 'w177b_number_check', 'overleaf_51a6fb5')
OUT_NAMES = {'json': 'w177b_number_check.json', 'md': 'w177b_number_check.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'typing_json': 'w177b_bool_typing_test.json', 'typing_log': 'w177b_bool_typing_test.log',
             'post': 'manifest_post_run_sha256.json'}

CLONE = W164.DEFAULT_MANUSCRIPT_DIR                       # manuscript/6a67305f25e8348fb71380c3 (read only)
DECLARED_COMMIT = '51a6fb5'
CLONE_HEAD = '51a6fb5b5f238dd62f6f1fe54ec4f3e5617f767b'
PREV_HEAD = '6ff8f3c'                                     # the order's minimum HEAD (Addendum 76); 51a6fb5 = 6ff8f3c - 19 lines
TEX_SHA = {
    'main.tex': '61fe630cc427d622a855e127f33f96d4be1a7362fe23552b87d2d54318a39346',
    'highlights.tex': '001352eebb3ac647f8fe1f5c2211025816ca5143c74c02a01af731630a8fea08',
    'response_to_reviewers_draft.tex': '334e74ea88924abc56426e7eee73f82c0bbc01b7a265fe5027aa4c09cccb6578',
    'cover_letter.tex': '363ad82139c8bbf1873e64c8984934d4e049a67af593fa90936a54f3601d7508',
    'section2_expert_draft.tex': '7e347dc830d1d9a8f0c6892a963a534d63d90846c6508108821e9cfac9bcabd1',
}
MAIN_LINES_N = 2312
FRAME = W177A.FRAME                                       # section4_expert_draft.tex (4f941b29, e73b6d91)
FZ_REL, FZ_SHA = W164.FZ_REL, W164.FZ_SHA

# scope line ranges of main.tex at 51a6fb5 (computed at run time from the markers; must equal these)
EXPECTED_SCOPES = collections.OrderedDict([
    ('abstract', (138, 143)), ('highlights (main.tex block)', (155, 161)), ('4 preamble', (1144, 1171)),
    ('4.1', (1172, 1222)), ('4.2', (1223, 1314)), ('4.3', (1315, 1363)), ('4.4', (1364, 1402)), ('4.5', (1403, 1438)),
    ('4.6', (1439, 1495)), ('4.7', (1496, 1526)), ('4.8', (1527, 1541)), ('Conclusions', (1545, 1563))])
SUBSEC_LABELS = {'4.1': 'subsec:res_baseline', '4.2': 'subsec:res_breakeven', '4.3': 'subsec:res_degradation',
                 '4.4': 'subsec:res_coordination', '4.5': 'subsec:res_multiscenario', '4.6': 'subsec:res_sensitivities',
                 '4.7': 'subsec:res_certification', '4.8': 'subsec:res_limitations'}
IN_SCOPE = ('abstract', 'highlights (main.tex block)', 'highlights.tex', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7',
            'Conclusions')
SEPARATE = ('4 preamble', '4.8')

W177A_JSON = os.path.join(S53, 'w177a_section4_frame', 'w177a_section4_frame_check.json')
EXTRA_INPUTS = {
    'W177A_SCRIPT': 'p515_s53_w177a_section4_frame_check.py',
    'W177A_JSON': W177A_JSON,
    'W177A_REPORT': 'P5_15_ADDENDUM73_W177A_REPORT.md',
    'W176_SCRIPT': 'p515_s53_w176_manuscript_number_check.py',
    'CHANGE_LOG': os.path.join('manuscript_review', 'W177B_CHANGE_LOG.md'),
    'SRP1_PARAMS': W164.SRP1_PARAMS,
    'SPEC_3X3': W171.SPEC_3X3,
}
EXTRA_JSON = ('W177A_JSON', 'SRP1_PARAMS', 'SPEC_3X3')
STATUSES = ('match', 'approximate', 'derived', 'MISMATCH', 'no counterpart', 'unchecked', 'excluded')

UNIT = 'CHECK:headline_V_minus_I_settled'                 # the reference unit (node 7, 0.25 MVA / 1 MWh, 2025), value form
REF4 = 'B:n7_4h_e1'                                       # the same plan, F form (cost excess)
C5 = 'C:y2025__n5_p0.25_e0.5'
C59 = 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5'
C57 = 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5'
LF2 = 'L:y2030__n5_p0.25_e0.5__n7_p1_e3'
G = ('G:n7_2h_e1_y2030:gross', 'G:n7_2h_e1_y2035:gross', 'G:n7_4h_e2_y2030:gross', 'G:n7_4h_e2_y2035:gross')
BM = 'benchmark'
AF = 'benchmark.w160_additions.arms_in_full'
YL = 'year_ladder'
WC = ('rec', 'W153_CERT')
M15 = ('rec', 'FIG2_DATA', 'values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value')
M175 = ('rec', 'FIG2_DATA', 'values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value')
M2 = ('rec', 'FIG2_DATA', 'values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value')

# main.tex lines whose whitespace-normalized text equals a frame line: W177a's declaration of that frame line is carried
CARRY = {1148: 25, 1154: 31, 1160: 37, 1165: 42, 1167: 44, 1168: 45, 1390: 305, 1417: 353, 1425: 361, 1426: 362, 1427: 363,
         1428: 364, 1429: 365, 1431: 367, 1432: 368, 1433: 369, 1501: 437, 1505: 441, 1511: 447, 1519: 453, 1523: 457,
         1531: 465, 1535: 469, 1539: 473}


# ======================================================================================================================
#  1. new spec kinds
# ======================================================================================================================
def DER(w, fn, what):
    """a derived number: fn(X) -> {'value': in the written unit, 'arithmetic': str, 'consistent': bool (optional)}."""
    return dict(w=w, kind='derived', fn=fn, desc=what)


def COMP(w, tens, units, src, desc):
    """a hyphenated number word split by the tokenizer: the tens part (w) or the units part; both evaluated as tens+units."""
    return dict(w=w, kind='compound', tens=tens, units=units, src=src, desc=desc)


def SRC(X, src):
    return W177A._source_value(X, {'src': src, 'desc': None})[0]


def _k(x):
    return f'{x / 1e3:,.2f} kEUR'


def der_108(X):
    n9 = len(X.j['CASE9']['nodes'])
    n33 = [len(X.j[k]['nodes']) for k in ('CASE33_1', 'CASE33_2', 'CASE33_3')]
    return {'value': n9 + sum(n33), 'arithmetic': f"case9_2025.json nodes {n9} + case33_1/2/3_2025.json nodes {n33} = "
            f"{n9 + sum(n33)}"}


def der_breakeven_pct(which):
    def f(X):
        be = X.t['break_even']
        e = be['e_cost_eur_per_MWh']
        lo_hi = be['conservative_A64']['breakeven_range']
        m = be['conservative_A64']['margin_to_cost_range']
        estar = lo_hi[1] if which == 0 else lo_hi[0]
        v = 100.0 * (1.0 - estar / e)
        v2 = 100.0 * m[which] / e
        return {'value': v, 'consistent': abs(v - v2) < 1e-9,
                'arithmetic': f"1 - e*/e_cost = 1 - {estar:,.2f} / {e:,.2f} = {m[which]:,.2f} / {e:,.2f} = {v:.4f} % "
                              f"(T3 break_even.conservative_A64.{'breakeven_range[1]' if which == 0 else 'breakeven_range[0]'}, "
                              f"margin_to_cost_range[{which}], e_cost_eur_per_MWh)"}
    return f


def der_one_minus_r(row, label):
    def f(X):
        r = X.fzp(f'three_by_three.rows[{row}].value')
        return {'value': 100.0 * (1.0 - r), 'arithmetic': f'1 - R({label}) = 1 - {r:.6f} = {100 * (1 - r):.4f} % (T11 '
                                                          f'three_by_three.rows[{row}].value)'}
    return f


def der_meur(path, label):
    def f(X):
        v = X.fzp(path)
        return {'value': v / 1e6, 'arithmetic': f'{label} {v:,.2f} EUR = {v / 1e3:,.1f} kEUR = {v / 1e6:.4f} MEUR ({FZ_REL} '
                                                f'tables.{path})'}
    return f


def der_vs_c3(arm):
    claim = {'C2_calfade': 'E:n7_4h_e1_C2_calfade:vs_C3', 'C2': 'E:n7_4h_e1_C2:vs_C3', 'C4': 'E:n7_4h_e1_C4:vs_C3'}[arm]

    def f(X):
        c = X.CL[claim]
        va, vc = _arm(X, arm)['value'], _arm(X, 'C3_unit')['value']
        d8 = va - vc
        r_a, r_c = round(va / 1e3, 1), round(vc / 1e3, 1)
        return {'value': c['d_gross'] / 1e3, 'consistent': abs(d8 - c['d_gross']) < 0.01,
                'arithmetic': f"T8 value({arm}) {va:,.2f} - value(C3_unit) {vc:,.2f} = {d8:,.2f} EUR = T1 {claim} d_gross "
                              f"{c['d_gross']:,.2f} EUR ({c['gross_verdict']}, {c['gross_multiple']:.2f} x); from the printed "
                              f"T8 values {r_a:.1f} - {r_c:.1f} = {r_a - r_c:.1f}"}
    return f


def der_six(X):
    c = X.w173['item2']['F2']['c_L_claims_box_neighbours']
    box = [k for k in X.CL if k.startswith('L:') and 'not_a_neighbour' not in k]
    n_det = sum(1 for k in box if X.CL[k]['gross_verdict'] == 'determinate' and X.CL[k]['d_gross'] > 0)
    n_rest = len(box) - n_det
    v = c['n'] - c['n_positive_determinate']
    return {'value': v, 'consistent': v == c['n_positive_within_bar'] + c['n_negative_neighbour_better'] == n_rest and
            len(box) == c['n'] and n_det == c['n_positive_determinate'],
            'arithmetic': f"W173 F2 L neighbours {c['n']} - determinate {c['n_positive_determinate']} = {v} (= within the bar, "
                          f"plan better {c['n_positive_within_bar']} + neighbour better within the bar "
                          f"{c['n_negative_neighbour_better']}); frozen T1 L box-neighbour rows {len(box)}, determinate and "
                          f"positive {n_det}, the rest {n_rest}"}


def der_net_range(which):
    def f(X):
        nets = {'2025': -X.CL[UNIT]['d_net'], '2030': X.fzp(f'{YL}.per_year.2030.M_net_w153_form'),
                '2035': X.fzp(f'{YL}.per_year.2035.M_net_w153_form')}
        v = (min if which == 'min' else max)(nets.values())
        return {'value': v / 1e3, 'arithmetic': f"{which} of M net over the investment years {{2025: {nets['2025']:,.2f} "
                                                f"(T1 {UNIT} d_net, sign M = -(value - I)), 2030: {nets['2030']:,.2f}, 2035: "
                                                f"{nets['2035']:,.2f}}} EUR (T4) = {v:,.2f} EUR"}
    return f


def _base_plans(X):
    """the evaluated base-price plans other than the reference unit: the other cells of the T1 B / C / G rows."""
    cells = sorted({X.CL[k]['other_cell'] for k in X.CL if k.startswith(('B:', 'C:', 'G:'))} - {X.CL[REF4]['other_cell']})
    return {c: {n: v for n, v in X.cand(c)['nodes'].items() if v[0] or v[1]} for c in cells}


def _e_steps(X):
    es = [X.cand(X.other(c))['nodes']['7'][1] for c in ('H:m2:value_minus_I', 'I:m2:e2_value_minus_I', 'J:e3_value_minus_I',
                                                       'J:e4_value_minus_I', 'J:e5_value_minus_I')]
    return _same_vals([b - a for a, b in zip(es, es[1:])])


def _incr(X, which):
    vals = [X.CL[c]['d_gross'] for c in ('I:m2:second_MWh', 'J:e2_to_e3', 'J:e3_to_e4', 'J:e4_to_e5')]
    return (min if which == 'min' else max)(vals)


def _h_positive_determinate(X):
    rows = [X.A64['H:m1.75:value_minus_I'], X.CL['H:m2:value_minus_I'], X.CL['H:m1.5:value_minus_I']]
    return sum(1 for r in rows if r['d_gross'] > 0 and r['gross_verdict'] == 'determinate')


def _unc_three(X):
    return sum(1 for c in ('J:e4_value_minus_I', 'J:e3_to_e4', 'J:e4_to_e5')
               if X.CL[c]['uncertified_form_beside_A64_ruling4']['multiple_Q'] >= 1.0)


def _stop_reasons(X, kind):
    it = X.w173['item1']
    if kind == 'poll':
        return sum(1 for k in it if 'unit_poll_failed' in it[k]['termination']['reason'] or
                   'poll_failure_at_unit_mesh' in it[k]['termination']['reason'])
    return sum(1 for k in it if 'completion_cap' in it[k]['termination']['reason'])


def _twelve(X):
    return len(X.j['SRP1_JSON']['Years']) * len(X.j['SRP1_JSON']['Days'])


# ======================================================================================================================
#  2. declarations (main.tex at 51a6fb5, highlights.tex)
# ======================================================================================================================
def ladder_row(line, claim, nodes, year, w):
    """tab:res_value_ladder at 51a6fb5: node(s) [, year], S, E, duration, F(x) - F(0), threshold or bar, multiple."""
    sp = [P(str(n), 'node', claim, [n]) for n in nodes]
    if year != 2025:
        sp.append(P(str(year), 'year', claim))
    sp += [P(w[0], 'S', claim, nodes), P(w[1], 'E', claim, nodes), P(w[2], 'dur', claim, nodes),
           S(w[3], 'k', _cl(claim, 'd_gross'), signed=True), S(w[4], 'k', _cl(claim, 'gross_threshold_or_bar')),
           S(w[5], 'x', _cl(claim, 'gross_multiple'))]
    return D(line, None, *sp)


def declarations():
    L = []
    M = 'main.tex'

    def add(d, file_=M):
        d['file'] = file_
        L.append(d)

    # ---- abstract -------------------------------------------------------------------------------------------------------
    add(D(141, None, U('two', 'compound adjective', '"two-stage" (the model class, not a count)')))
    add(D(142, None, S('15', 'int', lambda X: sum(X.j['SRP1_JSON']['Years'].values()), 'SRP1.json Years: sum of the block lengths'),
          DER('108', der_108, 'buses of the integrated system'), S('13.9', 'pct', f'{BM}.benefit_relative'),
          P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), S('64', 'k', _cl(REF4, 'd_gross')),
          S('1.75', 'raw', M175, 'fig2 data (W165): the multiplier of the m = 1.75 cells'),
          S('171', 'kmwh', 'break_even.conservative_A64.breakeven_range[0]'),
          S('193', 'kmwh', 'break_even.conservative_A64.breakeven_range[1]'),
          S('254', 'kmwh', 'break_even.e_cost_eur_per_MWh'),
          S('3', 'int', ('rec', 'CASE_3X3', 'NumMarketScenarios'), 'market scenarios of the 3 x 3 instance'),
          S('3', 'int', ('rec', 'CASE_3X3', 'TransmissionNetwork.num_operation_scenarios'), 'operation scenarios of the 3 x 3 instance'),
          S('0.91', 'r3', 'three_by_three.rows[8].value'), S('0.93', 'r3', 'three_by_three.rows[7].value'), SINGLE('single')))
    # ---- highlights ------------------------------------------------------------------------------------------------------
    add(D(160, None, S('13.9', 'pct', f'{BM}.benefit_relative')))
    add(D(14, None, S('13.9', 'pct', f'{BM}.benefit_relative')), 'highlights.tex')
    # ---- 4.1 -------------------------------------------------------------------------------------------------------------
    add(D(1176, None, NOT('0'), S('fourteen', 'int', ('rec', 'W173', 'item2.x0.b_box.n_feasible'),
                                  'W173: admissible unit neighbours of x = 0 (n_evaluated 14, n_not_evaluated 0)')))
    add(D(1177, None, P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('7', 'node', UNIT, [7]), P('2025', 'year', UNIT),
          S('253.5', 'k', 'ageing.rows[C2_calfade].value'), S('318.0', 'k', _cl(UNIT, 'I_other')),
          S('64.4', 'k', _cl(REF4, 'd_gross')), S('4.93', 'x', _cl(REF4, 'gross_multiple')),
          P('5', 'node', C5, [5]), P('0.25', 'S', C5, [5]), P('0.5', 'E', C5, [5]), S('49.7', 'k', _cl(C5, 'd_gross'))))
    add(D(1182, None, SINGLE('single'), P('2025', 'year', REF4)))
    add(D(1187, None, NOT('0')))
    LADDER = ((1189, C5, [5], 2025, ('0.25', '0.5', '2', '49.7', '12.6', '3.94')),
              (1190, 'B:n5_4h_e1', [5], 2025, ('0.25', '1', '4', '58.0', '13.2', '4.39')),
              (1191, 'B:n7_4h_e1', [7], 2025, ('0.25', '1', '4', '64.4', '13.1', '4.93')),
              (1192, 'B:n7_4h_e2', [7], 2025, ('0.5', '2', '4', '135.5', '13.4', '10.13')),
              (1193, 'B:n7_4h_e3', [7], 2025, ('0.75', '3', '4', '208.3', '12.6', '16.49')),
              (1194, 'B:n7_4h_e4', [7], 2025, ('1', '4', '4', '274.0', '13.6', '20.19')),
              (1195, 'B:n7_4h_e5', [7], 2025, ('1.25', '5', '4', '347.2', '23.9', '14.52')),
              (1196, 'B:n7_2h_e1', [7], 2025, ('0.5', '1', '2', '114.1', '12.6', '9.03')),
              (1197, 'B:n7_2h_e2', [7], 2025, ('1', '2', '2', '231.7', '24.2', '9.56')),
              (1198, 'B:n7_2h_e3', [7], 2025, ('1.5', '3', '2', '361.7', '41.4', '8.74')),
              (1199, 'B:n7_2h_e4', [7], 2025, ('2', '4', '2', '494.4', '13.4', '36.91')),
              (1200, 'B:n7_2h_e5', [7], 2025, ('2.5', '5', '2', '616.6', '13.4', '45.94')),
              (1201, 'B:n9_4h_e1', [9], 2025, ('0.25', '1', '4', '61.1', '12.6', '4.84')),
              (1202, 'B:n9_4h_e3', [9], 2025, ('0.75', '3', '4', '199.1', '13.5', '14.78')),
              (1203, C59, [5, 9], 2025, ('0.25', '0.5', '2', '102.1', '12.6', '8.09')),
              (1204, C57, [5, 7], 2030, ('0.25', '0.5', '2', '93.9', '13.4', '7.00')))
    for ln, claim, nodes, year, w in LADDER:
        add(ladder_row(ln, claim, nodes, year, w))
    add(D(1213, None, NOT('0'), S('70.7', 'k', _cl('B:n7_4h_e3', 'uncertified_form_beside_A64_ruling4.bar')),
          S('2.94', 'x', _cl('B:n7_4h_e3', 'uncertified_form_beside_A64_ruling4.multiple_Q'))))
    add(D(1219, None, P('7', 'node', REF4, [7]), S('64.4', 'k', _cl(REF4, 'd_gross')), S('347.2', 'k', _cl('B:n7_4h_e5', 'd_gross')),
          P('1', 'E', REF4, [7]), P('5', 'E', 'B:n7_4h_e5', [7]), P('4', 'dur', 'B:n7_4h_e5', [7]),
          S('114.1', 'k', _cl('B:n7_2h_e1', 'd_gross')), S('616.6', 'k', _cl('B:n7_2h_e5', 'd_gross')),
          P('2', 'dur', 'B:n7_2h_e5', [7]), P('two', 'nnodes', C59), S('102.1', 'k', _cl(C59, 'd_gross')),
          P('5', 'node', C59, [5]), P('9', 'node', C59, [9]), P('2025', 'year', C59), S('93.9', 'k', _cl(C57, 'd_gross')),
          P('5', 'node', C57, [5]), P('7', 'node', C57, [7]), P('2030', 'year', C57), S('84.0', 'k', _cl(C57, 'd_net'))))
    # ---- 4.2 -------------------------------------------------------------------------------------------------------------
    BEC = DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4')
    add(D(1227, None, P('2', 'dur', 'B:n7_2h_e1', [7]), P('4', 'dur', REF4, [7]), P('7', 'node', REF4, [7]),
          P('4', 'dur', REF4, [7]), dict(BEC), dict(BEC), S('256,320', 'e10', 'break_even.p_cost_eur_per_MVA'),
          S('six', 'int', 'break_even.conservative_A64.certified_only.n'),
          S('183,080', 'e10', 'break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost'),
          S('ten', 'int', 'break_even.conservative_A64.banded_mid.n'),
          S('171,310', 'e10', 'break_even.conservative_A64.breakeven_range[0]'),
          S('192,560', 'e10', 'break_even.conservative_A64.breakeven_range[1]'),
          S('61,320', 'e10', 'break_even.conservative_A64.margin_to_cost_range[0]'),
          S('253,880', 'e10', 'break_even.e_cost_eur_per_MWh'),
          DER('24', der_breakeven_pct(0), 'energy-cost reduction to pay, lower end (upper break-even)'),
          DER('33', der_breakeven_pct(1), 'energy-cost reduction to pay, upper end (lower break-even)'),
          P('4', 'dur', REF4, [7]), P('7', 'node', REF4, [7])))
    add(D(1234, None, P('7', 'node', REF4, [7]), P('2025', 'year', REF4), P('4', 'dur', REF4, [7]),
          P('2', 'dur', 'B:n7_2h_e1', [7]), DEF('2', 'CAPTIONS', 'bar + 2 tau', 'the interval of the banded fit (captions.md, W165)')))
    add(D(1240, None, P('4', 'dur', REF4, [7]), P('7', 'node', REF4, [7]), S('256,320', 'e10', 'break_even.p_cost_eur_per_MVA')))
    add(D(1241, None, S('253,880', 'e10', 'break_even.e_cost_eur_per_MWh')))
    add(D(1249, None, S('4', 'int', lambda X: len(X.t['break_even']['conservative_A64']['uncertified_as_intervals']),
                        'T3 conservative A64: intervals'),
          S('6', 'int', 'break_even.conservative_A64.certified_only.n'),
          S('183,080', 'e10', 'break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost'),
          S('171,310', 'e10', 'break_even.conservative_A64.breakeven_range[0]'),
          S('192,560', 'e10', 'break_even.conservative_A64.breakeven_range[1]'),
          S('61,320', 'e10', 'break_even.conservative_A64.margin_to_cost_range[0]'),
          S('82,570', 'e10', 'break_even.conservative_A64.margin_to_cost_range[1]')))
    add(D(1250, None, S('3', 'int', lambda X: len(X.t['break_even']['committed_W145']['uncertified_as_intervals']),
                        'T3 committed W145: intervals'),
          S('7', 'int', 'break_even.committed_W145.certified_only.n'),
          S('182,700', 'e10', 'break_even.committed_W145.certified_only.breakeven_marginal_4h_energy_cost'),
          S('173,490', 'e10', 'break_even.committed_W145.breakeven_range[0]'),
          S('190,380', 'e10', 'break_even.committed_W145.breakeven_range[1]'),
          S('63,500', 'e10', 'break_even.committed_W145.margin_to_cost_range[0]'),
          S('80,390', 'e10', 'break_even.committed_W145.margin_to_cost_range[1]')))
    add(D(1256, None, S('14.2', 'k', _t10('H:m1.75:value_minus_I', 'd_gross')), S('1.75', 'raw', M175),
          S('46.3', 'k', _cl('H:m2:value_minus_I', 'd_gross')), S('1.5', 'raw', M15), S('1.75', 'raw', M175),
          W3('1.63', 'S1c', 1.6258759476587705, 'linear interpolation of value - I between m = 1.5 and 1.75'),
          S('1.5', 'raw', M15), S('14.5', 'k', _cl('H:m1.5:value_minus_I', 'd_gross')),
          S('18.8', 'k', _cl('H:m1.5:value_minus_I', 'gross_threshold_or_bar')),
          S('two', 'int', _h_positive_determinate, 'T1 / T10 H rows: positive and determinate margins (m = 1.75, 2)')))
    add(D(1262, None, S('1.5', 'raw', M15)))
    FLEX = ((1275, '1.00', UNIT, _cl, 'CHECK:headline_V_minus_I_settled.other_cell.ref:bd504ecf.flex_price_multiplier',
             ('64.4', '13.1', '4.93')),
            (1276, '1.50', 'H:m1.5:value_minus_I', _cl, 'H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier',
             ('14.5', '18.8', '0.77')),
            (1277, '1.75', 'H:m1.75:value_minus_I', _t10, 'H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier',
             ('14.2', '10.8', '1.32')),
            (1278, '2.00', 'H:m2:value_minus_I', _cl, 'H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier',
             ('46.3', '12.3', '3.77')))
    for ln, m, cid, f, key, w in FLEX:
        add(D(ln, None, S(m, 'raw', ('rec', 'FIG2_DATA', f'values[{key}].value'), 'fig2 data: the multiplier'),
              S(w[0], 'k', f(cid, 'd_gross'), signed=True), S(w[1], 'k', f(cid, 'gross_threshold_or_bar')),
              S(w[2], 'x', f(cid, 'gross_multiple'))))
    J5 = 'J:e5_value_minus_I'
    add(D(1284, None, P('4', 'dur', J5, [7]), P('7', 'node', J5, [7]), S('46.3', 'k', _cl('H:m2:value_minus_I', 'd_gross')),
          P('1', 'E', 'H:m2:value_minus_I', [7]), S('228.4', 'k', _cl(J5, 'd_gross')), P('5', 'E', J5, [7]),
          S('1', 'raw', _e_steps, 'the E step between consecutive cells of the m = 2 ladder (1-5 MWh at node 7)'),
          S('41.0', 'k', lambda X: _incr(X, 'min'), 'T1 I/J increments: smallest (J:e2_to_e3)'),
          S('51.8', 'k', lambda X: _incr(X, 'max'), 'T1 I/J increments: largest (J:e4_to_e5)'),
          P('two', 'nnodes', 'ref:5ca4f86c'), P('2030', 'year', 'ref:5ca4f86c'),
          S('seventeen', 'int', ('rec', 'W173', 'item2.F2.b_box.n_evaluated_in_search')),
          COMP('sixty', 60, 1, ('rec', 'W173', 'item2.F2.b_box.n_feasible'), 'W173 F2 admissible unit neighbours (sixty-one)'),
          COMP('one', 60, 1, ('rec', 'W173', 'item2.F2.b_box.n_feasible'), 'the units part of "sixty-one"'),
          S('thirteen', 'int', ('rec', 'W173', 'item2.F2.c_L_claims_box_neighbours.n')),
          S('seven', 'int', ('rec', 'W173', 'item2.F2.c_L_claims_box_neighbours.n_positive_determinate')),
          DER('six', der_six, '"within resolution of the other six"')))
    add(D(1290, None, P('4', 'dur', J5, [7]), P('7', 'node', J5, [7]), P('2025', 'year', J5)))
    EL = ((1297, 'H:m2:value_minus_I', None, ('1', '46.3', '12.3', '3.77')),
          (1298, 'I:m2:e2_value_minus_I', 'I:m2:second_MWh', ('2', '92.8', '12.8', '7.26', '46.5', '12.8', '3.64')),
          (1299, 'J:e3_value_minus_I', 'J:e2_to_e3', ('3', '133.8', '27.7', '4.83', '41.0', '27.7', '1.48')),
          (1300, 'J:e4_value_minus_I', 'J:e3_to_e4', ('4', '176.6', '9.7', '18.14', '42.8', '27.7', '1.55')),
          (1301, 'J:e5_value_minus_I', 'J:e4_to_e5', ('5', '228.4', '26.5', '8.60', '51.8', '26.5', '1.95')))
    for ln, c1, c2, w in EL:
        sp = [P(w[0], 'E', c1, [7]), S(w[1], 'k', _cl(c1, 'd_gross'), signed=True), S(w[2], 'k', _cl(c1, 'gross_threshold_or_bar')),
              S(w[3], 'x', _cl(c1, 'gross_multiple'))]
        if c2:
            sp += [S(w[4], 'k', _cl(c2, 'd_gross'), signed=True), S(w[5], 'k', _cl(c2, 'gross_threshold_or_bar')),
                   S(w[6], 'x', _cl(c2, 'gross_multiple'))]
        add(D(ln, None, *sp))
    add(D(1309, None, S('15.9', 'k', _cl('J:e4_value_minus_I', 'uncertified_form_beside_A64_ruling4.bar')),
          S('11.12', 'x', _cl('J:e4_value_minus_I', 'uncertified_form_beside_A64_ruling4.multiple_Q')),
          P('4', 'E', 'J:e4_value_minus_I', [7]), P('5', 'E', J5, [7]),
          S('27.7', 'k', _cl('J:e3_to_e4', 'uncertified_form_beside_A64_ruling4.bar')),
          S('26.5', 'k', _cl('J:e4_to_e5', 'uncertified_form_beside_A64_ruling4.bar')),
          S('1.55', 'x', _cl('J:e3_to_e4', 'uncertified_form_beside_A64_ruling4.multiple_Q')),
          S('1.95', 'x', _cl('J:e4_to_e5', 'uncertified_form_beside_A64_ruling4.multiple_Q')),
          S('three', 'int', _unc_three, 'the three comparisons of the note, each >= 1 x in the uncertified form')))
    # ---- 4.3 -------------------------------------------------------------------------------------------------------------
    add(D(1319, None, P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('7', 'node', UNIT, [7]), P('2025', 'year', UNIT),
          S('4.1', 'k', 'ageing.rows[no_ageing].value_minus_I'), S('31.6', 'k', 'ageing.rows[C2].value_minus_I'),
          S('64.4', 'k', 'ageing.rows[C2_calfade].value_minus_I'),
          S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh'), 'SRP1_ESS_Params.json minimum_soh'),
          S('31.6', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'max'), 'T8 aged arms: smallest loss (C2)'),
          S('73.7', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'min'), 'T8 aged arms: largest loss (C3_unit)')))
    add(D(1320, None, LANG('one', '"from one another"'),
          S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit: R = exp(-N D / k), N D = 8,000'),
          DER('9.3', der_vs_c3('C2_calfade'), 'baseline - retention 50 % (O3)'),
          DER('42.1', der_vs_c3('C2'), 'cycling only - retention 50 % (O3)'),
          DER('28.5', der_vs_c3('C4'), 'datasheet - retention 50 % (O3)')))
    add(D(1326, None, S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh'))))
    add(D(1332, None, P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('7', 'node', UNIT, [7]), P('2025', 'year', UNIT)))
    AG = ((1339, 'C2_calfade', 'E:n7_4h_e1_C2_calfade:value_minus_I', None, ('253.5', '64.4', '13.1', '4.93', '2035', '0.793', '0.86')),
          (1340, 'C2', 'E:n7_4h_e1_C2:value_minus_I', None, ('286.3', '31.6', '12.6', '2.50', None, '0.888', '1.19')),
          (1341, 'C3_unit', 'E:C3_unit_value_minus_I', '50', ('244.2', '73.7', '12.8', '5.74', '2035', '0.786', '0.76')),
          (1342, 'C3_midblock', 'E:n7_4h_e1_C3_midblock:value_minus_I', '50', ('252.1', '65.9', '12.9', '5.12', '2035', '0.836', '0.76')),
          (1343, 'C4', 'E:n7_4h_e1_C4:value_minus_I', '8,000', ('272.8', '45.2', '12.6', '3.58', None, '0.834', '1.13')),
          (1344, 'no_ageing', 'E:n7_4h_e1_no_ageing:value_minus_I', None, ('313.8', '4.1', '12.9', '0.32', None, '1.000', '1.29')))
    for ln, arm, claim, lead, w in AG:
        sp = []
        if lead == '50':
            sp.append(S('50', 'pct0', (lambda a: (lambda X: _retention(X, a)))(arm), f'T8 k of {arm}: R = exp(-N D / k)'))
        elif lead == '8,000':
            sp.append(S('8,000', 'int', lambda X: round(_arm(X, 'C4')['k'] * -math.log(0.70)), 'T8 k of C4 x (-ln 0.70)'))
        sp += [S(w[0], 'k', f'ageing.rows[{arm}].value'), S(w[1], 'k', f'ageing.rows[{arm}].value_minus_I', signed=True),
               S(w[2], 'k', _cl(claim, 'gross_threshold_or_bar')), S(w[3], 'x', _cl(claim, 'gross_multiple'))]
        if w[4]:
            sp.append(S(w[4], 'int', f'ageing.rows[{arm}].floor_year_070'))
        sp += [S(w[5], 'r3', f'ageing.rows[{arm}].AE'), S(w[6], 'efc', f'ageing.rows[{arm}].EFC')]
        add(D(ln, None, *sp))
    S050 = S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh')
    add(D(1346, None, dict(S050), S('59.5', 'k', _t10('E:soh050:value_minus_I', 'd_gross'), signed=True),
          S('12.6', 'k', _t10('E:soh050:value_minus_I', 'gross_threshold_or_bar')),
          S('4.71', 'x', _t10('E:soh050:value_minus_I', 'gross_multiple'))))
    add(D(1354, None, S('318.0', 'k', 'ageing.rows[C2_calfade].I'), S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')),
          dict(S050)))
    add(D(1360, None, S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')),
          S('2035', 'int', lambda X: _same(X, ('C2_calfade', 'C3_unit', 'C3_midblock'), 'floor_year_070'),
            'T8 floor binds (0.70): baseline, C3_unit, C3_midblock (all 2035; the other three never)'),
          S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit'),
          S('8,000', 'int', lambda X: round(_arm(X, 'C4')['k'] * -math.log(0.70)), 'T8 k of C4 x (-ln 0.70)'),
          S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')), dict(S050),
          W3('0.18', 'N70', 0.17541226610870808, 'EFC/day at 0.50 - at 0.70, 2025'),
          W3('0.20', 'N71', 0.19994545338686776, 'EFC/day at 0.50 - at 0.70, 2030'),
          W3('0.27', 'N72', 0.27253194078611853, 'EFC/day at 0.50 - at 0.70, 2035'),
          *[S(y, 'int', (lambda yy: (lambda X: int(yy) if yy in X.t['a64']['scored']['B']['efc_per_day'] else None))(y),
              'a64.scored.B.efc_per_day: the block-year keys') for y in ('2025', '2030', '2035')],
          S('4.9', 'k', _t10('E:soh050:delta_value_vs_070', 'd_gross')),
          S('59.5', 'k', _t10('E:soh050:value_minus_I', 'd_gross'))))
    # ---- 4.4 -------------------------------------------------------------------------------------------------------------
    add(D(1368, None, S('three', 'int', lambda X: len(X.t['benchmark']['arms']) + 1, 'T6: coordinated + the two NRF arms'),
          S('653,873.7', 'k', f'{BM}.coordinated_Q'), S('744,770.3', 'k', f'{BM}.arms.price_taker.Q_best'),
          S('895,049.9', 'k', f'{BM}.arms.passive.Q_best'),
          S('two', 'int', lambda X: len(X.t['benchmark']['arms']), 'T6: two NRF arms'),
          S('13.9', 'pct', f'{BM}.benefit_relative'),
          S('36.9', 'pct', f'{BM}.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative'),
          S('90,896.6', 'k', f'{BM}.benefit'), S('1,264', 'x', f'{BM}.multiple')))
    add(D(1373, None, S('three', 'int', lambda X: _same_vals([X.fzp(f'{AF}.passive.n_starts'), X.fzp(f'{AF}.price_taker.n_starts')]),
                        'T6 n_starts (both arms)')))
    add(D(1380, None, S('653,873.7', 'k', f'{BM}.coordinated_Q'), S('71.9', 'k', f'{BM}.coordinated_reproducibility_band')))
    add(D(1381, None, S('744,770.3', 'k', f'{BM}.arms.price_taker.Q_best'), S('0.0', 'k', f'{BM}.arms.price_taker.multimodality_band'),
          S('90,896.6', 'k', f'{BM}.decomposition.price_taker_NRF_minus_coordinated_eur'), S('13.9', 'pct', f'{BM}.benefit_relative'),
          S('807.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted'),
          S('0.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted')))
    add(D(1382, None, S('895,049.9', 'k', f'{BM}.arms.passive.Q_best'), S('25.8', 'k', f'{BM}.arms.passive.multimodality_band'),
          S('241,176.2', 'k', f'{BM}.decomposition.passive_NRF_minus_coordinated_eur'),
          S('36.9', 'pct', f'{BM}.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative'),
          S('781.7', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.TSO.mwh_day_weighted'),
          S('201.2', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.DSO.mwh_day_weighted')))
    add(D(1395, None, S('77.2', 'pct', ('rec', 'FIG5_DATA', 'computed[share_TSO].value'), 'W165 fig5 data (W130 totals)'),
          S('22.8', 'pct', ('rec', 'FIG5_DATA', 'computed[share_DSO].value'), 'W165 fig5 data (W130 totals)'),
          S('0.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted'),
          S('807.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted')))
    add(D(1399, None, S('four', 'int', f'{BM}.reverse_flow_interface_hours'),
          S('one', 'int', f'{BM}.sweep.sweep_passive_cold.n_blocks'), S('twelve', 'int', _twelve, 'SRP1.json: 3 years x 4 days = 12 TN blocks'),
          S('eight', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_blocks'), S('twelve', 'int', _twelve, 'SRP1.json: 12 blocks')))
    # ---- 4.5 -------------------------------------------------------------------------------------------------------------
    add(D(1407, None, S('three', 'int', ('rec', 'CASE_3X3', 'NumMarketScenarios')),
          S('three', 'int', ('rec', 'CASE_3X3', 'TransmissionNetwork.num_operation_scenarios')),
          S('five', 'int', lambda X: len(X.j['CASE_3X3']['Years']), '3 x 3 case: Years'),
          S('three', 'int', lambda X: _same_vals(X.j['CASE_3X3']['Years'].values()), '3 x 3 case: block length'),
          NOT('0'), S('236.7', 'k', 'three_by_three.rows[0].value'), S('318.0', 'k', 'three_by_three.rows[1].value'),
          S('81.3', 'k', 'three_by_three.rows[2].value'), S('4.52', 'x', 'three_by_three.rows[4].value'),
          S('18.0', 'k', 'three_by_three.rows[3].value'), SINGLE('single'),
          S('ten', 'int', ('rec', 'SPEC_3X3', 'required_consecutive_cycles'),
            '3 x 3 campaign spec required_consecutive_cycles (= SRP1_params.json minimum_consecutive_converged_cycles)')))
    add(D(1411, None, S('6.2', 'k', 'three_by_three.rows[5].value'), SINGLE('single'),
          S('0.934', 'r3', 'three_by_three.rows[7].value'), S('0.909', 'r3', 'three_by_three.rows[8].value'),
          S('0.9331', 'r4', 'three_by_three.rows[9].value'),
          DER('7', der_one_minus_r(7, 'upper'), 'value lower in the 3 x 3 instance, lower end'),
          DER('9', der_one_minus_r(8, 'lower'), 'value lower in the 3 x 3 instance, upper end')))
    add(D(1412, None, LANG('two', '"the two evaluated instances"')))
    add(D(1424, None, NOT('0'), S('72', 'int', ('rec', 'W98R2_RESULTS', 'replay_gate.N'), 'W98 r2: the 3 x 3 x = 0 stopping cycle N'),
          S('236.7', 'k', 'three_by_three.rows[0].value')))
    add(D(1430, None, S('0.934', 'r3', 'three_by_three.rows[7].value'), SINGLE('single')))
    # ---- 4.6 -------------------------------------------------------------------------------------------------------------
    add(D(1443, None, P('2030', 'year', 'yl_y2030'), S('68.5', 'k', f'{YL}.per_year.2030.M_gross_w153_form'),
          P('2035', 'year', 'yl_y2035'), S('111.0', 'k', f'{YL}.per_year.2035.M_gross_w153_form'),
          S('11.8', 'k', f'{YL}.per_year.2030.salvage_last'), P('2030', 'year', 'yl_y2030'),
          S('56.5', 'k', f'{YL}.per_year.2035.salvage_last'), P('2035', 'year', 'yl_y2035'),
          S('two', 'int', lambda X: len(X.t['year_ladder']['per_year']), 'T4: the two deferral years'),
          S('42.5', 'k', f'{YL}.D_gross'), S('3.46', 'x', f'{YL}.gross_v6.multiple'), S('12.3', 'k', f'{YL}.gross_v6.threshold'),
          S('-2.3', 'k', f'{YL}.D_net'), S('0.19', 'x', f'{YL}.net_v6.multiple'),
          DER('54.5', der_net_range('min'), 'net cost over the investment years, lower end'),
          DER('64.4', der_net_range('max'), 'net cost over the investment years, upper end')))
    add(D(1445, None, P('7', 'node', G[0], [7])))
    add(D(1446, None, P('0.5', 'S', G[0], [7]), P('1', 'E', G[0], [7]), P('2030', 'year', G[0]),
          S('99.0', 'k', _cl(G[0], 'd_gross')), S('88.3', 'k', _cl(G[0], 'd_net')), P('2035', 'year', G[1]),
          S('139.7', 'k', _cl(G[1], 'd_gross')), S('85.1', 'k', _cl(G[1], 'd_net')), P('0.5', 'S', G[2], [7]),
          P('2', 'E', G[2], [7]), P('2030', 'year', G[2]), S('137.4', 'k', _cl(G[2], 'd_gross')), S('113.1', 'k', _cl(G[2], 'd_net')),
          P('2035', 'year', G[3]), S('223.6', 'k', _cl(G[3], 'd_gross')), S('109.1', 'k', _cl(G[3], 'd_net'))))
    add(D(1451, None, P('7', 'node', 'yl_y2030', [7]), P('0.25', 'S', 'yl_y2030', [7]), P('1', 'E', 'yl_y2030', [7]), NOT('0'),
          P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030'), S('3.46', 'x', f'{YL}.gross_v6.multiple'),
          S('12.3', 'k', f'{YL}.gross_v6.threshold'), S('0.19', 'x', f'{YL}.net_v6.multiple')))
    add(D(1458, None, P('2025', 'year', UNIT), S('64.4', 'k', _cl(UNIT, 'd_gross'), note='M = I + Q(x) - Q(0) = -(value - I)'),
          S('0.0', 'k', 'cells[ref:bd504ecf].salvage', note='the 2025 cohort: T^Rem = 0 (W177a CONFIRM C8)'),
          S('64.4', 'k', _cl(UNIT, 'd_net'), note='M net = -(value - I) net')))
    add(D(1459, None, P('2030', 'year', 'yl_y2030'), S('68.5', 'k', f'{YL}.per_year.2030.M_gross_w153_form'),
          S('11.8', 'k', f'{YL}.per_year.2030.salvage_last'), S('56.7', 'k', f'{YL}.per_year.2030.M_net_w153_form')))
    add(D(1460, None, P('2035', 'year', 'yl_y2035'), S('111.0', 'k', f'{YL}.per_year.2035.M_gross_w153_form'),
          S('56.5', 'k', f'{YL}.per_year.2035.salvage_last'), S('54.5', 'k', f'{YL}.per_year.2035.M_net_w153_form')))
    add(D(1461, None, P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030'), S('42.5', 'k', f'{YL}.D_gross', signed=True),
          S('2.3', 'k', f'{YL}.D_net', signed=True)))
    add(D(1467, None, DEF('2025', 'T7_TEX', 'I paid in 2025', 'T7 caption'), S('277.1', 'k', 'discount[0.0].V'),
          S('203.4', 'k', 'discount[0.08].V'), S('8', 'pct0', 'discount[0.08].rate'), S('0', 'pct0', 'discount[0.0].rate'),
          S('8', 'pct0', 'discount[0.08].rate'), S('40.8', 'k', 'discount[0.0].value_minus_I'), S('2.57', 'x', 'discount[0.0].multiple'),
          S('114.6', 'k', 'discount[0.08].value_minus_I'), S('8.78', 'x', 'discount[0.08].multiple')))
    add(D(1472, None, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('2025', 'year', UNIT),
          DEF('one', 'T7_TEX', 'one discount factor per representative year', 'T7 caption'),
          DEF('five', 'T7_TEX', 'applied to the five years of its block', 'T7 caption'),
          DEF('2025', 'T7_TEX', 'I paid in 2025', 'T7 caption')))
    for ln, rate, w in ((1479, '0.0', ('0', '277.1', '318.0', '40.8', '15.9', '2.57')),
                        (1480, '0.02', ('2', '253.5', '318.0', '64.4', '13.1', '4.93')),
                        (1481, '0.05', ('5', '225.2', '318.0', '92.7', '13.1', '7.10')),
                        (1482, '0.08', ('8', '203.4', '318.0', '114.6', '13.1', '8.78'))):
        add(D(ln, None, S(w[0], 'pct0', f'discount[{rate}].rate'), S(w[1], 'k', f'discount[{rate}].V'),
              S(w[2], 'k', f'discount[{rate}].I'), S(w[3], 'k', f'discount[{rate}].value_minus_I', signed=True),
              S(w[4], 'k', f'discount[{rate}].threshold_a61_conservative'), S(w[5], 'x', f'discount[{rate}].multiple')))
    add(D(1488, None, S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims'), 'W153: the comparisons scored gross and net'),
          S('two', 'int', ('rec', 'W153_SALV', 'result.totals.n_verdict_differs'), 'W153: verdicts that change, gross -> net',
            note='as written ("Across the 60 pairwise comparisons ... It changes two verdicts") the count is of the 60: W153 '
                 'records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3). The second verdict the sentence lists, the 2035 - 2030 '
                 'investment-year comparison, is NOT one of the 60 (T4 year_ladder, outside T1 / W153): its gross verdict '
                 'determinate (3.46 x) becomes within resolution net (0.19 x) and its sign changes (D_gross +42,458.65 EUR, '
                 'D_net -2,286.25 EUR). "Two" holds for the 60 + 1 comparisons, not for the 60; the "changes no sign" clause '
                 'then fails for that pair (within resolution net, so its net sign is indeterminate). Part 2 (c), W177b-2'),
          LANG('one', '"one of its unit neighbours"'), S('0.95', 'x', _cl(LF2, 'gross_multiple')),
          S('1.22', 'x', _cl(LF2, 'net_multiple'))))
    add(D(1490, None, S('0', 'int', ('rec', 'W153_SALV', 'result.totals.n_sign_differs')),
          S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims'))))
    add(D(1491, None, XREF('4.6', 'the section 4.6 placeholder (Overleaf cfee0ba)'),
          S('one', 'int', ('rec', 'W153_SALV', 'result.totals.n_verdict_differs'),
            'the placeholder\'s "one verdict changes category" = W153 n_verdict_differs')))
    add(D(1492, None, U('two', 'quotation', 'the letter\'s R3.1 count, quoted in the comment; the letter is outside this scope '
                                       '(the count is Part 2 (c), W177b-2; W153 gives one among the 60, see l. 1488)'),
          P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030')))
    add(D(1493, None, S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims')), LANG('one', '"is one of the 60"'),
          S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims'))))
    # ---- 4.7 -------------------------------------------------------------------------------------------------------------
    add(D(1500, None, SINGLE('single')))
    add(D(1516, None, S('three', 'int', lambda X: len(X.w173['item1']), 'W173 item1: s47, s51, s53'),
          S('41', 'int', lambda X: sum(X.w173['item1'][k]['n_new_evaluations_total'] for k in X.w173['item1']), 'W173 sum'),
          S('14', 'int', ('rec', 'W173', 'item1.s47.n_new_evaluations_total')),
          S('20', 'int', ('rec', 'W173', 'item1.s51.n_new_evaluations_total')),
          S('7', 'int', ('rec', 'W173', 'item1.s53.n_new_evaluations_total')),
          S('20', 'int', ('rec', 'W173', 'item1.s47.max_new_evaluations')),
          S('20', 'int', ('rec', 'W173', 'item1.s51.max_new_evaluations')),
          S('60', 'int', ('rec', 'W173', 'item1.s53.max_new_evaluations'))))
    add(D(1517, None, S('Two', 'int', lambda X: _stop_reasons(X, 'poll'), 'W173 item1 termination: unit poll failed (s47, s53)'),
          S('one', 'int', lambda X: _stop_reasons(X, 'cap'), 'W173 item1 termination: STOP_FOR_REVIEW_completion_cap (s51)'),
          S('two', 'int', lambda X: _stop_reasons(X, 'poll'), '"the first two" (s47, s53)'),
          LANG('third', 'ordinal ("the third")')))
    add(D(1518, None, S('14', 'int', ('rec', 'W173', 'item1.s47.n_new_evaluations_total')),
          S('20', 'int', ('rec', 'W173', 'item1.s47.max_new_evaluations')),
          S('20', 'int', ('rec', 'W173', 'item1.s51.n_new_evaluations_total')),
          S('20', 'int', ('rec', 'W173', 'item1.s51.max_new_evaluations')),
          S('7', 'int', ('rec', 'W173', 'item1.s53.n_new_evaluations_total')),
          S('60', 'int', ('rec', 'W173', 'item1.s53.max_new_evaluations')),
          S('41', 'int', lambda X: sum(X.w173['item1'][k]['n_new_evaluations_total'] for k in X.w173['item1']), 'W173 sum')))
    # ---- Conclusions -----------------------------------------------------------------------------------------------------
    add(D(1549, None, U('two', 'compound adjective', '"two-stage" (the model class)'),
          LANG('single', '"a single, discrete investment plan" (one plan; not the instance name)')))
    add(D(1553, None, DER('108', der_108, 'buses of the integrated system'), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]),
          S('64.4', 'k', _cl(UNIT, 'd_gross')), S('318.0', 'k', _cl(UNIT, 'I_other')),
          S('0.5', 'raw', lambda X: min(v[1] for p in _base_plans(X).values() for v in p.values()),
            'smallest per-node E over the evaluated base-price plans other than the unit (T1 B / C / G other cells)'),
          S('5', 'raw', lambda X: max(v[1] for p in _base_plans(X).values() for v in p.values()),
            'largest per-node E over the same plans'),
          S('one', 'int', lambda X: min(len(p) for p in _base_plans(X).values()), 'fewest nodes over the same plans'),
          S('two', 'int', lambda X: max(len(p) for p in _base_plans(X).values()), 'most nodes over the same plans'),
          S('31.6', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'max'), 'T8 aged arms: smallest loss'),
          S('73.7', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'min'), 'T8 aged arms: largest loss'),
          S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')), dict(S050),
          P('4', 'dur', REF4, [7]), P('7', 'node', REF4, [7]),
          S('171', 'kmwh', 'break_even.conservative_A64.breakeven_range[0]'),
          S('193', 'kmwh', 'break_even.conservative_A64.breakeven_range[1]'),
          S('254', 'kmwh', 'break_even.e_cost_eur_per_MWh'), S('1.75', 'raw', M175), S('2', 'raw', M2), S('2', 'raw', M2),
          P('two', 'nnodes', 'ref:5ca4f86c'), S('0', 'pct0', 'discount[0.0].rate'), S('8', 'pct0', 'discount[0.08].rate'),
          P('2030', 'year', 'yl_y2030'), P('2035', 'year', 'yl_y2035'),
          DER('7', der_one_minus_r(7, 'upper'), 'storage value lower at the 3 x 3 instance, lower end'),
          DER('9', der_one_minus_r(8, 'lower'), 'storage value lower at the 3 x 3 instance, upper end'),
          S('13.9', 'pct', f'{BM}.benefit_relative'), DER('90.9', der_meur(f'{BM}.benefit', 'benefit'), 'value of coordination')))
    add(D(1557, None, LANG('Two', '"Two implications"'),
          W3('15', 'N4', 15, 'x0: 14,970.62 EUR after the residual-rule certificate N = 132'),
          W3('21', 'N5', 21, 'unit: 20,806.32 EUR after N = 112'),
          DER('654', der_meur(f'{BM}.coordinated_Q', 'coordinated Q (x = 0, cycle 181)'), 'operating cost')))
    add(D(1561, None, S('one', 'int', lambda X: _w173_year_dims(X), 'W173 box_definition: one year coordinate zY')))
    return L


def carried_declarations(main_lines, frame_lines):
    """W177a's declaration of each CARRY frame line, re-lined to main.tex, iff the two lines are identical (normalized)."""
    by_line = {d['line']: d for d in W177A.declarations()}
    out, refused = [], []
    for m_ln, f_ln in sorted(CARRY.items()):
        same = ' '.join(main_lines[m_ln - 1].split()) == ' '.join(frame_lines[f_ln - 1].split())
        d = by_line.get(f_ln)
        if not same or d is None:
            refused.append({'main_line': m_ln, 'frame_line': f_ln, 'identical': same, 'w177a_declaration_found': d is not None})
            continue
        nd = dict(d)
        nd['line'], nd['file'] = m_ln, 'main.tex'
        nd['carried_from'] = {'script': EXTRA_INPUTS['W177A_SCRIPT'], 'frame': FRAME, 'frame_line': f_ln}
        out.append(nd)
    return out, refused


# ======================================================================================================================
#  3. evaluation
# ======================================================================================================================
def evaluate(X, spec, t, line):
    kind = spec['kind']
    if kind == 'derived':
        r = spec['fn'](X)
        wv = W177A._written_value(t, line)
        dp = _dp(t['written']) if t['type'] != 'word' else 0
        shown = _fmt(abs(r['value']), dp)
        holds = float(shown) == abs(wv) and r.get('consistent', True) is True
        return {'status': 'derived' if holds else 'MISMATCH', 'derived': True, 'holds': holds,
                'counterpart': 'derived: ' + r['arithmetic'], 'counterpart_value': r['value'],
                'value_at_written_precision': shown, 'rule': f'derived; compared at the written precision ({dp} dp)',
                'consistency_of_the_arithmetic': r.get('consistent', True), 'note': spec.get('desc')}
    if kind == 'compound':
        val = SRC(X, spec['src'])
        written = spec['tens'] + spec['units']
        ok = float(val) == float(written)
        return {'status': 'match' if ok else 'MISMATCH', 'counterpart': W177A._source_value(X, spec)[1],
                'counterpart_value': val, 'value_at_written_precision': str(val),
                'rule': f"compound number word (tens {spec['tens']} + units {spec['units']} = {written}; both parts assigned)",
                'note': spec.get('desc')}
    r = W177A.evaluate(X, spec, t, line)
    if spec.get('note') and 'note' in r and spec['note'] not in str(r.get('note')):
        r['note'] = '; '.join(x for x in (spec['note'], r.get('note')) if x)
    return r


# ======================================================================================================================
#  4. scopes, Part 0, flags
# ======================================================================================================================
def compute_scopes(lines):
    def first(pred, start=0):
        return next(i + 1 for i in range(start, len(lines)) if pred(lines[i]))
    out = collections.OrderedDict()
    out['abstract'] = (first(lambda s: '\\begin{abstract}' in s), first(lambda s: '\\end{abstract}' in s))
    out['highlights (main.tex block)'] = (first(lambda s: '\\begin{highlights}' in s), first(lambda s: '\\end{highlights}' in s))
    res = first(lambda s: s.startswith('\\section{Results}'))
    concl_comment = first(lambda s: 'CONCLUSIONS' in s and s.lstrip().startswith('%'), res)
    banner = concl_comment
    while re.fullmatch(r'%\s*=+\s*', lines[banner - 2]):
        banner -= 1
    subs = [i + 1 for i in range(res, banner - 1) if lines[i].startswith('\\subsection{')]
    out['4 preamble'] = (res, subs[0] - 1)
    names = ['4.%d' % (k + 1) for k in range(len(subs))]
    for k, s in enumerate(subs):
        out[names[k]] = (s, (subs[k + 1] - 1) if k + 1 < len(subs) else banner - 1)
    concl = first(lambda s: s.startswith('\\section{Conclusions}'))
    out['Conclusions'] = (concl, first(lambda s: s.startswith('\\appendix'), concl) - 1)
    labels = {nm: next((re.search(r'\\label\{([^}]+)\}', lines[i]).group(1) for i in range(a - 1, min(a + 2, b))
                        if '\\label{' in lines[i]), None) for nm, (a, b) in out.items() if nm.startswith('4.')}
    return out, labels


def scope_of(file_, ln, scopes):
    if file_ == 'highlights.tex':
        return 'highlights.tex'
    for nm, (a, b) in scopes.items():
        if a <= ln <= b:
            return nm
    return None


def _hits(lines, rx, rng=None, flags=0):
    a, b = rng or (1, len(lines))
    return [{'line': i, 'text': lines[i - 1].strip()[:300]} for i in range(a, b + 1) if re.search(rx, lines[i - 1], flags)]


def _quote(lines, ln, needle, width=220):
    s = lines[ln - 1]
    k = s.find(needle)
    if k < 0:
        return s.strip()[:width]
    return s[max(0, k - 20):k + width].strip()


def part0(X, lines, scopes, confirm_now, w177a_json, toks, prev_main_lines):
    """each W177a finding against main.tex at the HEAD (evidence by text check), and the [CONFIRM -- W177] comments."""
    rows = []
    sec4 = (scopes['4 preamble'][0], scopes['4.8'][1])
    cmap = {c['id']: c for c in confirm_now}
    committed = {c['id']: c['outcome'] for c in w177a_json['confirm']}

    def row(id_, finding, status, evidence, where=None):
        rows.append({'id': id_, 'w177a_finding': finding, 'status': status, 'evidence': evidence, 'where': where})

    # B1 / C1 salvage equation
    eq = _hits(lines, r'\\label\{eq:salvage_credit\}')
    eq_ln = eq[0]['line'] if eq else None
    i_term = _hits(lines, re.escape(r'\frac{I_{e,y^\text{Inv}}}'), sec4)
    soh_term = _hits(lines, re.escape(r'\; SoH_{e,y^\text{Inv},y^\text{End}},'), sec4)
    s_term = _hits(lines, re.escape(r'c^{\text{Inv,S}}_{y^\text{Inv},c} S^{\text{Inv}}_{e,y^\text{Inv}}'), sec4)
    flines = X.frame.split('\n')
    f_eq = next(i + 1 for i, s in enumerate(flines) if '\\label{eq:salvage_credit}' in s)
    off = (eq_ln or 0) - f_eq
    block_m = [' '.join(lines[i - 1].split()) for i in range(eq_ln - 7, eq_ln + 12)] if eq_ln else []
    block_f = [' '.join(flines[i - off - 1].split()) for i in range(eq_ln - 7, eq_ln + 12)] if eq_ln else []
    c1 = cmap['C1']
    still = bool(i_term and soh_term and s_term)
    row('Blocked 1 / C1', 'eq:salvage_credit needs two corrections (energy cost only; health factor (SoH - SoH^Min)/(1 - '
        'SoH^Min)); items (a)-(d) hold', 'still present' if still else 'fixed',
        {'equation_line': eq_ln, 'cohort_cost_with_power_term': s_term, 'factor_I_over_discount': i_term,
         'factor_SoH_unnormalized': soh_term,
         'block_equals_frame_lines': f'main.tex l. {eq_ln - 7}-{eq_ln + 11} = frame l. {eq_ln - 7 - off}-{eq_ln + 11 - off}: '
                                     f'{block_m == block_f}',
         'recomputed_C1': {'outcome': c1['outcome'], 'production_reproduces_records': c1['production_reproduces_records'],
                           'frame_equation_reproduces_records': c1['frame_equation_reproduces_records'],
                           'frame_over_recorded': [round(x['frame_over_recorded'], 4) for x in c1['evidence']['reproduction']],
                           'items': {k: v['outcome'] for k, v in c1['items'].items()}},
         'corrected_text_W177a': c1['corrected_text']}, f'l. {eq_ln - 6}-{eq_ln + 7}' if eq_ln else None)
    # B2 / C4 F2 plan
    s42 = scopes['4.2']
    sizes = _hits(lines, r'1~MVA|3~MWh|3\.5~MWh|1 MVA', s42)
    f2 = _hits(lines, r'two-node investment plan', s42)
    row('Blocked 2 / C4', 'the F2 plan misdescribed: node 7 is 1 MVA / 3.5 MWh, not 3 MWh (frame l. 131)',
        'no longer applicable' if not sizes else 'still present',
        {'plan_sizes_printed_in_4.2': sizes, 'F2_sentence': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'The planning search under this price')} for h in f2],
         'note': 'the pasted text names the plan as "a two-node investment plan installed in 2030" without sizes; the two '
                 'nodes and 2030 match the frozen candidate of ref:5ca4f86c (Part 1, l. ' + (str(f2[0]['line']) if f2 else '?') + ')',
         'recomputed_C4': {'outcome': cmap['C4']['outcome'], 'summary': cmap['C4']['summary']}},
        f"l. {f2[0]['line']}" if f2 else None)
    # B3 J 4 MWh
    j4 = _hits(lines, r'^\s*4\$\^\{f\}\$\s*&', scopes['4.2'])
    j4n = _hits(lines, r'Treating the flagged evaluation as uncertified', scopes['4.2'])
    j4_main = bool(j4) and '9.7' in j4[0]['text'] and '18.14' in j4[0]['text']
    j4_note = bool(j4n) and '15.9' in j4n[0]['text'] and '11.12' in j4n[0]['text']
    row('Blocked 3', 'Addendum 64 ruling 4 (J 4 MWh at its uncertified bar) vs tab:res_energy_ladder_m2: the 4 MWh row prints '
        'the flagged certified threshold 9.7 / 18.14 x; the uncertified bar 15.9 / 11.12 x is beside it',
        'still present' if j4_main and j4_note else ('fixed' if not j4_main else 'moved'),
        {'row': j4, 'note': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'Treating')} for h in j4n],
         'main_columns_certified_threshold': j4_main, 'uncertified_bar_in_note': j4_note,
         'remark': 'the uncertified bar moved from the caption (frame) to the table note; no verdict depends on the choice'},
        f"l. {j4[0]['line']}, note l. {j4n[0]['line']}" if j4 and j4n else None)
    # B4 predictions
    row('Blocked 4', 'two of three Addendum 73 predictions failed (scoring record)', 'no longer applicable',
        {'note': 'a scoring record, not manuscript text; scored in Addendum 74/76 by the Planner'})
    # B5 literal references
    lit = _hits(lines, r'Section~4|Section 4\b|Sections~4|Subsection~4|\(Section~?4')
    refs = [h for h in _hits(lines, r'\\ref\{(sec:results|subsec:res_[a-z]+)\}') if not (sec4[0] <= h['line'] <= sec4[1])]
    row('Blocked 5', 'literal "Section~4.x" references (l. 377, 415, 445, 631, 635, 671, 1066, 1119 at 8b76423)',
        'fixed' if not lit else 'still present',
        {'literal_section_4_references': lit,
         'refs_to_section_4_labels_outside_section_4': [{'line': h['line'], 'refs': re.findall(r'\\ref\{((?:sec:results|subsec:res_[a-z]+))\}', lines[h['line'] - 1])} for h in refs]})
    # B6 author-slot facts
    prose41 = [h for h in _hits(lines, r'4\.39|45\.94', scopes['4.1']) if '&' not in h['text']]
    row('Blocked 6 (l. 58)', '"every evaluated plan ... 4.39 x to 45.94 x" spans the B rows only (table range 3.94-45.94 x)',
        'no longer applicable' if not prose41 else 'still present',
        {'range_in_4.1_prose': prose41, 'note': 'no multiple range is stated in the 4.1 prose; the table prints every row'})
    parts = _hits(lines, r'parts would|more than their parts|than their parts', sec4)
    twonode = _hits(lines, r'two-node plans likewise remain uneconomic', scopes['4.1'])
    row('Blocked 6 (l. 63)', '"Two-node plans lose more than their parts would" is not resolved by the records',
        'fixed' if not parts else 'still present',
        {'old_claim_found': parts, 'new_text': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'The evaluated two-node plans')} for h in twonode]},
        f"l. {twonode[0]['line']}" if twonode else None)
    same = _hits(lines, r'same remaining life|identical .{0,20}net', sec4)
    row('Blocked 6 (l. 65)', '"identical d net, since all cohorts reach y^End with the same remaining life" (the reason is '
        'that the 2025 cohort\'s remaining life is zero)', 'no longer applicable' if not same else 'still present',
        {'old_claim_found': same, 'note': 'not written; tab:res_year_ladder prints the 2025 salvage as 0.0'})
    little = _hits(lines, r'changes little', sec4)
    defer = _hits(lines, r'Deferring the reference unit does not make it pay', scopes['4.6'])
    row('Blocked 6 (l. 381)', '"Deferring the reference unit changes little gross and nothing net" contradicts the figures',
        'fixed' if not little else 'still present',
        {'old_claim_found': little, 'new_text': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'Deferring', 400)} for h in defer],
         'note': 'the new text makes no 2025-vs-2030/2035 net resolution claim; "net of salvage, the unit costs 54.5--64.4 '
                 'kEUR more than the plan without storage in every investment year" is checked in Part 1'},
        f"l. {defer[0]['line']}" if defer else None)
    ki = _hits(lines, r'Key Insights|subsec:res_insights|datasheet calibration and the 0\.70 floor')
    removed = [ln for ln in difflib.unified_diff(prev_main_lines, lines, lineterm='', n=0)
               if ln.startswith('-') and not ln.startswith('---')]
    row('Blocked 6 (l. 484)', '4.9 (i) "with the datasheet calibration and the 0.70 floor" ambiguous (baseline vs C4 row)',
        'no longer applicable' if not ki else 'still present',
        {'found_now': ki, 'main_tex_lines_removed_6ff8f3c_to_51a6fb5': len(removed),
         'removed_key_insights_heading': any('Key Insights' in ln for ln in removed),
         'removed_datasheet_clause': any('datasheet calibration and the 0.70 floor' in ln for ln in removed),
         'note': 'section 4.9 Key Insights was deleted at 51a6fb5 (git show 6ff8f3c:main.tex in the clone, read only)'})
    floor = _hits(lines, r'Baseline, 0\.50 floor', scopes['4.3'])
    cells = [c.strip() for c in floor[0]['text'].rstrip('\\').split('&')] if floor else []
    row('Blocked 7 / C5', 'optional fill of the 0.50-floor row: value 258.5 kEUR, AE 0.778, EFC/day 1.08, floor never (W155, '
        'not in the frozen T1-T11)', 'still open (row not filled; author\'s option)' if cells and cells[1] == '---' else 'filled',
        {'row': floor, 'cells': cells, 'W177a_values': cmap['C5']['fill_for_row_262'],
         'note': 'value, AE and EFC/day print "---" ("not reported", per the note); value - I, threshold, multiple and '
                 '"never" are checked in Part 1'}, f"l. {floor[0]['line']}" if floor else None)
    # C2, C3: from the Part 1 tokens of the rows
    def plan_ok(rng):
        ts = [t for t in toks if t['file'] == 'main.tex' and rng[0] <= t['line'] <= rng[1] and t.get('rule') == 'plan (candidate record)']
        return len(ts), sum(1 for t in ts if t['status'] == 'match')
    n2, m2 = plan_ok((1189, 1204))
    row('C2', 'S / E / duration of each B and C cell confirmed (16/16)', 'holds' if n2 and n2 == m2 and cmap['C2']['outcome'] == 'confirmed' else 'differs',
        {'plan_tokens_rows_1189_1204': {'n': n2, 'match': m2}, 'recomputed_C2': cmap['C2']['summary']}, 'l. 1189-1204')
    n3, m3 = plan_ok((1290, 1301))
    row('C3', 'I/J cells = node-7 4 h plans of 2-5 MWh (keys confirmed)', 'holds' if n3 and n3 == m3 and cmap['C3']['outcome'] == 'confirmed' else 'differs',
        {'plan_tokens_caption_and_rows_1290_1301': {'n': n3, 'match': m3}, 'recomputed_C3': cmap['C3']['summary']}, 'l. 1290-1301')
    # C6 16 cycles
    c16 = _hits(lines, r'16 cycles|sixteen|16~cycles', sec4)
    cont = _hits(lines, r'was continued beyond its stopping point', scopes['4.5'])
    capd = _hits(lines, r'evaluation continued past it', scopes['4.5'])
    row('C6', 'the 3 x 3 "16 cycles" confirmed; wording: the evaluation, not the fit, was continued',
        'fixed' if cont and not c16 else 'still present',
        {'16_cycles_printed': c16, 'prose': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'the evaluation of the plan without storage was continued', 260)} for h in cont],
         'caption': [{'line': h['line'], 'quote': _quote(lines, h['line'], '$D$ is the post-hoc', 160)} for h in capd],
         'recomputed_C6': cmap['C6']['summary']}, ', '.join(f"l. {h['line']}" for h in cont + capd))
    # C7 W153
    two = _hits(lines, r'changes two verdicts|It changes two verdicts', scopes['4.6'])
    yl_in_60 = [c['claim_id'] for c in X.w153s['claims'] if set(c.get('ref_cells', []) + c.get('other_cells', [])) &
                {'yl_y2030', 'yl_y2035'}]
    row('C7', 'W153: 0 sign changes in 60 claims; ONE verdict changes category (the F2 neighbour)',
        'changed (the text now says two verdicts)' if two else 'holds',
        {'text': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'Across the 60', 420)} for h in two],
         'recomputed_C7': cmap['C7']['summary'],
         'year_ladder_pair_among_the_60': bool(yl_in_60), 'W153_claims_with_a_year_ladder_cell': yl_in_60,
         'year_ladder': {'D_gross': X.t['year_ladder']['D_gross'], 'D_net': X.t['year_ladder']['D_net'],
                         'gross_verdict': X.t['year_ladder']['gross_v6']['verdict'], 'net_verdict': X.t['year_ladder']['net_v6']['verdict']},
         'note': 'carried to Part 1 (l. 1488 "two": MISMATCH against W153 n_verdict_differs = 1) and to W177b-2 Part 2 (c)'},
        f"l. {two[0]['line']}" if two else None)
    # C8, C9
    r25 = _hits(lines, r'^\s*2025 \(reference\)\s*&', scopes['4.6'])
    row('C8', 'the 2025 cohort\'s salvage is zero (the row may print 0.0)',
        'holds' if r25 and '0.0' in r25[0]['text'].split('&')[2] and cmap['C8']['outcome'] == 'confirmed' else 'differs',
        {'row': r25, 'recomputed_C8': cmap['C8']['summary'][:300]}, f"l. {r25[0]['line']}" if r25 else None)
    r41 = _hits(lines, r'41 new recourse evaluations', scopes['4.7'])
    row('C9', '41 new recourse evaluations (14, 20, 7 against 20, 20, 60) confirmed',
        'holds' if r41 and cmap['C9']['outcome'] == 'confirmed' else 'differs',
        {'text': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'made 41', 120)} for h in r41], 'recomputed_C9': cmap['C9']['summary']},
        f"l. {r41[0]['line']}" if r41 else None)

    # the [CONFIRM -- W177] comments still in main.tex
    confirms = _hits(lines, r'\[CONFIRM')
    g_ok = all(X.CL[g]['gross_verdict'] == 'determinate' and X.CL[g]['net_verdict_recomputed_here'] == 'determinate' and
               X.CL[g]['net_verdict_W153'] == 'determinate' for g in G)
    g_toks = [t for t in toks if t['file'] == 'main.tex' and t['line'] in (1445, 1446) and t['status'] not in ('excluded',)]
    re_c = []
    for h in confirms:
        ln = h['line']
        if 'eq:salvage_credit' in h['text']:
            re_c.append({'line': ln, 'point': 'salvage equation items (a)-(d)', 'outcome':
                         'items (a)-(d) hold as worded (recomputed C1: ' + ', '.join(f"{k.split(')')[0]})" + ' ' + v.split(' ')[0]
                                                                                    for k, v in {k: v['outcome'] for k, v in c1['items'].items()}.items()) +
                         '); the equation above them still needs the two corrections (Blocked 1)'})
        elif 'G rows' in h['text']:
            re_c.append({'line': ln, 'point': 'G rows of T1 (node 7, 0.5 MVA / 1 and 2 MWh, 2030 / 2035) gross / net, all determinate',
                         'outcome': ('confirmed' if g_ok and all(t['status'] == 'match' for t in g_toks) else 'differs') +
                         f": {sum(t['status'] == 'match' for t in g_toks)}/{len(g_toks)} tokens of l. 1445-1446 match the frozen "
                         f"T1 G claims; every G claim determinate gross and net (W153 and recomputed): {g_ok}"})
        elif 'W153' in h['text']:
            re_c.append({'line': ln, 'point': 'W153 totals; the neighbour at 0.95 x gross / 1.22 x net; the 2035 - 2030 pair',
                         'outcome': f"60 claims, 0 sign changes, 1 verdict change ({X.w153s['totals']['claims_verdict_differs']}) "
                                    f"-- recomputed C7 {cmap['C7']['outcome']}; the 2035 - 2030 investment-year pair is "
                                    f"{'AMONG' if yl_in_60 else 'NOT among'} the 60 (no W153 claim has a yl_y2030 / yl_y2035 "
                                    f"cell; it is T4 year_ladder): gross determinate {X.t['year_ladder']['gross_v6']['multiple']:.2f} x "
                                    f"-> net within resolution {X.t['year_ladder']['net_v6']['multiple']:.2f} x, sign + -> -"})
        elif 'W173' in h['text']:
            re_c.append({'line': ln, 'point': 'W173 realized use 14/20, 20/20, 7/60; 41 is their sum',
                         'outcome': f"recomputed C9 {cmap['C9']['outcome']}: {cmap['C9']['summary']}"})
        else:
            re_c.append({'line': ln, 'point': h['text'][:200], 'outcome': 'not recognized by this script'})
    same_as_committed = {k: {'committed_W177a': committed.get(k), 'recomputed_now': cmap[k]['outcome'],
                             'equal': committed.get(k) == cmap[k]['outcome']} for k in cmap}
    return rows, {'comments': confirms, 'rechecked': re_c, 'n': len(confirms)}, same_as_committed


def flag_checks(X, lines):
    out = []

    def chk(ln, what, flags_exp, extra_ok=True, words=()):
        r = lines[ln - 1]
        fl = {k: f'^{{{k}}}' in r for k in ('u', 'f', 'w')}
        ok = fl == flags_exp and extra_ok and all(w in r for w in words)
        out.append({'line': ln, 'what': what, 'flags_written': fl, 'flags_expected': flags_exp, 'words': list(words), 'ok': ok})

    def flags_of(claim):
        c = X.CL[claim]
        return {'u': X.cells[c['other_cell']]['status'] == 'uncertified', 'f': bool(c.get('uncertified_form_beside_A64_ruling4')),
                'w': False}
    ladder = ('C:y2025__n5_p0.25_e0.5', 'B:n5_4h_e1', 'B:n7_4h_e1', 'B:n7_4h_e2', 'B:n7_4h_e3', 'B:n7_4h_e4', 'B:n7_4h_e5',
              'B:n7_2h_e1', 'B:n7_2h_e2', 'B:n7_2h_e3', 'B:n7_2h_e4', 'B:n7_2h_e5', 'B:n9_4h_e1', 'B:n9_4h_e3', C59, C57)
    for k, claim in enumerate(ladder):
        chk(1189 + k, f'tab:res_value_ladder {claim} (caption: "All reported differences are determinate": '
                      f"{X.CL[claim]['gross_verdict']})", flags_of(claim), X.CL[claim]['gross_verdict'] == 'determinate')
    for ln, cid, src in ((1275, UNIT, 'cl'), (1276, 'H:m1.5:value_minus_I', 'cl'), (1277, 'H:m1.75:value_minus_I', 't10'),
                         (1278, 'H:m2:value_minus_I', 'cl')):
        c = X.CL[cid] if src == 'cl' else X.A64[cid]
        unc = (X.cells[c['other_cell']]['status'] == 'uncertified' if src == 'cl' else
               any(v['status'] != 'certified' for v in c['cells'].values()))
        chk(ln, f'tab:res_flex_ladder {cid}', {'u': unc, 'f': False, 'w': False}, words=(c['gross_verdict'],))
    for ln, c1, c2 in ((1297, 'H:m2:value_minus_I', None), (1298, 'I:m2:e2_value_minus_I', 'I:m2:second_MWh'),
                       (1299, 'J:e3_value_minus_I', 'J:e2_to_e3'), (1300, 'J:e4_value_minus_I', 'J:e3_to_e4'),
                       (1301, 'J:e5_value_minus_I', 'J:e4_to_e5')):
        det = all(X.CL[c]['gross_verdict'] == 'determinate' for c in (c1, c2) if c)
        chk(ln, f'tab:res_energy_ladder_m2 {c1} / {c2} (caption: "All differences are determinate")', flags_of(c1), det)
    for ln, arm in ((1339, 'C2_calfade'), (1340, 'C2'), (1341, 'C3_unit'), (1342, 'C3_midblock'), (1343, 'C4'), (1344, 'no_ageing')):
        r = _arm(X, arm)
        fy = 'never' if r['floor_year_070'] is None else str(r['floor_year_070'])
        if arm == 'no_ageing':
            fy = 'n/a'
        chk(ln, f"tab:res_ageing {arm} ({r['verdict']}; floor {fy})", {'u': False, 'f': False, 'w': r['verdict'] == 'within resolution'},
            words=(fy,))
    a = X.A64['E:soh050:value_minus_I']
    chk(1346, f"tab:res_ageing 0.50 floor ({a['gross_verdict']}; floor never binds)", {'u': False, 'f': False, 'w': False},
        a['gross_verdict'] == 'determinate', words=('never',))
    yl = X.t['year_ladder']
    cap = lines[1451 - 1]
    out.append({'line': 1451, 'what': 'tab:res_year_ladder caption: "determinate gross ... within resolution net"',
                'ok': yl['gross_v6']['verdict'] == 'determinate' and yl['net_v6']['verdict'] == 'within resolution' and
                'determinate gross' in cap and 'within resolution net' in cap})
    out.append({'line': 1472, 'what': 'tab:res_discount caption: "every difference is determinate"',
                'ok': all(d['verdict'] == 'determinate' for d in X.t['discount']) and 'every difference is determinate' in lines[1472 - 1]})
    return out


# ======================================================================================================================
#  5. guards, post-run, summary
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    _g, pk_a, _ok = W177A.guards_state()
    counts = dict(pk_a['counts'], w177b=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W177b post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W177b post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def _md_cell(s, n=None):
    s = str(s if s is not None else '').replace('|', '/').replace('\n', ' ')
    return s[:n] if n else s


def md_summary(o):
    L = [f"# W177b-1 -- Parts 0 and 1 at Overleaf {o['clone']['head'][:7]} (main.tex sha256 {o['clone']['tex_sha256']['main.tex'][:8]})", '',
         f"Zero solves; script `{SCRIPT_REL}`; frozen tables `{FZ_REL}` (sha256 {FZ_SHA[:8]}); clone `{CLONE}` (read only). "
         f"Objective convention: {o['objective_convention']}", '', '## Predictions (expert, Addendum 76)', '']
    for p in o['prediction']:
        L.append(f"- **{p['item']}** -- {p['outcome']}")
    L += ['', '## Part 0 -- W177a findings at this HEAD', '', '| id | W177a finding | status | where | evidence |', '|---|---|---|---|---|']
    for r in o['part0']['rows']:
        L.append(f"| {r['id']} | {_md_cell(r['w177a_finding'])} | {r['status']} | {_md_cell(r['where'])} | "
                 f"{_md_cell(json.dumps(r['evidence'], default=str), 700)} |")
    L += ['', '### [CONFIRM -- W177] comments in main.tex', '']
    for c in o['part0']['confirm_comments']['rechecked']:
        L.append(f"- l. {c['line']} ({c['point']}): {c['outcome']}")
    L += ['', 'W177a CONFIRM outcomes recomputed now vs committed: ' +
          ', '.join(f"{k} {'=' if v['equal'] else '!='} ({v['recomputed_now']})" for k, v in o['part0']['confirm_recomputed_vs_committed'].items()), '']
    s = o['summary']
    L += ['## Part 1 -- counts by scope', '', '| scope | lines | ' + ' | '.join(STATUSES) + ' | total |',
          '|---|---|' + '---|' * (len(STATUSES) + 1)]
    for sc, cnt in s['by_scope'].items():
        L.append(f"| {sc}{' (reported separately)' if sc in SEPARATE else ''} | {s['scope_lines'].get(sc, '')} | " +
                 ' | '.join(str(cnt.get(k, 0)) for k in STATUSES) + f" | {sum(cnt.values())} |")
    L += ['', f"In scope (Addendum 76 Part 1): {s['in_scope']}; reported separately (section 4 preamble and 4.8): {s['separate']}", '']
    L += ['### MISMATCH', '']
    for t in o['tokens']:
        if t['status'] == 'MISMATCH':
            L.append(f"- {t['file']} l. {t['line']}:{t['col']} ({t['scope_name']}, {t['part']}) written `{t['written']}` -> "
                     f"{t.get('value_at_written_precision')} [{_md_cell(t.get('counterpart'), 300)}] {_md_cell(t.get('note'), 900)}")
    L += ['', '### DERIVED (arithmetic)', '']
    for t in o['tokens']:
        if t.get('derived'):
            L.append(f"- {t['file']} l. {t['line']}:{t['col']} ({t['scope_name']}) `{t['written']}` -- {t['status']} "
                     f"(holds {t['holds']}): {_md_cell(t.get('counterpart'), 600)}")
    L += ['', '### Per-number table (every assigned token that is not excluded)', '',
          '| file | line | scope | part | written | status | value at written precision | source / category | note |',
          '|---|---|---|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['status'] == 'excluded':
            continue
        src = t.get('counterpart') if t['status'] != 'unchecked' else f"{t.get('category')}"
        L.append(f"| {t['file']} | {t['line']} | {t['scope_name']} | {t['part']} | {_md_cell(t['written'])} | {t['status']} | "
                 f"{_md_cell(t.get('value_at_written_precision'), 40)} | {_md_cell(src, 220)} | {_md_cell(t.get('note') or t.get('line_note'), 200)} |")
    L += ['', '## Flag and verdict checks (tables)', '']
    for c in o['flag_checks']:
        L.append(f"- l. {c['line']} {c['what']}: {'ok' if c['ok'] else 'NOT OK'}")
    L += ['', '## Integrity', '', f"checks: {sum(v is True for v in o['checks'].values())}/{len(o['checks'])} True; failed "
          f"{o['failed']}; guards ok {o['guards_ok']}; exit {o['exit_code']}", '']
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  6. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--overleaf-commit', required=True)
    ap.add_argument('--expect', action='append', default=[], help='file.tex=<sha256> for every *.tex of the clone')
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or OUT_DIR
    tag = 'W177b'
    if args.overleaf_commit != DECLARED_COMMIT:
        _log(f'[{tag} PRECONDITION FAILED] --overleaf-commit {args.overleaf_commit} != declared {DECLARED_COMMIT}')
        sys.exit(1)
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    pre = []
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        extra = sorted(set(os.listdir(od)) - {OUT_NAMES['log']})
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
    # ---- clone: commit, every *.tex declared and equal ------------------------------------------------------------------
    mabs = os.path.join(REPO, CLONE)
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex'], capture_output=True, text=True).stdout.strip()
    if not head.startswith(args.overleaf_commit) or head != CLONE_HEAD:
        pre.append(f'clone HEAD {head} does not match --overleaf-commit {args.overleaf_commit} / declared {CLONE_HEAD}')
    expect = {}
    for e in args.expect:
        if '=' not in e:
            pre.append(f'--expect {e!r}: not file=sha256')
            continue
        k, v = e.split('=', 1)
        expect[k] = v
    tex_now = sorted(f for f in os.listdir(mabs) if f.endswith('.tex'))
    tex_sha = {f: _sha(os.path.join(CLONE, f)) for f in tex_now}
    if sorted(expect) != tex_now:
        pre.append(f'--expect files {sorted(expect)} != the clone *.tex {tex_now}')
    if sorted(TEX_SHA) != tex_now:
        pre.append(f'declared TEX_SHA files {sorted(TEX_SHA)} != the clone *.tex {tex_now}')
    for f in tex_now:
        if expect.get(f) != tex_sha[f]:
            pre.append(f'{f}: sha256 {tex_sha[f]} != --expect {expect.get(f)}')
        if TEX_SHA.get(f) != tex_sha[f]:
            pre.append(f'{f}: sha256 {tex_sha[f]} != declared {TEX_SHA.get(f)}')
    blob_eq = {}
    for f in tex_now:
        hb = subprocess.run(['git', '-C', mabs, 'hash-object', f], capture_output=True, text=True).stdout.strip()
        hh = subprocess.run(['git', '-C', mabs, 'rev-parse', f'HEAD:{f}'], capture_output=True, text=True).stdout.strip()
        blob_eq[f] = bool(hb) and hb == hh
    if porcelain:
        pre.append(f'clone *.tex not clean: {porcelain!r}')
    # ---- inputs ---------------------------------------------------------------------------------------------------------
    all_inputs = dict(W177A.INPUTS)
    all_inputs.update(EXTRA_INPUTS)
    inputs = {}
    for key, rel in all_inputs.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'capture path missing: {rel}')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel), 'blob': _git('rev-parse', f'HEAD:{rel}')}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if 'FROZEN_JSON' in inputs and inputs['FROZEN_JSON']['sha256'] != FZ_SHA:
        pre.append(f'frozen JSON sha256 != {FZ_SHA}')
    if 'FRAME' in inputs and inputs['FRAME']['sha256'] != W177A.FRAME_SHA:
        pre.append(f'{FRAME} sha256 != {W177A.FRAME_SHA}')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] preconditions hold: clone {head[:8]} ({len(tex_now)} *.tex as declared: ' +
         ', '.join(f'{f} {tex_sha[f][:8]}' for f in tex_now) + f'); {len(inputs)} inputs committed clean; script committed '
         f'clean {script_clean}')

    # ---- context --------------------------------------------------------------------------------------------------------
    X = W177A.Ctx(inputs)
    for k in EXTRA_JSON:
        X.j[k] = _json(inputs[k]['path'])
    X.texts['BE_CONV'] = X.t['break_even']['objective_convention']
    X.texts['CAPTIONS'] = X.texts['CAPTIONS_MD']
    X.texts['W170_FORMULA'] = X.j['W170']['formula']
    X.def_src = {'BE_CONV': f'{FZ_REL} tables.break_even.objective_convention', 'CAPTIONS': inputs['CAPTIONS_MD']['path'],
                 'W170_FORMULA': f"{inputs['W170']['path']} formula", 'T7_TEX': inputs['T7_TEX']['path']}
    sal, consts, coh2025 = W177A.confirm_salvage(X)
    X.coh2025 = coh2025
    confirm_now = W177A.confirm_checks(X, sal, consts, coh2025)

    main_rel = os.path.join(CLONE, 'main.tex')
    main_text = _text(main_rel)
    main_lines = main_text.split('\n')
    hl_rel = os.path.join(CLONE, 'highlights.tex')
    hl_text = _text(hl_rel)
    frame_lines = X.frame.split('\n')
    prev = subprocess.run(['git', '-C', mabs, 'show', f'{PREV_HEAD}:main.tex'], capture_output=True, text=True)
    prev_main_lines = prev.stdout.split('\n') if prev.returncode == 0 else []
    scopes, labels = compute_scopes(main_lines)
    scopes_ok = list(scopes.items()) == list(EXPECTED_SCOPES.items())
    labels_ok = all(labels.get(k) == v for k, v in SUBSEC_LABELS.items())
    # the highlights blocks are the same five bullets
    hl_main = [s.strip() for s in main_lines[scopes['highlights (main.tex block)'][0]:scopes['highlights (main.tex block)'][1] - 1]
               if s.strip().startswith('\\item')]
    hl_file = [s.strip() for s in hl_text.split('\n') if s.strip().startswith('\\item')]

    # ---- tokens ---------------------------------------------------------------------------------------------------------
    toks = []
    for fname, text in (('main.tex', main_text), ('highlights.tex', hl_text)):
        doc = W164.Doc(fname, text)
        for t in W174.tokenize_v3(doc):
            t['file'] = fname
            sc = scope_of(fname, t['line'], scopes)
            if sc is None or (fname == 'main.tex' and sc not in IN_SCOPE + SEPARATE):
                continue
            t['scope_name'] = sc
            t['status'] = None
            if t['excluded']:
                t.update(status='excluded', category=f"structure: {t['excluded']}")
            elif t['type'] in W177A.IDENT_TYPES:
                t.update(status='excluded', category=f"identifier ({t['type']})")
            toks.append(t)
    decls = declarations()
    carried, carry_refused = carried_declarations(main_lines, frame_lines)
    decls_all = decls + carried
    texts = {'main.tex': main_lines, 'highlights.tex': hl_text.split('\n')}
    stale, claimed_twice, decl_out, dup_decl = [], [], [], []
    by_line = collections.defaultdict(list)
    for t in toks:
        by_line[(t['file'], t['line'])].append(t)
    seen = collections.Counter((d['file'], d['line']) for d in decls_all)
    dup_decl = [{'file': k[0], 'line': k[1]} for k, n in seen.items() if n > 1]
    for d in decls_all:
        key = (d['file'], d['line'])
        text = texts[d['file']][d['line'] - 1]
        cand = [t for t in by_line[key] if t['status'] != 'excluded']
        if len(cand) != len(d['specs']):
            stale.append({'file': d['file'], 'line': d['line'], 'why': f"{len(d['specs'])} specs for {len(cand)} tokens",
                          'tokens': [t['written'] for t in cand], 'specs': [s['w'] for s in d['specs']]})
            continue
        bad = [(t['written'], s['w']) for t, s in zip(cand, d['specs']) if t['written'].lower() != s['w'].lower()]
        if bad:
            stale.append({'file': d['file'], 'line': d['line'], 'why': 'written differs from the declaration', 'pairs': bad})
            continue
        res = []
        for t, s in zip(cand, d['specs']):
            if t['status'] is not None:
                claimed_twice.append(t['key'])
                continue
            try:
                r = evaluate(X, s, t, text)
            except Exception as e:  # noqa: BLE001 -- a declaration that cannot be evaluated is a run failure, listed
                stale.append({'file': d['file'], 'line': d['line'], 'why': f'evaluation error {type(e).__name__}: {e}',
                              'written': t['written']})
                r = {'status': None}
            if r.get('status') is None:
                continue
            t.update(r)
            t['declared'] = True
            if d.get('carried_from'):
                t['carried_from'] = d['carried_from']
            res.append({'written': t['written'], 'col': t['col'], 'status': r['status']})
        decl_out.append({'file': d['file'], 'line': d['line'], 'tokens': res, 'note': d.get('note'),
                         'carried_from': d.get('carried_from')})
        if d.get('note'):
            for t in cand:
                t.setdefault('line_note', d['note'])
    unassigned = [t for t in toks if t['status'] is None]
    for t in toks:
        envs = t.get('envs') or []
        if t['scope'] == 'comment':
            t['part'] = 'comment'
        elif 'minipage' in envs and 'table' in envs:
            t['part'] = 'table note'
        elif 'tabular' in envs:
            t['part'] = 'table body'
        elif 'table' in envs:
            t['part'] = 'table caption'
        elif 'figure' in envs:
            t['part'] = 'figure caption'
        elif any(e.startswith('equation') for e in envs):
            t['part'] = 'equation'
        else:
            t['part'] = 'prose'
    # ---- summaries --------------------------------------------------------------------------------------------------------
    order = list(EXPECTED_SCOPES)[:2] + ['highlights.tex'] + list(EXPECTED_SCOPES)[2:]
    by_scope = collections.OrderedDict((sc, dict(collections.Counter(t['status'] for t in toks if t['scope_name'] == sc)))
                                       for sc in order)
    scope_lines = {sc: f'{a}-{b}' for sc, (a, b) in scopes.items()}
    scope_lines['highlights.tex'] = f"1-{len(hl_text.rstrip(chr(10)).split(chr(10)))}"
    ins = dict(collections.Counter(t['status'] for t in toks if t['scope_name'] in IN_SCOPE))
    sep = dict(collections.Counter(t['status'] for t in toks if t['scope_name'] in SEPARATE))
    by_part = {p: dict(collections.Counter(t['status'] for t in toks if t['part'] == p)) for p in sorted({t['part'] for t in toks})}
    mism = [t for t in toks if t['status'] == 'MISMATCH']
    tab_parts = ('table body', 'table caption', 'table note', 'figure caption')
    mism_tab = [t for t in mism if t['part'] in tab_parts]
    mism_frozen = [t for t in mism if FZ_REL in str(t.get('counterpart')) or 'cells[' in str(t.get('counterpart'))]
    ders = [t for t in toks if t.get('derived')]
    named = {'24': None, '33': None, '7': [], '9': [], '90.9': None, '654': None, '9.3': None, '42.1': None, '28.5': None}
    for t in ders:
        if t['written'] in named:
            if isinstance(named[t['written']], list):
                named[t['written']].append({'line': t['line'], 'holds': t['holds']})
            else:
                named[t['written']] = {'line': t['line'], 'holds': t['holds']}
    p0rows, p0conf, p0same = part0(X, main_lines, scopes, confirm_now, X.j['W177A_JSON'], toks, prev_main_lines)
    flags = flag_checks(X, main_lines)
    prediction = [
        {'item': 'Part 1: zero MISMATCH in numbers carried from the tables',
         'outcome': (('HELD' if not mism else 'FAILED as worded') +
                     f": {len(mism)} MISMATCH among {sum(1 for t in toks if t['scope_name'] in IN_SCOPE and t['status'] != 'excluded')} "
                     f"in-scope tokens ({len(mism_tab)} in table bodies / captions / notes; {len(mism_frozen)} against a frozen-table "
                     f"field, the rest against a named record)" +
                     ('' if not mism else ' -- ' + '; '.join(f"l. {t['line']} '{t['written']}' -> {t.get('value_at_written_precision')} "
                                                            f"({str(t.get('counterpart'))[:120]})" for t in mism)))},
        {'item': 'Part 1: the derived numbers hold',
         'outcome': ('HELD' if all(t['holds'] for t in ders) else 'FAILED') +
                    f": {sum(t['holds'] for t in ders)}/{len(ders)} derived tokens hold at the written precision; named: " +
                    ', '.join(f'{k} {v}' for k, v in named.items())},
    ]
    checks = {
        'clone_head_is_declared': head == CLONE_HEAD,
        'every_clone_tex_sha_equals_declared_and_expect': all(TEX_SHA.get(f) == tex_sha[f] == expect.get(f) for f in tex_now),
        'every_clone_tex_equals_its_HEAD_blob': all(blob_eq.values()),
        'clone_tex_clean': porcelain == '',
        'main_tex_line_count': len(main_lines) - (1 if main_text.endswith('\n') else 0) == MAIN_LINES_N,
        'previous_head_readable': bool(prev_main_lines),
        'scopes_equal_declared': scopes_ok,
        'subsection_labels_as_declared': labels_ok,
        'highlights_main_block_equals_highlights_tex': hl_main == hl_file and len(hl_main) == 5,
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'every_carry_line_identical_to_its_frame_line': not carry_refused,
        'every_declaration_found_and_evaluated': not stale,
        'no_line_declared_twice': not dup_decl,
        'no_token_claimed_twice': not claimed_twice,
        'every_token_assigned': not unassigned,
        'statuses_in_vocabulary': all(t['status'] in STATUSES for t in toks),
        'every_input_committed_clean': all(v['committed_clean'] for v in inputs.values()),
        'production_salvage_form_reproduces_records': next(c for c in confirm_now if c['id'] == 'C1')['production_reproduces_records'],
        'w177a_confirm_outcomes_reproduced': all(v['equal'] for v in p0same.values()),
    }
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
    out = {'schema': 'p515_s53_w177b_number_check', 'version': 1,
           'stage': 'P5.15 Addendum 76 W177b-1 -- Part 0 (W177a findings) and Part 1 (numbers) at Overleaf 51a6fb5',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {EXTRA_INPUTS['W177A_SCRIPT']: _sha(EXTRA_INPUTS['W177A_SCRIPT'])}},
           'command_line': sys.argv,
           'clone': {'dir': CLONE, 'head': head, 'declared_commit': DECLARED_COMMIT, 'tex_sha256': tex_sha,
                     'tex_equals_head_blob': blob_eq, 'porcelain_tex': porcelain, 'edited': False,
                     'main_tex_lines': len(main_lines) - (1 if main_text.endswith('\n') else 0),
                     'previous_head_for_part0': PREV_HEAD},
           'objective_convention': X.fz.get('objective_convention'),
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA},
           'scopes': {'main.tex': {k: list(v) for k, v in scopes.items()}, 'highlights.tex': scope_lines['highlights.tex'],
                      'in_scope': list(IN_SCOPE), 'reported_separately': list(SEPARATE), 'labels': labels,
                      'not_carried': 'main.tex outside the abstract, highlights, section 4 and the Conclusions; the letter; '
                                     'cover_letter.tex; section2_expert_draft.tex (W176b results at 8b76423 stand; not '
                                     're-located here)'},
           'inputs': inputs,
           'prediction': prediction,
           'part0': {'rows': p0rows, 'confirm_comments': p0conf, 'confirm_recomputed_vs_committed': p0same,
                     'confirm_recomputed': confirm_now, 'w177a_json': inputs['W177A_JSON']},
           'summary': {'n_tokens': len(toks), 'by_status': dict(collections.Counter(t['status'] for t in toks)),
                       'by_scope': by_scope, 'scope_lines': scope_lines, 'in_scope': ins, 'separate': sep, 'by_part': by_part,
                       'n_mismatch': len(mism), 'n_mismatch_tables_captions_notes': len(mism_tab),
                       'n_derived': len(ders), 'n_derived_hold': sum(t['holds'] for t in ders), 'named_derived': named,
                       'n_declarations': len(decls), 'n_carried': len(carried), 'n_declarations_applied': len(decl_out)},
           'conventions': {'column': '1-based', 'line': '1-based',
                           'statuses': 'match = equal to the frozen JSON (or the named record) at the export\'s printed rounding '
                                       '(export/README.md); approximate = printed coarser than that rule and equal at its own '
                                       'precision; derived = computed from frozen values (arithmetic recorded) and equal at the '
                                       'written precision; MISMATCH (a derived number that does not hold is a MISMATCH); no '
                                       'counterpart; unchecked (category); excluded (structure / identifier)',
                           'sign': 'a written sign (- / + / $-$) is compared signed; an unsigned figure by magnitude (the prose '
                                   'states the direction); an unsigned table cell for a negative value is a MISMATCH'},
           'declarations_applied': decl_out, 'carry_refused': carry_refused, 'stale_declarations': stale,
           'lines_declared_twice': dup_decl, 'claimed_twice': claimed_twice,
           'unassigned': [{'file': t['file'], 'line': t['line'], 'written': t['written']} for t in unassigned],
           'flag_checks': flags, 'highlights': {'main_block': hl_main, 'highlights_tex': hl_file},
           'checks': checks, 'failed': failed, 'tokens': toks, 'guards': guards, 'pickle_guard': pk, 'guards_ok': guards_ok,
           'exit_code': code, 'wall_s': time.time() - t0}

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if not callable(v)}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        return o
    out = _clean(out)
    written = {}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        full = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        with open(full, 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        with open(full, 'rb') as h:
            written[rel] = _sha_bytes(h.read())
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    for f in tex_now:
        man[os.path.join(CLONE, f)] = tex_sha[f]
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log ------------------------------------------------------------------------------------------------------------
    for r in p0rows:
        _log(f"[{tag}] PART 0 {r['id']}: {r['status']} ({r['where']}) -- {r['w177a_finding'][:160]}")
    for c in p0conf['rechecked']:
        _log(f"[{tag}] CONFIRM comment l. {c['line']}: {c['outcome'][:400]}")
    _log(f"[{tag}] W177a CONFIRM recomputed vs committed: {p0same}")
    for p in prediction:
        _log(f"[{tag}] PREDICTION {p['item']}: {p['outcome']}")
    _log(f"[{tag}] tokens {len(toks)}: {dict(collections.Counter(t['status'] for t in toks))}; in scope {ins}; separate {sep}")
    for sc, cnt in by_scope.items():
        _log(f'[{tag}]   scope {sc} ({scope_lines.get(sc)}): {cnt}')
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no counterpart') or t.get('derived'):
            _log(f"[{tag}]   {t['file']} l. {t['line']}:{t['col']} ({t['scope_name']}, {t['part']}) {t['status']}: written "
                 f"{t['written']!r} -> {t.get('value_at_written_precision')!r} [{str(t.get('counterpart'))[:260]}] "
                 f"{(t.get('note') or '')[:300]}")
    for c in flags:
        if not c['ok']:
            _log(f"[{tag}]   flag check NOT OK: {json.dumps(c, default=str)[:400]}")
    _log(f"[{tag}] flag checks ok {sum(c['ok'] for c in flags)}/{len(flags)}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if stale or claimed_twice or unassigned or carry_refused or dup_decl:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; carry refused {carry_refused}; declared twice {dup_decl}; '
             f"unassigned {[(t['file'], t['line'], t['written']) for t in unassigned]}")
    if failed:
        _log(f'[{tag}] FAILED: {failed}')
    _log(f"[{tag}] wrote {', '.join(f'{k} {v[:8]}' for k, v in written.items())}")
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle ok {pk['ok']}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
