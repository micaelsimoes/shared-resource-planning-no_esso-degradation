"""P5.15 Addendum 76 + its Amendment, Planner task W177b-1' -- PARTS 0 AND 1 OF THE W177(b) ORDER RE-RUN AT OVERLEAF
e10c104, PLUS THE AMENDMENT'S TWO NUMBER ITEMS: (1) every number of the section 3 opening ("The 15-year planning horizon
is represented in two instances ...") against the case files, and (2) every number of the letter
(response_to_reviewers_draft.tex at e10c104 = manuscript_review/response_to_reviewers_v2.tex) against the frozen tables
(590088fe), the W160 export or a named record, with the reviewer quotations re-checked against
manuscript_review/Reviewers Comments.docx (W171a's check, unchanged). ZERO SOLVES, NO MODEL LOADS.

A NEW FILE. The W177b-1 script (p515_s53_w177b_number_check.py, 404e9181) is IMPORTED and NOT edited; through it the
W177a script and the chain W176b -> W174a -> W171a -> W164 -> ... -> W160 are imported, each arming its guards. USED from
W177b-1: its token evaluator (`evaluate`: derived / compound kinds over W177a's export rounding rules), its spec
constructors and derived functions, its declarations (`declarations()`, by line at 51a6fb5), its frame-carry table
(`CARRY`), its scope finder (`compute_scopes`) and its table flag checks (`flag_checks`). From W171a: the reviewers'
document reader and the quotation check. Every output is a new file opened 'x' in a new directory.

HOW THE 51a6fb5 DECLARATIONS ARE CARRIED. main.tex at 51a6fb5 (read with `git show` in the clone, read only) is mapped
line by line onto main.tex at e10c104 (difflib equal blocks). A W177b-1 declaration (and a W177b-1 frame carry) whose
51a6fb5 line lies in an equal block is RE-LOCATED to its e10c104 line unchanged (the written text of every token must
still equal the declaration, else the run exits 3); a declaration whose line changed is RETIRED and must be named in
RETIRED below with the e10c104 line(s) whose NEW declaration replaces it (else the run exits 3). Every in-scope token
is marked "unchanged line (51a6fb5 l. N)" or "new or changed line"; the statuses of the tokens on unchanged lines are
compared with W177b-1's committed JSON (an integrity check). W177b-1's line-pinned Part 0 and flag checks are evaluated
at e10c104: the flag checks by calling W177b-1's `flag_checks` on a 51a6fb5-indexed view of the e10c104 lines (every line
it reads is in an equal block; the view is checked), Part 0 by the port below (`part0`), whose line ranges are mapped.

SCOPES (main.tex at e10c104, sha256 1b532035..., 2,345 lines; ranges computed from the section markers and pinned):
  Part 1 in scope: abstract, highlights (main.tex block + highlights.tex), section 4.1-4.7, Conclusions;
  reported separately: section 4 preamble and 4.8 (as W177b-1);
  Amendment (1): section 3 opening (from \\section{Case Study} to the line before \\subsection{Investment Costs});
  Amendment (2): the letter, every line.
LETTER RULES (in this order): structure exclusions (the W164 tokenizer); a token inside \\rcomment{...} is a reviewer
quotation (status from W171a's quotation check against the reviewers' document); labels / hashes / dates / DOIs / alnum
are identifiers (excluded); a number inside a Section / Table / Figure / Eq. / Appendix / Algorithm pointer (W171a's
XREF_RX) is "unchecked: pointer, checked by W177b-3'" (Reviewer numbers: "unchecked: enumerator"); every other token is
DECLARED by line below, and pointers the regex misses are declared as pointers.

RECORDED PREDICTIONS (expert, Addendum 76): Part 1 -- zero MISMATCH in numbers carried from the tables; the derived
numbers hold. Scored in the outputs at e10c104, beside the 51a6fb5 outcome.

OBJECTIVE CONVENTION. Q = gross_operational_cost, settlement excluded (the frozen primary); net = Q - terminal salvage
credit; value = Q(0) - Q(x); F = Q + I.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W177b-1 chain arms; `pickle.load` / `pickle.loads` blocked for the whole run, every counter
verified at 0. git reads only (`git show` / `git rev-parse` / `git ls-files` / `git status` / `git log` /
`git hash-object` without -w), also in the manuscript clone, which is never pulled or modified.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w177b_number_check/overleaf_e10c104 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b1r_number_check.py \\
        --overleaf-commit e10c104 --expect main.tex=1b532035... --expect highlights.tex=001352ee... \\
        --expect response_to_reviewers_draft.tex=bc270fe2... --expect cover_letter.tex=363ad821... \\
        --expect section2_expert_draft.tex=7e347dc8... \\
        > data/SRP1/Results/P515S53/w177b_number_check/overleaf_e10c104/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w177b_number_check/overleaf_e10c104/w177b1r_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w177b_number_check/overleaf_e10c104/w177b1r_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b1r_number_check.py \\
        --overleaf-commit e10c104 --post-run
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

GUARD = SolveProfileGuard(permitted=(), label="P5.15 W177b-1' number check at Overleaf e10c104 (never solves)").install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError("W177b-1': pickle.load called -- no model loads are permitted")


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError("W177b-1': pickle.loads called -- no model loads are permitted")


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w177b_number_check as B1  # noqa: E402 -- arms its guard, imports W177a -> W176b -> ... -> W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W177A, W176, W174, W171, W164, W160 = B1.W177A, B1.W176, B1.W174, B1.W171, B1.W164, B1.W160
GUARDS = W160.W157._dedupe((('w177b1r_number_check', GUARD),) + tuple(B1.GUARDS))
_log, _sha, _sha_bytes, _git = W160._log, W160._sha, W160._sha_bytes, W160._git
S, U, P, W3, DEF, D = B1.S, B1.U, B1.P, B1.W3, B1.DEF, B1.D
NOT, SINGLE, LANG, XREF, DER = B1.NOT, B1.SINGLE, B1.LANG, B1.XREF, B1.DER
_cl, _t10, _text, _json, _norm = B1._cl, B1._t10, B1._text, B1._json, B1._norm
_arm, _retention, _same, _same_vals, _min_max_aged, _w173_year_dims = (B1._arm, B1._retention, B1._same, B1._same_vals,
                                                                        B1._min_max_aged, B1._w173_year_dims)
UNIT, REF4, C57, C59, LF2, BM, AF, YL, WC = B1.UNIT, B1.REF4, B1.C57, B1.C59, B1.LF2, B1.BM, B1.AF, B1.YL, B1.WC

SCRIPT_REL = os.path.basename(__file__)
B1_SCRIPT = os.path.basename(B1.__file__)
S53 = W160.S53
OUT_DIR = os.path.join(S53, 'w177b_number_check', 'overleaf_e10c104')
OUT_NAMES = {'json': 'w177b1r_number_check.json', 'md': 'w177b1r_number_check.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'typing_json': 'w177b1r_bool_typing_test.json', 'typing_log': 'w177b1r_bool_typing_test.log',
             'post': 'manifest_post_run_sha256.json'}

CLONE = W164.DEFAULT_MANUSCRIPT_DIR                       # manuscript/6a67305f25e8348fb71380c3 (read only)
DECLARED_COMMIT = 'e10c104'
CLONE_HEAD = 'e10c104f93e52506130ce48119cbaafc48bab558'
PREV_COMMIT = B1.CLONE_HEAD                               # 51a6fb5 (W177b-1)
PREV_6FF = B1.PREV_HEAD                                   # 6ff8f3c (the order's minimum HEAD; W177b-1's Part 0 comparison)
LETTER_PREV_CHECK = '8b76423'                             # the last number check of the letter (W176b)
TEX_SHA = {
    'main.tex': '1b532035abc65a7f8a08cea6304ac14fcd4ec2e6da553b9a1f250a0b09b5ba1b',
    'highlights.tex': '001352eebb3ac647f8fe1f5c2211025816ca5143c74c02a01af731630a8fea08',
    'response_to_reviewers_draft.tex': 'bc270fe2cf4b44f7e9547e17ccd85ecd94b56e8f99dd717486d9cf863c20a642',
    'cover_letter.tex': '363ad82139c8bbf1873e64c8984934d4e049a67af593fa90936a54f3601d7508',
    'section2_expert_draft.tex': '7e347dc830d1d9a8f0c6892a963a534d63d90846c6508108821e9cfac9bcabd1',
}
MAIN_LINES_N = 2345
LETTER = W164.LETTER                                      # response_to_reviewers_draft.tex
LETTER_V2 = os.path.join('manuscript_review', 'response_to_reviewers_v2.tex')
BIB = 'bibliography.bib'                                  # in the clone (read only; sha256 and HEAD blob recorded)
FRAME = B1.FRAME
FZ_REL, FZ_SHA = B1.FZ_REL, B1.FZ_SHA

EXPECTED_SCOPES = collections.OrderedDict([
    ('abstract', (140, 150)), ('highlights (main.tex block)', (162, 168)), ('4 preamble', (1176, 1203)),
    ('4.1', (1204, 1254)), ('4.2', (1255, 1346)), ('4.3', (1347, 1395)), ('4.4', (1396, 1434)), ('4.5', (1435, 1470)),
    ('4.6', (1471, 1527)), ('4.7', (1528, 1558)), ('4.8', (1559, 1573)), ('Conclusions', (1577, 1596))])
SEC3 = '3 opening'
EXPECTED_SEC3 = (951, 992)
SEC3_CORE = (955, 962)                                    # the order's "l. ~954-965" paragraph pair (M1 (a) + the timing paragraph)
IN_SCOPE, SEPARATE = B1.IN_SCOPE, B1.SEPARATE
STATUSES = B1.STATUSES + ('reviewer quotation, verified',)

# W177b-1 declarations whose 51a6fb5 line changed at e10c104, with the e10c104 line(s) of the new declaration
RETIRED = {141: ((145,), 'abstract rewritten (Overleaf after 51a6fb5): "two-stage" now at l. 145'),
           142: ((146,), 'abstract rewritten: 108 / 15 / 13.9 / 0.25 / 1 / 64 now at l. 146; 1.75, 171, 193, 254, 3 x 3 and '
                         '0.91-0.93 are no longer in the abstract'),
           1561: ((1593,), 'Conclusions M3: the future-work sentence split into l. 1593-1594; "one investment year" now at l. 1593')}

EXTRA_INPUTS = {
    'B1_SCRIPT': B1_SCRIPT,
    'B1_JSON': os.path.join(S53, 'w177b_number_check', 'overleaf_51a6fb5', 'w177b_number_check.json'),
    'B1_REPORT': 'P5_15_W177B_PART01.md',
    'W176B_JSON': os.path.join(S53, 'w176_manuscript_check', 'overleaf_8b76423', 'w176_manuscript_number_check.json'),
    'DOCX': W171.DOCX,
    'LETTER_V2': LETTER_V2,
    'FIXES_M1': os.path.join('manuscript_review', 'section3_intro_conclusions_fixes.tex'),
    'INSTANCE_3X3': os.path.join(S53, 'w89_3x3', 'instance', 'instance_record.json'),
    'CODE_DEFS': 'definitions.py',
    'A1_CAMPAIGN': W164.A1_CAMPAIGN,
    'P2': W164.P2_REL,
    'W167_Q4': os.path.join(S53, 'w167_nomenclature_years', 'question4_cost_file.json'),
}
EXTRA_JSON = ('B1_JSON', 'W176B_JSON', 'INSTANCE_3X3', 'W167_Q4')
EXTRA_TEXT = ('LETTER_V2', 'FIXES_M1', 'CODE_DEFS', 'A1_CAMPAIGN', 'P2')
# the case files of the section 3 opening (sha256 and git blob recorded; the 3 x 3 campaign spec records both sha256)
CASE_FILES = {'SRP1_JSON': W164.SRP1_JSON, 'CASE_3X3': W171.INST_3X3}


# ======================================================================================================================
#  1. new spec kinds and derived numbers
# ======================================================================================================================
def SUBF(w, frag, sub_text, desc):
    """a figure of the SUBMITTED version: `frag` is found in the committed submitted main.tex (whitespace-normalized),
    contains `sub_text`, and sub_text has the written value."""
    return dict(w=w, kind='subfrag', frag=frag, sub_text=sub_text, desc=desc)


def NOC(w, why):
    return dict(w=w, kind='noc', why=why)


def PTR(w, why='numbered pointer (Section / Table / Figure / Eq. / Appendix) the pointer regex does not catch'):
    return U(w, "pointer, checked by W177b-3'", why)


def _cases(X):
    return X.j['SRP1_JSON'], X.j['CASE_3X3']


def _op_scen(case):
    return _same_vals([case['TransmissionNetwork']['num_operation_scenarios']] +
                      [d['num_operation_scenarios'] for d in case['DistributionNetworks']])


def _year_in(key, y):
    return lambda X: int(y) if y in X.j[key]['Years'] else None


def _n_bus(X, which):
    if which == 9:
        return len(X.j['CASE9']['nodes'])
    return _same_vals([len(X.j[k]['nodes']) for k in ('CASE33_1', 'CASE33_2', 'CASE33_3')])


def _two_instances(X):
    di = X.j['SPEC_3X3']['configuration']['derived_instance']
    ok = (di['case_path'] == CASE_FILES['CASE_3X3'] and di['source_case_path'] == CASE_FILES['SRP1_JSON'] and
          di['case_sha256'] == _sha(CASE_FILES['CASE_3X3']) and di['source_case_sha256'] == _sha(CASE_FILES['SRP1_JSON']) and
          all('w90_3x3' in r['source'] for r in X.t['three_by_three']['rows'][:3]))
    return len({di['case_path'], di['source_case_path']}) if ok else None


def _one_install_per_node(X):
    cells = list(X.t['cells'].values()) + list(X.t['cells_appended_w160'].values())
    ok = all(isinstance(c['candidate_canonical']['investment_year'], int) and
             all(isinstance(v, list) and len(v) == 2 for v in c['candidate_canonical']['nodes'].values()) for c in cells)
    return 1 if ok and cells else None


def der_nine(X):
    s, c = _cases(X)
    m, o = c['NumMarketScenarios'], _op_scen(c)
    ir = X.j['INSTANCE_3X3']['derive_overrides']
    return {'value': m * o, 'consistent': ir == {'num_market_scenarios': m, 'num_operation_scenarios': o},
            'arithmetic': f"SRP1__s53_3x3.json NumMarketScenarios {m} x num_operation_scenarios {o} (TN and every DN) = "
                          f"{m * o}; instance_record.json derive_overrides {ir}"}


def der_48(X):
    s = X.j['SRP1_JSON']
    ny, nd, nn = len(s['Years']), len(s['Days']), 1 + len(s['DistributionNetworks'])
    return {'value': ny * nd * nn, 'arithmetic': f"SRP1.json Years {ny} x Days {nd} x network operators (1 TN + "
                                                f"{nn - 1} DN) {nn} = {ny * nd * nn}"}


def _divisors(X):
    q = X.j['W167_Q4']
    now = q['formula_recompute_7ce1d1ab']['formula_templates']['energy']
    pred = json.dumps(q['predecessor_072b1310_selected'])
    dn = re.search(r'\)/(\d)\*1000', now)
    dp = re.search(r"\)/(\d)\*1000\*\('Cost breakdown NREL'!\$B\$2\+", pred)
    return (int(dn.group(1)) if dn else None), (int(dp.group(1)) if dp else None)


def der_125(X):
    dn, dp = _divisors(X)
    return {'value': dp / dn, 'arithmetic': f"W167 question4_cost_file.json: submitted (predecessor 072b1310) energy formula "
                                            f"divisor {dp} / corrected (7ce1d1ab) divisor {dn} = {dp / dn}"}


def _cite_numbers(X):
    """citation numbers by order of first \\cite in main.tex at the HEAD (elsarticle-num.bst has no SORT: unsorted, by
    first citation); comments stripped."""
    order = []
    for ln in X.main_lines:
        body = W164.split_comment(ln)[0]
        for m in re.finditer(r'\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}', body):
            for k in m.group(1).split(','):
                k = k.strip()
                if k and k not in order:
                    order.append(k)
    return {k: i + 1 for i, k in enumerate(order)}


def _cite_in_sentence(X, needle):
    for ln in X.main_lines:
        body = W164.split_comment(ln)[0]
        k = body.find(needle)
        if k >= 0:
            m = re.search(r'\\cite\{([^}]*)\}', body[k:])
            return [x.strip() for x in m.group(1).split(',')] if m else None
    return None


def _bib_dois(X):
    out = {}
    for m in re.finditer(r'@\w+\{([^,\s]+),(.*?)(?=\n@|\Z)', X.bib, re.S):
        d = re.search(r'doi\s*=\s*\{([^}]*)\}', m.group(2))
        if d:
            out[m.group(1)] = re.sub(r'^https?://(dx\.)?doi\.org/', '', d.group(1).strip())
    return out


def der_cite(which):
    """which: 'R7' (the key cited after "rented to renewable producers"), 'R8' / 'R9' (the two keys cited after
    "congestion management"), 'lo' / 'hi' (the range of the three reviewer DOIs' citation numbers)."""
    def f(X):
        num = _cite_numbers(X)
        dois = _bib_dois(X)
        rev = re.findall(r'10\.\d{4,}/[^\s;}]+', X.r28_comment)
        rev = [d.rstrip('.') for d in rev]
        keys_rev = sorted(k for k, d in dois.items() if d in rev)
        nums_rev = sorted(num[k] for k in keys_rev if k in num)
        if not nums_rev or len(nums_rev) != len(keys_rev):
            return {'value': None, 'consistent': False, 'arithmetic': f'reviewer DOIs {rev} -> bib keys {keys_rev}; '
                                                                       f'citation numbers {nums_rev} (incomplete)'}
        k7 = _cite_in_sentence(X, 'rented to renewable producers')
        k89 = _cite_in_sentence(X, 'congestion management~')
        consistent = len(keys_rev) == 3 and None not in nums_rev and k7 is not None and k89 is not None and \
            sorted(k7 + k89) == keys_rev
        if which == 'R7':
            v = num.get(k7[0]) if k7 and len(k7) == 1 else None
        elif which in ('R8', 'R9'):
            v = sorted(num.get(k) for k in k89)[0 if which == 'R8' else 1] if k89 and len(k89) == 2 else None
        else:
            v = (nums_rev[0] if which == 'lo' else nums_rev[-1]) if nums_rev else None
        return {'value': v, 'consistent': consistent and nums_rev == list(range(nums_rev[0], nums_rev[0] + 3)),
                'arithmetic': f"citation numbers by first \\cite in main.tex at the HEAD (elsarticle-num, unsorted): "
                              f"{ {k: num.get(k) for k in keys_rev} }; reviewer DOIs {rev} -> bib keys {keys_rev} "
                              f"(bibliography.bib doi fields); \"rented to renewable producers\" cites {k7}, \"congestion "
                              f"management\" cites {k89}; static order, not a compile"}
    return f


def _n_reviewers_docx(X):
    return len(set(re.findall(r'Reviewer\s*#\s*(\d)', X.docx_text)))


def _n_letter_paragraphs(X):
    pre = X.letter_raw.split('\\section*{Reviewer 1}', 1)[0]
    return len(re.findall(r'^\\paragraph\{', pre, re.M))


def _n_r28_dois(X):
    return len(re.findall(r'10\.\d{4,}/[^\s;}]+', X.r28_comment))


def _n_subfloats(X, label):
    raw = '\n'.join(W164.split_comment(ln)[0] for ln in X.main_lines)
    for m in re.finditer(r'\\begin\{(figure\*?)\}(.*?)\\end\{\1\}', raw, re.S):
        if f'\\label{{{label}}}' in m.group(2):
            return len(re.findall(r'\\subfloat', m.group(2)))
    return None


def _new_explanatory_figures(X):
    labels = ('fig:search_generic', 'fig:investment_lattice')
    raw = '\n'.join(W164.split_comment(ln)[0] for ln in X.main_lines)
    now = [lb for lb in labels if f'\\label{{{lb}}}' in raw]
    sub = [lb for lb in labels if f'\\label{{{lb}}}' in X.texts['SUBMITTED_MAIN']]
    return len(now) if len(now) == len(labels) and not sub else None


def _soc(X, which):
    m = re.search(rf'^ENERGY_STORAGE_{which}_ENERGY_STORED = ([0-9.]+)', X.texts['CODE_DEFS'], re.M)
    return float(m.group(1)) if m else None


def _lattice(X, which):
    m = re.search(rf'^LATTICE_{which}_STEP = ([0-9.]+)', X.texts['A1_CAMPAIGN'], re.M)
    return float(m.group(1)) if m else None


def _scorecard(X):
    txt = X.texts['P2'].split('## (v) Prediction scorecard', 1)[1].split('**Not included:**', 1)[0]
    rows = [int(m.group(1)) for m in re.finditer(r'^\| (\d+) \|', txt, re.M)]
    return len(rows) if rows == list(range(1, len(rows) + 1)) else None


def _one_period(X):
    n9, n10 = X.W163.get('N9'), X.W163.get('N10')
    ok = n9 and n10 and n9['status'] == 'match' and n10['status'] == 'match' and float(n10['value']) > 1.0 and \
        float(n9['value']) >= 1.0
    return 1 if ok else None


def _fix_texts(fx):
    """the expert's fix file: every \\textcolor{blue}{...} block (brace-matched) and every other non-comment line."""
    body = '\n'.join(ln for ln in fx.split('\n') if not ln.lstrip().startswith('%'))
    blocks, i = [], 0
    key = '\\textcolor{blue}{'
    while True:
        k = body.find(key, i)
        if k < 0:
            break
        j, depth = k + len(key), 1
        while j < len(body) and depth:
            depth += {'{': 1, '}': -1}.get(body[j], 0)
            j += 1
        blocks.append((k, j, body[k + len(key):j - 1]))
        i = j
    rest = body
    for k, j, _b in reversed(blocks):
        rest = rest[:k] + rest[j:]
    plain = [ln.strip() for ln in rest.split('\n') if ln.strip()]
    return [' '.join(b.split()) for _k, _j, b in blocks], [' '.join(p.split()) for p in plain]


def _ret50_both(X):
    a, b = round(_retention(X, 'C3_unit'), 9), round(_retention(X, 'C3_midblock'), 9)
    return a if a == b else None


def _disc_both(X):
    s, c = _cases(X)
    return s['DiscountFactor'] if s['DiscountFactor'] == c['DiscountFactor'] else None


def _twelve(X):
    return B1._twelve(X)


# ======================================================================================================================
#  2. declarations at e10c104 (new lines; section 3 opening; the letter)
# ======================================================================================================================
def main_new_declarations():
    """main.tex lines changed since 51a6fb5 that carry in-scope tokens (RETIRED names what each replaces)."""
    L = []

    def add(d):
        d['file'] = 'main.tex'
        L.append(d)
    add(D(145, None, U('two', 'compound adjective', '"two-stage" (the model class, not a count)')))
    add(D(146, None, DER('108', B1.der_108, 'buses of the integrated system'),
          S('15', 'int', lambda X: sum(X.j['SRP1_JSON']['Years'].values()), 'SRP1.json Years: sum of the block lengths'),
          S('13.9', 'pct', f'{BM}.benefit_relative'), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]),
          S('64', 'k', _cl(REF4, 'd_gross'), note='"operational savings are approximately 64 kEUR below investment cost" = '
                                                    '-(value - I) of the reference unit (F(x) - F(0))')))
    add(D(1593, None, S('one', 'int', lambda X: _w173_year_dims(X), 'W173 box_definition: one year coordinate zY')))
    return L


def sec3_declarations():
    """the section 3 opening (Amendment (1)): every token from \\section{Case Study} to the line before 3.1."""
    L = []
    SR, C3 = 'SRP1_JSON', 'CASE_3X3'

    def add(d):
        d['file'] = 'main.tex'
        L.append(d)
    add(D(954, None, S('9', 'int', lambda X: _n_bus(X, 9), 'case9_2025.json nodes (IEEE 9-bus)'),
          S('three', 'int', lambda X: len(X.j[SR]['DistributionNetworks']), 'SRP1.json DistributionNetworks (count)'),
          S('33', 'int', lambda X: _n_bus(X, 33), 'case33_1/2/3_2025.json nodes (each)'),
          U('single', 'compound adjective', '"single-line diagram"')))
    add(D(955, None, S('three', 'int', lambda X: len(X.j[SR]['DistributionNetworks']), 'SRP1.json DistributionNetworks (count) '
                                                                                        '= the interface nodes')))
    add(D(957, None,
          S('15', 'int', lambda X: _same_vals([sum(X.j[SR]['Years'].values()), sum(X.j[C3]['Years'].values())]),
            'sum of the block lengths: SRP1.json Years and SRP1__s53_3x3.json Years (both 15)'),
          S('two', 'int', _two_instances, 'the instances with results: SRP1.json (T1-T10) and its derived 3 x 3 instance '
                                          '(T11); the 3 x 3 campaign spec derived_instance names both, with their sha256 '
                                          '(checked against the files now)'),
          SINGLE('single'),
          S('three', 'int', lambda X: len(X.j[SR]['Years']), 'SRP1.json Years (count)'),
          S('2025', 'int', _year_in(SR, '2025'), 'SRP1.json Years'), S('2030', 'int', _year_in(SR, '2030'), 'SRP1.json Years'),
          S('2035', 'int', _year_in(SR, '2035'), 'SRP1.json Years'),
          S('five', 'int', lambda X: _same_vals(X.j[SR]['Years'].values()), 'SRP1.json Years: block length (each)'),
          S('one', 'int', ('rec', SR, 'NumMarketScenarios'), 'SRP1.json NumMarketScenarios'),
          S('one', 'int', lambda X: _op_scen(X.j[SR]), 'SRP1.json num_operation_scenarios (TN and every DN)'),
          S('five', 'int', lambda X: len(X.j[C3]['Years']), 'SRP1__s53_3x3.json Years (count)'),
          *[S(y, 'int', _year_in(C3, y), 'SRP1__s53_3x3.json Years') for y in ('2025', '2028', '2031', '2034', '2037')],
          S('three', 'int', lambda X: _same_vals(X.j[C3]['Years'].values()), 'SRP1__s53_3x3.json Years: block length (each)'),
          S('three', 'int', ('rec', C3, 'NumMarketScenarios'), 'SRP1__s53_3x3.json NumMarketScenarios'),
          S('three', 'int', lambda X: _op_scen(X.j[C3]), 'SRP1__s53_3x3.json num_operation_scenarios (TN and every DN)'),
          DER('nine', der_nine, 'operating scenarios per representative day of the 3 x 3 instance'),
          S('four', 'int', lambda X: _same_vals([len(X.j[SR]['Days']), len(X.j[C3]['Days'])]),
            'Days (count) of both instances (Spring, Summer, Autumn, Winter)')))
    add(D(961, None, S('single', 'int', lambda X: _w173_year_dims(X), 'W173 box_definition: one year coordinate zY common to '
                                                                      'all nodes'),
          S('one', 'int', _one_install_per_node, 'every frozen cell\'s candidate_canonical: one investment_year and one '
                                                 '[S, E] pair per node (no staged additions)')))
    add(D(967, None, U('Single', 'comment (not typeset)', 'commented-out figure caption ("Single-line diagram")')))
    add(D(973, None, S('9', 'int', lambda X: _n_bus(X, 9), 'case9_2025.json nodes')))
    add(D(977, None, S('33', 'int', lambda X: _n_bus(X, 33), 'case33_1/2/3_2025.json nodes (each)')))
    add(D(985, None, S('Three', 'int', lambda X: len(X.expected_cost('energy', 2025)[1]),
                       'SRP1_ESS.xlsx sheet Scenarios: the investment-cost trajectories (probabilities)'),
          PTR('3.1', 'literal "Subsection~3.1" (not a \\ref): a numbered pointer, checked by W177b-3\'')))
    add(D(987, None, S('1', 'raw', lambda X: X.j['ESS_PARAMS']['budget'] / 1e6, 'SRP1_ESS_Params.json budget (EUR) / 1e6'),
          S('5.00', 'raw', ('rec', 'ESS_PARAMS', 'max_capacity'), 'SRP1_ESS_Params.json max_capacity (MWh per node)'),
          S('15', 'int', ('rec', 'ESS_PARAMS', 'ageing.calendar_life_years'), 'SRP1_ESS_Params.json ageing.calendar_life_years'),
          S('2.00', 'pct2', _disc_both, 'DiscountFactor of SRP1.json and SRP1__s53_3x3.json (equal)')))
    add(D(990, None, LANG('Two', '"Two comparisons are used" (the two comparisons enumerated in the sentence)'), NOT('0'),
          S('two', 'int', lambda X: len(X.t['benchmark']['arms']), 'T6: the two NRF arms (passive, price-taker)')))
    return L


def letter_declarations():
    """the letter at e10c104 (= response_to_reviewers_v2.tex): every token not covered by the letter rules."""
    L = []
    SR, C3 = 'SRP1_JSON', 'CASE_3X3'

    def add(d):
        d['file'] = LETTER
        L.append(d)
    nyrs = S('three', 'int', lambda X: len(X.j[SR]['Years']), 'SRP1.json Years (count)')
    blk5 = S('five', 'int', lambda X: _same_vals(X.j[SR]['Years'].values()), 'SRP1.json Years: block length (each)')
    add(D(24, None, U('1', 'enumerator', 'revision number (title block)')))
    add(D(33, None, S('three', 'int', _n_reviewers_docx, 'manuscript_review/Reviewers Comments.docx: distinct "Reviewer #n"')))
    add(D(35, None, S('two', 'int', _n_letter_paragraphs, 'the letter itself: \\paragraph{} blocks before \\section*{Reviewer 1}')))
    add(D(40, None, SUBF('0.50', 'relative optimality gap of 0.50', '0.50', 'submitted figure (the gap withdrawn)')))
    add(D(42, None, PTR('2.2')))
    add(D(48, None, P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('7', 'node', UNIT, [7]),
          S('64.4', 'k', _cl(UNIT, 'd_gross'), note='"loses 64.4 kEUR" = -(value - I) of the reference unit')))
    add(D(51, None, SUBF('18.25', 'reduces operating costs by up to 18.25', '18.25', 'submitted abstract figure (withdrawn)')))
    add(D(53, None, DER('90.9', B1.der_meur(f'{BM}.benefit', 'benefit'), 'value of coordination (MEUR)'),
          S('13.9', 'pct', f'{BM}.benefit_relative')))
    add(D(58, None, LANG('Two', '"Two requests were not met in full"'),
          SUBF('five', 'represented by five years', 'five', 'the submitted instance: five representative years'),
          SUBF('twenty-five', 'resulting in 25 operating scenarios', '25', 'the submitted instance: 5 x 5 scenarios'),
          S('3', 'int', ('rec', C3, 'NumMarketScenarios'), '3 x 3 instance: market scenarios'),
          S('3', 'int', lambda X: _op_scen(X.j[C3]), '3 x 3 instance: operation scenarios')))
    add(D(77, None, SINGLE('single'), S('3', 'int', ('rec', C3, 'NumMarketScenarios'), '3 x 3 instance: market scenarios'),
          S('3', 'int', lambda X: _op_scen(X.j[C3]), '3 x 3 instance: operation scenarios')))
    add(D(78, None, S('five', 'int', lambda X: len(X.j[C3]['Years']), 'SRP1__s53_3x3.json Years (count): "five-block horizon"'),
          S('0.909', 'r3', 'three_by_three.rows[8].value'), S('0.934', 'r3', 'three_by_three.rows[7].value'), SINGLE('single'),
          S('0.933', 'r4', 'three_by_three.rows[9].value', note='R predicted from the mean-profile spread (0.9331)'),
          S('9', 'int', lambda X: _n_bus(X, 9), 'case9_2025.json nodes'),
          S('three', 'int', lambda X: len(X.j[SR]['DistributionNetworks']), 'SRP1.json DistributionNetworks (count)'),
          S('33', 'int', lambda X: _n_bus(X, 33), 'case33_1/2/3_2025.json nodes (each)'),
          DER('108', B1.der_108, 'buses of the integrated system')))
    add(D(80, None, SUBF('18.25', 'reduces operating costs by up to 18.25', '18.25', 'submitted figure (withdrawn)'),
          SUBF('92.16', 'RES curtailment by up to 92.16', '92.16', 'submitted figure (withdrawn)'),
          S('13.9', 'pct', f'{BM}.benefit_relative'),
          S('15', 'int', lambda X: sum(X.j[SR]['Years'].values()), 'SRP1.json Years: sum of the block lengths')))
    add(D(81, None, dict(nyrs), dict(blk5), SINGLE('single'),
          S('five', 'int', lambda X: len(X.j[C3]['Years']), 'SRP1__s53_3x3.json Years (count)'),
          S('three', 'int', lambda X: _same_vals(X.j[C3]['Years'].values()), 'SRP1__s53_3x3.json Years: block length')))
    add(D(97, None, S('0', 'pct0', 'discount[0.0].rate'), S('2', 'pct0', 'discount[0.02].rate'),
          S('5', 'pct0', 'discount[0.05].rate'), S('8', 'pct0', 'discount[0.08].rate'),
          S('-40.8', 'k', 'discount[0.0].value_minus_I'), S('-64.4', 'k', 'discount[0.02].value_minus_I'),
          S('-92.7', 'k', 'discount[0.05].value_minus_I'), S('-114.6', 'k', 'discount[0.08].value_minus_I'),
          S('2', 'pct0', ('rec', SR, 'DiscountFactor'), 'SRP1.json DiscountFactor (the production rate)')))
    add(D(102, None, S('two', 'int', lambda X: _n_subfloats(X, 'fig:tso_dso_coordination_cs1_network_diagram'),
                       'main.tex at the HEAD: \\subfloat panels of fig:tso_dso_coordination_cs1_network_diagram')))
    add(D(113, None, U('two', 'compound adjective', '"two-stage"')))
    add(D(114, None, LANG('one', '"selects one scenario-independent multi-year plan" (one plan)')))
    add(D(116, None, LANG('single', '"a single vector" (article sense)')))
    add(D(118, None, LANG('single', '"that single plan" (article sense)')))
    add(D(125, None, S('0.25', 'raw', lambda X: _lattice(X, 'P'), f'{W164.A1_CAMPAIGN} LATTICE_P_STEP (MVA)'),
          S('0.5', 'raw', lambda X: _lattice(X, 'E'), f'{W164.A1_CAMPAIGN} LATTICE_E_STEP (MWh)'),
          S('2', 'raw', ('rec', 'ESS_PARAMS', 'min_energy_to_power_factor'), 'SRP1_ESS_Params.json min E/P (h)'),
          S('4', 'raw', ('rec', 'ESS_PARAMS', 'max_energy_to_power_factor'), 'SRP1_ESS_Params.json max E/P (h)'),
          LANG('One', '"One recourse evaluation is made per candidate plan" (method statement)'),
          LANG('one', '"one local problem per network block" (method statement)'), dict(nyrs),
          S('four', 'int', lambda X: len(X.j[SR]['Days']), 'SRP1.json Days (count)'),
          S('four', 'int', lambda X: 1 + len(X.j[SR]['DistributionNetworks']), 'SRP1.json: 1 TN + the DistributionNetworks'),
          SINGLE('single'), DER('48', der_48, 'network blocks per ADMM cycle at the single-scenario instance'),
          LANG('one', '"one per shared-ESS agent" (method statement)'),
          S('three', 'int', lambda X: len({d['connection_node_id'] for d in X.j[SR]['DistributionNetworks']}),
            'SRP1.json DistributionNetworks connection_node_id (distinct) = the interface nodes = the shared-ESS agents'),
          LANG('one', '"one per interface node" (method statement)')))
    add(D(135, None, PTR('2.2')))
    add(D(141, None, PTR('2.2.7', 'the submitted version\'s section number ("former 2.2.7 Benders\' cuts")')))
    add(D(142, None, PTR('2.2.7', '"new 2.2.7" (the revised section number)')))
    add(D(153, None, W3('three', 'N7', 3, 'at_least_3_turning_points')))
    add(D(155, None, S('one', 'int', _one_period, 'W163 N9 (W_MIN 20) and N10 (W_FACTOR 1.1), both committed checks status '
                                                  'match: W = max(20, ceil(1.1 P_hat)) >= 1.1 P_hat > P_hat, i.e. a window at '
                                                  'least one measured period long (as W176b)')))
    add(D(157, None, S('42', 'int', WC + ('result.totals.n_cells',)), S('32', 'int', WC + ('result.totals.n_certified',))))
    add(D(158, None, S('10', 'int', WC + ('result.totals.n_uncertified',))))
    add(D(159, None, W3('0.9', 'N25', 0.9, 'largest movement past a settling-rule certificate: 0.882 tau')))
    add(D(163, None, U('two', 'compound adjective', '"two-phase schedule"')))
    add(D(171, None, S('three', 'int', lambda X: _same_vals([X.fzp(f'{AF}.passive.n_starts'), X.fzp(f'{AF}.price_taker.n_starts')]),
                       'T6 n_starts (both arms)')))
    add(D(182, None, S('three', 'int', _n_r28_dois, 'the letter itself: DOIs quoted in the R2.8 comment'),
          DER('7', der_cite('lo'), 'references [7]--[9]: the lowest citation number of the three'),
          DER('9', der_cite('hi'), 'references [7]--[9]: the highest citation number of the three'),
          DER('7', der_cite('R7'), '"[7]": the key cited in main.tex after "rented to renewable producers"'),
          DER('8', der_cite('R8'), '"[8, 9]": the keys cited in main.tex after "congestion management"'),
          DER('9', der_cite('R9'), '"[8, 9]": the keys cited in main.tex after "congestion management"'),
          S('three', 'int', _n_r28_dois, 'the letter itself: DOIs quoted in the R2.8 comment')))
    add(D(205, None, LANG('two', '"couples the two" (the network models and the agent)')))
    add(D(220, None, P('two', 'nnodes', C57, None)))
    add(D(221, None, S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims'), 'W153: the comparisons scored gross and net'),
          S('two', 'int', ('rec', 'W153_SALV', 'result.totals.n_verdict_differs'), 'W153: verdicts that change, gross -> net',
            note='as written ("It changes no sign in the 60 reported comparisons and two verdicts") the count is of the 60: '
                 'W153 records ONE (L:y2030__n5_p0.25_e0.5__n7_p1_e3, 0.95 x gross -> 1.22 x net). The second verdict the '
                 'sentence names, the 2035 - 2030 investment-year comparison, is NOT one of the 60 (T4 year_ladder): '
                 'determinate gross (3.46 x) -> within resolution net (0.19 x), and its sign changes (+42,458.65 -> '
                 '-2,286.25 EUR). The same count as main.tex section 4.6 (W177b-1 l. 1488; here l. 1520)'),
          P('two', 'nnodes', LF2, None), P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030'),
          S('+42.5', 'k', f'{YL}.D_gross'), S('-2.3', 'k', f'{YL}.D_net'),
          S('10', 'pct0', lambda X: _soc(X, 'MIN'), 'definitions.py ENERGY_STORAGE_MIN_ENERGY_STORED (SoC^Min)'),
          S('90', 'pct0', lambda X: _soc(X, 'MAX'), 'definitions.py ENERGY_STORAGE_MAX_ENERGY_STORED (SoC^Max)')))
    add(D(224, None, SUBF('0.50', 'relative optimality gap of 0.50', '0.50', 'submitted figure quoted in the heading')))
    add(D(231, None, W3('4', 'N18', 4.0, 'TAU = DELTA_R * R_REF / 4'), W3('0.07', 'N19', 0.07, 'DELTA_R'), SINGLE('single')))
    add(D(232, None, W3('two', 'N22', 2, 'R = V / V_SRP1: a ratio of two values')))
    add(D(241, None, LANG('one', '"one representative year" (the block definition)'),
          LANG('one', '"one representative day" (the block definition)'), LANG('single', '"a single ... schedule" (article)'),
          LANG('one', '"one scenario-free variable per hour" (model statement)'),
          LANG('one', '"this one schedule" (model statement)')))
    add(D(256, None, S('0.985', 'raw', ('rec', 'ESS_PARAMS', 'ageing.calendar_retention_per_year'),
                       'SRP1_ESS_Params.json ageing.calendar_retention_per_year (the C2_calfade baseline)'),
          S('31.6', 'k', 'ageing.rows[C2].value_minus_I'), S('64.4', 'k', 'ageing.rows[C2_calfade].value_minus_I')))
    add(D(263, None, P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('7', 'node', UNIT, [7]),
          S('-4.1', 'k', 'ageing.rows[no_ageing].value_minus_I'), S('-31.6', 'k', 'ageing.rows[C2].value_minus_I'),
          S('-64.4', 'k', 'ageing.rows[C2_calfade].value_minus_I'),
          S('8,000', 'int', lambda X: round(_arm(X, 'C4')['k'] * -math.log(0.70)), 'T8 k of C4 x (-ln 0.70)'),
          S('-45.2', 'k', 'ageing.rows[C4].value_minus_I'),
          S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit: R = exp(-N D / k), N D = 8,000'),
          S('-73.7', 'k', 'ageing.rows[C3_unit].value_minus_I'), S('-65.9', 'k', 'ageing.rows[C3_midblock].value_minus_I')))
    add(D(264, None, S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')),
          S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh'),
          S('+4.9', 'k', _t10('E:soh050:delta_value_vs_070', 'd_gross')),
          S('2035', 'int', lambda X: _same(X, ('C2_calfade', 'C3_unit', 'C3_midblock'), 'floor_year_070'),
            'T8 floor binds (0.70): baseline, C3_unit, C3_midblock (all 2035; the other aged arms never)'),
          S('50', 'pct0', _ret50_both, 'T8 k of C3_unit and C3_midblock: R = exp(-8,000 / k), both 0.50')))
    add(D(272, None, DEF('one', 'T7_TEX', 'one discount factor per representative year', 'T7 caption')))
    add(D(282, None, dict(blk5)))
    add(D(283, None, S('1', 'int', f'{BM}.sweep.sweep_passive_cold.n_blocks'),
          S('12', 'int', _twelve, 'SRP1.json: 3 years x 4 days = 12 TN blocks'),
          S('8', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_blocks'),
          S('12', 'int', _twelve, 'SRP1.json: 12 blocks'), PTR('4.4')))
    add(D(284, None, W3('10^{-5}', 'N33', 1e-5, 'RES availability slack (EQUALITY_TOLERANCE)')))
    add(D(285, None, NOC('single', '"single machine": no committed record states the machine count (W176b: the author\'s '
                                   'attestation); the machine is named in section 4.7 (W163 N55-N57)'),
          W3('single', 'N45', 1, 'single-threaded (thread caps)')))
    add(D(286, None, S('49', 'int', _scorecard, f'{W164.P2_REL} section (v) prediction scorecard: rows numbered 1..n')))
    add(D(287, None, S('4', 'int', lambda X: _divisors(X)[0], 'W167 question4_cost_file.json formula_recompute_7ce1d1ab '
                                                               '(the corrected energy formula divides the 4-h system cost by 4)'),
          S('5', 'int', lambda X: _divisors(X)[1], 'W167 question4_cost_file.json predecessor_072b1310_selected (submitted: /5)'),
          S('4', 'int', lambda X: _divisors(X)[0], 'W167: the corrected divisor'),
          DER('1.25', der_125, 'the corrected values are x 5/4')))
    add(D(288, None, S('two', 'int', _new_explanatory_figures, 'main.tex at the HEAD: fig:search_generic and '
                                                               'fig:investment_lattice present, neither in the submitted main.tex')))
    return L


# ======================================================================================================================
#  3. evaluation
# ======================================================================================================================
def evaluate(X, spec, t, line):
    k = spec['kind']
    if k == 'subfrag':
        flat = re.sub(r'\s+', ' ', X.texts['SUBMITTED_MAIN'])
        found = spec['frag'] in flat
        lines = W177A._find_lines(X.texts['SUBMITTED_MAIN'], spec['frag'][:30])
        tok_ok = spec['sub_text'] in spec['frag'] and W174.wv3(spec['sub_text']) == W174.wv3(t['written'])
        return {'status': 'match' if found and tok_ok else 'MISMATCH',
                'counterpart': f"{X.inputs['SUBMITTED_MAIN']['path']} (sha256 {X.inputs['SUBMITTED_MAIN']['sha256'][:8]}) "
                               f"lines {lines}: {spec['frag']!r}", 'counterpart_value': spec['sub_text'],
                'value_at_written_precision': spec['sub_text'], 'rule': 'submitted-version figure (fragment in the '
                                                                        'committed submitted main.tex)', 'note': spec.get('desc')}
    if k == 'noc':
        return {'status': 'no counterpart', 'note': spec['why'], 'rule': 'no committed record'}
    return B1.evaluate(X, spec, t, line)


def apply_declarations(X, toks, decls, texts):
    """W177b-1's mechanism: per declared (file, line), the unassigned tokens of the line in order must equal the specs
    (count and written text), else the declaration is stale; a declared line twice and a token claimed twice are listed."""
    stale, claimed_twice, decl_out = [], [], []
    by_line = collections.defaultdict(list)
    for t in toks:
        by_line[(t['file'], t['line'])].append(t)
    seen = collections.Counter((d['file'], d['line']) for d in decls)
    dup = [{'file': k[0], 'line': k[1]} for k, n in seen.items() if n > 1]
    for d in decls:
        key = (d['file'], d['line'])
        text = texts[d['file']][d['line'] - 1]
        cand = [t for t in by_line[key] if t['status'] not in ('excluded',) and not t.get('rule_assigned')]
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
            for f in ('carried_from', 'relocated_from_51a6fb5_line'):
                if d.get(f):
                    t[f] = d[f]
            res.append({'written': t['written'], 'col': t['col'], 'status': r['status']})
        decl_out.append({'file': d['file'], 'line': d['line'], 'tokens': res, 'note': d.get('note'),
                         'carried_from': d.get('carried_from'), 'relocated_from_51a6fb5_line': d.get('relocated_from_51a6fb5_line')})
    return stale, claimed_twice, decl_out, dup


# ======================================================================================================================
#  4. line maps, relocation, Part 0 (ported with mapped lines), flags
# ======================================================================================================================
def line_map(old, new):
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    m, hunks = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
        else:
            hunks.append({'op': tag, 'old': [i1 + 1, i2], 'new': [j1 + 1, j2]})
    return m, hunks


def relocate_b1(M, main_lines, frame_lines):
    """W177b-1's declarations (51a6fb5 lines) and its frame carries -> e10c104 lines; changed lines -> RETIRED."""
    out, retired, unmapped = [], [], []
    for d in B1.declarations():
        if d['file'] != 'main.tex':
            nd = dict(d)
            nd['relocated_from_51a6fb5_line'] = d['line']
            out.append(nd)
            continue
        nl = M.get(d['line'])
        if nl is None:
            (retired if d['line'] in RETIRED else unmapped).append(
                {'line_51a6fb5': d['line'], 'specs': [s['w'] for s in d['specs']],
                 'replaced_by': RETIRED.get(d['line'], (None, None))[0], 'why': RETIRED.get(d['line'], (None, None))[1]})
            continue
        nd = dict(d)
        nd['line'], nd['relocated_from_51a6fb5_line'] = nl, d['line']
        out.append(nd)
    by_frame = {d['line']: d for d in W177A.declarations()}
    carried, refused = [], []
    for m_old, f_ln in sorted(B1.CARRY.items()):
        m_new = M.get(m_old)
        same = m_new is not None and ' '.join(main_lines[m_new - 1].split()) == ' '.join(frame_lines[f_ln - 1].split())
        d = by_frame.get(f_ln)
        if not same or d is None:
            refused.append({'main_line_51a6fb5': m_old, 'main_line_e10c104': m_new, 'frame_line': f_ln, 'identical': same,
                            'w177a_declaration_found': d is not None})
            continue
        nd = dict(d)
        nd['line'], nd['file'] = m_new, 'main.tex'
        nd['relocated_from_51a6fb5_line'] = m_old
        nd['carried_from'] = {'script': B1.EXTRA_INPUTS['W177A_SCRIPT'], 'frame': FRAME, 'frame_line': f_ln,
                              'via': f'{B1_SCRIPT} CARRY (51a6fb5 l. {m_old})'}
        carried.append(nd)
    return out, carried, retired, unmapped, refused


_hits, _quote = B1._hits, B1._quote


def part0(X, lines, scopes, confirm_now, w177a_json, toks, prev6ff_lines, M):
    """W177b-1's part0, ported to e10c104: the same checks, with W177b-1's pinned 51a6fb5 ranges mapped through M."""
    rows = []
    sec4 = (scopes['4 preamble'][0], scopes['4.8'][1])
    cmap = {c['id']: c for c in confirm_now}
    committed = {c['id']: c['outcome'] for c in w177a_json['confirm']}

    def row(id_, finding, status, evidence, where=None):
        rows.append({'id': id_, 'w177a_finding': finding, 'status': status, 'evidence': evidence, 'where': where})

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
    s42 = scopes['4.2']
    sizes = _hits(lines, r'1~MVA|3~MWh|3\.5~MWh|1 MVA', s42)
    f2 = _hits(lines, r'two-node investment plan', s42)
    row('Blocked 2 / C4', 'the F2 plan misdescribed: node 7 is 1 MVA / 3.5 MWh, not 3 MWh (frame l. 131)',
        'no longer applicable' if not sizes else 'still present',
        {'plan_sizes_printed_in_4.2': sizes,
         'F2_sentence': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'The planning search under this price')} for h in f2],
         'note': 'the text names the plan as "a two-node investment plan installed in 2030" without sizes; the two nodes and '
                 '2030 match the frozen candidate of ref:5ca4f86c (Part 1, l. ' + (str(f2[0]['line']) if f2 else '?') + ')',
         'recomputed_C4': {'outcome': cmap['C4']['outcome'], 'summary': cmap['C4']['summary']}},
        f"l. {f2[0]['line']}" if f2 else None)
    j4 = _hits(lines, r'^\s*4\$\^\{f\}\$\s*&', scopes['4.2'])
    j4n = _hits(lines, r'Treating the flagged evaluation as uncertified', scopes['4.2'])
    j4_main = bool(j4) and '9.7' in j4[0]['text'] and '18.14' in j4[0]['text']
    j4_note = bool(j4n) and '15.9' in j4n[0]['text'] and '11.12' in j4n[0]['text']
    row('Blocked 3', 'Addendum 64 ruling 4 (J 4 MWh at its uncertified bar) vs tab:res_energy_ladder_m2: the 4 MWh row prints '
        'the flagged certified threshold 9.7 / 18.14 x; the uncertified bar 15.9 / 11.12 x is beside it',
        'still present' if j4_main and j4_note else ('fixed' if not j4_main else 'moved'),
        {'row': j4, 'note': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'Treating')} for h in j4n],
         'main_columns_certified_threshold': j4_main, 'uncertified_bar_in_note': j4_note,
         'remark': 'the uncertified bar is in the table note (W177b-1: moved there from the frame caption); no verdict '
                   'depends on the choice'},
        f"l. {j4[0]['line']}, note l. {j4n[0]['line']}" if j4 and j4n else None)
    row('Blocked 4', 'two of three Addendum 73 predictions failed (scoring record)', 'no longer applicable',
        {'note': 'a scoring record, not manuscript text; scored in Addendum 74/76 by the Planner'})
    lit = _hits(lines, r'Section~4|Section 4\b|Sections~4|Subsection~4|\(Section~?4')
    refs = [h for h in _hits(lines, r'\\ref\{(sec:results|subsec:res_[a-z]+)\}') if not (sec4[0] <= h['line'] <= sec4[1])]
    lit_any = [h for h in _hits(lines, r'(?:Sub)?[Ss]ections?~\d|Appendix~[A-E]\b|Table~\d|Figure~\d|Fig\.~\d|Eq\.~\(\d')
               if not h['text'].lstrip().startswith('%')]
    row('Blocked 5', 'literal "Section~4.x" references (l. 377, 415, 445, 631, 635, 671, 1066, 1119 at 8b76423)',
        'fixed' if not lit else 'still present',
        {'literal_section_4_references': lit,
         'refs_to_section_4_labels_outside_section_4': [{'line': h['line'], 'refs': re.findall(r'\\ref\{((?:sec:results|subsec:res_[a-z]+))\}', lines[h['line'] - 1])} for h in refs],
         'any_literal_numbered_pointer_in_main_tex (wider search, this run)': lit_any,
         'note': 'the wider search (any literal "Section~n" / "Subsection~n" / "Table~n" / "Figure~n" / "Eq.~(n" / '
                 '"Appendix~X" outside comments) is new in this run; its hits are pointers for W177b-3\''})
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
         'note': '"net of salvage, the unit costs 54.5--64.4 kEUR more than the plan without storage in every investment year" '
                 'is checked in Part 1 (derived)'},
        f"l. {defer[0]['line']}" if defer else None)
    ki = _hits(lines, r'Key Insights|subsec:res_insights|datasheet calibration and the 0\.70 floor')
    removed = [ln for ln in difflib.unified_diff(prev6ff_lines, lines, lineterm='', n=0)
               if ln.startswith('-') and not ln.startswith('---')]
    row('Blocked 6 (l. 484)', '4.9 (i) "with the datasheet calibration and the 0.70 floor" ambiguous (baseline vs C4 row)',
        'no longer applicable' if not ki else 'still present',
        {'found_now': ki, 'main_tex_lines_removed_6ff8f3c_to_e10c104': len(removed),
         'removed_key_insights_heading': any('Key Insights' in ln for ln in removed),
         'removed_datasheet_clause': any('datasheet calibration and the 0.70 floor' in ln for ln in removed),
         'note': 'section 4.9 Key Insights was deleted at 51a6fb5 (W177b-1) and is still absent at e10c104'})
    floor = _hits(lines, r'Baseline, 0\.50 floor', scopes['4.3'])
    cells = [c.strip() for c in floor[0]['text'].rstrip('\\').split('&')] if floor else []
    row('Blocked 7 / C5', 'optional fill of the 0.50-floor row: value 258.5 kEUR, AE 0.778, EFC/day 1.08, floor never (W155, '
        'not in the frozen T1-T11)', 'still open (row not filled; author\'s option)' if cells and cells[1] == '---' else 'filled',
        {'row': floor, 'cells': cells, 'W177a_values': cmap['C5']['fill_for_row_262'],
         'note': 'value, AE and EFC/day print "---"; value - I, threshold, multiple and "never" are checked in Part 1'},
        f"l. {floor[0]['line']}" if floor else None)

    def plan_ok(rng):
        ts = [t for t in toks if t['file'] == 'main.tex' and rng[0] <= t['line'] <= rng[1] and t.get('rule') == 'plan (candidate record)']
        return len(ts), sum(1 for t in ts if t['status'] == 'match')
    r2 = (M[1189], M[1204])
    n2, m2 = plan_ok(r2)
    row('C2', 'S / E / duration of each B and C cell confirmed (16/16)',
        'holds' if n2 and n2 == m2 and cmap['C2']['outcome'] == 'confirmed' else 'differs',
        {'plan_tokens_rows': {'lines': list(r2), 'n': n2, 'match': m2}, 'recomputed_C2': cmap['C2']['summary']}, f'l. {r2[0]}-{r2[1]}')
    r3 = (M[1290], M[1301])
    n3, m3 = plan_ok(r3)
    row('C3', 'I/J cells = node-7 4 h plans of 2-5 MWh (keys confirmed)',
        'holds' if n3 and n3 == m3 and cmap['C3']['outcome'] == 'confirmed' else 'differs',
        {'plan_tokens_caption_and_rows': {'lines': list(r3), 'n': n3, 'match': m3}, 'recomputed_C3': cmap['C3']['summary']},
        f'l. {r3[0]}-{r3[1]}')
    c16 = _hits(lines, r'16 cycles|sixteen|16~cycles', sec4)
    cont = _hits(lines, r'was continued beyond its stopping point', scopes['4.5'])
    capd = _hits(lines, r'evaluation continued past it', scopes['4.5'])
    row('C6', 'the 3 x 3 "16 cycles" confirmed; wording: the evaluation, not the fit, was continued',
        'fixed' if cont and not c16 else 'still present',
        {'16_cycles_printed': c16,
         'prose': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'the evaluation of the plan without storage was continued', 260)} for h in cont],
         'caption': [{'line': h['line'], 'quote': _quote(lines, h['line'], '$D$ is the post-hoc', 160)} for h in capd],
         'recomputed_C6': cmap['C6']['summary']}, ', '.join(f"l. {h['line']}" for h in cont + capd))
    two = _hits(lines, r'changes two verdicts|It changes two verdicts', scopes['4.6'])
    yl_in_60 = [c['claim_id'] for c in X.w153s['claims'] if set(c.get('ref_cells', []) + c.get('other_cells', [])) &
                {'yl_y2030', 'yl_y2035'}]
    row('C7', 'W153: 0 sign changes in 60 claims; ONE verdict changes category (the F2 neighbour)',
        'changed (the text says two verdicts)' if two else 'holds',
        {'text': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'Across the 60', 420)} for h in two],
         'recomputed_C7': cmap['C7']['summary'],
         'year_ladder_pair_among_the_60': bool(yl_in_60), 'W153_claims_with_a_year_ladder_cell': yl_in_60,
         'year_ladder': {'D_gross': X.t['year_ladder']['D_gross'], 'D_net': X.t['year_ladder']['D_net'],
                         'gross_verdict': X.t['year_ladder']['gross_v6']['verdict'], 'net_verdict': X.t['year_ladder']['net_v6']['verdict']},
         'note': 'carried to Part 1 ("two": MISMATCH against W153 n_verdict_differs = 1) and to W177b-2\' Part 2 (c)'},
        f"l. {two[0]['line']}" if two else None)
    r25 = _hits(lines, r'^\s*2025 \(reference\)\s*&', scopes['4.6'])
    row('C8', 'the 2025 cohort\'s salvage is zero (the row may print 0.0)',
        'holds' if r25 and '0.0' in r25[0]['text'].split('&')[2] and cmap['C8']['outcome'] == 'confirmed' else 'differs',
        {'row': r25, 'recomputed_C8': cmap['C8']['summary'][:300]}, f"l. {r25[0]['line']}" if r25 else None)
    r41 = _hits(lines, r'41 new recourse evaluations', scopes['4.7'])
    row('C9', '41 new recourse evaluations (14, 20, 7 against 20, 20, 60) confirmed',
        'holds' if r41 and cmap['C9']['outcome'] == 'confirmed' else 'differs',
        {'text': [{'line': h['line'], 'quote': _quote(lines, h['line'], 'made 41', 120)} for h in r41], 'recomputed_C9': cmap['C9']['summary']},
        f"l. {r41[0]['line']}" if r41 else None)

    confirms = _hits(lines, r'\[CONFIRM')
    g_ok = all(X.CL[g]['gross_verdict'] == 'determinate' and X.CL[g]['net_verdict_recomputed_here'] == 'determinate' and
               X.CL[g]['net_verdict_W153'] == 'determinate' for g in B1.G)
    g_lines = (M[1445], M[1446])
    g_toks = [t for t in toks if t['file'] == 'main.tex' and t['line'] in g_lines and t['status'] not in ('excluded',)]
    re_c = []
    for h in confirms:
        ln = h['line']
        if 'eq:salvage_credit' in h['text']:
            re_c.append({'line': ln, 'point': 'salvage equation items (a)-(d)', 'outcome':
                         'items (a)-(d) hold as worded (recomputed C1: ' + ', '.join(f"{k.split(')')[0]})" + ' ' + v['outcome'].split(' ')[0]
                                                                                    for k, v in c1['items'].items()) +
                         '); the equation above them still needs the two corrections (Blocked 1)'})
        elif 'G rows' in h['text']:
            re_c.append({'line': ln, 'point': 'G rows of T1 (node 7, 0.5 MVA / 1 and 2 MWh, 2030 / 2035) gross / net, all determinate',
                         'outcome': ('confirmed' if g_ok and g_toks and all(t['status'] == 'match' for t in g_toks) else 'differs') +
                         f": {sum(t['status'] == 'match' for t in g_toks)}/{len(g_toks)} tokens of l. {g_lines[0]}-{g_lines[1]} "
                         f"match the frozen T1 G claims; every G claim determinate gross and net (W153 and recomputed): {g_ok}"})
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


def flag_checks_e10c104(X, new_lines, old_n, M):
    """W177b-1's flag_checks (51a6fb5-pinned lines) on a 51a6fb5-indexed view of the e10c104 lines."""
    view = [new_lines[M[k] - 1] if k in M else '' for k in range(1, old_n + 1)]
    out = B1.flag_checks(X, view)
    for c in out:
        c['line_51a6fb5'] = c['line']
        c['line'] = M.get(c['line'])
        c['mapped'] = c['line'] is not None
    return out


def part_of(t):
    envs = t.get('envs') or []
    if t['scope'] == 'comment':
        return 'comment'
    if t['file'] == LETTER:
        return {'rcomment': 'reviewer quotation', 'rresponse': 'response', 'rchanges': 'changes'}.get(t.get('macro'), 'prose')
    if 'minipage' in envs and 'table' in envs:
        return 'table note'
    if 'tabular' in envs:
        return 'table body'
    if 'table' in envs:
        return 'table caption'
    if 'figure' in envs or 'figure*' in envs:
        return 'figure caption'
    if any(e.startswith('equation') for e in envs):
        return 'equation'
    return 'prose'


# ======================================================================================================================
#  5. guards, post-run, summary
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    _g, pk_b1, _ok = B1.guards_state()           # its own "blocked" flag is False here (this script's blockers are installed)
    counts = dict(pk_b1['counts'], w177b1r=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f"[W177b-1' post-run PRECONDITION FAILED] {rel} missing")
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W177b-1' post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


_md_cell = B1._md_cell


def _tok_line(t):
    return (f"{t['file']} l. {t['line']}:{t['col']} ({t['scope_name']}, {t['part']}) `{t['written']}` -- {t['status']}: "
            f"{_md_cell(t.get('counterpart') or t.get('category'), 260)}")


def md_summary(o):
    L = [f"# W177b-1' -- Parts 0 and 1 at Overleaf {o['clone']['head'][:7]} (main.tex sha256 {o['clone']['tex_sha256']['main.tex'][:8]}), "
         f"section 3 opening, letter", '',
         f"Zero solves; script `{SCRIPT_REL}` (imports `{B1_SCRIPT}`, not edited); frozen tables `{FZ_REL}` (sha256 {FZ_SHA[:8]}); "
         f"clone `{CLONE}` (read only). Objective convention: {o['objective_convention']}", '',
         '## Predictions (expert, Addendum 76)', '']
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
    L += ['## Part 1 -- counts by scope (main.tex, highlights.tex)', '', '| scope | lines | ' + ' | '.join(STATUSES[:7]) + ' | total |',
          '|---|---|' + '---|' * 8]
    for sc, cnt in s['by_scope'].items():
        L.append(f"| {sc}{' (reported separately)' if sc in SEPARATE else ''} | {s['scope_lines'].get(sc, '')} | " +
                 ' | '.join(str(cnt.get(k, 0)) for k in STATUSES[:7]) + f" | {sum(cnt.values())} |")
    L += ['', f"In scope: {s['in_scope']}; reported separately: {s['separate']}", '',
          '### Lines changed since 51a6fb5 inside the Part 1 scopes', '']
    for h in o['changes_since_51a6fb5']['in_scope_hunks']:
        L.append(f"- {h['op']} 51a6fb5 l. {h['old'][0]}-{h['old'][1]} -> e10c104 l. {h['new'][0]}-{h['new'][1]} ({h['scope']}): "
                 f"tokens removed {h['tokens_removed']}, added {h['tokens_added']}, kept {h['tokens_kept']}")
    L += ['', '### New or changed tokens (in scope)', '']
    for t in o['tokens']:
        if t['file'] != LETTER and t['scope_name'] in IN_SCOPE + SEPARATE and t.get('since_51a6fb5') == 'new or changed line' \
                and t['status'] != 'excluded':
            L.append('- ' + _tok_line(t))
    L += ['', '### MISMATCH (all scopes)', '']
    for t in o['tokens']:
        if t['status'] == 'MISMATCH':
            L.append(f"- {_tok_line(t)} [{_md_cell(t.get('value_at_written_precision'))}] {_md_cell(t.get('note'), 700)}")
    L += ['', '### DERIVED (arithmetic, all scopes)', '']
    for t in o['tokens']:
        if t.get('derived'):
            L.append(f"- {t['file']} l. {t['line']}:{t['col']} ({t['scope_name']}) `{t['written']}` -- {t['status']} "
                     f"(holds {t['holds']}): {_md_cell(t.get('counterpart'), 700)}")
    L += ['', '## Amendment (1) -- section 3 opening', '', f"Scope l. {o['sec3']['scope'][0]}-{o['sec3']['scope'][1]}; counts "
          f"{o['sec3']['counts']}; M1 texts at the HEAD: {o['sec3']['m1_present']}; submitted sentences gone: "
          f"{o['sec3']['submitted_sentences_absent']}", '', '| line | written | status | value | source |', '|---|---|---|---|---|']
    for t in o['tokens']:
        if t['scope_name'] == SEC3 and t['status'] != 'excluded':
            L.append(f"| {t['line']} | {_md_cell(t['written'])} | {t['status']} | {_md_cell(t.get('value_at_written_precision'), 30)} | "
                     f"{_md_cell(t.get('counterpart') or t.get('category'), 220)} |")
    L += ['', 'Case files: ' + '; '.join(f"{k} {v['path']} sha256 {v['sha256'][:8]} blob {v['blob'][:8]} (campaign spec "
                                         f"records {v['spec_sha256'][:8]}, equal {v['equal_to_spec']})" for k, v in o['sec3']['case_files'].items()), '']
    L += ['## Amendment (2) -- the letter', '', f"counts {o['letter']['counts']}; by rule {o['letter']['by_rule']}; letter = "
          f"response_to_reviewers_v2.tex: {o['letter']['equals_v2']}", '', '| line | part | written | status | value | source / category | note |',
          '|---|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == LETTER and t['status'] != 'excluded':
            L.append(f"| {t['line']} | {t['part']} | {_md_cell(t['written'])} | {t['status']} | "
                     f"{_md_cell(t.get('value_at_written_precision'), 30)} | {_md_cell(t.get('counterpart') or t.get('category'), 200)} | "
                     f"{_md_cell(t.get('note'), 160)} |")
    L += ['', '### Reviewer quotations (W171a check)', '', '| quote | segment | letter lines | chars | result | ratio | docx paragraph | differences |',
          '|---|---|---|---|---|---|---|---|']
    for r in o['letter']['quotations']:
        d = '; '.join(f"{x['op']} {x['letter']!r} -> {x['document']!r} (l. {x['letter_line']})" for x in r['diffs'])
        L.append(f"| {r['quote']} | {r['segment']} | {r['lines']} | {r['chars']} | {r['status']} | {r['ratio']:.3f} | "
                 f"{r['docx_paragraph']} | {_md_cell(d, 600)} |")
    L += ['', '## Flag and verdict checks (tables)', '']
    for c in o['flag_checks']:
        L.append(f"- l. {c['line']} (51a6fb5 l. {c['line_51a6fb5']}) {c['what']}: {'ok' if c['ok'] else 'NOT OK'}")
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
    tag = "W177b-1'"
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
    # ---- clone ----------------------------------------------------------------------------------------------------------
    mabs = os.path.join(REPO, CLONE)
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex', BIB], capture_output=True,
                               text=True).stdout.strip()
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
    for f in tex_now + [BIB]:
        hb = subprocess.run(['git', '-C', mabs, 'hash-object', f], capture_output=True, text=True).stdout.strip()
        hh = subprocess.run(['git', '-C', mabs, 'rev-parse', f'HEAD:{f}'], capture_output=True, text=True).stdout.strip()
        blob_eq[f] = bool(hb) and hb == hh
    if porcelain:
        pre.append(f'clone *.tex / {BIB} not clean: {porcelain!r}')
    # ---- inputs ---------------------------------------------------------------------------------------------------------
    all_inputs = dict(W177A.INPUTS)
    all_inputs.update(B1.EXTRA_INPUTS)
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
    if 'B1_SCRIPT' in inputs and inputs['B1_SCRIPT']['last_commit'] is None:
        pre.append(f'{B1_SCRIPT} has no commit')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] preconditions hold: clone {head[:8]} ({len(tex_now)} *.tex as declared: ' +
         ', '.join(f'{f} {tex_sha[f][:8]}' for f in tex_now) + f'); {len(inputs)} inputs committed clean; script committed '
         f'clean {script_clean}')

    # ---- context --------------------------------------------------------------------------------------------------------
    X = W177A.Ctx(inputs)
    for k in B1.EXTRA_JSON + EXTRA_JSON:
        X.j[k] = _json(inputs[k]['path'])
    for k in EXTRA_TEXT:
        X.texts[k] = _text(inputs[k]['path'])
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
    X.main_lines = main_lines
    hl_text = _text(os.path.join(CLONE, 'highlights.tex'))
    letter_text = _text(os.path.join(CLONE, LETTER))
    X.letter_raw = letter_text
    X.bib = _text(os.path.join(CLONE, BIB))
    dx = W171.docx_text(W171.DOCX)
    X.docx_text = dx['text']
    m28 = re.search(r'\\subsection\*\{R2\.8[^}]*\}\s*\\rcomment\{(.*?)\n\}', letter_text, re.S)
    X.r28_comment = m28.group(1) if m28 else ''
    frame_lines = X.frame.split('\n')

    def show(commit, f):
        r = subprocess.run(['git', '-C', mabs, 'show', f'{commit}:{f}'], capture_output=True, text=True)
        return r.stdout.split('\n') if r.returncode == 0 else []
    prev_lines = show(PREV_COMMIT, 'main.tex')
    prev6ff_lines = show(PREV_6FF, 'main.tex')
    letter_prev = show(PREV_COMMIT, LETTER)
    letter_w176 = show(LETTER_PREV_CHECK, LETTER)
    M, hunks = line_map(prev_lines, main_lines)
    ML_prev, lhunks_prev = line_map(letter_prev, letter_text.split('\n'))
    ML_w176, lhunks_w176 = line_map(letter_w176, letter_text.split('\n'))
    scopes, labels = B1.compute_scopes(main_lines)
    scopes_ok = list(scopes.items()) == list(EXPECTED_SCOPES.items())
    labels_ok = all(labels.get(k) == v for k, v in B1.SUBSEC_LABELS.items())
    s3a = next(i + 1 for i, s in enumerate(main_lines) if s.startswith('\\section{Case Study}'))
    s3b = next(i + 1 for i in range(s3a, len(main_lines)) if main_lines[i].startswith('\\subsection{Investment Costs}')) - 1
    sec3_ok = (s3a, s3b) == EXPECTED_SEC3
    all_scopes = collections.OrderedDict(scopes)
    all_scopes[SEC3] = (s3a, s3b)
    hl_main = [s.strip() for s in main_lines[scopes['highlights (main.tex block)'][0]:scopes['highlights (main.tex block)'][1] - 1]
               if s.strip().startswith('\\item')]
    hl_file = [s.strip() for s in hl_text.split('\n') if s.strip().startswith('\\item')]

    # ---- tokens: main.tex scopes, highlights.tex, section 3 opening ------------------------------------------------------
    toks = []
    inv = {v: k for k, v in M.items()}
    for fname, text in (('main.tex', main_text), ('highlights.tex', hl_text)):
        doc = W164.Doc(fname, text)
        for t in W174.tokenize_v3(doc):
            t['file'] = fname
            sc = B1.scope_of(fname, t['line'], all_scopes)
            if sc is None or (fname == 'main.tex' and sc not in IN_SCOPE + SEPARATE + (SEC3,)):
                continue
            t['scope_name'] = sc
            t['status'] = None
            if t['excluded']:
                t.update(status='excluded', category=f"structure: {t['excluded']}")
            elif t['type'] in W177A.IDENT_TYPES:
                t.update(status='excluded', category=f"identifier ({t['type']})")
            if fname == 'main.tex':
                t['since_51a6fb5'] = (f"unchanged line (51a6fb5 l. {inv[t['line']]})" if t['line'] in inv else 'new or changed line')
                t['line_51a6fb5'] = inv.get(t['line'])
            else:
                t['since_51a6fb5'] = 'unchanged file (sha256 as at 51a6fb5)'
                t['line_51a6fb5'] = t['line']
            toks.append(t)
    # ---- tokens: the letter ----------------------------------------------------------------------------------------------
    ldoc = W164.Doc(LETTER, letter_text)
    ltoks = W174.tokenize_v3(ldoc)
    for t in ltoks:
        t['file'] = LETTER
    qrows = W171.quotation_check(ldoc, dx)
    qstat = W171.quotation_token_status(qrows, ldoc, ltoks)
    inv_l_prev = {v: k for k, v in ML_prev.items()}
    inv_l_w176 = {v: k for k, v in ML_w176.items()}
    for t in ltoks:
        t['scope_name'] = 'letter'
        t['status'] = None
        t['since_51a6fb5'] = (f"unchanged line (51a6fb5 l. {inv_l_prev[t['line']]})" if t['line'] in inv_l_prev else 'new or changed line')
        t['since_8b76423_W176b'] = (f"unchanged line (8b76423 l. {inv_l_w176[t['line']]})" if t['line'] in inv_l_w176
                                    else 'new or changed line')
        t['line_8b76423'] = inv_l_w176.get(t['line'])
        if t['excluded']:
            t.update(status='excluded', category=f"structure: {t['excluded']}")
        elif t['scope'] == 'body' and t.get('macro') == 'rcomment':
            r = qstat.get(t['key'])
            if r is None:
                continue
            t.update({k: v for k, v in r.items() if v is not None})
            t['rule_assigned'] = 'reviewer quotation (W171a quotation check against the reviewers\' document)'
        elif t['type'] in ('label', 'hash', 'date', 'alnum', 'doi'):
            t.update(status='excluded', category=f"identifier ({t['type']})")
        elif t['scope'] == 'body' and W171._in_xref(ldoc, t):
            ln = ldoc.lines[t['line'] - 1]
            mt = next(m for m in W171.XREF_RX.finditer(ln) if m.start() <= t['col0'] < m.end())
            if mt.group().startswith('Reviewer'):
                t.update(status='unchecked', category='enumerator', note=f'reviewer number ({mt.group()!r})')
            else:
                t.update(status='unchecked', category="pointer, checked by W177b-3'", note=f'pointer {mt.group()!r}')
            t['rule_assigned'] = 'pointer rule (W171a XREF_RX)'
    toks += ltoks

    # ---- declarations -----------------------------------------------------------------------------------------------------
    relocated, carried, retired, unmapped, carry_refused = relocate_b1(M, main_lines, frame_lines)
    new_main = main_new_declarations()
    sec3_d = sec3_declarations()
    letter_d = letter_declarations()
    retired_replaced_ok = all(set(r['replaced_by'] or ()) <= {d['line'] for d in new_main} for r in retired) and \
        sorted(r['line_51a6fb5'] for r in retired) == sorted(RETIRED)
    decls_all = relocated + carried + new_main + sec3_d + letter_d
    texts = {'main.tex': main_lines, 'highlights.tex': hl_text.split('\n'), LETTER: letter_text.split('\n')}
    stale, claimed_twice, decl_out, dup_decl = apply_declarations(X, toks, decls_all, texts)
    unassigned = [t for t in toks if t['status'] is None]
    for t in toks:
        t['part'] = part_of(t)

    # ---- section 3 opening: M1 identity, submitted sentences, case files ---------------------------------------------------
    m1_blocks, m23_lines = _fix_texts(X.texts['FIXES_M1'])
    sec3_flat = ' '.join(' '.join(main_lines[s3a - 1:s3b]).split())
    main_flat = ' '.join(' '.join(W164.split_comment(ln)[0] for ln in main_lines).split())
    m1_present = {f"M1 ({'abc'[i] if i < 3 else i})": b in sec3_flat for i, b in enumerate(m1_blocks)}
    m1_present.update({f"M{2 + i} (anywhere in main.tex, outside comments)": p in main_flat for i, p in enumerate(m23_lines)})
    sub_old = ('represented by five years', 'five load/RES scenarios', '25 operating scenarios',
               'used as a baseline to quantify the overall benefits')
    sub_absent = {s: s not in sec3_flat for s in sub_old}
    di = X.j['SPEC_3X3']['configuration']['derived_instance']
    case_files = {}
    for k, rel in CASE_FILES.items():
        spec_sha = di['case_sha256'] if k == 'CASE_3X3' else di['source_case_sha256']
        case_files[k] = {'path': rel, 'sha256': _sha(rel), 'blob': _git('rev-parse', f'HEAD:{rel}'),
                         'committed_clean': W160.W157.L132._committed_clean(rel), 'last_commit': W160._last_commit(rel),
                         'spec_sha256': spec_sha, 'equal_to_spec': _sha(rel) == spec_sha,
                         'instance_record_sha256': (X.j['INSTANCE_3X3']['case_sha256'] if k == 'CASE_3X3'
                                                    else X.j['INSTANCE_3X3']['source_case']['sha256'])}
    sec3_toks = [t for t in toks if t['scope_name'] == SEC3]

    # ---- summaries -------------------------------------------------------------------------------------------------------
    order = list(EXPECTED_SCOPES)[:2] + ['highlights.tex'] + list(EXPECTED_SCOPES)[2:]
    main_toks = [t for t in toks if t['file'] != LETTER and t['scope_name'] != SEC3]
    by_scope = collections.OrderedDict((sc, dict(collections.Counter(t['status'] for t in main_toks if t['scope_name'] == sc)))
                                       for sc in order)
    scope_lines = {sc: f'{a}-{b}' for sc, (a, b) in all_scopes.items()}
    scope_lines['highlights.tex'] = f"1-{len(hl_text.rstrip(chr(10)).split(chr(10)))}"
    ins = dict(collections.Counter(t['status'] for t in main_toks if t['scope_name'] in IN_SCOPE))
    sep = dict(collections.Counter(t['status'] for t in main_toks if t['scope_name'] in SEPARATE))
    by_part = {p: dict(collections.Counter(t['status'] for t in main_toks if t['part'] == p)) for p in sorted({t['part'] for t in main_toks})}
    mism = [t for t in main_toks if t['status'] == 'MISMATCH']
    mism_in = [t for t in mism if t['scope_name'] in IN_SCOPE]
    tab_parts = ('table body', 'table caption', 'table note', 'figure caption')
    tab_toks = [t for t in main_toks if t['scope_name'] in IN_SCOPE and t['part'] in tab_parts and t['status'] != 'excluded']
    mism_tab = [t for t in mism_in if t['part'] in tab_parts]
    mism_frozen = [t for t in mism_in if FZ_REL in str(t.get('counterpart')) or 'cells[' in str(t.get('counterpart'))]
    ders = [t for t in main_toks if t.get('derived') and t['scope_name'] in IN_SCOPE + SEPARATE]
    ders_all = [t for t in toks if t.get('derived')]
    named = {'24': None, '33': None, '7': [], '9': [], '90.9': None, '654': None, '9.3': None, '42.1': None, '28.5': None}
    for t in ders:
        if t['written'] in named:
            if isinstance(named[t['written']], list):
                named[t['written']].append({'line': t['line'], 'holds': t['holds']})
            else:
                named[t['written']] = {'line': t['line'], 'holds': t['holds']}
    # the in-scope hunks since 51a6fb5 (token multisets from W177b-1's committed JSON vs now)
    b1_toks = [t for t in X.j['B1_JSON']['tokens'] if t['file'] == 'main.tex']
    in_hunks = []
    for h in hunks:
        a, b = h['new']                           # e10c104 lines a..b (empty when b < a: a pure deletion)
        oa, ob = h['old']                         # 51a6fb5 lines oa..ob (empty when ob < oa: a pure insertion)
        scs = [sc for sc, (x, y) in all_scopes.items() if sc in IN_SCOPE + SEPARATE and b >= a and not (b < x or a > y)]
        scs += [sc for sc, (x, y) in B1.EXPECTED_SCOPES.items() if sc in IN_SCOPE + SEPARATE and ob >= oa and
                not (ob < x or oa > y) and sc not in scs]
        if not scs:
            continue
        old_w = collections.Counter(t['written'] for t in b1_toks if oa <= t['line'] <= ob and t['status'] != 'excluded')
        new_w = collections.Counter(t['written'] for t in main_toks if t['file'] == 'main.tex' and a <= t['line'] <= b and
                                    t['status'] != 'excluded')
        in_hunks.append({**h, 'scope': ', '.join(scs), 'tokens_removed': sorted((old_w - new_w).elements()),
                         'tokens_added': sorted((new_w - old_w).elements()), 'tokens_kept': sorted((old_w & new_w).elements())})
    # W177b-1 statuses reproduced on unchanged lines
    b1_status = {(t['file'], t['line'], t['col'], t['written']): t['status'] for t in X.j['B1_JSON']['tokens']}
    repro = {'n': 0, 'equal': 0, 'differ': []}
    for t in main_toks:
        if t['scope_name'] not in IN_SCOPE + SEPARATE or t.get('line_51a6fb5') is None:
            continue
        k = (t['file'], t['line_51a6fb5'], t['col'], t['written'])
        if k not in b1_status:
            repro['differ'].append({'key': list(k), 'why': 'not in W177b-1 JSON'})
            continue
        repro['n'] += 1
        if b1_status[k] == t['status']:
            repro['equal'] += 1
        else:
            repro['differ'].append({'key': list(k), 'w177b1': b1_status[k], 'now': t['status']})
    # the letter
    lt = [t for t in toks if t['file'] == LETTER]
    l_counts = dict(collections.Counter(t['status'] for t in lt))
    l_rule = dict(collections.Counter(t.get('rule_assigned') or ('declared' if t.get('declared') else t.get('category', 'excluded'))
                                      for t in lt))
    l_mism = [t for t in lt if t['status'] == 'MISMATCH']
    l_changed = dict(collections.Counter((t['since_8b76423_W176b'] != 'new or changed line', t['status']) for t in lt
                                         if t['status'] != 'excluded'))
    w176_l = {(t['line'], t['col'], t['written']): t['status'] for t in X.j['W176B_JSON']['tokens'] if t['file'] == LETTER}
    w176_cmp = collections.Counter()
    w176_diff = []
    for t in lt:
        if t.get('line_8b76423') is None or t['status'] == 'excluded':
            continue
        k = (t['line_8b76423'], t['col'], t['written'])
        old = w176_l.get(k)
        w176_cmp[(old, t['status'])] += 1
        if old is not None and old != t['status'] and not (old == 'no table counterpart' and t['status'] == 'no counterpart'):
            w176_diff.append({'line': t['line'], 'line_8b76423': t['line_8b76423'], 'written': t['written'], 'w176b': old,
                              'now': t['status'], 'category_now': t.get('category')})
    w176_q = X.j['W176B_JSON'].get('quotations', [])
    q_summary = {'now': dict(collections.Counter(r['status'] for r in qrows)), 'n_now': len(qrows),
                 'w176b_8b76423': dict(collections.Counter(r['status'] for r in w176_q)), 'n_w176b': len(w176_q)}
    letter_equals_v2 = tex_sha.get(LETTER) == inputs['LETTER_V2']['sha256']

    p0rows, p0conf, p0same = part0(X, main_lines, scopes, confirm_now, X.j['W177A_JSON'], toks, prev6ff_lines, M)
    flags = flag_checks_e10c104(X, main_lines, len(prev_lines), M)
    b1_pred = {p['item']: p['outcome'] for p in X.j['B1_JSON']['prediction']}
    prediction = [
        {'item': 'Part 1: zero MISMATCH in numbers carried from the tables',
         'outcome': (('HELD' if not mism_in else 'FAILED as worded') +
                     f": {len(mism_in)} MISMATCH among {sum(1 for t in main_toks if t['scope_name'] in IN_SCOPE and t['status'] != 'excluded')} "
                     f"in-scope tokens; {len(mism_tab)} among the {len(tab_toks)} non-excluded tokens in table bodies / captions / "
                     f"notes / figure captions; {len(mism_frozen)} against a frozen-table field, the rest against a named record" +
                     ('' if not mism_in else ' -- ' + '; '.join(f"l. {t['line']} '{t['written']}' -> {t.get('value_at_written_precision')} "
                                                               f"({str(t.get('counterpart'))[:120]})" for t in mism_in))),
         'at_51a6fb5_W177b1': b1_pred.get('Part 1: zero MISMATCH in numbers carried from the tables')},
        {'item': 'Part 1: the derived numbers hold',
         'outcome': ('HELD' if all(t['holds'] for t in ders) else 'FAILED') +
                    f": {sum(t['holds'] for t in ders)}/{len(ders)} derived tokens (Part 1 scopes) hold at the written "
                    f"precision; named: " + ', '.join(f'{k} {v}' for k, v in named.items()) +
                    f"; all scopes incl. section 3 opening and the letter: {sum(t['holds'] for t in ders_all)}/{len(ders_all)} hold",
         'at_51a6fb5_W177b1': b1_pred.get('Part 1: the derived numbers hold')},
    ]
    checks = {
        'clone_head_is_declared': head == CLONE_HEAD,
        'every_clone_tex_sha_equals_declared_and_expect': all(TEX_SHA.get(f) == tex_sha[f] == expect.get(f) for f in tex_now),
        'every_clone_tex_and_bib_equals_its_HEAD_blob': all(blob_eq.values()),
        'clone_tex_and_bib_clean': porcelain == '',
        'main_tex_line_count': len(main_lines) - (1 if main_text.endswith('\n') else 0) == MAIN_LINES_N,
        'previous_heads_readable': bool(prev_lines) and bool(prev6ff_lines) and bool(letter_prev) and bool(letter_w176),
        'scopes_equal_declared': scopes_ok,
        'section3_opening_scope_equals_declared': sec3_ok,
        'subsection_labels_as_declared': labels_ok,
        'highlights_main_block_equals_highlights_tex': hl_main == hl_file and len(hl_main) == 5,
        'letter_equals_response_to_reviewers_v2': letter_equals_v2,
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'every_carry_line_identical_to_its_frame_line': not carry_refused,
        'every_w177b1_declaration_relocated_or_retired_with_replacement': not unmapped and retired_replaced_ok,
        'every_declaration_found_and_evaluated': not stale,
        'no_line_declared_twice': not dup_decl,
        'no_token_claimed_twice': not claimed_twice,
        'every_token_assigned': not unassigned,
        'statuses_in_vocabulary': all(t['status'] in STATUSES for t in toks),
        'every_input_committed_clean': all(v['committed_clean'] for v in inputs.values()),
        'case_files_committed_clean_and_equal_to_campaign_spec': all(v['committed_clean'] and v['equal_to_spec'] and
                                                                     v['sha256'] == v['instance_record_sha256'] for v in case_files.values()),
        'production_salvage_form_reproduces_records': next(c for c in confirm_now if c['id'] == 'C1')['production_reproduces_records'],
        'w177a_confirm_outcomes_reproduced': all(v['equal'] for v in p0same.values()),
        'w177b1_statuses_reproduced_on_unchanged_lines': repro['n'] > 0 and repro['equal'] == repro['n'] and not repro['differ'],
        'flag_check_lines_all_mapped': all(c['mapped'] for c in flags),
        'every_quotation_segment_located': all(r['status'] != 'not found' for r in qrows),
        'r28_comment_found': bool(X.r28_comment),
    }
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
    out = {'schema': 'p515_s53_w177b1r_number_check', 'version': 1,
           'stage': "P5.15 Addendum 76 + Amendment, W177b-1' -- Part 0, Part 1, section 3 opening and letter numbers at Overleaf e10c104",
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {B1_SCRIPT: _sha(B1_SCRIPT), B1.EXTRA_INPUTS['W177A_SCRIPT']: _sha(B1.EXTRA_INPUTS['W177A_SCRIPT'])}},
           'command_line': sys.argv,
           'clone': {'dir': CLONE, 'head': head, 'declared_commit': DECLARED_COMMIT, 'tex_sha256': tex_sha,
                     'bib_sha256': _sha(os.path.join(CLONE, BIB)), 'tex_and_bib_equal_head_blob': blob_eq,
                     'porcelain_tex_and_bib': porcelain, 'edited': False,
                     'main_tex_lines': len(main_lines) - (1 if main_text.endswith('\n') else 0),
                     'previous_commit_W177b1': PREV_COMMIT, 'letter_previous_check_W176b': LETTER_PREV_CHECK},
           'objective_convention': X.fz.get('objective_convention'),
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA},
           'scopes': {'main.tex': {k: list(v) for k, v in all_scopes.items()}, 'highlights.tex': scope_lines['highlights.tex'],
                      'in_scope': list(IN_SCOPE), 'reported_separately': list(SEPARATE), 'section3_opening': [s3a, s3b],
                      'section3_core_paragraphs': list(SEC3_CORE), 'letter': f'1-{len(letter_text.rstrip(chr(10)).split(chr(10)))}',
                      'labels': labels},
           'inputs': inputs,
           'prediction': prediction,
           'part0': {'rows': p0rows, 'confirm_comments': p0conf, 'confirm_recomputed_vs_committed': p0same,
                     'confirm_recomputed': confirm_now, 'w177a_json': inputs['W177A_JSON']},
           'changes_since_51a6fb5': {'main_tex_hunks': hunks, 'in_scope_hunks': in_hunks, 'retired_declarations': retired,
                                     'unmapped_declarations': unmapped, 'w177b1_statuses_on_unchanged_lines': repro,
                                     'letter_hunks_vs_51a6fb5': lhunks_prev},
           'summary': {'n_tokens_main_scopes': len(main_toks), 'by_status': dict(collections.Counter(t['status'] for t in main_toks)),
                       'by_scope': by_scope, 'scope_lines': scope_lines, 'in_scope': ins, 'separate': sep, 'by_part': by_part,
                       'n_mismatch_in_scope': len(mism_in), 'n_mismatch_tables_captions_notes': len(mism_tab),
                       'n_table_tokens_in_scope': len(tab_toks), 'n_derived_part1': len(ders),
                       'n_derived_part1_hold': sum(t['holds'] for t in ders), 'named_derived': named,
                       'n_derived_all': len(ders_all), 'n_derived_all_hold': sum(t['holds'] for t in ders_all),
                       'n_declarations': {'relocated_w177b1': len(relocated), 'carried_frame': len(carried), 'new_main': len(new_main),
                                          'section3_opening': len(sec3_d), 'letter': len(letter_d)},
                       'n_declarations_applied': len(decl_out)},
           'sec3': {'scope': [s3a, s3b], 'counts': dict(collections.Counter(t['status'] for t in sec3_toks)),
                    'm1_blocks_in_fix_file': len(m1_blocks), 'm1_present': m1_present, 'submitted_sentences_absent': sub_absent,
                    'case_files': case_files},
           'letter': {'path': os.path.join(CLONE, LETTER), 'sha256': tex_sha.get(LETTER), 'equals_v2': letter_equals_v2,
                      'counts': l_counts, 'by_rule': l_rule, 'n_mismatch': len(l_mism),
                      'by_unchanged_since_w176b_and_status': {f'{k[0]}|{k[1]}': v for k, v in l_changed.items()},
                      'hunks_vs_8b76423': lhunks_w176, 'w176b_status_pairs_on_unchanged_lines': {f'{k[0]}->{k[1]}': v for k, v in w176_cmp.items()},
                      'w176b_status_differences': w176_diff, 'quotations': qrows, 'quotation_summary': q_summary,
                      'docx': {k: v for k, v in dx.items() if k not in ('text', 'para_of')}},
           'conventions': {'column': '1-based', 'line': '1-based',
                           'statuses': 'match = equal to the frozen JSON (or the named record) at the export\'s printed rounding '
                                       '(export/README.md); approximate = printed coarser than that rule and equal at its own '
                                       'precision; derived = computed from frozen values / named records (arithmetic recorded) '
                                       'and equal at the written precision; MISMATCH; no counterpart; unchecked (category); '
                                       'excluded (structure / identifier); reviewer quotation, verified (letter \\rcomment: the '
                                       'token lies in a part of the quotation equal to the reviewers\' document)',
                           'sign': 'a written sign (- / + / $-$) is compared signed; an unsigned figure by magnitude'},
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
    for v in case_files.values():
        man[v['path']] = v['sha256']
    for f in tex_now + [BIB]:
        man[os.path.join(CLONE, f)] = _sha(os.path.join(CLONE, f))
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log ------------------------------------------------------------------------------------------------------------
    for r in p0rows:
        _log(f"[{tag}] PART 0 {r['id']}: {r['status']} ({r['where']}) -- {r['w177a_finding'][:160]}")
    for c in p0conf['rechecked']:
        _log(f"[{tag}] CONFIRM comment l. {c['line']}: {c['outcome'][:400]}")
    _log(f"[{tag}] W177a CONFIRM recomputed vs committed: {p0same}")
    for p in prediction:
        _log(f"[{tag}] PREDICTION {p['item']}: {p['outcome']}")
    _log(f"[{tag}] main-scope tokens {len(main_toks)}: {dict(collections.Counter(t['status'] for t in main_toks))}; in scope {ins}; "
         f"separate {sep}")
    for sc, cnt in by_scope.items():
        _log(f'[{tag}]   scope {sc} ({scope_lines.get(sc)}): {cnt}')
    _log(f"[{tag}] in-scope hunks since 51a6fb5: {[(h['old'], h['new'], h['scope'], h['tokens_removed'], h['tokens_added']) for h in in_hunks]}")
    _log(f"[{tag}] W177b-1 statuses on unchanged lines: {repro['equal']}/{repro['n']} equal; differ {repro['differ'][:10]}")
    _log(f"[{tag}] section 3 opening ({s3a}-{s3b}): {dict(collections.Counter(t['status'] for t in sec3_toks))}; M1 {m1_present}; "
         f"submitted sentences absent {sub_absent}")
    for k, v in case_files.items():
        _log(f"[{tag}]   case file {v['path']}: sha256 {v['sha256'][:8]} blob {v['blob'][:8]} spec {v['spec_sha256'][:8]} "
             f"equal {v['equal_to_spec']}")
    _log(f"[{tag}] letter ({LETTER} = v2 {letter_equals_v2}): {l_counts}; rules {l_rule}; quotations {q_summary}")
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
    if stale or claimed_twice or unassigned or carry_refused or dup_decl or unmapped:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; carry refused {carry_refused}; declared twice {dup_decl}; '
             f"unmapped {unmapped}; unassigned {[(t['file'], t['line'], t['written']) for t in unassigned]}")
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
