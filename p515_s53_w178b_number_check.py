"""P5.15 Addendum 77, Planner task W178b -- ITEMS (b) AND (c) OF THE W178 ORDER AT OVERLEAF 9fd19e7 (main.tex =
manuscript_review/main_820ee1b_fixed.tex, sha256 d18384d6...). ZERO SOLVES, NO MODEL LOADS.

(b) For each of the fixes A1-A20 and B1-B4 of manuscript_review/w177b_fixes.tex (the letter edits L1-L4 are W178c's):
    applied as worded at 9fd19e7 (verbatim / with changes / absent / declined), located by CONTENT, whitespace-insensitive
    (every whitespace character removed before comparison; comment text kept apart from typeset text). The fixes file is
    parsed: the uncommented lines of each "% A<n> --" / "% B<n> --" block are the new text (grouped into fragments; the
    lone "\\textcolor{blue}{" / "}" wrapper lines are dropped), the old text named in the block's comments is declared
    below (OLD_ABSENT) and must be absent from the typeset text. Instructions given only in a comment (a pointer to change,
    a comment to delete, a counter line to add after each longtable) are declared below (EXTRA_NEW, STRUCT). A difference
    between the fix and the HEAD is DECLARED below (DIFFS) with the observed text, which must be found, and a judgement of
    whether it alters meaning; an undeclared difference fails an integrity check. Inside a multi-line fix the typeset text
    between two consecutively found lines (the residue) must be only the \\textcolor{blue}{ ... } wrapper, so that text
    inserted inside a fix is seen; B1's range end is checked against 9a697cc, the HEAD the fixes were written against.
    Then: every W177(b) MISMATCH / H / M / L row in main.tex (P5_15_W177B_PART01_E10C104.md, P5_15_W177B_PART2_E10C104.md;
    with the Part 0 carry-overs and the decision-6 Part 3 rows the fixes address) re-scored at 9fd19e7 (ROWS: the e10c104
    text must exist at e10c104; "still" requires it at 9fd19e7, "resolved" requires it gone and the new text present).
    And the NUMBER CHECK over the addressed passages (ADDRESSED: the 9fd19e7 lines each fix lands on, pinned and
    re-derived from the located fragments), with new declarations: every token of those lines is assigned. Declarations
    are W177b-1' specs carried by (e10c104 line, written text, occurrence) where the quantity is unchanged (OLD; the
    W177b-1' declarations at e10c104 are rebuilt by calling its own relocate_b1 / main_new_declarations /
    sec3_declarations, not re-typed), relocated unchanged on lines equal to e10c104, or new specs evaluated now against
    the frozen JSON (590088fe) and named records; one token is carried from W174a's committed check (CARRY174: the source
    files it cites must still have the sha256 it recorded).
    The new salvage equation (A8) against shared_energy_storage_data.py:866-905 and the reproduction of the recorded
    yl_y2030 / yl_y2035 salvage by the PRINTED form (energy cost only, health (SoH - SoH^Min)/(1 - SoH^Min), converter not
    credited); the J 4 MWh row at its uncertified bar (Addendum 64 ruling 4); the new sentences of 2.2.7, 4.5, 4.6, 4.8 and
    the Conclusions (numbers here; their claims in ROWS).
(c) The 0.50-floor row: value 258.5 kEUR = I + (value - I); AE and EFC/day, PV-weighted exactly as the other T8 rows,
    from the committed W155 record (w155_summary_after_04_e_soh050.json reports.e_soh050.AE / EFC), recomputed from the
    primary evaluation_record.json with the same reader (p515_s53_w142_resettle_ext_v6_campaign.ageing_series) and formula
    (p515_s46_ageing_mechanism.pv) that produced T8's AE / EFC columns; the definition traced through the code that
    computed T8 (W157 ageing() <- W142-ext summary item_E.per_arm <- M46A.pv) and checked numerically (the W155 0.70
    neutrality cell reproduces T8's C2_calfade AE / EFC exactly); stated at T8's printed rounding (AE 3 dp, EFC/day 2 dp)
    beside what main.tex prints in that row.

A NEW FILE. p515_s53_w177b1r_number_check.py (W177b-1', f78836a9) is IMPORTED and NOT edited; through it the W177b-1, W177a
and W176b -> ... -> W160 scripts are imported, each arming its guards. Also imported, not edited:
p515_s53_w142_resettle_ext_v6_campaign (ageing_series; arms its guards) and p515_s46_ageing_mechanism (pv; arms its guard).
Every output is a new file opened 'x' in a new directory.

OBJECTIVE CONVENTION. Q = gross_operational_cost, settlement excluded (the frozen primary); net = Q - terminal salvage
credit; value = Q(0) - Q(x); F = Q + I. Amounts in EUR unless stated (kEUR in the manuscript).

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import, verified at exactly 0 at the end with
every guard the imported chain arms; pickle.load / pickle.loads blocked for the whole run (counters verified 0). git
reads only (git show / rev-parse / ls-files / status / log / hash-object without -w), also in the manuscript clone, which
is never pulled or modified. Refuses to run beside another copy of itself.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w178b_number_check/overleaf_9fd19e7 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w178b_number_check.py \\
        --overleaf-commit 9fd19e7 --expect main.tex=d18384d6... (full sha256 of every *.tex of the clone) \\
        > data/SRP1/Results/P515S53/w178b_number_check/overleaf_9fd19e7/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w178b_number_check/overleaf_9fd19e7/w178b_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w178b_number_check/overleaf_9fd19e7/w178b_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w178b_number_check.py \\
        --overleaf-commit 9fd19e7 --post-run
REFUSALS ARE KEPT: a refused launch writes only its launch log; a later launch must log to launch_attempt<N>.log beside
it (the write-once check admits launch*.log files only, and the JSON lists each with its sha256).
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES, differences and open rows are
FINDINGS and never change the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard
fault (nothing written).
"""
import argparse
import collections
import difflib
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W178b number check at Overleaf 9fd19e7 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W178b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W178b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w177b1r_number_check as B1R  # noqa: E402 -- arms its guard; imports W177b-1 -> W177a -> ... -> W160 (none edited)
import p515_s53_w142_resettle_ext_v6_campaign as LX  # noqa: E402 -- ageing_series (the T8 AE / EFC reader; arms its guards)
import p515_s46_ageing_mechanism as M46A  # noqa: E402 -- pv (the T8 AE / EFC formula; arms its guard)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

B1, W177A, W176, W174, W171, W164, W160 = B1R.B1, B1R.W177A, B1R.W176, B1R.W174, B1R.W171, B1R.W164, B1R.W160
GUARDS = W160.W157._dedupe((('w178b_number_check', GUARD),) + tuple(B1R.GUARDS) + tuple(LX.GUARDS) +
                           (('s46_ageing_mechanism', M46A.GUARD),))
_log, _sha, _sha_bytes, _git = W160._log, W160._sha, W160._sha_bytes, W160._git
S, U, P, W3, D = B1R.S, B1R.U, B1R.P, B1R.W3, B1R.D
NOT, SINGLE, LANG, DER, SUBF = B1R.NOT, B1R.SINGLE, B1R.LANG, B1R.DER, B1R.SUBF
_cl, _t10, _text, _json = B1R._cl, B1R._t10, B1R._text, B1R._json
REF4 = B1R.REF4

SCRIPT_REL = os.path.basename(__file__)
B1R_SCRIPT = os.path.basename(B1R.__file__)
S53 = W160.S53
OUT_DIR = os.path.join(S53, 'w178b_number_check', 'overleaf_9fd19e7')
OUT_NAMES = {'json': 'w178b_number_check.json', 'md': 'w178b_number_check.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'typing_json': 'w178b_bool_typing_test.json', 'typing_log': 'w178b_bool_typing_test.log',
             'post': 'manifest_post_run_sha256.json'}
LAUNCH_LOG_RX = re.compile(r'launch(_attempt\d+)?\.log')

CLONE = W164.DEFAULT_MANUSCRIPT_DIR                       # manuscript/6a67305f25e8348fb71380c3 (read only)
DECLARED_COMMIT = '9fd19e7'
CLONE_HEAD = '9fd19e7498e2fed23d00c74efaff88140ab5e84e'
PREV_COMMIT = B1R.CLONE_HEAD                              # e10c104 (W177b-1', W177b-2')
PREV51 = B1R.PREV_COMMIT                                  # 51a6fb5 (W177b-1)
FIXES_WRITTEN_AT = '9a697cc'                               # the HEAD w177b_fixes.tex was written against (B1 range)
TEX_SHA = {
    'main.tex': 'd18384d669b7a4fbbcfb95e7ead5181e6ffdc5001f51f60d773053e1473df15d',
    'highlights.tex': '001352eebb3ac647f8fe1f5c2211025816ca5143c74c02a01af731630a8fea08',
    'response_to_reviewers_draft.tex': '18992d8f3bcb45f2030a16f981c6cc6f474e6332fe791e23ee7030d130c9857b',
    'cover_letter.tex': '363ad82139c8bbf1873e64c8984934d4e049a67af593fa90936a54f3601d7508',
    'section2_expert_draft.tex': '7e347dc830d1d9a8f0c6892a963a534d63d90846c6508108821e9cfac9bcabd1',
}
MAIN_LINES_N = 2511
LETTER = W164.LETTER
BIB = 'bibliography.bib'
FZ_REL, FZ_SHA = B1R.FZ_REL, B1R.FZ_SHA

FIXES = os.path.join('manuscript_review', 'w177b_fixes.tex')
EXTRA_INPUTS = {
    'B1R_SCRIPT': B1R_SCRIPT,
    'B1R_JSON': os.path.join(S53, 'w177b_number_check', 'overleaf_e10c104', 'w177b1r_number_check.json'),
    'B2R_JSON': os.path.join(S53, 'w177b_part2_e10c104', 'w177b2r_checks.json'),
    'B3R_JSON': os.path.join(S53, 'w177b_part3_e10c104', 'w177b3r_checks.json'),
    'W174A_JSON': os.path.join(S53, 'w174_manuscript_check', 'overleaf_260bd83', 'w174_manuscript_number_check.json'),
    'PART01_REPORT': 'P5_15_W177B_PART01_E10C104.md',
    'PART2_REPORT': 'P5_15_W177B_PART2_E10C104.md',
    'W177B_REPORT': 'P5_15_W177B_REPORT.md',
    'FIXES': FIXES,
    'FOLLOWUP_DIFF': os.path.join('manuscript_review', 'main_820ee1b_fixes.diff'),
    'FIXED_COPY': os.path.join('manuscript_review', 'main_820ee1b_fixed.tex'),
    'LETTER_V4': os.path.join('manuscript_review', 'response_to_reviewers_v4.tex'),
    'RESETTLE_V6': os.path.join(S53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json'),
    'RESETTLE_EXT_V3': os.path.join(S53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json'),
    'A64_SPEC': os.path.join(S53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json'),
    'SX_SUMMARY': os.path.join(S53, 'w142_resettle_ext_v6', 'w142_ext_summary_after_07_pb_y2025_n5_v6.json'),
    'W155_RECORD': os.path.join(S53, 'w155_a64_cells', 'campaign_s53_w155_a64_r2_e_soh050', 'evals',
                                '3f01b9eaaab13c82_e_soh050', 'evaluation_record.json'),
    'CODE_W157': 'p515_s53_w157_step6_tables.py',
    'CODE_W142X': 'p515_s53_w142_resettle_ext_v6_campaign.py',
    'CODE_W155': 'p515_s53_w155_a64_campaign.py',
    'CODE_M46A': 'p515_s46_ageing_mechanism.py',
    'ELSARTICLE_DTX': os.path.join(CLONE, 'elsarticle.dtx'),
}
EXTRA_JSON = ('B1R_JSON', 'B2R_JSON', 'B3R_JSON', 'W174A_JSON', 'RESETTLE_V6', 'RESETTLE_EXT_V3', 'A64_SPEC', 'SX_SUMMARY',
              'W155_RECORD')
EXTRA_TEXT = ('FIXES', 'FOLLOWUP_DIFF', 'FIXED_COPY', 'LETTER_V4', 'CODE_W157', 'CODE_W142X', 'CODE_W155', 'CODE_M46A',
              'ELSARTICLE_DTX', 'PART01_REPORT', 'PART2_REPORT', 'W177B_REPORT')
STATUSES = B1.STATUSES
ITEM_IDS = tuple(f'A{i}' for i in range(1, 21)) + ('B1', 'B2', 'B3', 'B4')


# ======================================================================================================================
#  1. text helpers
# ======================================================================================================================
def nows(s):
    return re.sub(r'\s+', '', s)


def nsp(s):
    return re.sub(r'\s+', ' ', s).strip()


class Located:
    """main.tex split into typeset (body) and comment text; whitespace-removed copies with a char -> line index."""

    def __init__(self, lines):
        self.lines = lines
        self.body_chars, self.body_line, self.com_chars, self.com_line = [], [], [], []
        for i, ln in enumerate(lines, 1):
            body, com, _ = W164.split_comment(ln)
            for ch in body:
                if not ch.isspace():
                    self.body_chars.append(ch)
                    self.body_line.append(i)
            for ch in (com or ''):
                if not ch.isspace():
                    self.com_chars.append(ch)
                    self.com_line.append(i)
        self.body = ''.join(self.body_chars)
        self.com = ''.join(self.com_chars)

    def find(self, frag, where='body'):
        """all occurrences of the whitespace-removed fragment: [(first line, last line)]."""
        s, idx = (self.body, self.body_line) if where == 'body' else (self.com, self.com_line)
        f = nows(frag)
        out, k = [], s.find(f)
        while f and k >= 0:
            out.append((idx[k], idx[k + len(f) - 1]))
            k = s.find(f, k + 1)
        return out

    def local(self, ranges):
        """the typeset text of the given line ranges only (whitespace removed) with its char -> line index."""
        keep = [(c, ln) for c, ln in zip(self.body_chars, self.body_line) if any(a <= ln <= b for a, b in ranges)]
        return ''.join(c for c, _ in keep), [ln for _, ln in keep]

    def find_in(self, frag, ranges):
        s, idx = self.local(ranges)
        f = nows(frag)
        k = s.find(f) if f else -1
        return [(idx[k], idx[k + len(f) - 1])] if k >= 0 else []

    def ordered(self, frag_lines, ranges):
        """each (fix line no, text) of a fragment found in order inside the ranges: [(fix line, found, (a, b) | None,
        found_anywhere_in_ranges)], whether the order held, and the RESIDUES: the typeset text (whitespace removed)
        between two consecutively found lines of the fragment, so that text inserted inside a multi-line fix is seen."""
        s, idx = self.local(ranges)
        pos, out, order_ok, resid, prev = 0, [], True, [], None
        for fl, text in frag_lines:
            f = nows(text)
            if not f:
                continue
            k = s.find(f, pos)
            anywhere = s.find(f)
            if k >= 0:
                if prev is not None and k > prev[1]:
                    resid.append({'after_fix_line': prev[0], 'before_fix_line': fl, 'text': s[prev[1]:k],
                                  'lines_now': [idx[prev[1]], idx[k - 1]]})
                out.append((fl, True, (idx[k], idx[k + len(f) - 1]), True))
                pos = k + len(f)
                prev = (fl, pos)
            else:
                if anywhere >= 0:
                    order_ok = False
                out.append((fl, False, (idx[anywhere], idx[anywhere + len(f) - 1]) if anywhere >= 0 else None,
                            anywhere >= 0))
        return out, order_ok, resid


def parse_fixes(text):
    """{item id: {'fragments': [{'first_line', 'text', 'core'}], 'comment': str}} from w177b_fixes.tex."""
    items, cur = collections.OrderedDict(), None
    for i, ln in enumerate(text.split('\n'), 1):
        m = re.match(r'%\s*([AB]\d+)\s*--', ln)
        if m:
            cur = items.setdefault(m.group(1), {'header_line': i, 'lines': []})
            continue
        if cur is not None:
            cur['lines'].append((i, ln))
    for it in items.values():
        frags, f, com = [], [], []
        for i, ln in it['lines']:
            s = ln.strip()
            if s.startswith('%') or not s:
                if s.startswith('%'):
                    com.append(s)
                if f:
                    frags.append(f)
                    f = []
                continue
            f.append((i, ln))
        if f:
            frags.append(f)
        it['fragments'] = [{'first_line': fr[0][0], 'last_line': fr[-1][0], 'text': '\n'.join(ln for _, ln in fr),
                            'core': '\n'.join(ln for _, ln in fr if ln.strip() not in ('\\textcolor{blue}{', '}')),
                            'core_lines': [(i, ln) for i, ln in fr if ln.strip() not in ('\\textcolor{blue}{', '}')]}
                           for fr in frags]
        it['comment'] = '\n'.join(com)
        del it['lines']
    return items


def word_diff(a, b):
    """word-level difference of two LaTeX texts: [(op, removed, added)] for the non-equal opcodes."""
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    return [{'op': op, 'fix': ' '.join(ta[i1:i2]), 'head': ' '.join(tb[j1:j2])}
            for op, i1, i2, j1, j2 in sm.get_opcodes() if op != 'equal']


# ======================================================================================================================
#  2. (b) the fixes: declared old text, comment-only instructions, structure, declared differences
# ======================================================================================================================
# old text named in each block's comments: must be ABSENT from the typeset text at the HEAD (whitespace-insensitive)
OLD_ABSENT = {
    'A1': ['transmission--distribution test system over a 15-year planning horizon'],
    'A2': ['$\\pi_t$, $\\bar{\\pi}_t$', '$t_{\\text{sum}}$, $s$', 'settling slack (uncertified form)'],
    'A3': [],
    'A4': ['remaining disagreement at the interface, priced, must be within'],
    'A5': ['(Subsection~3.1)'],
    'A6': ["an evaluation continued from an earlier run is capped 100 cycles after that run's stopping cycle",
           'the cache was seeded with the single-node ladders of the initial design, run with and without the budget, and '
           'their combinations'],
    'A7': ['Every reported number is a certified recourse evaluation or a difference between two of them'],
    'A8': ['\\frac{I_{e,y^\\text{Inv}}}{(1+r)^{\\,y^\\text{End}-y_0}}', '\\; SoH_{e,y^\\text{Inv},y^\\text{End}},',
           'is the expected undiscounted cost of the cohort installed at node'],
    'A9': ['4$^{f}$ & 176.6 & 9.7 & 18.14', 'Treating the flagged evaluation as uncertified gives a comparison bar'],
    'A10': ['Baseline, 0.50 floor & --- &'],
    'A11': ['The planning search again returned the plan without shared storage',
            'every local solve successful; Section~\\ref{sec:case_settings}), rather than by the certification rule'],
    'A12': [],
    'A13': ['Across the 60 pairwise comparisons reported in this section', 'It changes two verdicts'],
    'A14': ['they are counted as clean under the rule of Subsection~\\ref{subsubsec:certification}'],
    'A15': [],
    'A16': ['instead, each evaluation is certified by an explicit stopping rule, and every planning comparison',
            'and every other evaluated plan, from 0.5 to 5~MWh and at one or two nodes',
            'the value of shared storage cannot be assessed apart from its ageing or from the coordination'],
    'A17': ['or from the stopping cycle of the earlier run in an evaluation continued from one.'],
    'A18': ['the same with the adaptive barrier strategy (Section~\\ref{sec:case_settings})'],
    'A19': ['from the next cycle in a fresh run or from the stopping cycle of the earlier run in an evaluation continued from '
            'one', "the residual mismatch between the operators' interface schedules at the interface price"],
    'A20': ['certification parameters of Subsection~\\ref{subsubsec:certification}'],
    'B1': [],          # the defect B1 repairs (duplicate labels, missing eq:cohort_rated) is checked in STRUCT
    'B2': ['the available-energy product in \\eqref{eq:cohort_available} and'],
    'B3': ['or when an evaluation limit is reached.'],
    'B4': [],
}
# instructions written only in a comment of the block: (description, fragment that must be in the typeset text)
EXTRA_NEW = {
    'A5': [('the pointer becomes a \\ref', 'at the planning level (Subsection~\\ref{subsec:case_investment_costs}).'),
           ('the label sits directly under \\subsection{Investment Costs}', '\\subsection{Investment Costs}\n'
                                                                            '\\label{subsec:case_investment_costs}')],
    'A11': [('pointer (...; \\ref{app:admm_updated_settings})',
             'with every local solve successful; \\ref{app:admm_updated_settings}), rather than')],
    'A18': [('pointer after the recovery tiers', 'the same with the adaptive barrier strategy '
                                                 '(\\ref{app:admm_updated_settings});')],
    'A16': [('(c) continues with "For the method, ..." unchanged', 'while storage becomes worthwhile where flexibility is '
                                                                   'expensive. For the method, a residual-based stopping')],
    'A9': [('the note keeps its second sentence', 'its certified threshold would be 9.7~k\\euro{} ($18.14\\times$). The '
                                                  'increments to 4 and 5~MWh have bars of 27.7 and 26.5')],
    'A3': [('placed after "...market prices."', 'operational scenarios of load, RES generation and market prices. Demand '
                                               'growth and RES capacity follow predefined trajectories')],
    'A4': [('inserted as the first sentence of the paragraph', '\\paragraph{Stopping and certification} No convergence '
                                                              'guarantee is available'),
           ('followed by the old first sentence', 'stopped by an explicit rule. Each recourse evaluation runs the '
                                                  'coordination procedure')],
    'A12': [('T7 caption clause', '$I$ paid in 2025; rates other than 2\\,\\% re-weight the 2\\,\\% evaluations; every '
                                  'difference is determinate.'),
            ('placed after "...and every difference is determinate."', 'and every difference is determinate. The rates '
                                                                       'other than 2\\,\\% re-weight')],
    'A15': [('placed after "...applied at block boundaries."', 'the degradation chain is applied at block boundaries. The '
                                                               'multi-scenario evidence is the $3 \\times 3$ instance')],
    'A8': [('"The credit is reported, not optimized: ..." is kept', 'the converter is not credited. The credit is reported, '
                                                                    'not optimized: the recourse does not see it'),
           ('eq:salvage_remaining_life is kept', 'T^\\text{Rem}_{e,y^\\text{Inv}} = \\max\\left\\{0,\\; y^\\text{Inv} + '
                                                  'T^\\text{Cal}_{e,y^\\text{Inv}} - y^\\text{End}\\right\\}')],
    'A7': [('the rest of the sentence is kept', 'stopped by the production exit; a difference is called determinate or '
                                                'within resolution by the rule of')],
    'A13': [('the [CONFIRM -- W177] comment under the paragraph is deleted (checked in STRUCT)', 'on it the credit changes '
                                                                                                 'the verdict, from '
                                                                                                 'determinate in gross '
                                                                                                 'terms to within '
                                                                                                 'resolution.')],
    'B1': [('the heading of 2.3.2 (follow-up hunk 1)', '\\subsubsection{\\textcolor{blue}{Installed and Available '
                                                       'Capacity}}')],
}
# comment text that a block says to DELETE: must be absent from the comments at the HEAD
COMMENT_ABSENT = {
    'A8': ['[CONFIRM — W177] eq:salvage_credit against _get_terminal_salvage_value_*'],
    'A13': ['[CONFIRM — W177] W153 totals'],
}
# the declined item: the sentence stays commented out
B4_SENTENCE = 'The decomposition is computational; it is not a leader--follower structure.'
# differences between the fix as worded and the HEAD, each with the observed text (must be found) and a judgement
DIFFS = {
    'A8': [{'what': 'the three-line [CONFIRM -- W177] comment under the where-clause was not deleted',
            'fix': '(delete the comment)',
            'head_where': 'comment',
            'head': '[CONFIRM — W177] eq:salvage_credit against _get_terminal_salvage_value_*: (a) y^End is the end of the '
                    'last block',
            'explains': ['comment_present:[CONFIRM — W177] eq:salvage_credit against _get_terminal_salvage_value_*'],
            'alters_meaning': False, 'impact': 'none (LaTeX comment, not typeset)',
            'judgement': 'a review comment left in the source; it is not typeset. Its four points (a)-(d) are W177a C1, '
                         'confirmed as worded and reproduced again by this run (salvage section).'}],
    'A11': [{'what': 'the pointer keeps the word "Section" before the appendix \\ref',
             'fix': '(ten consecutive cycles passing the residual test with every local solve successful; '
                    '\\ref{app:admm_updated_settings})',
             'head_where': 'body',
             'head': '(ten consecutive cycles passing the residual test with every local solve successful; '
                     'Section~\\ref{app:admm_updated_settings})',
             'explains': ['extra_missing:pointer (...; \\ref{app:admm_updated_settings})'],
             'alters_meaning': False, 'impact': 'L (wording)',
             'judgement': 'the target is the one the fix names (A.8, the solver and coordination settings). Under '
                          'elsarticle\'s \\appendix (\\thesection = "\\appendixname~\\Alph{section}", elsarticle.dtx in '
                          'the clone) the \\ref of an appendix subsection already reads "Appendix A.<n>", so the line '
                          'renders "Section Appendix A.8" (static reading; no compile). Every other appendix pointer of '
                          'the fixes is written without a prefix.'}],
    'B1': [{'what': 'the cohort-rating equation is printed as two numbered equations (S, then E), the label '
                    'eq:cohort_rated on the S equation and a new label eq:cohort_rated_energy on the E equation (the '
                    'author\'s split at 820ee1b with the expert\'s follow-up hunk 2)',
            'fix': 'S^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y} = \\hat{S}^{\\text{Inv}}_{e,y^\\text{Inv}}, \\qquad '
                   'E^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y} = \\hat{E}^{\\text{Inv}}_{e,y^\\text{Inv}} '
                   '\\label{eq:cohort_rated}',
            'head_where': 'body',
            'head': 'S^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y} = \\hat{S}^{\\text{Inv}}_{e,y^\\text{Inv}} '
                    '\\label{eq:cohort_rated} \\end{equation} } \\textcolor{blue}{ \\begin{equation} '
                    'E^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y} = \\hat{E}^{\\text{Inv}}_{e,y^\\text{Inv}} '
                    '\\label{eq:cohort_rated_energy}',
            'explains': ['frag140:line145:missing', 'frag140:line147:missing', 'frag140:order',
                         'frag140:insertion:144-146', 'frag140:insertion:146-148'],
            'alters_meaning': False, 'impact': 'none',
            'judgement': 'the same two definitions; one more equation number. Appendix A.2 cites the range '
                         '\\eqref{eq:cohort_rated}--\\eqref{eq:available_capacity}, which now starts at the S equation and '
                         'so covers both ratings; eq:cohort_rated_energy is defined once and cited nowhere (checked).'}],
}
STRUCT_NOTES = {
    'A2': 'after EACH of the three nomenclature \\end{longtable}: the next non-blank line is \\addtocounter{table}{-1}',
    'B1': 'no duplicate label in the typeset text; eq:cohort_rated, eq:cohort_available, eq:cohort_available_energy, '
          'eq:available_power and eq:available_capacity each defined once; every \\eqref to them resolves',
}
LOCATION = {
    'A1': 'Abstract', 'A2': 'Nomenclature (three longtables; Variables and Functions)', 'A3': 'Introduction, "To address '
    'this gap" paragraph', 'A4': 'Section 2.2.7, "Stopping and certification"', 'A5': 'Section 3 opening + 3.1 label',
    'A6': 'Section 3.6 (cap sentence; cache clause)', 'A7': 'Section 4 opening', 'A8': 'Section 4 opening: salvage '
    'equation and where-clause', 'A9': 'tab:res_energy_ladder_m2, row 4 MWh and note', 'A10': 'tab:res_ageing, 0.50-floor '
    'row', 'A11': 'Section 4.5', 'A12': 'Section 4.6 discount paragraph + T7 caption', 'A13': 'Section 4.6 salvage '
    'paragraph', 'A14': 'Section 4.7 reproducibility', 'A15': 'Section 4.8 scope of the study', 'A16': 'Conclusions '
    '(a)-(c)', 'A17': 'Appendix A.3, last sentence', 'A18': 'Appendix A.4 (recovery tiers pointer)', 'A19': 'Appendix A.5 '
    '(hold origin; C4)', 'A20': 'Algorithm 2 input line', 'B1': 'Section 2.3.2 Installed and Available Capacity', 'B2':
    'Section 2.3.4 last paragraph', 'B3': 'Section 2.1 MADS paragraph', 'B4': 'Section 2 opening (leader-follower sentence)'}
# the addressed passages (9fd19e7 lines), pinned; the run re-derives the lines of every located fragment and refuses on a
# fragment outside its item's ranges
ADDRESSED = collections.OrderedDict([
    ('A1', [(146, 146)]), ('A2', [(224, 224), (273, 273), (287, 287), (312, 313), (342, 342), (345, 345)]),
    ('A3', [(380, 381)]), ('A4', [(732, 734)]), ('A5', [(1136, 1136), (1144, 1145)]), ('A6', [(1315, 1316)]),
    ('A7', [(1332, 1334)]), ('A8', [(1336, 1362)]), ('A9', [(1493, 1493), (1502, 1503)]), ('A10', [(1540, 1540)]),
    ('A11', [(1601, 1603)]), ('A12', [(1663, 1665), (1670, 1670)]), ('A13', [(1686, 1686)]), ('A14', [(1717, 1717)]),
    ('A15', [(1733, 1735)]), ('A16', [(1745, 1745), (1749, 1749), (1753, 1754)]), ('A17', [(1908, 1908)]),
    ('A18', [(1925, 1925)]), ('A19', [(1932, 1933), (1941, 1941)]), ('A20', [(1962, 1962)]), ('B1', [(767, 833)]),
    ('B2', [(1051, 1051)]), ('B3', [(524, 524)]), ('B4', [(411, 411)])])


# ======================================================================================================================
#  3. number check: helpers and declarations at 9fd19e7
# ======================================================================================================================
def OLD(e10_line, w, k=0):
    """a W177b-1' spec carried from main.tex at e10c104 (line, written text, k-th occurrence among that line's specs)."""
    return {'kind': '_old', 'e10_line': e10_line, 'w': w, 'k': k}


def SAME(e10_line):
    """every spec of the e10c104 line, in order (the token sequence must be unchanged)."""
    return {'kind': '_same', 'e10_line': e10_line}


def CARRY174(w, line260, col260, frag, why):
    """W174a's committed check of the token at 260bd83 (line, col), carried: the context fragment must be in both lines and
    every source it cites with '(sha256 xxxxxxxx)' must still have that sha256."""
    return {'w': w, 'kind': 'carry174', 'line260': line260, 'col260': col260, 'frag': frag, 'why': why}


def _rs(X, row_id, key):
    return W177A.jget(X.j['B3R_JSON'], f'item2_run_settings.rows[{row_id}].detail.{key}')


def _all_same(vals):
    vals = list(vals)
    return vals[0] if vals and all(v == vals[0] for v in vals) else None


def _cells_all(X):
    out = []
    for key in ('RESETTLE_V6', 'RESETTLE_EXT_V3', 'A64_SPEC'):
        c = X.j[key]['cells']
        out += [(key, k, v) for k, v in (c.items() if isinstance(c, dict) else [(x.get('cell'), x) for x in c])]
    return out


def cap_after_record(X):
    """cells with a fixed cap: cap - N_old (formula 'N_old + 100') over the three frozen re-settling specs."""
    d = [v['cap_rule']['cap'] - v['N_old'] for _k, _c, v in _cells_all(X) if v['cap_rule'].get('kind') == 'fixed']
    ok = all(v['cap_rule'].get('formula') == 'N_old + 100' for _k, _c, v in _cells_all(X)
             if v['cap_rule'].get('kind') == 'fixed')
    return _all_same(d) if ok else None


def fresh_cap(field):
    def f(X):
        return _all_same(v['cap_rule'][field] for _k, _c, v in _cells_all(X) if v['cap_rule'].get('kind') == 'dynamic')
    return f


def n_searches_budget(X, b):
    return sum(1 for k in ('s47', 's51', 's53') if _rs(X, 'budgets', k)[0] == b)


def n_searches_m2_cache(X):
    c = W177A.jget(X.j['B3R_JSON'], 'item2_run_settings.rows[cache_seeding].detail')
    return sum(1 for k in ('s47', 's51', 's53') if set(c[k]['by_source']) == {'committed_certified_record_at_m2'})


def year_multi_node_s53(X):
    c = W177A.jget(X.j['B3R_JSON'], 'item2_run_settings.rows[cache_seeding].detail.s53.multi_node_entries')
    ys = {int(re.match(r'y(\d{4})__', e).group(1)) for e in c}
    return ys.pop() if len(ys) == 1 else None


VALUE_LADDER_CLAIMS = tuple(r[1] for r in W177A.VALUE_LADDER_ROWS)


def _ladder_others(X):
    return [c for c in VALUE_LADDER_CLAIMS if c != REF4]


def ladder_plan_stat(which):
    """over the plans of tab:res_value_ladder other than the reference unit: per-node E (min / max), nodes (min / max)."""
    def f(X):
        es, nn = [], []
        for c in _ladder_others(X):
            nodes = {n: v for n, v in X.cand(X.other(c))['nodes'].items() if v[0] or v[1]}
            es += [v[1] for v in nodes.values()]
            nn.append(len(nodes))
        return {'Emin': min(es), 'Emax': max(es), 'nmin': min(nn), 'nmax': max(nn)}[which]
    return f


def der_floor050_value(X):
    a = X.A64['E:soh050:value_minus_I']
    v = a['d_gross'] + a['I_other']
    q = X.j['W155_SOH050']['scored']['claims']['E:soh050:value_minus_I']
    v2 = q['ref']['Q'] - q['other']['Q']
    return {'value': v / 1e3, 'consistent': abs(v - v2) < 0.01,
            'arithmetic': f"value = I + (value - I) = {a['I_other']:,.2f} + ({a['d_gross']:,.2f}) = {v:,.2f} EUR "
                          f"({FZ_REL} tables.a64.rows[E:soh050:value_minus_I] I_other + d_gross) = Q(0) - Q(e_soh050) "
                          f"{v2:,.2f} EUR ({X.inputs['W155_SOH050']['path']} scored.claims.E:soh050:value_minus_I ref.Q - "
                          f"other.Q); printed 318.0 - 59.5 = 258.5"}


def declarations_9fd():
    """main.tex at 9fd19e7: one entry per addressed line carrying non-excluded tokens and not equal to an e10c104 line."""
    J4 = 'J:e4_value_minus_I'
    two_pct = S('2', 'pct0', ('rec', 'SRP1_JSON', 'DiscountFactor'), 'SRP1.json DiscountFactor (x 100): the production rate')
    ctr = U('-1', 'LaTeX argument', '\\addtocounter{table}{-1} (A2): compensates the counter step of a caption-less '
                                    'nomenclature longtable; checked structurally in part (b)')
    three_ref = OLD(1555, 'three', 0)
    L = [
        D(146, None, SAME(146)),
        D(380, None, U('two', 'compound adjective', '"two-stage stochastic program" (the model class)'),
          LANG('single', '"a single, scenario-independent investment plan"')),
        D(224, None, ctr), D(287, None, ctr), D(345, None, ctr),
        D(734, None, W3('2', 'N12', 2.0, 'the gap clause |t_sum| <= tau/2 (settling_criterion_v2.GAP_BOUND = TAU / 2)')),
        D(1136, None, OLD(985, 'Three')),
        D(1315, None,
          S('100', 'int', cap_after_record, 'frozen re-settling specs v6 96c23404, ext v3 84775dc4, A64 44a2dce8: every '
                                            'fixed-cap cell (continued or replay-gated re-run) has cap = N_old + 100'),
          LANG('one', 'pronoun ("a fresh one")'),
          S('109', 'int', fresh_cap('after_first_k0'), 'the same specs: every dynamic-cap cell min(k0_run + 109, ceiling)'),
          S('300', 'int', fresh_cap('ceiling'), 'the same specs: the ceiling of every dynamic-cap cell'),
          S('4', 'int', lambda X: _all_same([_rs(X, 'delta0', 's47'), _rs(X, 'delta0', 's51')]),
            'W177b-3\' run settings (search specs s47 / s51 poll_design.delta_0)'),
          S('20', 'int', lambda X: _all_same([_rs(X, 'budgets', 's47')[0], _rs(X, 'budgets', 's51')[0]]),
            'W177b-3\' run settings: max_new_evaluations of the variant-A searches s47 / s51'),
          S('two', 'int', lambda X: n_searches_budget(X, 20), 'the searches with budget 20 (s47, s51; variant A)'),
          S('60', 'int', lambda X: _rs(X, 'budgets', 's53')[0], 'W177b-3\' run settings: max_new_evaluations of s53 '
                                                                '(variant B)'),
          S('30', 'int', lambda X: _all_same(_rs(X, 'completion_cap', k) for k in ('s47', 's51', 's53')),
            'W177b-3\' run settings: poll_design.completion_cap (all three)'),
          S('18,449.66', 'raw', lambda X: _all_same(_rs(X, 'sigma_Q', k) for k in ('s47', 's51', 's53')),
            'W177b-3\' run settings: extra.sigma_Q.sigma_Q_eur (all three)')),
        D(1316, None,
          S('ten', 'int', lambda X: _all_same(_rs(X, 'production_exit', k)[0] for k in ('s47', 's51', 's53')),
            'W177b-3\' run settings: required_consecutive_cycles (all three)'),
          S('500', 'int', lambda X: _all_same(_rs(X, 'production_exit', k)[1] for k in ('s47', 's51', 's53')),
            'W177b-3\' run settings: cap (all three)'),
          S('two', 'int', lambda X: _rs(X, 'barrier_stop', 's53_barrier_stop')[0], 'barrier stop: two in one poll '
                                                                                  '(s53 barrier_stop[0])'),
          LANG('one', '"in one poll"'),
          S('three', 'int', lambda X: _rs(X, 'barrier_stop', 's53_barrier_stop')[1], 'barrier stop: three in all '
                                                                                    '(s53 barrier_stop[1])'),
          S('60', 'int', lambda X: _all_same(_rs(X, 'max_polls', k) for k in ('s47', 's51', 's53')),
            'W177b-3\' run settings: poll_design.max_polls (all three)'),
          U('single', 'compound adjective', '"single-node ladders"'),
          S('two', 'int', n_searches_m2_cache, 'the searches whose cache holds only committed certified records at m = 2 '
                                               '(s51, s53 extra.cache_table by_source)'),
          S('2030', 'int', year_multi_node_s53, 'the installation year of every multi-node entry of the s53 cache')),
        D(1332, None, OLD(1180, 'single'), OLD(1180, '3', 0), OLD(1180, '3', 1)),
        D(1333, None, OLD(1180, 'two')),
        D(1334, None, OLD(1180, '0'), OLD(1180, '15'), OLD(1180, '2')),
        D(1341, None, OLD(1186, '1')),
        D(1343, None, U('1', 'notation', 'the 1 of (1 - SoH^Min) in the health factor of eq:salvage_credit (formula; '
                                         'production: (e_available - min_soh * e_rated) / (1.00 - min_soh), '
                                         'shared_energy_storage_data.py:890; salvage section)')),
        D(1358, None, OLD(1197, '0')),
        D(1493, None, OLD(1332, '4'), OLD(1332, '176.6'),
          S('15.9', 'k', _cl(J4, 'uncertified_form_beside_A64_ruling4.bar'), 'A64 ruling 4: the uncertified bar'),
          S('11.12', 'x', _cl(J4, 'uncertified_form_beside_A64_ruling4.multiple_Q'), 'A64 ruling 4: the uncertified multiple'),
          OLD(1332, '42.8'), OLD(1332, '27.7'), OLD(1332, '1.55')),
        D(1502, None, P('4', 'E', J4, [7]),
          S('15.9', 'k', _cl(J4, 'uncertified_form_beside_A64_ruling4.bar')),
          S('11.12', 'x', _cl(J4, 'uncertified_form_beside_A64_ruling4.multiple_Q')),
          S('9.7', 'k', _cl(J4, 'gross_threshold_or_bar'), 'the certified threshold (flagged certificate)'),
          S('18.14', 'x', _cl(J4, 'gross_multiple'), 'the certified multiple (flagged certificate)')),
        D(1503, None, OLD(1341, '4'), OLD(1341, '5'), OLD(1341, '27.7'), OLD(1341, '26.5'), OLD(1341, '1.55'),
          OLD(1341, '1.95'), OLD(1341, 'three')),
        D(1540, None, OLD(1378, '0.50'), DER('258.5', der_floor050_value, 'the 0.50-floor row value (A10)'),
          OLD(1378, '59.5'), OLD(1378, '12.6'), OLD(1378, '4.71')),
        D(1601, None, OLD(1439, 'three', 0), OLD(1439, 'three', 1), OLD(1439, 'five'), OLD(1439, 'three', 2)),
        D(1603, None, OLD(1439, '236.7'), OLD(1439, '318.0'), OLD(1439, '81.3'), OLD(1439, '4.52'), OLD(1439, '18.0'),
          OLD(1439, 'single'), OLD(1439, 'ten')),
        D(1663, None, SAME(1499)),
        D(1664, None, dict(two_pct), dict(two_pct)),
        D(1670, None, OLD(1504, '7'), OLD(1504, '0.25'), OLD(1504, '1'), OLD(1504, '2025', 0), OLD(1504, 'one'),
          OLD(1504, 'five'), OLD(1504, '2025', 1), dict(two_pct), dict(two_pct)),
        D(1686, None, OLD(1520, '60'),
          S('one', 'int', ('rec', 'W153_SALV', 'result.totals.n_verdict_differs'), 'W153: verdict changes over the 60'),
          OLD(1520, 'one'), OLD(1520, '0.95'), OLD(1520, '1.22'), OLD(1520, '60')),
        D(1717, None, SAME(1555)),
        D(1733, None, OLD(1571, 'one', 0), OLD(1571, 'three', 0), OLD(1571, 'five', 0), OLD(1571, 'five', 1),
          OLD(1571, 'three', 1), OLD(1571, 'four', 0)),
        D(1734, None, S('3', 'int', ('rec', 'CASE_3X3', 'NumMarketScenarios'), 'market scenarios of the 3 x 3 instance'),
          S('3', 'int', ('rec', 'CASE_3X3', 'TransmissionNetwork.num_operation_scenarios'),
            'operation scenarios of the 3 x 3 instance'),
          SUBF('5', 'five load/RES scenarios are combined with five market-price scenarios, resulting in 25 operating '
                    'scenarios per representative day', 'five', 'the submitted version\'s scenario set (5 x 5)'),
          SUBF('5', 'five load/RES scenarios are combined with five market-price scenarios, resulting in 25 operating '
                    'scenarios per representative day', 'five', 'the submitted version\'s scenario set (5 x 5)')),
        D(1735, None, OLD(1571, 'one', 1), OLD(1571, '9'), OLD(1571, 'three', 2), OLD(1571, '33'), OLD(1571, '0.70'),
          OLD(1571, '1.04'), OLD(1571, '1.85'), OLD(1571, '0.41'), OLD(1571, '0.62'), OLD(1571, 'three', 3),
          OLD(1571, '0.50')),
        D(1745, None, SAME(1581)),
        D(1749, None, OLD(1585, '108'), OLD(1585, '0.25'), OLD(1585, '1'), OLD(1585, '64.4'), OLD(1585, '318.0'),
          S('0.5', 'raw', ladder_plan_stat('Emin'), 'tab:res_value_ladder plans other than the unit: smallest per-node E'),
          S('5', 'raw', ladder_plan_stat('Emax'), 'tab:res_value_ladder plans other than the unit: largest per-node E'),
          S('one', 'int', ladder_plan_stat('nmin'), 'tab:res_value_ladder plans other than the unit: fewest nodes'),
          S('two', 'int', ladder_plan_stat('nmax'), 'tab:res_value_ladder plans other than the unit: most nodes'),
          *[OLD(1585, w, k) for w, k in (('31.6', 0), ('73.7', 0), ('0.70', 0), ('0.50', 0), ('4', 0), ('7', 0),
                                         ('171', 0), ('193', 0), ('254', 0), ('1.75', 0), ('2', 0), ('2', 1),
                                         ('two', 1), ('0', 0), ('8', 0), ('2030', 0), ('2035', 0), ('7', 1), ('9', 0),
                                         ('13.9', 0), ('90.9', 0))]),
        D(1753, None, OLD(1589, 'Two')),
        D(1754, None, OLD(1589, '15'), OLD(1589, '21'), OLD(1589, '654')),
        D(1908, None,
          U('one', 'method statement', '"with one value for all agents"'),
          S('1.5', 'raw', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.increase_factor')),
          S('1.5', 'raw', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.decrease_factor')),
          S('5', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.residual_balance_ratio')),
          S('5', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.residual_balance_ratio')),
          S('3', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.residual_balance_ratio_pf_decrease')),
          S('10^{-4}', 'sci', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.min')),
          S('10^{4}', 'sci', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.max')),
          S('one', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.balancing_exempt_until.ess.dual_ratio_below')),
          S('five', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.balancing_exempt_until.ess.consecutive_cycles')),
          U('two', 'compound adjective', '"two-phase schedule"'),
          S('ten', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.freeze_after_unchanged_cycles')),
          S('200', 'int', ('rec', 'SRP1_PARAMS', 'admm.penalty_update.freeze_backstop_cycle')),
          three_ref),
        D(1925, None, S('10^{-6}', 'sci', ('rec', 'RESETTLE_V6', 'inputs_in_force_now.configuration_now.convergence_depth_tail.compl_inf_tol'),
                        'the tight tail (frozen spec v6 96c23404)'),
          CARRY174('two', 1644, 509, 'retried in two tiers', 'the two recovery tiers (W174a check A309: network.py, '
                                                             'shared_energy_storage_data.py)')),
        D(1932, None, three_ref),
        D(1941, None, W3('2', 'N12', 2.0, 'the gap clause |t_sum| <= tau/2')),
    ]
    for d in L:
        d['file'] = 'main.tex'
    return L


# ======================================================================================================================
#  4. re-scored rows (W177(b) at e10c104 -> 9fd19e7)
# ======================================================================================================================
# status: resolved (the e10c104 text is gone and the new text is in) / still (the e10c104 text is in) / changed (the new
# text is in and leaves a different difference, rated) / not addressed. 'old' must be in main.tex at e10c104.
ROWS = [
    # ---- Part 1 (numbers) and Part 0 (W177a carry-overs), P5_15_W177B_PART01_E10C104.md -----------------------------
    {'id': 'P1-M1 / P0-C7', 'src': 'PART01 Part 1 MISMATCH; Part 0 C7', 'e10_line': 1520, 'rating_e10': 'MISMATCH (H in Part 2)',
     'old': 'It changes two verdicts', 'fix': 'A13', 'new': ['applying the salvage credit changes no sign and one verdict'],
     'status': 'resolved', 'rating_now': '-', 'token': (1686, 'one'),
     'evidence': 'the l. 1686 token "one" is checked by the number check against W153 result.totals.n_verdict_differs'},
    {'id': 'P0-C1', 'src': 'PART01 Part 0 Blocked 1 / C1; W177B_REPORT decision 6', 'e10_line': 1186,
     'rating_e10': 'expert item (printed form 4.22x / 1.86x the recorded credit)',
     'old': '\\frac{I_{e,y^\\text{Inv}}}{(1+r)^{\\,y^\\text{End}-y_0}}', 'fix': 'A8',
     'new': ['\\frac{\\bar{c}^{\\text{Inv,E}}_{y^\\text{Inv}}\\, E^{\\text{Inv}}_{e,y^\\text{Inv}}}{(1+r)^{\\,y^\\text{End}-y_0}}',
             '\\frac{SoH_{e,y^\\text{Inv},y^\\text{End}} - SoH^{\\text{Min}}_e}{1 - SoH^{\\text{Min}}_e}'],
     'status': 'resolved', 'rating_now': '-', 'salvage': True,
     'evidence': 'the printed form reproduces the recorded yl_y2030 / yl_y2035 salvage (salvage section)'},
    {'id': 'P0-B3', 'src': 'PART01 Part 0 Blocked 3 (A64 ruling 4)', 'e10_line': 1332, 'rating_e10': 'author item',
     'old': '4$^{f}$   & 176.6 & 9.7  & 18.14', 'fix': 'A9', 'new': ['4$^{f}$   & 176.6 & 15.9 & 11.12 & 42.8',
                                                                'its certified threshold would be 9.7~k\\euro{} ($18.14\\times$)'],
     'status': 'resolved', 'rating_now': '-', 'token': (1493, '15.9'),
     'evidence': 'row prints the uncertified bar 15.9 / 11.12x; the note gives the certified 9.7 / 18.14x (tokens checked)'},
    {'id': 'P0-C5', 'src': 'PART01 Part 0 Blocked 7 / C5', 'e10_line': 1378, 'rating_e10': 'author item (row unfilled)',
     'old': 'Baseline, 0.50 floor & --- & $-$59.5', 'fix': 'A10', 'new': ['Baseline, 0.50 floor & 258.5 & $-$59.5 & 12.6 & '
                                                                         '4.71 & never & --- & ---'],
     'status': 'changed', 'rating_now': 'author item (AE, EFC/day still "---")', 'token': (1540, '258.5'),
     'evidence': 'value 258.5 filled and derived; AE and EFC/day still print "---" (part (c) gives 0.778 and 1.08)'},
    {'id': 'P0-B5', 'src': 'PART01 Part 0 Blocked 5 (literal pointer)', 'e10_line': 985, 'rating_e10': 'pointer (W177b-3\')',
     'old': '(Subsection~3.1)', 'fix': 'A5', 'new': ['(Subsection~\\ref{subsec:case_investment_costs})'],
     'status': 'resolved', 'rating_now': '-', 'evidence': 'label at l. 1145 under \\subsection{Investment Costs}'},
    # ---- Part 2 (claims), P5_15_W177B_PART2_E10C104.md ------------------------------------------------------------------
    {'id': '4.2-1', 'src': 'PART2 1', 'e10_line': 1316, 'rating_e10': 'L', 'fix': None,
     'old': 'the value of every evaluated 4~h configuration at node~7 exceeds its investment cost', 'new': [],
     'status': 'still', 'rating_now': 'L', 'evidence': 'not addressed by A1-A20 / B1-B4; text unchanged (l. 1477)'},
    {'id': '4.2-2', 'src': 'PART2 1', 'e10_line': 1316, 'rating_e10': 'L', 'fix': None,
     'old': 'determinately better than seven of them and within resolution of the other six', 'new': [],
     'status': 'still', 'rating_now': 'L', 'evidence': 'not addressed; text unchanged (l. 1477)'},
    {'id': '4.3-1', 'src': 'PART2 1', 'e10_line': 1392, 'rating_e10': 'L', 'fix': None,
     'old': 'persists across the tested ageing calibrations and both floor settings', 'new': [],
     'status': 'still', 'rating_now': 'L', 'evidence': 'not addressed; text unchanged (l. 1554)'},
    {'id': '4.5-1', 'src': 'PART2 1 (H); W177B_REPORT decision 2', 'e10_line': 1439, 'rating_e10': 'H', 'fix': 'A11',
     'old': 'The planning search again returned the plan without shared storage',
     'new': ['The planning search was not repeated at this instance; the reference unit was evaluated against the plan '
             'without storage.'],
     'status': 'resolved', 'rating_now': '-',
     'evidence': 'W177b-2\' item (e): only the pair (x = 0, reference unit) and the x = 0 continuation ran on the 3 x 3; the '
                 'sentence now says so. The pointer kept "Section~" (A11 difference, L).'},
    {'id': '4.6-1', 'src': 'PART2 1 (M); W177B_REPORT decision 4', 'e10_line': 1499, 'rating_e10': 'M', 'fix': 'A12',
     'old': None, 'new': ['The rates other than 2\\,\\% re-weight the per-year costs of the 2\\,\\% evaluations rather than '
                          're-solving them; this is exact at the plan without storage and first-order at the reference '
                          'unit, whose ageing couples the years.',
                          'rates other than 2\\,\\% re-weight the 2\\,\\% evaluations'],
     'status': 'resolved', 'rating_now': '-',
     'evidence': 'the rate enters only the block weights (_get_admm_block_weight, W177b-2\' LM-7); at x = 0 no state links the '
                 'years, so the re-weighting is exact; at the unit the 2 %-optimal schedule is held fixed, so the value '
                 'at r != 2 % is first-order in the rate change (envelope argument, local optimum); its size is unmeasured '
                 '(needs solves) and the verdicts sit at 2.57x-8.78x'},
    {'id': '4.6-2', 'src': 'PART2 1 (H); W177B_REPORT decision 1', 'e10_line': 1520, 'rating_e10': 'H', 'fix': 'A13',
     'old': 'It changes two verdicts',
     'new': ['The investment-year comparison above is not among the 60; on it the credit changes the verdict, from '
             'determinate in gross terms to within resolution.'],
     'status': 'changed', 'rating_now': 'M (until Appendix E is rebuilt)',
     'evidence': 'count resolved (one verdict among the 60; the 2035 - 2030 pair outside the 60, 3.46x -> 0.19x). NEW: the '
                 'sentence points to "the full claims table (\\ref{app:results})", and Appendix E at 9fd19e7 still holds '
                 'the submitted per-block results table (tab:cs3_results_summary_years_days), no claims table -- the same '
                 'gap as letter row LM-9; Addendum 77 ruling 3 has the expert composing Appendix E from the export. Not '
                 'stated by the sentence: the sign of the investment-year difference also flips net of salvage '
                 '(+42.5 -> -2.3 kEUR); "changes the verdict ... to within resolution" is true as written.'},
    {'id': 'C-1', 'src': 'PART2 1 (M)', 'e10_line': 1581, 'rating_e10': 'M', 'fix': 'A16(a)',
     'old': 'instead, each evaluation is certified by an explicit stopping rule',
     'new': ['each evaluation behind a reported comparison is either certified by an explicit stopping rule or reported in '
             'an uncertified form with its error bar'],
     'status': 'changed', 'rating_now': 'L',
     'evidence': 'the uncertified evaluations are now covered. Remaining: the evaluations behind the 3 x 3 comparison '
                 '(4.5, stopped by the production exit, resolution 18.0 kEUR from the pair campaign, not the uncertified '
                 'form) and the 41 search evaluations behind 4.1\'s "beyond the search resolution" are neither; the '
                 'section 4 opening (A7) names that third status, the Conclusions sentence does not. No number or verdict '
                 'changes.'},
    {'id': 'C-2', 'src': 'PART2 1 (M)', 'e10_line': 1585, 'rating_e10': 'M', 'fix': 'A16(b)',
     'old': 'and every other evaluated plan, from 0.5 to 5~MWh and at one or two nodes, also loses determinately',
     'new': ['every other plan of Table~\\ref{tab:res_value_ladder}, from 0.5 to 5~MWh and at one or two nodes, also loses '
             'determinately'],
     'status': 'resolved', 'rating_now': '-', 'ladder_check': True,
     'evidence': 'scoped to tab:res_value_ladder: its 15 other plans are 0.5-5 MWh, 1-2 nodes and each loses determinately '
                 '(frozen T1 gross verdicts, computed below); the three-node and search-resolution-only plans are outside '
                 'the table'},
    {'id': 'C-3', 'src': 'PART2 1 (M)', 'e10_line': 1589, 'rating_e10': 'M', 'fix': 'A16(c)',
     'old': 'cannot be assessed apart from its ageing or from the coordination',
     'new': ['ageing decides the value of shared storage on this system, the same unit being at break-even without it and '
             'losing with it'],
     'status': 'resolved', 'rating_now': '-',
     'evidence': 'T8: no ageing -4.1 kEUR within resolution (0.32x), every aged calibration determinate (2.50x-5.74x); T6: '
                 'coordination 90.9 MEUR; H rows: the unit pays at 1.75x and 2x the flexibility price'},
    {'id': 'flag-1180', 'src': 'PART2 1 (section 4 preamble flag, not rated); W177B_REPORT decision 6', 'e10_line': 1180,
     'rating_e10': 'flag (expert)', 'fix': 'A7',
     'old': 'Every reported number is a certified recourse evaluation or a difference between two of them',
     'new': ['and its status is stated: certified, uncertified with its bar, or, at the multi-scenario instance, stopped by '
             'the production exit'],
     'status': 'resolved', 'rating_now': '-', 'evidence': 'three statuses named, matching T2/T9 (certified / uncertified) and '
                                                         'T11 (production exit)'},
    {'id': 'S3-12', 'src': 'PART2 2 (section 3 opening)', 'e10_line': 990, 'rating_e10': 'L', 'fix': None,
     'old': 'Every candidate plan is compared with the plan without shared storage', 'new': [],
     'status': 'still', 'rating_now': 'L', 'evidence': 'not addressed; text unchanged (l. 1141)'},
    # ---- letter rows about text in main.tex (manuscript side only; the letter side is W178c's) --------------------------
    {'id': 'LM-4 (manuscript side)', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'M', 'fix': 'A15', 'old': None,
     'new': ['the $5 \\times 5$ scenario set of the submitted version could not be evaluated within the memory available '
             'and is left to future work'], 'status': 'resolved', 'rating_now': '-',
     'evidence': 'REVISION_CONTEXT.md l. 446 / 464 and brief Addendum 44 (memory), as W177b-2\' found'},
    {'id': 'LM-5 (manuscript side)', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'M', 'fix': 'A1', 'old': None,
     'new': ['test system with utility-scale lithium-ion storage'], 'status': 'resolved', 'rating_now': '-',
     'evidence': 'chemistry also in 3.5 (lithium-ion)'},
    {'id': 'LM-6 (manuscript side)', 'src': 'PART2 3; W177B_REPORT decision 3', 'e10_line': None, 'rating_e10': 'M',
     'fix': 'A3', 'old': None,
     'new': ['Demand growth and RES capacity follow predefined trajectories, while policy changes, infrastructure '
             'retirements and the cost of capital as a stochastic quantity are outside the present formulation'],
     'status': 'resolved', 'rating_now': '-', 'evidence': 'M2 now in the introduction'},
    {'id': 'LM-7 (manuscript side)', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'M', 'fix': 'A12', 'old': None,
     'new': ['this is exact at the plan without storage and first-order at the reference unit'], 'status': 'resolved',
     'rating_now': '-', 'evidence': 'see row 4.6-1'},
    {'id': 'LM-8 (manuscript side)', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'M', 'fix': 'A4', 'old': None,
     'new': ['No convergence guarantee is available for consensus ADMM on a nonconvex recourse of this kind'],
     'status': 'resolved', 'rating_now': '-',
     'evidence': 'the manuscript says "guarantee"; whether the letter\'s "theorem" wording matches is W178c\'s'},
    {'id': 'LM-9', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'M (until the rebuild)', 'fix': None,
     'old': None, 'new': ['\\label{app:results}'], 'status': 'still', 'rating_now': 'M (until the rebuild)',
     'still_needle': 'tab:cs3_results_summary_years_days',
     'evidence': 'Appendix E still holds the submitted per-block table; now also cited by the new 4.6 sentence (row 4.6-2)'},
    {'id': 'LM-10 (manuscript side)', 'src': 'PART2 3; W177B_REPORT decision 3; Addendum 77 ruling 3 (L4)',
     'e10_line': None, 'rating_e10': 'M', 'fix': None, 'old': None, 'new': [],
     'status': 'moot in main.tex (letter trimmed, L4)', 'rating_now': '- in main.tex (letter side: W178c)',
     'absent_needles': ['representative-day-weighted'],
     'evidence': 'ruling 3 trims R3.7/3.8 instead of adding the wording to the paper ("the paper has no annual figures"); '
                 'main.tex still has no "representative-day-weighted" (searched, typeset and comments), as intended. The '
                 'letter side is W178c\'s.'},
    {'id': 'LL-8 (manuscript side)', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'L', 'fix': None, 'old': None,
     'new': [], 'status': 'still', 'rating_now': 'L', 'absent_needles': ['different local solution', 'penalty schedule'],
     'evidence': '4.8 still states no penalty-schedule limitation (searched: "different local solution", "penalty '
                 'schedule")'},
    {'id': 'LL-9 (manuscript side)', 'src': 'PART2 3', 'e10_line': None, 'rating_e10': 'L', 'fix': 'A8', 'old': None,
     'new': ['the converter is not credited'], 'status': 'resolved', 'rating_now': '-', 'evidence': 'see row P0-C1'},
    # ---- Part 3 rows the fixes address (W177B_REPORT decision 6; P5_15_W177B_PART3_E10C104.md) --------------------------
    {'id': 'A-1', 'src': 'PART3 (M); decision 6', 'e10_line': 1767, 'rating_e10': 'M', 'fix': 'A6, A17, A19',
     'old': 'from the next cycle in a fresh run or from the stopping cycle of the earlier run',
     'new': ["or, in the three reference evaluations continued from an earlier run, from that run's stopping cycle (an "
             "evaluation re-run on an earlier record replays it bitwise to its first residual pass and holds from the "
             "cycle after it)", "an evaluation continued from, or re-run on, an earlier record is capped 100 cycles after "
                                "that record's stopping cycle"],
     'status': 'changed', 'rating_now': 'L',
     'evidence': 'the three cases are named and the re-runs\' hold (k0 + 1) and cap (N_old + 100, two above 300) now match '
                 'the specs. Remaining: for the three reference continuations the text holds "from that run\'s stopping '
                 'cycle", while the hooks apply the holds "for every cycle > N" (p515_s53_w101_settling_continuation_hooks.py'
                 ':19, as W177b-3 found): one cycle early, unchanged from e10c104'},
    {'id': 'A-3', 'src': 'PART3 (M); decision 6', 'e10_line': 1775, 'rating_e10': 'M', 'fix': 'A19, A4',
     'old': "interface schedules at the interface price",
     'new': ["interface active-power schedules at the expected wholesale price of each hour, block-weighted",
             'priced at the expected wholesale price, must be within'],
     'status': 'resolved', 'rating_now': '-',
     'evidence': 'p515_s53_w118_resettle_hooks.py:336-369: t_sum = sum w(y,d) pi_bar (p_DSO - p_TSO), active power only '
                 '(W177b-3 A-3); "the interface price" no longer occurs in main.tex'},
    {'id': 'nomenclature', 'src': 'PART3 (d); decision 6', 'e10_line': 271, 'rating_e10': 'author item', 'fix': 'A2',
     'old': '$\\pi_t$, $\\bar{\\pi}_t$', 'new': ['Expected wholesale price of hour $t$ over the market scenarios',
                                                 'Operating cost of scenario $s$ over the horizon, weighted and discounted'],
     'status': 'resolved', 'rating_now': '-', 'evidence': 'pi_t row -> pi-bar; s dropped from the t_sum row; C^Op_s added '
                                                          '(and the daily C^Op_{y,d,m,o} row kept, follow-up hunk 3)'},
    {'id': 'stale pointers', 'src': 'PART3 (2c); decision 6', 'e10_line': 1555, 'rating_e10': 'author item',
     'fix': 'A14, A20, A18, A11',
     'old': 'they are counted as clean under the rule of Subsection~\\ref{subsubsec:certification}',
     'new': ['they are counted as clean under the rule of \\ref{app:admm_updated_certification}',
             'certification parameters of \\ref{app:admm_updated_certification}',
             'adaptive barrier strategy (\\ref{app:admm_updated_settings})', 'Section~\\ref{app:admm_updated_settings})'],
     'status': 'resolved', 'rating_now': '- (A11 keeps "Section~", L)', 'evidence': 'four pointers now name A.5 / A.8'},
    {'id': 'cache clause', 'src': 'PART3 (2b); decision 6', 'e10_line': 1164, 'rating_e10': 'author item', 'fix': 'A6',
     'old': 'run with and without the budget, and their combinations',
     'new': ['the cache of the baseline search was seeded with the single-node ladders of the initial design, including '
             'ladder points over the budget'], 'status': 'resolved', 'rating_now': '-',
     'evidence': 'W177b-3\' cache tables: s47 30 single-node ladder points (15 over budget) + c_star + x = 0; s51 6 and s53 '
                 '26 committed certified records at m = 2; s53 holds the 13 multi-node 2030 plans (numbers checked on l. '
                 '1316). Not stated, unchanged since e10c104: the s47 cache also held c_star (the re-certified earlier '
                 'result, off-lattice, multi-node) and x = 0 (L at most; no number or verdict depends on it)'},
    {'id': 'table numbering', 'src': 'PART3 letter pointers; decision 5', 'e10_line': None, 'rating_e10': 'author item',
     'fix': 'A2', 'old': None, 'new': ['\\addtocounter{table}{-1}'], 'status': 'resolved', 'rating_now': '- (static)',
     'evidence': 'three \\addtocounter{table}{-1}, one after each nomenclature longtable (STRUCT A2); the pointers are W178c\'s'},
]


# ======================================================================================================================
#  5. evaluation, application
# ======================================================================================================================
SHA_CITE_RX = re.compile(r'([\w./-]+\.(?:json|py|xlsx|tex|md))(?::\d+)?\s*\(sha256 ([0-9a-f]{8})')


def evaluate(X, spec, t, line):
    if spec['kind'] == 'carry174':
        w = X.w174tok.get((spec['line260'], spec['col260']))
        l260 = X.lines260[spec['line260'] - 1] if spec['line260'] <= len(X.lines260) else ''
        ok_tok = w is not None and str(w.get('written')).lower() == spec['w'].lower()
        ok_frag = nsp(spec['frag']) in nsp(l260) and nsp(spec['frag']) in nsp(line)
        cites = SHA_CITE_RX.findall(str((w or {}).get('counterpart')))
        sha_now = {p: (_sha(p)[:8] if os.path.exists(os.path.join(REPO, p)) else None) for p, _h in cites}
        ok_src = bool(cites) and all(sha_now[p] == h for p, h in cites)
        st = w['status'] if (ok_tok and ok_frag and ok_src) else 'MISMATCH'
        return {'status': st, 'counterpart': f"W174a committed check {w.get('check_id') if w else None} at 260bd83 l. "
                f"{spec['line260']}:{spec['col260']} ({X.inputs['W174A_JSON']['path']}, sha256 "
                f"{X.inputs['W174A_JSON']['sha256'][:8]}): {(w or {}).get('counterpart')}",
                'counterpart_value': (w or {}).get('counterpart_value'), 'rule': 'carried committed check (token, context '
                'fragment in both lines, every cited source sha256 unchanged)',
                'carry': {'token_ok': ok_tok, 'fragment_ok': ok_frag, 'cited_sha256_then_now': {p: [h, sha_now[p]] for p, h in cites}},
                'note': spec.get('why')}
    return B1R.evaluate(X, spec, t, line)


def resolve_specs(decls, e10):
    """OLD / SAME -> the W177b-1' spec objects; unresolved references are listed (an integrity failure)."""
    bad = []
    for d in decls:
        out = []
        for s in d['specs']:
            if s.get('kind') == '_same':
                src = e10.get(s['e10_line'])
                if src is None:
                    bad.append({'line': d['line'], 'ref': s})
                    continue
                out += [dict(x, carried_from_e10c104=s['e10_line']) for x in src['specs']]
            elif s.get('kind') == '_old':
                src = e10.get(s['e10_line'])
                hits = [x for x in (src['specs'] if src else []) if str(x['w']).lower() == s['w'].lower()]
                if len(hits) <= s['k']:
                    bad.append({'line': d['line'], 'ref': s, 'found': len(hits)})
                    continue
                out.append(dict(hits[s['k']], carried_from_e10c104=s['e10_line']))
            else:
                out.append(s)
        d['specs'] = out
    return bad


def apply_declarations(X, toks, decls, lines):
    stale, claimed_twice, decl_out = [], [], []
    by_line = collections.defaultdict(list)
    for t in toks:
        by_line[t['line']].append(t)
    seen = collections.Counter(d['line'] for d in decls)
    dup = [ln for ln, n in seen.items() if n > 1]
    for d in decls:
        text = lines[d['line'] - 1]
        cand = [t for t in by_line[d['line']] if t['status'] != 'excluded']
        if len(cand) != len(d['specs']):
            stale.append({'line': d['line'], 'why': f"{len(d['specs'])} specs for {len(cand)} tokens",
                          'tokens': [t['written'] for t in cand], 'specs': [s['w'] for s in d['specs']]})
            continue
        bad = [(t['written'], s['w']) for t, s in zip(cand, d['specs']) if t['written'].lower() != str(s['w']).lower()]
        if bad:
            stale.append({'line': d['line'], 'why': 'written differs from the declaration', 'pairs': bad})
            continue
        res = []
        for t, s in zip(cand, d['specs']):
            if t['status'] is not None:
                claimed_twice.append(t['key'])
                continue
            try:
                r = evaluate(X, s, t, text)
            except Exception as e:  # noqa: BLE001 -- listed as a run failure
                stale.append({'line': d['line'], 'why': f'evaluation error {type(e).__name__}: {e}', 'written': t['written']})
                continue
            if r.get('status') is None:
                continue
            t.update(r)
            t['declared_by'] = d.get('declared_by', 'new declaration (W178b)')
            if s.get('carried_from_e10c104'):
                t['carried_from_e10c104_line'] = s['carried_from_e10c104']
            res.append({'written': t['written'], 'col': t['col'], 'status': r['status']})
        decl_out.append({'line': d['line'], 'declared_by': d.get('declared_by', 'new declaration (W178b)'), 'tokens': res})
    return stale, claimed_twice, decl_out, dup


# ======================================================================================================================
#  6. checks: (b) items, rows, salvage, J row, (c)
# ======================================================================================================================
def in_ranges(lo, hi, ranges):
    return any(a <= lo and hi <= b for a, b in ranges)


def check_items(X, LOC, lines, fixes):
    out = []
    body_all = '\n'.join(W164.split_comment(ln)[0] for ln in lines)
    labels_body = collections.Counter(re.findall(r'\\label\{([^}]+)\}', body_all))
    refs_body = re.findall(r'\\(?:eq)?ref\{([^}]+)\}', body_all)
    for iid in ITEM_IDS:
        it = fixes.get(iid, {'fragments': [], 'comment': '', 'header_line': None})
        ranges = ADDRESSED[iid]
        row = {'id': iid, 'where': LOCATION[iid], 'fix_header_line': it.get('header_line'), 'addressed_ranges': ranges,
               'fragments': [], 'old_absent': [], 'extra_new': [], 'comment_absent': [], 'struct': None, 'diffs': [],
               'lines_now': []}
        fails = []
        for f in it['fragments']:
            res, order_ok, resid = LOC.ordered(f['core_lines'], ranges)
            # a residue is allowed only if it is the \textcolor{blue}{ ... } wrapper boilerplate the parser drops
            ins = [dict(r, wrapper_only=not re.sub(r'\\textcolor\{blue\}\{|\}', '', r['text'])) for r in resid]
            elsewhere = {fl: LOC.find(txt) for fl, txt in f['core_lines'] if nows(txt) and not
                         any(r[0] == fl and r[1] for r in res)}
            row['fragments'].append({'fix_lines': [f['first_line'], f['last_line']], 'n_lines': len(res),
                                     'n_found_in_order': sum(1 for r in res if r[1]), 'order_ok': order_ok,
                                     'missing': [{'fix_line': r[0], 'text': nsp(dict(f['core_lines'])[r[0]])[:200],
                                                  'in_ranges_out_of_order': r[3], 'elsewhere_in_main': elsewhere.get(r[0])}
                                                 for r in res if not r[1]],
                                     'found_lines_now': sorted({x for r in res if r[1] for x in r[2]}),
                                     'residues_between_found_lines': ins,
                                     'text_head': nsp(f['core'])[:160]})
            row['lines_now'] += [x for r in res if r[1] for x in range(r[2][0], r[2][1] + 1)]
            fails += [f"frag{f['first_line']}:line{r[0]}:missing" for r in res if not r[1]]
            fails += [f"frag{f['first_line']}:insertion:{r['after_fix_line']}-{r['before_fix_line']}" for r in ins
                      if not r['wrapper_only']]
            if not order_ok:
                fails.append(f"frag{f['first_line']}:order")
        for o in OLD_ABSENT.get(iid, []):
            present = LOC.find(o)
            row['old_absent'].append({'old': o, 'absent_from_typeset': not present, 'lines_now': present,
                                      'in_comments': LOC.find(o, 'comment')})
            if present:
                fails.append(f'old_present:{o}')
        for desc, frag in EXTRA_NEW.get(iid, []):
            hits = LOC.find_in(frag, ranges)
            row['extra_new'].append({'what': desc, 'fragment': frag, 'found': bool(hits), 'lines_now': hits,
                                     'elsewhere_in_main': LOC.find(frag) if not hits else None})
            row['lines_now'] += [x for a, b in hits for x in range(a, b + 1)]
            if not hits:
                fails.append(f'extra_missing:{desc}')
        for c in COMMENT_ABSENT.get(iid, []):
            hits = LOC.find(c, 'comment')
            row['comment_absent'].append({'comment': c, 'absent': not hits, 'lines_now': hits})
            if hits:
                fails.append(f'comment_present:{c}')
        if iid == 'A2':
            ends = [i for i, ln in enumerate(lines, 1) if W164.split_comment(ln)[0].strip() == '\\end{longtable}']
            nxt = []
            for e in ends:
                j = e
                while j < len(lines) and not W164.split_comment(lines[j])[0].strip():
                    j += 1
                nxt.append({'end_longtable_line': e, 'next_line': j + 1, 'next': W164.split_comment(lines[j])[0].strip()})
            n_add = sum(1 for ln in lines if W164.split_comment(ln)[0].strip() == '\\addtocounter{table}{-1}')
            ok = len(ends) == 3 and n_add == 3 and all(x['next'] == '\\addtocounter{table}{-1}' for x in nxt)
            row['struct'] = {'rule': STRUCT_NOTES['A2'], 'end_longtable': nxt, 'n_addtocounter': n_add, 'ok': ok}
            row['lines_now'] += [x['next_line'] for x in nxt]
            if not ok:
                fails.append('struct')
        if iid == 'B1':
            want = ('eq:cohort_rated', 'eq:cohort_rated_energy', 'eq:cohort_available', 'eq:cohort_available_energy',
                    'eq:available_power', 'eq:available_capacity')
            dup = sorted(k for k, n in labels_body.items() if n > 1)
            undefined = sorted({r for r in refs_body if r not in labels_body})
            # B1's range ends at the comment "% ---- C.2 New: storage operation ..." (at 9a697cc, the HEAD the fix was
            # written against). The paragraph "The storage agent publishes ..." after the block at the HEAD must lie
            # AFTER that comment at 9a697cc, i.e. outside the replaced range (retained, not an addition).
            c2, para = '% ---- C.2 New: storage operation', 'The storage agent publishes $S^{\\text{Av}}_{e,y}$'
            l97 = getattr(X, 'lines_9a697cc', [])
            b97 = {'c2_comment_line': [i for i, ln in enumerate(l97, 1) if c2 in ln],
                   'paragraph_line': [i for i, ln in enumerate(l97, 1) if para in ln]}
            bnow = {'c2_comment_line': [i for i, ln in enumerate(lines, 1) if c2 in ln],
                    'paragraph_line': [i for i, ln in enumerate(lines, 1) if para in ln],
                    'eq_available_capacity_label_line': [i for i, ln in enumerate(lines, 1)
                                                         if '\\label{eq:available_capacity}' in ln],
                    'next_subsubsection_line': [i for i, ln in enumerate(lines, 1)
                                                if 'Storage Operation in the Network Models' in ln]}
            boundary_ok = (len(b97['c2_comment_line']) == 1 and len(b97['paragraph_line']) == 1
                           and b97['paragraph_line'][0] > b97['c2_comment_line'][0] and not bnow['c2_comment_line']
                           and len(bnow['paragraph_line']) == 1
                           and bnow['eq_available_capacity_label_line'][0] < bnow['paragraph_line'][0]
                           < bnow['next_subsubsection_line'][0])
            ok = all(labels_body.get(k, 0) == 1 for k in want) and not dup and not undefined and boundary_ok
            row['b1_range_end'] = {'at_9a697cc': b97, 'at_head': bnow, 'ok': boundary_ok,
                                   'reading': 'the paragraph after eq:available_capacity was outside B1\'s replaced range '
                                              'at 9a697cc and is retained; the boundary comment line was removed at the '
                                              'HEAD (a comment, not typeset)'}
            row['struct'] = {'rule': STRUCT_NOTES['B1'], 'labels_defined': {k: labels_body.get(k, 0) for k in want},
                             'duplicate_labels_typeset': dup, 'undefined_refs_typeset': undefined,
                             'refs_to_cohort_rated_energy': refs_body.count('eq:cohort_rated_energy'),
                             'a2_range_cited': bool(LOC.find('\\eqref{eq:cohort_rated}--\\eqref{eq:available_capacity}')),
                             'a2_range_lines': LOC.find('\\eqref{eq:cohort_rated}--\\eqref{eq:available_capacity}'), 'ok': ok}
            if not ok:
                fails.append('struct')
        if iid == 'B4':
            row['b4'] = {'sentence': B4_SENTENCE, 'commented_lines': LOC.find(B4_SENTENCE, 'comment'),
                         'typeset': bool(LOC.find(B4_SENTENCE))}
            row['lines_now'] += [x for a, b in row['b4']['commented_lines'] for x in range(a, b + 1)]
        explained_ids = set()
        for dd in DIFFS.get(iid, []):
            hits = LOC.find(dd['head'], 'comment' if dd['head_where'] == 'comment' else 'body')
            row['diffs'].append(dict(dd, head_found=bool(hits), head_lines=hits,
                                     word_diff=word_diff(dd['fix'], dd['head']) if dd['head_where'] == 'body' else None))
            if hits:
                explained_ids |= set(dd['explains'])
        row['failing_checks'] = fails
        row['unexplained_failing_checks'] = [x for x in fails if x not in explained_ids]
        row['declared_differences_without_failing_check'] = [x for x in explained_ids if x not in fails]
        if iid == 'B4':
            outcome = ('declined (author; Addendum 77)' if row['b4']['commented_lines'] and not row['b4']['typeset']
                       else 'B4 APPLIED OR MOVED')
            explained = not fails
        elif not fails and not row['diffs']:
            outcome, explained = 'applied verbatim', True
        elif row['diffs'] and not row['unexplained_failing_checks'] and all(d['head_found'] for d in row['diffs']) \
                and not row['declared_differences_without_failing_check']:
            outcome, explained = 'applied with changes', True
        else:
            outcome, explained = 'ABSENT or UNDECLARED DIFFERENCE', False
        row['outcome'], row['explained'] = outcome, explained
        row['lines_now'] = sorted(set(row['lines_now']))
        out.append(row)
    return out


def check_rows(X, LOC, LOC_E10, toks):
    tok = {}
    for t in toks:
        tok.setdefault((t['line'], t['written']), t)          # the first occurrence on the line
    out = []
    for r in ROWS:
        o = dict(r)
        o['old_at_e10c104'] = (bool(LOC_E10.find(r['old'])) or bool(LOC_E10.find(r['old'], 'comment'))) if r['old'] else None
        o['old_lines_e10c104'] = LOC_E10.find(r['old']) if r['old'] else None
        o['old_now'] = bool(LOC.find(r['old'])) if r['old'] else None
        o['old_lines_now'] = LOC.find(r['old']) if r['old'] else None
        o['new_now'] = [{'text': n, 'found': bool(LOC.find(n)), 'lines_now': LOC.find(n)} for n in r['new']]
        if r.get('still_needle'):
            o['still_needle_now'] = LOC.find(r['still_needle'])
        if r.get('absent_needles'):
            o['absent_needles_now'] = {n: LOC.find(n) + LOC.find(n, 'comment') for n in r['absent_needles']}
        if r.get('token'):
            t = tok.get(r['token'])
            o['token_now'] = {'line': r['token'][0], 'written': r['token'][1], 'status': t and t['status'],
                              'counterpart': t and str(t.get('counterpart'))[:300]}
        st = r['status']
        new_ok = all(n['found'] for n in o['new_now'])
        if st == 'still':
            ok = (o['old_now'] is True if r['old'] else True) and (not r.get('still_needle') or bool(o['still_needle_now'])) \
                and (not r.get('absent_needles') or not any(o['absent_needles_now'].values())) and new_ok
        elif st == 'resolved':
            ok = (o['old_now'] is False if r['old'] else True) and new_ok
        else:
            ok = new_ok
        if r.get('absent_needles') and st != 'still':
            ok = ok and not any(o['absent_needles_now'].values())
        if r['old'] and not o['old_at_e10c104']:
            ok = False
        if r.get('token'):
            ok = ok and o['token_now']['status'] in ('match', 'derived')
        o['consistent_with_text'] = bool(ok)
        out.append(o)
    return out


def code_loc(X, key, needle, expect_line=None):
    lines = X.texts[key].split('\n') if key in X.texts else _text(X.inputs[key]['path']).split('\n')
    hits = [i + 1 for i, ln in enumerate(lines) if needle in ln]
    return {'file': X.inputs[key]['path'], 'needle': needle, 'lines': hits, 'expected_line': expect_line,
            'ok': (hits == [expect_line]) if expect_line else bool(hits)}


SESD_866_905 = ((866, 'def _build_terminal_salvage_value_expression(shared_ess_data, model, shared_ess_idx):'),
                (871, 'terminal_year_idx = len(shared_ess_data.years) - 1'),
                (872, 'terminal_discount = _get_terminal_discount_factor(shared_ess_data)'),
                (879, 'min_soh = shared_energy_storage.soh_min'),
                (883, 'e_rated = model.es_e_rated_per_unit[y_inv, terminal_year_idx]'),
                (884, 'e_available = model.es_e_available_per_unit[y_inv, terminal_year_idx]'),
                (889, 'e_available = e_rated * model.es_soh_per_unit_cumul[y_inv, terminal_year_idx]'),
                (890, 'usable_energy_above_eol = (e_available - min_soh * e_rated) / (1.00 - min_soh)'),
                (892, 'params.recycling_floor_fraction * e_rated'),
                (893, '+ (1.00 - params.recycling_floor_fraction) * usable_energy_above_eol'),
                (895, 'expected_unit_cost = _get_expected_energy_investment_cost(shared_ess_data, year_inv)'),
                (896, '_, _, remaining_life_fraction = _get_remaining_calendar_life('),
                (900, 'terminal_discount'), (901, '* params.energy_recovery_fraction'), (902, '* expected_unit_cost'),
                (903, '* remaining_life_fraction'), (904, '* residual_energy'))


def salvage_section(X, confirm_now, sal):
    lines = X.texts['CODE_SESD'].split('\n')
    code = [{'line': ln, 'needle': nd, 'ok': nd in lines[ln - 1]} for ln, nd in SESD_866_905]
    sv = X.j['ESS_PARAMS']['salvage_value']
    rep = []
    for x in sal['reproduction']:
        printed = x['DF_term'] * x['expected_c_E'] * x['E_MWh'] * (x['T_rem'] / float(X.j['ESS_PARAMS']['ageing']['calendar_life_years'])) \
            * x['normalized_health']
        rep.append({'cell': x['cell'], 'investment_year': x['investment_year'], 'node': x['node'], 'S_MVA': x['S_MVA'],
                    'E_MWh': x['E_MWh'], 'terminal_SoH': x['terminal_SoH_recorded'], 'terminal_SoH_source': x['terminal_SoH_source'],
                    'expected_c_E_eur_per_MWh': x['expected_c_E'], 'probabilities_omega_c': x['probabilities'],
                    'DF_term': x['DF_term'], 'T_rem': x['T_rem'], 'T_cal': X.j['ESS_PARAMS']['ageing']['calendar_life_years'],
                    'normalized_health': x['normalized_health'], 'printed_form_eur': printed,
                    'production_form_eur': x['production_form_eur'], 'recorded_salvage_eur': x['recorded_salvage_eur'],
                    'recorded_source': x['recorded_source'], 'printed_rel_diff_vs_recorded': printed / x['recorded_salvage_eur'] - 1.0,
                    'old_printed_form_over_recorded': x['frame_over_recorded']})
    c1 = next(c for c in confirm_now if c['id'] == 'C1')
    mapping = [
        ('sum over e in E^S', 'one ESSO node model per shared ESS; get_salvage_value sums model.salvage_value over the nodes (W177a C1 locations)'),
        ('sum over y^Inv in Y', 'for y_inv in model.years (l. 876)'),
        ('cbar^{Inv,E}_{y^Inv} = sum_c omega_c c^{Inv,E}_{y^Inv,c}', '_get_expected_energy_investment_cost (l. 895; l. 925-929): '
         'probability-weighted ENERGY cost of the installation year; the probabilities are read from SRP1_ESS.xlsx sheet '
         'Scenarios into shared_ess_data.prob_market_scenarios (shared_energy_storage_data.py:1743, after '
         'shared_resources_planning.py:9003) -- the cost-trajectory probabilities omega_c despite the attribute name'),
        ('E^Inv_{e,y^Inv}', 'e_rated = es_e_rated_per_unit[y_inv, terminal] (l. 883): the cohort\'s rated energy (W177a C1 '
                            'reproduction uses the cell\'s E; the rated-unit rows are W178a\'s)'),
        ('1 / (1+r)^(y^End - y0)', 'terminal_discount = 1 / (1 + r)^(sum of block lengths) (l. 872, l. 910-912); y^End = '
                                   'years[0] + 15 = 2040'),
        ('T^Rem / T^Cal', 'remaining_life_fraction = min(max(T_cal - age_at_terminal, 0) / T_cal, 1) (l. 896-898, l. 915-922); '
                          'the min(., 1) never binds since y^Inv <= y^End'),
        ('(SoH - SoH^Min) / (1 - SoH^Min)', 'usable_energy_above_eol / e_rated = (e_available - min_soh e_rated) / ((1 - min_soh) '
                                            'e_rated) with e_available = e_rated SoH (l. 884 / 889-890): terminal end-of-block '
                                            'SoH, also under the mid-block variant'),
        ('(not printed) energy_recovery_fraction f_rec', f"multiplies the credit (l. 901); = {sv['energy_recovery_fraction']} in "
                                                          f"SRP1_ESS_Params.json: the printed form omits a factor equal to 1"),
        ('(not printed) recycling_floor_fraction phi', f"residual = phi E_rated + (1 - phi) usable_above_eol (l. 891-894); = "
                                                        f"{sv['recycling_floor_fraction']}: the printed form omits a term equal to 0"),
        ('converter not credited', 'no S / power-cost term anywhere in l. 866-905 (only expected_unit_cost of energy)'),
    ]
    return {'code_lines_866_905': code, 'code_ok': all(c['ok'] for c in code),
            'code_file': {'path': X.inputs['CODE_SESD']['path'], 'sha256': X.inputs['CODE_SESD']['sha256'],
                          'committed_clean': X.inputs['CODE_SESD']['committed_clean']},
            'parameters': {k: sv[k] for k in ('enabled', 'energy_recovery_fraction', 'recycling_floor_fraction', 'cost_basis',
                                              'health_basis', 'calendar_life_basis')},
            'printed_equals_coded_at_these_parameters': sv['energy_recovery_fraction'] == 1.0 and sv['recycling_floor_fraction'] == 0.0,
            'symbol_to_code': [{'printed': a, 'code': b} for a, b in mapping],
            'reproduction': rep, 'printed_reproduces_recorded': all(abs(r['printed_rel_diff_vs_recorded']) < 1e-9 for r in rep),
            'w177a_c1_recomputed': {'outcome': c1['outcome'], 'production_reproduces_records': c1['production_reproduces_records'],
                                    'items_a_to_d': {k: v['outcome'] for k, v in c1['items'].items()}}}


def part_c(X, lines_now):
    w = X.j['W155_SOH050']
    rep = w['reports']['e_soh050']
    series, ok = LX.ageing_series(X.j['W155_RECORD'])
    ae, efc = M46A.pv(series['soh_used']), M46A.pv(series['efc_per_day'])
    g070 = w['reports']['g070_neutrality']
    t8 = {r['arm']: r for r in X.t['ageing']['rows']}
    hdr = next(i for i, ln in enumerate(lines_now, 1) if 'Calibration & Value' in ln and 'EFC/day' in ln)
    row = next(i for i in range(hdr + 1, len(lines_now) + 1) if lines_now[i - 1].strip().startswith('Baseline, 0.50 floor'))
    cells = [c.strip() for c in lines_now[row - 1].split('\\\\')[0].split('&')]
    cols = [c.strip() for c in lines_now[hdr - 1].split('\\\\')[0].split('&')]
    t8_printed = {}
    for i in range(hdr + 1, row):
        ln = lines_now[i - 1]
        if '&' in ln:
            c = [x.strip() for x in ln.split('\\\\')[0].split('&')]
            t8_printed[c[0]] = {'AE': c[cols.index('$AE$')], 'EFC/day': c[cols.index('EFC/day')]}
    defs = {
        'T8_frozen_fields': f"{FZ_REL} tables.ageing.rows[].AE / .EFC; tables.ageing.column_labels carries no AE / EFC label "
                            f"({sorted(X.t['ageing']['column_labels'])})",
        'W157_builder': code_loc(X, 'CODE_W157', "'floor_year_070': r.get('floor_year'), 'AE': r.get('AE'), 'EFC': r.get('EFC')"),
        'W157_source': code_loc(X, 'CODE_W157', "'SX': (os.path.join(S53, 'w142_resettle_ext_v6', "
                                                "'w142_ext_summary_after_07_pb_y2025_n5_v6.json'),"),
        'W142X_compute_AE': code_loc(X, 'CODE_W142X', "'AE': M46A.pv(series['soh_used']) if ok else None,"),
        'W142X_compute_EFC': code_loc(X, 'CODE_W142X', "'EFC': M46A.pv(series['efc_per_day']) if ok else None})"),
        'W142X_definition_AE': X.j['SX_SUMMARY']['definitions']['per_cell']['AE'],
        'W142X_definition_EFC': X.j['SX_SUMMARY']['definitions']['per_cell']['EFC'],
        'W155_compute': code_loc(X, 'CODE_W155', "'AE': M46A.pv(series['soh_used']) if ok else None, 'EFC': "
                                                 "M46A.pv(series['efc_per_day']) if ok else None})"),
        'W155_reader': code_loc(X, 'CODE_W155', 'series, ok = LX.ageing_series(rec)'),
        'W155_definition': w['definitions']['per_cell']['AE_EFC'],
        'pv_formula': code_loc(X, 'CODE_M46A', 'return sum(w * v for w, v in zip(W, values)) / sum(W)'),
        'pv_weights': {'W': list(M46A.W), 'YEARS': list(M46A.YEARS), 'rule': code_loc(X, 'CODE_M46A',
                       'W = [1.0 / 1.02 ** (y - 2025) for y in YEARS]')},
        'manuscript_note': '$AE$ and EFC/day are the present-value-weighted available-energy fraction and equivalent full cycles '
                           'per day',
    }
    defs['manuscript_note_found'] = bool(Located(lines_now).find(defs['manuscript_note']))
    same_code = all(defs[k]['ok'] for k in ('W157_builder', 'W157_source', 'W142X_compute_AE', 'W142X_compute_EFC',
                                            'W155_compute', 'W155_reader', 'pv_formula'))
    ident = {'g070_neutrality_AE': g070['AE'], 'g070_neutrality_EFC': g070['EFC'], 'T8_C2_calfade_AE': t8['C2_calfade']['AE'],
             'T8_C2_calfade_EFC': t8['C2_calfade']['EFC'],
             'equal_exactly': g070['AE'] == t8['C2_calfade']['AE'] and g070['EFC'] == t8['C2_calfade']['EFC']}
    printed_ae, printed_efc = f'{ae:.3f}', f'{efc:.2f}'
    return {
        'record': {'path': X.inputs['W155_SOH050']['path'], 'sha256': X.inputs['W155_SOH050']['sha256'],
                   'last_commit': X.inputs['W155_SOH050']['last_commit'], 'fields': ['reports.e_soh050.AE', 'reports.e_soh050.EFC'],
                   'AE': rep['AE'], 'EFC': rep['EFC'], 'cell': rep['cell'], 'eval_key': rep['eval_key'], 'status': rep['status'],
                   'k_star': rep['k_star'], 'minimum_soh': rep['minimum_soh'], 'arm': rep['arm'],
                   'candidate': rep['candidate_canonical'], 'floor_year': rep['floor_year'], 'ageing_series': rep['ageing_series']},
        'primary_record': {'path': X.inputs['W155_RECORD']['path'], 'sha256': X.inputs['W155_RECORD']['sha256'],
                           'series_ok': ok, 'soh_used': series['soh_used'], 'efc_per_day': series['efc_per_day'],
                           'block_years': series['block_years'], 'AE_recomputed': ae, 'EFC_recomputed': efc,
                           'equal_to_summary': ae == rep['AE'] and efc == rep['EFC']},
        'value': {'I_plus_value_minus_I_eur': X.A64['E:soh050:value_minus_I']['I_other'] + X.A64['E:soh050:value_minus_I']['d_gross'],
                  'value_minus_I_eur': X.A64['E:soh050:value_minus_I']['d_gross'], 'I_eur': X.A64['E:soh050:value_minus_I']['I_other']},
        'definition': defs, 'definition_same_code_path': same_code, 'numerical_identity_g070_vs_T8_C2_calfade': ident,
        'at_T8_rounding': {'AE_3dp': printed_ae, 'EFC_per_day_2dp': printed_efc},
        'T8_printed_other_rows': t8_printed,
        'main_tex_row': {'line': row, 'cells': cells, 'columns': cols,
                         'AE_printed': cells[cols.index('$AE$')], 'EFC_printed': cells[cols.index('EFC/day')],
                         'value_printed': cells[cols.index('Value (k\\euro)')],
                         'AE_matches': cells[cols.index('$AE$')] == printed_ae,
                         'EFC_matches': cells[cols.index('EFC/day')] == printed_efc},
    }


def ladder_loses(X):
    rows = []
    for c in VALUE_LADDER_CLAIMS:
        cl = X.CL[c]
        unc = cl.get('uncertified_form_beside_A64_ruling4') or {}
        rows.append({'claim': c, 'other_cell': cl['other_cell'], 'd_gross_F_minus_F0': cl['d_gross'],
                     'gross_verdict': cl['gross_verdict'], 'gross_multiple': cl['gross_multiple'],
                     'uncertified_verdict': unc.get('verdict'), 'uncertified_multiple': unc.get('multiple_Q'),
                     'is_reference_unit': c == REF4})
    others = [r for r in rows if not r['is_reference_unit']]
    return {'rows': rows, 'n_others': len(others),
            'all_others_lose_determinately': all(r['d_gross_F_minus_F0'] > 0 and r['gross_verdict'] == 'determinate' and
                                                 (r['uncertified_verdict'] in (None, 'determinate')) for r in others)}


# ======================================================================================================================
#  7. guards, post-run, summary
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': {'w178b': dict(PICKLE_COUNTS), 'w177b1r': dict(B1R.PICKLE_COUNTS)}, 'pickle_load_and_loads_blocked': blocked}
    pk['ok'] = blocked and all(v == {'load': 0, 'loads': 0} for v in pk['counts'].values())
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('man', 'json', 'md', 'typing_json', 'typing_log')]
    rels += [os.path.join(out_dir, f) for f in sorted(os.listdir(os.path.join(REPO, out_dir))) if LAUNCH_LOG_RX.fullmatch(f)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W178b post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W178b post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def other_copies():
    r = subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    out = []
    for ln in r:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(a.endswith(SCRIPT_REL) for a in argv[1:3]) \
                and '--post-run' not in argv:
            out.append(ln.strip())
    return out


def _cell(s, n=200):
    s = str(s).replace('|', '\\|').replace('\n', ' ')
    return s if len(s) <= n else s[:n - 1] + '…'


def md_summary(o):
    L = [f"# W178b number check at Overleaf {o['clone']['head'][:7]} (zero-solve)", '',
         f"main.tex sha256 {o['clone']['tex_sha256']['main.tex'][:8]}; exit {o['exit_code']}; failed checks {o['failed']}", '',
         '## (b) fixes applied', '', '| item | where | outcome | lines (9fd19e7) |', '|---|---|---|---|']
    for r in o['items']:
        L.append(f"| {r['id']} | {_cell(r['where'], 60)} | {r['outcome']} | {_cell(r['lines_now'], 80)} |")
    L += ['', '## Re-scored rows', '', '| row | e10c104 | status | now | consistent |', '|---|---|---|---|---|']
    for r in o['rows']:
        L.append(f"| {r['id']} | {r['rating_e10']} | {r['status']} | {r['rating_now']} | {r['consistent_with_text']} |")
    s = o['summary']
    L += ['', '## Number check (addressed passages)', '', f"tokens {s['n_tokens']}: {s['by_status']}", '',
          'MISMATCH: ' + (', '.join(f"l. {t['line']} {t['written']}" for t in o['tokens'] if t['status'] == 'MISMATCH') or 'none'),
          '', '## (c) 0.50-floor row', '', f"AE {o['part_c']['at_T8_rounding']['AE_3dp']}, EFC/day "
          f"{o['part_c']['at_T8_rounding']['EFC_per_day_2dp']}; main.tex prints AE "
          f"'{o['part_c']['main_tex_row']['AE_printed']}', EFC/day '{o['part_c']['main_tex_row']['EFC_printed']}'", '']
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  8. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--overleaf-commit', required=True)
    ap.add_argument('--expect', action='append', default=[], help='file.tex=<sha256> for every *.tex of the clone')
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    ap.add_argument('--launch-log', default=OUT_NAMES['log'], help='the name of THIS launch\'s log in the output directory '
                                                                   '(launch.log, or launch_attempt<N>.log after a refusal)')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or OUT_DIR
    tag = 'W178b'
    if not LAUNCH_LOG_RX.fullmatch(args.launch_log):
        _log(f'[{tag} PRECONDITION FAILED] --launch-log {args.launch_log!r} is not launch.log / launch_attempt<N>.log')
        sys.exit(1)
    if args.overleaf_commit != DECLARED_COMMIT:
        _log(f'[{tag} PRECONDITION FAILED] --overleaf-commit {args.overleaf_commit} != declared {DECLARED_COMMIT}')
        sys.exit(1)
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    pre = []
    others = other_copies()
    if others:
        pre.append(f'another copy of {SCRIPT_REL} is running: {others}')
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    refused_logs = []
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        present = sorted(os.listdir(od))
        extra = [f for f in present if not LAUNCH_LOG_RX.fullmatch(f)]
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
        refused_logs = [f for f in present if LAUNCH_LOG_RX.fullmatch(f) and f != args.launch_log]
    # ---- clone ----------------------------------------------------------------------------------------------------------
    mabs = os.path.join(REPO, CLONE)
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex', BIB], capture_output=True,
                               text=True).stdout.strip()
    if head != CLONE_HEAD:
        pre.append(f'clone HEAD {head} != declared {CLONE_HEAD}')
    expect = dict(e.split('=', 1) for e in args.expect if '=' in e)
    tex_now = sorted(f for f in os.listdir(mabs) if f.endswith('.tex'))
    tex_sha = {f: _sha(os.path.join(CLONE, f)) for f in tex_now}
    if sorted(expect) != tex_now or sorted(TEX_SHA) != tex_now:
        pre.append(f'--expect files {sorted(expect)} / declared {sorted(TEX_SHA)} != the clone *.tex {tex_now}')
    for f in tex_now:
        if expect.get(f) != tex_sha[f] or TEX_SHA.get(f) != tex_sha[f]:
            pre.append(f'{f}: sha256 {tex_sha[f]} != --expect {expect.get(f)} / declared {TEX_SHA.get(f)}')
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
    all_inputs.update(B1R.EXTRA_INPUTS)
    all_inputs.update(EXTRA_INPUTS)
    inputs = {}
    for key, rel in all_inputs.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'capture path missing: {rel}')
            continue
        tracked_in_clone = rel.startswith(CLONE + os.sep)
        inputs[key] = {'path': rel, 'sha256': _sha(rel),
                       'committed_clean': (True if tracked_in_clone else W160.W157.L132._committed_clean(rel)),
                       'last_commit': None if tracked_in_clone else W160._last_commit(rel),
                       'blob': None if tracked_in_clone else _git('rev-parse', f'HEAD:{rel}')}
        if tracked_in_clone:
            r = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', os.path.relpath(rel, CLONE)],
                               capture_output=True, text=True).stdout.strip()
            inputs[key]['committed_clean'] = r == ''
            inputs[key]['clone_tracked'] = True
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if 'FROZEN_JSON' in inputs and inputs['FROZEN_JSON']['sha256'] != FZ_SHA:
        pre.append(f'frozen JSON sha256 != {FZ_SHA}')
    if inputs.get('FIXED_COPY', {}).get('sha256') != TEX_SHA['main.tex']:
        pre.append('main.tex at the HEAD is not byte-identical to manuscript_review/main_820ee1b_fixed.tex')
    if inputs.get('LETTER_V4', {}).get('sha256') != TEX_SHA[LETTER]:
        pre.append('the letter at the HEAD is not byte-identical to manuscript_review/response_to_reviewers_v4.tex')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] ' + ' | '.join(pre))
        sys.exit(1)
    _log(f'[{tag}] preconditions hold: clone {head[:8]} ({len(tex_now)} *.tex as declared); {len(inputs)} inputs committed '
         f'clean; script committed clean {script_clean}; earlier launch logs kept {refused_logs}')

    # ---- context (as W177b-1') ------------------------------------------------------------------------------------------
    X = W177A.Ctx(inputs)
    for k in B1.EXTRA_JSON + B1R.EXTRA_JSON + EXTRA_JSON:
        X.j[k] = _json(inputs[k]['path'])
    for k in B1R.EXTRA_TEXT + EXTRA_TEXT:
        X.texts[k] = _text(inputs[k]['path'])
    X.texts['BE_CONV'] = X.t['break_even']['objective_convention']
    X.texts['CAPTIONS'] = X.texts['CAPTIONS_MD']
    X.texts['W170_FORMULA'] = X.j['W170']['formula']
    X.def_src = {'BE_CONV': f'{FZ_REL} tables.break_even.objective_convention', 'CAPTIONS': inputs['CAPTIONS_MD']['path'],
                 'W170_FORMULA': f"{inputs['W170']['path']} formula", 'T7_TEX': inputs['T7_TEX']['path']}
    sal, consts, coh2025 = W177A.confirm_salvage(X)
    X.coh2025 = coh2025
    confirm_now = W177A.confirm_checks(X, sal, consts, coh2025)

    main_text = _text(os.path.join(CLONE, 'main.tex'))
    lines = main_text.split('\n')
    X.main_lines = lines

    def show(commit, f='main.tex'):
        r = subprocess.run(['git', '-C', mabs, 'show', f'{commit}:{f}'], capture_output=True, text=True)
        return r.stdout.split('\n') if r.returncode == 0 else []
    e10 = show(PREV_COMMIT)
    l51 = show(PREV51)
    X.lines260 = show('260bd83')
    X.lines_9a697cc = show(FIXES_WRITTEN_AT)
    X.w174tok = {(t['line'], t['col']): t for t in X.j['W174A_JSON']['tokens'] if t['file'].endswith('main.tex')}

    # ---- the W177b-1' declarations at e10c104, rebuilt by its own functions ----------------------------------------------
    M51, _h51 = B1R.line_map(l51, e10)
    relocated, carried, _retired, unmapped, refused = B1R.relocate_b1(M51, e10, X.frame.split('\n'))
    e10_decls = [d for d in relocated if d['file'] == 'main.tex'] + carried + B1R.main_new_declarations() + B1R.sec3_declarations()
    E10 = {}
    e10_dup = []
    for d in e10_decls:
        if d['line'] in E10:
            e10_dup.append(d['line'])
        E10[d['line']] = d

    # ---- addressed lines; tokens ----------------------------------------------------------------------------------------
    addr = sorted({ln for rs in ADDRESSED.values() for a, b in rs for ln in range(a, b + 1)})
    addr_set = set(addr)
    doc = W164.Doc('main.tex', main_text)
    toks = []
    for t in W174.tokenize_v3(doc):
        if t['line'] not in addr_set:
            continue
        t['file'] = 'main.tex'
        t['status'] = None
        t['item'] = [iid for iid, rs in ADDRESSED.items() if any(a <= t['line'] <= b for a, b in rs)]
        if t['excluded']:
            t.update(status='excluded', category=f"structure: {t['excluded']}")
        elif t['type'] in W177A.IDENT_TYPES:
            t.update(status='excluded', category=f"identifier ({t['type']})")
        toks.append(t)
    # lines equal to e10c104 (difflib): relocate the e10c104 declaration unchanged
    M2, hunks = B1R.line_map(e10, lines)
    inv2 = {v: k for k, v in M2.items()}
    new_decls = declarations_9fd()
    new_lines = {d['line'] for d in new_decls}
    reloc = []
    for ln in addr:
        if ln in inv2 and inv2[ln] in E10 and ln not in new_lines:
            d = dict(E10[inv2[ln]])
            d['line'], d['declared_by'] = ln, f"relocated unchanged from W177b-1' (e10c104 l. {inv2[ln]}; line equal)"
            d['specs'] = list(d['specs'])
            reloc.append(d)
    for d in new_decls:
        d['declared_by'] = 'new declaration (W178b)'
    decls = reloc + new_decls
    unresolved = resolve_specs(decls, E10)
    stale, claimed_twice, decl_out, dup_decl = apply_declarations(X, toks, decls, lines)
    unassigned = [t for t in toks if t['status'] is None]
    for t in toks:
        t['part'] = B1R.part_of(dict(t, scope=t['scope']))
        t['since_e10c104'] = (f"unchanged line (e10c104 l. {inv2[t['line']]})" if t['line'] in inv2 else 'new or changed line')

    # ---- (b) items, rows ------------------------------------------------------------------------------------------------
    fixes = parse_fixes(X.texts['FIXES'])
    LOC, LOC_E10 = Located(lines), Located(e10)
    items = check_items(X, LOC, lines, fixes)
    rows = check_rows(X, LOC, LOC_E10, toks)
    lad = ladder_loses(X)
    for r in rows:
        if r.get('ladder_check'):
            r['ladder'] = {'n_others': lad['n_others'], 'all_others_lose_determinately': lad['all_others_lose_determinately']}
            r['consistent_with_text'] = r['consistent_with_text'] and lad['all_others_lose_determinately']
    salv = salvage_section(X, confirm_now, sal)
    for r in rows:
        if r.get('salvage'):
            r['consistent_with_text'] = r['consistent_with_text'] and salv['printed_reproduces_recorded'] and salv['code_ok']
    pc = part_c(X, lines)
    # the located fragment lines must lie in the declared ranges; every addressed range must hold a located fragment
    # a fix text missing inside its item's ranges must not be found elsewhere in main.tex (else the ranges are wrong)
    frag_lines_ok = (all(m['in_ranges_out_of_order'] or not m.get('elsewhere_in_main') for r in items for f in r['fragments']
                         for m in f['missing']) and
                     all(e['found'] or not e['elsewhere_in_main'] for r in items for e in r['extra_new']))
    range_hit = {iid: [any(a <= x <= b for x in next(r for r in items if r['id'] == iid)['lines_now']) for a, b in rs]
                 for iid, rs in ADDRESSED.items()}
    n_verbatim = sum(1 for r in items if r['outcome'] == 'applied verbatim')
    n_changes = sum(1 for r in items if r['outcome'] == 'applied with changes')
    n_declined = sum(1 for r in items if r['outcome'].startswith('declined'))
    n_absent = sum(1 for r in items if r['outcome'].startswith('ABSENT'))
    by_status = dict(collections.Counter(t['status'] for t in toks))
    mism = [t for t in toks if t['status'] == 'MISMATCH']
    prediction = {'item': '(b) all applied (expert, Addendum 77)',
                  'outcome': (('HELD' if n_absent == 0 else 'FAILED') +
                              f': of A1-A20 and B1-B3 (B4 optional), {n_verbatim} verbatim, {n_changes} with changes '
                              f"({', '.join(r['id'] for r in items if r['outcome'] == 'applied with changes')}), "
                              f'{n_absent} absent; B4 declined by the author ({n_declined}); no change alters meaning: '
                              f"{all(not d['alters_meaning'] for r in items for d in r['diffs'])}")}
    checks = {
        'clone_head_is_declared': head == CLONE_HEAD,
        'every_clone_tex_sha_equals_declared_and_expect': all(TEX_SHA.get(f) == tex_sha[f] == expect.get(f) for f in tex_now),
        'every_clone_tex_and_bib_equals_its_HEAD_blob': all(blob_eq.values()),
        'clone_tex_and_bib_clean': porcelain == '',
        'main_tex_line_count': len(lines) - (1 if main_text.endswith('\n') else 0) == MAIN_LINES_N,
        'main_tex_equals_main_820ee1b_fixed': tex_sha['main.tex'] == inputs['FIXED_COPY']['sha256'],
        'letter_equals_v4': tex_sha[LETTER] == inputs['LETTER_V4']['sha256'],
        'previous_heads_readable': bool(e10) and bool(l51) and bool(X.lines260) and bool(X.lines_9a697cc),
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'w177b1r_declarations_rebuilt_without_loss': not unmapped and not refused and not e10_dup,
        'fixes_file_has_every_item': all(i in fixes for i in ITEM_IDS),
        'every_item_outcome_explained': all(r['explained'] for r in items),
        'every_declared_difference_found': all(d['head_found'] for r in items for d in r['diffs']),
        'located_fragments_inside_declared_ranges': frag_lines_ok,
        'every_declared_range_holds_a_located_fragment': all(all(v) for v in range_hit.values()),
        'every_row_consistent_with_its_texts': all(r['consistent_with_text'] for r in rows),
        'every_OLD_SAME_reference_resolved': not unresolved,
        'every_declaration_found_and_evaluated': not stale,
        'no_line_declared_twice': not dup_decl,
        'no_token_claimed_twice': not claimed_twice,
        'every_token_assigned': not unassigned,
        'statuses_in_vocabulary': all(t['status'] in STATUSES for t in toks),
        'every_input_committed_clean': all(v['committed_clean'] for v in inputs.values()),
        'salvage_code_lines_as_cited': salv['code_ok'],
        'production_salvage_form_reproduces_records': salv['w177a_c1_recomputed']['production_reproduces_records'],
        'printed_salvage_form_reproduces_records': salv['printed_reproduces_recorded'],
        'part_c_primary_record_equals_summary': pc['primary_record']['equal_to_summary'],
        'part_c_definition_same_code_path': pc['definition_same_code_path'] and pc['definition']['manuscript_note_found'],
        'part_c_g070_reproduces_T8_C2_calfade': pc['numerical_identity_g070_vs_T8_C2_calfade']['equal_exactly'],
    }
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
    out = {'schema': 'p515_s53_w178b_number_check', 'version': 1,
           'stage': 'P5.15 Addendum 77, W178b -- (b) A1-A20 / B1-B4 applied; W177(b) rows re-scored; number check over the '
                    'addressed passages; (c) the 0.50-floor row, at Overleaf 9fd19e7',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {B1R_SCRIPT: _sha(B1R_SCRIPT), os.path.basename(LX.__file__): _sha(os.path.basename(LX.__file__)),
                                  os.path.basename(M46A.__file__): _sha(os.path.basename(M46A.__file__))}},
           'command_line': sys.argv, 'this_launch_log': args.launch_log,
           'earlier_launch_logs_kept': {f: _sha(os.path.join(out_dir, f)) for f in refused_logs},
           'clone': {'dir': CLONE, 'head': head, 'declared_commit': DECLARED_COMMIT, 'tex_sha256': tex_sha,
                     'bib_sha256': _sha(os.path.join(CLONE, BIB)), 'tex_and_bib_equal_head_blob': blob_eq,
                     'porcelain_tex_and_bib': porcelain, 'edited': False, 'main_tex_lines': MAIN_LINES_N,
                     'previous_commit': PREV_COMMIT},
           'objective_convention': X.fz.get('objective_convention'), 'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA},
           'inputs': inputs, 'prediction': prediction,
           'items': items, 'fixes_parsed': {k: {'header_line': v['header_line'], 'n_fragments': len(v['fragments'])}
                                            for k, v in fixes.items()},
           'rows': rows, 'value_ladder_others': lad, 'salvage': salv, 'part_c': pc,
           'addressed': {k: v for k, v in ADDRESSED.items()}, 'addressed_lines': addr, 'range_hit': range_hit,
           'hunks_e10c104_to_9fd19e7': hunks,
           'summary': {'n_tokens': len(toks), 'by_status': by_status, 'n_mismatch': len(mism),
                       'by_item': {iid: dict(collections.Counter(t['status'] for t in toks if iid in t['item']))
                                   for iid in ADDRESSED},
                       'n_declarations': {'relocated_unchanged': len(reloc), 'new': len(new_decls)},
                       'items': {'verbatim': n_verbatim, 'with_changes': n_changes, 'declined': n_declined, 'absent': n_absent}},
           'conventions': {'column': '1-based', 'line': '1-based (main.tex at 9fd19e7 unless stated)',
                           'statuses': "W177b-1's: match / approximate / derived / MISMATCH / no counterpart / unchecked "
                                       '(category) / excluded; carried specs are evaluated now, not copied'},
           'declarations_applied': decl_out, 'unresolved_references': unresolved, 'stale_declarations': stale,
           'lines_declared_twice': dup_decl, 'claimed_twice': claimed_twice,
           'unassigned': [{'line': t['line'], 'written': t['written']} for t in unassigned],
           'checks': checks, 'failed': failed, 'tokens': toks, 'guards': guards, 'pickle_guard': pk, 'guards_ok': guards_ok,
           'exit_code': code, 'wall_s': time.time() - t0}

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items() if not callable(v)}
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
    for f in tex_now + [BIB]:
        man[os.path.join(CLONE, f)] = _sha(os.path.join(CLONE, f))
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log ------------------------------------------------------------------------------------------------------------
    _log(f"[{tag}] PREDICTION {prediction['item']}: {prediction['outcome']}")
    for r in items:
        _log(f"[{tag}] {r['id']:>3} {r['outcome']:<32} lines {r['lines_now']} "
             f"{'; '.join(d['what'] for d in r['diffs'])}")
    for r in rows:
        _log(f"[{tag}] ROW {r['id']}: {r['rating_e10']} -> {r['status']} ({r['rating_now']}); consistent {r['consistent_with_text']}")
    _log(f"[{tag}] tokens {len(toks)} on {len(addr)} addressed lines: {by_status}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no counterpart') or t.get('derived') or t['line'] in (1493, 1502, 1540, 1686):
            _log(f"[{tag}]   l. {t['line']}:{t['col']} {t['status']}: {t['written']!r} -> {t.get('value_at_written_precision')!r} "
                 f"[{str(t.get('counterpart'))[:220]}]")
    for x in salv['reproduction']:
        _log(f"[{tag}] SALVAGE {x['cell']}: printed {x['printed_form_eur']:.6f} recorded {x['recorded_salvage_eur']:.6f} "
             f"rel {x['printed_rel_diff_vs_recorded']:.2e} (old printed / recorded {x['old_printed_form_over_recorded']:.3f})")
    _log(f"[{tag}] PART C: AE {pc['record']['AE']!r} EFC {pc['record']['EFC']!r} -> {pc['at_T8_rounding']}; main.tex row "
         f"l. {pc['main_tex_row']['line']} {pc['main_tex_row']['cells']}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if stale or claimed_twice or unassigned or dup_decl or unresolved:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; declared twice {dup_decl}; unresolved {unresolved}; '
             f"unassigned {[(t['line'], t['written']) for t in unassigned]}")
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
