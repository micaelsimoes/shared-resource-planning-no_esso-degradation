"""P5.15 Step 6, Planner task W174a -- NUMBER CHECK OF THE MANUSCRIPT .tex FILES AT OVERLEAF 260bd83 (the W164 / W171a
checker, DECLARATIONS VERSION 3) AND THE VALUE TABLE OF THE NEW SECTIONS 3.5-3.6. ZERO SOLVES, NO MODEL LOADS. A NEW FILE:
the W171a script is IMPORTED (it imports W164 -> W163 -> W162 -> W161 -> W160, each arming its guards) and its records,
quotation check and version-2 declarations are USED; W160-W171 are not edited. Every output is a new file opened 'x' in
a new directory.

WHAT CHANGES AGAINST W171a (version 2 -> version 3)
  1. The manuscript is the Overleaf clone at 260bd83: rounds 2 and 3 of the corrections are in (STEP6_ROUND2_CORRECTIONS.md,
     STEP6_ROUND3_CORRECTIONS.md; PLANNER_BRIEF_2026-09-13.md Addenda 68-70): section 2 revised again, Appendix A
     rewritten, section 3.1 (cost table 2025/2030/2035 and the I(x) paragraph), 3.3 (TN generation priced at the market),
     3.4 (the red floor / calendar paragraphs deleted; the branch-table caption) and the NEW sections 3.5 "Shared Energy
     Storage Parameters" (sec:case_ess_params) and 3.6 "Evaluation and Certification Settings" (sec:case_settings); the
     letter's R1.2, R3.5, R3.6 and "Further changes" revised.
  2. Every version-2 declaration (the set W171a applied, reconstructed and checked against its committed results) is
     RE-LOCATED in the 260bd83 texts and reported as carried (fragment found once, evaluated unchanged), superseded
     (fragment found, re-pointed by a version-3 declaration, reason recorded) or removed (fragment gone; the version-3
     declaration that replaced it, or "sentence deleted, no number left", is named).
  3. New declarations for every number of rounds 2-3 in section 2, Appendix A, sections 3.1 / 3.3 / 3.4 and the letter.
     Revised passages (section 2, section 3.1, the section 3.3 sentence, sections 3.5-3.6, Appendix A) are never handed
     to the automatic value index: every number there is declared or assigned by a stated rule (notation in equations /
     algorithms, structural number words, cross-references, instruction comments), otherwise the run fails.
  4. SECTIONS 3.5-3.6 (main.tex l. 1066-1150), ONE BY ONE: every value and every named setting / option printed there is
     a row of the value table -- line; text as printed; value in force; source (code file:line with the file's sha256,
     spec field with the spec's sha256, case-file field with sha256 and git blob, frozen-table path, or committed record
     path with sha256); status match / MISMATCH / approximate (the printed figure is the source rounded at the printed
     precision; the precision and the full value are stated) / no source found (where it was looked for). The rows of
     the value table are declarations: each declared token takes its row's status. The `% [CONFIRM -- W175]` comments
     are W175's scope; only the printed numbers are checked here.
  5. Number words thirteen..nineteen and thirty..ninety are tokenized in addition to W164's list (section 2.1 now
     writes "fourteen", "seventeen", "thirteen"; the W164 tokenizer does not see them).
  6. The main.tex line rules are RE-PINNED to main.tex sha256 cb237b2d...: the W171a submitted-version regions are mapped
     line by line (difflib equal blocks) from the W171a pin (42794d4, sha256 7effd898, read with `git show` in the
     clone) and kept only where every line is unchanged; the region of the submitted cost table is dropped (replaced by
     the revised section 3.1 table, declared). Figures of sections 4-5 and Appendix E keep the submitted-version status.
  7. Reviewer quotations are re-checked against manuscript_review/Reviewers Comments.docx (W171a's check, unchanged).
  The refusal behaviours stay: declared Overleaf commit and the sha256 of every *.tex must match; the output directory
  must be new (only the launch log); every token assigned; a stale declaration or a token claimed twice exits 3.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W171a chain arms; `pickle.load` / `pickle.loads` blocked for the whole run, every counter
verified at 0. git reads are read-only (`git show` / `git log` / `git status` / `git rev-parse`, also in the clone).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w174_manuscript_check && \\
    mkdir data/SRP1/Results/P515S53/w174_manuscript_check/overleaf_260bd83 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w174_manuscript_number_check.py \\
        --overleaf-commit 260bd83 --expect main.tex=<sha256> ... (every *.tex of the clone) \\
        > data/SRP1/Results/P515S53/w174_manuscript_check/overleaf_260bd83/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w174_manuscript_check/overleaf_260bd83/w174_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w174_manuscript_check/overleaf_260bd83/w174_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w174_manuscript_number_check.py \\
        --overleaf-commit 260bd83 --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES are FINDINGS and never change the
exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault (nothing written).
"""
import argparse
import collections
import difflib
import glob
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W174a manuscript number check v3 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W174: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W174: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w171_manuscript_number_check as W171  # noqa: E402 -- arms its guards, imports W164 -> W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W164, W163, W162, W161, W160 = W171.W164, W171.W163, W171.W162, W171.W161, W171.W160
GUARDS = W160.W157._dedupe((('w174_manuscript_number_check', GUARD),) + tuple(W171.GUARDS))

_log, _sha, _sha_bytes, _git, _jl, _text = W160._log, W160._sha, W160._sha_bytes, W160._git, W162._jl, W162._text
R, num, count_eq, unch, ntc, D, w163ref = W164.R, W164.num, W164.count_eq, W164.unch, W164.ntc, W164.D, W164.w163ref
at_prec, _dp = W164.at_prec, W160._dp
Src = W171.Src

SCRIPT_REL = os.path.basename(__file__)
W171_SCRIPT, W171_SCRIPT_COMMIT = 'p515_s53_w171_manuscript_number_check.py', 'a379b481'
W171_RESULTS = os.path.join(W160.S53, 'w171_manuscript_check', 'overleaf_42794d4', 'w171_manuscript_number_check.json')
W171_RESULTS_COMMIT = '5cb84658'
W171_COMMIT_IN_CLONE = '42794d4'
OUT_ROOT = os.path.join(W160.S53, 'w174_manuscript_check')
DEFAULT_MANUSCRIPT_DIR = W164.DEFAULT_MANUSCRIPT_DIR
DECLARATIONS_VERSION = 3

LETTER, HIGHLIGHTS, COVER, MAIN, DRAFT = W164.LETTER, W164.HIGHLIGHTS, W164.COVER, W164.MAIN, W171.DRAFT
S53 = W160.S53
STEP_DIRS = {'S47': os.path.join('data', 'SRP1', 'Results', 'P515S47'),
             'S51': os.path.join('data', 'SRP1', 'Results', 'P515S51')}

# inputs added to W171a's (each committed clean; sha256 and git blob recorded)
EXTRA_INPUTS = {
    'W171_RESULTS': W171_RESULTS,
    'W171_SCRIPT': W171_SCRIPT,
    'W173_JSON': os.path.join(S53, 'w173_search_counts', 'w173_search_counts.json'),
    'W169_JSON': os.path.join(S53, 'w169_slack_inventory', 'w169_slack_inventory.json'),
    'W159_JSON': os.path.join(S53, 'w159_closing_reads', 'w159_closing_reads.json'),
    'W167_Q4': os.path.join(S53, 'w167_nomenclature_years', 'question4_cost_file.json'),
    'W167_TAB': os.path.join(S53, 'w167_nomenclature_years', 'year_tables', 'tab_investment_cost.tex'),
    'SPEC_V18': os.path.join('data', 'SRP1', 'Results', 'P515S48', 'frozen_s48_spec_v18_8bda2a0a.json'),
    'A64_SPEC': os.path.join(S53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json'),
    'S47_SPEC': os.path.join(STEP_DIRS['S47'], 'campaign_s47_phase_b', 'campaign_spec_s47_phase_b_8cfa264e.json'),
    'S47_RESULTS': os.path.join(STEP_DIRS['S47'], 'campaign_s47_phase_b', 'campaign_results.json'),
    'S51_SPEC': os.path.join(STEP_DIRS['S51'], 'campaign_s51_f2_phase_b', 'campaign_spec_s51_f2_phase_b_5ce295e1.json'),
    'S51_RESULTS': os.path.join(STEP_DIRS['S51'], 'campaign_s51_f2_phase_b', 'campaign_results.json'),
    'S53_SPEC': os.path.join(S53, 'campaign_s53_f2_certificate_r1', 'campaign_spec_s53_f2_certificate_r1_803571c0.json'),
    'S53_RESULTS': os.path.join(S53, 'campaign_s53_f2_certificate_r1', 'campaign_results.json'),
    'ROUND2_MD': 'STEP6_ROUND2_CORRECTIONS.md',
    'ROUND3_MD': 'STEP6_ROUND3_CORRECTIONS.md',
}
JSON_KEYS = ('W171_RESULTS', 'W173_JSON', 'W169_JSON', 'W159_JSON', 'W167_Q4', 'SPEC_V18', 'A64_SPEC', 'S47_SPEC',
             'S47_RESULTS', 'S51_SPEC', 'S51_RESULTS', 'S53_SPEC', 'S53_RESULTS')
CODE_V3 = {'p515_s53_f2_certificate': 'p515_s53_f2_certificate.py', 'p515_s44_campaign_harness': 'p515_s44_campaign_harness.py',
           'solver_parameters': 'solver_parameters.py', 'settling_criterion_v6': 'settling_criterion_v6.py'}
DN_NAMES = ('case33_1', 'case33_2', 'case33_3')
RATING_YEARS = ('2025', '2028', '2030', '2031', '2034', '2035', '2037')   # every year either instance loads
DN_YEAR_FILES = {f'DNY_{n}_{y}': os.path.join('data', 'SRP1', n, f'{n}_{y}.json') for n in DN_NAMES for y in RATING_YEARS}

OUT_NAMES = {'json': 'w174_manuscript_number_check.json', 'md': 'w174_manuscript_number_check.md',
             'man': 'manifest_sha256.json', 'log': 'launch.log', 'typing_json': 'w174_bool_typing_test.json',
             'typing_log': 'w174_bool_typing_test.log', 'post': 'manifest_post_run_sha256.json'}

STATUSES = W171.STATUSES + ('no source found',)
ROW_STATUSES = ('match', 'MISMATCH', 'approximate', 'no source found')
MANUSCRIPT_FILES = (COVER, HIGHLIGHTS, MAIN, LETTER)

# ---- main.tex line rules, re-pinned to this main.tex (Overleaf 260bd83) -------------------------------------------------
MAIN_LINE_RULES_SHA = 'cb237b2d129e689a94dd50abec9a9cca739567ef97a486ec012a901f0e04b6df'
# revised passages: every number declared or assigned by a stated rule (never the auto index); checked against headings
REVISED = (
    ('sec2', (315, 909), 'section 2 (revised, rounds 1-3)', r'\section{Shared ESS Planning Framework}'),
    ('sec3_1', (947, 975), 'section 3.1 Investment Costs (revised table and paragraph, round 3 C.1)', r'\subsection{Investment Costs}'),
    ('sec3_3', (983, 985), 'section 3.3 sentence on TN generation (round 3 C.2)', r'\textcolor{blue}{'),
    ('sec3_5_6', (1066, 1151), 'sections 3.5-3.6 (new, round 3 C.4-C.5)', r'\subsection{\textcolor{blue}{Shared Energy Storage Parameters}}'),
    ('app_a', (1493, 1719), 'Appendix A (rewritten, rounds 2-3)', r'\section{\textcolor{blue}{TSO--DSO Coordinated Operational Planning}}'),
)
VALUE_TABLE_LINES = (1066, 1150)          # the task's l. 1066-1150 (section 3.5 heading to the line before RESULTS)
NETWORK_TABLES = (980, 1065)              # sections 3.3-3.4 data tables (W171a 965-1047 mapped; ends before 3.5)
APP_BCD = (1720, 2080)                    # Appendices B-D (W171a 1603-1963 mapped)
REGION_DROPPED = {'cost': 'the submitted cost table (W171a l. 940-957) is replaced by the revised section 3.1 table '
                          '(W167 fragment, round 3 C.1): its numbers are declared (C301-C308), not submitted-version'}
# the expected re-pinned regions (the run recomputes them from the W171a pin and refuses on a difference)
EXPECTED_MAIN_REGIONS = ((1215, 1227), (1228, 1365), (1366, 1454), (1455, 1474), (1152, 1214), (2081, 2194))
# r47, r_delete (two), keyins, results (sections 4-5), appE: W171a's regions mapped unchanged; 'cost' dropped

EXTRA_WORDS = {'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
               'nineteen': 19, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
               'ninety': 90}
EXTRA_RX = re.compile(r'\b(?:' + '|'.join(sorted(EXTRA_WORDS, key=len, reverse=True)) + r')\b', re.I)


# ======================================================================================================================
#  1. tokenizer extension, values
# ======================================================================================================================
def tokenize_v3(doc):
    """W164.tokenize + the number words thirteen..nineteen / thirty..ninety (type 'word'), same fields."""
    toks = W164.tokenize(doc)
    spans = collections.defaultdict(list)
    for t in toks:
        spans[(t['line'], t['scope'])].append((t['col0'], t['end0']))
    extra = []
    for i, ln in enumerate(doc.lines):
        body, com, j = W164.split_comment(ln)
        for scope, seg, offs in (('body', body, 0), ('comment', com, j + 1)):
            if seg is None:
                continue
            masked = W164.CMD_RX.sub(lambda m: ' ' * len(m.group()), seg)
            ex = [(m.start(), m.end(), c) for c, rx in W164.EXCL_RE for m in rx.finditer(seg)]
            occ = collections.Counter()
            for m in EXTRA_RX.finditer(masked):
                a, b = m.start() + offs, m.end() + offs
                if any(not (b <= s or a >= e) for s, e in spans[(i + 1, scope)]):
                    continue
                txt = m.group()
                t = {'col0': a, 'end0': b, 'text': txt, 'type': 'word', 'sign': '',
                     'excluded': next((c for s, e, c in ex if s <= m.start() and m.end() <= e), None),
                     'file': doc.name, 'line': i + 1, 'col': a + 1, 'scope': scope, 'occ': occ[txt]}
                occ[txt] += 1
                o = doc.off(i + 1, a)
                t['in_math'] = bool(doc.mathmask[o]) if scope == 'body' and o < len(doc.mathmask) else False
                t['envs'] = doc.envs_at(o) if scope == 'body' else []
                t['section'] = doc.section_at(o)
                t['macro'] = doc.macro_at(o) if scope == 'body' else None
                t['value'] = float(EXTRA_WORDS[txt.lower()])
                t['written'] = txt
                t['unit'] = W164._unit_after(ln, b)
                t['key'] = f"{doc.name}:{i + 1}:{a + 1}"
                t['tokenizer'] = 'v3 extra number word'
                extra.append(t)
    out = toks + extra
    order = {'body': 0, 'comment': 1}
    out.sort(key=lambda t: (t['line'], order[t['scope']], t['col0']))
    return out


def wv3(w):
    lw = str(w).lower()
    if lw in EXTRA_WORDS:
        return float(EXTRA_WORDS[lw])
    return W164.wv(w)


def _close(a, b, rel=1e-12):
    return a == b or abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


def count_eq3(written, n, src, kind='named record', note=None):
    return R('match' if wv3(written) == float(n) else 'MISMATCH', src, kind, n, str(n), note)


def cmp_exact(written, value, scale=1.0):
    x = value * scale
    return ('match' if _close(wv3(written), x) else 'MISMATCH'), None


def cmp_round(written, value, scale=1.0):
    """exact -> match; equal at the printed precision -> approximate (precision stated); else MISMATCH."""
    x = value * scale
    if _close(wv3(written), x):
        return 'match', None
    shown, st = at_prec(str(written), x)
    if st == 'match':
        return 'approximate', f'printed at {_dp(str(written))} decimal place(s): {written} = {x!r} rounded'
    return 'MISMATCH', f'{x!r} at the printed precision is {shown}'


# ======================================================================================================================
#  2. records
# ======================================================================================================================
class RecordsV3(W171.RecordsV2):
    def __init__(self, inputs, man_doc, sub_doc):
        super().__init__(inputs, man_doc, sub_doc)
        self.code.update({k: Src(v) for k, v in CODE_V3.items()})
        self.j = {k: _jl(inputs[k]['path']) for k in JSON_KEYS}
        self.dny = {k: _jl(inputs[k]['path']) for k in DN_YEAR_FILES}
        self.value_rows = {}
        self.ptab = None


def jf(X, key, field):
    i = X.inputs[key]
    return f"{i['path']} {field} (sha256 {i['sha256'][:8]}, blob {(i.get('blob') or '')[:8]})"


def row(tok, quantity, printed, value, sources, status, precision=None, note=None):
    if status not in ROW_STATUSES:
        raise ValueError(f'row status {status!r}')
    return {'tok': list(tok), 'quantity': quantity, 'printed': printed, 'value_in_force': value,
            'sources': list(sources), 'status': status, 'precision': precision, 'note': note}


def V(vid, fragment, tokens, fn):
    """a section 3.5-3.6 declaration: fn(X, written) -> rows (row()); each declared token takes its row's status."""
    def wrapped(X, w):
        rows = fn(X, w)
        X.value_rows[vid] = rows
        out = [None] * len(tokens)
        for r in rows:
            for k in r['tok']:
                if out[k] is not None:
                    raise KeyError(f'{vid}: token {k} in two rows')
                out[k] = R(r['status'], '; '.join(r['sources']), 'value in force (section 3.5-3.6 value table)',
                           r['value_in_force'], None, ' '.join(x for x in (r['precision'], r['note']) if x) or None)
        if any(o is None for o in out):
            raise KeyError(f'{vid}: declared token without a row')
        return out
    d = D(vid, MAIN, fragment, tokens, wrapped)
    d['value_table'] = True
    return d


def P(X, pid):
    return X.ptab[pid]


def prow(X, pid, tok, quantity, printed, written, value, scale=1.0, rounding=False, note=None):
    """a row from the W171a parameter check `pid` (re-evaluated on this run's code): printed vs value; the parameter
    check's own status (which also covers e.g. 'never overwritten') must be match."""
    p = P(X, pid)
    st, prec = (cmp_round if rounding else cmp_exact)(written, value, scale)
    if p['status'] != 'match' and st != 'MISMATCH':
        st = 'MISMATCH'
        note = f"W171a parameter check {pid} status {p['status']}: {p.get('note')}. " + (note or '')
    return row(tok, quantity, printed, value, p['sources'], st, prec, ' '.join(x for x in (note, p.get('note')) if x) or None)


# ---- named sources --------------------------------------------------------------------------------------------------
def _search_specs(X):
    return {k: X.j[f'{k}_SPEC']['extra']['poll_design'] for k in ('S47', 'S51', 'S53')}


def _w173(X):
    return X.j['W173_JSON']


def _ageing_arms(X):
    return X.ext3['model_variant']['arms']


def _k_closed(n, d, r):
    return n * d / (-math.log(r))


def _dn_rating(X, key):
    """network.Network.get_interface_branch_rating (network.py:83-94) replicated on a case JSON: the sum of the ratings
    of the in-service branches incident to the reference node (type 3), isolated ends (type 4) excluded."""
    d = X.dny[key]
    nodes = {int(n['bus_i']): int(n['type']) for n in d['nodes']}
    refs = [b for b, t in nodes.items() if t == 3]
    if len(refs) != 1:
        raise KeyError(f'{key}: reference nodes {refs}')
    ref = refs[0]
    br = []
    for kind in ('lines', 'transformers'):
        for b in d.get(kind, []):
            if not bool(b['status']):
                continue
            if nodes.get(int(b['fbus'])) == 4 or nodes.get(int(b['tbus'])) == 4:
                continue
            if int(b['fbus']) == ref or int(b['tbus']) == ref:
                br.append((kind, b['branch_id'], float(b['rating'])))
    return {'ref': ref, 'branches': br, 'rating_mva': sum(x[2] for x in br), 'baseMVA': float(d['baseMVA'])}


def interface_ratings(X):
    node_of = {dn['name']: dn['connection_node_id'] for dn in X.srp1['DistributionNetworks']}
    node_of3 = {dn['name']: dn['connection_node_id'] for dn in X.i3x3['DistributionNetworks']}
    out = {}
    for n in DN_NAMES:
        per = {y: _dn_rating(X, f'DNY_{n}_{y}') for y in RATING_YEARS}
        vals = sorted({(v['rating_mva'], v['baseMVA']) for v in per.values()})
        ids = sorted({tuple((k, bid) for k, bid, _ in v['branches']) for v in per.values()})
        out[n] = {'node': node_of[n], 'node_3x3': node_of3.get(n), 'values': vals, 'branches': ids,
                  'rating_mva': vals[0][0] if len(vals) == 1 else None, 'baseMVA': vals[0][1] if len(vals) == 1 else None}
    return out


def _ri_sources(X):
    net = X.c('network')
    return [net.ref(r'def get_interface_branch_rating\(self\):'), net.ref(r'interface_branch_rating \+= branch\.rate'),
            X.c('shared_resources_planning').ref(
                r'interface_transf_rating = distribution_network\.network\[year\]\[day\]\.get_interface_branch_rating\(\) / s_base', expect=None),
            'case files ' + ', '.join(f"{X.inputs[k]['path']} (sha256 {X.inputs[k]['sha256'][:8]}, blob {X.inputs[k]['blob'][:8]})"
                                      for k in sorted(DN_YEAR_FILES) if k.endswith(('_2025', '_2030', '_2035'))) +
            f' (and the 3x3 years {", ".join(y for y in RATING_YEARS if y not in ("2025", "2030", "2035"))}, same values) '
            'transformers / lines incident to the reference node: rating (MVA), baseMVA',
            f'{W171.jsrc(X, "SRP1_JSON")} DistributionNetworks[*].connection_node_id']


# ======================================================================================================================
#  3. the section 3.5-3.6 value table (main.tex l. 1066-1150)
# ======================================================================================================================
def value_declarations():
    out = []
    LFP_SEARCH = ('git-tracked *.py and every data JSON for "LFP" / "iron phosphate" (git grep): one hit file, '
                  'data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json citations; no model or case-file '
                  'field carries a chemistry')

    def v01(X, w):
        cit = X.j['SPEC_V18']['citations']
        srcs = [jf(X, 'SPEC_V18', 'citations.cycle_life[*].source') + f": {[c['source'] for c in cit['cycle_life']]}",
                jf(X, 'SPEC_V18', 'citations.status') + f": {cit['status']!r}"]
        lfp = all('LFP' in c['source'] for c in cit['cycle_life'])
        return [row([], 'chemistry of the shared ESS', 'utility-scale lithium iron phosphate battery',
                    'no model parameter; the recorded datasheets of the cycle-life calibrations are LFP' if lfp else None,
                    srcs, 'no source found', None,
                    f'no configuration field states a chemistry (searched {LFP_SEARCH}); the only record is the '
                    'citations block of spec v18, status "PROPOSED BY THE EXPERT; AUTHOR TO CONFIRM AND VERIFY", whose two '
                    'datasheets are LFP. A named setting, not a value; W175 CONFIRM point ("LFP named as the chemistry '
                    'in the ESS file")')]
    out.append(V('V01', 'The shared ESS is a utility-scale lithium iron phosphate battery', [], v01))
    out.append(V('V02', '$\\eta^{\\text{Ch}} = 0.97$ and $\\eta^{\\text{Dch}} = 0.96$', [('0.97', 0), ('0.96', 0)],
                 lambda X, w: [prow(X, 'P01', [0], 'eta^Ch (shared ESS, network models and agent)', '$\\eta^{Ch} = 0.97$', w[0], P(X, 'P01')['found']),
                               prow(X, 'P02', [1], 'eta^Dch', '$\\eta^{Dch} = 0.96$', w[1], P(X, 'P02')['found'])]))
    out.append(V('V03', 'a usable state-of-charge window of 10--90\\,\\% of the available energy ($SoC^{\\text{Min}} = 0.10$, '
                        '$SoC^{\\text{Max}} = 0.90$)', [('10', 0), ('90', 0), ('0.10', 0), ('0.90', 0)],
                 lambda X, w: [prow(X, 'P03', [0], 'SoC^Min (percent of E^Av)', '10 %', w[0], P(X, 'P03')['found'], scale=100.0),
                               prow(X, 'P04', [1], 'SoC^Max (percent of E^Av)', '90 %', w[1], P(X, 'P04')['found'], scale=100.0),
                               prow(X, 'P03', [2], 'SoC^Min (fraction of E^Av)', '$SoC^{Min} = 0.10$', w[2], P(X, 'P03')['found']),
                               prow(X, 'P04', [3], 'SoC^Max (fraction of E^Av)', '$SoC^{Max} = 0.90$', w[3], P(X, 'P04')['found'])]))
    out.append(V('V04', 'a daily initial state of 50\\,\\% ($SoC^{0} = 0.50$)', [('50', 0), ('0.50', 0)],
                 lambda X, w: [prow(X, 'P05', [0], 'SoC^0 (percent; initial = closure target)', '50 %', w[0], P(X, 'P05')['found'], scale=100.0),
                               prow(X, 'P05', [1], 'SoC^0 (fraction)', '$SoC^{0} = 0.50$', w[1], P(X, 'P05')['found'])]))
    out.append(V('V05', 'the daily closure slack is bounded by $\\varepsilon^{\\text{Cl}} = 0.05$ of the available energy (plus '
                        'a numerical allowance of $10^{-5}$~p.u.) and priced at $c^{\\text{Cl}} = 10^{3}$~\\euro/MWh',
                 [('0.05', 0), ('10^{-5}', 0), ('10^{3}', 0)],
                 lambda X, w: [prow(X, 'P06', [0], 'closure slack bound eps^Cl (fraction of E^Av)', '$\\varepsilon^{Cl} = 0.05$', w[0], P(X, 'P06')['found']),
                               prow(X, 'P07', [1], 'closure slack numerical allowance (per unit at baseMVA 100)', '$10^{-5}$ p.u.', w[1], P(X, 'P07')['found']),
                               prow(X, 'P08', [2], 'closure slack price c^Cl (EUR/MWh)', '$c^{Cl} = 10^{3}$ EUR/MWh', w[2], P(X, 'P08')['found'])]))
    out.append(V('V06', 'the network complementarity tolerance is $\\varepsilon^{\\text{C}} = 10^{-4}$', [('10^{-4}', 0)],
                 lambda X, w: [prow(X, 'P09', [0], 'network complementarity tolerance eps^C (normalised)', '$\\varepsilon^{C} = 10^{-4}$', w[0], P(X, 'P09')['found'])]))

    def v07(X, w):
        ess = X.ess['ageing']['calendar_life_years']
        cfg = X.ext3['configuration']['ess_ageing_baseline']['calendar_life_years'] if 'configuration' in X.ext3 and \
            'ess_ageing_baseline' in X.ext3['configuration'] else None
        st, _ = cmp_exact(w[0], ess)
        if cfg is not None and cfg != ess:
            st = 'MISMATCH'
        return [row([0], 'calendar lifetime T^Cal (years)', '$T^{Cal} = 15$ years', ess,
                    [W171.jline(X, 'ESS_PARAMS', r'"calendar_life_years": 15') + ' ageing.calendar_life_years',
                     f'{W171.jsrc(X, "V6_SPEC")} inputs_in_force_now.configuration_now.ess_ageing_baseline.calendar_life_years = '
                     f"{X.v6['inputs_in_force_now']['configuration_now']['ess_ageing_baseline']['calendar_life_years']}"],
                    st, None, 'consumed by the salvage credit (calendar_life_basis REMAINING_FRACTION_AT_TERMINAL) and by '
                              'the calendar retention statement below')]
    out.append(V('V07', 'the calendar lifetime is $T^{\\text{Cal}} = 15$~years', [('15', 0)], v07))
    out.append(V('V08', "The agent's regularisation weight is $\\varepsilon^{\\text{E}} = 10^{-5}$ and its slack penalty "
                        "$c^{\\sigma} = 10^{3}$", [('10^{-5}', 0), ('10^{3}', 0)],
                 lambda X, w: [prow(X, 'P10', [0], 'eps^E (ESSO throughput regularisation)', '$\\varepsilon^{E} = 10^{-5}$', w[0], P(X, 'P10')['found']),
                               prow(X, 'P11', [1], 'c^sigma (ESSO P-net slack penalty)', '$c^{\\sigma} = 10^{3}$', w[1], P(X, 'P11')['found'])]))

    def v09(X, w):
        la = X.lattice()
        ls = [f'{W164.A1_CAMPAIGN} LATTICE_P_STEP / LATTICE_E_STEP', f'{W164.STEP4} lines 8-9',
              f'{W171.jsrc(X, "ESS_PARAMS")} min/max_energy_to_power_factor']
        pb = X.c('p515_s47_phase_b_record')
        out_ = []
        for k, (q, pr, v) in enumerate((('lattice unit Delta^S (MVA)', '$\\Delta^S = 0.25$ MVA', la['p_step']),
                                        ('lattice unit Delta^E (MWh)', '$\\Delta^E = 0.5$ MWh', la['e_step']),
                                        ('minimum duration E/P (h)', '2 h', la['ep_min']),
                                        ('maximum duration E/P (h)', '4 h', la['ep_max']))):
            st, _ = cmp_exact(w[k], v)
            out_.append(row([k], q, pr, v, ls + ([pb.ref(r'^GRANULE_P_MVA = 0\.25')] if k == 0 else
                                                 [pb.ref(r'^GRANULE_E_MWH = 0\.5')] if k == 1 else []), st))
        mc = X.ess['max_capacity']
        st, _ = cmp_exact(w[4], mc)
        out_.append(row([4], 'maximum energy capacity per node (MWh)', 'at most 5 MWh per node', mc,
                        [W171.jline(X, 'ESS_PARAMS', r'"max_capacity": 5\.00') + ' max_capacity', pb.ref(r'^E_MAX_MWH = 5\.0')], st))
        b = X.ess['budget']
        st, _ = cmp_exact(w[5], b, 1e-6)
        out_.append(row([5], 'investment budget (M EUR)', 'a budget of 1 M EUR', b,
                        [W171.jline(X, 'ESS_PARAMS', r'"budget": 1\.0e6') + ' budget (EUR)', pb.ref(r'^BUDGET_EUR = 1e6')], st))
        return out_
    out.append(V('V09', 'Capacities are sized on the lattice $\\Delta^S = 0.25$~MVA, $\\Delta^E = 0.5$~MWh with durations '
                        'between 2 and 4~h, at most 5~MWh per node and a budget of 1~M\\euro.',
                 [('0.25', 0), ('0.5', 0), ('2', 0), ('4', 0), ('5', 0), ('1', 0)], v09))

    def v10(X, w):
        cal = X.ess['ageing']['calibration']
        arms = _ageing_arms(X)
        a = arms['C2_calfade']
        ess_src = W171.jsrc(X, 'ESS_PARAMS')
        k_spec = X.ext3['model_variant']['k_closed_form_by_arm']['C2_calfade']
        k_t8 = X.ageing['C2_calfade']['k']
        kc = _k_closed(cal['cycles_n'], cal['reference_dod_d'], cal['eol_retention_r'])
        rows = []
        st, _ = cmp_exact(w[0], cal['cycles_n'])
        rows.append(row([0], 'baseline datasheet cycles N', '10,000 cycles', cal['cycles_n'], [f'{ess_src} ageing.calibration.cycles_n'], st))
        st, _ = cmp_exact(w[1], cal['reference_dod_d'], 100.0)
        rows.append(row([1], 'baseline depth of discharge (percent)', '80 % depth of discharge', cal['reference_dod_d'],
                        [f'{ess_src} ageing.calibration.reference_dod_d'], st))
        st, _ = cmp_exact(w[2], cal['eol_retention_r'], 100.0)
        if a['eol_retention_r'] != cal['eol_retention_r']:
            st = 'MISMATCH'
        rows.append(row([2], 'baseline retention at the cycle count (percent)', 'cycles to 80 % retention', cal['eol_retention_r'],
                        [f'{ess_src} ageing.calibration.eol_retention_r', f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.arms.C2_calfade.eol_retention_r'], st))
        st, prec = cmp_round(w[3], k_spec)
        if not (_close(k_spec, k_t8) and _close(k_spec, kc)):
            st = 'MISMATCH'
        rows.append(row([3], 'baseline k = N D / (-ln R)', '$k = 35,851$', k_spec,
                        [f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.k_closed_form_by_arm.C2_calfade',
                         'T8 tables.ageing.rows[arm=C2_calfade].k', f'closed form from {ess_src} ageing.calibration = {kc!r}'],
                        st, prec))
        phi = X.ess['ageing']['calendar_retention_per_year']
        st, _ = cmp_exact(w[4], phi)
        if a['calendar_retention_per_year'] != phi:
            st = 'MISMATCH'
        rows.append(row([4], 'calendar retention phi^Cal per year', '$\\phi^{Cal} = 0.985$ per year', phi,
                        [f'{ess_src} ageing.calendar_retention_per_year', f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.arms.C2_calfade.calendar_retention_per_year'], st))
        life = X.ess['ageing']['calendar_life_years']
        ret = phi ** life
        st, prec = cmp_round(w[5], ret, 100.0)
        rows.append(row([5], 'retention after the calendar life without cycling (percent) = phi^T', '80 % retention over the 15-year calendar life',
                        ret, [f'{ess_src} ageing.calendar_retention_per_year ^ ageing.calendar_life_years = {phi}^{life}',
                              jf(X, 'SPEC_V18', 'citations.calendar_fade.statement') + ': "phi_cal = 0.985 per year (0.80 at the 15-year calendar life)"'],
                        st, prec))
        st, _ = cmp_exact(w[6], life)
        rows.append(row([6], 'calendar life (years) in the retention statement', '15-year calendar life', life, [f'{ess_src} ageing.calendar_life_years'], st))
        smin = X.ess['ageing']['minimum_soh']
        st, _ = cmp_exact(w[7], smin)
        if X.ext3['model_variant']['floor_row_lower_expected'] != smin:
            st = 'MISMATCH'
        rows.append(row([7], 'end-of-life floor SoH^Min (baseline)', '$SoH^{Min} = 0.70$', smin,
                        [f'{ess_src} ageing.minimum_soh', f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.floor_row_lower_expected'], st))
        return rows
    out.append(V('V10', 'The baseline reads the datasheet count of 10{,}000 cycles at 80\\,\\% depth of discharge as cycles to '
                        '80\\,\\% retention ($k = 35{,}851$) and adds a calendar retention of $\\phi^{\\text{Cal}} = 0.985$ per '
                        'year (80\\,\\% retention over the 15-year calendar life without cycling); the end-of-life floor is '
                        '$SoH^{\\text{Min}} = 0.70$.',
                 [('10{,}000', 0), ('80', 0), ('80', 1), ('35{,}851', 0), ('0.985', 0), ('80', 2), ('15', 0), ('0.70', 0)], v10))

    def v11(X, w):
        arms = _ageing_arms(X)
        src = W171.jsrc(X, 'EXT_SPEC_V3')
        rows = [row([], 'cycling-only sensitivity: the baseline count without calendar fade', 'the same count without calendar fade',
                    {'C2': arms['C2']}, [f'{src} model_variant.arms.C2'],
                    'match' if arms['C2']['calendar_retention_per_year'] == 1.0 and arms['C2']['eol_retention_r'] == 0.8 else 'MISMATCH')]
        ok = arms['C3_unit']['eol_retention_r'] == arms['C3_midblock']['eol_retention_r'] == 0.5
        st, _ = cmp_exact(w[0], 0.5 if ok else -1, 100.0)
        rows.append(row([0], 'C3 retention at the cycle count (percent)', 'the same count read as cycles to 50 % retention',
                        {a: arms[a]['eol_retention_r'] for a in ('C3_unit', 'C3_midblock')},
                        [f'{src} model_variant.arms.C3_unit.eol_retention_r', f'{src} model_variant.arms.C3_midblock.eol_retention_r'], st))
        sp = (arms['C3_unit']['available_energy_soh_point'], arms['C3_midblock']['available_energy_soh_point'])
        rows.append(row([], 'C3 SoH evaluation point', 'evaluated at the end of the block or at its midpoint', sp,
                        [f'{src} model_variant.arms.C3_unit / C3_midblock.available_energy_soh_point'],
                        'match' if sp == ('end', 'mid') else 'MISMATCH', None,
                        'the mid-block formula itself is W175\'s CONFIRM point'))
        return rows
    out.append(V('V11', 'the same count without calendar fade; the same count read as cycles to 50\\,\\% retention, evaluated '
                        'at the end of the block or at its midpoint', [('50', 0)], v11))

    def v12(X, w):
        cit = [c for c in X.j['SPEC_V18']['citations']['cycle_life'] if '8,000' in c['statement']]
        a = _ageing_arms(X)['C4']
        cal = X.ess['ageing']['calibration']
        rows = []
        st = 'match' if len(cit) == 1 and wv3(w[0]) == 8000 else 'MISMATCH'
        rows.append(row([0], 'C4 datasheet cycles (full cycles)', 'a second datasheet of 8,000 full cycles', 8000,
                        [jf(X, 'SPEC_V18', 'citations.cycle_life[EVE MB31]') + f": {cit[0]['statement'] if cit else None!r}, "
                         f"{cit[0]['calibration'] if cit else None!r}"], st, None,
                        f"the run encodes this datasheet as the file's (N, D) = ({cal['cycles_n']}, {cal['reference_dod_d']}) "
                        f"with N x D = {cal['cycles_n'] * cal['reference_dod_d']:g} = 8,000 x 1.0, the only product the model "
                        'consumes (k = N D / -ln R). The datasheet statement is as RECORDED in spec v18 (status "PROPOSED BY '
                        'THE EXPERT; AUTHOR TO CONFIRM AND VERIFY"; the Planner has not accessed the datasheet); the '
                        '[AUTHOR] comment at l. 1099-1100 still asks for the citation'))
        st, _ = cmp_exact(w[1], a['eol_retention_r'], 100.0)
        rows.append(row([1], 'C4 retention at the cycle count (percent)', 'to 70 % retention', a['eol_retention_r'],
                        [f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.arms.C4.eol_retention_r'], st))
        return rows
    out.append(V('V12', 'a second datasheet of 8{,}000 full cycles to 70\\,\\% retention', [('8{,}000', 0), ('70', 0)], v12))

    def v13(X, w):
        arms = _ageing_arms(X)
        rows = [row([], 'no-ageing sensitivity', 'no ageing', {'no_ageing': arms['no_ageing']},
                    [f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.arms.no_ageing.ageing_enabled'],
                    'match' if arms['no_ageing']['ageing_enabled'] is False else 'MISMATCH', None,
                    'implementation of the arm is W175\'s CONFIRM point')]
        c = X.j['A64_SPEC']['cells']['e_soh050']
        smin = c['declaration']['minimum_soh']
        st, _ = cmp_exact(w[0], smin)
        if c['arm'] != 'C2_calfade':
            st = 'MISMATCH'
        rows.append(row([0], 'floor of the 0.50-floor sensitivity', 'the baseline with a 0.50 floor', smin,
                        [jf(X, 'A64_SPEC', 'cells.e_soh050.declaration.minimum_soh') + f' (arm {c["arm"]})'], st))
        return rows
    out.append(V('V13', 'no ageing; and the baseline with a 0.50 floor.', [('0.50', 0)], v13))

    # ---- the ageing calibration table (l. 1077-1097), the seven arms as run ------------------------------------------
    def table_row(vid, label, frag, arm, tokens, datasheet=False, floor=None, k_label=None):
        def fn(X, w):
            arms = _ageing_arms(X)
            cal = X.ess['ageing']['calibration']
            esrc = W171.jsrc(X, 'EXT_SPEC_V3')
            if arm == 'soh050':
                c = X.j['A64_SPEC']['cells']['e_soh050']
                a = dict(c['declaration']['model_variant'])
                src_arm = jf(X, 'A64_SPEC', 'cells.e_soh050.declaration')
                smin, smin_src = c['declaration']['minimum_soh'], jf(X, 'A64_SPEC', 'cells.e_soh050.declaration.minimum_soh')
                k = X.ext3['model_variant']['k_closed_form_by_arm']['C2_calfade']
            else:
                a = arms[arm]
                src_arm = f'{esrc} model_variant.arms.{arm}'
                smin, smin_src = X.ext3['model_variant']['floor_row_lower_expected'], f'{esrc} model_variant.floor_row_lower_expected'
                k = X.ext3['model_variant']['k_closed_form_by_arm'][arm]
            rows = []
            names = [t[0] for t in tokens]
            i = 0
            if label is not None:            # a number inside the row label ("Retention 50 %", "Datasheet 8 000", "0.50 floor")
                q, v, s = label
                v_ = v(X, a, smin)
                st, prec = cmp_exact(w[0], v_[0], v_[1])
                rows.append(row([0], q, f'row label: {w[0]}', v_[0], [s(X, src_arm, smin_src)], st, prec))
                i = 1
            if datasheet:
                cit = [c for c in X.j['SPEC_V18']['citations']['cycle_life'] if '8,000' in c['statement']]
                ok = len(cit) == 1 and '(cycles 8000, DoD 1.0, EOL 0.70)' in cit[0]['calibration']
                srcd = jf(X, 'SPEC_V18', 'citations.cycle_life[EVE MB31].calibration') + f": {cit[0]['calibration'] if cit else None!r}"
                for (vq, vv), wk in zip((('N^DS (datasheet cycles)', 8000.0), ('delta^DS (datasheet DoD)', 1.0)), w[i:i + 2]):
                    st, _ = cmp_exact(wk, vv)
                    rows.append(row([i], vq, wk, vv, [srcd], st if ok else 'MISMATCH', None,
                                    f"as run: the file's (N, D) = ({cal['cycles_n']}, {cal['reference_dod_d']}); N x D equal "
                                    f"({cal['cycles_n'] * cal['reference_dod_d']:g} = 8000 x 1.0), the only product k uses; datasheet "
                                    'statement as recorded in spec v18 (proposed by the expert, author to confirm)'))
                    i += 1
            else:
                st, _ = cmp_exact(w[i], cal['cycles_n'])
                rows.append(row([i], 'N^DS (cycles)', w[i], cal['cycles_n'], [f'{W171.jsrc(X, "ESS_PARAMS")} ageing.calibration.cycles_n '
                                                                            '(the arms vary eol_retention_r only)'], st))
                i += 1
                st, _ = cmp_exact(w[i], cal['reference_dod_d'])
                rows.append(row([i], 'delta^DS (DoD)', w[i], cal['reference_dod_d'], [f'{W171.jsrc(X, "ESS_PARAMS")} ageing.calibration.reference_dod_d'], st))
                i += 1
            st, _ = cmp_exact(w[i], a['eol_retention_r'])
            rows.append(row([i], 'R^DS (retention)', w[i], a['eol_retention_r'], [f'{src_arm}.eol_retention_r'], st))
            i += 1
            st, prec = cmp_round(w[i], k)
            kc = _k_closed(cal['cycles_n'], cal['reference_dod_d'], a['eol_retention_r'])
            if not _close(k, kc):
                st = 'MISMATCH'
            rows.append(row([i], 'k_e', w[i], k, [f'{esrc} model_variant.k_closed_form_by_arm.{"C2_calfade" if arm == "soh050" else arm}',
                                                 f'closed form N D / -ln R = {kc!r}'] +
                            ([f'T8 tables.ageing.rows[arm={arm}].k = {X.ageing[arm]["k"]!r}'] if arm in X.ageing else []), st, prec))
            i += 1
            st, _ = cmp_exact(w[i], a['calendar_retention_per_year'])
            rows.append(row([i], 'phi^Cal', w[i], a['calendar_retention_per_year'], [f'{src_arm}.calendar_retention_per_year'], st))
            i += 1
            want = {'end': 'end', 'mid': 'mid'}[a['available_energy_soh_point']]
            rows.append(row([], 'SoH point', k_label, a['available_energy_soh_point'], [f'{src_arm}.available_energy_soh_point'],
                            'match' if k_label == want else 'MISMATCH'))
            st, _ = cmp_exact(w[i], smin)
            rows.append(row([i], 'SoH^Min', w[i], smin, [smin_src], st))
            i += 1
            if i != len(tokens):
                raise KeyError(f'{vid}: {i} tokens used of {len(tokens)} ({names})')
            return rows
        return V(vid, frag, tokens, fn)

    ten = ('10\\,000', 0)
    out.append(table_row('V14', None, 'Baseline (cycling + calendar) & (10\\,000, 0.80, 0.80) & 35\\,851 & 0.985 & end & 0.70 \\\\',
                         'C2_calfade', [ten, ('0.80', 0), ('0.80', 1), ('35\\,851', 0), ('0.985', 0), ('0.70', 0)], k_label='end'))
    out.append(table_row('V15', None, 'Cycling only & (10\\,000, 0.80, 0.80) & 35\\,851 & 1.000 & end & 0.70 \\\\',
                         'C2', [ten, ('0.80', 0), ('0.80', 1), ('35\\,851', 0), ('1.000', 0), ('0.70', 0)], k_label='end'))
    lab50 = ('row label: retention (percent)', lambda X, a, s: (a['eol_retention_r'], 100.0), lambda X, sa, ss: f'{sa}.eol_retention_r')
    out.append(table_row('V16', lab50, 'Retention 50\\,\\% & (10\\,000, 0.80, 0.50) & 11\\,542 & 1.000 & end & 0.70 \\\\',
                         'C3_unit', [('50', 0), ten, ('0.80', 0), ('0.50', 0), ('11\\,542', 0), ('1.000', 0), ('0.70', 0)], k_label='end'))
    out.append(table_row('V17', lab50, 'Retention 50\\,\\%, mid-block & (10\\,000, 0.80, 0.50) & 11\\,542 & 1.000 & mid & 0.70 \\\\',
                         'C3_midblock', [('50', 0), ten, ('0.80', 0), ('0.50', 0), ('11\\,542', 0), ('1.000', 0), ('0.70', 0)], k_label='mid'))
    lab8 = ('row label: datasheet cycles', lambda X, a, s: (8000.0, 1.0),
            lambda X, sa, ss: jf(X, 'SPEC_V18', 'citations.cycle_life[EVE MB31].statement'))
    out.append(table_row('V18', lab8, 'Datasheet 8\\,000 cycles & (8\\,000, 1.00, 0.70) & 22\\,429 & 1.000 & end & 0.70 \\\\',
                         'C4', [('8\\,000', 0), ('8\\,000', 1), ('1.00', 0), ('0.70', 0), ('22\\,429', 0), ('1.000', 0), ('0.70', 1)],
                         datasheet=True, k_label='end'))
    lab050 = ('row label: floor', lambda X, a, s: (s, 1.0), lambda X, sa, ss: ss)
    out.append(table_row('V19', lab050, 'Baseline, 0.50 floor & (10\\,000, 0.80, 0.80) & 35\\,851 & 0.985 & end & 0.50 \\\\',
                         'soh050', [('0.50', 0), ten, ('0.80', 0), ('0.80', 1), ('35\\,851', 0), ('0.985', 0), ('0.50', 1)], k_label='end'))

    def v20(X, w):
        a = _ageing_arms(X)['no_ageing']
        h = X.c('p515_s44_campaign_harness')
        esrc = W171.jsrc(X, 'EXT_SPEC_V3')
        st, _ = cmp_exact(w[0], a['calendar_retention_per_year'])
        return [row([], 'no ageing: k_e', '$\\infty$', {'ageing_enabled': a['ageing_enabled'], 'k_field': X.ext3['model_variant']['k_closed_form_by_arm']['no_ageing']},
                    [f'{esrc} model_variant.arms.no_ageing.ageing_enabled', h.ref(r'ageing_enabled\s+bool \(False: SoH == 1 everywhere\)')],
                    'match' if a['ageing_enabled'] is False else 'MISMATCH', None,
                    'ageing disabled (the harness docstring: SoH == 1 everywhere), the reading of k = infinity; the k field of '
                    'the arm (11,541.56) is not consumed when ageing is off -- the implementation is W175\'s CONFIRM point'),
                row([0], 'no ageing: phi^Cal', w[0], a['calendar_retention_per_year'], [f'{esrc} model_variant.arms.no_ageing.calendar_retention_per_year'], st),
                row([], 'no ageing: (N, delta, R), SoH point, SoH^Min', '--- / --- / ---',
                    {'eol_retention_r_field': a['eol_retention_r'], 'soh_point_field': a['available_energy_soh_point']},
                    [f'{esrc} model_variant.arms.no_ageing'], 'match' if a['ageing_enabled'] is False else 'MISMATCH', None,
                    'not applicable with ageing off (SoH == 1, so no floor binds); the fields the arm carries are inert')]
    out.append(V('V20', 'No ageing & --- & $\\infty$ & 1.000 & --- & --- \\\\', [('1.000', 0)], v20))

    def v21(X, w):
        sp = (_ageing_arms(X)['C2_calfade']['available_energy_soh_point'], _ageing_arms(X)['C3_midblock']['available_energy_soh_point'])
        return [row([], 'SoH point definition (caption)', 'the state of health that sets a block\'s available energy (end of block, or its midpoint)',
                    sorted({a['available_energy_soh_point'] for a in _ageing_arms(X).values()}),
                    [f'{W171.jsrc(X, "EXT_SPEC_V3")} model_variant.arms[*].available_energy_soh_point'],
                    'match' if set(sp) == {'end', 'mid'} else 'MISMATCH', None, 'the formula is W175\'s CONFIRM point')]
    out.append(V('V21', 'SoH point: the state of health that sets a block\'s available energy (end of block, or its midpoint).', [], v21))

    # ---- section 3.6 ----------------------------------------------------------------------------------------------------
    def v22(X, w):
        c3 = X.j['W159_JSON']['c']['c3']
        c2 = X.j['W159_JSON']['c']['c2']['tally_d_4a82a64a_all_manifest_logs']
        ip = c3['ipopt']['version_output']['stdout']
        srcs = [jf(X, 'W159_JSON', 'c.c3.ipopt.version_output.stdout') + f': {ip.strip()!r}',
                jf(X, 'W159_JSON', 'c.c2.tally_d_4a82a64a_all_manifest_logs.versions') + f": {c2['versions']} ({c2['n_files']} logs)"]
        ver = re.search(r'Ipopt (\d+\.\d+\.\d+)', ip)
        rows = [row([0], 'IPOPT version', 'IPOPT 3.14.18', ver.group(1) if ver else None, srcs,
                    'match' if ver and ver.group(1) == w[0] and c2['versions'] == [w[0]] else 'MISMATCH', None,
                    'scope of the record: the W159 closing reads (2026-10-06) of the v6 campaign\'s logs and binary; the '
                    'environment is recorded unchanged since the first v6 start, not for the earlier search campaigns')]
        rows.append(row([1], 'Pyomo version', 'Pyomo 6.9.5', c3['pyomo_version'], [jf(X, 'W159_JSON', 'c.c3.pyomo_version')],
                        'match' if c3['pyomo_version'] == w[1] else 'MISMATCH'))
        pv = c3['sys_version'].split()[0]
        rows.append(row([2], 'Python version', 'Python 3.11.11', pv, [jf(X, 'W159_JSON', 'c.c3.sys_version') + f": {c3['sys_version']!r}"],
                        'match' if pv == w[2] else 'MISMATCH'))
        return rows
    out.append(V('V22', 'Local problems were solved with IPOPT~3.14.18 through Pyomo~6.9.5 on Python~3.11.11',
                 [('3.14.18', 0), ('6.9.5', 0), ('3.11.11', 0)], v22))

    def v23(X, w):
        c2 = X.j['W159_JSON']['c']['c2']['tally_d_4a82a64a_all_manifest_logs']
        c1 = X.j['W159_JSON']['c']['c1']
        ls = {k: v['solver']['options']['linear_solver'] for k, v in X.cp.items()}
        es = X.ess['solver']['options']['linear_solver']
        rows = [row([0], 'network linear solver', 'MA97', ls,
                    [W171.jline(X, f'CP_{k}', r'"linear_solver": "ma97"', expect=None) for k in X.cp] +
                    [jf(X, 'W159_JSON', 'c.c2.tally_d_4a82a64a_all_manifest_logs.network') + f": {c2['network']}"],
                    'match' if set(ls.values()) == {w[0].lower()} and set(c2['network']) == {w[0].lower()} else 'MISMATCH'),
                row([1], 'storage-agent linear solver', 'MA57', es,
                    [W171.jline(X, 'ESS_PARAMS', r'"linear_solver": "ma57"'),
                     jf(X, 'W159_JSON', 'c.c2.tally_d_4a82a64a_all_manifest_logs.esso') + f": {c2['esso']}"],
                    'match' if es == w[1].lower() and set(c2['esso']) == {w[1].lower()} else 'MISMATCH')]
        th = c1['all_table_evaluations_ran_with_OMP_NUM_THREADS_1']
        h = X.c('p515_s44_campaign_harness')
        rows.append(row([2], 'threading', 'single-threaded throughout', 'OMP/MKL/OpenBLAS/vecLib/NumExpr threads = 1',
                        [jf(X, 'W159_JSON', 'c.c1.all_table_evaluations_ran_with_OMP_NUM_THREADS_1') + f' = {th}',
                         h.ref(r"^THREAD_CAP_ENV = \{"), h.ref(r"raise SystemExit\(f'CHILD REFUSES: thread caps not in force")],
                        'match' if th is True else 'MISMATCH', None,
                        'record scope: every evaluation the frozen tables use (W159); "throughout" beyond the tables (the '
                        'search campaigns) runs through the same harness and its child refusal'))
        return rows
    out.append(V('V23', 'the network models with the MA97 linear solver and the storage agent with MA57, single-threaded '
                        'throughout', [('MA97', 0), ('MA57', 0), ('single', 0)], v23))

    def v24(X, w):
        sol = {k: v['solver']['options'] for k, v in X.cp.items()}
        net = X.c('network')
        cps = lambda rx: [W171.jline(X, f'CP_{k}', rx, expect=None) for k in X.cp]  # noqa: E731
        rows = []
        tol = {k: o['tol'] for k, o in sol.items()}
        st, _ = cmp_exact(w[0], tol['case9'])
        rows.append(row([0], 'network convergence tolerance (tol)', '$10^{-5}$', tol, cps(r'"tol": 1e-05|"tol": 1e-5'),
                        st if len(set(tol.values())) == 1 else 'MISMATCH'))
        at = {k: o['acceptable_tol'] for k, o in sol.items()}
        st, _ = cmp_exact(w[1], at['case9'])
        rows.append(row([1], 'network acceptable tolerance', '$10^{-4}$', at, cps(r'"acceptable_tol": 0\.0001|"acceptable_tol": 1e-4'),
                        st if len(set(at.values())) == 1 else 'MISMATCH'))
        tso = sol['case9'].get('compl_inf_tol')
        st, _ = cmp_exact(str(wv3(w[2]) * wv3(w[3])), tso)
        rows.append(row([2, 3], 'TSO complementarity tolerance (compl_inf_tol)', '$5 \\times 10^{-4}$', tso,
                        [W171.jline(X, 'CP_case9', r'"compl_inf_tol"')], st))
        dso = {k: sol[k].get('compl_inf_tol') for k in DN_NAMES}
        dflt = net.value(r'^IPOPT_DEFAULT_COMPL_INF_TOL = ([0-9.e-]+)')
        st, _ = cmp_exact(w[4], dflt)
        rows.append(row([4], 'DSO complementarity tolerance', '$10^{-4}$ (DSOs)', {'case_files': dso, 'ipopt_default': dflt},
                        [net.ref(r'^IPOPT_DEFAULT_COMPL_INF_TOL = ')] + [f'{X.inputs["CP_" + k]["path"]}: no compl_inf_tol key' for k in DN_NAMES],
                        st if set(dso.values()) == {None} else 'MISMATCH', None, 'IPOPT default (no DSO case file sets the option)'))
        mi = net.value(r"options\['max_iter'\] = (\d+)", cast=int)
        over = {k: o.get('max_iter') for k, o in sol.items()}
        st, _ = cmp_exact(w[5], mi)
        rows.append(row([5], 'network iteration limit (max_iter)', 'at most 500 iterations', mi, [net.ref(r"options\['max_iter'\] = 500")] +
                        [f'case files: max_iter {over} (no override)'], st if set(over.values()) == {None} else 'MISMATCH'))
        t6 = X.v6['inputs_in_force_now']['configuration_now']['convergence_depth_tail']
        t3 = X.s3x3['configuration']['convergence_depth_tail']
        st, _ = cmp_exact(w[6], t6['compl_inf_tol'])
        rows.append(row([6], 'tight-tail complementarity tolerance', '$10^{-6}$', {'v6': t6, '3x3': t3},
                        [f'{W171.jsrc(X, "V6_SPEC")} inputs_in_force_now.configuration_now.convergence_depth_tail',
                         f'{W171.jsrc(X, "SPEC_3X3")} configuration.convergence_depth_tail'],
                        st if t6 == t3 and t6.get('enabled') is True else 'MISMATCH'))
        return rows
    out.append(V('V24', 'Network solves use a convergence tolerance of $10^{-5}$, an acceptable tolerance of $10^{-4}$, a '
                        'complementarity tolerance of $5 \\times 10^{-4}$ (TSO) and $10^{-4}$ (DSOs), and at most 500 '
                        'iterations; the tight tail sets the complementarity tolerance to $10^{-6}$.',
                 [('10^{-5}', 0), ('10^{-4}', 0), ('5', 0), ('10^{-4}', 1), ('10^{-4}', 2), ('500', 0), ('10^{-6}', 0)], v24))

    def v25(X, w):
        sesd = X.c('shared_energy_storage_data')
        ov = {'tol': 1e-10, 'acceptable_tol': 1e-9}
        sesd.cite(r"^ESSO_TOL_OVERRIDES = \{'tol': 1e-10, 'acceptable_tol': 1e-9\}")
        srcs = [sesd.ref(r'^ESSO_TOL_OVERRIDES = '), sesd.ref(r'option_overrides=ESSO_TOL_OVERRIDES,'),
                sesd.ref(r'if option_overrides:'), W171.jline(X, 'ESS_PARAMS', r'"tol": 1e-6') + ' (file value, overridden)']
        a, _ = cmp_exact(w[0], ov['tol'])
        b, _ = cmp_exact(w[1], ov['acceptable_tol'])
        return [row([0], 'storage-agent tolerance', '$10^{-10}$', ov['tol'], srcs, a, None, 'applied after the file options to every ESSO solve'),
                row([1], 'storage-agent acceptable tolerance', '$10^{-9}$', ov['acceptable_tol'], srcs[:2], b)]
    out.append(V('V25', 'The storage agent solves to a tolerance of $10^{-10}$ with an acceptable tolerance of $10^{-9}$.',
                 [('10^{-10}', 0), ('10^{-9}', 0)], v25))

    def v26(X, w):
        net, sesd, spm = X.c('network'), X.c('shared_energy_storage_data'), X.c('solver_parameters')
        rec = {k: v['solver'].get('recovery_options') for k, v in X.cp.items()}
        pol = {k: v['solver'].get('recovery') for k, v in X.cp.items()}
        epol = X.ess['solver'].get('recovery')
        erec = X.ess['solver'].get('recovery_options')
        rows = []
        trig = [net.ref(r'po\.TerminationCondition\.internalSolverError,', expect=None),
                net.ref(r'po\.TerminationCondition\.maxIterations,', expect=None),
                net.ref(r'po\.TerminationCondition\.infeasible,', expect=None),
                sesd.ref(r'def _is_recoverable_shared_ess_failure\(result, params, node_id\):')]
        rows.append(row([], 'retry triggers', 'ends at the iteration limit, infeasible or in a solver error',
                        ['maxIterations', 'infeasible', 'internalSolverError'], trig, 'match', None,
                        'network.py _is_recoverable_network_failure and shared_energy_storage_data '
                        '_is_recoverable_shared_ess_failure test exactly these three termination conditions'))
        en = all(p is None for p in pol.values()) and epol is None
        rows.append(row([], 'retry enabled for every agent (tiers 1 and 2)', 'is retried ... and then once more',
                        {'case_file_recovery_blocks': pol, 'ess_recovery_block': epol, 'defaults': (True, True)},
                        [spm.ref(r'self\.recovery_enabled = True'), spm.ref(r'self\.recovery_tier2_enabled = True'),
                         spm.ref(r"parameters\.recovery_enabled = bool\(recovery_policy\.get\('enabled', True\)\)")],
                        'match' if en else 'MISMATCH', None, 'no case file or ESS file carries a solver.recovery block'))
        rows.append(row([], 'tier 1 = cold restart', 'retried as a cold restart with the agent\'s recovery settings',
                        "warm_start_init_point = 'no'", [net.ref(r"recovery_options\['warm_start_init_point'\] = 'no'"),
                                                         sesd.ref(r"recovery_options\['warm_start_init_point'\] = 'no'")], 'match'))
        tr = rec['case9'] or {}
        st, _ = cmp_exact(w[0], tr.get('acceptable_tol', -1))
        rows.append(row([0], 'TSO recovery acceptable_tol', 'TSO: acceptable tolerance $10^{-4}$', tr.get('acceptable_tol'),
                        [W171.jline(X, 'CP_case9', r'"recovery_options"') + ' solver.recovery_options.acceptable_tol', net.ref(r'recovery_options = \{')], st))
        st, _ = cmp_exact(w[1], tr.get('acceptable_iter', -1))
        rows.append(row([1], 'TSO recovery acceptable_iter', 'after one acceptable iteration', tr.get('acceptable_iter'),
                        [W171.jline(X, 'CP_case9', r'"recovery_options"') + ' solver.recovery_options.acceptable_iter'], st))
        dso = {k: rec[k] for k in DN_NAMES}
        rows.append(row([], 'DSO recovery settings', 'DSOs: their primary settings', dso,
                        [f'{X.inputs["CP_" + k]["path"]} solver.recovery_options = {rec[k]!r}' for k in DN_NAMES] +
                        [net.ref(r"key: value for key, value in \(params\.solver_params\.recovery_options or \{\}\)\.items\(\)")],
                        'match' if all(not v for v in dso.values()) else 'MISMATCH', None,
                        'empty or absent recovery_options: the cold restart keeps the primary options '
                        f"(acceptable_tol {X.cp['case33_1']['solver']['options']['acceptable_tol']}, acceptable_iter "
                        f"{X.cp['case33_1']['solver']['options']['acceptable_iter']})"))
        eff = dict(erec or {})
        eff.update({'tol': 1e-10, 'acceptable_tol': 1e-9})
        st, _ = cmp_exact(w[2], eff['acceptable_tol'])
        rows.append(row([2], 'storage-agent recovery acceptable_tol', 'storage agent: acceptable tolerance $10^{-9}$', eff['acceptable_tol'],
                        [W171.jline(X, 'ESS_PARAMS', r'"recovery_options"') + f' solver.recovery_options = {erec!r}',
                         sesd.ref(r'recovery_options\.update\(esso_option_overrides\)'), sesd.ref(r'^ESSO_TOL_OVERRIDES = ')], st, None,
                        'the file\'s 1e-4 is replaced by ESSO_TOL_OVERRIDES (applied to the retry too)'))
        st, _ = cmp_exact(w[3], eff.get('acceptable_iter', -1))
        rows.append(row([3], 'storage-agent recovery acceptable_iter', 'after one iteration', eff.get('acceptable_iter'),
                        [W171.jline(X, 'ESS_PARAMS', r'"recovery_options"') + ' solver.recovery_options.acceptable_iter'], st))
        rows.append(row([], 'tier 2', 'and then once more with the adaptive barrier strategy', "mu_strategy = 'adaptive', one attempt",
                        [net.ref(r"tier2_options\['mu_strategy'\] = 'adaptive'"), sesd.ref(r"tier2_options\['mu_strategy'\] = 'adaptive'"),
                         net.ref(r"and _is_recoverable_network_failure\(recovery_result, params\)")], 'match', None,
                        'tier 2 runs only after a failed tier-1 retry whose failure is itself recoverable; same recovery options plus '
                        'mu_strategy adaptive'))
        return rows
    out.append(V('V26', "A solve that ends at the iteration limit, infeasible or in a solver error is retried as a cold restart "
                        "with the agent's recovery settings (TSO: acceptable tolerance $10^{-4}$ after one acceptable "
                        "iteration; DSOs: their primary settings; storage agent: acceptable tolerance $10^{-9}$ after one "
                        "iteration) and then once more with the adaptive barrier strategy.",
                 [('10^{-4}', 0), ('one', 0), ('10^{-9}', 0), ('one', 1)], v26))
    out.append(V('V27', 'common objective scale $\\sigma = 93{,}635{,}360$', [('93{,}635{,}360', 0)],
                 lambda X, w: [prow(X, 'P19', [0], 'sigma (common objective scale)', '$\\sigma = 93,635,360$', w[0], P(X, 'P19')['found'],
                                    note=f"the 3x3 spec runs the same case file ({W171.jsrc(X, 'SPEC_3X3')} configuration.case_file "
                                         f"= {X.s3x3['configuration']['case_file']})")]))
    out.append(V('V28', 'reference converter rating $S^{\\text{ref}} = 2.5$~MVA', [('2.5', 0)],
                 lambda X, w: [prow(X, 'P12', [0], 'S^ref (MVA)', '$S^{ref} = 2.5$ MVA', w[0], P(X, 'P12')['found'])]))

    def v29(X, w):
        ri = interface_ratings(X)
        order = sorted(DN_NAMES, key=lambda n: ri[n]['node'])
        srcs = _ri_sources(X)
        rows = []
        ok_fixed = all(len(ri[n]['values']) == 1 for n in DN_NAMES) and all(ri[n]['node'] == ri[n]['node_3x3'] for n in DN_NAMES)
        for k, n in enumerate(order):
            v = ri[n]
            st, _ = cmp_exact(w[k], (v['rating_mva'] or -1) / (v['baseMVA'] or 1))
            rows.append(row([k], f'R^I ({n}, TN node {v["node"]}) p.u.', w[k], (v['rating_mva'] or 0) / (v['baseMVA'] or 1), srcs,
                            st if ok_fixed else 'MISMATCH', None,
                            f'fixed in every year file {", ".join(RATING_YEARS)}: {v["values"]}; incident branches {v["branches"]}'))
        for k, n in enumerate(order):
            st, _ = cmp_exact(w[3 + k], ri[n]['rating_mva'] or -1)
            rows.append(row([3 + k], f'R^I ({n}) MVA', w[3 + k], ri[n]['rating_mva'], srcs[3:4], st if ok_fixed else 'MISMATCH'))
        bases = sorted({ri[n]['baseMVA'] for n in DN_NAMES})
        st, _ = cmp_exact(w[6], bases[0] if len(bases) == 1 else -1)
        rows.append(row([6], 'base (MVA) of the per-unit ratings', '100 MVA base', bases, srcs[3:4] + [f'W171a parameter check P29 baseMVA {P(X, "P29")["found"]}'], st))
        for k, n in enumerate(order):
            st, _ = cmp_exact(w[7 + k], ri[n]['node'])
            rows.append(row([7 + k], f'TN node of {n}', w[7 + k], ri[n]['node'], srcs[4:5], st))
        return rows
    out.append(V('V29', 'interface-transformer ratings $R^{\\text{I}} = 2.0$, $1.0$ and $1.5$~p.u. (200, 100 and 150~MVA on a '
                        '100~MVA base) at TN nodes 5, 7 and 9',
                 [('2.0', 0), ('1.0', 0), ('1.5', 0), ('200', 0), ('100', 0), ('150', 0), ('100', 1), ('5', 0), ('7', 0), ('9', 0)], v29))

    def v30(X, w):
        p = P(X, 'P20')
        f = p['found']
        rows = [row([], 'kappa^E formula', '$\\kappa^{E} = \\sigma/\\operatorname{median}_b w_b$', X.srp1p['admm']['esso_al_scale'],
                    [W171.jline(X, 'SRP1_PARAMS', r'"esso_al_scale"'),
                     X.c('shared_resources_planning').ref(r'al_scale_esso = objective_scale_used / median_block_weight')],
                    'match' if X.srp1p['admm']['esso_al_scale'] == 'sigma_over_median_block_weight' else 'MISMATCH')]
        for k, (key, q) in enumerate((('SRP1', 'kappa^E single-scenario instance'), ('3x3', 'kappa^E multi-scenario instance'))):
            st, prec = cmp_round(w[k], f[key])
            rows.append(row([k], q, w[k], f[key], p['sources'], st if p['status'] == 'match' else 'MISMATCH', prec,
                            f"recomputed: sigma / median w_b, median {f[key + '_median_w_b']!r} over {f[key + '_n_blocks']} blocks"))
        return rows
    out.append(V('V30', "storage-agent scale $\\kappa^{\\text{E}} = \\sigma/\\operatorname{median}_b w_b$ ($227{,}211$ at the "
                        "single-scenario instance, $386{,}259$ at the multi-scenario instance)",
                 [('227{,}211', 0), ('386{,}259', 0)], v30))

    def v31(X, w):
        rho = P(X, 'P13')['found']
        srcs = P(X, 'P13')['sources']
        rows = []
        for k, (ch, q) in enumerate((('v', 'rho^V_0'), ('pf', 'rho^PF_0'), ('ess', 'rho^E_0'))):
            vals = sorted(set(rho[ch].values()))
            st, _ = cmp_exact(w[k], vals[0] if len(vals) == 1 else -1)
            rows.append(row([k], q, w[k], rho[ch], srcs, st, None, 'one value for every agent'))
        return rows
    out.append(V('V31', 'initial penalties $\\rho^{V}_0 = 0.0077$, $\\rho^{\\text{PF}}_0 = 0.198$, $\\rho^{\\text{E}}_0 = 0.01$',
                 [('0.0077', 0), ('0.198', 0), ('0.01', 0)], v31))

    def v32(X, w):
        pu = X.srp1p['admm']['penalty_update']
        s = lambda rx: W171.jline(X, 'SRP1_PARAMS', rx)  # noqa: E731
        ex = pu['balancing_exempt_until']['ess']
        items = [([0], 'residual-balance ratio mu', pu['residual_balance_ratio'], [s(r'"residual_balance_ratio": 5')]),
                 ([1], 'ratio for the decrease of rho^PF', pu['residual_balance_ratio_pf_decrease'], [s(r'"residual_balance_ratio_pf_decrease"')]),
                 ([2], 'increase / decrease factor', pu['increase_factor'] if pu['increase_factor'] == pu['decrease_factor'] else None,
                  [s(r'"increase_factor"'), s(r'"decrease_factor"')]),
                 ([3], 'rho lower bound', pu['min'], [s(r'"min": 1e-4|"min": 0\.0001')]),
                 ([4], 'rho upper bound', pu['max'], [s(r'"max": 1e4|"max": 10000')]),
                 ([5], 'per-channel freeze after unchanged cycles', pu['freeze_after_unchanged_cycles'], [s(r'"freeze_after_unchanged_cycles"')]),
                 ([6], 'freeze backstop cycle', pu['freeze_backstop_cycle'], [s(r'"freeze_backstop_cycle"')]),
                 ([7], 'storage-channel exemption: dual ratio below', ex['dual_ratio_below'], [s(r'"ess": \{"dual_ratio_below"')]),
                 ([8], 'storage-channel exemption: consecutive cycles', ex['consecutive_cycles'], [s(r'"ess": \{"dual_ratio_below"')])]
        rows = []
        for tk, q, v, srcs in items:
            st, _ = cmp_exact(w[tk[0]], v if v is not None else float('nan'))
            rows.append(row(tk, q, w[tk[0]], v, srcs + [f'{W171.jsrc(X, "SRP1_PARAMS")} admm.adaptive_penalty = {X.srp1p["admm"]["adaptive_penalty"]}'],
                            st if X.srp1p['admm']['adaptive_penalty'] is True else 'MISMATCH'))
        return rows
    out.append(V('V32', 'residual balancing with ratio 5 (3 for the decrease of $\\rho^{\\text{PF}}$), factor 1.5, bounds '
                        '$[10^{-4}, 10^{4}]$, a per-channel freeze after ten unchanged cycles and a backstop at cycle 200, the '
                        'storage channel exempt until its dual ratio has been below one on five consecutive cycles',
                 [('5', 0), ('3', 0), ('1.5', 0), ('10^{-4}', 0), ('10^{4}', 0), ('ten', 0), ('200', 0), ('one', 0), ('five', 0)], v32))

    def v33(X, w):
        bo = P(X, 'P16')['found']
        a, _ = cmp_exact(w[0], bo['eps_abs'])
        b, _ = cmp_exact(w[1], bo['eps_rel'])
        return [row([0], 'Boyd eps_abs', w[0], bo['eps_abs'], P(X, 'P16')['sources'], a),
                row([1], 'Boyd eps_rel', w[1], bo['eps_rel'], P(X, 'P16')['sources'], b)]
    out.append(V('V33', 'residual tolerances $\\varepsilon_{\\text{abs}} = 10^{-5}$, $\\varepsilon_{\\text{rel}} = 10^{-4}$',
                 [('10^{-5}', 0), ('10^{-4}', 0)], v33))

    def v34(X, w):
        aac = X.srp1p['admm']['anderson_acceleration']
        p = P(X, 'P22')
        a, _ = cmp_exact(w[0], aac['memory'])
        b, _ = cmp_exact(w[1], aac['regularization'])
        return [row([], 'Anderson acceleration enabled', 'Anderson acceleration', aac['enabled'], p['sources'][:1] + p['sources'][3:4],
                    'match' if aac['enabled'] is True and p['status'] == 'match' else 'MISMATCH'),
                row([0], 'AA memory', w[0], aac['memory'], [W171.jline(X, 'SRP1_PARAMS', r'"memory": 5')], a),
                row([1], 'AA regularisation (Tikhonov)', w[1], aac['regularization'], [W171.jline(X, 'SRP1_PARAMS', r'"regularization": 1e-10')], b)]
    out.append(V('V34', 'Anderson acceleration with memory 5 and regularisation $10^{-10}$', [('5', 0), ('10^{-10}', 0)], v34))

    def v35(X, w):
        mc, rc = X.srp1p['admm']['minimum_consecutive_converged_cycles'], X.s3x3['required_consecutive_cycles']
        a, _ = cmp_exact(w[0], mc)
        b, _ = cmp_exact(w[1], X.s3x3['cap'])
        return [row([0], 'production exit: consecutive passing cycles', 'ten consecutive passing cycles', {'SRP1_params': mc, '3x3_spec': rc},
                    [W171.jline(X, 'SRP1_PARAMS', r'"minimum_consecutive_converged_cycles"'), f'{W171.jsrc(X, "SPEC_3X3")} required_consecutive_cycles',
                     X.c('shared_resources_planning').ref(r'convergence = \(consecutive_converged_cycles >= ')],
                    a if mc == rc else 'MISMATCH'),
                row([1], 'cycle cap at the multi-scenario instance', 'a cap of 500 cycles', X.s3x3['cap'], [f'{W171.jsrc(X, "SPEC_3X3")} cap'], b)]
    out.append(V('V35', 'The production exit is ten consecutive passing cycles, with a cap of 500 cycles at the multi-scenario '
                        'instance.', [('ten', 0), ('500', 0)], v35))

    def v36(X, w):
        al = P(X, 'P27')['found']
        a, _ = cmp_exact(w[0], 0.5 if al == [json.dumps({'alpha': 0.5, 'floor': None})] else -1)
        pin = P(X, 'P28')['found']
        b, _ = cmp_exact(str(wv3(w[1]) * wv3(w[2])), pin)
        return [row([0], 'commitment premium alpha (3x3)', '$\\alpha = 0.5$', al, P(X, 'P27')['sources'], a),
                row([1, 2], 'voltage regularisation weight (3x3, solver-side)', '$9 \\times 10^{4}$', pin, P(X, 'P28')['sources'], b, None,
                    'excluded from the reported Q')]
    out.append(V('V36', 'the commitment premium is $\\alpha = 0.5$ and the solver-side voltage regularisation weight is '
                        '$9 \\times 10^{4}$.', [('0.5', 0), ('9', 0), ('10^{4}', 0)], v36))

    def v37(X, w):
        sc = X.c('settling_criterion')
        dr, rr = sc.value(r'^DELTA_R = ([0-9.]+)'), sc.value(r'^R_REF = ([0-9.]+)')
        tau = dr * rr / 4.0
        pm = X.v6['stop_rule']['p_max']
        a, _ = cmp_exact(w[0], dr)
        b, _ = cmp_exact(w[1], rr)
        c, prec = cmp_round(w[2], tau)
        d_, _ = cmp_exact(w[3], pm['value'])
        return [row([0], 'delta_R', w[0], dr, [sc.ref(r'^DELTA_R = ')], a),
                row([1], 'reference value V (EUR)', w[1], rr, [sc.ref(r'^R_REF = ')], b),
                row([2], 'tau = delta_R V / 4 (EUR)', w[2], tau, [sc.ref(r'^TAU = DELTA_R \* R_REF / 4\.0')], c, prec),
                row([3], 'P_max (cycles)', w[3], pm['value'], [f'{W171.jsrc(X, "V6_SPEC")} stop_rule.p_max = {pm}'], d_, None,
                    f"L_MONO = 2 P_max = {pm.get('L')}")]
    out.append(V('V37', 'uses $\\delta_R = 0.07$ on the reference value $\\mathcal{V} = 259{,}375.33$~\\euro{}, hence $\\tau = '
                        '4{,}539.07$~\\euro{}, and $P_{\\max} = 30$ cycles',
                 [('0.07', 0), ('259{,}375.33', 0), ('4{,}539.07', 0), ('30', 0)], v37))

    def v38(X, w):
        sc2 = X.c('settling_criterion_v2')
        gated, ungated, bad = [], [], []
        for spec_key, cells in (('V6_SPEC', X.v6['cells']), ('EXT_SPEC_V3', X.ext3['cells']), ('A64_SPEC', X.j['A64_SPEC']['cells'])):
            for k, c in cells.items():
                cr = c.get('cap_rule') or {}
                g = c.get('gated')
                if cr.get('kind') == 'fixed':
                    gated.append((spec_key, k))
                    if not (cr.get('formula') == 'N_old + 100' and cr.get('cap') == (c.get('N_old') or 0) + 100 and g is True):
                        bad.append((spec_key, k, 'fixed'))
                elif cr.get('kind') == 'dynamic':
                    ungated.append((spec_key, k))
                    if not (cr.get('after_first_k0') == 109 and cr.get('ceiling') == 300 and not g):
                        bad.append((spec_key, k, 'dynamic'))
                else:
                    bad.append((spec_key, k, cr.get('kind')))
        a, _ = cmp_exact(w[0], 100 if gated and not bad else -1)
        b, _ = cmp_exact(w[1], sc2.value(r'^CAP_AFTER_K0 = (\d+)', cast=int))
        c, _ = cmp_exact(w[2], sc2.value(r'^CAP_CEILING = (\d+)', cast=int))
        srcs = [f'{W171.jsrc(X, "V6_SPEC")} cells[*].cap_rule / gated / N_old', f'{W171.jsrc(X, "EXT_SPEC_V3")} cells[*].cap_rule',
                jf(X, 'A64_SPEC', 'cells[*].cap_rule')]
        note = (f'continued (replay-gated on the earlier run, cap = N_old + 100): {len(gated)} cells; fresh (dynamic cap): '
                f'{len(ungated)} cells; violations {bad}')
        return [row([0], 'cap of a continued evaluation (cycles after the earlier run\'s stop)', 'capped 100 cycles after that run\'s stopping cycle',
                    'N_old + 100', srcs, a if not bad else 'MISMATCH', None, note),
                row([1], 'cap of a fresh evaluation: cycles after k0', '$k_0 + 109$', 109, srcs + [sc2.ref(r'^CAP_AFTER_K0 = ')], b if not bad else 'MISMATCH'),
                row([2], 'cap of a fresh evaluation: ceiling', '300', 300, srcs + [sc2.ref(r'^CAP_CEILING = ')], c if not bad else 'MISMATCH')]
    out.append(V('V38', "an evaluation continued from an earlier run is capped 100 cycles after that run's stopping cycle, a fresh "
                        "one at $\\min\\{k_0 + 109, 300\\}$", [('100', 0), ('109', 0), ('300', 0)], v38))

    def v39(X, w):
        sp = _search_specs(X)
        pb, f2 = X.c('p515_s47_phase_b_record'), X.c('p515_s53_f2_certificate')
        rows = []
        d0 = {k: sp[k].get('delta_0') for k in ('S47', 'S51')}
        st, _ = cmp_exact(w[0], pb.value(r'^DELTA_0 = (\d+)', cast=int))
        rows.append(row([0], 'Delta_0 (variant A)', '$\\Delta_0 = 4$', d0, [pb.ref(r'^DELTA_0 = 4'), jf(X, 'S47_SPEC', 'extra.poll_design.delta_0'),
                                                                        jf(X, 'S51_SPEC', 'extra.poll_design.delta_0')],
                        st if set(d0.values()) == {4} else 'MISMATCH'))
        nb = {k: sp[k]['max_new_evaluations'] for k in sp}
        st, _ = cmp_exact(w[1], nb['S47'])
        rows.append(row([1], 'budget of new evaluations, variant A', 'budgets of 20 new evaluations', {k: nb[k] for k in ('S47', 'S51')},
                        [pb.ref(r'^MAX_NEW_EVALUATIONS = 20'), jf(X, 'S47_SPEC', 'extra.poll_design.max_new_evaluations'),
                         jf(X, 'S51_SPEC', 'extra.poll_design.max_new_evaluations')], st if nb['S47'] == nb['S51'] else 'MISMATCH'))
        va = [k for k in sp if sp[k]['design'] == 'orthomads_n_plus_1_neg']
        st, _ = cmp_exact(w[2], len(va))
        rows.append(row([2], 'number of variant-A searches', 'the two variant-A searches', va,
                        [jf(X, f'{k}_SPEC', 'extra.poll_design.design') + f" = {sp[k]['design']}" for k in sp], st))
        st, _ = cmp_exact(w[3], nb['S53'])
        rows.append(row([3], 'budget of new evaluations, variant B', '60 for the variant-B certificate', nb['S53'],
                        [f2.ref(r'^MAX_NEW_EVALUATIONS = 60'), jf(X, 'S53_SPEC', 'extra.poll_design.max_new_evaluations')],
                        st if sp['S53']['design'] == 'orthomads_2n' else 'MISMATCH'))
        cc = {k: sp[k]['completion_cap'] for k in sp}
        st, _ = cmp_exact(w[4], pb.value(r'^COMPLETION_CAP = (\d+)', cast=int))
        rows.append(row([4], 'completion cap (points)', 'a completion cap of 30 points', cc,
                        [pb.ref(r'^COMPLETION_CAP = 30'), f2.ref(r'^COMPLETION_CAP = PB\.COMPLETION_CAP')] +
                        [jf(X, f'{k}_SPEC', 'extra.poll_design.completion_cap') for k in sp], st if set(cc.values()) == {30} else 'MISMATCH'))
        sq = {}
        for k in ('S47', 'S51', 'S53'):
            vals = sorted(set(float(x) for x in re.findall(r'"sigma_Q[a-z_]*": ([0-9.]+)', json.dumps(X.j[f'{k}_RESULTS']))))
            sq[k] = vals
        one = sorted({v for vs in sq.values() for v in vs})
        st, prec = cmp_round(w[5], one[0] if len(one) == 1 else float('nan'))
        rows.append(row([5], 'sigma_Q (EUR)', '$\\sigma_Q = 18,449.66$ EUR', one, [jf(X, f'{k}_RESULTS', 'sigma_Q_eur (every occurrence)') for k in sq] +
                        [jf(X, 'S47_RESULTS', 'A6') + ': "sigma_Q = phase_a_tables T3 residual_max_abs_eur = 18,449.66 EUR"'],
                        st if len(one) == 1 else 'MISMATCH', prec, 'one value in all three search campaigns'))
        return rows
    out.append(V('V39', 'used $\\Delta_0 = 4$ lattice units, budgets of 20 new evaluations for the two variant-A searches and 60 '
                        'for the variant-B certificate, a completion cap of 30 points, and the measured resolution $\\sigma_Q = '
                        '18{,}449.66$~\\euro{} of the recourse.',
                 [('4', 0), ('20', 0), ('two', 0), ('60', 0), ('30', 0), ('18{,}449.66', 0)], v39))

    def v40(X, w):
        arms = X.t['benchmark']['w160_additions']['arms_in_full']
        st, _ = cmp_exact(w[0], len(arms))
        return [row([0], 'number of static arrangements', 'two static arrangements', sorted(arms),
                    ['T6 tables.benchmark.w160_additions.arms_in_full (keys)'], st, None,
                    'the arm definitions are W175\'s CONFIRM point')]
    out.append(V('V40', 'is measured against two static arrangements of the same system', [('two', 0)], v40))

    def v41(X, w):
        r = W164._three_start(X)
        st, _ = cmp_exact(w[0], 3 if r['status'] == 'match' else -1)
        return [row([0], 'solver starts per arrangement', 'the best of three solver starts', r['counterpart_value'], [r['counterpart']], st,
                    None, r['note'])]
    out.append(V('V41', 'Each arrangement is reported as the best of three solver starts.', [('three', 0)], v41))
    return out


VALUE_EXCLUDED = ('the named settings of the benchmark paragraph (l. 1138-1143: no-reverse-flow rule, passive and price-taker '
                  'arm definitions, TSO dispatch at fixed interface exchanges) are the `% [CONFIRM -- W175]` analysis and are '
                  'not rows here; only its printed numbers ("two", "three") are')


# ======================================================================================================================
#  4. declarations version 3: section 2, Appendix A, sections 3.1 / 3.3 / 3.4, letter
# ======================================================================================================================
def _pb(X):
    return X.c('p515_s47_phase_b_record')


def _f2(X):
    return X.c('p515_s53_f2_certificate')


def _src_specs(X, field):
    return [jf(X, f'{k}_SPEC', field) for k in ('S47', 'S51', 'S53')]


def _w173_f2(X):
    return _w173(X)['item2']['F2']


def sec2_declarations_v3():
    M = MAIN
    out = []

    def s301(X, w):
        pb = _pb(X)
        n = pb.value(r'^N_VARS = 2 \* len\(ACTIVE_NODES\) \+ 1', expect=1, group=0, cast=str)
        nodes = re.search(r'^ACTIVE_NODES = \((.*)\)', pb.text, re.M).group(1)
        k = len([x for x in nodes.split(',') if x.strip()])
        src = '; '.join([pb.ref(r'^N_VARS = 2 \* len\(ACTIVE_NODES\) \+ 1'), pb.ref(r'^ACTIVE_NODES = ')])
        return [R('match' if wv3(w[0]) == 2 else 'MISMATCH', src, 'code (named record)', 2, '2', f'(zP, zE) per node; |E^S| = {k}; {n}'),
                R('match' if wv3(w[1]) == 1 else 'MISMATCH', src, 'code (named record)', 1, '+1', 'the common investment-year index')]
    out.append(D('S301', M, '\\in \\mathbb{Z}^{2|E^S|+1}$', [('2', 0), ('1', 0)], s301))

    def s302(X, w):
        pb, f2, sp = _pb(X), _f2(X), _search_specs(X)
        d0 = {k: sp[k].get('delta_0') for k in ('S47', 'S51')}
        return [R('match' if wv3(w[0]) == pb.value(r'^DELTA_0 = (\d+)', cast=int) and set(d0.values()) == {4} else 'MISMATCH',
                  '; '.join([pb.ref(r'^DELTA_0 = 4'), jf(X, 'S47_SPEC', 'extra.poll_design.delta_0'), jf(X, 'S51_SPEC', 'extra.poll_design.delta_0')]),
                  'code + spec fields', d0, '4'),
                R('match' if wv3(w[1]) == 1 and sp['S53'].get('delta') == 1 else 'MISMATCH',
                  '; '.join([f2.ref(r'^DELTA_UNIT = PB\.DELTA_MIN'), _pb(X).ref(r'^DELTA_MIN = 1'), jf(X, 'S53_SPEC', 'extra.poll_design.delta')]),
                  'code + spec field', sp['S53'].get('delta'), '1')]
    out.append(D('S302', M, '($\\Delta_0 = 4$ lattice units in variant A; $\\Delta = 1$ throughout in variant B)', [('4', 0), ('1', 0)], s302))

    def s303(X, w):
        sp = _search_specs(X)
        mp = {k: sp[k]['max_polls'] for k in sp}
        return R('match' if wv3(w[0]) == _pb(X).value(r'^MAX_POLLS = (\d+)', cast=int) and set(mp.values()) == {60} else 'MISMATCH',
                 '; '.join([_pb(X).ref(r'^MAX_POLLS = 60'), _f2(X).ref(r'^MAX_POLLS = PB\.MAX_POLLS')] + _src_specs(X, 'extra.poll_design.max_polls')),
                 'code + spec fields', mp, '60')
    out.append(D('S303', M, '\\While{polls remain (at most 60) and a poll can be launched}', [('60', 0)], s303))

    def s304(X, w):
        sp = _search_specs(X)
        nd = {k: (sp[k]['design'], sp[k]['n_directions'], sp[k]['n_vars']) for k in ('S47', 'S51')}
        ok = all(v == ('orthomads_n_plus_1_neg', 8, 7) for v in nd.values()) and wv3(w[0]) == 1
        return R('match' if ok else 'MISMATCH', '; '.join([_pb(X).ref(r"^POLL_DESIGN = 'orthomads_n_plus_1_neg'"),
                                                            _pb(X).ref(r'neg = \[-sum\(c\[i\] for c in cols\) for i in range\(n\)\]'),
                                                            jf(X, 'S47_SPEC', 'extra.poll_design'), jf(X, 'S51_SPEC', 'extra.poll_design')]),
                 'code + spec fields', nd, 'n + 1', 'n Householder columns and minus their sum: n_directions 8 = n_vars 7 + 1')
    out.append(D('S304', M, '$\\mathcal{P} \\gets$ the $n+1$ OrthoMADS points', [('1', 0)], s304))

    def s305(X, w):
        pb = _pb(X)
        sp = _search_specs(X)
        return [R('match' if wv3(w[0]) == 1 and sp['S47'].get('unit_poll_completion') is True else 'MISMATCH',
                  '; '.join([pb.ref(r'completion = lattice\.completion\(tuple\(inc'), pb.ref(r'^UNIT_POLL_COMPLETION = True'),
                             jf(X, 'S47_SPEC', 'extra.poll_design.unit_poll_completion')]), 'code + spec field', 1, '1',
                  'the full admissible unit box is added at every unit poll of variant A'),
                W171._poll_rule(X, 'neighbour', w[1]),
                R('match' if wv3(w[2]) == pb.value(r'^COMPLETION_CAP = (\d+)', cast=int) else 'MISMATCH',
                  '; '.join([pb.ref(r'^COMPLETION_CAP = 30'), pb.ref(r"if completion is not None and completion\['over_cap'\]:")]),
                  'code (named record)', 30, '30', 'refuses (stop for review), never truncates')]
    out.append(D('S305', M, '\\If{$\\Delta = 1$}{ $\\mathcal{P} \\gets \\mathcal{P} \\cup$ every admissible lattice point within one '
                            'unit step of $\\boldsymbol{z}^{\\text{inc}}$ in the $\\infty$-norm; \\lIf{$|\\mathcal{P}| > 30$}',
                 [('1', 0), ('one', 0), ('30', 0)], s305))
    out.append(D('S306', M, '\\Else(variant B, $\\Delta = 1$ throughout)', [('1', 0)], lambda X, w: R(
        'match' if wv3(w[0]) == 1 and _search_specs(X)['S53'].get('delta') == 1 else 'MISMATCH',
        '; '.join([_f2(X).ref(r'^DELTA_UNIT = PB\.DELTA_MIN'), jf(X, 'S53_SPEC', 'extra.poll_design.delta')]), 'code + spec field', 1, '1')))

    def s307(X, w):
        f2, sp = _f2(X), _search_specs(X)['S53']
        return [R('match' if sp['design'] == 'orthomads_2n' and sp['n_directions'] == 2 * sp['n_vars'] else 'MISMATCH',
                  '; '.join([f2.ref(r'^N_DIRECTIONS = 2 \* N_VARS'), f2.ref(r"^POLL_DESIGN = 'orthomads_2n'"), jf(X, 'S53_SPEC', 'extra.poll_design')]),
                  'code + spec field', sp['n_directions'], '2n'),
                R('match' if wv3(w[1]) == 1 and sp['min_feasible_poll_points'] == sp['n_vars'] + 1 else 'MISMATCH',
                  '; '.join([f2.ref(r'^MIN_FEASIBLE_POLL_POINTS = N_VARS \+ 1'), jf(X, 'S53_SPEC', 'extra.poll_design.min_feasible_poll_points')]),
                  'code + spec field', sp['min_feasible_poll_points'], 'n + 1'),
                R('match' if wv3(w[2]) == sp['completion_cap'] == 30 else 'MISMATCH',
                  '; '.join([f2.ref(r'^COMPLETION_CAP = PB\.COMPLETION_CAP'), jf(X, 'S53_SPEC', 'extra.poll_design.completion_cap')]),
                  'code + spec field', sp['completion_cap'], '30', 'the cap refuses the poll (STOP_FOR_REVIEW_completion_cap)')]
    out.append(D('S307', M, '$\\mathcal{P} \\gets$ the $2n$ OrthoMADS points $\\boldsymbol{z}^{\\text{inc}} \\pm \\boldsymbol{v}$, each '
                            'inadmissible point snapped to the nearest admissible point of the incumbent\'s unit frame; if fewer than '
                            '$n+1$ distinct admissible points result, add every admissible unit neighbour (the poll is refused for '
                            'review beyond 30 points)', [('2n', 0), ('1', 0), ('30', 0)], s307))

    def s308(X, w):
        specs = {k: (X.j[f'{k}_SPEC']['required_consecutive_cycles'], X.j[f'{k}_SPEC']['cap']) for k in ('S47', 'S51', 'S53')}
        pb = _pb(X)
        return [R('match' if {v[0] for v in specs.values()} == {wv3(w[0])} else 'MISMATCH',
                  '; '.join(_src_specs(X, 'required_consecutive_cycles') + [pb.ref(r'^REQUIRED_CONSECUTIVE_CYCLES = 10')]),
                  'spec fields', specs, '10'),
                R('match' if {v[1] for v in specs.values()} == {wv3(w[1])} else 'MISMATCH',
                  '; '.join(_src_specs(X, 'cap') + [pb.ref(r'^CAP = 500')]), 'spec fields', specs, '500'),
                R('match' if wv3(w[2]) == 10 and 'step over the last 10 cycles' in pb.text else 'MISMATCH',
                  pb.ref(r"'step over the last 10 cycles; a missing bar makes the resolution \+inf'\)"), 'code (named record)', 10, '10',
                  'RESOLUTION_RULE: bar = record max gross step over the last 10 cycles')]
    out.append(D('S308', M, 'under the production exit (ten consecutive passing cycles, cap 500); record $F$, its bar (the largest '
                            'objective step over the last ten cycles) and the exit', [('ten', 0), ('500', 0), ('ten', 1)], s308))

    def s309(X, w):
        pb, sp = _pb(X), _search_specs(X)['S53']
        bs = (pb.value(r'^BARRIER_STOP_PER_POLL = (\d+)', cast=int), pb.value(r'^BARRIER_STOP_OVERALL = (\d+)', cast=int))
        src = '; '.join([pb.ref(r'^BARRIER_STOP_PER_POLL = 2'), pb.ref(r'^BARRIER_STOP_OVERALL = 3'), jf(X, 'S53_SPEC', 'extra.poll_design.barrier_stop')])
        ok = list(bs) == list(sp['barrier_stop'])
        return [R('match' if ok and wv3(w[0]) == bs[0] else 'MISMATCH', src, 'code + spec field', bs, '2', 'barrier points in one poll'),
                unch('method statement', '"in one poll" (the scope of the per-poll count)'),
                R('match' if ok and wv3(w[2]) == bs[1] else 'MISMATCH', src, 'code + spec field', bs, '3', 'barrier points in all')]
    out.append(D('S309', M, '(two in one poll, or three in all, stop the search for review)', [('two', 0), ('one', 0), ('three', 0)], s309))

    def s310(X, w):
        pb, sp = _pb(X), _search_specs(X)['S53']
        return [R('match' if wv3(w[0]) == 2 else 'MISMATCH', '; '.join([pb.ref(r'^\s+delta \*= 2\s*$'), jf(X, 'S47_SPEC', 'extra.poll_design.success')]),
                  'code + spec field', _search_specs(X)['S47'].get('success'), '2'),
                R('match' if wv3(w[1]) == 1 and sp.get('on_success') == 'Delta stays 1' else 'MISMATCH',
                  jf(X, 'S53_SPEC', 'extra.poll_design.on_success'), 'spec field', sp.get('on_success'), '1')]
    out.append(D('S310', M, '$\\Delta \\gets 2\\Delta$ (variant A) or $1$ (variant B)', [('2', 0), ('1', 0)], s310))
    out.append(D('S311', M, '\\lIf{$\\Delta = 1$}{\\textbf{terminate}', [('1', 0)], lambda X, w: W171._poll_rule(X, 'unit', w[0])))
    out.append(D('S312', M, '$\\Delta \\gets \\Delta/2$', [('2', 0)], lambda X, w: W171._poll_rule(X, 'halve', w[0])))

    def s313(X, w):
        sp = _search_specs(X)
        designs = {k: sp[k]['design'] for k in sp}
        return [count_eq3(w[0], len(set(designs.values())), '; '.join(_src_specs(X, 'extra.poll_design.design')), 'spec fields',
                          f'{designs}'),
                R('match' if {designs['S47'], designs['S51']} == {'orthomads_n_plus_1_neg'} and wv3(w[1]) == 1 else 'MISMATCH',
                  '; '.join(_src_specs(X, 'extra.poll_design.design')[:2]), 'spec fields', designs, 'n + 1')]
    out.append(D('S313', M, 'Two poll variants were used: variant A (the $n+1$ OrthoMADS directions', [('Two', 0), ('1', 0)], s313))
    out.append(D('S314', M, 'and variant B (the $2n$ directions snapped to the unit frame)', [('2n', 0)], lambda X, w: R(
        'match' if _search_specs(X)['S53']['design'] == 'orthomads_2n' else 'MISMATCH', jf(X, 'S53_SPEC', 'extra.poll_design.design'),
        'spec field', _search_specs(X)['S53']['design'], '2n')))

    def s315(X, w):
        b = _w173(X)['item2']['x0']['b_box']
        return count_eq3(w[0], b['n_evaluated'] if b['n_feasible'] == b['n_evaluated'] else -1,
                         jf(X, 'W173_JSON', 'item2.x0.b_box') + f' {b}', 'committed record', 'full box: n_feasible = n_evaluated')
    out.append(D('S315', M, 'every admissible unit neighbour was evaluated (fourteen, the full box', [('fourteen', 0)], s315))

    def s316(X, w):
        b = _w173_f2(X)['b_box']
        st = _w173_f2(X)['c_search_time_evaluated']
        src = jf(X, 'W173_JSON', 'item2.F2.b_box')
        note = f'search-time classes of the 17: {st}; none better at search time'
        return [count_eq3(w[0], b['n_evaluated_in_search'], src + '.n_evaluated_in_search', 'committed record', note),
                count_eq3(w[1], b['n_feasible'], src + '.n_feasible', 'committed record'),
                count_eq3(w[2], b['n_in_final_poll_set'], src + '.n_in_final_poll_set', 'committed record'),
                count_eq3(w[3], b['n_cached_outside_poll_set'], src + '.n_cached_outside_poll_set', 'committed record')]
    out.append(D('S316', M, 'seventeen of its 61 admissible unit neighbours were evaluated (the ten points of the final poll and '
                            'seven earlier evaluations)', [('seventeen', 0), ('61', 0), ('ten', 0), ('seven', 0)], s316))

    def s317(X, w):
        c = _w173_f2(X)['c_L_claims_box_neighbours']
        src = jf(X, 'W173_JSON', 'item2.F2.c_L_claims_box_neighbours')
        neg = c.get('negative') or []
        note = f'negative (neighbour better) {len(neg)}: {json.dumps(neg)[:300]}'
        return [count_eq3(w[0], c['n'], src + '.n', 'committed record', note),
                count_eq3(w[1], c['n_positive_plan_better'], src + '.n_positive_plan_better', 'committed record'),
                count_eq3(w[2], c['n_positive_determinate'], src + '.n_positive_determinate', 'committed record'),
                count_eq3(w[3], c['n_positive_within_bar'], src + '.n_positive_within_bar', 'committed record')]
    out.append(D('S317', M, 'of the thirteen re-evaluated under the certification rule the plan is better than twelve, seven '
                            'determinately and five within resolution', [('thirteen', 0), ('twelve', 0), ('seven', 0), ('five', 0)], s317))

    def s318(X, w):
        sc2 = X.c('settling_criterion_v2')
        return R('match' if wv3(w[0]) == 3 else 'MISMATCH', sc2.ref(r'p_hat = T\[-1\]\[0\] - T\[-3\]\[0\]') + ' (inherited by v6)',
                 'code (named record)', 'T[-1] - T[-3]', 'three most recent', 'P_hat spans the three most recent turning points')
    out.append(D('S318', M, 'the number of cycles spanned by the three most recent turning points', [('three', 0)], s318))

    def s319(X, w):
        sc = X.c('settling_criterion')
        return [R('match' if wv3(w[0]) == 100 else 'MISMATCH', sc.ref(r'^EPS0 = TAU / 100\.0'), 'code (named record)', 'TAU / 100', '100'),
                R('match' if wv3(w[1]) == sc.value(r'^K_EXCL = (\d+)', cast=int) else 'MISMATCH', sc.ref(r'^K_EXCL = 3'),
                  'code (named record)', 3, '3')]
    out.append(D('S319', M, 'a step smaller than $\\tau/100$ carries no sign, and the first three cycles after $k_0$ do not enter '
                            'the count', [('100', 0), ('three', 0)], s319))
    out.append(D('S320', M, 'or every step below $\\tau/100$), its range is at most', [('100', 0)], lambda X, w: R(
        'match' if wv3(w[0]) == 100 else 'MISMATCH', '; '.join([X.c('settling_criterion').ref(r'^EPS0 = TAU / 100\.0'),
                                                                 X.c('settling_criterion').ref(r'stationary\) OR mean \|dQ\| over the second half of')]),
        'code (named record)', 'max |dQ| < EPS0 (stationary sub-case)', '100')))

    def s321(X, w):
        doc = X.main
        a = doc.raw.find('\\item the objective has shown at least three turning points')
        b = doc.raw.find('\\end{enumerate}', a)
        items = doc.raw[a:b].split('\\item')[1:] if a >= 0 and b > a else []
        ok4 = len(items) == 5 and 'gap' in items[3]
        ok5 = len(items) == 5 and 'cleanly' in items[4]
        src = f'{MAIN} at the declared commit: the enumerate list of the certification rule ({len(items)} items)'
        return [R('match' if ok4 and wv3(w[0]) == 4 else 'MISMATCH', src, 'structure of the text', 'item 4: the gap clause', '4'),
                R('match' if ok5 and wv3(w[1]) == 5 else 'MISMATCH', src, 'structure of the text', 'item 5: the clean veto', '5')]
    out.append(D('S321', M, 'and conditions 4 and 5 hold on that window', [('4', 0), ('5', 0)], s321))
    out.append(D('S322', M, 'The multi-scenario instance was evaluated under the production exit (ten consecutive passing cycles '
                            'with every local solve successful)', [('ten', 0)], lambda X, w: W171._prod_exit(X, w[0])))
    out.append(D('S323', M, 'A difference between two certified evaluations is called determinate when it is at least '
                            '$\\max\\{3\\,b, 2\\tau\\}$, where $b$ is the larger of the two certification bands',
                 [('two', 0), ('3', 0), ('2', 0), ('two', 1)],
                 lambda X, w: [w163ref(X, 'N73', w[0], 2, 'two', None), w163ref(X, 'N28', w[1], 3, '3', None),
                               w163ref(X, 'N29', w[2], 2, '2', None), w163ref(X, 'N73', w[3], 2, 'two', None)]))
    out.append(D('S324', M, 'is determinate only if it exceeds three times the largest of their consensus gaps and settling slacks',
                 [('three', 0)], lambda X, w: w163ref(X, 'N26', w[0], 3, 'three', None)))

    def s325(X, w):
        det = X.j['W169_JSON']['detector']
        mx = det['max ratio at P over certified points']
        bound = wv3(w[0]) * wv3(w[1])
        return [R('match' if mx < bound else 'MISMATCH', jf(X, 'W169_JSON', 'detector."max ratio at P over certified points"') +
                  f' = {mx!r} ({det["argmax cell"]})', 'committed record', mx, f'{bound:g}', f'bound: {mx:.3g} < {bound:g}')] * 2
    out.append(D('S325', M, 'at the certified points it was below $4 \\times 10^{-5}$ of the rating', [('4', 0), ('10^{-5}', 0)], s325))
    out.append(D('S326', M, 'over a window of $2P_{\\max}$ cycles starting after $k_0$ with $P_{\\max}$ the longest period '
                            'measured on the instance', [('2P_', 0)],
                 lambda X, w: w163ref(X, 'N15', None, None, '2 P_max = 60', None,
                                      'the factor 2 of 2 P_max: v6 spec stop_rule.W.monotone L = 2 * P_MAX')))
    return out


def appendix_a_declarations_v3():
    M = MAIN
    out = []

    def pu(X):
        return X.srp1p['admm']['penalty_update']

    def jl(X, rx):
        return W171.jline(X, 'SRP1_PARAMS', rx)
    out.append(D('A301', M, '$\\rho_g \\gets 1.5\\,\\rho_g$ when', [('1.5', 0)],
                 lambda X, w: count_eq3(w[0], pu(X)['increase_factor'], jl(X, r'"increase_factor"'), 'case file')))
    out.append(D('A302', M, '$\\rho_g \\gets \\rho_g/1.5$ when', [('1.5', 0)],
                 lambda X, w: count_eq3(w[0], pu(X)['decrease_factor'], jl(X, r'"decrease_factor"'), 'case file')))
    out.append(D('A303', M, "with $\\mu = 5$ and $\\mu' = 5$ ($\\mu' = 3$ for the decrease of $\\rho^{\\text{PF}}$), clamped to "
                            "$[10^{-4}, 10^{4}]$", [('5', 0), ('5', 1), ('3', 0), ('10^{-4}', 0), ('10^{4}', 0)],
                 lambda X, w: [count_eq3(w[0], pu(X)['residual_balance_ratio'], jl(X, r'"residual_balance_ratio": 5'), 'case file', 'increase test'),
                               count_eq3(w[1], pu(X)['residual_balance_ratio'], jl(X, r'"residual_balance_ratio": 5'), 'case file',
                                         'decrease test (V and E channels)'),
                               count_eq3(w[2], pu(X)['residual_balance_ratio_pf_decrease'], jl(X, r'"residual_balance_ratio_pf_decrease"'), 'case file'),
                               count_eq3(w[3], pu(X)['min'], jl(X, r'"min": 1e-4|"min": 0\.0001'), 'case file'),
                               count_eq3(w[4], pu(X)['max'], jl(X, r'"max": 1e4|"max": 10000'), 'case file')]))
    out.append(D('A304', M, 'until its dual ratio has been below one on five consecutive cycles (its two-phase schedule), a channel '
                            'freezes once it has gone ten cycles without changing after having acted, every channel freezes at cycle 200',
                 [('one', 0), ('five', 0), ('ten', 0), ('200', 0)],
                 lambda X, w: [count_eq3(w[0], pu(X)['balancing_exempt_until']['ess']['dual_ratio_below'], jl(X, r'"ess": \{"dual_ratio_below"'), 'case file'),
                               count_eq3(w[1], pu(X)['balancing_exempt_until']['ess']['consecutive_cycles'], jl(X, r'"ess": \{"dual_ratio_below"'), 'case file'),
                               count_eq3(w[2], pu(X)['freeze_after_unchanged_cycles'], jl(X, r'"freeze_after_unchanged_cycles"'), 'case file'),
                               count_eq3(w[3], pu(X)['freeze_backstop_cycle'], jl(X, r'"freeze_backstop_cycle"'), 'case file')]))
    out.append(D('A305', M, '$\\varepsilon_{\\text{abs}} = 10^{-5}$, $\\varepsilon_{\\text{rel}} = 10^{-4}$. A cycle passes',
                 [('10^{-5}', 0), ('10^{-4}', 0)],
                 lambda X, w: [count_eq3(w[0], X.srp1p['admm']['tol']['boyd']['eps_abs'], jl(X, r'"boyd": \{"eps_abs"'), 'case file'),
                               count_eq3(w[1], X.srp1p['admm']['tol']['boyd']['eps_rel'], jl(X, r'"boyd": \{"eps_abs"'), 'case file')]))
    out.append(D('A306', M, 'The production exit is ten consecutive passing cycles; it stopped the multi-scenario evaluations.',
                 [('ten', 0)], lambda X, w: W171._prod_exit(X, w[0])))
    out.append(D('A307', M, 'with memory 5 and a Tikhonov regularisation of $10^{-10}$', [('5', 0), ('10^{-10}', 0)],
                 lambda X, w: [count_eq3(w[0], X.srp1p['admm']['anderson_acceleration']['memory'], jl(X, r'"memory": 5'), 'case file'),
                               count_eq3(w[1], X.srp1p['admm']['anderson_acceleration']['regularization'], jl(X, r'"regularization": 1e-10'), 'case file')]))
    out.append(D('A308', M, 'tightened to $10^{-6}$ (the tight tail)', [('10^{-6}', 0)],
                 lambda X, w: count_eq3(w[0], X.v6['inputs_in_force_now']['configuration_now']['convergence_depth_tail']['compl_inf_tol'],
                                        f'{W171.jsrc(X, "V6_SPEC")} inputs_in_force_now.configuration_now.convergence_depth_tail', 'spec field')))

    def a309(X, w):
        net, sesd = X.c('network'), X.c('shared_energy_storage_data')
        src = '; '.join([net.ref(r"recovery_result, recovery_log_path = _run_smopf_solver_attempt\("),
                         net.ref(r"tier2_result, tier2_log_path = _run_smopf_solver_attempt\("),
                         sesd.ref(r"tier2_options\['mu_strategy'\] = 'adaptive'")])
        return count_eq3(w[0], 2, src, 'code (named record)', 'tier 1 cold restart with the recovery options; tier 2 the same plus '
                                                              'mu_strategy adaptive')
    out.append(D('A309', M, 'is retried in two tiers, a cold restart with the agent\'s recovery settings', [('two', 0)], a309))
    out.append(D('A310', M, 'production: ten consecutive passing cycles; single-scenario tables', [('ten', 0)],
                 lambda X, w: W171._prod_exit(X, w[0])))
    return out


def sec3_declarations_v3():
    M = MAIN
    out = []

    def q4(X):
        return X.j['W167_Q4']['as_read_by_production']

    def yrs(X, w):
        return [R('match' if x in X.years() else 'MISMATCH', f'{X.src("SRP1_JSON")} Years {sorted(X.years())}', 'named record',
                  sorted(X.years()), x, 'a representative year of the revised instance') for x in w]
    out.append(D('C301', M, 'Parameter & Scn. & Prob., [\\%] & 2025 & 2030 & 2035', [('2025', 0), ('2030', 0), ('2035', 0)], yrs))

    def cost_row(cid, frag, kind, sc, toks):
        def fn(X, w):
            q = q4(X)
            costs = q['cost_power_eur_per_MVA' if kind == 'P' else 'cost_energy_eur_per_MWh'][str(sc)]
            src = jf(X, 'W167_Q4', f"as_read_by_production.{'cost_power_eur_per_MVA' if kind == 'P' else 'cost_energy_eur_per_MWh'}.{sc}")
            out_ = [count_eq3(w[0], sc if sc <= q['num_scenarios'] else -1, jf(X, 'W167_Q4', 'as_read_by_production.num_scenarios'),
                              'committed record', 'scenario index'),
                    num(w[1], q['scenario_probabilities'][sc - 1], jf(X, 'W167_Q4', f'as_read_by_production.scenario_probabilities[{sc - 1}]'),
                        'committed record', scale=100.0)]
            for k, y in enumerate(('2025', '2030', '2035')):
                out_.append(num(w[2 + k], costs[y], f'{src}.{y} (EUR; printed in k EUR)', 'committed record', scale=1e-3,
                                note=f"W167 fragment {X.inputs['W167_TAB']['path']} (sha256 {X.inputs['W167_TAB']['sha256'][:8]}) "
                                     'generated through production\'s reader'))
            return out_
        return D(cid, M, frag, toks, fn)
    out.append(cost_row('C302', '$c^\\text{Inv,S}_{y,c}$, & 1 & 35.00\\% & 214.17 & 168.72 & 153.93', 'P', 1,
                        [('1', 0), ('35.00', 0), ('214.17', 0), ('168.72', 0), ('153.93', 0)]))
    out.append(cost_row('C303', '$\\left[\\text{k\\euro/MVA}\\right]$ & 2 & 55.00\\% & 267.53 & 224.25 & 206.96', 'P', 2,
                        [('2', 0), ('55.00', 0), ('267.53', 0), ('224.25', 0), ('206.96', 0)]))
    out.append(cost_row('C304', '~ & 3 & 10.00\\% & 342.17 & 278.10 & 268.58', 'P', 3,
                        [('3', 0), ('10.00', 0), ('342.17', 0), ('278.10', 0), ('268.58', 0)]))
    out.append(cost_row('C305', '$c^\\text{Inv,E}_{y,c}$, & 1 & 35.00\\% & 212.13 & 167.11 & 152.46', 'E', 1,
                        [('1', 0), ('35.00', 0), ('212.13', 0), ('167.11', 0), ('152.46', 0)]))
    out.append(cost_row('C306', '$\\left[\\text{k\\euro/MWh}\\right]$ & 2 & 55.00\\% & 264.98 & 222.12 & 204.99', 'E', 2,
                        [('2', 0), ('55.00', 0), ('264.98', 0), ('222.12', 0), ('204.99', 0)]))
    out.append(cost_row('C307', '~ & 3 & 10.00\\% & 338.91 & 275.45 & 266.02', 'E', 3,
                        [('3', 0), ('10.00', 0), ('338.91', 0), ('275.45', 0), ('266.02', 0)]))

    def c308(X, w):
        q = q4(X)
        return [num(w[0], q['expected_power_eur_per_MVA']['2025'], jf(X, 'W167_Q4', 'as_read_by_production.expected_power_eur_per_MVA.2025'),
                    'committed record', scale=1e-3, note=q['expected_formula']),
                num(w[1], q['expected_energy_eur_per_MWh_production_function']['2025'],
                    jf(X, 'W167_Q4', 'as_read_by_production.expected_energy_eur_per_MWh_production_function.2025'), 'committed record', scale=1e-3),
                R('match' if w[2] in X.years() else 'MISMATCH', f'{X.src("SRP1_JSON")} Years', 'named record', sorted(X.years()), w[2])]
    out.append(D('C308', M, '(256.32~k\\euro/MVA and 253.88~k\\euro/MWh in 2025)', [('256.32', 0), ('253.88', 0), ('2025', 0)], c308))

    def divisors(X):
        q = X.j['W167_Q4']
        now = q['formula_recompute_7ce1d1ab']['formula_templates']['energy']
        pred = json.dumps(q['predecessor_072b1310_selected'])
        dn = re.search(r'\)/(\d)\*1000', now)
        dp = re.search(r'\)/(\d)\*1000\*\(\'Cost breakdown NREL\'!\$B\$2\+', pred)
        return (int(dn.group(1)) if dn else None, int(dp.group(1)) if dp else None,
                jf(X, 'W167_Q4', 'formula_recompute_7ce1d1ab.formula_templates.energy') + f': {now[:90]!r}...',
                jf(X, 'W167_Q4', 'predecessor_072b1310_selected[sheet Investment Cost, Energy].first_formula_per_row') + ' (".../5*1000*...")')

    def c309(X, w):
        dn, dp, sn, sp = divisors(X)
        return [count_eq3(w[0], dn, sn, 'committed record', 'the 4-h system cost: the divisor of the current energy formula'),
                count_eq3(w[1], dp, sp, 'committed record', 'divisor of the submitted (predecessor 072b1310) file'),
                count_eq3(w[2], dn, sn, 'committed record', 'divisor of the corrected file (7ce1d1ab = HEAD)')]
    out.append(D('C309', M, 'derived from the 4-h system cost by dividing by 5 instead of 4.', [('4', 0), ('5', 0), ('4', 1)], c309))
    out.append(D('C310', M, 'The three trajectories enter the plan\'s investment cost', [('three', 0)],
                 lambda X, w: count_eq3(w[0], q4(X)['num_scenarios'], jf(X, 'W167_Q4', 'as_read_by_production.num_scenarios'), 'committed record')))

    def c311(X, w):
        ri = interface_ratings(X)
        order = sorted(DN_NAMES, key=lambda n: ri[n]['node'])
        srcs = _ri_sources(X)
        b1 = all(all(bid == 1 and kind == 'transformers' for kind, bid in br) for n in DN_NAMES for br in ri[n]['branches'])
        doc = X.main
        row_line = next((ln for ln in doc.lines if re.match(r'\s*1 & 1 & 2 & ', ln)), '')
        printed = row_line.split('&')[6].strip() if row_line.count('&') >= 6 else None
        out_ = [R('match' if b1 and wv3(w[0]) == 1 else 'MISMATCH', '; '.join(srcs[:2] + srcs[3:4]), 'case files', ri[DN_NAMES[0]]['branches'], '1',
                  'the only in-service branch incident to the DN reference node is transformer branch_id 1 (bus 1 - 2)')]
        for k, n in enumerate(order):
            out_.append(count_eq3(w[1 + k], ri[n]['rating_mva'], '; '.join(srcs[3:4]), 'case files', f'{n} (TN node {ri[n]["node"]})'))
        for k, n in enumerate(order):
            out_.append(count_eq3(w[4 + k], ri[n]['node'], srcs[4], 'named record', f'{n}'))
        n5 = next(n for n in DN_NAMES if ri[n]['node'] == 5)
        out_.append(R('match' if printed is not None and float(printed) == ri[n5]['rating_mva'] and wv3(w[7]) == 5 else 'MISMATCH',
                      f'{MAIN} branch table row "1 & 1 & 2 & ..." column S^Rated = {printed}; {srcs[3]}', 'structure of the text + case file',
                      ri[n5]['rating_mva'], printed, f'node 5 = {n5}'))
        return out_
    out.append(D('C311', M, 'Branch 1 is the interface transformer, rated 200, 100 and 150~MVA for the ADNs at TN nodes 5, 7 and 9; the '
                            'table prints the node-5 value.',
                 [('1', 0), ('200', 0), ('100', 0), ('150', 0), ('5', 0), ('7', 0), ('9', 0), ('5', 1)], c311))
    return out


def letter_declarations_v3():
    L = LETTER
    out = []

    def l301(X, w):
        y1 = set(X.years().values())
        y3 = X.i3x3['Years']
        return [count_eq3(w[0], 5 if y1 == {5} else -1, f'{X.src("SRP1_JSON")} Years {X.years()} (block lengths)', 'named record'),
                count_eq3(w[1], len(y3), f'{W171.jsrc(X, "INST_3X3")} Years {y3} (count)', 'named record'),
                count_eq3(w[2], 3 if set(y3.values()) == {3} else -1, f'{W171.jsrc(X, "INST_3X3")} Years (block lengths)', 'named record')]
    out.append(D('L301', L, 'five-year blocks at the single-scenario instance; five three-year blocks at the multi-scenario instance',
                 [('five', 0), ('five', 1), ('three', 0)], l301))

    def l302(X, w):
        fy = {r['arm']: r['floor_year_070'] for r in X.t['ageing']['rows']}
        named = {'C2_calfade', 'C3_midblock', 'C3_unit'}            # baseline, mid-block, unit-retention
        others = {a for a in fy if a not in named and a != 'no_ageing'}
        ok = all(fy[a] == 2035 for a in named) and all(fy[a] is None for a in others)
        return [R('match' if ok and w[0] == '2035' else 'MISMATCH', 'T8 tables.ageing.rows[*].floor_year_070', 'frozen-table cells', fy, '2035',
                  'baseline = C2_calfade, mid-block = C3_midblock, unit-retention = C3_unit'),
                count_eq3(w[1], len(others) if ok else -1, 'T8 tables.ageing.rows[*].floor_year_070 (aged arms with None)', 'frozen-table cells',
                          f'the other two aged arms: {sorted(others)}')]
    out.append(D('L302', L, 'binds (in 2035 under the baseline, the mid-block and the unit-retention calibrations; not within the '
                            'horizon under the other two)', [('2035', 0), ('two', 0)], l302))

    def l303(X, w):
        q = X.j['W167_Q4']
        now = q['formula_recompute_7ce1d1ab']['formula_templates']['energy']
        pred = json.dumps(q['predecessor_072b1310_selected'])
        dn = re.search(r'\)/(\d)\*1000', now)
        dp = re.search(r"\)/(\d)\*1000\*\('Cost breakdown NREL'!\$B\$2\+", pred)
        dn, dp = (int(dn.group(1)) if dn else None), (int(dp.group(1)) if dp else None)
        sn = jf(X, 'W167_Q4', 'formula_recompute_7ce1d1ab.formula_templates.energy')
        sp = jf(X, 'W167_Q4', 'predecessor_072b1310_selected (energy formula ".../5*1000*...")')
        return [count_eq3(w[0], dn, sn, 'committed record', 'the 4-h system cost'),
                count_eq3(w[1], dp, sp, 'committed record'),
                count_eq3(w[2], dn, sn, 'committed record'),
                num(w[3], dp / dn if dn and dp else float('nan'), f'{sp}; {sn} (ratio of the divisors)', 'committed record',
                    note='every energy cost x 5/4 against the submitted file (W167 year_tables_crosscheck: 15 energy cells differ)')]
    out.append(D('L304', L, 'the horizon (three representative years standing for five-year blocks at the single-scenario '
                            'instance', [('three', 0)],
                 lambda X, w: count_eq3(w[0], len(X.years()), f'{X.src("SRP1_JSON")} Years {sorted(X.years())} (count)', 'named record')))
    out.append(D('L303', L, 'the energy cost per MWh was derived from the 4-h system cost by dividing by 5 instead of 4 in the '
                            'submitted version; the corrected values (×1.25)', [('4', 0), ('5', 0), ('4', 1), ('1.25', 0)], l303))
    return out


def all_declarations_v3():
    return value_declarations() + sec2_declarations_v3() + appendix_a_declarations_v3() + sec3_declarations_v3() + \
        letter_declarations_v3()


# v2 declarations whose fragment is still found but which are re-pointed in version 3 (reason)
SUPERSEDED_V3 = {}
# v2 declarations whose fragment is gone: the version-3 declaration that replaced the sentence, or why no number is left
REPLACED_BY_V3 = {
    'L43v2': ('-', 'R3.5: the clause "The sentence of the submitted version announcing 0.5 % and 2 % cases ... was removed" is '
                   'deleted (Addendum 69 D8: the reviewers never saw that text); no number left'),
    'L14': ('L304, L301', 'R1.2 (round 2 B.6 second part): "the horizon (three representative years standing for five-year '
                          'blocks)" -> "... five-year blocks at the single-scenario instance; five three-year blocks at the '
                          'multi-scenario instance"'),
    'M12': ('V10, V14-V18', 'the red section 3.4 paragraph (70 % floor) is deleted (round 3 C.3); the floor SoH^Min = 0.70 is '
                            'printed in section 3.5 (text and ageing table)'),
    'M13': ('-', 'deleted with the red paragraph (Addendum 68 D6; round 3 C.3): the 60 % / 80 % floors were never evaluated; '
                 'no number left'),
    'M14': ('V10', 'the red calendar paragraph (1 %/yr, 14 %, 15-year, 20 years) is deleted (round 3 C.3); the calendar '
                   'retention 0.985 / 80 % over the 15-year calendar life is printed in section 3.5'),
    'M15': ('-', 'deleted with M14 (round 3 C.3): the 0.5 % / 2.0 % calendar cases were never run; no number left'),
    'S205': ('S318', '"(measured between the first and third turning points)" -> "the number of cycles spanned by the three '
                     'most recent turning points" (round 2 B.2; Addendum 69 D3)'),
    'S210': ('S326', '"over a window of $2P_{\\max}$ cycles with $P_{\\max}$" -> "... cycles starting after $k_0$ with '
                     '$P_{\\max}$" (round 2 B.2)'),
    'S217': ('S323', '"called determinate when it exceeds" -> "... when it is at least" (round 2; Addendum 69 D3)'),
    'S218': ('S324', '"three times the larger of that evaluation\'s consensus gap and settling slack" -> "three times the '
                     'largest of their consensus gaps and settling slacks" (round 2; Addendum 69 D3)'),
    'S219': ('S304, S307, S313, S314', 'Algorithm 1 rewritten to the search as run (round 2 A.1-A.4; Addendum 69 D4): '
                                       'variant A n + 1 directions, variant B 2n'),
    'S220': ('S305', 'the completion line rewritten: "every admissible lattice point within one unit step of z^inc in the '
                     'infinity-norm" (variant A, at Delta = 1)'),
    'S225': ('S305, S307', '"\\If{$|\\mathcal{P}| < n + 1$}" -> variant A "\\If{$\\Delta = 1$}" (unit-box completion) and '
                           'variant B "if fewer than $n+1$ distinct admissible points result" (round 2 A.2, round 3 B.c)'),
    'S226': ('S310', '"$\\Delta^p \\gets 2\\Delta^p$" -> "$\\Delta \\gets 2\\Delta$ (variant A) or $1$ (variant B)"'),
    'S227': ('S311', '"\\lIf{$\\Delta^p = 1$}" -> "\\lIf{$\\Delta = 1$}"'),
    'S228': ('S312', '"$\\Delta^p \\gets \\Delta^p / 2$" -> "$\\Delta \\gets \\Delta/2$"'),
    'S230': ('S322', '"evaluated under the production exit --- ten consecutive cycles passing the residual test" -> "... '
                     'under the production exit (ten consecutive passing cycles with every local solve successful)" '
                     '(round 2 B.3)'),
}


# ======================================================================================================================
#  5. main.tex line rules v3 (re-pinned regions) and rules
# ======================================================================================================================
def map_regions(old_lines, new_lines):
    """the W171a submitted-version regions mapped into this main.tex: a region is kept iff every one of its lines lies in
    an unchanged (difflib equal) block and the mapped lines are consecutive."""
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    lm = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                lm[i1 + k + 1] = j1 + k + 1
    kept, dropped = [], []
    for (a, b), key, kind, why in W171.MAIN_REGIONS:
        ms = [lm.get(x) for x in range(a, b + 1)]
        ok = all(m is not None for m in ms) and all(ms[i + 1] == ms[i] + 1 for i in range(len(ms) - 1))
        if ok:
            kept.append(((ms[0], ms[-1]), key, kind, why))
        else:
            dropped.append({'old': [a, b], 'key': key, 'n_lines_changed': sum(m is None for m in ms),
                            'reason': REGION_DROPPED.get(key, 'TEXT CHANGED (no reason declared)')})
    return tuple(kept), dropped, lm


def region_of(ln):
    for key, (a, b), *_ in REVISED:
        if a <= ln <= b:
            return key
    return None


def rule_assign_main_v3(X, doc, t, map_xref_numbers):
    typ, ln, txt = t['type'], t['line'], t['text']
    line = doc.lines[ln - 1]
    reg = region_of(ln)
    if t['scope'] == 'comment':
        sub_ = 'template boilerplate' if ln <= 60 or line.lstrip().startswith('%%') else \
            ('[CONFIRM] / [AUTHOR] instruction comment in a revised passage (' + reg + '; the CONFIRM analysis is W172 / '
             'W173 / W175 scope)') if reg else 'commented-out text'
        return unch('comment (not typeset)', f'{sub_}; comments are not manuscript text')
    lines_ok = X.main_sha == MAIN_LINE_RULES_SHA
    if lines_ok and reg is None:
        for (a, b), key, kind, why in X.main_regions_v3:
            if a <= ln <= b:
                return W164._map_sub(X, key, kind, why)
    if typ == 'doi':
        return unch('reference identifier', 'DOI (reference-list identifier)')
    if typ == 'label':
        if re.fullmatch(r'R\d\.\d+', txt):
            return unch('enumerator', 'reviewer item label')
        if re.fullmatch(r'T\d+', txt):
            return unch('enumerator', 'table identifier of the frozen tables')
        if re.fullmatch(r'C\d', txt):
            return unch('enumerator', 'ageing calibration name')
        return unch('identifier', 'label containing digits')
    if typ in ('hash', 'date', 'alnum'):
        return unch('notation' if reg else 'identifier', f'{typ} in a revised passage (notation)' if reg else f'{typ} in the submitted text')
    if typ in ('num', 'dotted') and W171._in_xref(doc, t):
        return unch('cross-reference', 'section / table / figure / equation / algorithm number written in prose' +
                    W171._xref_note(X, doc, t))
    if typ == 'word' and W164._compound(doc, t):
        return unch('not a figure', f'compound adjective ({txt}{line[t["end0"]:t["end0"] + 12].split()[0]})')
    if typ == 'word' and txt.lower() == 'single':
        return unch('not a figure', 'article sense ("a single ...")')
    if reg:
        where = dict((k, d) for k, _r, d, _h in REVISED)[reg]
        if typ == 'word':
            return unch('method statement', f'number word in {where} describing structure (no value to check; the statements '
                        'are audited against the code by W171b / W172 / W174b)')
        if 'algorithm' in t['envs'] or t['in_math']:
            return unch('notation', f'formula constant in {where} (an index offset, exponent, bound or coefficient of the printed '
                        'formula; the equations and algorithms are audited against the code by W171b / W172 / W174b)')
        return None
    if typ == 'num' and re.fullmatch(r'(19|20)\d\d', txt):
        if txt in X.years():
            return R('match', f'{X.src("SRP1_JSON")} Years {sorted(X.years())}', 'named record', sorted(X.years()), txt,
                     'a representative year of the revised instance')
        return W164._map_sub(X, 'years', 'input replaced', 'non-representative year of the submitted horizon')
    if typ == 'num' and re.search(r'IEEE\s*$', line[:t['col0']]) and line[t['end0']:t['end0'] + 4] == '-bus':
        return W164._bus(X, 'case9', t['written']) if txt == '9' else W164._bus(X, 'case33', t['written']) if txt == '33' else None
    if typ == 'num' and re.search(r'[Nn]odes?(?:~|\s)*(?:\d+\s*,\s*(?:and\s*)?|\d+\s*,?\s*and\s*)*$', line[:t['col0']]) \
            and 'tabular' not in ' '.join(t['envs']):
        ids = sorted(d['connection_node_id'] for d in X.srp1['DistributionNetworks'])
        return R('match' if int(float(txt)) in ids else 'MISMATCH', f'{X.src("SRP1_JSON")} DistributionNetworks[*].connection_node_id '
                 f'{ids}', 'named record', ids, txt, 'TN interface node of a distribution network (an identifier)')
    if 'postcode=' in line:
        return unch('address', 'postcode of an affiliation')
    if not lines_ok:
        return None
    if 'algorithm' in t['envs']:
        return unch('notation', f'constant in an algorithm (audited: {W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appA"])})')
    if t['in_math']:
        return unch('notation', 'formula constant (equations are audited against the code, W166 / W171b / W172)')
    if any(e.startswith('tabular') for e in t['envs']):
        if NETWORK_TABLES[0] <= ln <= NETWORK_TABLES[1]:
            return unch('network/data parameter', 'case-data table (RES capacity / generator and node IDs), outside the frozen '
                        f'tables ({W164.MAP_REL} line {X.map_line(W164.MAP_CITES["years"])})')
        if APP_BCD[0] <= ln <= APP_BCD[1]:
            return unch('network/data parameter', 'IEEE 33-bus / market data table, outside the frozen tables '
                        f'({W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appBCD"])}: keep)')
    if typ == 'word':
        if txt.lower() == 'third':
            return unch('enumerator', 'ordinal word')
        return unch('descriptive count', 'number word in unrevised submitted prose (counts of lists, layers, agents, reference '
                    'cases); wording checks are not required this round')
    return None


def rule_assign_v3(X, doc, role, t, map_xref_numbers, qstat):
    if role == 'main':
        return rule_assign_main_v3(X, doc, t, map_xref_numbers)
    if t['type'] == 'word' and t.get('tokenizer') == 'v3 extra number word' and t['scope'] == 'comment':
        return unch('comment (not typeset)', 'number word in a comment')
    return W171.rule_assign_v2(X, doc, role, t, map_xref_numbers, qstat)


# ======================================================================================================================
#  6. findings
# ======================================================================================================================
def letter_findings_v3(X, docs, qrows):
    letter = docs[LETTER]

    def lno(frag):
        hits, _n = letter.find_fragment(frag)
        return letter.flat_idx[hits[0]][0] if len(hits) == 1 else None
    fy = {r['arm']: r['floor_year_070'] for r in X.t['ageing']['rows']}
    further = letter.raw.split('\\section*{Further changes', 1)[1] if '\\section*{Further changes' in letter.raw else ''
    stray = [i + 1 for i, ln in enumerate(letter.lines) if '(Section~4.5)".' in ln]
    b6 = 'five three-year blocks at the multi-scenario instance'
    vc = X.verdict_changes()
    rchanges_fig2 = [i + 1 for i, ln in enumerate(letter.lines) if 'Figure~2 and the graphical abstract' in ln]
    sub_has = 'annual rates of 0.5' in X.sub.raw
    main_has = 'annual rates of 0.5' in X.main.raw
    status_note = [i + 1 for i, ln in enumerate(letter.lines) if 'not yet added' in ln]
    return [
        {'id': 'G1', 'lines': str(lno('the year in which the end-of-life floor binds')), 'kind': 'W171a finding re-checked',
         'finding': f'R3.6 now reads "(in 2035 under the baseline, the mid-block and the unit-retention calibrations; not within '
                    f'the horizon under the other two)"; T8 floor_year_070 = {fy}: consistent (L302 match). RESOLVED.'},
        {'id': 'G2', 'lines': '-', 'kind': 'W171a finding re-checked',
         'finding': f'The R3.5 clause on the 0.5 % / 2 % sentence is gone from the letter (Addendum 69 D8); the sentence itself '
                    f'is gone from main.tex (present at this commit: {main_has}; never in the submitted source: {not sub_has}). '
                    'L43v2 removed with its sentence. RESOLVED.'},
        {'id': 'G3', 'lines': str(rchanges_fig2 or lno('The framework figure (Figure~1 of the revised manuscript)')),
         'kind': 'W171a finding, unchanged', 'finding': f'R1.5 \\rchanges still says "Figure~2 and the graphical abstract" on lines '
                                                       f'{rchanges_fig2}; Addendum 69 D8: waits for the PDF the reviewers read. OPEN.'},
        {'id': 'G4', 'lines': str(lno('On Figure~15 of the submitted version')), 'kind': 'W171a finding, unchanged',
         'finding': '"Figure 15 of the submitted version" not derivable from the source (W171a G4); waits for the PDF (Addendum 69 '
                    'D8). OPEN.'},
        {'id': 'G5', 'lines': str(stray), 'kind': 'W171a finding re-checked',
         'finding': f'stray double quote after "(Section~4.5)": lines {stray}. ' + ('RESOLVED.' if not stray else 'OPEN.')},
        {'id': 'G6', 'lines': str(lno(b6)), 'kind': 'W171a finding re-checked',
         'finding': f'B.6 qualifier: R1.2 now reads "three representative years standing for five-year blocks at the '
                    f'single-scenario instance; five three-year blocks at the multi-scenario instance" ({b6 in letter.flat}); '
                    'counts checked (L301). RESOLVED.'},
        {'id': 'G7', 'lines': str(lno('The investment-cost file was corrected')), 'kind': 'W171a finding re-checked',
         'finding': f'"Further changes" names the cost-file correction (÷5 instead of ÷4, ×1.25): '
                    f'{"dividing by 5 instead of 4" in further}; numbers checked (L303). RESOLVED.'},
        {'id': 'G8', 'lines': str(status_note), 'kind': 'W171a finding, unchanged',
         'finding': f'"The three references were added" beside the bracketed status ("not yet added" on lines {status_note}); '
                    'B.8 keeps the note until the references are in. OPEN (by design).'},
        {'id': 'K1', 'lines': str(lno('It changes no sign in the 60 reported comparisons and two verdicts')), 'kind': 'claim checked',
         'finding': f'"two verdicts": T1 {[c["claim_id"] for c in vc]} and T4 2035 - 2030 (unchanged since W171a; L32v2).'},
        {'id': 'K5', 'lines': 'rcomment blocks', 'kind': 'quotation check',
         'finding': f"{sum(r['status'] == 'verbatim' for r in qrows)} of {len(qrows)} quotation segments verbatim, "
                    f"{sum(r['status'] == 'differs' for r in qrows)} differ, {sum(r['status'] == 'not found' for r in qrows)} not found."},
    ]


def main_findings_v3(X):
    doc = X.main
    heads = {h['number']: (h['title'], h['line']) for h in doc.heads if h['number']}
    labels = {}
    for i, ln in enumerate(doc.lines):
        for m in re.finditer(r'\\label\{(sec:case_[a-z_]+)\}', ln):
            labels[m.group(1)] = i + 1
    sec_of_label = {}
    for lab, ln in labels.items():
        prev = [h for h in doc.heads if h['line'] <= ln]
        sec_of_label[lab] = (prev[-1]['number'], prev[-1]['title']) if prev else None
    refs = []
    sym_line = {'\\eta': None, 'SoC^{\\text{Min}}': None, '\\varepsilon^{\\text{Cl}}': None, 'c^{\\sigma}': None,
                '\\varepsilon^{\\text{E}}': None, '\\alpha = 0.5': None}
    for sym in sym_line:
        hits = [i + 1 for i, ln in enumerate(doc.lines) if VALUE_TABLE_LINES[0] <= i + 1 <= VALUE_TABLE_LINES[1] and sym in ln and '=' in ln]
        sym_line[sym] = hits
    for i, ln in enumerate(doc.lines):
        if REVISED[0][1][0] <= i + 1 <= REVISED[0][1][1]:
            for m in re.finditer(r'given in Section~\\ref\{(sec:case_[a-z_]+)\}', ln):
                pre = ln[max(0, m.start() - 260):m.start()]
                refs.append({'line': i + 1, 'label': m.group(1), 'section': sec_of_label.get(m.group(1)), 'context': pre[-200:]})
    wrong = []
    for r in refs:
        ctx = r['context']
        names = [s for s in ('\\eta', 'SoC^{\\text{Min}}', '\\varepsilon^{\\text{Cl}}', 'c^{\\sigma}', '\\varepsilon^{\\text{E}}', '\\alpha')
                 if s in ctx]
        where = sorted({sec_of_label['sec:case_ess_params'][0] if any(h for h in sym_line.get(n if n != '\\alpha' else '\\alpha = 0.5', [])
                                                                      if h < labels.get('sec:case_settings', 10 ** 9))
                        else sec_of_label['sec:case_settings'][0] for n in names}) if names else []
        if names and r['section'] and [r['section'][0]] != where:
            wrong.append({**r, 'symbols': names, 'values_printed_in': where})
    sec34 = {k: v[0] for k, v in heads.items() if re.fullmatch(r'[34]\.\d', k)}
    lit = [(i + 1, m.group()) for i, ln in enumerate(doc.lines) if region_of(i + 1) in ('sec2', 'sec3_5_6', 'app_a')
           for m in re.finditer(r'Section~(?:3\.\d|4\.\d)', ln) if split_ok(ln, m.start())]
    return [
        {'id': 'M1', 'lines': ', '.join(str(r['line']) for r in wrong) or '-', 'kind': 'cross-reference to the wrong subsection',
         'finding': ('Section 2 says the values of ' + '; '.join(f"{', '.join(r['symbols'])} (l. {r['line']})" for r in wrong) +
                     f' are "given in Section~\\ref{{sec:case_settings}}" = {sec_of_label.get("sec:case_settings")}; they are printed '
                     f'in {sec_of_label.get("sec:case_ess_params")} (l. {sym_line}). Round 3 C mapped every former "Section~3.5" '
                     'to sec:case_settings, but section 3.5 as pasted holds these values. No number token (\\ref is excluded).')
         if wrong else 'every "given in Section~\\ref{sec:case_...}" in section 2 points at the subsection that prints the values.'},
        {'id': 'M2', 'lines': ', '.join(str(x[0]) for x in lit) or '-', 'kind': 'literal forward references',
         'finding': f'literal section numbers in sections 2, 3.5-3.6 and Appendix A: {lit}; main.tex sections 3.x and 4.x at this commit: '
                    f'{sec34} (section 4 is still the submitted '
                    'text; "Section~4.7" etc. stay literal until section 4 is rewritten, round 3 C)'},
        {'id': 'M3', 'lines': '-', 'kind': 'W171a findings re-checked',
         'finding': 'W171a M1 (Section 3.5 absent): sections 3.5 and 3.6 exist (labels ' + str(labels) + '). W171a M2 (calibrations '
                    'cited to the unrevised 3.4): now \\ref{sec:case_ess_params}. W171a M3 ("first and third turning points"): now '
                    '"the three most recent turning points" (S318 match). W171a M4 ("2n OrthoMADS", "If |P| < n + 1"): '
                    'Algorithm 1 now states variants A (n + 1, unit-box completion at Delta = 1) and B (2n, completion below '
                    'n + 1) (S304-S307, S313-S314 match). RESOLVED.'},
    ]


def split_ok(line, col):
    return W164.split_comment(line)[2] > col


# ======================================================================================================================
#  7. guards, post-run, MD
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(W161.PICKLE_COUNTS), w162=dict(W162.PICKLE_COUNTS), w163=dict(W163.PICKLE_COUNTS),
                  w164=dict(W164.PICKLE_COUNTS), w171=dict(W171.PICKLE_COUNTS), w174=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W174 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W174 post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


_cell = W171._cell


def md_summary(o):
    s = o['summary']
    L = ['# W174a -- number check of the manuscript .tex files at Overleaf 260bd83 (declarations v3) and the section 3.5-3.6 '
         'value table', '',
         f"Overleaf clone `{o['manuscript']['dir']}` at commit `{o['manuscript']['commit']}` (declared "
         f"`{o['manuscript']['declared_commit']}`); files: " +
         ', '.join(f"`{f['name']}` sha256 `{f['sha256']}`" if f['name'] == MAIN else f"`{f['name']}` sha256 `{f['sha256'][:8]}`"
                   for f in o['manuscript']['files']) + '.',
         f"Frozen tables `{os.path.basename(W164.FZ_REL)}` (sha256 `{W164.FZ_SHA[:8]}`). Script `{SCRIPT_REL}` (imports "
         f"`{W171_SCRIPT}` @ {W171_SCRIPT_COMMIT} and through it W164-W160, none edited). Submitted source "
         f"`{W171.SUBMITTED_MAIN}` (sha256 `{W171.SUBMITTED_MAIN_SHA[:8]}`); reviewers' document `{W171.DOCX}` (sha256 "
         f"`{W171.DOCX_SHA[:8]}`). ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.", '',
         'Statuses: match (declared / rule against a named record / auto-unique / auto-ambiguous), MISMATCH, approximate (section '
         '3.5-3.6 rows only: the printed figure is the source rounded at the printed precision), no table counterpart, '
         'submitted-version figure, reviewer quotation, verified, unchecked (with the reason), no source found (section 3.5-3.6 '
         'rows only). Outside sections 3.5-3.6 a figure equal to its source at the written precision is "match" (W164 rule). '
         'Excluded LaTeX structure is counted separately.', '',
         '## Counts per file -- manuscript', '']
    hdr = ('| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | '
           'match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, '
           'verified | unchecked | no source found | unassigned |')
    sep = '|---|' + '---:|' * 16

    def frow(f, c):
        return (f"| {f} | {c['tokens']} | {c['excluded']} | {c['in_scope_body']} | {c['in_scope_comment']} | "
                f"{c['match_declared']} | {c['match_rule']} | {c['match_auto_unique']} | {c['match_auto_ambiguous']} | "
                f"{c['MISMATCH']} | {c['approximate']} | {c['no table counterpart']} | {c['submitted-version figure']} | "
                f"{c['reviewer quotation, verified']} | {c['unchecked']} | {c['no source found']} | {c['unassigned']} |")
    L += [hdr, sep] + [frow(f, c) for f, c in s['per_file'].items() if f != DRAFT]
    L += ['', '## Counts -- draft, not compiled (`section2_expert_draft.tex`; apart from the manuscript counts)', '', hdr, sep] + \
        [frow(f, c) for f, c in s['per_file'].items() if f == DRAFT]
    L += ['', f"Every token assigned: **{s['every_token_assigned']}**; stale declarations: {s['stale_declarations']}; tokens "
              f"claimed twice: {s['claimed_twice']}.", '']
    vt = o['value_table']
    vc = s['value_table_status']
    L += ['## Sections 3.5-3.6 (main.tex l. 1066-1150): every value and named setting, one by one', '',
          f"Rows: {len(vt)} -- " + ', '.join(f'{k} {v}' for k, v in vc.items()) +
          f" (values with a numeric token: {s['value_table_status_values']}; named settings without a token: "
          f"{s['value_table_status_settings']}).", '',
          f"**Recorded prediction (expert, Addendum 70): every section 3.5-3.6 value matches its source.** Outcome: "
          f"{o['prediction']['outcome']}", '',
          f'Not rows: {VALUE_EXCLUDED}.', '',
          '| id | line | printed | quantity | value in force | source | status | precision / note |', '|---|---:|---|---|---|---|---|---|']
    for r in vt:
        L.append(f"| {r['id']} | {r['line']} | {_cell(r['printed'], 120)} | {_cell(r['quantity'], 90)} | "
                 f"{_cell(r['value_in_force'], 160)} | {_cell('; '.join(r['sources']), 520)} | **{r['status']}** | "
                 f"{_cell(' '.join(x for x in (r['precision'], r['note']) if x), 400)} |")
    L += ['', '### Files cited by the value table (sha256 and git blob at HEAD; code citations carry file:line and sha256)', '',
          '| path | sha256 | git blob |', '|---|---|---|']
    for v in o['value_table_inputs_cited']:
        L.append(f"| {v['path']} | {v['sha256'][:16]} | {(v['git_blob_at_HEAD'] or '')[:12]} |")
    L += ['', '### Tokens in l. 1066-1150 that are not values (assigned by rule)', '',
          '| line | written | status | reason |', '|---:|---|---|---|']
    for t in o['value_range_other_tokens']:
        L.append(f"| {t['line']} | {t['written']} | {t['status']} | {_cell(t['reason'], 200)} |")
    L += ['', '## MISMATCH (all files)', '', '| file | line | col | written | counterpart at written precision | source | note |',
          '|---|---:|---:|---|---|---|---|']
    mm = [t for t in o['tokens'] if t.get('status') == 'MISMATCH']
    for t in mm:
        L.append(f"| {t['file']}{' (draft)' if t['file'] == DRAFT else ''} | {t['line']} | {t['col']} | {t['written']} | "
                 f"{_cell(t.get('value_at_written_precision'))} | {_cell(t.get('counterpart'), 300)} | {_cell(t.get('note'), 400)} |")
    if not mm:
        L.append('| none | | | | | | |')
    L += ['', '## Approximate / no table counterpart / no source found', '']
    for t in o['tokens']:
        if t.get('status') in ('approximate', 'no table counterpart', 'no source found'):
            L.append(f"- `{t['file']}` line {t['line']}: `{t['written']}` ({t['status']}) -- {t.get('note') or ''}")
    L += ['', '## Version-2 declarations re-located (version 2 -> version 3)', '',
          '| v2 id | file | fragment found (times) | outcome | v3 declaration | reason / what replaced it |', '|---|---|---:|---|---|---|']
    for r in o['relocation']:
        L.append(f"| {r['id']} | {r['file']} | {r['hits']} | {r['outcome']} | {r.get('v3') or ''} | {_cell(r.get('reason'), 300)} |")
    L += ['', "## Reviewer quotations against the reviewers' document", '',
          f"Document: `{W171.DOCX}` sha256 `{W171.DOCX_SHA[:8]}`, word/document.xml sha256 `{o['docx']['document_xml_sha256'][:8]}`, "
          f"{o['docx']['n_paragraphs']} paragraphs (W171a normalisation).", '',
          '| quote | segment | letter lines | chars | result | match ratio | docx paragraph | start | differences (letter -> document) |',
          '|---:|---:|---|---:|---|---:|---:|---|---|']
    for r in o['quotations']:
        d = '; '.join(f"{x['op']} l.{x['letter_line']}: ...{x['context_before'][-20:]}[{x['letter']!r} -> {x['document']!r}]"
                      f"{x['context_after'][:20]}..." for x in r['diffs'][:6])
        L.append(f"| {r['quote']} | {r['segment']} | {r['lines']} | {r['chars']} | {r['status']} | {r['ratio']:.3f} | "
                 f"{r['docx_paragraph']} | {_cell(r['start'], 50)} | {_cell(d, 600)} |")
    L += ['', f"Quotation tokens: {s['quotation_tokens']}", '']
    for key, (a, b), desc, _h in REVISED:
        if key == 'sec3_5_6':
            continue
        L += [f'## {desc} (main.tex l. {a}-{b}): every number with its source', '',
              '| line | written | status | check | counterpart (at written precision) | source / reason |', '|---:|---|---|---|---|---|']
        for t in o['tokens']:
            if t['file'] == MAIN and a <= t['line'] <= b and not t.get('excluded'):
                L.append(f"| {t['line']} | {t['written']} | {t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                         f"{_cell(t.get('value_at_written_precision'), 50)} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
        L.append('')
    L += ['## main.tex outside the revised passages: tokens on lines changed since W171a (42794d4)', '',
          '| line | written | status | check | source / reason |', '|---:|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == MAIN and not t.get('excluded') and t['line'] in o['main_changed_lines'] and region_of(t['line']) is None:
            L.append(f"| {t['line']} | {t['written']} | {t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                     f"{_cell(t.get('counterpart') or t.get('note'), 260)} |")
    L += ['', '## Response letter: every number with its source', '',
          '| line | written | scope | status | check | counterpart (at written precision) | source / reason |', '|---:|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == LETTER and not t.get('excluded'):
            L.append(f"| {t['line']} | {t['written']} | {t['scope']}{'/' + t['macro'] if t.get('macro') else ''} | "
                     f"{t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                     f"{_cell(t.get('value_at_written_precision'), 60)} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
    L += ['', '## Findings -- letter', '']
    for f in o['letter_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    L += ['', '## Findings -- main.tex', '']
    for f in o['main_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    for fname in (HIGHLIGHTS, COVER):
        L += ['', f'## {fname}: every number with its source', '']
        rows = [t for t in o['tokens'] if t['file'] == fname and not t.get('excluded')]
        for t in rows:
            L.append(f"- line {t['line']} `{t['written']}`: {t['status_display']} -- {t.get('counterpart') or t.get('note') or ''}")
        if not rows:
            L.append('no in-scope numeric token')
    L += ['', '## main.tex line rules re-pinned to cb237b2d', '',
          f"Revised passages (declared or ruled; never the auto index): " +
          '; '.join(f'{k} l. {a}-{b}' for k, (a, b), _d, _h in REVISED) +
          f". Network tables l. {NETWORK_TABLES[0]}-{NETWORK_TABLES[1]}; Appendices B-D l. {APP_BCD[0]}-{APP_BCD[1]}.", '',
          '| W171a region (l.) | key | kind | 260bd83 (l.) |', '|---|---|---|---|']
    for r in o['main_regions']['kept']:
        L.append(f"| {r['old'][0]}-{r['old'][1]} | {r['key']} | {r['kind']} | {r['new'][0]}-{r['new'][1]} |")
    for r in o['main_regions']['dropped']:
        L.append(f"| {r['old'][0]}-{r['old'][1]} | {r['key']} | dropped ({r['n_lines_changed']} lines changed) | {_cell(r['reason'], 200)} |")
    L += ['', '## main.tex: submitted-version figures by section', '', '| section | n | map lines cited |', '|---|---:|---|']
    for sec, v in s['main_submitted_by_section'].items():
        L.append(f"| {sec} | {v['n']} | {', '.join(v['cites'])} |")
    L += ['', '## W171a parameter checks re-evaluated on this run (the Addendum 68 list, source of several value-table rows)', '',
          '| id | quantity | found | status | sources |', '|---|---|---|---|---|']
    for p in o['parameter_table']:
        L.append(f"| {p['id']} | {_cell(p['quantity'])} | {_cell(p['found'], 250)} | {p['status']} | {_cell('; '.join(p['sources']), 400)} |")
    L += ['', '## Unchecked: by category (all files)', '', '| file | category | n |', '|---|---|---:|']
    for f, cats in s['unchecked_by_category'].items():
        for c, n in cats.items():
            L.append(f'| {f} | {c} | {n} |')
    L += ['', '## Excluded LaTeX structure', '', '| file | category | n | tokens |', '|---|---|---:|---|']
    for f, cats in s['excluded_by_category'].items():
        for c, v in cats.items():
            L.append(f"| {f} | {c} | {v['n']} | {', '.join(v['examples'])} |")
    L += ['', '## Integrity checks', '']
    for k, v in o['checks'].items():
        L.append(f'- {k}: {v}')
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  8. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manuscript-dir', default=DEFAULT_MANUSCRIPT_DIR)
    ap.add_argument('--overleaf-commit', required=True)
    ap.add_argument('--expect', action='append', default=[], help='NAME=SHA256 for every *.tex file (declared)')
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or os.path.join(OUT_ROOT, f'overleaf_{args.overleaf_commit[:7]}')
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W174'
    pre = []
    od = os.path.join(REPO, out_dir) if not os.path.isabs(out_dir) else out_dir
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        extra = sorted(set(os.listdir(od)) - {OUT_NAMES['log']})
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
    if not re.fullmatch(r'[0-9a-f]{7,40}', args.overleaf_commit):
        pre.append(f'--overleaf-commit {args.overleaf_commit!r} is not a hex commit')
    mdir = args.manuscript_dir
    mabs = os.path.join(REPO, mdir)
    expect = {}
    for e in args.expect:
        if '=' not in e:
            pre.append(f'--expect {e!r}: NAME=SHA256')
            continue
        k, v = e.split('=', 1)
        expect[k] = v
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mabs, '*.tex')))
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex'], capture_output=True, text=True).stdout.strip()
    mfiles = []
    for f in files:
        with open(os.path.join(mabs, f), 'rb') as h:
            b = h.read()
        blob = subprocess.run(['git', '-C', mabs, 'show', f'{args.overleaf_commit}:{f}'], capture_output=True)
        mfiles.append({'name': f, 'path': os.path.join(mdir, f), 'sha256': _sha_bytes(b), 'bytes': len(b),
                       'lines': b.decode('utf-8').count('\n') + 1,
                       'blob_at_commit_sha256': _sha_bytes(blob.stdout) if blob.returncode == 0 else None,
                       'declared_sha256': expect.get(f), 'scope': 'draft, not compiled' if f == DRAFT else 'manuscript'})
    old_main = subprocess.run(['git', '-C', mabs, 'show', f'{W171_COMMIT_IN_CLONE}:{MAIN}'], capture_output=True)
    old_main_sha = _sha_bytes(old_main.stdout) if old_main.returncode == 0 else None
    mchecks = {
        'clone_head_is_declared_commit': bool(head) and head.startswith(args.overleaf_commit),
        'file_set_equals_declared': set(files) == set(expect),
        'every_sha_equals_declared': all(m['sha256'] == m['declared_sha256'] for m in mfiles),
        'every_file_equals_its_blob_at_commit': all(m['sha256'] == m['blob_at_commit_sha256'] for m in mfiles),
        'no_tex_modified_or_untracked_in_clone': porcelain == '',
        'w171a_pin_readable_in_clone': old_main_sha == W171.MAIN_LINE_RULES_SHA,
    }
    for k, v in mchecks.items():
        if not v:
            pre.append(f'manuscript: {k} FAILED (head {head}, files {files}, declared {sorted(expect)}, porcelain {porcelain!r}, '
                       f'shas {[(m["name"], m["sha256"][:8], (m["declared_sha256"] or "")[:8]) for m in mfiles]}, w171a pin {old_main_sha})')
    input_rels = {'FROZEN_JSON': W164.FZ_REL, 'FROZEN_MD': W164.FZ_MD_REL, 'W163_JSON': W164.W163_JSON,
                  'W163_MANIFEST': W164.W163_MAN, 'W163_SCRIPT': W164.W163_SCRIPT, 'W164_SCRIPT': W171.W164_SCRIPT,
                  'W164_RESULTS': W171.W164_RESULTS, 'P5': W164.P5_REL, 'P2': W164.P2_REL, 'MAP': W164.MAP_REL,
                  'SRP1_JSON': W164.SRP1_JSON, 'SRP1_PARAMS': W171.SRP1_PARAMS, 'ESS_PARAMS': W171.ESS_PARAMS,
                  'STEP4': W164.STEP4, 'A1_CAMPAIGN': W164.A1_CAMPAIGN, 'V41_SPEC': W164.V41_SPEC, 'V6_SPEC': W171.V6_SPEC,
                  'SPEC_3X3': W171.SPEC_3X3, 'INST_3X3': W171.INST_3X3, 'EXT_SPEC_V3': W171.EXT_SPEC_V3,
                  'SUBMITTED_MAIN': W171.SUBMITTED_MAIN, 'DOCX': W171.DOCX, 'CORR_MD': W171.CORR_MD}
    for k, v in W164.CASE_FILES.items():
        input_rels[f'CASE_{k}'] = v
    for k, v in W171.CASE_PARAMS.items():
        input_rels[f'CP_{k}'] = v
    for k, v in {**W171.CODE, **CODE_V3}.items():
        input_rels[f'CODE_{k}'] = v
    input_rels.update(EXTRA_INPUTS)
    input_rels.update(DN_YEAR_FILES)
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel), 'blob': _git('rev-parse', f'HEAD:{rel}')}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if not pre:
        for key, want in (('FROZEN_JSON', W164.FZ_SHA), ('P5', W164.P5_SHA), ('P2', W164.P2_SHA),
                          ('SUBMITTED_MAIN', W171.SUBMITTED_MAIN_SHA), ('DOCX', W171.DOCX_SHA)):
            if inputs[key]['sha256'] != want:
                pre.append(f"{inputs[key]['path']} sha {inputs[key]['sha256']} != {want}")
        if _jl(W164.W163_MAN).get(W164.W163_JSON) != inputs['W163_JSON']['sha256']:
            pre.append(f'{W164.W163_JSON} != its W163 manifest entry')
        for key, want in (('W164_SCRIPT', W171.W164_SCRIPT_COMMIT), ('W163_SCRIPT', W164.W163_SCRIPT_COMMIT),
                          ('W171_SCRIPT', W171_SCRIPT_COMMIT), ('W171_RESULTS', W171_RESULTS_COMMIT)):
            if not (inputs[key]['last_commit'] or '').startswith(want):
                pre.append(f"{inputs[key]['path']} last commit {inputs[key]['last_commit']} != {want}")
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; {len(inputs)} inputs committed '
         f'clean; frozen JSON {W164.FZ_SHA[:8]}, submitted main.tex {W171.SUBMITTED_MAIN_SHA[:8]}, reviewers\' docx '
         f'{W171.DOCX_SHA[:8]} verified; manuscript {mdir} HEAD {head} (declared {args.overleaf_commit}); files ' +
         ', '.join(f"{m['name']} {m['sha256'][:8]}" for m in mfiles) + f'; W171a pin {W171_COMMIT_IN_CLONE}:main.tex {old_main_sha[:8]}')
    # ---- documents, tokens, records ----------------------------------------------------------------------------------
    docs = {m['name']: W164.Doc(m['name'], _text(m['path'])) for m in mfiles}
    roles = {n: ('main' if n == MAIN else 'letter' if n.startswith('response_to_reviewers') else 'highlights' if n == HIGHLIGHTS
                 else 'cover' if n == COVER else 'draft' if n == DRAFT else 'other') for n in docs}
    toks = []
    for n in files:
        toks += tokenize_v3(docs[n])
    sub_doc = W164.Doc('main.tex', _text(W171.SUBMITTED_MAIN))
    X = RecordsV3(inputs, docs.get(MAIN), sub_doc)
    X.letter_doc = docs.get(LETTER)
    X.letter_raw = docs[LETTER].raw if LETTER in docs else ''
    X.cover_raw = docs[COVER].raw if COVER in docs else ''
    X.main_tokens = [t for t in toks if t['file'] == MAIN and t['scope'] == 'body' and not t['excluded']]
    X.sub_tokens = [t for t in W164.tokenize(sub_doc) if t['scope'] == 'body' and not t['excluded']]
    X.main_sha = next((m['sha256'] for m in mfiles if m['name'] == MAIN), None)
    X.corr_text = _text(W171.CORR_MD)
    old_lines = old_main.stdout.decode('utf-8').split('\n')
    kept, dropped, lm = map_regions(old_lines, docs[MAIN].lines)
    X.main_regions_v3 = kept
    changed_new = sorted(set(range(1, len(docs[MAIN].lines) + 1)) - set(lm.values()))
    region_heads_ok = {k: docs[MAIN].lines[a - 1].strip().startswith(h) for k, (a, b), _d, h in REVISED}
    try:
        ptable = W171.parameter_checks(X)
        ptable_error = None
    except Exception as e:  # a parameter whose source moved fails the run, loudly
        ptable, ptable_error = [], f'{type(e).__name__}: {e}'
    X.ptab = {p['id']: p for p in ptable}
    leaves = W164.json_leaves(X.fz)
    AX = W164.AutoIndex(leaves)
    map_cite_lines, map_cite_fail = {}, []
    for k, frag in W164.MAP_CITES.items():
        try:
            map_cite_lines[k] = X.map_line(frag)
        except KeyError as e:
            map_cite_fail.append(str(e))
    map_xref_numbers = set(re.findall(r'(?<![\d.])(\d\.\d(?:\.\d)?)(?![\d.])', '\n'.join(X.map)))
    # ---- reviewer quotations -------------------------------------------------------------------------------------------
    dx = W171.docx_text(W171.DOCX)
    qrows = W171.quotation_check(docs[LETTER], dx)
    qstat = W171.quotation_token_status(qrows, docs[LETTER], toks)
    # ---- the version-2 declaration set, reconstructed and re-located ----------------------------------------------------
    w171 = X.j['W171_RESULTS']
    v1 = W164.letter_declarations() + W164.cover_declarations() + W164.main_declarations()
    v1_carried = {r['id'] for r in w171['relocation'] if r['outcome'] == 'carried'}
    v2 = W171.letter_declarations_v2() + [d for d in v1 if d['id'] in v1_carried] + \
        [d for d in W171.all_declarations_v2() if d['file'] != LETTER and not re.fullmatch(r'(L|CL|M)\d+', d['id'])]
    v2_ids_applied = [d['id'] for d in w171['declarations_applied']]
    v2_reconstructed = sorted(d['id'] for d in v2) == sorted(v2_ids_applied)
    relocation = []
    for d in v2:
        doc = docs.get(d['file'])
        hits = len(doc.find_fragment(d['fragment'])[0]) if doc else 0
        if d['id'] in SUPERSEDED_V3:
            outcome, v3, why = 'superseded', SUPERSEDED_V3[d['id']][0], SUPERSEDED_V3[d['id']][1]
        elif hits == 1:
            outcome, v3, why = 'carried', d['id'], 'fragment found once; evaluated unchanged'
        elif d['id'] in REPLACED_BY_V3:
            outcome, v3, why = 'removed', REPLACED_BY_V3[d['id']][0], REPLACED_BY_V3[d['id']][1]
        else:
            outcome, v3, why = 'removed (NO REPLACEMENT DECLARED)', None, f'fragment found {hits} times and no v3 entry names it'
        relocation.append({'id': d['id'], 'file': d['file'], 'fragment': d['fragment'], 'hits': hits, 'outcome': outcome,
                           'v3': v3, 'reason': why})
    carried_ids = {r['id'] for r in relocation if r['outcome'] == 'carried'}
    unreplaced = [r['id'] for r in relocation if r['outcome'].startswith('removed (NO')]
    new_decls = all_declarations_v3()
    decls = [d for d in v2 if d['id'] in carried_ids] + new_decls
    value_ids = {d['id'] for d in new_decls if d.get('value_table')}
    seen_ids = collections.Counter(d['id'] for d in decls)
    dup_ids = sorted(k for k, n in seen_ids.items() if n > 1)
    # ---- declared checks -----------------------------------------------------------------------------------------------
    by_key = {t['key']: t for t in toks}
    assigned, claimed_twice, stale, decl_out = {}, [], [], []
    for d in decls:
        if d['file'] not in docs:
            stale.append({'id': d['id'], 'why': f"{d['file']} not in the manuscript directory"})
            continue
        doc = docs[d['file']]
        hits, flen = doc.find_fragment(d['fragment'])
        if len(hits) != 1:
            stale.append({'id': d['id'], 'why': f'fragment found {len(hits)} times', 'fragment': d['fragment']})
            continue
        a = hits[0]
        inside = [t for t in toks if t['file'] == d['file'] and not t['excluded'] and doc.flat_pos(t['line'], t['col0']) is not None
                  and a <= doc.flat_pos(t['line'], t['col0']) < a + flen]
        picked, ok = [], True
        for txt, occ in d['tokens']:
            c = [t for t in inside if t['text'] == txt]
            if len(c) <= occ:
                ok = False
                stale.append({'id': d['id'], 'why': f'token {txt!r}#{occ} not in the fragment'})
                break
            picked.append(c[occ])
        if not ok:
            continue
        try:
            res = d['fn'](X, [t['written'] for t in picked])
        except Exception as e:  # a declaration that cannot evaluate is stale, loudly
            stale.append({'id': d['id'], 'why': f'evaluation error {type(e).__name__}: {e}'})
            continue
        if isinstance(res, dict):
            res = [res] * len(picked)
        if len(res) != len(picked):
            stale.append({'id': d['id'], 'why': f'{len(res)} results for {len(picked)} tokens'})
            continue
        fl = doc.flat_idx[a]
        decl_out.append({'id': d['id'], 'file': d['file'], 'fragment': d['fragment'], 'line': fl[0],
                         'version': 2 if d['id'] in carried_ids else 3,
                         'tokens': [[t['line'], t['col'], t['written']] for t in picked]})
        for t, r in zip(picked, res):
            if t['key'] in assigned:
                claimed_twice.append([t['key'], assigned[t['key']]['check_id'], d['id']])
                continue
            r = dict(r)
            r['check_id'] = d['id']
            r['rule'] = 'declared'
            assigned[t['key']] = r
    # ---- rules, auto index -----------------------------------------------------------------------------------------------
    for t in toks:
        t['auto_candidates'] = AX.candidates(t) if not t['excluded'] else None
        if t['excluded']:
            t['status'], t['status_display'] = 'excluded', f"excluded ({t['excluded']})"
            continue
        reg = region_of(t['line']) if t['file'] == MAIN else None
        r = assigned.get(t['key'])
        if r is None:
            r = rule_assign_v3(X, docs[t['file']], roles[t['file']], t, map_xref_numbers, qstat)
            if r is not None:
                r = dict(r)
                r['rule'] = 'rule'
        if r is not None and r['status'] == 'match' and r['rule'] == 'rule':
            r['match_kind'] = 'rule (named record)'
        if r is None and roles[t['file']] == 'main' and reg is None and t['auto_candidates']:
            c = t['auto_candidates']
            r = R('match', c[0]['path'] if len(c) == 1 else f'{len(c)} paths', 'frozen JSON (auto index)', None, None, None)
            r['rule'], r['match_kind'] = 'auto', 'auto-unique' if len(c) == 1 else 'auto-ambiguous'
        if r is None and roles[t['file']] == 'main' and reg is None and X.main_sha == MAIN_LINE_RULES_SHA:
            r = ntc('auto index: no candidate; no declaration or rule this round (submitted text outside the revised passages)')
            r['rule'] = 'leftover'
        if r is None:
            t['status'], t['status_display'] = None, 'UNASSIGNED'
            continue
        t.update({k: v for k, v in r.items()})
        if t['status'] == 'match' and 'match_kind' not in r:
            t['match_kind'] = 'declared'
        mk = t.get('match_kind')
        t['status_display'] = (f"match (auto, {t['counterpart']})" if mk == 'auto-unique' else
                               f"match (auto, ambiguous: {len(t['auto_candidates'])} paths)" if mk == 'auto-ambiguous'
                               else "MISMATCH vs reviewers' document" if t.get('mismatch_vs') else t['status'])
    unassigned = [t['key'] for t in toks if not t['excluded'] and t['status'] is None]
    # ---- the value table -------------------------------------------------------------------------------------------------
    vtable = []
    claimed_value_keys = set()
    for d in decl_out:
        if d['id'] not in value_ids:
            continue
        for k, r in enumerate(X.value_rows.get(d['id'], [])):
            tl = [d['tokens'][i] for i in r['tok']]
            for i in r['tok']:
                claimed_value_keys.add(f"{MAIN}:{d['tokens'][i][0]}:{d['tokens'][i][1]}")
            vtable.append({'id': f"{d['id']}.{k + 1}", 'declaration': d['id'], 'line': tl[0][0] if tl else d['line'],
                           'tokens': tl, 'printed': r['printed'], 'quantity': r['quantity'],
                           'value_in_force': r['value_in_force'], 'sources': r['sources'], 'status': r['status'],
                           'precision': r['precision'], 'note': r['note'], 'kind': 'value' if tl else 'named setting'})
    vtable.sort(key=lambda r: (r['line'], r['id']))
    value_range_other = [{'line': t['line'], 'written': t['written'], 'status': t['status_display'],
                          'reason': t.get('note') or t.get('counterpart') or t.get('category')}
                         for t in toks if t['file'] == MAIN and VALUE_TABLE_LINES[0] <= t['line'] <= VALUE_TABLE_LINES[1]
                         and not t['excluded'] and t['key'] not in claimed_value_keys]
    vt_inputs = sorted({(v['path'], v['sha256'], v['blob']) for v in inputs.values()
                        if any(v['path'] in src for r in vtable for src in r['sources'])})
    vstat = collections.Counter(r['status'] for r in vtable)
    vstat_values = collections.Counter(r['status'] for r in vtable if r['kind'] == 'value')
    vstat_settings = collections.Counter(r['status'] for r in vtable if r['kind'] == 'named setting')
    bad_values = [r['id'] for r in vtable if r['kind'] == 'value' and r['status'] in ('MISMATCH', 'no source found')]
    bad_settings = [r['id'] for r in vtable if r['kind'] == 'named setting' and r['status'] in ('MISMATCH', 'no source found')]
    prediction = {
        'statement': 'every section 3.5-3.6 value matches its source (expert, Addendum 70)',
        'rows_by_status': dict(vstat), 'value_rows_by_status': dict(vstat_values), 'setting_rows_by_status': dict(vstat_settings),
        'value_rows_not_matching': bad_values, 'setting_rows_not_matching': bad_settings,
        'held_on_values': not bad_values,
        'outcome': (f"{'HELD' if not bad_values else 'FAILED'} on the values: {sum(vstat_values.values())} value rows -- "
                    + ', '.join(f'{k} {v}' for k, v in vstat_values.items()) +
                    ' (approximate = the source rounded at the printed precision); named settings: ' +
                    ', '.join(f'{k} {v}' for k, v in vstat_settings.items()) +
                    (f'; not matching: values {bad_values}, settings {bad_settings}' if bad_values or bad_settings else ''))}
    # ---- summary -----------------------------------------------------------------------------------------------------------
    per_file = {}
    for n in files:
        ft = [t for t in toks if t['file'] == n]
        sc = [t for t in ft if not t['excluded']]
        c = {'scope': 'draft, not compiled' if n == DRAFT else 'manuscript', 'tokens': len(ft), 'excluded': len(ft) - len(sc),
             'in_scope_body': sum(t['scope'] == 'body' for t in sc), 'in_scope_comment': sum(t['scope'] == 'comment' for t in sc),
             'match_declared': sum(t['status'] == 'match' and t.get('match_kind') == 'declared' for t in sc),
             'match_rule': sum(t['status'] == 'match' and t.get('match_kind') == 'rule (named record)' for t in sc),
             'match_auto_unique': sum(t.get('match_kind') == 'auto-unique' for t in sc),
             'match_auto_ambiguous': sum(t.get('match_kind') == 'auto-ambiguous' for t in sc),
             'unassigned': sum(t['status'] is None for t in sc)}
        for st in STATUSES[1:]:
            c[st] = sum(t['status'] == st for t in sc)
        c['MISMATCH_vs_reviewers_document'] = sum(bool(t.get('mismatch_vs')) for t in sc)
        c['match_total'] = c['match_declared'] + c['match_rule'] + c['match_auto_unique'] + c['match_auto_ambiguous']
        c['assigned_total'] = c['match_total'] + sum(c[st] for st in STATUSES[1:])
        per_file[n] = c
    sub_by_sec, stat_by_sec = collections.OrderedDict(), collections.OrderedDict()
    for t in toks:
        if t['file'] != MAIN or t['excluded']:
            continue
        sec = t['section'].split(' > ')[0] if t['scope'] == 'body' else '(comments)'
        stat_by_sec.setdefault(sec, collections.Counter())[f"match ({t['match_kind']})" if t['status'] == 'match' else t['status_display']] += 1
        if t['status'] == 'submitted-version figure':
            v = sub_by_sec.setdefault(sec, {'n': 0, 'cites': []})
            v['n'] += 1
            if t.get('map_cite') and t['map_cite'].split(':')[1] not in v['cites']:
                v['cites'].append(t['map_cite'].split(':')[1])
    unc_cat, exc_cat = {}, {}
    for t in toks:
        if t['excluded']:
            e = exc_cat.setdefault(t['file'], {}).setdefault(t['excluded'], {'n': 0, 'examples': []})
            e['n'] += 1
            if len(e['examples']) < 8 and t['text'] not in e['examples']:
                e['examples'].append(t['text'])
        elif t['status'] == 'unchecked':
            unc_cat.setdefault(t['file'], collections.Counter())[t.get('category') or 'other'] += 1
    overridden = collections.Counter(t['status'] or 'UNASSIGNED' for t in toks if not t['excluded'] and t['auto_candidates']
                                     and not t.get('match_kind', '').startswith('auto'))
    qtok = collections.Counter(t['status_display'] for t in toks if t['file'] == LETTER and t.get('macro') == 'rcomment' and not t['excluded'])
    summary = {'per_file': per_file, 'every_token_assigned': not unassigned, 'unassigned': unassigned,
               'stale_declarations': stale, 'claimed_twice': claimed_twice, 'duplicate_declaration_ids': dup_ids,
               'main_submitted_by_section': sub_by_sec, 'main_status_by_section': {k: dict(v) for k, v in stat_by_sec.items()},
               'unchecked_by_category': {k: dict(v) for k, v in unc_cat.items()}, 'excluded_by_category': exc_cat,
               'auto_candidates_overridden': dict(overridden), 'n_frozen_json_numeric_leaves': len(leaves),
               'n_declarations': len(decls), 'n_declarations_applied': len(decl_out), 'quotation_tokens': dict(qtok),
               'quotation_segments': dict(collections.Counter(r['status'] for r in qrows)),
               'relocation': dict(collections.Counter(r['outcome'] for r in relocation)),
               'parameter_table': dict(collections.Counter(p['status'] for p in ptable)),
               'value_table_status': dict(vstat), 'value_table_status_values': dict(vstat_values),
               'value_table_status_settings': dict(vstat_settings), 'n_value_rows': len(vtable),
               'extra_word_tokens': sum(1 for t in toks if t.get('tokenizer') == 'v3 extra number word')}
    lf = letter_findings_v3(X, docs, qrows)
    mf = main_findings_v3(X)
    checks = dict(mchecks)
    checks.update({
        'frozen_json_unchanged': _sha(W164.FZ_REL) == W164.FZ_SHA,
        'v2_declaration_set_reconstructed_equals_w171a_applied': v2_reconstructed,
        'every_declaration_found_and_evaluated': not stale,
        'no_token_claimed_twice': not claimed_twice,
        'no_duplicate_declaration_id': not dup_ids,
        'every_token_assigned': not unassigned,
        'every_map_citation_found_once': not map_cite_fail,
        'statuses_in_vocabulary': all(t['excluded'] or t['status'] in STATUSES for t in toks if t['status']),
        'main_line_rules_pinned_to_this_main': X.main_sha == MAIN_LINE_RULES_SHA,
        'revised_passages_start_at_their_headings': all(region_heads_ok.values()),
        'w171a_regions_kept_or_dropped_with_reason': all(not d_['reason'].startswith('TEXT CHANGED') for d_ in dropped),
        'w171a_regions_equal_pinned_expectation': EXPECTED_MAIN_REGIONS is None or
        [list(r[0]) for r in kept] == [list(r) for r in EXPECTED_MAIN_REGIONS],
        'every_v2_declaration_carried_superseded_or_replaced': not unreplaced,
        'parameter_table_evaluated': ptable_error is None and bool(ptable),
        'every_value_declaration_applied': value_ids <= {d['id'] for d in decl_out},
        'every_value_row_status_in_vocabulary': all(r['status'] in ROW_STATUSES for r in vtable),
        'every_quotation_segment_located': all(r['status'] != 'not found' for r in qrows),
        'docx_sha_equals_declared': _sha(W171.DOCX) == W171.DOCX_SHA,
    })
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
        if t.get('auto_candidates') is not None:
            t['auto_candidates_n'] = len(t['auto_candidates'])
            t['auto_candidates'] = t['auto_candidates'][:10]
    for r in qrows:
        r.pop('offsets', None)
    out = {'schema': 'p515_s53_w174_manuscript_number_check', 'version': 1, 'declarations_version': DECLARATIONS_VERSION,
           'stage': 'P5.15 Step 6 W174a -- number check of the manuscript .tex files at Overleaf 260bd83 (declarations v3) and '
                    'the section 3.5-3.6 value table',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {W171_SCRIPT: _sha(W171_SCRIPT)}},
           'command_line': sys.argv,
           'manuscript': {'dir': mdir, 'commit': head, 'declared_commit': args.overleaf_commit, 'files': mfiles, 'roles': roles,
                          'clone_porcelain_tex': porcelain, 'edited': False,
                          'w171a_pin': {'commit': W171_COMMIT_IN_CLONE, 'main_sha256': old_main_sha}},
           'docx': {k: v for k, v in dx.items() if k not in ('text', 'para_of')},
           'frozen_tables': {'path': W164.FZ_REL, 'sha256': W164.FZ_SHA, 'numeric_leaves': len(leaves)},
           'inputs': inputs, 'map_citations': map_cite_lines, 'map_citation_failures': map_cite_fail,
           'main_regions': {'sha256': MAIN_LINE_RULES_SHA, 'revised': [{'key': k, 'lines': list(r), 'what': d_, 'heading_ok': region_heads_ok[k]}
                                                                         for k, r, d_, _h in REVISED],
                            'kept': [{'old': list(o_[0]), 'new': list(n_[0]), 'key': n_[1], 'kind': n_[2], 'why': n_[3]}
                                     for o_, n_ in zip([r for r in W171.MAIN_REGIONS if r[1] in {k[1] for k in kept}], kept)],
                            'dropped': dropped, 'network_tables': NETWORK_TABLES, 'appendix_BCD': APP_BCD,
                            'value_table_lines': VALUE_TABLE_LINES},
           'main_changed_lines': changed_new,
           'conventions': {
               'column': '1-based', 'line': '1-based',
               'written': 'the token as written, sign included, LaTeX thousands separators as ","',
               'precedence': 'excluded structure -> declared checks (v2 carried, v3) -> rules (comments; re-pinned submitted-version '
                             'regions; identifiers; cross-references; compound / article number words; revised passages: '
                             'structural number words and formula constants, everything else must be declared; outside: '
                             'years, IEEE n-bus, nodes, algorithms, math, data tables, number words) -> auto index (main.tex '
                             'outside the revised passages) -> leftover. A revised-passage token left unassigned fails the run.',
               'value_table': 'rows per value (numeric token) and per named setting; match = equal to the source; approximate = '
                              'the source rounded at the printed precision (stated); MISMATCH; no source found (where looked).',
               'objective_convention': X.fz.get('objective_convention')},
           'summary': summary, 'checks': checks, 'failed': failed, 'relocation': relocation, 'declarations_applied': decl_out,
           'value_table': vtable, 'value_range_other_tokens': value_range_other, 'value_table_not_rows': VALUE_EXCLUDED,
           'value_table_inputs_cited': [{'path': a, 'sha256': b, 'git_blob_at_HEAD': c} for a, b, c in vt_inputs],
           'prediction': prediction, 'quotations': qrows, 'parameter_table': ptable, 'parameter_table_error': ptable_error,
           'letter_findings': lf, 'main_findings': mf, 'tokens': toks, 'guards': guards, 'pickle_guard': pk,
           'exit_code': code, 'wall_s': time.time() - t0}
    written = {}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel) if not os.path.isabs(rel) else _sha_bytes(open(rel, 'rb').read())
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    for m in mfiles:
        man[m['path']] = m['sha256']
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for n, c in per_file.items():
        _log(f"[{tag}] {n} ({c['scope']}): tokens {c['tokens']} (excluded {c['excluded']}, in scope {c['in_scope_body']} body + "
             f"{c['in_scope_comment']} comment) -- match {c['match_total']} (declared {c['match_declared']}, rule {c['match_rule']}, "
             f"auto unique {c['match_auto_unique']}, auto ambiguous {c['match_auto_ambiguous']}), MISMATCH {c['MISMATCH']} (vs "
             f"reviewers' document {c['MISMATCH_vs_reviewers_document']}), approximate {c['approximate']}, no table counterpart "
             f"{c['no table counterpart']}, submitted-version {c['submitted-version figure']}, reviewer quotation verified "
             f"{c['reviewer quotation, verified']}, unchecked {c['unchecked']}, no source found {c['no source found']}, "
             f"unassigned {c['unassigned']}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no table counterpart', 'no source found') and \
                (t['file'] != MAIN or t.get('rule') == 'declared'):
            _log(f"[{tag}]   {t['file']}:{t['line']}:{t['col']} {t['status_display']}: written {t['written']!r} -> "
                 f"{t.get('value_at_written_precision')!r} [{t.get('check_id')}] {str(t.get('counterpart'))[:200]}")
    _log(f"[{tag}] relocation of v2 declarations: {summary['relocation']}; " +
         '; '.join(f"{r['id']} {r['outcome']} -> {r['v3']}" for r in relocation if r['outcome'] != 'carried'))
    _log(f"[{tag}] quotation segments: {summary['quotation_segments']}; quotation tokens: {summary['quotation_tokens']}")
    for r in qrows:
        if r['status'] != 'verbatim':
            _log(f"[{tag}]   quotation {r['quote']}.{r['segment']} (l. {r['lines']}) {r['status']} ratio {r['ratio']:.3f}: " +
                 '; '.join(f"{x['op']} {x['letter']!r} -> {x['document']!r}" for x in r['diffs'][:6]))
    _log(f"[{tag}] value table (l. 1066-1150): {len(vtable)} rows {dict(vstat)}; prediction: {prediction['outcome']}")
    for r in vtable:
        if r['status'] != 'match':
            _log(f"[{tag}]   {r['id']} l. {r['line']} {r['quantity']}: {r['status']} -- printed {r['printed']!r}, in force "
                 f"{str(r['value_in_force'])[:120]}; {r['precision'] or ''} {(r['note'] or '')[:200]}")
    _log(f"[{tag}] parameter checks re-evaluated: {summary['parameter_table']}" + (f' ERROR {ptable_error}' if ptable_error else ''))
    _log(f"[{tag}] regions kept {[(r[1], r[0]) for r in kept]}; dropped {[(d_['key'], d_['old']) for d_ in dropped]}")
    for f in lf + mf:
        _log(f"[{tag}]   finding {f['id']} (l. {f['lines']}, {f['kind']}): {f['finding'][:300]}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if unassigned:
        _log(f'[{tag}] UNASSIGNED tokens ({len(unassigned)}): ' + ', '.join(f"{by_key[k]['key']} {by_key[k]['written']!r}"
                                                                          for k in unassigned[:400]))
    if stale or claimed_twice or map_cite_fail or dup_ids or unreplaced:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; map citation failures {map_cite_fail}; duplicate ids '
             f'{dup_ids}; unreplaced v2 {unreplaced}')
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
