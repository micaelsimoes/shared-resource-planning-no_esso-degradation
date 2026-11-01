"""P5.15 Step 6, Planner task W164 -- NUMBER CHECK OF THE MANUSCRIPT .tex FILES (the W163 checker extended to the
Overleaf clone) AGAINST THE FROZEN TABLES AND NAMED COMMITTED RECORDS. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the
W163 script is IMPORTED (it imports W162 -> W161 -> W160, each arming its guards) and its helpers and committed check
registry are USED; W160-W163 are not edited. Every output is a new file opened 'x' in a new directory.

RE-RUN EVERY REVIEW ROUND (STEP6_REVISION_MAP.md section E). The input is a manuscript directory (default: the Overleaf
clone `manuscript/6a67305f25e8348fb71380c3/`, its own git repository, untracked by this repository, never modified
here). The run DECLARES the Overleaf commit and the sha256 of every `*.tex` file (glob, so new files are picked up);
it refuses to run when the clone's HEAD, the file set or any sha differs from the declaration, when a tracked .tex is
modified in the clone, or when a file differs from its blob at the declared commit. Outputs go to
`data/SRP1/Results/P515S53/w164_manuscript_check/overleaf_<commit7>/`, created by the launch command (`mkdir`, which
fails on an existing directory); the script refuses to write if that directory holds anything but the launch log.

WHAT IT DOES
  1. Tokenizes every numeric token of every .tex file, LaTeX-aware: `\\,` / `{,}` / `,` thousands, `\\%`, `~`, `$...$`
     and display math, `\\times`, `10^{-5}`, ranges `0.909--0.934` (two tokens), `k\\euro{}` / `M\\euro{}` units,
     number words (W162's list + twelve, twenty, twenty-five), ordinals (8th), labels with digits (R1.2, T7, C2), dates,
     hashes, DOIs. Comments (`% ...`) are a separate scope. LaTeX structure is EXCLUDED and counted separately:
     class/package options (12pt, margin=1in), lengths, \\label / \\ref / \\cite keys, figure options and filenames,
     environment options, table structure (\\multicolumn / \\cmidrule arguments), macro definitions and parameters.
     Calendar years in prose are NOT excluded: they are checked.
  2. Assigns every in-scope token ONE status: match / MISMATCH / approximate / no table counterpart / submitted-version
     figure (main.tex only; cites the STEP6_REVISION_MAP.md line that removes or replaces it) / unchecked (with the
     reason). An unassigned token, a declaration whose fragment is not found exactly once, or a token claimed twice
     FAILS the run (exit 3).
  3. Counterparts. (a) DECLARED checks, located by a text fragment (so a declaration goes stale, loudly, when the text
     changes): for the response letter, highlights and cover letter one check per number against a frozen-table cell
     (JSON path) or a named record (file + sha256 + field); figures the letter repeats from the methods text reuse the
     committed W163 check registry (w163_paragraphs_v4_figure_check.json) and are located again in paragraphs_v5.md.
     (b) main.tex: declared checks for the front matter and the case-study parameters, then the map's section rules
     (submitted-version figures), then context rules (notation in math and algorithms, data tables, words, comments),
     then the AUTOMATIC VALUE INDEX over the frozen JSON (every numeric leaf with its path; transforms raw, x100 when
     the token is followed by %, /1e3 for k EUR, /1e6 for M EUR; signed, or absolute when the text writes a loss as a
     positive number): a unique match is "match (auto, path)", several "match (auto, ambiguous: n paths)". The index is
     ALSO computed for every other numeric token and recorded beside its status, so the effect of the precedence is
     visible (main.tex is still the submitted text: a coincidental value equality in a passage the map removes is not
     a trace).
  4. Letter findings: sentences whose claim contradicts a frozen table or a recorded value are listed as findings,
     never edited. Definition / wording checks of main.tex are not required this round.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W163 / W162 / W161 / W160 imports arm; `pickle.load` / `pickle.loads` blocked for the whole
run, every counter verified at 0. git reads are `git show` / `git log` / `git status` (read-only, also in the clone).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w164_manuscript_check && \\
    mkdir data/SRP1/Results/P515S53/w164_manuscript_check/overleaf_6191c6c && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w164_manuscript_number_check.py \\
        --overleaf-commit 6191c6c --expect main.tex=<sha256> --expect highlights.tex=<sha256> ... \\
        > data/SRP1/Results/P515S53/w164_manuscript_check/overleaf_6191c6c/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w164_manuscript_check/overleaf_6191c6c/w164_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w164_manuscript_check/overleaf_6191c6c/w164_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w164_manuscript_number_check.py \\
        --overleaf-commit 6191c6c --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES are FINDINGS and never change the
exit code); 3 = written, an integrity check failed (listed: unassigned tokens, stale declarations, ...); 1 =
precondition or guard fault (nothing written).
"""
import argparse
import bisect
import collections
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W164 manuscript number check (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W164: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W164: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w163_paragraphs_v4_check as W163  # noqa: E402 -- arms its guards, imports W162 / W161 / W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W162 = W163.W162
W161 = W162.W161
W160 = W162.W160
GUARDS = W160.W157._dedupe((('w164_manuscript_number_check', GUARD),) + tuple(W163.GUARDS))

_log = W160._log
_sha = W160._sha
_sha_bytes = W160._sha_bytes
_git = W160._git
_jl = W162._jl
_text = W162._text
at_prec = W162.at_prec

SCRIPT_REL = os.path.basename(__file__)
S53 = W160.S53
OUT_ROOT = os.path.join(S53, 'w164_manuscript_check')
DEFAULT_MANUSCRIPT_DIR = os.path.join('manuscript', '6a67305f25e8348fb71380c3')

FZ_REL, FZ_SHA = W162.FZ_REL, W162.FZ_SHA
FZ_MD_REL = FZ_REL[:-len('.json')] + '.md'
W163_JSON, W163_MAN = W163.OUT_JSON, W163.OUT_MAN
W163_SCRIPT, W163_SCRIPT_COMMIT, W163_RESULTS_COMMIT = W163.SCRIPT_REL, '7648a27a', 'ff66c75f'
P5_REL = os.path.join(W162.EXPORT, 'paragraphs_v5.md')
P5_SHA = 'b86657069901ed62ccd80209295bf2d4ecc075a149f47b283148273a9c1a9841'
P5_COMMIT = '097421f8'
P2_REL, P2_SHA = W162.P2_REL, W162.P2_SHA
MAP_REL = 'STEP6_REVISION_MAP.md'
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
SRP1_PARAMS = W162.SRP1_PARAMS
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
STEP4 = 'STEP4_DFO_METHOD.md'
A1_CAMPAIGN = 'p515_s45_a1_campaign.py'
V41_SPEC = W162.V41_SPEC
REP_YEARS = ('2025', '2030', '2035')
CASE_FILES = {f'case9_{y}': os.path.join('data', 'SRP1', 'case9', f'case9_{y}.json') for y in REP_YEARS}
for _n in (1, 2, 3):
    for _y in REP_YEARS:
        CASE_FILES[f'case33_{_n}_{_y}'] = os.path.join('data', 'SRP1', f'case33_{_n}', f'case33_{_n}_{_y}.json')

OUT_NAMES = {'json': 'w164_manuscript_number_check.json', 'md': 'w164_manuscript_number_check.md',
             'man': 'manifest_sha256.json', 'log': 'launch.log', 'typing_json': 'w164_bool_typing_test.json',
             'typing_log': 'w164_bool_typing_test.log', 'post': 'manifest_post_run_sha256.json'}

STATUSES = ('match', 'MISMATCH', 'approximate', 'no table counterpart', 'submitted-version figure', 'unchecked')
NUMWORDS = {'twenty-five': 25, 'twenty': 20, 'twelve': 12, 'eleven': 11, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'single': 1, 'third': 3,
            'hundred': 100}


# ======================================================================================================================
#  1. tokenizer (LaTeX-aware)
# ======================================================================================================================
EXCL = (
    ('package/class option or argument', r'\\(?:documentclass|usepackage)(?:\[[^\]]*\])?(?:\{[^}]*\})?'),
    ('label/ref/cite key', r'\\(?:label|ref|eqref|autoref|cref|Cref|pageref|cite[tp]?|citeauthor|citeyear|nocite)\*?'
                           r'\{[^}]*\}'),
    ('figure option/filename', r'\\includegraphics(?:\[[^\]]*\])?\{[^}]*\}'),
    ('url', r'\\(?:url|href)\{[^}]*\}|https?://\S+'),
    ('environment option/argument', r'\\begin\{[^}]*\}(?:\[[^\]]*\])?(?:\{[^}]*\})*'),
    ('table structure', r'\\(?:multicolumn|multirow)\{[^}]*\}(?:\{[^}]*\})?|\\(?:cmidrule|cline)(?:\([^)]*\))?'
                        r'\{[^}]*\}'),
    ('macro definition', r'\\(?:newcommand|renewcommand|providecommand)\{[^}]*\}(?:\[\d+\])?'),
    ('macro parameter', r'#\d'),
    ('length/layout', r'\\(?:vspace|hspace|setlength|addtolength|resizebox|scalebox|rule|parbox|makebox)\*?'
                      r'(?:\{[^}]*\})+|\b\d*\.?\d+(?:pt|mm|cm|in|em|ex|bp)\b'
                      r'|\d*\.?\d+\\(?:textwidth|linewidth|columnwidth|textheight|hsize)'
                      r'|(?:width|height|scale|angle)\s*=\s*[\d.]+'),
    ('counter', r'\\setcounter\{[^}]*\}\{[^}]*\}'),
)
EXCL_RE = tuple((c, re.compile(p)) for c, p in EXCL)
TOK = re.compile(
    r'(?P<doi>\b10\.\d{4,9}/[^\s;,}]+)'
    r'|(?P<hash>\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b)'
    r'|(?P<date>\b\d{4}-\d{2}-\d{2}\b|\b\d{4}/\d{2}/\d{2}\b)'
    r'|(?P<label>(?<![\\\w])[A-Za-z]{1,5}\d+(?:\.\d+)*(?:[a-z])?\b)'
    r'|(?P<ordinal>(?<![\w.])\d+(?:st|nd|rd|th)\b)'
    r'|(?P<alnum>(?<![\w.])\d+(?:\.\d+)?[A-Za-z]+\w*)'
    r'|(?P<sci>(?<![\w.])10\^\{?[-−]?\d+\}?)'
    r'|(?P<dotted>(?<![\w.])\d+\.\d+\.\d+(?![\w.]*\d))'
    r'|(?P<num>(?<![\w.])\d{1,3}(?:(?:,|\{,\}|\\,)\d{3})+(?:\.\d+)?(?![\w])'
    r'|(?<![\w.])\d+(?:\.\d+)?(?![\w]|\.\d))'
    r'|(?P<word>\b(?:' + '|'.join(sorted(NUMWORDS, key=len, reverse=True)) + r')\b)', re.I)


def split_comment(line):
    """(body, comment text or None, column of the '%'): the first '%' not escaped by an odd run of backslashes."""
    i = 0
    while True:
        j = line.find('%', i)
        if j < 0:
            return line, None, len(line)
        k, bs = j - 1, 0
        while k >= 0 and line[k] == '\\':
            bs += 1
            k -= 1
        if bs % 2 == 0:
            return line[:j], line[j + 1:], j
        i = j + 1


CMD_RX = re.compile(r'\\[A-Za-z]+')


def _tokens_of(seg, offset):
    spans = []
    for cat, rx in EXCL_RE:
        for m in rx.finditer(seg):
            spans.append((m.start(), m.end(), cat))
    masked = CMD_RX.sub(lambda m: ' ' * len(m.group()), seg)   # \\times3 -> '      3'; positions kept
    out = []
    for m in TOK.finditer(masked):
        a, b, t, typ = m.start(), m.end(), m.group(), m.lastgroup
        cat = next((c for s, e, c in spans if s <= a and b <= e), None)
        sgn = ''
        if typ == 'num' and a > 0 and seg[a - 1] in '+-−' and (a < 2 or not (seg[a - 2].isalnum() or seg[a - 2] in '-_')):  # noqa: E501
            sgn = '-' if seg[a - 1] in '-−' else '+'
        out.append({'col0': a + offset, 'end0': b + offset, 'text': t, 'type': typ, 'sign': sgn, 'excluded': cat})
    return out


def _numval(t):
    typ, s = t['type'], t['text']
    if typ == 'num':
        v = float(s.replace('{,}', '').replace('\\,', '').replace(',', ''))
        return -v if t['sign'] == '-' else v
    if typ == 'word':
        return float(NUMWORDS[s.lower()])
    if typ == 'sci':
        return 10.0 ** int(re.search(r'[-−]?\d+\}?$', s.split('^', 1)[1]).group().strip('{}').replace('−', '-'))
    if typ == 'ordinal':
        return float(re.match(r'\d+', s).group())
    return None


def _written(t):
    if t['type'] == 'num':
        return t['sign'] + t['text'].replace('{,}', ',').replace('\\,', ',')
    return t['text']


UNIT_RX = ((r'\\%', '%'), (r'k\\euro', 'kEUR'), (r'M\\euro', 'MEUR'), (r'\\euro', 'EUR'), (r'MWh', 'MWh'),
           (r'MVA', 'MVA'), (r'MW\b', 'MW'), (r'GB\b', 'GB'), (r'p\.u\.', 'pu'), (r'h\b', 'h'), (r's\b', 's'),
           (r'-?years?\b', 'year'), (r'-bus\b', 'bus'))


def _unit_after(line, end0):
    s = line[end0:]
    s = re.sub(r'^(?:\s|~|\\,|\\ |\{\}|\$)+', '', s)
    for rx, u in UNIT_RX:
        if re.match(rx, s):
            return u
    return None


class Doc:
    """One .tex file: lines, offsets, comment columns, environment spans, math mask, section headers, macro-argument
    spans (the letter's \\rcomment / \\rresponse / \\rchanges) and the flattened text used to locate fragments."""

    def __init__(self, name, raw):
        self.name, self.raw = name, raw
        self.lines = raw.split('\n')
        self.starts, o = [], 0
        for ln in self.lines:
            self.starts.append(o)
            o += len(ln) + 1
        self.com = [split_comment(ln)[2] for ln in self.lines]
        self.blank = '\n'.join(ln[:c] + ' ' * (len(ln) - c) for ln, c in zip(self.lines, self.com))
        self._envs()
        self._math()
        self._sections()
        self._macro_args()
        self._flatten()

    def off(self, line, col0):
        return self.starts[line - 1] + col0

    def _envs(self):
        self.envs, stack = [], []
        for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', self.blank):
            if m.group(1) == 'begin':
                stack.append((m.group(2), m.start()))
            else:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == m.group(2):
                        nm, a = stack.pop(i)
                        self.envs.append((nm, a, m.end()))
                        break

    def envs_at(self, o):
        return sorted({nm for nm, a, b in self.envs if a <= o < b})

    def _math(self):
        n = len(self.blank)
        mask = bytearray(n)
        mathenv = re.compile(r'(equation|align|gather|multline|eqnarray|displaymath|math)\*?$')
        for nm, a, b in self.envs:
            if mathenv.match(nm):
                for i in range(a, b):
                    mask[i] = 1
        s, i, inl = self.blank, 0, False
        start = 0
        while i < n:
            c = s[i]
            if c == '\\' and i + 1 < n:
                if s[i + 1] in '[(' and not inl:
                    j = s.find('\\]' if s[i + 1] == '[' else '\\)', i + 2)
                    j = n if j < 0 else j + 2
                    for k in range(i, j):
                        mask[k] = 1
                    i = j
                    continue
                i += 2
                continue
            if c == '$':
                if s.startswith('$$', i):
                    j = s.find('$$', i + 2)
                    j = n if j < 0 else j + 2
                    for k in range(i, j):
                        mask[k] = 1
                    i = j
                    continue
                if not inl:
                    inl, start = True, i
                else:
                    for k in range(start, i + 1):
                        mask[k] = 1
                    inl = False
            i += 1
        self.mathmask = mask

    def _brace_end(self, i):
        """index just after the brace group opening at self.blank[i] == '{'."""
        depth, s = 0, self.blank
        while i < len(s):
            if s[i] == '\\':
                i += 2
                continue
            if s[i] == '{':
                depth += 1
            elif s[i] == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return len(s)

    def _sections(self):
        self.heads = []
        sec = sub = subsub = 0
        app = False
        for m in re.finditer(r'\\(appendix)\b|\\(section|subsection|subsubsection)(\*?)\{', self.blank):
            if m.group(1):
                app, sec, sub, subsub = True, 0, 0, 0
                continue
            lvl, star = m.group(2), m.group(3)
            e = self._brace_end(m.end() - 1)
            title = re.sub(r'\\textcolor\{[^}]*\}', '', self.blank[m.end():e - 1]).strip('{} ')
            if not star:
                if lvl == 'section':
                    sec, sub, subsub = sec + 1, 0, 0
                elif lvl == 'subsection':
                    sub, subsub = sub + 1, 0
                else:
                    subsub += 1
                s0 = chr(ord('A') + sec - 1) if app else str(sec)
                num = s0 if lvl == 'section' else f'{s0}.{sub}' if lvl == 'subsection' else f'{s0}.{sub}.{subsub}'
            else:
                num = None
            self.heads.append({'offset': m.start(), 'line': self.blank.count('\n', 0, m.start()) + 1, 'level': lvl,
                               'star': bool(star), 'number': num, 'title': title})
        self.head_offsets = [h['offset'] for h in self.heads]

    def section_at(self, o):
        i = bisect.bisect_right(self.head_offsets, o) - 1
        path = {}
        for h in self.heads[:i + 1]:
            if h['level'] == 'section':
                path = {'section': h}
            elif h['level'] == 'subsection':
                path = {'section': path.get('section'), 'subsection': h}
            else:
                path = {'section': path.get('section'), 'subsection': path.get('subsection'), 'subsubsection': h}
        parts = []
        for k in ('section', 'subsection', 'subsubsection'):
            h = path.get(k)
            if h:
                parts.append(f"{h['number'] + ' ' if h['number'] else ''}{h['title']}".strip())
        envs = self.envs_at(o)
        for fm in ('abstract', 'highlights', 'keyword', 'graphicalabstract'):
            if fm in envs:
                return 'Front matter > ' + fm
        if not parts:
            return 'Front matter / preamble'
        return ' > '.join(parts)

    def _macro_args(self):
        self.margs = []
        for m in re.finditer(r'\\(rcomment|rresponse|rchanges)\{', self.blank):
            self.margs.append((m.group(1), m.end() - 1, self._brace_end(m.end() - 1)))

    def macro_at(self, o):
        return next((nm for nm, a, b in self.margs if a <= o < b), None)

    def _flatten(self):
        chars, idx = [], []
        prev_space = False
        for li, ln in enumerate(self.lines):
            for ci, ch in enumerate(ln + '\n'):
                sp = ch.isspace()
                if sp and prev_space:
                    continue
                chars.append(' ' if sp else ch)
                idx.append((li + 1, ci))
                prev_space = sp
        self.flat = ''.join(chars)
        self.flat_idx = idx
        self.flat_of = {v: i for i, v in enumerate(idx)}

    def find_fragment(self, frag):
        f = re.sub(r'\s+', ' ', frag.strip())
        hits = [m.start() for m in re.finditer(re.escape(f), self.flat)]
        return hits, len(f)

    def flat_pos(self, line, col0):
        return self.flat_of.get((line, col0))


def tokenize(doc):
    toks = []
    for i, ln in enumerate(doc.lines):
        body, com, j = split_comment(ln)
        seen = collections.Counter()
        for scope, seg, offs in (('body', body, 0), ('comment', com, j + 1)):
            if seg is None:
                continue
            for t in _tokens_of(seg, offs):
                key = (scope, t['text'])
                t.update(file=doc.name, line=i + 1, col=t['col0'] + 1, scope=scope, occ=seen[key])
                seen[key] += 1
                o = doc.off(i + 1, t['col0'])
                t['in_math'] = bool(doc.mathmask[o]) if scope == 'body' and o < len(doc.mathmask) else False
                t['envs'] = doc.envs_at(o) if scope == 'body' else []
                t['section'] = doc.section_at(o)
                t['macro'] = doc.macro_at(o) if scope == 'body' else None
                t['value'] = _numval(t)
                t['written'] = _written(t)
                t['unit'] = _unit_after(ln, t['end0'])
                t['key'] = f"{doc.name}:{i + 1}:{t['col']}"
                toks.append(t)
    return toks


# ======================================================================================================================
#  2. the automatic value index over the frozen JSON
# ======================================================================================================================
ID_KEYS = ('claim_id', 'cell', 'arm', 'quantity', 'rate', 'id')


def json_leaves(obj, path=''):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += json_leaves(v, f'{path}.{k}' if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sel = next((f'{k}={v[k]}' for k in ID_KEYS if isinstance(v, dict) and k in v and
                        isinstance(v[k], (str, int, float)) and not isinstance(v[k], bool)), None)
            out += json_leaves(v, f'{path}[{sel if sel is not None else i}]')
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if math.isfinite(obj):
            out.append((path, float(obj)))
    return out


TRANSFORMS = {'raw': lambda v: v, 'x100': lambda v: 100.0 * v, '/1e3': lambda v: v / 1e3, '/1e6': lambda v: v / 1e6}
UNIT_TRANSFORMS = {'%': ('raw', 'x100'), 'kEUR': ('/1e3',), 'MEUR': ('/1e6',)}


class AutoIndex:
    def __init__(self, leaves):
        self.leaves = leaves
        self.by_dp = {}

    def _fmt(self, x, dp):
        s = f'{x:.{dp}f}'
        return s[1:] if s.startswith('-') and float(s) == 0.0 else s

    def _index(self, dp):
        if dp not in self.by_dp:
            ix = collections.defaultdict(list)
            for p, v in self.leaves:
                for tn, f in TRANSFORMS.items():
                    x = f(v)
                    ix[self._fmt(x, dp)].append((p, tn, False))
                    if x < 0:
                        ix[self._fmt(-x, dp)].append((p, tn, True))
            self.by_dp[dp] = ix
        return self.by_dp[dp]

    def candidates(self, tok):
        if tok['type'] != 'num' or tok['value'] is None:
            return None
        w = tok['written']
        dp = W160._dp(w)
        allowed = ('raw',) + UNIT_TRANSFORMS.get(tok['unit'], ())
        if tok['unit'] in ('kEUR', 'MEUR'):
            allowed = UNIT_TRANSFORMS[tok['unit']]
        key = self._fmt(W160._wval(w), dp)
        hits = [h for h in self._index(dp).get(key, []) if h[1] in allowed and (not h[2] or not w.startswith('-'))]
        return [{'path': p, 'transform': tn, 'abs': ab} for p, tn, ab in hits]


# ======================================================================================================================
#  3. records and check helpers
# ======================================================================================================================
def R(status, counterpart=None, kind=None, value=None, shown=None, note=None, cite=None, category=None):
    return {'status': status, 'counterpart': counterpart, 'counterpart_kind': kind, 'counterpart_value': value,
            'value_at_written_precision': shown, 'note': note, 'map_cite': cite, 'category': category}


def num(written, value, src, kind='frozen-table cell', note=None, scale=1.0, absval=False):
    x = value * scale
    if absval:
        x = abs(x)
    shown, st = at_prec(written, x)
    return R(st, src, kind, value, shown, note)


def wv(w):
    """the value a written token states (number words mapped; 10^{-k} parsed)."""
    lw = str(w).lower()
    if lw in NUMWORDS:
        return float(NUMWORDS[lw])
    if '^' in lw:
        return 10.0 ** int(re.search(r'[-−]?\d+', lw.split('^', 1)[1]).group().replace('−', '-'))
    return float(W160._wval(str(w)))


def count_eq(written, n, src, kind='named record', note=None):
    """the written token's value against a counted / recorded value n (n = -1: the record is not single-valued)."""
    return R('match' if wv(written) == float(n) else 'MISMATCH', src, kind, n, str(n), note)


def unch(category, reason):
    return R('unchecked', None, None, None, None, reason, category=category)


def ntc(note, named=None):
    return R('no table counterpart', named, 'named record' if named else None, None, None, note)


def sel(lst, **kw):
    hits = [x for x in lst if all(x.get(k) == v for k, v in kw.items())]
    if len(hits) != 1:
        raise KeyError(f'selector {kw}: {len(hits)} hits')
    return hits[0]


class Records:
    def __init__(self, inputs, man_doc):
        self.inputs = inputs
        self.fz = _jl(FZ_REL)
        self.t = self.fz['tables']
        self.claims = {c['claim_id']: c for c in self.t['claims']}
        self.a64 = {c['claim_id']: c for c in self.t['a64']['rows']}
        self.ageing = {r['arm']: r for r in self.t['ageing']['rows']}
        self.fzmd = _text(FZ_MD_REL)
        self.w163 = _jl(W163_JSON)
        self.w163c = {c['id']: c for c in self.w163['checks_in_v4'] + self.w163['nonmanuscript_checks']}
        self.w163u = self.w163['unchecked_tokens']
        self.p5 = _text(P5_REL)
        self.p5flat = re.sub(r'\s+', ' ', ' '.join(ln.lstrip('>').strip() for ln in self.p5.split('\n')))
        self.p2 = _text(P2_REL)
        self.map = _text(MAP_REL).split('\n')
        self.srp1 = _jl(SRP1_JSON)
        self.srp1p = _jl(SRP1_PARAMS)
        self.ess = _jl(ESS_PARAMS)
        self.step4 = _text(STEP4).split('\n')
        self.a1 = _text(A1_CAMPAIGN)
        self.v41 = _jl(V41_SPEC)
        self.cases = {k: _jl(v) for k, v in CASE_FILES.items()}
        self.main = man_doc

    def src(self, key):
        i = self.inputs[key]
        return f"{i['path']} (sha256 {i['sha256'][:8]}, last commit {(i.get('last_commit') or '')[:8]})"

    def map_line(self, frag):
        hits = [i + 1 for i, ln in enumerate(self.map) if frag in ln]
        if len(hits) != 1:
            raise KeyError(f'map fragment {frag!r}: {hits}')
        return hits[0]

    def main_line(self, frag):
        if self.main is None:
            raise KeyError('main.tex not in this manuscript directory')
        hits, _n = self.main.find_fragment(frag)
        if len(hits) != 1:
            raise KeyError(f'main.tex fragment {frag!r}: {len(hits)} hits')
        return self.main.flat_idx[hits[0]][0]

    def p5_line(self, frag):
        f = re.sub(r'\s+', ' ', frag)
        if f not in self.p5flat:
            return None
        lines = self.p5.split('\n')
        first = f.split(' ')[:4]
        for i, ln in enumerate(lines):
            if ' '.join(first) in re.sub(r'\s+', ' ', ln.lstrip('>')):
                return i + 1
        return 0

    # ---- named-record quantities ----
    def years(self):
        return self.srp1['Years']

    def n_dns(self):
        return len(self.srp1['DistributionNetworks'])

    def bus_counts(self):
        return {k: len(v['nodes']) for k, v in self.cases.items()}

    def lattice(self):
        p = re.search(r'^LATTICE_P_STEP = ([0-9.]+)', self.a1, re.M)
        e = re.search(r'^LATTICE_E_STEP = ([0-9.]+)', self.a1, re.M)
        s4p = any('**0.25 MVA**' in ln for ln in self.step4[:12])
        s4e = any('**0.5 MWh**' in ln for ln in self.step4[:12])
        s4d = any('2 h ≤ E/P ≤ 4 h' in ln for ln in self.step4[:12])
        return {'p_step': float(p.group(1)) if p else None, 'e_step': float(e.group(1)) if e else None,
                'step4_p': s4p, 'step4_e': s4e, 'step4_duration': s4d,
                'ep_min': self.ess['min_energy_to_power_factor'], 'ep_max': self.ess['max_energy_to_power_factor']}

    def scorecard_rows(self):
        txt = self.p2.split('## (v) Prediction scorecard', 1)[1].split('**Not included:**', 1)[0]
        return [int(m.group(1)) for m in re.finditer(r'^\| (\d+) \|', txt, re.M)]

    def verdict_changes(self):
        out = []
        for c in self.t['claims']:
            nv = c.get('net_verdict_recomputed_here') or c.get('net_verdict_W153')
            if c['gross_verdict'] != nv:
                out.append({'claim_id': c['claim_id'], 'family': c['family'], 'gross': c['gross_verdict'], 'net': nv,
                            'statement': c['statement']})
        return out

    def sign_changes(self):
        sg = lambda x: (x > 0) - (x < 0)  # noqa: E731
        return [c['claim_id'] for c in self.t['claims'] if sg(c['d_gross']) != sg(c['d_net'])]


def w163ref(X, cid, written_value, figure, w163_written, v5_frag, note=None):
    """A letter figure that repeats a methods figure: the committed W163 check (status match, the same written figure)
    and the v5 wording that still carries it."""
    c = X.w163c.get(cid)
    ln = X.p5_line(v5_frag) if v5_frag else None
    same = (written_value is None) or wv(written_value) == float(figure)
    ok = c is not None and c['status'] == 'match' and c.get('written_v4') == w163_written and same and \
        (v5_frag is None or ln is not None)
    cpt = (c.get('counterpart') or c.get('table_ref')) if c else 'NOT FOUND'
    vwp = (c.get('value_at_written_precision', c.get('table_value_at_written_precision'))) if c else None
    cp = (f"W163 {cid} ({W163_JSON}, results commit {W163_RESULTS_COMMIT}): {cpt}; "
          f"W163 status {c['status'] if c else None}, written {c.get('written_v4') if c else None!r}, value at written "
          f"precision {vwp!r}" +
          (f"; paragraphs_v5.md line {ln} ({P5_COMMIT})" if ln else (
              '; v5 fragment NOT FOUND' if v5_frag else '')))
    return R('match' if ok else 'MISMATCH', cp, 'W163 registry (named record)',
             (c.get('value', c.get('table_value_full_precision'))) if c else None, vwp, note)


# ======================================================================================================================
#  4. declared checks (fragment-located; one evaluation per declared token)
# ======================================================================================================================
LETTER = 'response_to_reviewers_draft.tex'
HIGHLIGHTS = 'highlights.tex'
COVER = 'cover_letter.tex'
MAIN = 'main.tex'


def D(cid, fname, fragment, tokens, fn):
    """tokens: [(text, occurrence of that text inside the fragment)]; fn(X, [written...]) -> one R or a list of R."""
    return {'id': cid, 'file': fname, 'fragment': fragment, 'tokens': tokens, 'fn': fn}


def _claim(X, cid):
    return X.claims[cid]


def _unit_node(X, cid):
    inst = _claim(X, cid)['instance']
    return {cell: [n for n, pe in v['candidate_canonical']['nodes'].items() if pe != [0.0, 0.0]]
            for cell, v in inst.items()}


def _main_check(X, frag, written, main_text, note=None):
    """a letter figure that quotes the submitted text: the main.tex fragment (found once) carries `main_text` as a token,
    and the letter's value equals it."""
    try:
        ln = X.main_line(frag)
    except KeyError as e:
        return R('MISMATCH', f'main.tex fragment {frag!r}', 'submitted manuscript (named record)', None, None, str(e))
    hits, flen = X.main.find_fragment(frag)
    inside = [t for t in X.main_tokens if t['text'] == main_text and X.main.flat_pos(t['line'], t['col0']) is not None
              and hits[0] <= X.main.flat_pos(t['line'], t['col0']) < hits[0] + flen]
    ok = bool(inside) and wv(written) == wv(main_text)
    return R('match' if ok else 'MISMATCH', f"{MAIN} at the declared Overleaf commit, line {ln}: {frag!r} "
             '(the submitted text; main.tex is still the submitted version)', 'submitted manuscript (named record)',
             main_text, main_text, note)


def letter_declarations():
    L = LETTER
    out = []
    # ---- comments ---------------------------------------------------------------------------------------------------
    out.append(D('LC1', L, "expert's skeleton, 2026-10-06", [('2026-10-06', 0)], lambda X, w: R(
        'match' if f'2026-10-06' in X.map[2] and "Expert's plan for the author, 2026-10-06" in X.map[2] else 'MISMATCH',
        f'{MAP_REL} line 3 (header: "Expert\'s plan for the author, 2026-10-06")', 'named record', '2026-10-06',
        '2026-10-06', 'comment scope; the map the skeleton follows carries the same date')))
    out.append(D('LC2', L, 'frozen tables (590088fe)', [('590088fe', 0)], lambda X, w: R(
        'match' if FZ_SHA.startswith(w[0]) else 'MISMATCH', f'sha256 of {FZ_REL}', 'identifier', FZ_SHA[:8],
        FZ_SHA[:8], 'comment scope; identifier of the frozen tables')))
    # ---- To the Editor --------------------------------------------------------------------------------------------
    out.append(D('L01', L, 'the Editor and the three Reviewers', [('three', 0)], lambda X, w: count_eq(
        w[0], len(re.findall(r'\\section\*\{Reviewer \d\}', X.letter_raw)), 'the letter itself: \\section*{Reviewer n} '
        'blocks', 'structure of the text', 'the reviewers\' documents are not in the repository (map section C)')))
    out.append(D('L02', L, 'two points apply to the whole revision', [('two', 0)], lambda X, w: count_eq(
        w[0], len(re.findall(r'\\paragraph\{', X.letter_raw.split('\\section*{Reviewer 1}')[0])),
        'the letter itself: \\paragraph blocks before Reviewer 1', 'structure of the text')))
    out.append(D('L03', L, '0.50\\,\\% gap was a stopping tolerance', [('0.50', 0)], lambda X, w: _main_check(
        X, 'to a relative optimality gap of 0.50\\%', w[0], '0.50',
        f"submitted figure quoted; {SRP1_PARAMS} benders.tol_rel = {X.srp1p['benders']['tol_rel']} (= 0.50 %) at "
        'HEAD corroborates "a stopping tolerance" (the parameters of the submitted run itself are not re-read)')))
    out.append(D('L04', L, 'admissible unit loses 64.4~k\\euro{} over the horizon, determinately', [('64.4', 0)],
                 lambda X, w: num(w[0], _claim(X, 'CHECK:headline_V_minus_I_settled')['d_gross'],
                                  'T1 tables.claims[claim_id=CHECK:headline_V_minus_I_settled].d_gross (k EUR, absolute value)',
                                  scale=1e-3, absval=True,
                                  note=f"verdict {_claim(X, 'CHECK:headline_V_minus_I_settled')['gross_verdict']} "
                                       f"({_claim(X, 'CHECK:headline_V_minus_I_settled')['gross_multiple']:.2f}x); "
                                       f"the unit: {_unit_node(X, 'CHECK:headline_V_minus_I_settled')}")))
    out.append(D('L05', L, 'The 18.25\\,\\% operating-cost reduction of the submitted version', [('18.25', 0)],
                 lambda X, w: _main_check(X, 'by 18.25\\% and 92.16\\%, respectively', w[0], '18.25',
                                          'submitted figure quoted (abstract; also main.tex Table l. 1179 -18.25%); '
                                          'withdrawn in Addendum 49 (map section A row 3)')))
    out.append(D('L06', L, '90.9~M\\euro{} (13.9\\,\\%), with its mechanism', [('90.9', 0), ('13.9', 0)],
                 lambda X, w: [num(w[0], X.t['benchmark']['benefit'], 'T6 tables.benchmark.benefit (M EUR)', scale=1e-6,
                                   note=f"determinate {X.t['benchmark']['determinate']}, {X.t['benchmark']['multiple']:.2f}x "
                                        f"the larger band; definition: {X.t['benchmark']['definition']}"),
                               num(w[1], X.t['benchmark']['benefit_relative'],
                                   'T6 tables.benchmark.benefit_relative (%)', scale=100.0)]))
    out.append(D('L07', L, 'Two requests were not met in full', [('Two', 0)], lambda X, w: unch(
        'structural', 'count of the requests the paragraph names (Reviewer 1 attribution; Reviewer 3 re-optimisation); '
        'the third sentence (the 5 x 5 instance) is not a reviewer request')))
    out.append(D('L08', L, 'submitted version (five representative years, twenty-five scenarios)',
                 [('five', 0), ('twenty-five', 0)], lambda X, w: [
                     _main_check(X, 'is represented by five years (2025, 2028, 2031, 2034, and 2037)', w[0], 'five',
                                 'the submitted five representative years'),
                     _main_check(X, 'resulting in 25 operating scenarios per representative day', w[1], '25',
                                 'the submitted 5 load/RES x 5 market-price scenarios')]))
    out.append(D('L09', L, 'results are given on a $3\\times3$ scenario instance', [('3', 0), ('3', 1)],
                 lambda X, w: [w163ref(X, 'N48', wi, 3, '3 × 3; x = 0', 'the 3 × 3 multi-scenario instance',
                                       'T11: the 3 x 3 instance (Addenda 52 / 54); map section B 3: 3 market x 3 '
                                       'operation scenarios') for wi in w]))
    # ---- R1.2 -------------------------------------------------------------------------------------------------------
    out.append(D('L10', L, 'single-scenario instance and on a $3\\times3$ market$\\times$operation scenario instance',
                 [('3', 0), ('3', 1)], lambda X, w: [w163ref(X, 'N48', wi, 3, '3 × 3; x = 0',
                                                              'the 3 × 3 multi-scenario instance') for wi in w]))
    out.append(D('L11', L, 'the storage value is 0.909--0.934 of its single-scenario value', [('0.909', 0),
                                                                                                ('0.934', 0)],
                 lambda X, w: [num(w[0], X.t['three_by_three']['R_range_derived'][0],
                                   'T11 tables.three_by_three.R_range_derived[0] (lower end, post-hoc settled descent)'),
                               num(w[1], X.t['three_by_three']['R_range_derived'][1],
                                   'T11 tables.three_by_three.R_range_derived[1] (upper end, at certification)')]))
    out.append(D('L12', L, 'against 0.933 predicted from the mean-profile price spread', [('0.933', 0)],
                 lambda X, w: num(w[0], sel(X.t['three_by_three']['rows'],
                                            quantity='R predicted from the mean-profile spread')['value'],
                                  'T11 tables.three_by_three.rows[quantity=R predicted from the mean-profile spread]'
                                  '.value (0.9331)')))
    out.append(D('L13', L, 'The test system (IEEE 9-bus transmission network with three IEEE 33-bus distribution '
                           'networks, 108 buses)', [('9', 0), ('three', 0), ('33', 0), ('108', 0)],
                 lambda X, w: [_bus(X, 'case9', w[0]), _bus(X, 'dns', w[1]), _bus(X, 'case33', w[2]),
                               _bus(X, 'total', w[3])]))
    out.append(D('L14', L, 'the horizon (three representative years standing for five-year blocks)',
                 [('three', 0), ('five', 0)], lambda X, w: [
                     count_eq(w[0], len(X.years()), f'{X.src("SRP1_JSON")} Years {X.years()} (count of keys)'),
                     count_eq(w[1], 5 if set(X.years().values()) == {5} else -1,
                              f'{X.src("SRP1_JSON")} Years {X.years()} (every block 5 years)')]))
    # ---- R1.4 -------------------------------------------------------------------------------------------------------
    out.append(D('L15', L, 'at 0, 2, 5 and 8\\,\\% the smallest unit', [('0', 0), ('2', 0), ('5', 0), ('8', 0)],
                 lambda X, w: [num(wi, X.t['discount'][i]['rate'], f'T7 tables.discount[{i}].rate (%)', scale=100.0)
                               for i, wi in enumerate(w)]))
    out.append(D('L16', L, 'is $-40.8$, $-64.4$, $-92.7$ and $-114.6$~k\\euro{} respectively, negative and determinate '
                           'at every rate', [('40.8', 0), ('64.4', 0), ('92.7', 0), ('114.6', 0)],
                 lambda X, w: [num(wi, X.t['discount'][i]['value_minus_I'],
                                   f"T7 tables.discount[{i}].value_minus_I (k EUR) at rate {X.t['discount'][i]['rate']}",
                                   scale=1e-3, note=f"verdict {X.t['discount'][i]['verdict']}")
                               for i, wi in enumerate(w)]))
    out.append(D('L17', L, 'The production rate remains 2\\,\\%.', [('2', 0)], lambda X, w: num(
        w[0], X.srp1['DiscountFactor'], f'{X.src("SRP1_JSON")} DiscountFactor (%)', 'named record', scale=100.0,
        note='T7 row 2 % is the CHECK headline (-64.4 k EUR)')))
    # ---- R2.1 / R2.2 ------------------------------------------------------------------------------------------------
    out.append(D('L18', L, 'stage selects one scenario-independent', [('one', 0)], lambda X, w: unch(
        'not a figure', 'article sense ("one ... plan" = a single plan)')))
    out.append(D('L19', L, 'the lattice (0.25~MVA and 0.5~MWh units, duration between 2 and 4~h',
                 [('0.25', 0), ('0.5', 0), ('2', 0), ('4', 0)], lambda X, w: _lattice_checks(X, w)))
    out.append(D('L20', L, 'with discounting. One recourse evaluation is made per candidate plan', [('One', 0)],
                 lambda X, w: unch('method statement', 'one evaluation per candidate (the evaluation cache keyed on '
                                   'the candidate); an algorithm statement, audited by the equation/algorithm audit '
                                   '(map section B 2.1), not a table figure')))
    out.append(D('L21', L, 'every ADMM cycle solves one local problem per network block (three representative years '
                           '$\\times$ four representative days $\\times$ four agents at the single-scenario instance, '
                           '48 blocks', [('one', 0), ('three', 0), ('four', 0), ('four', 1), ('single', 0), ('48', 0)],
                 lambda X, w: _block_checks(X, w)))
    # ---- R2.3 / R2.5 ------------------------------------------------------------------------------------------------
    out.append(D('L22', L, "Section~2.2 (former 2.2.7 ``Benders' cuts'' removed", [('2.2', 0), ('2.2.7', 0)],
                 lambda X, w: [unch('cross-reference', 'revised-manuscript section number (map section B 2.2)'),
                               _former_227(X, w[1])]))
    out.append(D('L23', L, "new 2.2.7 ``Recourse evaluation and", [('2.2.7', 0)], lambda X, w: R(
        'match' if X.map_line('replace by "2.2.7 Recourse evaluation') else 'MISMATCH',
        f'{MAP_REL} line {X.map_line(chr(34) + "2.2.7 Recourse evaluation")}: '
        '"replace by \\"2.2.7 Recourse evaluation and certification\\""', 'named record (revision map)', '2.2.7',
        '2.2.7')))
    out.append(D('L24', L, 'a settling criterion on the objective --- three measured turning points', [('three', 0)],
                 lambda X, w: w163ref(X, 'N7', w[0], 3, 'three', 'at least three turning points')))
    out.append(D('L25', L, 'over a window at least one period long', [('one', 0)], lambda X, w: _window_check(X)))
    out.append(D('L26', L, '42 evaluations under the rule, 32 certified, the 10 uncertified by cause',
                 [('42', 0), ('32', 0), ('10', 0)], lambda X, w: [
                     w163ref(X, 'C5', w[0], 42, '42', 'Of the 42 SRP1 evaluations run under this rule'),
                     w163ref(X, 'C6', w[1], 32, '32', '32 certified (24 oscillatory, 8 monotone)'),
                     w163ref(X, 'N30', w[2], 10, '10', 'The 10 uncertified evaluations divide as follows')]))
    out.append(D('L27', L, 'post-certification movement ($\\le 0.9\\,\\tau$ on continued cells)', [('0.9', 0)],
                 lambda X, w: w163ref(X, 'N25', w[0], 0.9, '0.9', 'moved by at most 0.9 τ',
                                      '"continued cells" = evaluations continued past a certificate under this rule '
                                      '(W163 N25 scope; the old residual-rule references are outside it)')))
    out.append(D('L28', L, 'the three-start bands of the benchmark arms', [('three', 0)], lambda X, w: _three_start(X)))
    # ---- R2.8 -------------------------------------------------------------------------------------------------------
    out.append(D('L29', L, 'The three references were added to the literature review', [('three', 0)],
                 lambda X, w: count_eq(w[0], len(re.findall(r'\b10\.\d{4,9}/', X.letter_raw.split(
                     '\\subsection*{R2.8')[1].split('\\rresponse')[0])), 'the letter itself: DOIs in the R2.8 comment',
                     'structure of the text', 'the same response says two of the three are "not yet added" '
                                              '(lines 215-216): see letter findings')))
    out.append(D('L30', L, '[AUTHOR: one clause per reference', [('one', 0)], lambda X, w: unch(
        'author placeholder', '[AUTHOR: ...] instruction, not manuscript text')))
    # ---- R2.10 / R2.11 ----------------------------------------------------------------------------------------------
    out.append(D('L31', L, 'storage schedule couples the two (Section', [('two', 0)], lambda X, w: unch(
        'not a figure', 'pronoun ("the two" = the network models and the storage agent)')))
    # ---- R3.1 -------------------------------------------------------------------------------------------------------
    out.append(D('L32', L, 'It changes no sign in the 60 reported comparisons and one verdict',
                 [('60', 0), ('one', 0)], lambda X, w: [
                     count_eq(w[0], len(X.t['claims']), 'T1 tables.claims (count)', 'frozen-table cell',
                              f'sign changes gross -> net: {len(X.sign_changes())} {X.sign_changes()}'),
                     count_eq(w[1], len(X.verdict_changes()), 'T1 tables.claims: gross_verdict != net verdict (count)',
                              'frozen-table cell', f'the changed verdict: {X.verdict_changes()}; the letter calls it '
                                                   '"a Phase B neighbour" -- see letter findings')]))
    out.append(D('L33', L, 'The investment-year comparison is the one result that depends on the convention',
                 [('one', 0)], lambda X, w: _one_result(X)))
    out.append(D('L34', L, '--- 2035 is worse than 2030 in gross terms (+42.5~k\\euro{}, determinate)',
                 [('2035', 0), ('2030', 0), ('42.5', 0)], lambda X, w: [
                     R('match' if '2035' in X.t['year_ladder']['per_year'] or 2035 in X.years() or '2035' in X.years()
                       else 'MISMATCH', 'T4 tables.year_ladder.per_year; ' + X.src('SRP1_JSON') + ' Years',
                       'frozen-table cell', sorted(X.t['year_ladder']['per_year']), '2035'),
                     R('match' if '2030' in X.t['year_ladder']['per_year'] or '2030' in X.years() else 'MISMATCH',
                       'T4 tables.year_ladder.per_year; ' + X.src('SRP1_JSON') + ' Years', 'frozen-table cell',
                       sorted(X.t['year_ladder']['per_year']), '2030'),
                     num(w[2], X.t['year_ladder']['D_gross'], 'T4 tables.year_ladder.D_gross (k EUR, 2035 - 2030)',
                         scale=1e-3, note=f"verdict {X.t['year_ladder']['gross_v6']['verdict']} "
                                          f"({X.t['year_ladder']['gross_v6']['multiple']:.2f}x)")]))
    out.append(D('L35', L, '($-2.3$~k\\euro{}, within resolution)', [('2.3', 0)], lambda X, w: num(
        w[0], X.t['year_ladder']['D_net'], 'T4 tables.year_ladder.D_net (k EUR)', scale=1e-3,
        note=f"verdict {X.t['year_ladder']['net_v6']['verdict']} ({X.t['year_ladder']['net_v6']['multiple']:.2f}x)")))
    out.append(D('L36', L, 'consistent with Table~[2] and printed', [('2', 0)], lambda X, w: unch(
        'cross-reference', 'Table [2] of the revised manuscript (placeholder; the reviewer cites "Table 2 in Section '
        '4.1" of the submitted version)')))
    # ---- R3.2 -------------------------------------------------------------------------------------------------------
    out.append(D('L37', L, 'R3.2 --- The 0.50\\,\\% gap is not an optimality certificate', [('0.50', 0)],
                 lambda X, w: _main_check(X, 'to a relative optimality gap of 0.50\\%', w[0], '0.50',
                                          'submitted figure quoted (heading)')))
    out.append(D('L38', L, '$\\tau = \\delta_R V/4$ with $\\delta_R = 0.07$ of the single-scenario storage value',
                 [('4', 0), ('0.07', 0), ('single', 0)], lambda X, w: [
                     w163ref(X, 'N18', w[0], 4, '4', 'τ = δR·V/4'),
                     w163ref(X, 'N19', w[1], 0.07, '0.07', 'with δR = 0.07',
                             'V = 259,375.33 EUR, the SRP1 (single-scenario) value of the reference unit (W163 N20)'),
                     unch('not a figure', 'compound adjective (single-scenario)')]))
    out.append(D('L39', L, 'so that a ratio of two values is resolved', [('two', 0)], lambda X, w: w163ref(
        X, 'N22', w[0], 2, 'two', 'a ratio of two values')))
    out.append(D('L40', L, 'On Figure~15 of the submitted version', [('15', 0)], lambda X, w: _figure_count(X, 15)))
    # ---- R3.5 / R3.6 ------------------------------------------------------------------------------------------------
    out.append(D('L41', L, 'annual retention factor of 0.985 in the baseline calibration', [('0.985', 0)],
                 lambda X, w: _calendar(X, w[0])))
    out.append(D('L42', L, 'the smallest unit loses 31.6~k\\euro{} without calendar fade and 64.4~k\\euro{} with it,',
                 [('31.6', 0), ('64.4', 0)], lambda X, w: [
                     num(w[0], X.ageing['C2']['value_minus_I'], 'T8 tables.ageing.rows[arm=C2].value_minus_I (k EUR, absolute value)',
                         scale=1e-3, absval=True, note=f"verdict {X.ageing['C2']['verdict']}"),
                     num(w[1], X.ageing['C2_calfade']['value_minus_I'],
                         'T8 tables.ageing.rows[arm=C2_calfade].value_minus_I (k EUR, absolute value)', scale=1e-3, absval=True,
                         note=f"verdict {X.ageing['C2_calfade']['verdict']}")]))
    out.append(D('L43', L, 'announcing 0.5\\,\\% and 2\\,\\% cases', [('0.5', 0), ('2', 0)], lambda X, w: [
        _main_check(X, 'annual rates of 0.5\\% and 2.0\\% are additionally considered', w[0], '0.5',
                    'submitted sentence quoted (map section A last row; section B 3.3-3.4 l. 1062)'),
        _main_check(X, 'annual rates of 0.5\\% and 2.0\\% are additionally considered', w[1], '2.0',
                    'written 2 against the submitted 2.0')]))
    out.append(D('L44', L, '(the smallest admissible unit at node~7)', [('7', 0)], lambda X, w: R(
        'match' if all(v == ['7'] for k, v in _unit_node(X, 'E:n7_4h_e1_C2_calfade:value_minus_I').items()
                       if k != 'ref:7aa017f0') else 'MISMATCH',
        'T1 tables.claims[claim_id=E:n7_4h_e1_C2_calfade:value_minus_I].instance (the non-zero node of the unit)',
        'frozen-table cell', _unit_node(X, 'E:n7_4h_e1_C2_calfade:value_minus_I'), '7',
        'T8 caption: "fixed plan (unit at node 7, 0.25 MVA / 1 MWh)"')))
    out.append(D('L45', L, 'without ageing $-4.1$~k\\euro{} (within resolution, i.e.\\ break-even); cycling only '
                           '$-31.6$~k\\euro{}; cycling and calendar (the baseline) $-64.4$~k\\euro{}; harsher '
                           'calibrations $-73.7$ and $-45.2$~k\\euro{}; all determinate except the first',
                 [('4.1', 0), ('31.6', 0), ('64.4', 0), ('73.7', 0), ('45.2', 0)], lambda X, w: [
                     num(wi, X.ageing[arm]['value_minus_I'], f'T8 tables.ageing.rows[arm={arm}].value_minus_I (k EUR)',
                         scale=1e-3, note=f"verdict {X.ageing[arm]['verdict']}"
                                          + ('; C3_midblock (-65.9, determinate) is not named; "harsher" holds against '
                                             'C2 (cycling only), not against the baseline' if arm == 'C4' else ''))
                     for wi, arm in zip(w, ('no_ageing', 'C2', 'C2_calfade', 'C3_unit', 'C4'))]))
    out.append(D('L46', L, 'end-of-life floor from 0.70 to 0.50 changes the value by $+4.9$~k\\euro{}, within '
                           'resolution', [('0.70', 0), ('0.50', 0), ('4.9', 0)], lambda X, w: [
                     w163ref(X, 'N38', w[0], 0.70, '0.70', None, 'baseline floor; SRP1_ESS_Params ageing.minimum_soh '
                             f"{X.ess['ageing']['minimum_soh']}"),
                     w163ref(X, 'N41', w[1], 0.50, '0.50', None, 'the A64 soh_min 0.50 row (T10)'),
                     num(w[2], X.a64['E:soh050:delta_value_vs_070']['d_gross'],
                         'T10 tables.a64.rows[claim_id=E:soh050:delta_value_vs_070].d_gross (k EUR)', scale=1e-3,
                         note=f"verdict {X.a64['E:soh050:delta_value_vs_070']['gross_verdict']}")]))
    # ---- R3.7 / further changes -------------------------------------------------------------------------------------
    out.append(D('L47', L, 'one discount factor per representative year applied to the five years of its block',
                 [('one', 0), ('five', 0)], lambda X, w: [
                     R('match' if 'one discount factor per representative year applied to the five years of its block'
                       in X.fzmd else 'MISMATCH', f'{FZ_MD_REL} T7 caption ("one discount factor per representative '
                       'year applied to the five years of its block")', 'frozen-table caption', 1, '1'),
                     count_eq(w[1], 5 if set(X.years().values()) == {5} else -1,
                              f'{X.src("SRP1_JSON")} Years {X.years()}')]))
    out.append(D('L48', L, 'the investment is paid in 2025 and not discounted', [('2025', 0)], lambda X, w: R(
        'match' if 'I paid in 2025' in X.fzmd and len({round(r['I'], 6) for r in X.t['discount']}) == 1
        else 'MISMATCH', f'{FZ_MD_REL} T7 caption ("I paid in 2025"); T7 tables.discount[*].I identical at every rate',
        'frozen-table caption + cells', sorted({round(r['I'], 2) for r in X.t['discount']}), '2025',
        'I = 317,957.01 EUR at 0, 2, 5 and 8 %: not discounted')))
    out.append(D('L49', L, 'The five-year block discretization', [('five', 0)], lambda X, w: count_eq(
        w[0], 5 if set(X.years().values()) == {5} else -1, f'{X.src("SRP1_JSON")} Years {X.years()}')))
    out.append(D('L50', L, 'distribution exchange in 1 of 12 (passive) and 8 of 12 (price-responsive)',
                 [('1', 0), ('12', 0), ('8', 0), ('12', 1)], lambda X, w: [
                     count_eq(w[0], X.t['benchmark']['sweep']['sweep_passive_cold']['n_blocks'],
                              'T6 tables.benchmark.sweep.sweep_passive_cold.n_blocks', 'frozen-table cell'),
                     count_eq(w[1], len(X.years()) * len(X.srp1['Days']), f'{X.src("SRP1_JSON")}: years x days',
                              note='T6 column "sweep: blocks TN cannot accept (of 12)"'),
                     count_eq(w[2], X.t['benchmark']['sweep']['sweep_price_taker_cold']['n_blocks'],
                              'T6 tables.benchmark.sweep.sweep_price_taker_cold.n_blocks', 'frozen-table cell',
                              '"price-responsive" = the price-taker arm'),
                     count_eq(w[3], len(X.years()) * len(X.srp1['Days']), f'{X.src("SRP1_JSON")}: years x days')]))
    out.append(D('L51', L, 'a $10^{-5}$~p.u.\\ renewable bound slack', [('10^{-5}', 0)], lambda X, w: w163ref(
        X, 'N33', w[0], 1e-5, '10⁻⁵', 'plus a 10⁻⁵ pu numerical slack')))
    out.append(D('L52', L, 'single machine, single thread, bitwise replays', [('single', 0), ('single', 1)],
                 lambda X, w: [
                     ntc('no committed record states the machine count: W163 listed "one machine" as unchecked, '
                         f"NO SOURCE FOUND ({W163_JSON} unchecked_tokens); the v6 spec records concurrency 1 "
                         '(one evaluation at a time, W163 N44) and W159 the environment of one host'),
                     w163ref(X, 'N45', w[1], 1, 'single-threaded; one thread', 'single-threaded')]))
    out.append(D('L53', L, 'prediction record of the revision (49 predictions, with outcomes)', [('49', 0)],
                 lambda X, w: count_eq(w[0], len(X.scorecard_rows()) if X.scorecard_rows() == list(
                     range(1, len(X.scorecard_rows()) + 1)) else -1,
                     f'{X.src("P2")} section (v) prediction scorecard: rows numbered 1..n', 'named record')))
    return out


def _bus(X, which, written):
    """bus / network counts from the case files at the representative years and SRP1.json."""
    bc = X.bus_counts()
    n9 = {k: v for k, v in bc.items() if k.startswith('case9')}
    n33 = {k: v for k, v in bc.items() if k.startswith('case33')}
    tot = {y: bc[f'case9_{y}'] + sum(bc[f'case33_{n}_{y}'] for n in (1, 2, 3)) for y in REP_YEARS}
    one = lambda d: min(d.values()) if len(set(d.values())) == 1 else -1  # noqa: E731
    src = 'case files data/SRP1/case9/case9_<year>.json and data/SRP1/case33_<n>/case33_<n>_<year>.json at ' \
          f'{"/".join(REP_YEARS)}: len(nodes) '
    if which == 'case9':
        return count_eq(written, one(n9), src + f'{n9}')
    if which == 'case33':
        return count_eq(written, one(n33), src + f'{n33}')
    if which == 'total':
        return count_eq(written, one(tot), src + f'9 + 3 x 33 per year {tot}')
    if which == 'dns':
        return count_eq(written, X.n_dns(), f'{X.src("SRP1_JSON")} DistributionNetworks (count)')
    raise KeyError(which)


def _lattice_checks(X, w):
    la = X.lattice()
    s = f'{A1_CAMPAIGN} LATTICE_P_STEP / LATTICE_E_STEP; {STEP4} lines 8-9 (author\'s decision 2026-09-19); ' \
        f'{X.src("ESS_PARAMS")} min/max_energy_to_power_factor'
    return [R('match' if la['p_step'] == float(w[0]) and la['step4_p'] else 'MISMATCH', s, 'named record',
              la['p_step'], str(la['p_step'])),
            R('match' if la['e_step'] == float(w[1]) and la['step4_e'] else 'MISMATCH', s, 'named record', la['e_step'],
              str(la['e_step'])),
            R('match' if la['ep_min'] == float(w[2]) and la['step4_duration'] else 'MISMATCH', s, 'named record',
              la['ep_min'], str(la['ep_min'])),
            R('match' if la['ep_max'] == float(w[3]) and la['step4_duration'] else 'MISMATCH', s, 'named record',
              la['ep_max'], str(la['ep_max']))]


def _block_checks(X, w):
    ny, nd, na = len(X.years()), len(X.srp1['Days']), 1 + X.n_dns()
    src = f'{X.src("SRP1_JSON")}: Years {sorted(X.years())}, Days {sorted(X.srp1["Days"])}, TransmissionNetwork + ' \
          f'{X.n_dns()} DistributionNetworks'
    note = ('network blocks only: the storage-operator (ESSO) problems are solved in the same cycle in addition '
            '(TASKS.md W132 / W133: 51 exits per cycle incl. the ESSO at SRP1) -- see letter findings')
    return [unch('method statement', 'one local problem per network block per cycle (algorithm statement; audited by '
                 'the equation/algorithm audit, map section B Appendix A)'),
            count_eq(w[1], ny, src), count_eq(w[2], nd, src),
            count_eq(w[3], na, src, note='"four agents" = 1 TSO + 3 DSOs; ' + note),
            unch('not a figure', 'compound adjective (single-scenario)'),
            count_eq(w[5], ny * nd * na, src + ' (3 x 4 x 4)', note=note)]


def _former_227(X, written):
    hs = [h for h in X.main.heads if h['level'] == 'subsubsection' and "Benders' Cuts" in h['title']] if X.main else []
    num_ = hs[0]['number'] if len(hs) == 1 else None
    return R('match' if num_ == written else 'MISMATCH', f'{MAIN} section structure (subsubsection "Benders\' Cuts" at '
             f'line {hs[0]["line"] if hs else None}; numbered by its position)', 'submitted manuscript (named record)',
             num_, num_)


def _window_check(X):
    n9, n10 = X.w163c.get('N9'), X.w163c.get('N10')
    ok = n9 and n10 and n9['status'] == n10['status'] == 'match' and float(n10['written_v4']) >= 1.0
    return R('match' if ok else 'MISMATCH', 'W163 N9 (W_MIN 20) and N10 (W_FACTOR 1.1): W = max(20, ceil(1.1 P_hat)) '
             '>= 1.1 P_hat > P_hat, i.e. at least one measured period', 'W163 registry (named record) + derived',
             {'W_MIN': n9 and n9['value'], 'W_FACTOR': n10 and n10['value']}, '>= 1 period')


def _three_start(X):
    arms = X.t['benchmark']['w160_additions']['arms_in_full']
    n = {a: len(v['Q_by_start']) for a, v in arms.items()}
    return R('match' if set(n.values()) == {3} else 'MISMATCH',
             'T6 tables.benchmark.w160_additions.arms_in_full.{passive,price_taker}.Q_by_start (count of starts)',
             'frozen-table cell', n, '3', 'cold, perturbed, warm_from_certified; each arm = the minimum over its starts')


def _one_result(X):
    vc = X.verdict_changes()
    yl = X.t['year_ladder']
    t4 = yl['gross_v6']['verdict'] != yl['net_v6']['verdict']
    n = len(vc) + int(t4)
    return R('match' if n == 1 else 'MISMATCH', 'T1 tables.claims (gross_verdict != net verdict) and T4 '
             'tables.year_ladder (gross_v6.verdict != net_v6.verdict): the comparisons whose verdict depends on the '
             'convention', 'frozen-table cells', {'T1': [c['claim_id'] for c in vc], 'T4_2035_minus_2030': t4}, str(n),
             'the letter calls the investment-year comparison "the one result that depends on the convention"; T1 '
             f"carries a second: {vc[0]['claim_id'] if vc else None} (gross within the uncertified bar, net "
             'determinate), the verdict the preceding sentence counts. Reading: "result" = a reported comparison')


def _figure_count(X, n):
    figs = [(nm, X.main.blank.count('\n', 0, a) + 1) for nm, a, b in X.main.envs if nm in ('figure', 'figure*')] \
        if X.main else []
    return unch('cross-reference', f'figure number of the submitted version: main.tex at this commit has {len(figs)} '
                f'figure environments (outside comments); its figure 15 by environment order is at line '
                f'{figs[n - 1][1] if len(figs) >= n else None}; the submitted numbering is not re-derived here')


def _calendar(X, written):
    e = X.ess['ageing']['calendar_retention_per_year']
    v = X.v41['configuration']['identical_to_w104']['ess_ageing_baseline']['calendar_retention_per_year']
    ok = e == v == float(written)
    return R('match' if ok else 'MISMATCH', f'{X.src("ESS_PARAMS")} ageing.calendar_retention_per_year; '
             f'{X.src("V41_SPEC")} configuration.identical_to_w104.ess_ageing_baseline.calendar_retention_per_year',
             'named record', {'ess_params': e, 'v41_spec': v}, f'{e:g}', 'the C2_calfade baseline (T8)')


def cover_declarations():
    C = COVER
    return [
        D('CL1', C, 'the manuscript combines four elements', [('four', 0)], lambda X, w: count_eq(
            w[0], len(re.findall(r'\\item\b', X.cover_raw.split('combines four elements', 1)[1].split('\\end{itemize}')[0])),
            'the cover letter itself: \\item entries of the list that follows', 'structure of the text')),
        D('CL2', C, 'over a 15-year planning horizon', [('15', 0)], lambda X, w: count_eq(
            w[0], sum(X.years().values()), f'{X.src("SRP1_JSON")} Years (sum of block lengths)',
            note='the cover letter is the original submission letter (Benders, voltage violations): its claims are '
                 'submitted-version text; the horizon length is unchanged')),
    ]


def main_declarations():
    M = MAIN

    def sub(key, note, kind='result'):
        return lambda X, w: R('submitted-version figure', f'{MAP_REL} line {X.map_line(MAP_CITES[key])}',
                              'revision map', None, None, note, cite=f'{MAP_REL}:{X.map_line(MAP_CITES[key])}',
                              category=kind)

    def subs(n, key, note, kind='result'):
        f = sub(key, note, kind)
        return lambda X, w: [f(X, w) for _ in range(n)]

    return [
        D('M01', M, 'evaluated over a 15-year planning horizon using parameters', [('15', 0)], lambda X, w: count_eq(
            w[0], sum(X.years().values()), f'{X.src("SRP1_JSON")} Years (sum of block lengths)',
            note='map section B 3: three representative years 2025/2030/2035, each a five-year block (15-year horizon)')),
        D('M02', M, 'an integrated 108-bus test system comprising the IEEE 9-bus transmission network and three IEEE '
                    '33-bus active distribution networks', [('108', 0), ('9', 0), ('three', 0), ('33', 0)],
          lambda X, w: [_bus(X, 'total', w[0]), _bus(X, 'case9', w[1]), _bus(X, 'dns', w[2]), _bus(X, 'case33', w[3])]),
        D('M03', M, 'In the final representative year, 2037, coordinated operation with shared ESSs reduces the '
                    'representative-day average operating cost and RES curtailment by 18.25\\% and 92.16\\%',
          [('2037', 0), ('18.25', 0), ('92.16', 0)], lambda X, w: [
              sub('years', 'submitted final representative year (the revised horizon ends with 2035)')(X, w),
              sub('a1825', 'withdrawn (Addendum 49); replaced by the 13.9 % benefit against the best static NRF '
                           'arrangement (T6)')(X, w),
              sub('a1825', 'submitted RES-curtailment reduction; not reproduced under the revised method')(X, w)]),
        D('M04', M, 'The test system comprises the IEEE 9-bus transmission network coupled with three Active '
                    'Distribution Networks (ADNs), each derived from the IEEE 33-bus distribution test system',
          [('9', 0), ('three', 0), ('33', 0)], lambda X, w: [_bus(X, 'case9', w[0]), _bus(X, 'dns', w[1]),
                                                            _bus(X, 'case33', w[2])]),
        D('M05', M, 'connection points of the three ADNs. The 15-year planning horizon is represented by five years '
                    '(2025, 2028, 2031, 2034, and 2037), spaced at three-year intervals. For each year, four seasonal '
                    'representative days', [('three', 0), ('15', 0), ('five', 0), ('2025', 0), ('2028', 0),
                                            ('2031', 0), ('2034', 0), ('2037', 0), ('three', 1), ('four', 0)],
          lambda X, w: [count_eq(w[0], X.n_dns(), f'{X.src("SRP1_JSON")} DistributionNetworks (count)'),
                        count_eq(w[1], sum(X.years().values()), f'{X.src("SRP1_JSON")} Years (sum of block lengths)'),
                        sub('instance', 'five representative years: the revision runs three (2025, 2030, 2035)',
                            'input replaced')(X, w),
                        R('match' if '2025' in X.years() else 'MISMATCH', f'{X.src("SRP1_JSON")} Years', 'named record',
                          sorted(X.years()), '2025', 'a representative year of the revised instance')] +
          [sub('years', 'submitted representative year (columns 2025/2028/.../2037 replaced by 2025/2030/2035)',
               'input replaced')(X, w) for _ in range(4)] +
          [sub('instance', 'three-year spacing: the revision uses five-year blocks', 'input replaced')(X, w),
           count_eq(w[9], len(X.srp1['Days']), f'{X.src("SRP1_JSON")} Days {sorted(X.srp1["Days"])}')]),
        D('M06', M, 'Three ESS investment-cost scenarios are considered at the planning level. At the operational '
                    'level, five load/RES scenarios are combined with five market-price scenarios, resulting in 25 '
                    'operating scenarios per representative day. The total investment budget for shared ESS deployment '
                    'is set to 1~M\\euro, and the maximum installable energy capacity at each interface node is limited '
                    'to 5.00~MWh. All shared ESSs are assumed to have a calendar lifetime of 15 years, and a discount '
                    'rate of 2.00\\%', [('Three', 0), ('five', 0), ('five', 1), ('25', 0), ('1', 0), ('5.00', 0),
                                        ('15', 0), ('2.00', 0)],
          lambda X, w: [sub('cost', 'the three cost-trajectory scenarios are not what was run (map section B 3.1)',
                            'input replaced')(X, w)] +
          [sub('instance', 'SRP1 uses one scenario; the multi-scenario instance uses 3 market x 3 operation scenarios',
               'input replaced')(X, w) for _ in range(3)] +
          [num(w[4], X.ess['budget'], f'{X.src("ESS_PARAMS")} budget (M EUR)', 'named record', scale=1e-6,
               note='map section B 2.2: budget 1 M EUR'),
           num(w[5], X.ess['max_capacity'], f'{X.src("ESS_PARAMS")} max_capacity (MWh)', 'named record',
               note='map section B 2.2: <= 5 MWh per node'),
           count_eq(w[6], X.ess['ageing']['calendar_life_years'], f'{X.src("ESS_PARAMS")} ageing.calendar_life_years'),
           num(w[7], X.srp1['DiscountFactor'], f'{X.src("SRP1_JSON")} DiscountFactor (%)', 'named record',
               scale=100.0)]),
        D('M07', M, 'Typical values for utility-scale lithium-ion batteries range from 2~h to 10~h', [('2', 0),
                                                                                                        ('10', 0)],
          lambda X, w: [unch('literature value', 'cited typical duration range (\\cite{nrel_ess_costs}); the model '
                             f"bounds are E/P in [{X.ess['min_energy_to_power_factor']}, "
                             f"{X.ess['max_energy_to_power_factor']}] h (SRP1_ESS_Params)") for _ in range(2)]),
        D('M17', M, 'increase at an average annual rate of 2.50\\%, whereas flexibility market prices increase at '
                    '2.00\\% per year', [('2.50', 0), ('2.00', 0)],
          lambda X, w: [unch('network/data parameter', 'market-price growth rate of the market data, outside the frozen '
                             'tables (map section B 3.2: keep)') for _ in range(2)]),
        D('M08', M, 'Within each ADN, all 32 loads are modeled', [('32', 0)], lambda X, w: count_eq(
            w[0], min(len(v['loads']) for k, v in X.cases.items() if k.startswith('case33'))
            if len({len(v['loads']) for k, v in X.cases.items() if k.startswith('case33')}) == 1 else -1,
            'case files data/SRP1/case33_{1,2,3}/case33_n_{2025,2030,2035}.json len(loads)')),
        D('M09', M, 'grow at average annual rates of 1.25\\% and 3.00\\%', [('1.25', 0), ('3.00', 0)],
          lambda X, w: [unch('network/data parameter', 'demand growth rate of the case data, outside the frozen tables '
                             '(not re-derived from the case files this round)') for _ in range(2)]),
        D('M10', M, 'illustrate the flexible load profiles adopted for the three ADNs', [('three', 0)],
          lambda X, w: count_eq(w[0], X.n_dns(), f'{X.src("SRP1_JSON")} DistributionNetworks (count)')),
        D('M11', M, 'equipped with an Apple M2 Max processor and 32~GB of RAM', [('M2', 0), ('32', 0)],
          lambda X, w: [w163ref(X, 'N56', None, None, 'Apple M2 Max', 'on an Apple M2 Max',
                                'label token M2 of "Apple M2 Max" (the W163 record names the processor)'),
                        w163ref(X, 'N57', w[1], 32, '32 GiB', 'with 32 GiB of memory',
                                'main.tex writes "32 GB"; the record is 34,359,738,368 bytes = 32 GiB')]),
        D('M12', M, 'A minimum SoH of 70\\% is imposed as the baseline end-of-life criterion', [('70', 0)],
          lambda X, w: num(w[0], X.ess['ageing']['minimum_soh'], f'{X.src("ESS_PARAMS")} ageing.minimum_soh (%)',
                           'named record', scale=100.0, note='T8 "Ageing arms at minimum SoH 0.70"')),
        D('M13', M, 'alternative minimum SoH values of 60\\% and 80\\% are evaluated', [('60', 0), ('80', 0)],
          lambda X, w: [ntc('the frozen tables carry minimum SoH 0.70 (T8) and 0.50 (T10 soh050 row) only; no 0.60 / '
                            '0.80 evaluation exists in them. This red note (main.tex l. 1058) is not cited by the '
                            'revision map') for _ in range(2)]),
        D('M14', M, 'corresponding to a 1\\% loss of remaining usable capacity per year. This results in approximately '
                    '14\\% calendar-induced capacity loss over the 15-year lifetime and is consistent with reported '
                    'calendar lifetimes exceeding 20 years for LFP cells', [('1', 0), ('14', 0), ('15', 0), ('20', 0)],
          lambda X, w: [sub('cal1062', '1 %/yr replaced by retention 0.985/yr (1.5 %/yr) in the baseline',
                            'input replaced')(X, w),
                        sub('cal1062', 'derived from the replaced 1 %/yr', 'input replaced')(X, w),
                        count_eq(w[2], X.ess['ageing']['calendar_life_years'],
                                 f'{X.src("ESS_PARAMS")} ageing.calendar_life_years'),
                        unch('literature value', 'reported LFP calendar lifetimes (literature claim in a sentence '
                             'the map replaces)')]),
        D('M15', M, 'annual rates of 0.5\\% and 2.0\\% are additionally considered', [('0.5', 0), ('2.0', 0)],
          subs(2, 'cal1062', 'the 0.5 % / 2 % sensitivities were never run; removed (map section A last row)',
               'input replaced')),
        D('M16', M, 'reduces operating costs by up to 18.25\\% and RES curtailment by up to 92.16\\%',
          [('18.25', 0), ('92.16', 0)], subs(2, 'a1825', 'submitted figures in the conclusions (paragraphs 2-3 '
                                                          'rewritten, map section B 5)')),
    ]


MAP_CITES = {
    'a_header': '| submitted claim | revised result',
    'a_plan': '| optimal plan 1.62 MVA / 3.24 MWh at node 7',
    'a_benders': '| Benders, 4 iterations, 0.50 % gap',
    'a1825': '| 18.25 % / 92.16 % vs uncoordinated',
    'a_disc': '| discretization sensitivity (5-year vs 1-year)',
    'a_cal': '| calendar ageing 1 %/yr with 0.5 % / 2 % sensitivities',
    'abstract': '- **Abstract** (l. 139–148): rewrite.',
    'highlights': '- **Highlights** (l. 157–163)',
    'alg1': 'Algorithm 1 (l. 398–430,',
    'master': '- **2.2 Master problem** (l. 544–723)',
    'cuts': "**2.2.7 Benders' cuts (l. 684–723): delete entirely**",
    'subproblem': '- **2.3 Subproblem** (l. 724–904)',
    'instance': '- Instance as run: IEEE 9-bus TN',
    'years': 'Replace the 2025/2028/…/2037 columns everywhere',
    'cost': '- **3.1 Investment costs** (l. 945–969)',
    'networks': '- **3.3–3.4 Networks** (l. 974–1067)',
    'cal1062': '**l. 1062 calendar ageing sentence**',
    'results': '### 4. Results (l. 1068–1390) — replace wholesale',
    'r47': '- **4.7 Certification statistics and computational performance.** Replace "four Benders iterations,',
    'r_delete': '- **Delete**: 4.5 "Impact of planning horizon discretization" (l. 1282–1370)',
    'keyins': '- **Key insights** (l. 1371–1390)',
    'concl': '### 5. Conclusions (l. 1391–1407)',
    'appA': '- **A. TSO–DSO coordinated operational planning** (l. 1409–1611)',
    'appBCD': '- **B. Market data, C. TN, D. DNs** (l. 1612–1972): keep',
    'appE': '- **E. Results** (l. 1973–2086)',
}

# The line-based main.tex rules below (MAIN_REGIONS and the line ranges in rule_assign) are written for this main.tex
# (Overleaf 6cb4492 = 6191c6c, the submitted text the revision map's line numbers refer to). For any other main.tex they
# are NOT applied and the run fails the check `main_line_rules_pinned_to_this_main` (exit 3): revise them first.
MAIN_LINE_RULES_SHA = '3cedbb6e37bfe31bf7812a76fcc93e7030053e9c7b80793a454ba7e11c965e19'
# main.tex line ranges the map removes or replaces (submitted-version figures), checked in this order
MAIN_REGIONS = (
    ((949, 966), 'cost', 'input replaced', 'investment-cost table of the three submitted cost trajectories'),
    ((1131, 1143), 'r47', 'result', 'Benders convergence and run time'),
    ((1144, 1281), 'r_delete', 'result', 'operational-planning subsections built on the old recourse'),
    ((1282, 1370), 'r_delete', 'result', 'planning-horizon discretization (not run under the revised method)'),
    ((1371, 1390), 'keyins', 'result', 'key insights of the submitted results'),
    ((1068, 1130), 'results', 'result', 'submitted investment plan and degradation results'),
    ((1973, 2086), 'appE', 'result', 'appendix results breakdown'),
)


def all_declarations():
    return letter_declarations() + cover_declarations() + main_declarations()


# ======================================================================================================================
#  5. rules (applied to tokens no declaration claims), per file role
# ======================================================================================================================
XREF_RX = re.compile(r'(?:Sections?|Appendix|Algorithm|Figures?|Fig\.|Tables?|Eqs?\.|Equations?|Reviewers?)'
                     r'(?:~|\s)*[\[(]?\d+(?:\.\d+)*[\])]?'
                     r'(?:\s*(?:,|and|--|-|–|or)\s*[\[(]?\d+(?:\.\d+)*[\])]?)*')


def _in_xref(doc, t):
    if t['scope'] != 'body':
        return False
    ln = doc.lines[t['line'] - 1]
    for m in XREF_RX.finditer(ln):
        if m.start() <= t['col0'] < m.end():
            return True
    return False


def _compound(doc, t):
    ln = doc.lines[t['line'] - 1]
    return t['type'] == 'word' and re.match(r'-[A-Za-z]', ln[t['end0']:t['end0'] + 2] or '') is not None


def _map_sub(X, key, kind, note):
    ml = X.map_line(MAP_CITES[key])
    return R('submitted-version figure', f'{MAP_REL} line {ml}', 'revision map', None, None, note,
             cite=f'{MAP_REL}:{ml}', category=kind)


def rule_assign(X, doc, role, t, map_xref_numbers):
    """(result) for a token no declaration claims; None = unassigned."""
    typ, ln, txt = t['type'], t['line'], t['text']
    line = doc.lines[ln - 1]
    # ---- comments --------------------------------------------------------------------------------------------------
    if t['scope'] == 'comment':
        if role == 'main':
            sub_ = 'template boilerplate' if ln <= 60 or line.lstrip().startswith('%%') else 'commented-out text'
            return unch('comment (not typeset)', f'{sub_}; comments are not manuscript text')
        return None
    lines_ok = role == 'main' and X.main_sha == MAIN_LINE_RULES_SHA
    # ---- main.tex: passages the revision map removes or replaces (before every other rule) ---------------------------
    if lines_ok:
        for (a, b), key, kind, why in MAIN_REGIONS:
            if a <= ln <= b:
                return _map_sub(X, key, kind, why)
    # ---- letter: reviewer quotations --------------------------------------------------------------------------------
    if role == 'letter' and t['macro'] == 'rcomment':
        note = 'reviewer quotation (the reviewers\' documents are not in the repository, map section C)'
        if X.main is not None and typ == 'num':
            hits = [x for x in X.main_tokens if x['type'] == 'num' and x['written'].lstrip('+-') == t['written']]
            if hits and txt in ('18.25', '92.16', '0.50'):
                note += f"; the figure appears in submitted main.tex lines {sorted({h['line'] for h in hits})}"
        return unch('reviewer quotation', note)
    # ---- identifiers ------------------------------------------------------------------------------------------------
    if typ == 'doi':
        return unch('reference identifier', 'DOI (reference-list identifier)')
    if typ == 'label':
        if re.fullmatch(r'R\d\.\d+', txt):
            return unch('enumerator', 'reviewer item label')
        if re.fullmatch(r'T\d+', txt):
            return unch('enumerator', 'table identifier of the frozen tables (placeholder [T..] in the letter)')
        if re.fullmatch(r'C\d', txt):
            return unch('enumerator', 'ageing calibration name (T8 arm labels C2 / C2_calfade)')
        return unch('identifier', 'label containing digits')
    if typ in ('hash', 'date', 'alnum'):
        return None if role != 'main' else unch('identifier', f'{typ} in the submitted text')
    # ---- cross-references written in prose --------------------------------------------------------------------------
    if typ in ('num', 'dotted') and _in_xref(doc, t):
        found = txt in map_xref_numbers
        return unch('cross-reference', 'section / table / figure / equation / algorithm / reviewer number written in '
                    'prose' + (f'; the revision map names section {txt}' if found and role == 'letter' else ''))
    if role == 'letter' and typ == 'num' and re.search(r'\\section\*\{Reviewer ' + re.escape(txt) + r'\}', line):
        return unch('enumerator', 'reviewer number (section heading)')
    if role == 'letter' and typ == 'num' and re.search(r'revision ' + re.escape(txt) + r'\}', line):
        return unch('enumerator', 'revision number (title block)')
    # ---- number words -----------------------------------------------------------------------------------------------
    if typ == 'word' and _compound(doc, t):
        return unch('not a figure', f'compound adjective ({txt}{line[t["end0"]:t["end0"] + 12].split()[0]})')
    if typ == 'word' and txt.lower() == 'single':
        return unch('not a figure', 'article sense ("a single ...")')
    if role != 'main':
        return None
    # ---- main.tex ---------------------------------------------------------------------------------------------------
    if typ == 'num' and re.fullmatch(r'(19|20)\d\d', txt):
        if txt in X.years():
            return R('match', f'{X.src("SRP1_JSON")} Years {sorted(X.years())}', 'named record', sorted(X.years()), txt,
                     'a representative year of the revised instance')
        return _map_sub(X, 'years', 'input replaced', 'non-representative year of the submitted horizon')
    if typ == 'num' and re.search(r'IEEE\s*$', line[:t['col0']]) and line[t['end0']:t['end0'] + 4] == '-bus':
        return _bus(X, 'case9', t['written']) if txt == '9' else _bus(X, 'case33', t['written']) if txt == '33' else None
    if typ == 'num' and re.search(r'[Nn]odes?(?:~|\s)*(?:\d+\s*,\s*(?:and\s*)?|\d+\s*,?\s*and\s*)*$', line[:t['col0']]) \
            and 'tabular' not in ' '.join(t['envs']):
        ids = sorted(d['connection_node_id'] for d in X.srp1['DistributionNetworks'])
        return R('match' if int(float(txt)) in ids else 'MISMATCH', f'{X.src("SRP1_JSON")} '
                 f'DistributionNetworks[*].connection_node_id {ids}', 'named record', ids, txt,
                 'TN interface node of a distribution network (an identifier)')
    if 'postcode=' in line:
        return unch('address', 'postcode of an affiliation')
    if not lines_ok:
        return None
    if 'algorithm' in t['envs']:
        ln1 = X.map_line(MAP_CITES['alg1'] if ln < 1000 else MAP_CITES['appA'])
        return unch('notation', f'constant in an algorithm (replaced or audited: {MAP_REL} line {ln1})')
    if t['in_math']:
        key = 'cuts' if 684 <= ln <= 723 else 'subproblem' if 724 <= ln <= 904 else \
            'appA' if 1409 <= ln <= 1611 else 'master' if 544 <= ln <= 683 else None
        extra = f'; passage cited by {MAP_REL} line {X.map_line(MAP_CITES[key])}' if key else ''
        return unch('notation', 'formula constant (equations are audited against the code, W166)' + extra)
    if any(e.startswith('tabular') for e in t['envs']):
        if 974 <= ln <= 1067:
            return unch('network/data parameter', 'case-data table (RES capacity / generator and node IDs), outside the '
                        f'frozen tables; to be regenerated for 2025/2030/2035 ({MAP_REL} line '
                        f'{X.map_line(MAP_CITES["years"])})')
        if 1612 <= ln <= 1972:
            return unch('network/data parameter', 'IEEE 33-bus / market data table, outside the frozen tables '
                        f'({MAP_REL} line {X.map_line(MAP_CITES["appBCD"])}: keep)')
    if typ == 'word':
        if txt.lower() == 'third':
            return unch('enumerator', 'ordinal word')
        return unch('descriptive count', 'number word in unrevised submitted prose (counts of lists, layers, agents, '
                    'reference cases); wording checks are not required this round')
    return None


# ======================================================================================================================
#  6. guards, post-run, summary
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(W161.PICKLE_COUNTS), w162=dict(W162.PICKLE_COUNTS),
                  w163=dict(W163.PICKLE_COUNTS), w164=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W164 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W164 post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " +
         ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def md_summary(o):
    s = o['summary']
    L = ['# W164 -- number check of the manuscript .tex files', '',
         f"Overleaf clone `{o['manuscript']['dir']}` at commit `{o['manuscript']['commit']}` (declared "
         f"`{o['manuscript']['declared_commit']}`); files: " +
         ', '.join(f"`{f['name']}` sha256 `{f['sha256'][:8]}`" for f in o['manuscript']['files']) + '.',
         f"Frozen tables `{os.path.basename(FZ_REL)}` (sha256 `{FZ_SHA[:8]}`). Script `{SCRIPT_REL}` (imports "
         f"`{W163_SCRIPT}`, not edited). ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.",
         '', 'Statuses: match (declared / rule against a named record / auto-unique / auto-ambiguous), MISMATCH, '
             'approximate, no table counterpart, '
             'submitted-version figure (main.tex only, cites the revision-map line), unchecked (with the reason). '
             'Excluded LaTeX structure is counted separately and is not in scope.', '',
         '## Counts per file', '',
         '| file | tokens | excluded (structure) | in scope (body) | comments | match declared | match rule | '
         'match auto-unique | match auto-ambiguous | MISMATCH | approximate | no table counterpart | '
         'submitted-version | unchecked |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f, c in s['per_file'].items():
        L.append(f"| {f} | {c['tokens']} | {c['excluded']} | {c['in_scope_body']} | {c['in_scope_comment']} | "
                 f"{c['match_declared']} | {c['match_rule']} | {c['match_auto_unique']} | {c['match_auto_ambiguous']} | "
                 f"{c['MISMATCH']} | "
                 f"{c['approximate']} | {c['no table counterpart']} | {c['submitted-version figure']} | "
                 f"{c['unchecked']} |")
    L += ['', f"Every token assigned: **{s['every_token_assigned']}**; stale declarations: {s['stale_declarations']}; "
              f"tokens claimed twice: {s['claimed_twice']}.", '',
          '## MISMATCH (all files)', '']
    mm = [t for t in o['tokens'] if t.get('status') == 'MISMATCH']
    for t in mm:
        L.append(f"- `{t['file']}` line {t['line']} col {t['col']}: written `{t['written']}` -- counterpart "
                 f"{t.get('value_at_written_precision')!r} ({t.get('counterpart')}). {t.get('note') or ''}")
    if not mm:
        L.append('none')
    L += ['', '## Approximate', '']
    ap = [t for t in o['tokens'] if t.get('status') == 'approximate']
    for t in ap:
        L.append(f"- `{t['file']}` line {t['line']}: written `{t['written']}`, value {t.get('counterpart_value')!r}. "
                 f"{t.get('note') or ''}")
    if not ap:
        L.append('none')
    L += ['', '## No table counterpart', '']
    for t in o['tokens']:
        if t.get('status') == 'no table counterpart':
            L.append(f"- `{t['file']}` line {t['line']}: `{t['written']}` -- {t.get('note') or ''}")
    L += ['', '## Response letter: every number with its source', '',
          '| line | written | scope | status | check | counterpart (at written precision) | source / reason |',
          '|---:|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == LETTER and not t.get('excluded'):
            src = (t.get('counterpart') or t.get('note') or '').replace('|', '/')
            L.append(f"| {t['line']} | {t['written']} | {t['scope']}{'/' + t['macro'] if t.get('macro') else ''} | "
                     f"{t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                     f"{str(t.get('value_at_written_precision') if t.get('value_at_written_precision') is not None else '').replace('|', '/')[:60]} | "
                     f"{src[:260]} |")
    L += ['', '## Response letter: findings (sentences against a table, a record or the letter itself)', '']
    for f in o['letter_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    for fname in (HIGHLIGHTS, COVER):
        L += ['', f'## {fname}: every number with its source', '']
        rows = [t for t in o['tokens'] if t['file'] == fname and not t.get('excluded')]
        for t in rows:
            L.append(f"- line {t['line']} `{t['written']}`: {t['status_display']} -- "
                     f"{t.get('counterpart') or t.get('note') or ''}")
        if not rows:
            L.append('no in-scope numeric token (structure only: ' + ', '.join(
                f"`{t['text']}` ({t['excluded']})" for t in o['tokens'] if t['file'] == fname) + ')')
    L += ['', '## main.tex: submitted-version figures by section', '', '| section | n | map lines cited |',
          '|---|---:|---|']
    for sec, v in s['main_submitted_by_section'].items():
        L.append(f"| {sec} | {v['n']} | {', '.join(v['cites'])} |")
    L += ['']
    by_sec = collections.OrderedDict()
    for t in o['tokens']:
        if t['file'] == MAIN and t.get('status') == 'submitted-version figure':
            by_sec.setdefault(t['section'].split(' > ')[0], []).append(f"l.{t['line']} {t['written']}")
    for sec, v in by_sec.items():
        L.append(f'- **{sec}** ({len(v)}): ' + ', '.join(v))
    L += ['', '## main.tex: statuses by section', '', '| section | status | n |', '|---|---|---:|']
    for sec, v in s['main_status_by_section'].items():
        for st, n in v.items():
            L.append(f'| {sec} | {st} | {n} |')
    L += ['', '## main.tex: auto-index matches', '']
    for t in o['tokens']:
        if t['file'] == MAIN and t.get('match_kind', '').startswith('auto'):
            L.append(f"- line {t['line']} `{t['written']}`: {t['status_display']} "
                     f"{[c['path'] for c in t['auto_candidates'][:5]]}")
    L += ['', f"Tokens assigned another status by precedence although the auto index has >= 1 candidate: "
              f"{s['auto_candidates_overridden']}", '',
          '## Unchecked: by category (all files)', '', '| file | category | n |', '|---|---|---:|']
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
#  7. main
# ======================================================================================================================
def letter_findings(X, by_key):
    vc = X.verdict_changes()
    t9 = {r['cell']: r for r in X.t['dead_zone']['cells']}
    cells = {**X.t['cells'], **X.t['cells_appended_w160']}
    unc = sorted(k for k, r in cells.items() if r.get('certification_stats_included') and r.get('status') == 'uncertified')
    dz = sorted(k for k in unc if k in t9 and ('gap-refused' in t9[k]['entry'] or 'signature' in t9[k]['entry']))
    figs = sorted((X.main.blank.count('\n', 0, a) + 1, nm) for nm, a, b in X.main.envs if nm in ('figure', 'figure*'))
    figs = [(nm, ln) for ln, nm in figs]
    F = [
        {'id': 'F1', 'lines': '254-256', 'kind': 'wording against a table',
         'finding': f'"one verdict (a Phase~B neighbour becomes determinate in the certificate\'s favour)": the one T1 '
                    f'claim whose verdict changes gross -> net is {vc[0]["claim_id"] if vc else None} (family '
                    f'{vc[0]["family"] if vc else None}: "{vc[0]["statement"][:90] if vc else ""}..."), an F2-certificate '
                    'neighbour of the m = 2 Phase B incumbent, both cells uncertified; T1 labels "Phase B certificate" '
                    'its C-family rows (against x = 0), which do not change. Count and direction match; the label '
                    'is ambiguous.'},
        {'id': 'F2', 'lines': '256', 'kind': 'claim against a table',
         'finding': '"The investment-year comparison is the one result that depends on the convention": T4 2035 - 2030 '
                    f"(gross {X.t['year_ladder']['gross_v6']['verdict']}, net {X.t['year_ladder']['net_v6']['verdict']}) "
                    f"is one; T1 carries a second, {vc[0]['claim_id'] if vc else None} (gross {vc[0]['gross'] if vc else None}, "
                    f"net {vc[0]['net'] if vc else None}), which the preceding sentence itself counts. Token status MISMATCH "
                    '(count 2 against "one").'},
        {'id': 'F3', 'lines': '327-328', 'kind': 'claim about a table',
         'finding': '"Table [T8] gives, per calibration, the year in which the end-of-life floor binds and the terminal '
                    'available-energy fraction, together with the equivalent full cycles per day": T8 carries "floor '
                    'binds (0.70)", "AE (PV-weighted)" and "EFC/day (PV-weighted)" -- the AE column is present-value-'
                    'weighted, not terminal (no terminal available-energy column in T8).'},
        {'id': 'F4', 'lines': '212-216', 'kind': 'internal contradiction',
         'finding': '"The three references were added to the literature review" against the bracketed status in the '
                    'same response: 10.1109/ISGTEUROPE62998.2024.10863557 and 10.1016/j.apenergy.2022.120569 "not yet '
                    'added" (author placeholder).'},
        {'id': 'F5', 'lines': '320-322', 'kind': 'wording against a table',
         'finding': '"harsher calibrations -73.7 and -45.2 k EUR": C4 (-45.2) loses less than the baseline C2_calfade '
                    '(-64.4); it is harsher than C2 (cycling only, -31.6), not than the baseline. T8\'s sixth arm '
                    f"C3_midblock ({X.ageing['C3_midblock']['value_minus_I'] / 1e3:.1f} k EUR, "
                    f"{X.ageing['C3_midblock']['verdict']}) is not named. Every named value matches T8."},
        {'id': 'F6', 'lines': '158-160', 'kind': 'wording against a record',
         'finding': '"one local problem per network block (three representative years x four representative days x '
                    'four agents ..., 48 blocks)": 3 x 4 x (1 TSO + 3 DSO) = 48 network blocks matches SRP1.json; '
                    'the storage-operator (ESSO) problems solved in the same cycle are not counted (TASKS.md W132 / '
                    'W133 lines: 51 exits per cycle including the ESSO; not re-read from the solve records here).'},
        {'id': 'F7', 'lines': '127-132', 'kind': 'not verifiable here',
         'finding': f'"Figure 2" (R1.5) is answered as the framework figure. In main.tex at this commit the framework '
                    f'figure is the first figure environment (line {figs[0][1] if figs else None}, blue caption) and the '
                    f'second is the network diagram (line {figs[1][1] if len(figs) > 1 else None}); the numbering of the '
                    'submitted PDF the reviewer read cannot be re-derived from this file.'},
        {'id': 'F8', 'lines': '24, 99-100', 'kind': 'internal inconsistency',
         'finding': 'the title block says "revision 1" while R1.2 says "we respectfully maintain the position of our '
                    'first reply"; one of the two is wrong unless an earlier reply exists.'},
        {'id': 'F9', 'lines': '360', 'kind': 'no record',
         'finding': '"single machine": no committed record states it (W163 listed "one machine" as NO SOURCE FOUND).'},
        {'id': 'K1', 'lines': '195-197', 'kind': 'claim checked, consistent',
         'finding': f'"the limitations section explains the mechanism behind most of them (a set-valued interface '
                    f'dual ...)": {len(dz)} of the {len(unc)} uncertified SRP1 evaluations are T9 dead-zone entries '
                    f'(gap-refused or by signature: {dz}); the others {sorted(set(unc) - set(dz))} are not.'},
        {'id': 'K2', 'lines': '253-255', 'kind': 'claim checked, consistent',
         'finding': f'"It changes no sign in the 60 reported comparisons": sign changes gross -> net in T1: '
                    f'{len(X.sign_changes())}.'},
    ]
    return F


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manuscript-dir', default=DEFAULT_MANUSCRIPT_DIR)
    ap.add_argument('--overleaf-commit', required=True, help='the Overleaf commit the run declares (>= 7 hex)')
    ap.add_argument('--expect', action='append', default=[], help='NAME=SHA256 for every *.tex file (declared)')
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or os.path.join(OUT_ROOT, f'overleaf_{args.overleaf_commit[:7]}')
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W164'
    pre = []
    # ---- the output directory: created by the launch, holding only the launch log -----------------------------------
    od = os.path.join(REPO, out_dir) if not os.path.isabs(out_dir) else out_dir
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        extra = sorted(set(os.listdir(od)) - {OUT_NAMES['log']})
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
    if not re.fullmatch(r'[0-9a-f]{7,40}', args.overleaf_commit):
        pre.append(f'--overleaf-commit {args.overleaf_commit!r} is not a hex commit')
    # ---- the manuscript: commit, file set, sha, blobs -----------------------------------------------------------------
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
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex'], capture_output=True,
                               text=True).stdout.strip()
    mfiles = []
    for f in files:
        with open(os.path.join(mabs, f), 'rb') as h:
            b = h.read()
        blob = subprocess.run(['git', '-C', mabs, 'show', f'{args.overleaf_commit}:{f}'], capture_output=True)
        mfiles.append({'name': f, 'path': os.path.join(mdir, f), 'sha256': _sha_bytes(b), 'bytes': len(b),
                       'lines': b.decode('utf-8').count('\n') + 1,
                       'blob_at_commit_sha256': _sha_bytes(blob.stdout) if blob.returncode == 0 else None,
                       'declared_sha256': expect.get(f)})
    mchecks = {
        'clone_head_is_declared_commit': bool(head) and head.startswith(args.overleaf_commit),
        'file_set_equals_declared': set(files) == set(expect),
        'every_sha_equals_declared': all(m['sha256'] == m['declared_sha256'] for m in mfiles),
        'every_file_equals_its_blob_at_commit': all(m['sha256'] == m['blob_at_commit_sha256'] for m in mfiles),
        'no_tex_modified_or_untracked_in_clone': porcelain == '',
    }
    for k, v in mchecks.items():
        if not v:
            pre.append(f'manuscript: {k} FAILED (head {head}, files {files}, declared {sorted(expect)}, porcelain '
                       f'{porcelain!r}, shas {[(m["name"], m["sha256"][:8], (m["declared_sha256"] or "")[:8]) for m in mfiles]})')
    # ---- the records ---------------------------------------------------------------------------------------------------
    input_rels = {'FROZEN_JSON': FZ_REL, 'FROZEN_MD': FZ_MD_REL, 'W163_JSON': W163_JSON, 'W163_MANIFEST': W163_MAN,
                  'W163_SCRIPT': W163_SCRIPT, 'W162_SCRIPT': W162.SCRIPT_REL, 'W161_SCRIPT': W161.SCRIPT_REL
                  if hasattr(W161, 'SCRIPT_REL') else 'p515_s53_w161_paragraphs_v2.py',
                  'W160_SCRIPT': 'p515_s53_w160_step6_freeze_export.py', 'P5': P5_REL, 'P2': P2_REL, 'MAP': MAP_REL,
                  'SRP1_JSON': SRP1_JSON, 'SRP1_PARAMS': SRP1_PARAMS, 'ESS_PARAMS': ESS_PARAMS, 'STEP4': STEP4,
                  'A1_CAMPAIGN': A1_CAMPAIGN, 'V41_SPEC': V41_SPEC}
    for k, v in CASE_FILES.items():
        input_rels[f'CASE_{k}'] = v
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if not pre:
        for key, want in (('FROZEN_JSON', FZ_SHA), ('P5', P5_SHA), ('P2', P2_SHA)):
            if inputs[key]['sha256'] != want:
                pre.append(f"{inputs[key]['path']} sha {inputs[key]['sha256']} != {want}")
        if _jl(W163_MAN).get(W163_JSON) != inputs['W163_JSON']['sha256']:
            pre.append(f'{W163_JSON} != its W163 manifest entry')
        if not (inputs['W163_SCRIPT']['last_commit'] or '').startswith(W163_SCRIPT_COMMIT):
            pre.append(f"{W163_SCRIPT} last commit {inputs['W163_SCRIPT']['last_commit']} != {W163_SCRIPT_COMMIT}")
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; {len(inputs)} inputs '
         f'committed clean; frozen JSON {FZ_SHA[:8]}, W163 JSON {inputs["W163_JSON"]["sha256"][:8]}, paragraphs_v5 '
         f'{P5_SHA[:8]} verified; manuscript {mdir} HEAD {head} (declared {args.overleaf_commit}); files ' +
         ', '.join(f"{m['name']} {m['sha256'][:8]}" for m in mfiles))
    # ---- tokenize --------------------------------------------------------------------------------------------------------
    docs = {m['name']: Doc(m['name'], _text(m['path'])) for m in mfiles}
    roles = {n: ('main' if n == MAIN else 'letter' if n.startswith('response_to_reviewers') else
                 'highlights' if n == HIGHLIGHTS else 'cover' if n == COVER else 'other') for n in docs}
    toks = []
    for n in files:
        toks += tokenize(docs[n])
    X = Records(inputs, docs.get(MAIN))
    X.letter_doc = docs.get(LETTER)
    X.letter_raw = docs[LETTER].raw if LETTER in docs else ''
    X.cover_raw = docs[COVER].raw if COVER in docs else ''
    X.main_tokens = [t for t in toks if t['file'] == MAIN and t['scope'] == 'body' and not t['excluded']]
    X.main_sha = next((m['sha256'] for m in mfiles if m['name'] == MAIN), None)
    leaves = json_leaves(X.fz)
    AX = AutoIndex(leaves)
    map_cite_lines, map_cite_fail = {}, []
    for k, frag in MAP_CITES.items():
        try:
            map_cite_lines[k] = X.map_line(frag)
        except KeyError as e:
            map_cite_fail.append(str(e))
    map_xref_numbers = set(re.findall(r'(?<![\d.])(\d\.\d(?:\.\d)?)(?![\d.])', '\n'.join(X.map)))
    # ---- declared checks -------------------------------------------------------------------------------------------------
    by_key = {t['key']: t for t in toks}
    assigned, claimed_twice, stale = {}, [], []
    decl_out = []
    for d in all_declarations():
        if d['file'] not in docs:
            stale.append({'id': d['id'], 'why': f"{d['file']} not in the manuscript directory"})
            continue
        doc = docs[d['file']]
        hits, flen = doc.find_fragment(d['fragment'])
        if len(hits) != 1:
            stale.append({'id': d['id'], 'why': f'fragment found {len(hits)} times', 'fragment': d['fragment']})
            continue
        a = hits[0]
        inside = [t for t in toks if t['file'] == d['file'] and not t['excluded'] and
                  doc.flat_pos(t['line'], t['col0']) is not None and a <= doc.flat_pos(t['line'], t['col0']) < a + flen]
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
                         'tokens': [[t['line'], t['col'], t['written']] for t in picked]})
        for t, r in zip(picked, res):
            if t['key'] in assigned:
                claimed_twice.append([t['key'], assigned[t['key']]['check_id'], d['id']])
                continue
            r = dict(r)
            r['check_id'] = d['id']
            r['rule'] = 'declared'
            assigned[t['key']] = r
    # ---- rules, auto index -------------------------------------------------------------------------------------------
    for t in toks:
        t['auto_candidates'] = AX.candidates(t) if not t['excluded'] else None
        if t['excluded']:
            t['status'], t['status_display'] = 'excluded', f"excluded ({t['excluded']})"
            continue
        r = assigned.get(t['key'])
        if r is None:
            r = rule_assign(X, docs[t['file']], roles[t['file']], t, map_xref_numbers)
            if r is not None:
                r = dict(r)
                r['rule'] = 'rule'
        if r is not None and r['status'] == 'match' and r['rule'] == 'rule':
            r['match_kind'] = 'rule (named record)'
        if r is None and roles[t['file']] == 'main' and t['auto_candidates']:
            c = t['auto_candidates']
            r = R('match', c[0]['path'] if len(c) == 1 else f'{len(c)} paths', 'frozen JSON (auto index)', None, None,
                  None)
            r['rule'], r['match_kind'] = 'auto', 'auto-unique' if len(c) == 1 else 'auto-ambiguous'
        if r is None and roles[t['file']] == 'main' and X.main_sha == MAIN_LINE_RULES_SHA:
            r = ntc('auto index: no candidate; no declaration or rule this round (main.tex is the submitted text)')
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
                               else t['status'])
    unassigned = [t['key'] for t in toks if not t['excluded'] and t['status'] is None]
    # ---- summary -------------------------------------------------------------------------------------------------------
    per_file = {}
    for n in files:
        ft = [t for t in toks if t['file'] == n]
        sc = [t for t in ft if not t['excluded']]
        c = {'tokens': len(ft), 'excluded': len(ft) - len(sc),
             'in_scope_body': sum(t['scope'] == 'body' for t in sc),
             'in_scope_comment': sum(t['scope'] == 'comment' for t in sc),
             'match_declared': sum(t['status'] == 'match' and t.get('match_kind') == 'declared' for t in sc),
             'match_rule': sum(t['status'] == 'match' and t.get('match_kind') == 'rule (named record)' for t in sc),
             'match_auto_unique': sum(t.get('match_kind') == 'auto-unique' for t in sc),
             'match_auto_ambiguous': sum(t.get('match_kind') == 'auto-ambiguous' for t in sc),
             'unassigned': sum(t['status'] is None for t in sc)}
        for st in STATUSES[1:]:
            c[st] = sum(t['status'] == st for t in sc)
        c['match_total'] = c['match_declared'] + c['match_rule'] + c['match_auto_unique'] + c['match_auto_ambiguous']
        c['assigned_total'] = c['match_total'] + sum(c[st] for st in STATUSES[1:])
        per_file[n] = c
    sub_by_sec = collections.OrderedDict()
    stat_by_sec = collections.OrderedDict()
    for t in toks:
        if t['file'] != MAIN or t['excluded']:
            continue
        sec = t['section'].split(' > ')[0] if t['scope'] == 'body' else '(comments)'
        stat_by_sec.setdefault(sec, collections.Counter())[
            f"match ({t['match_kind']})" if t['status'] == 'match' else t['status_display']] += 1
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
            d_ = unc_cat.setdefault(t['file'], collections.Counter())
            d_[t.get('category') or 'other'] += 1
    overridden = collections.Counter(t['status'] or 'UNASSIGNED' for t in toks if not t['excluded'] and t['auto_candidates'] and
                                     not t.get('match_kind', '').startswith('auto'))
    summary = {'per_file': per_file, 'every_token_assigned': not unassigned,
               'unassigned': unassigned, 'stale_declarations': stale, 'claimed_twice': claimed_twice,
               'main_submitted_by_section': sub_by_sec,
               'main_status_by_section': {k: dict(v) for k, v in stat_by_sec.items()},
               'unchecked_by_category': {k: dict(v) for k, v in unc_cat.items()},
               'excluded_by_category': exc_cat, 'auto_candidates_overridden': dict(overridden),
               'n_frozen_json_numeric_leaves': len(leaves), 'n_declarations': len(all_declarations()),
               'n_declarations_applied': len(decl_out)}
    findings = letter_findings(X, by_key) if LETTER in docs and MAIN in docs else []
    checks = dict(mchecks)
    checks.update({
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'every_declaration_found_and_evaluated': not stale,
        'no_token_claimed_twice': not claimed_twice,
        'every_token_assigned': not unassigned,
        'every_map_citation_found_once': not map_cite_fail,
        'statuses_in_vocabulary': all(t['excluded'] or t['status'] in STATUSES for t in toks if t['status']),
        'main_line_rules_pinned_to_this_main': X.main_sha in (None, MAIN_LINE_RULES_SHA),
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
    out = {'schema': 'p515_s53_w164_manuscript_number_check', 'version': 1,
           'stage': 'P5.15 Step 6 W164 -- number check of the manuscript .tex files (W163 checker extended)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {W163_SCRIPT: _sha(W163_SCRIPT)}},
           'command_line': sys.argv,
           'manuscript': {'dir': mdir, 'commit': head, 'declared_commit': args.overleaf_commit, 'files': mfiles,
                          'roles': roles, 'clone_porcelain_tex': porcelain, 'edited': False},
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA, 'numeric_leaves': len(leaves)},
           'inputs': inputs, 'map_citations': map_cite_lines, 'map_citation_failures': map_cite_fail,
           'conventions': {
               'column': '1-based', 'line': '1-based', 'occ': 'occurrence of the same text on the same line and scope',
               'written': 'the token as written, sign included, LaTeX thousands separators as ","',
               'precedence': 'excluded structure -> comments -> declared checks -> rules (reviewer quotations, '
                             'identifiers, cross-references, number words; main.tex: map regions, years, IEEE n-bus, '
                             'algorithms, math, data tables) -> auto index (main.tex) -> leftover (main.tex: no table '
                             'counterpart). The auto index is computed for every numeric token and recorded beside its '
                             'status.',
               'auto_index': 'every numeric leaf of the frozen JSON (bools excluded); transforms raw always, x100 if '
                             'the token is followed by %, /1e3 (only) for k EUR, /1e6 (only) for M EUR; matched at the '
                             'written precision, signed, or |leaf| when the token is written without a sign',
               'objective_convention': X.fz.get('objective_convention')},
           'summary': summary, 'checks': checks, 'failed': failed, 'declarations_applied': decl_out,
           'letter_findings': findings, 'tokens': toks, 'guards': guards, 'pickle_guard': pk, 'exit_code': code,
           'wall_s': time.time() - t0}
    written = {}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        with open(os.path.join(REPO, rel), 'xb') as h:      # an absolute trial path survives the join
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel)
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    for m in mfiles:
        man[m['path']] = m['sha256']
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log ----------------------------------------------------------------------------------------------------------
    for n, c in per_file.items():
        _log(f"[{tag}] {n}: tokens {c['tokens']} (excluded {c['excluded']}, in scope {c['in_scope_body']} body + "
             f"{c['in_scope_comment']} comment) -- match {c['match_total']} (declared {c['match_declared']}, rule "
             f"{c['match_rule']}, auto unique "
             f"{c['match_auto_unique']}, auto ambiguous {c['match_auto_ambiguous']}), MISMATCH {c['MISMATCH']}, "
             f"approximate {c['approximate']}, no table counterpart {c['no table counterpart']}, submitted-version "
             f"{c['submitted-version figure']}, unchecked {c['unchecked']}, unassigned {c['unassigned']}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no table counterpart') and t['file'] != MAIN:
            _log(f"[{tag}]   {t['file']}:{t['line']}:{t['col']} {t['status']}: written {t['written']!r} -> "
                 f"{t.get('value_at_written_precision')!r} [{t.get('check_id')}] {t.get('counterpart')}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'no table counterpart') and t['file'] == MAIN and t.get('rule') == 'declared':
            _log(f"[{tag}]   {t['file']}:{t['line']}:{t['col']} {t['status']}: written {t['written']!r} "
                 f"[{t.get('check_id')}] {t.get('note')}")
    _log(f"[{tag}] main.tex submitted-version figures by section: "
         f"{ {k: v['n'] for k, v in sub_by_sec.items()} }")
    _log(f"[{tag}] auto index: {len(leaves)} numeric leaves; overridden by precedence (status: n) {dict(overridden)}")
    for f in findings:
        _log(f"[{tag}]   letter finding {f['id']} (l. {f['lines']}, {f['kind']}): {f['finding'][:300]}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if unassigned:
        _log(f'[{tag}] UNASSIGNED tokens ({len(unassigned)}): ' + ', '.join(
            f"{by_key[k]['key']} {by_key[k]['written']!r}" for k in unassigned[:200]))
    if stale or claimed_twice or map_cite_fail:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; map citation failures {map_cite_fail}')
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
