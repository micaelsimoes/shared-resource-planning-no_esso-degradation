"""P5.15 Step 6 support, Planner task W167 -- NOMENCLATURE AUDIT, YEAR-DEPENDENT TABLES FOR 2025 / 2030 / 2035, AND
QUESTION (4) (THE INVESTMENT-COST FILE). ZERO SOLVES, NO MODEL LOADS. A NEW FILE; no production module, case file,
spec, frozen JSON, manuscript file or committed artifact is edited.

AUTHORITY. TASKS.md Step 6 support order (W167); STEP6_REVISION_MAP.md section B (Nomenclature, section 3 Case study,
3.1, Appendices B-D) and section E. The manuscript is the Overleaf clone manuscript/6a67305f25e8348fb71380c3/main.tex
(clone HEAD 6191c6c, the SUBMITTED text), pinned by sha256 below; it is only read.

TASK A -- NOMENCLATURE AUDIT (report, never edit).
  Parses the Nomenclature (the two tables between "\\section*{Nomenclature}" and the next "\\section") and every
  math-mode region of the rest of main.tex (comments stripped; $...$, $$...$$, \\[...\\], \\(...\\), and the
  equation / align / gather / multline / eqnarray environments). Each math region is tokenised into ATOMS
  (base, superscript, subscript). NORMALISATION OF SYMBOL IDENTITY (the key used for every comparison):
    * font wrappers \\text \\mathrm \\textrm \\mathit \\operatorname are removed (\\mathrm{Inv} == \\text{Inv} == Inv);
      \\boldsymbol / \\mathbf / \\bm are removed (bold x == x; the bold variant is recorded); \\textcolor{..}{..} is
      transparent; \\mathcal{X} -> cal(X); \\mathbb{R} -> bb(R) (number sets, excluded); decorations are kept as part
      of the base (\\hat{S} -> hat(S); \\widehat == \\hat, \\overline == \\bar, \\widetilde == \\tilde);
    * SUPERSCRIPT: kept as a label after removing whitespace, braces and ITERATION MARKERS -- "(l)", "(k)", "(\\ell)"
      and their +-1 forms, a nested "^k" / "^{k+1}" / "^{\\ell}", and comma-separated items equal to k, l, \\ell,
      k+-1, l+-1, \\ell+-1 or 0 (so S^{Inv(l)} == S^{Inv}, \\pi^{{E,P}^k} == \\pi^{E,P}, V^{I,0} == V^{I});
      a superscript that consisted ONLY of an iteration marker (\\alpha^{(l)}, y^{(k)}) leaves the bare base and the
      atom is flagged iteration_only; \\max -> max;
    * SUBSCRIPT: dropped from the identity (indices: e, y, d, t, {e,y^Inv,y}, ...) EXCEPT a label subscript (starts
      with an upper-case letter: \\Omega_C, \\Omega_O, \\Omega_M) or a digit string (y_0), which is kept. So D_d == D
      and Y_y == Y (the nomenclature itself uses both; reported under (iii) as an overload);
    * a run of letters is split into single-letter atoms unless it is one of MULTI_LETTER (SoH, LB, UB, CL, ...);
    * index letters (single lower-case latin letter or \\ell, no superscript) are classified INDEX, not symbols.
  Outputs (i) defined-never-used, (ii) used-not-defined (first line, count, every line, category), (iii) defined
  twice / overloaded / conflicting (automatic: duplicate nomenclature keys, keys that differ only by a dropped index
  subscript; declared: the Worker's reading of conflicting uses, each VERIFIED against the text by the script --
  a declared occurrence that is not found FAILS the run), (iv) symbols of the removed Benders method with every
  line, (v) the symbols the revision needs (map section B) and whether paragraphs_v5.md uses them, with its notation.

TASK B -- YEAR-DEPENDENT TABLES.
  Enumerates every table / figure / text line of main.tex indexed by a calendar year 2025-2039 (line ranges).
  For every year-indexed INPUT-DATA table (content from the case files) it writes year_tables/<label>.tex: the same
  rows, units, caption and label, columns 2025 / 2030 / 2035 only, read THROUGH PRODUCTION'S READERS:
    * RES capacity: network._read_network_from_json_file on data/SRP1/<net>/<net>_<year>.json (the file
      Network.read_network_from_json_file opens, network.py), capacity = generator.pmax * baseMVA (the gen_capacity
      production scales the RES profile with, network.py `_update_network_with_operational_data`); cross-checked
      against the raw JSON 'Pmax';
    * investment cost: shared_energy_storage_data._read_shared_energy_storage_data_from_file on SRP1_ESS.xlsx (HEAD
      blob == 7ce1d1ab blob, asserted), the scenario probabilities from sheet 'Scenarios'.
  The 2025 column produced is cross-checked CELL BY CELL against the 2025 column printed in the submitted table; the
  printed 2028 / 2031 / 2034 / 2037 columns are also cross-checked against the case files of those years (the
  multi-scenario 3 x 3 instance was run on that five-year horizon; see the instance record). A disagreement is a
  FINDING, reported row by row, never overwritten. Year-indexed RESULTS tables (outputs of the old solves) cannot be
  produced from the case files with zero solves: they are enumerated and classified, not regenerated.
  Year-indexed figures are listed with whether 2030 / 2035 versions would be needed (report only; no figure made).

TASK C -- QUESTION (4).
  SRP1_ESS.xlsx at 7ce1d1ab (git show into a temporary directory, deleted at exit) and at HEAD: identity, every sheet
  (layout, formulas, cached values), cost values for 2025 / 2030 / 2035 (and the printed years), scenario rows and
  probabilities; the cached values recomputed from the workbook's own formulas; the predecessor blob (072b1310) read
  for the change; production's reader and every consumer of the costs on the planning path (file:line located by
  exact text at run time); I(x) of the SRP1 unit candidate recomputed through p56a_oracle.investment_cost and compared
  with the committed record; a scoped search of the committed campaign specs for a cost-scenario selection field.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0;
pickle.load / pickle.loads blocked for the whole run, counters verified at 0. git reads are `git show` / `git log` /
`git rev-parse` / `git ls-files` / `git status --porcelain` only.

MODE (repo root, canonical interpreter; attached, both streams captured; mkdir refuses an existing directory):
    mkdir data/SRP1/Results/P515S53/w167_nomenclature_years && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w167_nomenclature_years.py \\
        > data/SRP1/Results/P515S53/w167_nomenclature_years/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w167_nomenclature_years/w167_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w167_nomenclature_years/w167_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w167_nomenclature_years.py --post-run
  --out-dir DIR   trial runs only (absolute or repo-relative); the directory must hold nothing but launch.log.
Exit: 0 = written, integrity checks hold, guards at 0 (cross-check disagreements are FINDINGS and never change the exit
code); 3 = written, an integrity check failed (listed in the JSON and the log); 1 = precondition or guard fault.
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W167 nomenclature / year tables / cost file (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W167: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W167: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_REL = os.path.basename(__file__)
CLONE_REL = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_REL = os.path.join(CLONE_REL, 'main.tex')
MAIN_SHA = '3cedbb6e37bfe31bf7812a76fcc93e7030053e9c7b80793a454ba7e11c965e19'
MAIN_SHA_AS_GIVEN_IN_TASK = '3cedbb6e39c8'          # the prefix in the W167 order; recorded, compared, reported
CLONE_HEAD_EXPECTED = '6191c6c'
P5_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen', 'export', 'paragraphs_v5.md')
P5_SHA = 'b86657069901ed62ccd80209295bf2d4ecc075a149f47b283148273a9c1a9841'
MAP_REL = 'STEP6_REVISION_MAP.md'
DFO_REL = 'STEP4_DFO_METHOD.md'
FZ_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen',
                      'frozen_step6_tables_v1_590088fe.json')
CASE_DIR = os.path.join('data', 'SRP1')
CASE_JSON_REL = os.path.join(CASE_DIR, 'SRP1.json')
INSTANCE_3X3_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
INSTANCE_3X3_RECORD_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'instance_record.json')
W101_SUMMARY_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w101_srp1_continuation',
                                'w101_three_reference_summary.json')
ESS_XLSX_REL = os.path.join(CASE_DIR, 'SharedESS', 'SRP1_ESS.xlsx')
COST_COMMIT = '7ce1d1ab'
COST_PREDECESSOR_COMMIT = '072b1310'      # the blob the branch carried before Addendum 27 (2cada62b)
TARGET_YEARS = (2025, 2030, 2035)
PRINTED_YEARS = (2025, 2028, 2031, 2034, 2037)
UNIT_CANDIDATE = {'node': 7, 'year': 2025, 's': 0.25, 'e': 1.0}   # n7_4h_e1 (STEP4 lattice unit at 4 h)

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w167_nomenclature_years')
OUT_NAMES = {'audit_json': 'nomenclature_audit.json', 'audit_md': 'nomenclature_audit.md',
             'cross': 'year_tables_crosscheck.json', 'enum': 'year_indexed_items.json',
             'q4': 'question4_cost_file.json', 'sources': 'sources_blob_hashes.json',
             'inputs_man': 'manifest_inputs_sha256.json', 'run': 'run_record.json',
             'man': 'manifest_sha256.json', 'post': 'manifest_post_run_sha256.json',
             'typing_json': 'w167_bool_typing_test.json', 'typing_log': 'w167_bool_typing_test.log',
             'log': 'launch.log'}
YEAR_TABLES_SUB = 'year_tables'

FAILED = []


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(rel):
    with open(os.path.join(REPO, rel), 'rb') as handle:
        return _sha_bytes(handle.read())


def _git(*args, cwd=REPO):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


def _git_bytes(*args, cwd=REPO):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True)
    return r.stdout if r.returncode == 0 else None


def _fail(msg):
    FAILED.append(msg)
    _log(f'[W167 INTEGRITY FAIL] {msg}')


def _norm(x):
    return json.loads(GRIO.dumps(x, sort_keys=True))


def _write_x(path, text):
    with open(path, 'x', encoding='utf-8') as handle:
        handle.write(text)


def _write_json(path, obj):
    _write_x(path, GRIO.dumps(obj, indent=1, sort_keys=False, ensure_ascii=False) + '\n')


def _locate(rel, needle, occurrence=0):
    """file:line of the `occurrence`-th line of `rel` (working tree == HEAD, checked by the caller) containing
    `needle` verbatim; None when absent."""
    with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
        lines = handle.read().split('\n')
    hits = [i + 1 for i, ln in enumerate(lines) if needle in ln]
    if len(hits) <= occurrence:
        return None
    n = hits[occurrence]
    return {'file': rel, 'line': n, 'text': lines[n - 1].strip(), 'n_hits': len(hits)}


# ======================================================================================================================
#  LaTeX reading
# ======================================================================================================================
def read_tex(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
        raw = handle.read()
    lines = raw.split('\n')
    return raw, lines


def strip_comments(lines):
    out = []
    for ln in lines:
        m = re.search(r'(?<!\\)%', ln)
        out.append(ln[:m.start()] if m else ln)
    return out


def section_map(lines):
    """line (1-based) -> the enclosing section / subsection title."""
    cur, out = 'front matter', {}
    app = False
    for i, ln in enumerate(lines, 1):
        if re.match(r'\s*\\appendix', ln):
            app = True
        m = re.match(r'\s*\\(section|subsection|subsubsection)\*?\{(.*)\}', ln)
        if m:
            title = re.sub(r'\\textcolor\{[^}]*\}\{(.*)\}', r'\1', m.group(2))
            cur = ('Appendix: ' if app and m.group(1) == 'section' else '') + f'{m.group(1)} {title}'
        out[i] = cur
    return out


MATH_ENVS = ('equation', 'align', 'gather', 'multline', 'eqnarray')


def math_regions(text):
    """(start, end, content) of every math region of comment-stripped `text`."""
    regions, mask = [], list(text)

    def take(a, b, ca, cb):
        regions.append((a, b, text[ca:cb]))
        for k in range(a, b):
            mask[k] = ' '

    env_re = re.compile(r'\\begin\{(' + '|'.join(MATH_ENVS) + r')(\*?)\}(.*?)\\end\{\1\2\}', re.S)
    for m in env_re.finditer(text):
        take(m.start(), m.end(), m.start(3), m.end(3))
    t2 = ''.join(mask)
    for pat in (r'\\\[(.*?)\\\]', r'\\\((.*?)\\\)', r'(?<!\\)\$\$(.*?)(?<!\\)\$\$'):
        for m in re.finditer(pat, t2, re.S):
            take(m.start(), m.end(), m.start(1), m.end(1))
        t2 = ''.join(mask)
    for m in re.finditer(r'(?<!\\)\$(.+?)(?<!\\)\$', t2, re.S):
        take(m.start(), m.end(), m.start(1), m.end(1))
    regions.sort()
    return regions


# ======================================================================================================================
#  math tokeniser
# ======================================================================================================================
GREEK = {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta', 'iota',
         'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma', 'tau', 'upsilon',
         'phi', 'varphi', 'chi', 'psi', 'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon',
         'Phi', 'Psi', 'Omega', 'ell'}
DECOR = {'hat': 'hat', 'widehat': 'hat', 'bar': 'bar', 'overline': 'bar', 'tilde': 'tilde', 'widetilde': 'tilde',
         'underline': 'underline', 'dot': 'dot', 'ddot': 'ddot', 'check': 'check', 'vec': 'vec', 'breve': 'breve'}
FONT_WORD = {'mathrm', 'text', 'textrm', 'mathit', 'operatorname', 'textit', 'mathsf', 'texttt', 'textnormal'}
FONT_BOLD = {'boldsymbol', 'mathbf', 'bm', 'textbf'}
FONT_CAL = {'mathcal', 'mathscr'}
FONT_BB = {'mathbb'}
SKIP_WITH_GROUP = {'label', 'ref', 'eqref', 'cite', 'tag', 'color'}
IGNORE_CMDS = {'sum', 'prod', 'max', 'min', 'ln', 'log', 'exp', 'frac', 'dfrac', 'tfrac', 'sqrt', 'forall', 'exists',
               'in', 'notin', 'leq', 'geq', 'le', 'ge', 'neq', 'cdot', 'times', 'infty', 'ldots', 'dots', 'cdots',
               'vdots', 'gets', 'to', 'rightarrow', 'leftarrow', 'Rightarrow', 'subseteq', 'subset', 'cup', 'cap',
               'Vert', 'lVert', 'rVert', 'vert', 'mid', 'partial', 'nabla', 'left', 'right', 'big', 'Big', 'bigg',
               'Bigg', 'bigl', 'bigr', 'Bigl', 'Bigr', 'quad', 'qquad', 'nonumber', 'notag', 'approx', 'sim', 'pm',
               'mp', 'setminus', 'emptyset', 'lfloor', 'rfloor', 'lceil', 'rceil', 'langle', 'rangle',
               'displaystyle', 'limits', 'arg', 'argmin', 'colon', 'prime', 'circ', 'ast', 'star', 'cdotp', 'leqslant',
               'geqslant', 'equiv', 'propto', 'land', 'lor', 'neg', 'perp', 'parallel', 'textcolor', 'hfill', 'hspace',
               'vspace', 'nolimits', 'substack', 'begin', 'end', 'mathrel', 'mathop', 'lim', 'sup', 'inf', 'det',
               'mathbin', 'phantom', 'ne', 'iff', 'implies', 'mapsto', 'uparrow', 'downarrow', 'percent', 'euro'}
MULTI_LETTER = ('SoH', 'SoC', 'LB', 'UB', 'CL', 'DSO', 'TSO', 'ESSO', 'NPV', 'EFC', 'DoD')
INDEX_LETTERS = set('cdegijklmnosty') | {'ell'}     # the letters this manuscript uses as indices / counters


def _read_group(s, i):
    depth, j = 0, i
    while j < len(s):
        if s[j] == '\\':
            j += 2
            continue
        if s[j] == '{':
            depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _read_cmd(s, i):
    j = i + 1
    while j < len(s) and s[j].isalpha():
        j += 1
    if j == i + 1:
        return s[i + 1:i + 2], i + 2
    return s[i + 1:j], j


def _skip_ws(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _read_arg(s, i):
    i = _skip_ws(s, i)
    if i >= len(s):
        return '', i
    if s[i] == '{':
        return _read_group(s, i)
    if s[i] == '\\':
        name, j = _read_cmd(s, i)
        jj = _skip_ws(s, j)
        if (name in FONT_WORD or name in FONT_BOLD or name in FONT_CAL or name in FONT_BB or name in DECOR) \
                and jj < len(s) and s[jj] == '{':
            _g, k = _read_group(s, jj)
            return s[i:k], k
        return s[i:j], j
    return s[i], i + 1


def _strip_fonts(x):
    prev = None
    while prev != x:
        prev = x
        x = re.sub(r'\\(?:' + '|'.join(sorted(FONT_WORD | FONT_BOLD)) + r')\s*\{([^{}]*)\}', r'\1', x)
        x = re.sub(r'\\textcolor\{[^{}]*\}', '', x)
    return x


ITER_ITEMS = {'k', 'l', r'\ell', 'k+1', 'k-1', 'l+1', 'l-1', r'\ell+1', r'\ell-1', '0'}


def normalise_sup(raw):
    """-> (label, iteration_only)."""
    if not raw:
        return '', False
    x = _strip_fonts(raw)
    x = re.sub(r'\s+', '', x).replace(r'\,', '').replace(r'\;', '').replace(r'\!', '')
    had_iter = False
    y = re.sub(r'\((?:l|k|\\ell)(?:[+-]1)?\)', '', x)
    had_iter |= y != x
    x = y
    y = re.sub(r'\^\{?(?:l|k|\\ell)(?:[+-]1)?\}?', '', x)
    had_iter |= y != x
    x = y.replace('{', '').replace('}', '')
    items = x.split(',')
    kept = [it for it in items if it not in ITER_ITEMS and not re.fullmatch(r'\d+', it)]
    had_iter |= len(kept) != len(items)
    lab = ','.join(kept).replace(r'\max', 'max').replace(r'\min', 'min')
    return lab, (had_iter and lab == '')


def normalise_sub(raw):
    if not raw:
        return ''
    x = re.sub(r'\s+', '', _strip_fonts(raw)).replace('{', '').replace('}', '')
    if re.fullmatch(r'[A-Z][A-Za-z]*', x) or re.fullmatch(r'\d+', x):
        return x
    return ''


def _expression_like(lab):
    return any(c in lab for c in '+-_(') or r'\times' in lab or r'\cdot' in lab


class Atom:
    __slots__ = ('base', 'sup_raw', 'sub_raw', 'bold', 'pos', 'kind', 'in_script')

    def __init__(self, base, pos, kind='symbol', bold=False, in_script=False):
        self.base, self.pos, self.kind, self.bold, self.in_script = base, pos, kind, bold, in_script
        self.sup_raw, self.sub_raw = '', ''

    def key(self):
        sup, _it = normalise_sup(self.sup_raw)
        sub = normalise_sub(self.sub_raw)
        return self.base + (f'^{{{sup}}}' if sup else '') + (f'_{{{sub}}}' if sub else '')

    def iteration_only(self):
        return normalise_sup(self.sup_raw)[1]

    def raw(self):
        b = ('bold ' if self.bold else '') + self.base
        return b + (f'^{{{self.sup_raw}}}' if self.sup_raw else '') + (f'_{{{self.sub_raw}}}' if self.sub_raw else '')

    def category(self):
        if self.kind != 'symbol':
            return self.kind
        if self.base in INDEX_LETTERS and not self.bold and not normalise_sup(self.sup_raw)[0] and not self.sub_raw:
            return 'index'
        return 'symbol'


def tokenize(s, offset=0, in_script=False, unknown=None):
    """Atoms of the math string `s` (positions are offsets into the comment-stripped document)."""
    atoms = []
    unknown = unknown if unknown is not None else set()
    i, n = 0, len(s)

    def scripts(atom, i):
        while True:
            j = _skip_ws(s, i)
            if j < n and s[j] in '^_':
                arg, k = _read_arg(s, j + 1)
                if s[j] == '^':
                    atom.sup_raw = (atom.sup_raw + ',' + arg) if atom.sup_raw else arg
                    lab, _it = normalise_sup(arg)
                    if _expression_like(lab):
                        atoms.extend(tokenize(arg, offset + j + 1, True, unknown))
                else:
                    atom.sub_raw = (atom.sub_raw + ',' + arg) if atom.sub_raw else arg
                    if not normalise_sub(arg):      # an index list is scanned; a label subscript (Omega_C) is not
                        atoms.extend(tokenize(arg, offset + j + 1, True, unknown))
                i = k
            elif j < n and s[j] == "'":
                atom.sup_raw += "'"
                i = j + 1
            else:
                return i

    while i < n:
        c = s[i]
        if c.isspace() or c in '{}':
            i += 1
            continue
        if c == '\\':
            name, j = _read_cmd(s, i)
            if name in SKIP_WITH_GROUP:
                jj = _skip_ws(s, j)
                if jj < n and s[jj] == '{':
                    _g, j = _read_group(s, jj)
                i = j
                continue
            if name == 'textcolor':
                jj = _skip_ws(s, j)
                if jj < n and s[jj] == '{':
                    _g, j = _read_group(s, jj)
                i = j
                continue
            if name in GREEK:
                a = Atom(name, offset + i, in_script=in_script)
                atoms.append(a)
                i = scripts(a, j)
                continue
            if name in DECOR or name in FONT_BOLD or name in FONT_CAL or name in FONT_BB or name in FONT_WORD:
                arg, k = _read_arg(s, j)
                if name in FONT_BB:
                    a = Atom(f'bb({arg.strip()})', offset + i, kind='number_set', in_script=in_script)
                    atoms.append(a)
                    i = scripts(a, k)
                    continue
                if name in FONT_CAL:
                    a = Atom(f'cal({arg.strip()})', offset + i, in_script=in_script)
                    atoms.append(a)
                    i = scripts(a, k)
                    continue
                if name in FONT_WORD:
                    word = re.sub(r'\s+', ' ', _strip_fonts(arg)).strip()
                    if re.fullmatch(r'[A-Za-z]+', word):
                        if len(word) == 1:
                            a = Atom(word, offset + i, in_script=in_script)
                        else:
                            a = Atom(word, offset + i, kind='symbol', in_script=in_script)
                        atoms.append(a)
                        i = scripts(a, k)
                    else:
                        i = k     # text such as "s.t.", "p.u." -- not a symbol
                    continue
                inner = tokenize(arg, offset + j, in_script, unknown)
                syms = [t for t in inner if not t.in_script]
                if not syms:
                    i = k
                    continue
                head = syms[0]
                if name in DECOR:
                    a = Atom(f'{DECOR[name]}({head.base})', offset + i, in_script=in_script)
                    a.sup_raw, a.sub_raw = head.sup_raw, head.sub_raw
                    atoms.extend(t for t in inner if t is not head)
                else:                                   # bold
                    a = head
                    a.bold = True
                    atoms.extend(t for t in inner if t is not head)
                atoms.append(a)
                i = scripts(a, k)
                continue
            if name in IGNORE_CMDS or not name.isalpha():
                i = j
                continue
            unknown.add(name)
            i = j
            continue
        if c.isalpha():
            j = i
            while j < n and s[j].isalpha():
                j += 1
            run = s[i:j]
            pieces = []
            p = 0
            while p < len(run):
                hit = next((w for w in MULTI_LETTER if run.startswith(w, p)), None)
                if hit:
                    pieces.append((hit, i + p))
                    p += len(hit)
                else:
                    pieces.append((run[p], i + p))
                    p += 1
            for q, (piece, pos) in enumerate(pieces):
                a = Atom(piece, offset + pos, in_script=in_script)
                atoms.append(a)
                if q == len(pieces) - 1:
                    i = scripts(a, j)
            continue
        if c in '^_':
            arg, k = _read_arg(s, i + 1)
            atoms.extend(tokenize(arg, offset + i + 1, True, unknown))
            i = k
            continue
        i += 1
    return atoms


# ======================================================================================================================
#  TASK A
# ======================================================================================================================
def nomenclature_bounds(lines):
    start = next(i for i, ln in enumerate(lines, 1) if ln.strip().startswith(r'\section*{Nomenclature}'))
    end = next(i for i, ln in enumerate(lines, 1) if i > start and re.match(r'\s*\\section\{', ln)) - 1
    return start, end


def parse_nomenclature(clines, start, end):
    entries, group, last = [], None, None
    for ln_no in range(start, end + 1):
        ln = clines[ln_no - 1].strip()
        if not ln or ln.startswith('\\begin') or ln.startswith('\\end') or ln.startswith('\\toprule') \
                or ln.startswith('\\midrule') or ln.startswith('\\bottomrule') or ln.startswith('\\centering'):
            continue
        m = re.search(r'\\textbf\{(Sets|Parameters)\}', ln)
        if m:
            group = m.group(1)
            continue
        if ln.startswith(r'\subsection*{'):
            group = re.sub(r'\\subsection\*\{(.*)\}', r'\1', ln)
            continue
        if '&' not in ln or r'\textbf{Symbol}' in ln or r'\textbf{Variable}' in ln:
            continue
        cell0, cell1 = ln.split('&', 1)
        desc = re.sub(r'\\\\\s*$', '', cell1).strip()
        desc = re.sub(r'\\textcolor\{[^}]*\}\{(.*?)\}', r'\1', desc).strip()
        c0 = cell0.strip()
        if c0 in ('', '~'):
            if last is not None:
                last['description'] = (last['description'] + ' ' + desc).strip()
                last['lines'].append(ln_no)
            continue
        maths = re.findall(r'(?<!\\)\$(.+?)(?<!\\)\$', c0)
        atoms = []
        for mth in maths:
            atoms.extend(t for t in tokenize(mth) if not t.in_script)
        if not atoms:
            continue
        head = atoms[0]
        last = {'line': ln_no, 'lines': [ln_no], 'group': group, 'tex': c0, 'raw': head.raw(), 'key': head.key(),
                'iteration_only_sup': head.iteration_only(), 'blue': 'textcolor{blue}' in c0, 'description': desc}
        entries.append(last)
    return entries


def body_atoms(ctext, clines, nom_start, nom_end):
    line_starts, pos = [], 0
    for ln in clines:
        line_starts.append(pos)
        pos += len(ln) + 1

    def line_of(p):
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= p:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    unknown = set()
    out = []
    for a, b, content in math_regions(ctext):
        ln0 = line_of(a)
        if nom_start <= ln0 <= nom_end:
            continue
        cstart = ctext.index(content, a) if content else a
        for at in tokenize(content, cstart, False, unknown):
            out.append((line_of(at.pos), at))
    return out, sorted(unknown)


# Benders / removed-method symbols (map section B: L^B, alpha^Down, cuts, cut multipliers, underestimator alpha^(l))
BENDERS_KEYS = {
    'L^{B}': 'set of Benders iterations (nomenclature l. 202)',
    'alpha^{Down}': 'lower bound on the Benders underestimator',
    'alpha': 'Benders underestimator alpha^(l) / alpha in the LP master of Algorithm 1',
    'mu^{S}': 'optimality / feasibility cut multiplier (power), mu^{S(k)}',
    'mu^{E}': 'optimality / feasibility cut multiplier (energy), mu^{E(k)}',
    'g^{S}': 'cut sensitivity g^{S,ell} in Algorithm 1',
    'g^{E}': 'cut sensitivity g^{E,ell} in Algorithm 1',
    'LB': 'Benders lower bound LB^ell',
    'UB': 'Benders upper bound UB^ell',
    'gap': 'Benders relative gap gap^ell',
    'hat(Q)': 'recourse value at the Benders iterate, Q-hat^ell',
    'hat(x)': 'Benders iterate x-hat^ell',
    'hat(S)^{Rated}': 'cut expansion point S-hat^{Rated,ell}',
    'hat(E)^{Rated}': 'cut expansion point E-hat^{Rated,ell}',
    'ell': 'Benders iteration counter ell (Algorithm 1)',
    'ell^{max}': 'Benders iteration cap ell^max',
    'epsilon^{rel}': 'Benders gap tolerance (Algorithm 1)',
    'epsilon^{abs}': 'Benders gap tolerance (Algorithm 1)',
}
BENDERS_RAW_PATTERNS = {
    'iteration superscript (l) on investment / rated variables (planning iteration of the Benders master)':
        r'\((?:l|k)\)',
    'y^{(k)} -- subproblem objective value at Benders iteration k (cut constant)': r'y\^\{\(k\)\}',
}
BENDERS_WORDS = r'Benders|[Cc]uts?\b|underestimator|optimality gap|lower bound|upper bound'

# Required symbols (map section B: lattice unit sizes, duration bounds, tau, k0, k*, settling-rule symbols)
REQUIRED = [
    {'id': 'lattice_unit_power', 'what': 'lattice unit size 0.25 MVA',
     'p5_patterns': [r'0\.25', r'lattice', r'granular'], 'main_patterns': [r'0\.25\s*~?\s*MVA', r'lattice'],
     'main_keys': [], 'dfo_notation': r'P_{n,y} \in 0.25\,\mathbb{Z}_{\ge 0} MVA (STEP4_DFO_METHOD.md 1.1)'},
    {'id': 'lattice_unit_energy', 'what': 'lattice unit size 0.5 MWh',
     'p5_patterns': [r'0\.5\s*MWh', r'lattice', r'granular'], 'main_patterns': [r'0\.50?\s*~?\s*MWh', r'lattice'],
     'main_keys': [], 'dfo_notation': r'E_{n,y} \in 0.5\,\mathbb{Z}_{\ge 0} MWh (STEP4_DFO_METHOD.md 1.1)'},
    {'id': 'duration_bounds', 'what': 'duration (E/P) bounds 2-4 h',
     'p5_patterns': [r'E/P', r'duration', r'\b2\s*h\b', r'\b4\s*h\b', r'2.{1,3}4\s*h'],
     'main_patterns': [r'2~h', r'10~h', r'energy-to-power'],
     'main_keys': ['phi^{Min}', 'phi^{Max}'], 'dfo_notation': r'2 h <= E/P <= 4 h (STEP4_DFO_METHOD.md header)'},
    {'id': 'tau', 'what': 'certification threshold tau', 'p5_patterns': [r'τ'], 'main_patterns': [],
     'main_keys': ['tau'], 'dfo_notation': None},
    {'id': 'k0', 'what': 'first residual pass k0', 'p5_patterns': [r'k₀'], 'main_patterns': [],
     'main_keys': ['k'], 'dfo_notation': None},
    {'id': 'kstar', 'what': 'certification cycle k*', 'p5_patterns': [r'k\\\*', r'k\*'], 'main_patterns': [],
     'main_keys': ['k'], 'dfo_notation': None},
    {'id': 'window_W', 'what': 'settling window length W = max(20, ceil(1.1 P-hat))',
     'p5_patterns': [r'\bW\b'], 'main_patterns': [], 'main_keys': ['W'], 'dfo_notation': None},
    {'id': 'period_Phat', 'what': 'measured period P-hat', 'p5_patterns': [r'P̂'], 'main_patterns': [],
     'main_keys': ['hat(P)', 'hat(P)^{E}', 'hat(P)^{I}'], 'dfo_notation': None},
    {'id': 'period_Pmax', 'what': 'longest measured period P_max', 'p5_patterns': [r'P_max'], 'main_patterns': [],
     'main_keys': ['P^{max}', 'P^{G,max}'], 'dfo_notation': None},
    {'id': 't_sum', 'what': 'priced interface-consensus gap t_sum', 'p5_patterns': [r't_sum'], 'main_patterns': [],
     'main_keys': ['t'], 'dfo_notation': None},
    {'id': 'deltaR', 'what': 'ratio resolution delta-R', 'p5_patterns': [r'δR'], 'main_patterns': [],
     'main_keys': ['delta'], 'dfo_notation': None},
    {'id': 'V_unit_value', 'what': 'reference unit value V (in tau = deltaR V / 4)', 'p5_patterns': [r'\bV\b'],
     'main_patterns': [], 'main_keys': ['V', 'V^{I}', 'hat(V)^{I}', 'V^{Base}', 'V^{max}', 'V^{min}', 'V^{S}'],
     'dfo_notation': None},
    {'id': 'eps_abs_rel', 'what': 'Boyd residual tolerances eps_abs, eps_rel', 'p5_patterns': [r'ε_abs', r'ε_rel'],
     'main_patterns': [], 'main_keys': ['epsilon^{abs}', 'epsilon^{rel}'], 'dfo_notation': None},
    {'id': 'rho', 'what': 'ADMM penalty rho (frozen from k0)', 'p5_patterns': [r'ρ'], 'main_patterns': [],
     'main_keys': ['rho^{E,P}', 'rho^{E,Q}'], 'dfo_notation': None},
    {'id': 'lambda', 'what': 'interface price lambda (consensus dual of the interface power), lambda_t',
     'p5_patterns': [r'λ'], 'main_patterns': [], 'main_keys': ['lambda', 'pi^{I,P}'], 'dfo_notation': None},
    {'id': 'pi_t', 'what': 'wholesale price pi_t', 'p5_patterns': [r'π_t'], 'main_patterns': [],
     'main_keys': ['pi'], 'dfo_notation': None},
    {'id': 'c_flex', 'what': 'flexibility price c_flex', 'p5_patterns': [r'c_flex'], 'main_patterns': [],
     'main_keys': [], 'dfo_notation': None},
    {'id': 'L_window', 'what': 'earlier window L = 44', 'p5_patterns': [r'\bL = 44'], 'main_patterns': [],
     'main_keys': ['L^{B}', 'cal(L)^{E,P}', 'cal(L)^{E,Q}'], 'dfo_notation': None},
    {'id': 'eps_AE', 'what': 'elasticity of value to available energy eps_AE', 'p5_patterns': [r'ε_AE'],
     'main_patterns': [], 'main_keys': [], 'dfo_notation': None},
]

# The Worker's reading of overloaded / conflicting uses. Every use declared here is VERIFIED against the text: its
# raw-atom regex (optionally restricted to a line range) must match at least one body atom or nomenclature entry, else
# the run FAILS. ALG1 = Algorithm 1 (\begin{algorithm} ... \end{algorithm}, lines located at run time).
ALG1 = 'ALG1'
DECLARED_OVERLOADS = [
    {'key': 'D', 'uses': [('set of representative days (nomenclature)', r'^D$', None),
                          ('number of calendar days represented by d, D_d (nomenclature)', r'^D_\{d\}$', None)],
     'note': 'D and D_d share the base letter; D_d is a count, D a set (both defined in the nomenclature)'},
    {'key': 'Y', 'uses': [('set of representative years (nomenclature)', r'^Y$', None),
                          ('number of calendar years represented by y, Y_y (nomenclature)', r'^Y_\{y\}$', None)],
     'note': 'same pattern as D / D_d'},
    {'key': 'y', 'uses': [('representative year (index)', r'^y$', None),
                          ('subproblem objective value at iteration k, y^{(k)} (Benders cuts)', r'^y\^\{\(k\)\}$', None)],
     'note': 'y^{(k)} reuses the year letter for a recourse value; removed with the Benders cuts'},
    {'key': 'r', 'uses': [('discount rate (nomenclature; (1+r)^(y-y0))', r'^r$', 'NOT_ALG1'),
                          ('ADMM loop counter "r = 1 to r^max" (Algorithm 1)', r'^r(\^\{\\max\})?$', ALG1),
                          ('penalty update rate r^{E,P}, r^{E,Q} (Appendix A)', r'^r\^\{E,[PQ]\}$', None),
                          ('branch resistance r_ij (Appendix D branch table)', r'^r_\{ij\}$', None)],
     'note': 'r carries four meanings; only the discount rate is in the nomenclature'},
    {'key': 'x', 'uses': [('first-stage investment vector x (bold in eq. 1, plain in Algorithm 1)', r'^(bold )?x$', None),
                          ('branch reactance x_ij (Appendix D branch table)', r'^x_\{ij\}$', None)],
     'note': 'x is undefined in the nomenclature in both meanings'},
    {'key': 'g', 'uses': [('generator index g (Appendix D generator table)', r'^g$', None),
                          ('bus conductance g_i (Appendix D bus table)', r'^g_\{i\}$', None),
                          ('cut sensitivities g^{S,ell}, g^{E,ell} (Algorithm 1)', r'^g\^\{[SE],', ALG1)],
     'note': 'three meanings of g; none in the nomenclature'},
    {'key': 'pi', 'uses': [('scenario probabilities pi_m pi_o in Algorithm 1', r'^pi_\{[mo]\}$', ALG1),
                           ('ADMM dual variables pi^{I,V}, pi^{E,P}, ... (Appendix A)', r'^pi\^\{.*\}', None)],
     'note': 'the nomenclature uses omega for probabilities; paragraphs_v5 uses pi_t for the wholesale price'},
    {'key': 'E', 'uses': [('set E with E^S subset of E (Appendix A list; salvage sum e in E)', r'^E$', None),
                          ('set of shared ESSs E^S (nomenclature)', r'^E\^\{S\}$', None),
                          ('energy variables E^{Inv}, E^{Rated}, ... (nomenclature)', r'^E\^\{\\(?:text|mathrm)\{(?:Inv|Rated)', None)],
     'note': 'E is both a set name and the energy-capacity letter'},
    {'key': 'C^{Inv} / c^{Inv}', 'uses': [('total investment budget c^{Inv} (nomenclature)', r'^c\^\{\\text\{Inv\}\}$', None),
                                          ('investment cost C^{Inv}(x), C_c^{Inv} (eq. 1, Algorithm 1, salvage)', r'^C(?:_\{c\})?\^\{\\(?:mathrm|text)\{Inv\}\}', None)],
     'note': 'the budget (lower-case c) and the investment-cost function (upper-case C) differ only by case'},
    {'key': 'epsilon^{rel} / epsilon^{abs}', 'uses': [('Benders gap tolerances in Algorithm 1', r'^epsilon\^\{\\mathrm\{(?:rel|abs)\}\}$', ALG1)],
     'note': 'paragraphs_v5 uses eps_abs = 1e-5 / eps_rel = 1e-4 for the ADMM residual test (different quantity)'},
    {'key': 'hat(P)', 'uses': [('local copy of the SO request, P-hat^{E^k}, P-hat^{I^k} (Appendix A)', r'^hat\(P\)', None)],
     'note': 'paragraphs_v5 uses P-hat for the measured objective period'},
    {'key': 'S^{Rated}', 'uses': [('rated power of the shared ESS (nomenclature, eqs.)', r'^S\^\{\\(?:text|mathrm)\{Rated[^}]*\}[^_]*(_\{(?!ij\}).*)?$', None),
                                  ('branch rating S^Rated_ij (Appendix D branch table)', r'^S\^\{\\text\{Rated\}\}_\{ij\}$', None)],
     'note': 'the branch-table header reuses S^Rated with branch indices'},
]

# Conflicts between a nomenclature description and the body (Worker reading), each pinned to main.tex lines that must
# contain the quoted substrings (verified; a miss FAILS the run).
DECLARED_CONFLICTS = [
    {'id': 'S_Rated_meaning',
     'finding': 'nomenclature defines S^{Rated}_{e,y^Inv,y} as the rated power of the unit installed in y^Inv (per unit); '
                'the body uses S^{Rated}_{e,y} for the TOTAL rated power and S^{Rated,Unit}_{e,y^Inv,y} (not in the '
                'nomenclature) for the per-unit quantity',
     'evidence': [(238, r'$S^{\text{Rated}}_{e,y^\text{Inv},y}$   & Rated power of unit $e$ installed'),
                  (769, r'S^{\text{Rated}}_{e,y} = \sum'), (769, r'S^{\text{Rated,Unit}}_{e,y^\text{Inv},y}')]},
    {'id': 'E_Rated_meaning',
     'finding': 'nomenclature defines E^{Rated}_{e,y^Inv,y} as the AVAILABLE energy capacity of the unit (the same '
                'description as E^{Av,Unit}, l. 244); the body uses E^{Rated}_{e,y} for the TOTAL rated energy and '
                'E^{Av,Unit} for the available (SoH-scaled) energy',
     'evidence': [(240, r'$E^{\text{Rated}}_{e,y^\text{Inv},y}$ & Available energy capacity of unit'),
                  (244, r'$E^{\text{Av,Unit}}_{e,y^\text{Inv},y}$ & Available energy capacity of unit'),
                  (773, r'E^{\text{Rated}}_{e,y} = \sum')]},
    {'id': 'E_Inv_scenario_index',
     'finding': 'the energy-to-power constraint writes E^{Inv(l)}_{e,y,c} with an investment-cost-scenario index c, '
                'while the plan is stated to be scenario-independent (S^{Inv(l)}_{e,y} on the same line)',
     'evidence': [(621, r'E^{\text{Inv}(l)}_{e,y,c}')]},
    {'id': 'phi_year_index',
     'finding': 'phi^{Min}_e / phi^{Max}_e carry only the unit index but are described "of unit e in year y"',
     'evidence': [(213, r'$\phi^{\text{Min}}_e$ & Minimum admissible energy-to-power ratio of unit $e$ in year $y$'),
                  (214, r'$\phi^{\text{Max}}_e$ & Maximum admissible energy-to-power ratio of unit $e$ in year $y$')]},
    {'id': 'SoH_min_index',
     'finding': 'nomenclature SoH^{min}_{e,y} "in year y"; the body uses SoH^{min}_{e,y^Inv} "for unit e installed in '
                'year y^Inv"',
     'evidence': [(217, r'$SoH_{e,y}^{\text{min}}$ & Minimum admissible SoH of unit $e$ in year $y$'),
                  (842, r'$SoH_{e,y^\text{Inv}}^{\text{min}}$ denotes the minimum admissible SoH for unit $e$ installed in year $y^\text{Inv}$')]},
]


def algorithm_ranges(clines):
    out, start = [], None
    for i, ln in enumerate(clines, 1):
        if re.search(r'\\begin\{algorithm\}', ln):
            start = i
        elif re.search(r'\\end\{algorithm\}', ln) and start:
            out.append((start, i))
            start = None
    return out


def task_a(lines, clines, ctext, p5_lines):
    smap = section_map(lines)
    nom_start, nom_end = nomenclature_bounds(lines)
    entries = parse_nomenclature(clines, nom_start, nom_end)
    batoms, unknown_cmds = body_atoms(ctext, clines, nom_start, nom_end)

    uses = {}
    for ln, at in batoms:
        cat = at.category()
        if cat == 'number_set':
            continue
        k = at.key()
        u = uses.setdefault(k, {'key': k, 'category': cat, 'lines': [], 'raw_forms': set(), 'iteration_only': False,
                                'bold_seen': False})
        u['lines'].append(ln)
        u['raw_forms'].add(at.raw())
        u['iteration_only'] |= at.iteration_only()
        u['bold_seen'] |= at.bold
        if cat == 'symbol':
            u['category'] = 'symbol'
    for u in uses.values():
        u['first_line'] = min(u['lines'])
        u['n_occurrences'] = len(u['lines'])
        u['lines'] = sorted(set(u['lines']))
        u['raw_forms'] = sorted(u['raw_forms'])
        u['first_section'] = smap.get(u['first_line'])

    nom_keys = {}
    for e in entries:
        nom_keys.setdefault(e['key'], []).append(e)

    # (i) defined, never used
    unused = []
    for e in entries:
        u = uses.get(e['key'])
        if u is None:
            unused.append({'key': e['key'], 'tex': e['tex'], 'nomenclature_line': e['line'],
                           'description': e['description']})
    # (ii) used, not defined
    def _base_key(k):
        return re.sub(r'_\{[^}]*\}$', '', k)

    nom_bases = {}
    for k in nom_keys:
        nom_bases.setdefault(re.sub(r'^(?:hat|bar|tilde)\((.*?)\)', r'\1', _base_key(k)), []).append(k)
    undefined = []
    for k, u in sorted(uses.items(), key=lambda kv: kv[1]['first_line']):
        if k in nom_keys:
            continue
        cat = u['category']
        if cat == 'symbol':
            m = re.match(r'^(hat|bar|tilde)\((.*?)\)(.*)$', k)
            if m and (m.group(2) + m.group(3)) in nom_keys:
                cat = 'decorated variant of a defined symbol'
            elif re.sub(r'_\{[^}]*\}$', '', k) in nom_keys:
                cat = 'label-subscript variant of a defined symbol'
        undefined.append({'key': k, 'category': cat, 'first_line': u['first_line'], 'first_section': u['first_section'],
                          'n_occurrences': u['n_occurrences'], 'lines': u['lines'], 'raw_forms': u['raw_forms'],
                          'iteration_only_seen': u['iteration_only']})
    # (iii) defined twice / overloaded
    dup = [{'key': k, 'entries': [{'line': e['line'], 'tex': e['tex'], 'description': e['description']} for e in es]}
           for k, es in nom_keys.items() if len(es) > 1]
    sub_overloads = []
    raw_by_key = {}
    for ln, at in batoms:
        raw_by_key.setdefault(at.key(), {}).setdefault(at.raw(), []).append(ln)
    for e in entries:
        pass
    # keys whose raw forms carry different index subscript letters on a SINGLE-letter subscript (D vs D_d etc.)
    for k, forms in raw_by_key.items():
        subs = set()
        for rf in forms:
            m = re.search(r'_\{([^{}]*)\}$', rf)
            subs.add(m.group(1) if m else '')
        if '' in subs and len(subs) > 1 and k in nom_keys:
            sub_overloads.append({'key': k, 'subscript_forms': sorted(subs),
                                  'forms': {rf: sorted(set(v))[:12] for rf, v in sorted(forms.items())}})
    alg1 = next((s_, e_) for s_, e_ in algorithm_ranges(clines) if 'alg:shared_ess_planning_admm' in
                '\n'.join(clines[s_ - 1:e_]))

    def in_scope(ln, scope):
        if scope is None:
            return True
        inside = alg1[0] <= ln <= alg1[1]
        return inside if scope == ALG1 else not inside

    declared = []
    for d in DECLARED_OVERLOADS:
        rec = {'key': d['key'], 'note': d['note'], 'uses': []}
        for meaning, rx, scope in d['uses']:
            hits = sorted({ln for ln, at in batoms if re.search(rx, at.raw()) and in_scope(ln, scope)})
            nom_hits = [e['line'] for e in entries if re.search(rx, e['raw'])] if scope in (None, 'NOT_ALG1') else []
            rec['uses'].append({'meaning': meaning, 'raw_regex': rx, 'body_lines': hits,
                                'nomenclature_lines': nom_hits})
            if not hits and not nom_hits:
                _fail(f'declared overload {d["key"]!r} use {meaning!r}: regex {rx!r} matches no atom')
        declared.append(rec)
    conflicts = []
    for c in DECLARED_CONFLICTS:
        ev = []
        for ln, sub in c['evidence']:
            ok = sub in lines[ln - 1]
            ev.append({'line': ln, 'must_contain': sub, 'found': ok, 'line_text': lines[ln - 1].strip()[:200]})
            if not ok:
                _fail(f"declared conflict {c['id']}: line {ln} does not contain {sub!r}")
        conflicts.append({'id': c['id'], 'finding': c['finding'], 'evidence': ev})
    # different nomenclature keys carrying the SAME description
    by_desc = {}
    for e in entries:
        by_desc.setdefault(re.sub(r'\s+', ' ', e['description']).strip(), []).append(e)
    same_desc = [{'description': dsc, 'entries': [{'line': e['line'], 'key': e['key'], 'tex': e['tex']} for e in es]}
                 for dsc, es in by_desc.items() if len({e['key'] for e in es}) > 1]
    # symbols occurring ONLY inside Algorithm 1 (replaced wholesale per the map)
    alg1_only = sorted(k for k, u in uses.items()
                       if u['category'] == 'symbol' and all(alg1[0] <= ln <= alg1[1] for ln in u['lines']))
    # local definitions in the body ("where $X$ denotes ...", itemised "$X$: ...") for the nomenclature keys
    body_defs = []
    for ln_no, ln in enumerate(clines, 1):
        if nom_start <= ln_no <= nom_end:
            continue
        for m in re.finditer(r'\$([^$]+)\$\s*(?::|denotes?|is|are|contains|represents)\s+([^$.;]{3,90})', ln):
            ats = [t for t in tokenize(m.group(1)) if not t.in_script]
            if not ats:
                continue
            k = ats[0].key()
            if k in nom_keys:
                body_defs.append({'key': k, 'line': ln_no, 'body_text': (m.group(1) + ' ' + m.group(2)).strip(),
                                  'nomenclature': [e['description'] for e in nom_keys[k]]})
    # (iv) Benders symbols
    benders = []
    for k, why in BENDERS_KEYS.items():
        u = uses.get(k)
        nom = [e['line'] for e in nom_keys.get(k, [])]
        benders.append({'key': k, 'meaning': why, 'nomenclature_lines': nom,
                        'body_lines': u['lines'] if u else [], 'raw_forms': u['raw_forms'] if u else [],
                        'n_occurrences': u['n_occurrences'] if u else 0})
    raw_pattern_hits = {}
    for label, rx in BENDERS_RAW_PATTERNS.items():
        hits = {}
        for ln, at in batoms:
            if re.search(rx, at.sup_raw if label.startswith('iteration') else at.raw()):
                hits.setdefault(at.key(), set()).add(ln)
        nom = [e['line'] for e in entries if re.search(rx, e['tex'])]
        raw_pattern_hits[label] = {'regex': rx, 'by_key': {k: sorted(v) for k, v in sorted(hits.items())},
                                   'nomenclature_lines': nom}
    word_lines = [{'line': i, 'section': smap.get(i), 'text': ln.strip()[:200]}
                  for i, ln in enumerate(clines, 1) if re.search(BENDERS_WORDS, ln)]
    # (v) required symbols
    p5_scope_from = next(i for i, ln in enumerate(p5_lines, 1) if ln.startswith('## (ii)'))
    required = []
    main_text_lines = [(i, ln) for i, ln in enumerate(clines, 1)]
    for r in REQUIRED:
        p5_hits = []
        for i, ln in enumerate(p5_lines, 1):
            for rx in r['p5_patterns']:
                for m in re.finditer(rx, ln):
                    lo, hi = max(0, m.start() - 40), min(len(ln), m.end() + 40)
                    p5_hits.append({'line': i, 'pattern': rx, 'in_manuscript_scope': i >= p5_scope_from,
                                    'context': ln[lo:hi].strip()})
        main_hits = []
        for i, ln in main_text_lines:
            for rx in r['main_patterns']:
                if re.search(rx, ln):
                    main_hits.append({'line': i, 'pattern': rx, 'text': ln.strip()[:160]})
        key_hits = {k: (uses[k]['lines'] if k in uses else []) for k in r['main_keys']}
        key_nom = {k: [e['line'] for e in nom_keys.get(k, [])] for k in r['main_keys']}
        required.append({'id': r['id'], 'what': r['what'], 'paragraphs_v5_used': bool(p5_hits),
                         'paragraphs_v5_hits': p5_hits, 'main_tex_text_hits': main_hits,
                         'main_tex_related_symbol_lines': key_hits, 'main_tex_related_symbol_in_nomenclature': key_nom,
                         'step4_dfo_notation': r['dfo_notation']})
    counts = {'nomenclature_entries': len(entries), 'distinct_nomenclature_keys': len(nom_keys),
              'body_atoms': len(batoms), 'distinct_body_keys': len(uses),
              'distinct_body_symbol_keys': sum(1 for u in uses.values() if u['category'] == 'symbol'),
              'i_defined_not_used': len(unused),
              'ii_used_not_defined_all': len(undefined),
              'ii_used_not_defined_by_category': {},
              'iii_duplicate_nomenclature_keys': len(dup),
              'iii_index_subscript_overloads_of_defined_keys': len(sub_overloads),
              'iii_declared_overloads': len(declared),
              'iii_declared_conflicts': len(conflicts),
              'iii_same_description_different_keys': len(same_desc),
              'symbols_only_inside_algorithm1': len(alg1_only),
              'iv_benders_keys_with_occurrences': sum(1 for b in benders if b['n_occurrences'] or b['nomenclature_lines']),
              'v_required_items': len(required),
              'v_required_used_in_paragraphs_v5': sum(1 for r in required if r['paragraphs_v5_used'])}
    for u in undefined:
        counts['ii_used_not_defined_by_category'][u['category']] = \
            counts['ii_used_not_defined_by_category'].get(u['category'], 0) + 1
    return {'normalisation': NORMALISATION_TEXT, 'nomenclature_lines': [nom_start, nom_end],
            'nomenclature_entries': entries, 'counts': counts,
            'i_defined_not_used': unused, 'ii_used_not_defined': undefined,
            'iii_duplicate_nomenclature_keys': dup, 'iii_index_subscript_overloads': sub_overloads,
            'iii_declared_overloads_verified': declared, 'iii_declared_conflicts_verified': conflicts,
            'iii_same_description_different_keys': same_desc,
            'iii_body_local_definitions_of_nomenclature_keys': body_defs,
            'algorithm1_lines': list(alg1), 'symbols_only_inside_algorithm1': alg1_only,
            'iv_benders_symbols': benders, 'iv_benders_raw_patterns': raw_pattern_hits,
            'iv_benders_word_lines': word_lines, 'v_required_symbols': required,
            'unknown_latex_commands_in_math': unknown_cmds,
            'body_symbol_index': {k: {kk: (sorted(vv) if isinstance(vv, set) else vv) for kk, vv in u.items()}
                                  for k, u in sorted(uses.items())}}


NORMALISATION_TEXT = (
    'identity key = base + "^{label}" + "_{label}". Font wrappers (\\text, \\mathrm, \\textrm, \\mathit, '
    '\\operatorname) removed; bold (\\boldsymbol, \\mathbf, \\bm) removed; \\textcolor transparent; \\mathcal{X} -> '
    'cal(X); \\mathbb -> excluded; decorations kept in the base (hat(S); widehat=hat, overline=bar, widetilde=tilde). '
    'Superscript: whitespace and braces removed, iteration markers removed ("(l)", "(k)", "(\\ell)" and +-1, nested '
    '"^k"/"^{k+1}"/"^{\\ell}", comma items k, l, \\ell, k+-1, l+-1, \\ell+-1 and digit strings such as 0 or 1 '
    '(LB^0, \\hat{x}^1: iteration counters)); \\max -> max. Subscript: dropped '
    '(index) unless it starts with an upper-case letter (\\Omega_C) or is all digits (y_0). Letter runs split into '
    'single letters except SoH, SoC, LB, UB, CL, DSO, TSO, ESSO, NPV, EFC, DoD. An atom is an INDEX (reported '
    'separately, not as an undefined symbol) when its base is one of the letters this manuscript uses as indices or '
    'counters (c d e g i j k l m n o s t y, \\ell), it is not bold, has no superscript label and no subscript at all; '
    'so x, u, f, h, r are symbols and g_i, x_{ij} are symbols. A label subscript (\\Omega_C) is not scanned for '
    'atoms; an index subscript is. '
    'Scripts are also scanned: every subscript, and a superscript that is an expression (contains + - _ ( \\times '
    '\\cdot after iteration markers are removed), contribute their own atoms (e.g. y^{Inv}, y_0 inside indices).')


def md_audit(a, main_sha):
    c = a['counts']
    L = ['# W167 Task A -- nomenclature audit of the submitted main.tex', '',
         f'main.tex sha256 `{main_sha}` (clone `{CLONE_REL}`); nomenclature lines {a["nomenclature_lines"][0]}-'
         f'{a["nomenclature_lines"][1]}. Report only; nothing edited. Line numbers are main.tex lines.', '',
         '## Normalisation', '', a['normalisation'], '',
         '## Counts', '']
    for k, v in c.items():
        L.append(f'- {k}: {v}')
    L += ['', '## (i) Defined in the nomenclature, never used in the text', '',
          '| key | nomenclature line | tex | description |', '|---|---|---|---|']
    for u in a['i_defined_not_used']:
        L.append(f"| `{u['key']}` | {u['nomenclature_line']} | `{u['tex']}` | {u['description']} |")
    L += ['', '## (ii) Used in the text, not in the nomenclature', '',
          'Category `symbol` is the substantive list; `index` are bound indices; the variant categories are forms of '
          'a defined symbol. Every line is in nomenclature_audit.json.', '']
    for cat in ('symbol', 'decorated variant of a defined symbol', 'label-subscript variant of a defined symbol',
                'index'):
        rows = [u for u in a['ii_used_not_defined'] if u['category'] == cat]
        L += [f'### {cat} ({len(rows)})', '', '| key | first line | section of first use | n | lines (first 15) | raw forms (first 4) |',
              '|---|---|---|---|---|---|']
        for u in rows:
            L.append(f"| `{u['key']}` | {u['first_line']} | {u['first_section']} | {u['n_occurrences']} | "
                     f"{', '.join(map(str, u['lines'][:15]))}{' ...' if len(u['lines']) > 15 else ''} | "
                     f"{'; '.join('`' + r + '`' for r in u['raw_forms'][:4])} |")
        L.append('')
    L += ['## (iii) Defined twice, overloaded or conflicting', '', '### Duplicate nomenclature keys', '']
    if not a['iii_duplicate_nomenclature_keys']:
        L.append('none')
    for d in a['iii_duplicate_nomenclature_keys']:
        L.append(f"- `{d['key']}`: " + '; '.join(f"l. {e['line']} `{e['tex']}` = {e['description']}" for e in d['entries']))
    L += ['', '### Defined keys that also occur with a dropped index subscript (same normalised key)', '']
    for d in a['iii_index_subscript_overloads']:
        L.append(f"- `{d['key']}`: subscript forms {d['subscript_forms']}")
    L += ['', '### Declared overloads (Worker reading; every use verified against the text)', '']
    for d in a['iii_declared_overloads_verified']:
        L.append(f"- **`{d['key']}`** -- {d['note']}")
        for u in d['uses']:
            L.append(f"  - {u['meaning']}: body lines {u['body_lines'][:20]}{' ...' if len(u['body_lines']) > 20 else ''}"
                     f"; nomenclature lines {u['nomenclature_lines']}")
    L += ['', '### Declared conflicts between nomenclature and body (Worker reading; lines verified)', '']
    for c in a['iii_declared_conflicts_verified']:
        L.append(f"- **{c['id']}**: {c['finding']} (lines " + ', '.join(str(e['line']) for e in c['evidence']) + ')')
    L += ['', '### Different nomenclature keys with the same description', '']
    for d in a['iii_same_description_different_keys']:
        L.append(f"- \"{d['description']}\": " + '; '.join(f"l. {e['line']} `{e['key']}`" for e in d['entries']))
    if not a['iii_same_description_different_keys']:
        L.append('none')
    L += ['', '### Local definitions in the body of nomenclature symbols (compare the wording)', '',
          '| key | line | body text | nomenclature |', '|---|---|---|---|']
    for d in a['iii_body_local_definitions_of_nomenclature_keys']:
        L.append(f"| `{d['key']}` | {d['line']} | {d['body_text']} | {' / '.join(d['nomenclature'])} |")
    L += ['', '## (iv) Symbols of the removed method (Benders)', '',
          '| key | meaning | nomenclature lines | body lines | n |', '|---|---|---|---|---|']
    for b in a['iv_benders_symbols']:
        L.append(f"| `{b['key']}` | {b['meaning']} | {b['nomenclature_lines']} | {b['body_lines']} | {b['n_occurrences']} |")
    L += ['']
    for label, h in a['iv_benders_raw_patterns'].items():
        L.append(f"- {label}: nomenclature lines {h['nomenclature_lines']}; body: " +
                 '; '.join(f"`{k}` {v}" for k, v in h['by_key'].items()))
    L += ['', f"Symbols occurring only inside Algorithm 1 (lines {a['algorithm1_lines']}, replaced wholesale per the map): "
          + ', '.join(f'`{k}`' for k in a['symbols_only_inside_algorithm1'])]
    L += ['', f"Lines containing the words Benders / cut(s) / underestimator / optimality gap / lower bound / upper bound: "
          f"{len(a['iv_benders_word_lines'])} -- " + ', '.join(str(w['line']) for w in a['iv_benders_word_lines']), '',
          '## (v) Symbols the revision needs (map section B) and paragraphs_v5.md', '',
          '| item | used in paragraphs_v5 | paragraphs_v5 notation (line: context) | main.tex today | STEP4 notation |',
          '|---|---|---|---|---|']
    for r in a['v_required_symbols']:
        ph = '; '.join(f"{h['line']}{'' if h['in_manuscript_scope'] else ' (comment, not manuscript)'}: {h['context']}"
                       for h in r['paragraphs_v5_hits'][:3]).replace('|', '/')
        mh = []
        for k, v in r['main_tex_related_symbol_lines'].items():
            if v:
                mh.append(f"`{k}` l. {v[:6]}{'...' if len(v) > 6 else ''}")
        for h in r['main_tex_text_hits'][:3]:
            mh.append(f"text l. {h['line']}")
        L.append(f"| {r['what']} | {r['paragraphs_v5_used']} | {ph or '-'} | {'; '.join(mh) or 'absent'} | "
                 f"{r['step4_dfo_notation'] or '-'} |")
    L += ['', f"Unknown LaTeX commands met inside math (ignored by the tokeniser): {a['unknown_latex_commands_in_math']}", '']
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  TASK B
# ======================================================================================================================
YEAR_RX = re.compile(r'(?<!\d)20(2[5-9]|3[0-9])(?!\d)')   # also inside file names (_2025_)


def enumerate_year_items(lines, clines):
    smap = section_map(lines)
    items = []
    envs = []
    stack = []
    for i, ln in enumerate(clines, 1):
        for m in re.finditer(r'\\(begin|end)\{(table|figure\*?|algorithm)\}', ln):
            if m.group(1) == 'begin':
                stack.append((m.group(2), i))
            elif stack:
                kind, s = stack.pop()
                envs.append((kind, s, i))
    covered = set()
    for kind, s, e in sorted(envs, key=lambda t: t[1]):
        body = '\n'.join(clines[s - 1:e])
        label = re.findall(r'\\label\{([^}]*)\}', body)
        caption = re.findall(r'\\caption\{(.*)\}', body)
        years = sorted({int('20' + y) for y in YEAR_RX.findall(body)})
        files = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}', body)
        for k in range(s, e + 1):
            covered.add(k)
        by_keyword = bool(re.search(r'representative year|multi_year', body))
        if not years and not by_keyword:
            continue
        items.append({'kind': kind, 'lines': [s, e], 'label': label[-1] if label else None,
                      'caption': caption[0] if caption else None, 'years': years, 'graphics': files,
                      'year_indexed_by': 'calendar-year literal' if years else 'keyword (representative year / multi_year)',
                      'section': smap.get(s)})
    text_lines = [{'line': i, 'section': smap.get(i), 'years': sorted({int('20' + y) for y in YEAR_RX.findall(ln)}),
                   'text': ln.strip()[:220]}
                  for i, ln in enumerate(clines, 1) if i not in covered and YEAR_RX.search(ln)]
    return items, text_lines


INPUT_TABLES = {
    'tab:investment_cost': 'cost',
    'tab:cs3_tn_res_generators': ('case9', None),
    'tab:cs3_adn_node_5_res_generators': ('case33_1', 5),
    'tab:cs3_adn_node_7_res_generators': ('case33_2', 7),
    'tab:cs3_adn_node_9_res_generators': ('case33_3', 9),
}


def parse_printed_res_table(clines, s, e):
    rows = []
    for ln in clines[s - 1:e]:
        cells = [c.strip() for c in re.sub(r'\\\\.*$', '', ln).split('&')]
        if len(cells) == 8 and re.fullmatch(r'\d+', cells[0]):
            rows.append({'gen_id': int(cells[0]), 'node': int(cells[1]), 'type': cells[2],
                         'values': dict(zip(PRINTED_YEARS, [float(v) for v in cells[3:]]))})
    return rows


def parse_printed_cost_table(clines, s, e):
    rows, param = [], None
    for ln in clines[s - 1:e]:
        cells = [c.strip() for c in re.sub(r'\\\\.*$', '', ln).split('&')]
        if len(cells) != 8 or not re.fullmatch(r'\d', cells[1]):
            continue
        if 'Inv,S' in cells[0]:
            param = 'power'
        elif 'Inv,E' in cells[0]:
            param = 'energy'
        rows.append({'parameter': param, 'scenario': int(cells[1]),
                     'probability_pct': float(cells[2].replace(r'\%', '')),
                     'values_k': dict(zip(PRINTED_YEARS, [float(v) for v in cells[3:]]))})
    return rows


def read_networks_production(net_name, years):
    """{year: [ {gen_id, bus, type, capacity_MW (production), Pmax_json} ]} via network._read_network_from_json_file."""
    import network as NW
    from definitions import GEN_RES_SOLAR, GEN_RES_WIND
    out, sources = {}, {}
    for y in years:
        rel = os.path.join(CASE_DIR, net_name, f'{net_name}_{y}.json')
        net = NW.Network()
        net.name, net.year = net_name, y
        NW._read_network_from_json_file(net, os.path.join(REPO, rel))
        with open(os.path.join(REPO, rel), encoding='utf-8') as handle:
            raw = json.load(handle)
        raw_pmax = {int(g['gen_id']): float(g['Pmax']) for g in raw['generators']}
        gens = []
        for g in net.generators:
            if g.gen_type in (GEN_RES_SOLAR, GEN_RES_WIND):
                cap = g.pmax * net.baseMVA
                gens.append({'gen_id': g.gen_id, 'bus': g.bus,
                             'type': 'Wind' if g.gen_type == GEN_RES_WIND else 'Solar PV',
                             'capacity_MW_production': cap, 'Pmax_json': raw_pmax[g.gen_id],
                             'production_equals_json': abs(cap - raw_pmax[g.gen_id]) < 1e-9})
        out[y] = gens
        sources[y] = rel
    return out, sources


def read_costs_production(xlsx_abs, years, discount):
    import shared_energy_storage_data as SED
    sed = SED.SharedEnergyStorageData()
    sed.years = {int(y): 5 for y in years}
    sed.discount_factor = discount
    SED._read_shared_energy_storage_data_from_file(sed, xlsx_abs)
    exp_e = {int(y): SED._get_expected_energy_investment_cost(sed, int(y)) for y in years}
    exp_s = {int(y): sum(p * sed.cost_investment['power'][m][int(y)] for m, p in enumerate(sed.prob_market_scenarios))
             for y in years}
    return sed, exp_s, exp_e


def fmt2(x):
    return f'{x:.2f}'


def res_fragment(label, caption, rows_by_year, order):
    yrs = TARGET_YEARS
    L = [r'\begin{table}[htbp!]', r'    \centering', f'    \\caption{{{caption}}}',
         r'    \begin{tabular}{ccc ccc}', r'        \toprule',
         r'        Generator   & Node  & Generator & \multicolumn{3}{c}{Installed capacity, [MW]}\\  \cmidrule{4-6}',
         r'        ID          & ID    & Type      & ' + ' & '.join(str(y) for y in yrs) + r' \\ ',
         r'        \midrule']
    for gid in order:
        g = {y: next(x for x in rows_by_year[y] if x['gen_id'] == gid) for y in yrs}
        g0 = g[yrs[0]]
        L.append(f"        {gid} & {g0['bus']} & {g0['type']} & " +
                 ' & '.join(fmt2(g[y]['capacity_MW_production']) for y in yrs) + r'\\')
    L += [r'        \bottomrule', r'    \end{tabular}', f'    \\label{{{label}}}', r'\end{table}']
    return '\n'.join(L) + '\n'


def cost_fragment(label, caption, sed, exp_s, exp_e):
    yrs = TARGET_YEARS
    probs = sed.prob_market_scenarios
    L = ['% W167: generated from data/SRP1/SharedESS/SRP1_ESS.xlsx (HEAD blob == 7ce1d1ab blob) through production\'s',
         '% reader; values in k EUR (file values / 1000, 2 dp). The planning objective uses the probability-weighted',
         '% sum over the three scenarios (expected 2025 / 2030 / 2035 below, comment only; not a row of the submitted table):',
         '%   E[c^Inv,S] k EUR/MVA: ' + ' / '.join(fmt2(exp_s[y] / 1000) for y in yrs),
         '%   E[c^Inv,E] k EUR/MWh: ' + ' / '.join(fmt2(exp_e[y] / 1000) for y in yrs),
         r'\begin{table}[htbp!]', r'    \centering', f'    \\caption{{{caption}}}',
         r'    \begin{tabular}{lcc ccc}', r'        \toprule',
         r'        Parameter & Scn. & Prob., [\%] & ' + ' & '.join(str(y) for y in yrs) + r' \\ ', r'        \midrule']
    heads = {'power': (r'$c^\text{Inv,S}_{y,c}$,    ', r'$\left[\text{k\euro/MVA}\right]$'),
             'energy': (r'$c^\text{Inv,E}_{y,c}$,', r'$\left[\text{k\euro/MWh}\right]$')}
    for pi, kind in enumerate(('power', 'energy')):
        for m in range(len(probs)):
            first = heads[kind][0] if m == 0 else (heads[kind][1] if m == 1 else '~                          ')
            L.append(f'        {first} & {m + 1} & {probs[m] * 100:.2f}\\% & ' +
                     ' & '.join(fmt2(sed.cost_investment[kind][m][y] / 1000) for y in yrs) + r'\\')
        if pi == 0:
            L.append(r'        \midrule')
    L += [r'        \bottomrule', r'    \end{tabular}', f'    \\label{{{label}}}', r'\end{table}']
    return '\n'.join(L) + '\n'


FIGURE_CODE_EVIDENCE = [
    ('market-price figures are plotted for the first representative year only', 'shared_resources_planning.py',
     'years_to_plot = list(self.years)[0]'),
    ('load / RES figures are plotted for the first representative year only', 'network_data.py',
     'years_to_plot = list(self.years)[0]'),
    ('flexibility figure: first representative year only', 'network_data.py',
     'years_to_plot = list(network_planning.years)[0]'),
    ('the synthetic profile pool is generated once per network (not per year)', 'network_data.py',
     'synthetic_profiles = _generate_operational_scenarios('),
    ('each (year, day) block draws its own realisation: the seed includes the year', 'network_data.py',
     "derive_random_seed(network_planning.random_seed, 'realization', int(year), str(day))"),
    ('RES realisation = sampled per-unit profile x installed capacity of that year', 'network.py',
     'gen_capacity = generator.pmax * network.baseMVA'),
    ('load realisation scaled by the cumulative load growth of that year', 'network.py',
     'load_growth_cumul = (1 + load_growth_factor) ** (network.year - initial_year)'),
    ('market price selection seed includes the year', 'shared_resources_planning.py',
     "random_state=derive_random_seed(market_seed, 'selection', 'energy', year, str(day)),"),
    ('market prices scaled by the cumulative energy-price growth of that year', 'shared_resources_planning.py',
     'energy_growth_cumul = (1 + energy_growth_factor) ** (year - initial_year)'),
    ('the RES figure plots the mean and std over the realised operation scenarios of network[year][season]',
     'network_data.py', 'pg_mean = pg.mean(axis=0)'),
]

FIGURE_ASSESSMENT = (
    'Input-data figures (market prices, TN / ADN RES, ADN loads, ADN flexibility): production plots them for the FIRST '
    'representative year only, from the realised operation scenarios of that year (mean +- std over the scenarios). '
    'Each (year, day) block draws its own realisation from a year-independent synthetic pool with a seed that includes '
    'the year, then scales it by the installed RES capacity / load growth / price growth of that year. So a 2030 or 2035 '
    'profile is a different draw from the pool, scaled, not the 2025 profile rescaled. The appendix sentences that the '
    '2025 profiles "are used as baseline trajectories in all representative years, scaled according to the '
    'installed-capacity evolution" (l. 1642) therefore describe the scaling but not the per-year draw. Whether 2030 / '
    '2035 versions are needed is for the author to decide. The submitted 2025 figures were produced for the '
    'submitted instance (l. 939: five operation scenarios); SRP1 draws ONE operation scenario per block and the 3 x 3 '
    'instance three, so a mean +- std band plotted from SRP1 would be a single curve (not checked against the PDFs). '
    'Results figures (operating cost / RES / curtailment "multi_year", the 2025 Winter voltage profile, the Benders '
    'convergence) are outputs of the submitted method: the map replaces or deletes them; no 2030 / 2035 version applies.')

RESULTS_TABLE_NOTE = {
    'tab:cs3_shared_ess_investment_plan': 'old Benders plan per cost scenario (map 4.1: replaced by the T1 headline)',
    'tab:cs3_shared_ess_specs': 'old SoH / available energy of the old plan (map section 4: replaced)',
    'tab:cs3_results_operational_planning_summary': 'old operational summary (map 4: deleted / regenerated from T6)',
    'tab:cs3_shared_ess_investment_plan_5year_discretization': 'discretization study (map 4.5: deleted)',
    'tab:cs3_shared_ess_specs_5year_discretization': 'discretization study (map 4.5: deleted)',
    'tab:cs3_shared_ess_investment_plan_1year_discretization': 'discretization study (map 4.5: deleted)',
    'tab:cs3_shared_ess_specs_1year_discretization': 'discretization study (map 4.5: deleted)',
    'tab:cs3_results_summary_years_days': 'Appendix E breakdown (map: replaced by the supplementary set)',
}


def task_b(lines, clines, out_dir, ess_head_abs):
    items, text_lines = enumerate_year_items(lines, clines)
    with open(os.path.join(REPO, CASE_JSON_REL), encoding='utf-8') as handle:
        case = json.load(handle)
    discount = float(case['DiscountFactor'])
    case_years = [int(y) for y in case['Years']]
    if tuple(case_years) != TARGET_YEARS:
        _fail(f'SRP1.json Years {case_years} != {TARGET_YEARS}')
    os.makedirs(os.path.join(out_dir, YEAR_TABLES_SUB), exist_ok=False)
    cross, fragments, classification = [], {}, []
    all_years = sorted(set(TARGET_YEARS) | set(PRINTED_YEARS))
    net_reads = {}
    for it in items:
        lab = it['label']
        if it['kind'] != 'table':
            continue
        kind = INPUT_TABLES.get(lab)
        if kind is None:
            classification.append({'label': lab, 'lines': it['lines'], 'class': 'RESULTS table (old solves)',
                                   'years': it['years'], 'regenerated': False,
                                   'why_not': 'content is solver output of the submitted (Benders) method; not a '
                                              'case-file quantity; zero-solve regeneration impossible',
                                   'map_disposition': RESULTS_TABLE_NOTE.get(lab, 'not in the map')})
            continue
        s, e = it['lines']
        if kind == 'cost':
            printed = parse_printed_cost_table(clines, s, e)
            sed, exp_s, exp_e = read_costs_production(ess_head_abs, all_years, discount)
            frag = cost_fragment(lab, it['caption'], sed, exp_s, exp_e)
            for r in printed:
                m = r['scenario'] - 1
                prob_file = sed.prob_market_scenarios[m] * 100
                cross.append({'table': lab, 'row': f"{r['parameter']} scn {r['scenario']}", 'column': 'probability',
                              'printed': r['probability_pct'], 'produced': round(prob_file, 2),
                              'agree': abs(prob_file - r['probability_pct']) < 0.005, 'year_scope': 'all'})
                for y in PRINTED_YEARS:
                    val = sed.cost_investment[r['parameter']][m][y] / 1000
                    cross.append({'table': lab, 'row': f"{r['parameter']} scn {r['scenario']}", 'column': y,
                                  'printed': r['values_k'][y], 'produced': round(val, 2),
                                  'produced_full_eur': sed.cost_investment[r['parameter']][m][y],
                                  'agree': abs(round(val, 2) - r['values_k'][y]) < 0.0051,
                                  'year_scope': 'target 2025' if y == 2025 else 'printed year (3x3 horizon)'})
            src = {'file': ESS_XLSX_REL, 'reader': 'shared_energy_storage_data._read_shared_energy_storage_data_from_file'}
        else:
            net_name, node = kind
            if net_name not in net_reads:
                net_reads[net_name] = read_networks_production(net_name, all_years)
            by_year, srcs = net_reads[net_name]
            printed = parse_printed_res_table(clines, s, e)
            order = [r['gen_id'] for r in printed]
            frag = res_fragment(lab, it['caption'], by_year, order)
            for r in printed:
                for y in PRINTED_YEARS:
                    g = next((x for x in by_year[y] if x['gen_id'] == r['gen_id']), None)
                    rec = {'table': lab, 'row': f"gen {r['gen_id']}", 'column': y, 'printed': r['values'][y],
                           'produced': g['capacity_MW_production'] if g else None,
                           'printed_node': r['node'], 'file_node': g['bus'] if g else None,
                           'printed_type': r['type'], 'file_type': g['type'] if g else None,
                           'year_scope': 'target 2025' if y == 2025 else 'printed year (3x3 horizon)'}
                    rec['agree'] = bool(g) and abs(g['capacity_MW_production'] - r['values'][y]) < 1e-9 \
                        and g['bus'] == r['node'] and g['type'] == r['type']
                    cross.append(rec)
            for y in all_years:
                for g in by_year[y]:
                    if not g['production_equals_json']:
                        _fail(f'{net_name} {y} gen {g["gen_id"]}: production capacity != raw JSON Pmax')
            src = {'files': {y: srcs[y] for y in all_years}, 'reader': 'network._read_network_from_json_file; '
                   'capacity = generator.pmax * baseMVA (network.py _update_network_with_operational_data)'}
        fname = lab.replace(':', '_') + '.tex'
        _write_x(os.path.join(out_dir, YEAR_TABLES_SUB, fname), frag)
        fragments[lab] = {'file': os.path.join(YEAR_TABLES_SUB, fname), 'source': src}
        classification.append({'label': lab, 'lines': it['lines'], 'class': 'INPUT-DATA table (case files)',
                               'years': it['years'], 'regenerated': True, 'fragment': fragments[lab]['file']})
    figures = []
    for it in items:
        if it['kind'] == 'table':
            continue
        is_result = any(g.startswith('results_') or g.startswith('voltage_profile') for g in it['graphics']) \
            or (it['label'] or '').startswith('fig:cs3_results')
        figures.append({'label': it['label'], 'lines': it['lines'], 'caption': it['caption'], 'graphics': it['graphics'],
                        'years': it['years'], 'year_indexed_by': it['year_indexed_by'],
                        'class': 'RESULTS figure (submitted method)' if is_result else 'INPUT-DATA figure (2025 profiles)',
                        'versions_2030_2035': 'not applicable (map replaces / deletes)' if is_result else
                        'author\'s choice; see figure_assessment (per-year draws differ from 2025, not a rescale)'})
    fig_ev = []
    for what, rel, needle in FIGURE_CODE_EVIDENCE:
        loc = _locate(rel, needle)
        if loc is None:
            _fail(f'figure code evidence not found: {rel}: {needle!r}')
        fig_ev.append({'what': what, 'located': loc})
    return {'items': items, 'text_lines': text_lines, 'classification': classification, 'fragments': fragments,
            'crosscheck': cross, 'figures': figures, 'figure_assessment': FIGURE_ASSESSMENT,
            'figure_code_evidence': fig_ev, 'net_reads': {k: v[0] for k, v in net_reads.items()}}


# ======================================================================================================================
#  TASK C
# ======================================================================================================================
def workbook_dump(xlsx_abs, years):
    import openpyxl
    wf = openpyxl.load_workbook(xlsx_abs, data_only=False)
    wv = openpyxl.load_workbook(xlsx_abs, data_only=True)
    sheets = []
    for ws in wf.worksheets:
        vs = wv[ws.title]
        hdr = [c.value for c in ws[1]]
        rows = []
        for r in range(1, ws.max_row + 1):
            cells = []
            for c in ws[r]:
                if c.value is None:
                    continue
                v = vs[c.coordinate].value
                cells.append({'cell': c.coordinate, 'formula_or_value': c.value if isinstance(c.value, (int, float, str)) else str(c.value),
                              'cached': v if isinstance(v, (int, float, str)) or v is None else str(v)})
            rows.append(cells)
        sel = {}
        if ws.title.startswith('Investment Cost'):
            for r in range(2, ws.max_row + 1):
                name = vs.cell(row=r, column=1).value
                if name is None:
                    continue
                sel[str(name)] = {int(y): vs.cell(row=r, column=j + 1).value for j, y in enumerate(hdr)
                                  if isinstance(y, int) and y in years}
        first_formula = {}
        for r in range(2, min(ws.max_row, 4) + 1):
            for c in ws[r]:
                if isinstance(c.value, str) and c.value.startswith('='):
                    first_formula[f'row {r}'] = {'cell': c.coordinate, 'formula': c.value}
                    break
        sheets.append({'sheet': ws.title, 'dimensions': ws.dimensions, 'max_row': ws.max_row,
                       'max_column': ws.max_column, 'header_row': [h if isinstance(h, (int, float, str)) or h is None
                                                                   else str(h) for h in hdr],
                       'first_formula_per_row': first_formula, 'values_selected_years': sel,
                       'n_nonempty_cells': sum(len(r) for r in rows), 'cells': rows})
    return sheets


def recompute_from_formulas(xlsx_abs, years):
    """The cached cost values recomputed from the workbook's own inputs, per the formulas it carries (the formula
    text is matched exactly first, so a different formula FAILS rather than being silently mis-modelled)."""
    import openpyxl
    wf = openpyxl.load_workbook(xlsx_abs, data_only=False)
    wv = openpyxl.load_workbook(xlsx_abs, data_only=True)
    rate = wv['Conversion rate']['C2'].value
    b = {r: wv['Cost breakdown NREL'][f'B{r}'].value for r in range(2, 10)}
    tot = sum(b.values())
    bos_half = sum(b[r] for r in range(3, 9)) / 2
    usd = wv['Investment Cost NREL, USD']
    hdr = {usd.cell(row=1, column=j).value: j for j in range(2, usd.max_column + 1)}
    out, checks = {}, []
    exp_p = ("=HLOOKUP({c}$1,'Investment Cost NREL, EUR'!$B$1:$AF$4,{r},FALSE)*1000*('Cost breakdown NREL'!$B$9+"
             "SUM('Cost breakdown NREL'!$B$3:$B$8)/2)/SUM('Cost breakdown NREL'!$B$2:$B$9)")
    exp_e = ("=HLOOKUP({c}$1,'Investment Cost NREL, EUR'!$B$1:$AF$4,{r},FALSE)/4*1000*('Cost breakdown NREL'!$B$2+"
             "SUM('Cost breakdown NREL'!$B$3:$B$8)/2)/SUM('Cost breakdown NREL'!$B$2:$B$9)")
    from openpyxl.utils import get_column_letter
    for kind, sheet, tmpl in (('power', 'Investment Cost, Power', exp_p), ('energy', 'Investment Cost, Energy', exp_e)):
        wsf, wsv = wf[sheet], wv[sheet]
        shdr = {wsf.cell(row=1, column=j).value: j for j in range(2, wsf.max_column + 1)}
        out[kind] = {}
        for m in range(3):
            out[kind][m] = {}
            for y in years:
                j = shdr[y]
                col = get_column_letter(j)
                formula = wsf.cell(row=m + 2, column=j).value
                expected_formula = tmpl.format(c=col, r=m + 2)
                eur_kw = usd.cell(row=m + 2, column=hdr[y]).value / rate
                val = eur_kw * 1000 * (b[9] + bos_half) / tot if kind == 'power' else eur_kw / 4 * 1000 * (b[2] + bos_half) / tot
                cached = wsv.cell(row=m + 2, column=j).value
                ok_f = formula == expected_formula
                ok_v = abs(val - cached) <= 1e-6 * abs(cached)
                checks.append({'kind': kind, 'scenario': m + 1, 'year': y, 'cell': f'{col}{m + 2}',
                               'formula_matches_template': ok_f, 'recomputed': val, 'cached': cached,
                               'recomputed_equals_cached': ok_v})
                if not ok_f:
                    _fail(f'{sheet} {col}{m + 2}: formula {formula!r} is not the template')
                if not ok_v:
                    _fail(f'{sheet} {col}{m + 2}: recomputed {val} != cached {cached}')
                out[kind][m][y] = val
    return {'conversion_rate_usd_per_eur': rate, 'cost_breakdown_usd_per_kwh': b,
            'power_share': (b[9] + bos_half) / tot, 'energy_share': (b[2] + bos_half) / tot,
            'formula_templates': {'power': exp_p, 'energy': exp_e}, 'checks': checks}


CODE_EVIDENCE = [
    ('SRP1.json names the cost file', CASE_JSON_REL, '"data_file": "SRP1_ESS.xlsx"'),
    ('file opened from data/SRP1/SharedESS', 'shared_energy_storage_data.py',
     "filename = os.path.join(self.data_dir, 'SharedESS', self.data_file)"),
    ('reader: scenario count and probabilities from sheet Scenarios', 'shared_energy_storage_data.py',
     "num_scenarios, shared_ess_data.prob_market_scenarios = _get_operational_scenarios_info_from_excel_file(filename, 'Scenarios')"),
    ('reader: all scenarios of the power sheet', 'shared_energy_storage_data.py',
     "investment_costs['power'] = _get_investment_costs_from_excel_file(filename, 'Investment Cost, Power', num_scenarios, shared_ess_data.years)"),
    ('reader: all scenarios of the energy sheet', 'shared_energy_storage_data.py',
     "investment_costs['energy'] = _get_investment_costs_from_excel_file(filename, 'Investment Cost, Energy', num_scenarios, shared_ess_data.years)"),
    ('reader: probabilities must sum to 1', 'shared_energy_storage_data.py',
     "if round(sum(prob_scenarios), 2) != 1.00:"),
    ('reader: row i+1 is scenario i, column = year', 'shared_energy_storage_data.py',
     "data[i][year] = float(df.iloc[i + 1, j + 1])"),
    ('budget-row and master-objective weight omega_m (both occurrences)', 'shared_energy_storage_data.py',
     "omega_m = shared_ess_data.prob_market_scenarios[s_m]"),
    ('master objective term (probability-weighted sum)', 'shared_energy_storage_data.py',
     "investment_cost += annualization * omega_m * model.es_s_investment[e, y] * c_inv_s"),
    ('master objective expression', 'shared_energy_storage_data.py',
     "model.investment_cost = pe.Expression(expr=investment_cost)"),
    ('budget row (probability-weighted)', 'shared_energy_storage_data.py',
     "model.energy_storage_investment.add(investment_cost_total <= shared_ess_data.params.budget)"),
    ('expected energy cost (salvage basis)', 'shared_energy_storage_data.py',
     "def _get_expected_energy_investment_cost(shared_ess_data, year):"),
    ('oracle I(x): the master expression transcribed', 'p56a_oracle.py', "def investment_cost(planning, candidate):"),
    ('oracle I(x): sum over the workbook scenarios', 'p56a_oracle.py',
     "for scenario, probability in enumerate(esso.prob_market_scenarios):"),
    ('oracle: every evaluation records I(x)', 'p56a_oracle.py',
     "result['investment_cost'] = investment_cost(planning, candidate)"),
    ('oracle: total objective = I(x) + net recourse', 'p56a_oracle.py',
     "result['total_objective'] = (result['investment_cost']"),
    ('method definition of I(x)', DFO_REL, r'I(x) = \sum_{n,y} \frac{1}{(1+d)^{\,y - y_1}} \sum_{m} \omega_m'),
    ('method: the weights are the workbook scenario weights', DFO_REL,
     'investment-cost scenario weights (0.35/0.55/0.10)'),
]

SELECTION_PATTERNS = [r'cost_scenario', r'investment_cost_scenario', r'cost_trajector', r'scenario_select',
                      r'selected_scenario', r'NumCostScenarios', r'num_cost_scenarios', r'cost_scenario_index',
                      r'prob_market_scenarios', r'investment.?cost.?prob']


def task_c(tmpdir):
    rec = {}
    head_blob = _git('rev-parse', f'HEAD:{ESS_XLSX_REL}')
    c_blob = _git('rev-parse', f'{COST_COMMIT}:{ESS_XLSX_REL}')
    p_blob = _git('rev-parse', f'{COST_PREDECESSOR_COMMIT}:{ESS_XLSX_REL}')
    c_bytes = _git_bytes('show', f'{COST_COMMIT}:{ESS_XLSX_REL}')
    p_bytes = _git_bytes('show', f'{COST_PREDECESSOR_COMMIT}:{ESS_XLSX_REL}')
    with open(os.path.join(REPO, ESS_XLSX_REL), 'rb') as handle:
        h_bytes = handle.read()
    c_abs = os.path.join(tmpdir, 'SRP1_ESS_7ce1d1ab.xlsx')
    p_abs = os.path.join(tmpdir, 'SRP1_ESS_072b1310.xlsx')
    with open(c_abs, 'wb') as handle:
        handle.write(c_bytes)
    with open(p_abs, 'wb') as handle:
        handle.write(p_bytes)
    rec['identity'] = {
        'commit_7ce1d1ab_full': _git('rev-parse', COST_COMMIT),
        'commit_7ce1d1ab_subject': _git('log', '-1', '--format=%ad %s', COST_COMMIT),
        'blob_at_7ce1d1ab': c_blob, 'blob_at_HEAD': head_blob, 'blob_at_072b1310_predecessor': p_blob,
        'sha256_at_7ce1d1ab': _sha_bytes(c_bytes), 'sha256_at_HEAD_working_tree': _sha_bytes(h_bytes),
        'sha256_at_072b1310': _sha_bytes(p_bytes),
        'HEAD_identical_to_7ce1d1ab': (c_blob == head_blob) and (c_bytes == h_bytes),
        'working_tree_clean': _git('status', '--porcelain', '--', ESS_XLSX_REL) == '',
        'history_of_path_on_branch': _git('log', '--format=%h %ad %s', '--date=short', '--', ESS_XLSX_REL).split('\n'),
        'how_read': 'git show <commit>:data/SRP1/SharedESS/SRP1_ESS.xlsx into a temporary directory (deleted at exit)'}
    if not rec['identity']['HEAD_identical_to_7ce1d1ab']:
        _fail('SRP1_ESS.xlsx at HEAD differs from the 7ce1d1ab blob')
    yrs = sorted(set(TARGET_YEARS) | set(PRINTED_YEARS))
    rec['sheets_7ce1d1ab'] = workbook_dump(c_abs, yrs)
    rec['formula_recompute_7ce1d1ab'] = recompute_from_formulas(c_abs, yrs)
    pred = workbook_dump(p_abs, yrs)
    rec['predecessor_072b1310_selected'] = [{'sheet': s['sheet'], 'first_formula_per_row': s['first_formula_per_row'],
                                             'values_selected_years': s['values_selected_years'],
                                             'cells': s['cells'] if s['sheet'] == 'Scenarios' else None}
                                            for s in pred if s['sheet'].startswith('Investment Cost')
                                            or s['sheet'] == 'Scenarios']
    with open(os.path.join(REPO, CASE_JSON_REL), encoding='utf-8') as handle:
        discount = float(json.load(handle)['DiscountFactor'])
    sed, exp_s, exp_e = read_costs_production(c_abs, yrs, discount)
    sed_h, exp_s_h, exp_e_h = read_costs_production(os.path.join(REPO, ESS_XLSX_REL), yrs, discount)
    if _norm(sed.cost_investment) != _norm(sed_h.cost_investment) or sed.prob_market_scenarios != sed_h.prob_market_scenarios:
        _fail('production read of the 7ce1d1ab blob != production read of the HEAD file')
    rec['as_read_by_production'] = {
        'reader': 'shared_energy_storage_data._read_shared_energy_storage_data_from_file (called on the 7ce1d1ab copy '
                  'and on the HEAD file; identical)',
        'num_scenarios': len(sed.prob_market_scenarios),
        'scenario_probabilities': list(sed.prob_market_scenarios),
        'cost_power_eur_per_MVA': {m + 1: {y: sed.cost_investment['power'][m][y] for y in yrs} for m in range(len(sed.prob_market_scenarios))},
        'cost_energy_eur_per_MWh': {m + 1: {y: sed.cost_investment['energy'][m][y] for y in yrs} for m in range(len(sed.prob_market_scenarios))},
        'expected_power_eur_per_MVA': exp_s, 'expected_energy_eur_per_MWh_production_function': exp_e,
        'expected_formula': 'sum_m prob[m] * cost[m][year] (energy: shared_energy_storage_data._get_expected_energy_'
                            'investment_cost; power: the same sum, computed here)'}
    # I(x) of the unit candidate through the oracle's transcription, against the committed record
    import p56a_oracle as O
    nodes = (5, 7, 9)
    cand = {'investment': {n: {y: {'s': 0.0, 'e': 0.0} for y in TARGET_YEARS} for n in nodes}}
    cand['investment'][UNIT_CANDIDATE['node']][UNIT_CANDIDATE['year']] = {'s': UNIT_CANDIDATE['s'], 'e': UNIT_CANDIDATE['e']}
    sed_t, _s, _e = read_costs_production(c_abs, TARGET_YEARS, discount)
    i_oracle = O.investment_cost(types.SimpleNamespace(shared_ess_data=sed_t), cand)
    i_expected_closed = UNIT_CANDIDATE['s'] * exp_s[2025] + UNIT_CANDIDATE['e'] * exp_e[2025]
    with open(os.path.join(REPO, W101_SUMMARY_REL), encoding='utf-8') as handle:
        w101 = json.load(handle)
    i_recorded = w101['predictions_scored']['expert_P2']['I']
    with open(os.path.join(REPO, INSTANCE_3X3_RECORD_REL), encoding='utf-8') as handle:
        inst = json.load(handle)
    i_rec_3x3 = inst['facts']['investment_cost']['n7_4h_e1']['I_x_eur_master_expression']
    rec['I_unit_check'] = {
        'candidate': UNIT_CANDIDATE, 'I_p56a_transcription': i_oracle, 'I_closed_form_expected_trajectory': i_expected_closed,
        'I_recorded_w101_three_reference_summary_expert_P2': i_recorded, 'I_recorded_3x3_instance_record': i_rec_3x3,
        'all_equal_to_1e-6': abs(i_oracle - i_recorded) < 1e-6 and abs(i_expected_closed - i_recorded) < 1e-6
                             and abs(i_rec_3x3 - i_recorded) < 1e-6,
        'single_scenario_values': {m + 1: UNIT_CANDIDATE['s'] * sed.cost_investment['power'][m][2025]
                                   + UNIT_CANDIDATE['e'] * sed.cost_investment['energy'][m][2025] for m in range(3)},
        'recorded_sources': [W101_SUMMARY_REL + ' predictions_scored.expert_P2.I', INSTANCE_3X3_RECORD_REL + ' facts.investment_cost.n7_4h_e1']}
    if not rec['I_unit_check']['all_equal_to_1e-6']:
        _fail('I(unit) recomputation does not reproduce the committed record')
    # T3 break-even: the energy-cost line the frozen tables compare against
    with open(os.path.join(REPO, FZ_REL), encoding='utf-8') as handle:
        fz = json.load(handle)
    be = fz.get('T3', {}) if isinstance(fz, dict) else {}
    found = []

    def walk(o, path):
        if isinstance(o, dict):
            if 'breakeven_marginal_4h_energy_cost' in o and 'margin_to_energy_cost_per_MWh' in o:
                found.append({'path': path, 'breakeven': o['breakeven_marginal_4h_energy_cost'],
                              'margin': o['margin_to_energy_cost_per_MWh'],
                              'implied_energy_cost': o['breakeven_marginal_4h_energy_cost'] + o['margin_to_energy_cost_per_MWh']})
            for k, v in o.items():
                walk(v, f'{path}/{k}')
        elif isinstance(o, list):
            for k, v in enumerate(o):
                walk(v, f'{path}[{k}]')
    walk(fz, '')
    rec['frozen_T3_energy_cost_line'] = {'source': FZ_REL, 'entries': found[:6], 'n_entries': len(found),
                                         'all_imply_expected_2025_energy_cost': all(
                                             abs(f['implied_energy_cost'] - exp_e[2025]) < 1e-6 for f in found) if found else None}
    del be
    # code evidence (file:line located by exact text)
    ev = []
    for what, rel, needle in CODE_EVIDENCE:
        loc = _locate(rel, needle)
        clean = _git('status', '--porcelain', '--', rel) == ''
        if loc is None:
            _fail(f'code evidence not found: {rel}: {needle!r}')
        ev.append({'what': what, 'located': loc, 'file_clean_vs_HEAD': clean,
                   'file_last_commit': _git('log', '-1', '--format=%h', '--', rel)})
    rec['code_evidence'] = ev
    # scoped search: committed campaign specs / frozen specs for a cost-scenario selection field
    spec_files = [f for f in _git('ls-files', 'data/SRP1/Results').split('\n')
                  if f.endswith('.json') and ('campaign_spec' in os.path.basename(f) or os.path.basename(f).startswith('frozen_'))]
    sel_hits, pin_hits = [], []
    for f in spec_files:
        with open(os.path.join(REPO, f), encoding='utf-8', errors='replace') as handle:
            txt = handle.read()
        for rx in SELECTION_PATTERNS:
            for m in re.finditer(rx, txt, re.I):
                lo = txt.rfind('\n', 0, m.start()) + 1
                hi = txt.find('\n', m.end())
                sel_hits.append({'file': f, 'pattern': rx, 'line_text': txt[lo:hi].strip()[:200]})
        if 'SRP1_ESS.xlsx' in txt:
            pins = re.findall(r'[0-9a-f]{64}', txt[txt.find('SRP1_ESS.xlsx'):txt.find('SRP1_ESS.xlsx') + 400])
            pin_hits.append({'file': f, 'sha256_near_path': sorted(set(pins))})
    rec['selection_field_search'] = {
        'scope': 'every git-tracked *.json under data/SRP1/Results whose basename contains "campaign_spec" or starts '
                 'with "frozen_" (at HEAD, working tree)', 'n_files_searched': len(spec_files),
        'patterns': SELECTION_PATTERNS, 'hits': sel_hits[:200], 'n_hits': len(sel_hits),
        'specs_pinning_SRP1_ESS_xlsx': len(pin_hits),
        'distinct_sha256_within_400_chars_after_the_path': sorted({p for h in pin_hits for p in h['sha256_near_path']}),
        'n_specs_pinning_the_7ce1d1ab_sha256': sum(1 for h in pin_hits if rec['identity']['sha256_at_7ce1d1ab'] in h['sha256_near_path']),
        'n_specs_pinning_the_072b1310_sha256': sum(1 for h in pin_hits if rec['identity']['sha256_at_072b1310'] in h['sha256_near_path']),
        'pins': pin_hits}
    return rec, c_abs


# ======================================================================================================================
#  sources, manifests
# ======================================================================================================================
def case_file_sources():
    with open(os.path.join(REPO, CASE_JSON_REL), encoding='utf-8') as handle:
        case = json.load(handle)
    with open(os.path.join(REPO, INSTANCE_3X3_REL), encoding='utf-8') as handle:
        inst = json.load(handle)
    years_srp1 = [int(y) for y in case['Years']]
    years_3x3 = [int(y) for y in inst['Years']]
    rels = [(CASE_JSON_REL, 'SRP1 case definition (shared_resources_planning._read_planning_problem)'),
            (os.path.join(CASE_DIR, case['PlanningParameters']['params_file']), 'planning parameters'),
            (os.path.join(CASE_DIR, 'MarketData', case['MarketData']), 'market data (shared_resources_planning: MarketData dir)'),
            (os.path.join(CASE_DIR, 'SharedESS', case['SharedEnergyStorage']['params_file']), 'ESS parameters'),
            (os.path.join(CASE_DIR, 'SharedESS', case['SharedEnergyStorage']['data_file']), 'ESS investment-cost workbook')]
    nets = [case['TransmissionNetwork']] + list(case['DistributionNetworks'])
    for nt in nets:
        nm = nt['name']
        rels.append((os.path.join(CASE_DIR, nm, nt['params_file']), f'{nm} parameters'))
        rels.append((os.path.join(CASE_DIR, nm, nt['operational_data_file']), f'{nm} operational data'))
        for y in sorted(set(years_srp1) | set(years_3x3)):
            scope = []
            if y in years_srp1:
                scope.append('SRP1')
            if y in years_3x3:
                scope.append('3x3 instance')
            rels.append((os.path.join(CASE_DIR, nm, f'{nm}_{y}.json'), f'{nm} network {y} (read by: {", ".join(scope)})'))
    rels.append((INSTANCE_3X3_REL, '3x3 derived case definition (Years / scenario counts)'))
    out = []
    for rel, role in rels:
        out.append({'path': rel, 'role': role, 'git_blob_HEAD': _git('rev-parse', f'HEAD:{rel}'),
                    'sha256_working_tree': _sha(rel), 'last_commit': _git('log', '-1', '--format=%h %ad', '--date=short', '--', rel),
                    'clean_vs_HEAD': _git('status', '--porcelain', '--', rel) == ''})
    for o in out:
        if not o['clean_vs_HEAD']:
            _fail(f"case file not clean vs HEAD: {o['path']}")
    reading_code = [
        _locate('network.py', "filename = os.path.join(self.data_dir, self.name, f'{self.name}_{self.year}.json')"),
        _locate('network.py', "generator.pmax = float(gen_data['Pmax']) / network.baseMVA"),
        _locate('network.py', 'gen_capacity = generator.pmax * network.baseMVA'),
        _locate('network_data.py', 'filename = os.path.join(network_planning.data_dir, network_planning.name, network_planning.operational_data_file)'),
        _locate('shared_resources_planning.py', "planning_problem.years[int(year)] = planning_data['Years'][year]"),
        _locate('shared_resources_planning.py', "filename = os.path.join(planning_problem.data_dir, 'MarketData', planning_problem.market_data_file)"),
        _locate('shared_energy_storage_data.py', "filename = os.path.join(self.data_dir, 'SharedESS', self.data_file)"),
        _locate('p56a_oracle.py', "planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')"),
    ]
    return {'years_srp1': years_srp1, 'years_3x3_instance': years_3x3,
            'srp1_Years_block': case['Years'], 'instance_3x3_Years_block': inst['Years'],
            'files': out, 'production_reading_code': reading_code}


def manifest(paths):
    return {p: _sha(p) for p in sorted(paths)}


def post_run(out_dir):
    names = [os.path.join(out_dir, n) for n in (OUT_NAMES['log'], OUT_NAMES['typing_json'], OUT_NAMES['typing_log'],
                                                OUT_NAMES['man'])]
    for rel in names:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W167 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = manifest(names)
    _write_json(os.path.join(REPO, out_dir, OUT_NAMES['post']), man)
    _log(f'[W167 post-run] wrote {OUT_NAMES["post"]}: ' + ', '.join(f'{os.path.basename(k)} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    out_rel = args.out_dir or OUT_DIR
    out_abs = out_rel if os.path.isabs(out_rel) else os.path.join(REPO, out_rel)
    if args.post_run:
        post_run(out_rel)
    t0 = time.time()
    _log(f'[W167] start; HEAD {_git("rev-parse", "HEAD")}; out {out_rel}')
    # ---- preconditions ------------------------------------------------------------------------------------------
    pre = []
    if not os.path.isdir(out_abs):
        pre.append(f'{out_rel} missing (mkdir it before the launch; the launch log goes there)')
    else:
        extra = [f for f in os.listdir(out_abs) if f != OUT_NAMES['log']]
        if extra:
            pre.append(f'{out_rel} holds {extra} (write-once: refuse)')
    if _sha(MAIN_REL) != MAIN_SHA:
        pre.append(f'main.tex sha256 {_sha(MAIN_REL)} != pinned {MAIN_SHA}')
    if _sha(P5_REL) != P5_SHA:
        pre.append(f'paragraphs_v5.md sha256 {_sha(P5_REL)} != pinned {P5_SHA}')
    clone_head = _git('rev-parse', '--short=7', 'HEAD', cwd=os.path.join(REPO, CLONE_REL))
    if clone_head != CLONE_HEAD_EXPECTED:
        pre.append(f'clone HEAD {clone_head} != {CLONE_HEAD_EXPECTED}')
    clone_main_clean = _git('status', '--porcelain', '--', 'main.tex', cwd=os.path.join(REPO, CLONE_REL)) == ''
    if not clone_main_clean:
        pre.append('main.tex modified in the clone')
    for rel in (P5_REL, FZ_REL, W101_SUMMARY_REL, INSTANCE_3X3_RECORD_REL, INSTANCE_3X3_REL, MAP_REL, DFO_REL):
        if _git('status', '--porcelain', '--', rel) != '':
            pre.append(f'{rel} not clean vs HEAD')
    if pre:
        for p in pre:
            _log(f'[W167 PRECONDITION FAILED] {p}')
        sys.exit(1)
    inputs = [MAIN_REL, P5_REL, MAP_REL, DFO_REL, FZ_REL, W101_SUMMARY_REL, INSTANCE_3X3_RECORD_REL, INSTANCE_3X3_REL,
              SCRIPT_REL, 'shared_energy_storage_data.py', 'network.py', 'network_data.py', 'shared_resources_planning.py',
              'p56a_oracle.py', 'definitions.py', 'p513_solve_profile_guard.py', 'gate_result_io.py']
    inputs_man = {'files': manifest(inputs),
                  'main_tex_clone_blob': _git('rev-parse', 'HEAD:main.tex', cwd=os.path.join(REPO, CLONE_REL)),
                  'main_tex_clone_head': _git('rev-parse', 'HEAD', cwd=os.path.join(REPO, CLONE_REL)),
                  'main_tex_sha256_prefix_in_task': MAIN_SHA_AS_GIVEN_IN_TASK,
                  'main_tex_sha256_matches_task_prefix': MAIN_SHA.startswith(MAIN_SHA_AS_GIVEN_IN_TASK)}
    # ---- read -----------------------------------------------------------------------------------------------------
    raw, lines = read_tex(MAIN_REL)
    clines = strip_comments(lines)
    ctext = '\n'.join(clines)
    with open(os.path.join(REPO, P5_REL), encoding='utf-8') as handle:
        p5_lines = handle.read().split('\n')
    # ---- Task A ---------------------------------------------------------------------------------------------------
    _log('[W167] Task A: nomenclature audit')
    audit = task_a(lines, clines, ctext, p5_lines)
    _log(f"[W167] Task A counts: {audit['counts']}")
    # ---- Task C first (the temp copy of the 7ce1d1ab blob feeds Task B's cross-read) ------------------------------
    with tempfile.TemporaryDirectory(prefix='w167_') as tmp:
        _log('[W167] Task C: cost file')
        q4, c_abs = task_c(tmp)
        _log(f"[W167] Task C: HEAD == 7ce1d1ab: {q4['identity']['HEAD_identical_to_7ce1d1ab']}; probabilities "
             f"{q4['as_read_by_production']['scenario_probabilities']}; expected 2025 power "
             f"{q4['as_read_by_production']['expected_power_eur_per_MVA'][2025]:.2f} EUR/MVA, energy "
             f"{q4['as_read_by_production']['expected_energy_eur_per_MWh_production_function'][2025]:.2f} EUR/MWh; "
             f"I(unit) check {q4['I_unit_check']['all_equal_to_1e-6']}")
        _log('[W167] Task B: year-dependent tables')
        yb = task_b(lines, clines, out_abs, os.path.join(REPO, ESS_XLSX_REL))
        n_dis = sum(1 for c in yb['crosscheck'] if not c['agree'])
        n_dis25 = sum(1 for c in yb['crosscheck'] if not c['agree'] and c['column'] in (2025, 'probability'))
        _log(f"[W167] Task B: {len(yb['fragments'])} fragments; cross-check cells {len(yb['crosscheck'])}, "
             f"disagree {n_dis} (2025 / probability column: {n_dis25})")
    sources = case_file_sources()
    sources['main_tex'] = {'path': MAIN_REL, 'sha256': MAIN_SHA, 'clone_head': inputs_man['main_tex_clone_head'],
                           'clone_blob': inputs_man['main_tex_clone_blob'], 'clean_in_clone': clone_main_clean,
                           'tracked_in_this_repository': _git('ls-files', '--', MAIN_REL) != ''}
    sources['cost_file_7ce1d1ab'] = q4['identity']
    # ---- guards ---------------------------------------------------------------------------------------------------
    gfail = GUARD.verify(0)
    if gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        _log(f'[W167 GUARD FAULT] {gfail} pickle {PICKLE_COUNTS}')
        sys.exit(1)
    run = {'stage': 'P5.15 Step 6 support W167 (zero solves)', 'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
           'git_HEAD': _git('rev-parse', 'HEAD'), 'started_utc_wall_s': round(time.time() - t0, 2),
           'interpreter': sys.executable, 'python': sys.version.split()[0],
           'guard': {'permitted': [], 'counts': GUARD.counts, 'verify_0_failures': gfail}, 'pickle_counts': PICKLE_COUNTS,
           'integrity_failures': FAILED, 'finished_utc': datetime.now(timezone.utc).isoformat()}
    cross_out = {'scope': 'every cell of every year-indexed INPUT-DATA table of main.tex: the printed value against the '
                          'case files read through production (the printed 2025 column against the target 2025 column; '
                          'the printed 2028-2037 columns against the files of those years, the 3x3 horizon)',
                 'n_cells': len(yb['crosscheck']), 'n_disagree': sum(1 for c in yb['crosscheck'] if not c['agree']),
                 'n_disagree_2025_or_probability': sum(1 for c in yb['crosscheck']
                                                       if not c['agree'] and c['column'] in (2025, 'probability')),
                 'cells': yb['crosscheck']}
    enum_out = {'tables_and_figures': yb['items'], 'classification': yb['classification'],
                'year_indexed_text_lines': yb['text_lines'], 'figures': yb['figures'], 'fragments': yb['fragments'],
                'figure_assessment': yb['figure_assessment'], 'figure_code_evidence': yb['figure_code_evidence'],
                'network_reads_production': yb['net_reads']}
    # ---- write ----------------------------------------------------------------------------------------------------
    p = lambda k: os.path.join(out_abs, OUT_NAMES[k])  # noqa: E731
    _write_json(p('audit_json'), audit)
    _write_x(p('audit_md'), md_audit(audit, MAIN_SHA))
    _write_json(p('cross'), cross_out)
    _write_json(p('enum'), enum_out)
    _write_json(p('q4'), q4)
    _write_json(p('sources'), sources)
    _write_json(p('inputs_man'), inputs_man)
    _write_json(p('run'), run)
    written = [os.path.relpath(os.path.join(dp, f), REPO) for dp, _d, fs in os.walk(out_abs) for f in fs
               if f not in (OUT_NAMES['log'], OUT_NAMES['man'])]
    _write_json(p('man'), manifest(written))
    _log(f'[W167] wrote {len(written)} files + manifest; integrity failures {len(FAILED)}; wall {time.time() - t0:.1f}s')
    sys.exit(3 if FAILED else 0)


if __name__ == '__main__':
    main()
