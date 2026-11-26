"""P5.15 Addendum 82 v2, W182 order items A3, A4, A5, A6 -- Planner task W182c: LABEL MAP 8f051a2 -> edd7a63 AND THE
LETTER v9 POINTER CHECK AGAINST IT (A3); DANGLING OR RE-POINTED REFERENCES AND PROSE MENTIONS (A4); TEXT WHOSE
\\textcolor{blue} WRAPPER WAS DROPPED (A5); APPENDIX E AT edd7a63 AGAINST gen_appE.py AND verify_appE.py (A6).
ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT. Read-only on the Overleaf clone (git rev-parse / rev-list /
show / diff / status / log only), on manuscript_review/, on every committed artifact and on every earlier script.

NUMBERING IS STATIC. No TeX installation exists on this machine (pdflatex, latex, latexmk, lualatex, xelatex absent).
Every number is the static elsarticle numbering of W177b-3'/W178c (`numbering_adjusted`: comment-stripped source,
sections / subsections / subsubsections, equation / align rows / subequations, captioned floats, longtables, the
\\addtocounter{table}{-1} after each caption-less nomenclature longtable, \\appendix as in the clone's elsarticle.dtx:
section, subsection and equation reset, figure and table counters NOT reset and printed <letter>.<n>). Nothing compiled.

WHAT IT DOES.
  0. Identity: clone HEAD (declared edd7a63, full hash checked), tracked files clean; commits after 8f051a2 and after
     d2ebb0d; files changed 8f051a2 -> HEAD; main.tex sha256 / lines at HEAD (= working copy) and at 8f051a2; the letter
     at HEAD = its blob at d2ebb0d (its last change); the letter at cd57efc = manuscript_review/response_to_reviewers_v8.tex;
     letter v8 -> HEAD hunks (U3 and U0) and the three Addendum 81 v9 sentences present; main.tex diff hunk count.
  1. Capture-path checklist (exit 4 naming the missing item) BEFORE any task runs.
  2. A3 label map: every label (table, figure, equation, section / subsection / subsubsection, appendix, algorithm) at
     8f051a2 and at edd7a63 -- number at each, unchanged / renumbered / removed / added -- with its caption or title at
     each; the numbered headings mapped old -> new (label, else exact title, else the heading whose own text holds the
     largest share of the old heading's sentences); unlabelled equations mapped by body. Cross-checks: the numbering
     at 8f051a2 equals that at cd57efc and e5322f8 (Addendum 81: "every label number identical to round 7").
  3. A3 pointers: every numbered pointer of letter v9 (the W177b-3' extractor `letter_pointers`, unchanged). Its
     intended target is the object that carried the printed number at 8f051a2 (the numbering the letter was written
     against; W180c found 0 MISMATCH at e5322f8). SUBMITTED pointers are W180c's five, carried through the v7 -> v9
     line map. The target is resolved at edd7a63: MATCH (same number), MISMATCH (renumbered, or target removed; ranges:
     an endpoint differs or a member is removed), SUBMITTED, AMBIGUOUS (no unique object at 8f051a2, or a removed
     unlabelled heading whose content is not located). Each pointer carries what its printed number denotes at
     edd7a63. Cross-check: for every v9 pointer on a line mapped to v7, the derived target equals W180c's declared
     target (resolved at 8f051a2). W180c's content needles are re-read in the target region at edd7a63 (informational).
  4. A4: undefined \\ref / \\eqref keys (static) at edd7a63 with every use line; a "Table/Figure/Section/Eq./Algorithm"
     word before a \\ref whose label is of another kind; labels kept whose object's caption / title changed; literal
     (typed) numbered pointers in main.tex with what each number denotes at 8f051a2 and edd7a63; prose mentions of
     removed objects ("ladder", "flexibility(-price) ladder", "shaded region", "illustration") in main.tex and letter.
  5. A5: \\textcolor{blue}{...} masks at both revisions (brace-matched, comment-stripped); lines aligned, then tokens
     inside changed line blocks; tokens blue at 8f051a2 and not blue at edd7a63 -- "unwrapped" (same words) and
     "rewritten, not blue" (a replaced run that was blue) -- listed by edd7a63 line with first words.
  6. A6: manuscript_review/gen_appE.py run IN-PROCESS with an explicit output path inside this stage's output
     directory (never the cwd default appE.tex, never manuscript_review/appendixE_evaluation_record.tex); bitwise
     location in main.tex at edd7a63 (and at 8f051a2 as the positive control); otherwise the line diff of the Appendix
     E block (W181a's block_diff) and the entry diff of Tables E.20-E.24 (W181a's raw_tables / diff_tables);
     manuscript_review/verify_appE.py run in-process (it regenerates into its own tempfile.mkdtemp()); PASS / FAIL.
  7. Negative controls (in memory, or a tempfile.mkdtemp() directory outside the repository).

IMPORTED, NOT EDITED: p515_s53_w179b_checks.py (and its chain W178c, W177b-3', W177b-3, W167, S47 phase-B record) for
numbering_adjusted, task_refs, letter_pointers, cite_order, region functions and task_elsarticle;
p515_s53_w181a_identity_appendixE.py for run_generator, run_verifier, locate_bytes, appendix_block, raw_tables,
diff_tables, block_diff. Each installs its own permitted=() guard, verified at 0 at exit.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0;
pickle.load / pickle.loads blocked and counted (0 expected); no production module imported (asserted). Refuses to run
concurrently with another copy of itself. JSON written through gate_result_io (string flags refused).

OUTPUT (write-once): data/SRP1/Results/P515S53/w182c_labels_pointers_appE/ must exist and hold nothing but launch.log;
writes w182c_labels_pointers_appE.json, w182c_labels_pointers_appE.md, appE_generated_w182c.tex, verify_appE_output.txt
and manifest_sha256.json. `--post-run` (after the run and the W100 typing test) writes manifest_post_run_sha256.json.
`--trial-out-dir DIR`: trial runs only (absolute path outside the repository).

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w182c_labels_pointers_appE && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182c_labels_pointers_appE.py \\
      > data/SRP1/Results/P515S53/w182c_labels_pointers_appE/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
      --out data/SRP1/Results/P515S53/w182c_labels_pointers_appE/w182c_bool_typing_test.json \\
      > data/SRP1/Results/P515S53/w182c_labels_pointers_appE/w182c_bool_typing_test.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182c_labels_pointers_appE.py --post-run
Exit: 0 = written, every integrity check holds, guards at 0 (MISMATCHES and failed predictions are FINDINGS and never
change the exit code); 3 = written, an integrity check failed (listed); 1 / 4 = precondition / capture refusal.
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W182c labels, pointers, blue, Appendix E at edd7a63 '
                                               '(never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W182c: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W182c: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s53_w179b_checks as B  # noqa: E402  (W179b; its own permitted=() guard; NOT edited)
import p515_s53_w181a_identity_appendixE as A  # noqa: E402  (W181a; its own permitted=() guard; NOT edited, not run)
import gate_result_io as GRIO  # noqa: E402

C, R, P3, PB, W167 = B.C, B.R, B.P3, B.PB, B.W167
os.chdir(REPO)
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w182c_labels_pointers_appE')
OUT_NAME, MD_NAME, MANIFEST_NAME = 'w182c_labels_pointers_appE.json', 'w182c_labels_pointers_appE.md', 'manifest_sha256.json'
GEN_NAME, VERIFY_NAME = 'appE_generated_w182c.tex', 'verify_appE_output.txt'
POST_NAME = 'manifest_post_run_sha256.json'
TYPING_NAMES = ('w182c_bool_typing_test.json', 'w182c_bool_typing_test.log')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER_NAME = 'response_to_reviewers_draft.tex'
LETTER = os.path.join(CLONE, LETTER_NAME)
DTX = os.path.join(CLONE, 'elsarticle.dtx')
MR = 'manuscript_review'
LETTER_V8 = os.path.join(MR, 'response_to_reviewers_v8.tex')
GEN_REL = os.path.join(MR, 'gen_appE.py')
VERIFY_REL = os.path.join(MR, 'verify_appE.py')
APPX_REL = os.path.join(MR, 'appendixE_evaluation_record.tex')
W181A_GEN = os.path.join(S53, 'w181a_identity_appendixE', 'appE_generated_w181a.tex')
W180C_JSON = os.path.join(S53, 'w180c_text_letter', 'w180c_text_letter.json')
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
REVIEW = os.path.join(MR, 'manuscript_critical_review_2026-10-25.md')

DECLARED_HEAD = 'edd7a63'
DECLARED_HEAD_FULL = 'edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838'
DECLARED_MAIN_SHA = 'b08a3365607e56f2ff8c56572a9a74bc9ea97d5fb461faa252fab7955ffa8f5b'   # review: fingerprint b08a3365607e56f2
OLD = '8f051a2'                     # the expert's compile with every label number identical to round 7 (Addendum 81)
ROUND7 = 'cd57efc'                  # W181: main.tex = main_e5322f8_round7.tex, letter = v8
W180C_HEAD = 'e5322f8'              # W180c's pointer check (letter v7): 160 MATCH / 5 SUBMITTED / 0 MISMATCH
LETTER_LAST = 'd2ebb0d'             # last commit touching the letter (v9)
V9_SENTENCES = ['cycle-life calibration bracketed by a manufacturer datasheet',
                'one of them read from a manufacturer datasheet [44]',
                'stated as a modelling assumption in the range reported for LFP cells [43]']
CHECK_FAILURES = []
APPE_MARK = '% APPENDIX E -- EVALUATION RECORD'
HEADING_KINDS = ('section', 'subsection', 'subsubsection')
FLOAT_KINDS = ('table', 'figure', 'algorithm')
PROSE_PATTERNS = [   # A4 prose mentions of removed / moved objects (case-insensitive)
    ('flexibility ladder', r'flexibility(?:[- ]price)?[- ]ladder'),
    ('price ladder', r'price[- ]ladder'),
    ('any "ladder"', r'\bladders?\b'),
    ('shaded region (fig:search_generic)', r'shaded region'),
    ('illustration (fig:search_generic)', r'\billustrat\w*'),
]
KIND_WORDS = {'table': 'table', 'tables': 'table', 'fig.': 'figure', 'figs.': 'figure', 'figure': 'figure',
              'figures': 'figure', 'eq.': 'equation', 'eqs.': 'equation', 'equation': 'equation',
              'equations': 'equation', 'algorithm': 'algorithm', 'section': 'heading', 'sections': 'heading',
              'subsection': 'heading', 'subsections': 'heading', 'appendix': 'heading', 'appendices': 'heading'}


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


_sha, _read = C._sha, C._read


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, text=True, check=True).stdout


def _git_bytes(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, check=True).stdout


def show(commit, path='main.tex'):
    return _git(['show', f'{commit}:{path}'], cwd=CLONE)


def sha_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()


def ws(s):
    return re.sub(r'\s+', ' ', s).strip()


def typeset(lines):
    return W167.strip_comments(lines)


def refuse_concurrent():
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me = os.getpid()
    others = []
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(os.path.basename(a) == SCRIPT_NAME for a in argv[1:]) \
                and '--post-run' not in argv:
            others.append(ln.strip())
    if others:
        print(f'REFUSED: another copy of {SCRIPT_NAME} is running: {others}', flush=True)
        sys.exit(1)


def natkey(num):
    out = []
    for p in str(num).split('.'):
        m = re.fullmatch(r'(\d+)([a-z]?)', p)
        out.append((0, int(m.group(1)), m.group(2)) if m else (1, 0, p))
    return tuple(out)


def unit_diff_hunks(a_text, b_text, n):
    return sum(1 for ln in difflib.unified_diff(a_text.split('\n'), b_text.split('\n'), lineterm='', n=n)
               if ln.startswith('@@'))


# ======================================================================================================================
#  0. identity
# ======================================================================================================================
def identity():
    head = _git(['rev-parse', '--short', 'HEAD'], cwd=CLONE).strip()
    head_full = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    n_after_old = int(_git(['rev-list', '--count', f'{OLD}..HEAD'], cwd=CLONE).strip())
    n_after_letter = int(_git(['rev-list', '--count', f'{LETTER_LAST}..HEAD'], cwd=CLONE).strip())
    files_changed = _git(['diff', '--name-only', OLD, 'HEAD'], cwd=CLONE).split()
    letter_log = _git(['log', '--format=%h %cI', '--', LETTER_NAME], cwd=CLONE).split('\n')[:3]
    mh, mo, m7 = show('HEAD'), show(OLD), show(ROUND7)
    lh, ll, l7, lw = show('HEAD', LETTER_NAME), show(LETTER_LAST, LETTER_NAME), show(ROUND7, LETTER_NAME), show(W180C_HEAD, LETTER_NAME)
    v8 = _read(LETTER_V8)
    main_diff = _git(['diff', '--no-color', '--no-ext-diff', OLD, 'HEAD', '--', 'main.tex'], cwd=CLONE)
    letter_diff_u3 = _git(['diff', '--no-color', '--no-ext-diff', ROUND7, 'HEAD', '--', LETTER_NAME], cwd=CLONE)
    letter_diff_u0 = _git(['diff', '--no-color', '--no-ext-diff', '-U0', ROUND7, 'HEAD', '--', LETTER_NAME], cwd=CLONE)
    d = {'clone_head': head, 'clone_head_full': head_full,
         'clone_head_date': _git(['log', '-1', '--format=%cI'], cwd=CLONE).strip(),
         'clone_tracked_modified': status, 'commits_after_8f051a2': n_after_old, 'commits_after_d2ebb0d': n_after_letter,
         'files_changed_8f051a2_to_head': files_changed, 'letter_log_top3': letter_log,
         'main_sha256_working_copy': _sha(MAIN_TEX), 'main_lines': mh.count('\n'),
         'main_sha256_head_blob': sha_text(mh), 'main_sha256_8f051a2': sha_text(mo), 'main_lines_8f051a2': mo.count('\n'),
         'letter_sha256_working_copy': _sha(LETTER), 'letter_sha256_head_blob': sha_text(lh),
         'letter_lines': lh.count('\n'), 'letter_v8_sha256': _sha(LETTER_V8),
         'main_diff_8f051a2_head_hunks_U3': sum(1 for ln in main_diff.split('\n') if ln.startswith('@@')),
         'letter_diff_cd57efc_head_hunks_U3': sum(1 for ln in letter_diff_u3.split('\n') if ln.startswith('@@')),
         'letter_diff_cd57efc_head_hunks_U0': sum(1 for ln in letter_diff_u0.split('\n') if ln.startswith('@@')),
         'letter_header_line': lh.split('\n')[0][:200],
         'v9_sentences_present': {s: (s in ws(lh)) for s in V9_SENTENCES},
         'repo_head': _git(['rev-parse', 'HEAD']).strip(),
         'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
         'last_production_code_commit_note': 'not measured here (W182 reports the code head from the repository)'}
    checks = {'head': head == DECLARED_HEAD, 'head_full': head_full == DECLARED_HEAD_FULL, 'clone_clean': status == '',
              'main_sha_declared': d['main_sha256_working_copy'] == DECLARED_MAIN_SHA,
              'main_head_blob_equals_working_copy': d['main_sha256_head_blob'] == d['main_sha256_working_copy'],
              'letter_head_blob_equals_working_copy': d['letter_sha256_head_blob'] == d['letter_sha256_working_copy'],
              'letter_head_equals_d2ebb0d_blob': lh == ll,
              'letter_cd57efc_equals_v8': l7 == v8,
              'letter_head_differs_from_v8': lh != v8,
              'letter_header_says_v9': lh.split('\n')[0].startswith('% Response to reviewers — v9'),
              'v9_sentences_all_present': all(d['v9_sentences_present'].values()),
              'letter_line_count_unchanged_since_e5322f8': lh.count('\n') == lw.count('\n'),
              'main_8f051a2_differs_from_cd57efc': mo != m7}
    d['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'identity check {k} failed')
    return d, {'main_head': mh, 'main_old': mo, 'main_r7': m7, 'main_w180c': show(W180C_HEAD), 'letter_head': lh,
               'letter_w180c': lw}


# ======================================================================================================================
#  1. capture-path checklist
# ======================================================================================================================
REQUIRED_CAPTURES = {
    'identity (HEAD, hashes, commit counts, letter v9 identity)': 'identity',
    'A3 label map (every label: number at each revision, status)': 'task_label_map',
    'A3 heading map old -> new (label / title / content)': 'heading_map',
    'A3 letter pointers (verdict, number at edd7a63, what the printed number denotes)': 'task_pointers',
    'A4 undefined refs, kind-word mismatches, retitled labels, literal pointers, prose mentions': 'task_dangling',
    'A5 dropped blue (unwrapped / rewritten not blue)': 'task_blue',
    'A6 Appendix E bitwise + verify_appE': 'task_appendix_e',
    'negative controls': 'task_negative_controls',
    'guards (solve 0, pickle 0, no production import)': 'verify_guards',
}
REQUIRED_SYMBOLS = {
    'C': (C, ('numbering_adjusted', 'task_refs', 'GUARD', 'PICKLE_COUNTS')),
    'R': (R, ('letter_pointers', 'cite_order', '_region_text', 'PTR', 'PTR_APP', 'GUARD', 'PICKLE_COUNTS')),
    'B': (B, ('region_v6', 'resolve_v6', 'task_elsarticle', 'GUARD', 'PICKLE_COUNTS', 'FORBIDDEN_MODULE_PREFIXES')),
    'A': (A, ('run_generator', 'run_verifier', 'locate_bytes', 'appendix_block', 'raw_tables', 'diff_tables',
              'block_diff', 'GUARD', 'PICKLE_COUNTS')),
}


def capture_checklist(out_dir):
    missing = []
    g = globals()
    for q, fn in REQUIRED_CAPTURES.items():
        if not callable(g.get(fn)):
            missing.append(f'capture function {fn} for "{q}"')
    for name, (mod, syms) in REQUIRED_SYMBOLS.items():
        for s in syms:
            if not hasattr(mod, s):
                missing.append(f'symbol {name}.{s}')
    for p in (MAIN_TEX, LETTER, DTX, LETTER_V8, GEN_REL, VERIFY_REL, APPX_REL, W180C_JSON, BRIEF, REVIEW,
              os.path.join(MR, 'export_W160', 'T1.csv')):
        if not os.path.isfile(p):
            missing.append(f'input {p}')
    gen_out = os.path.realpath(os.path.join(out_dir, GEN_NAME))
    if gen_out in (os.path.realpath(APPX_REL), os.path.realpath(os.path.join(MR, 'appE.tex'))):
        missing.append('generator target is a manuscript_review file')
    if not gen_out.startswith(os.path.realpath(out_dir) + os.sep):
        missing.append('generator target is not inside the output directory')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()},
            'all_present': True}


# ======================================================================================================================
#  2. A3 label map and heading map
# ======================================================================================================================
def _num(lines):
    before = list(C.CHECK_FAILURES)
    raw, adj = C.numbering_adjusted(lines)
    if C.CHECK_FAILURES != before:
        _fail(f'W178c numbering_adjusted reported: {C.CHECK_FAILURES[len(before):]}')
    return raw, adj


def objects_index(adj):
    """(kind, number) -> object, for headings, captioned floats, labelled longtables and equations."""
    labels_by = {}
    for k, v in adj['labels'].items():
        labels_by.setdefault((v['kind'], v['number']), []).append(k)
    idx = {}
    for o in adj['objects']:
        if o['number'] is None:
            continue
        kind = o['kind']
        if kind == 'table(longtable)':
            kind = 'table'
            if not [k for k in labels_by.get(('table', o['number']), []) if adj['labels'][k]['line'] == o['line']]:
                continue          # caption-less nomenclature longtables (their step is cancelled by \addtocounter)
        if kind.startswith('equation'):
            kind = 'equation'
        if kind == 'subequations':
            continue
        key = (kind, o['number'])
        labs = labels_by.get(key, [])
        if kind in FLOAT_KINDS:   # a float's labels are recorded at the float's \begin line
            labs = [k for k in labs if adj['labels'][k]['line'] == o['line']]
        rec = {'kind': kind, 'number': o['number'], 'line': o['line'],
               'desc': o.get('title') if kind in HEADING_KINDS else o.get('caption'),
               'labels': sorted(labs), 'appendix': bool(re.match(r'[A-Z]', str(o['number'])))}
        if key in idx:
            idx[key]['dup'] = idx[key].get('dup', 0) + 1
        else:
            idx[key] = rec
    return idx


def _clean_desc(s):
    s = re.sub(r'\\textcolor\{[^}]*\}\{', '', s or '')
    return ws(s.replace('}', ' ').replace('{', ' '))[:110]


def label_desc(adj, idx, label):
    v = adj['labels'].get(label)
    if v is None:
        return None
    o = idx.get(('equation' if v['kind'] == 'equation' else v['kind'], v['number']))
    return _clean_desc(o['desc']) if o and o.get('desc') else None


def label_class(label, v):
    k = v['kind']
    if k in HEADING_KINDS:
        return 'appendix' if re.match(r'[A-Z]', str(v['number'])) else 'section'
    return k or 'unknown'


def heading_own_text(cl, line):
    """Typeset text of a heading's own region: from its line to the next heading of any level (comment-stripped,
    blue wrappers removed, whitespace collapsed)."""
    i0 = line - 1
    e = next((k for k in range(i0 + 1, len(cl)) if re.match(r'\s*\\(section|subsection|subsubsection)\*?\{', cl[k])
              or re.match(r'\s*\\appendix\b', cl[k])), len(cl))
    return ws(strip_blue('\n'.join(cl[i0:e])))


def strip_blue(text):
    clean, _ = blue_mask_text(text)
    return clean


def heading_at(adj, line):
    """(kind, number) of the nearest numbered heading at or above a line (None in front matter)."""
    hs = [o for o in adj['objects'] if o['kind'] in HEADING_KINDS and o['number'] is not None and o['line'] <= line]
    return (hs[-1]['kind'], hs[-1]['number']) if hs else None


def shingles(text, k=6):
    w = text.split()
    return {' '.join(w[i:i + k]) for i in range(len(w) - k + 1)}


def heading_map(adj_o, adj_n, cl_o, cl_n):
    """Numbered headings old -> new. Successor rules, in order: (1) a label of the heading still defined; (2) the same
    title at the same level in the same part (body / appendices), else at another level in the same part; (3) at least
    two labels defined inside the old heading's own text, at least two thirds of them inside one new heading's own text.
    Otherwise 'not located'. 6-word-shingle containment of the old own text in each new heading's own text is recorded
    as a hint only (rewritten text defeats it; a merged or split section makes it point two ways)."""
    io, inn = objects_index(adj_o), objects_index(adj_n)
    new_heads = [o for o in inn.values() if o['kind'] in HEADING_KINDS]
    new_sh = {(o['kind'], o['number']): shingles(heading_own_text(cl_n, o['line'])) for o in new_heads}
    old_heads = sorted((o for o in io.values() if o['kind'] in HEADING_KINDS), key=lambda o: o['line'])
    out = {}
    for idx_h, o in enumerate(old_heads):
        key = (o['kind'], o['number'])
        rec = {'kind': o['kind'], 'number_old': o['number'], 'title_old': o['desc'], 'labels': o['labels'],
               'line_old': o['line']}
        succ, method = None, None
        lab = [lb for lb in o['labels'] if lb in adj_n['labels']]
        if lab:
            v = adj_n['labels'][lab[0]]
            succ, method = (v['kind'], v['number']), f'label {lab[0]}'
        else:
            same = [k for k, n in inn.items() if n['kind'] == o['kind'] and n['appendix'] == o['appendix'] and n['desc'] == o['desc']]
            if not same:
                same = [k for k, n in inn.items() if n['kind'] in HEADING_KINDS and n['appendix'] == o['appendix']
                        and n['desc'] == o['desc']]
            if len(same) == 1:
                succ, method = same[0], 'exact title'
            else:
                end = old_heads[idx_h + 1]['line'] if idx_h + 1 < len(old_heads) else 10 ** 9
                inside = [k for k, v in adj_o['labels'].items() if o['line'] <= v['line'] < end and k not in o['labels']]
                where = Counter(heading_at(adj_n, adj_n['labels'][k]['line']) for k in inside if k in adj_n['labels'])
                rec['contained_labels'] = {'n': len(inside), 'where_now': {f'{k[0]} {k[1]}' if k else 'front': v
                                                                           for k, v in where.items()}}
                if len(inside) >= 2 and where:
                    top, cnt = where.most_common(1)[0]
                    if top and cnt * 3 >= 2 * len(inside):
                        succ, method = top, f'contained labels ({cnt}/{len(inside)})'
                sh = shingles(heading_own_text(cl_o, o['line']))
                if sh:
                    sc = sorted(((len(sh & v) / len(sh), k) for k, v in new_sh.items()), key=lambda x: -x[0])[:3]
                    rec['shingle_hint_top3'] = [(f'{k[0]} {k[1]}', round(a, 3)) for a, k in sc]
        if succ is not None:
            n = inn.get(succ)
            rec.update({'kind_new': succ[0], 'number_new': succ[1], 'title_new': n['desc'] if n else None,
                        'line_new': n['line'] if n else None})
        else:
            rec.update({'kind_new': None, 'number_new': None, 'title_new': None, 'line_new': None})
        rec['method'] = method or 'not located'
        rec['heading_line_commented_at_edd7a63'] = None
        rec['status'] = ('removed / not located' if succ is None else
                         'unchanged' if succ[1] == o['number'] and rec['title_new'] == o['desc'] else
                         'retitled' if succ[1] == o['number'] else 'renumbered')
        out[key] = rec
    return out


def eq_bodies(adj, cl):
    """Unlabelled numbered equations: number -> (whitespace-collapsed environment text, row index)."""
    labelled = {v['number'] for v in adj['labels'].values() if v['kind'] == 'equation'}
    out, seen = {}, Counter()
    for o in adj['objects']:
        if not o['kind'].startswith('equation') or o['number'] is None:
            continue
        seen[o['line']] += 1
        if o['number'] in labelled:
            continue
        i0 = o['line'] - 1
        e = next((k for k in range(i0, len(cl)) if re.search(r'\\end\{(equation|align)\*?\}', cl[k])), i0)
        out[o['number']] = (ws('\n'.join(cl[i0:e + 1])), seen[o['line']])
    return out


def task_label_map(texts):
    lo, ln_ = texts['main_old'].split('\n'), texts['main_head'].split('\n')
    raw_o, adj_o = _num(lo)
    raw_n, adj_n = _num(ln_)
    _, adj_7 = _num(texts['main_r7'].split('\n'))
    _, adj_w = _num(texts['main_w180c'].split('\n'))
    io, inn = objects_index(adj_o), objects_index(adj_n)
    rows = []
    for lab in sorted(set(adj_o['labels']) | set(adj_n['labels']), key=lambda k: (k not in adj_o['labels'], k)):
        vo, vn = adj_o['labels'].get(lab), adj_n['labels'].get(lab)
        st = ('added' if vo is None else 'removed' if vn is None else
              'unchanged' if vo['number'] == vn['number'] else 'renumbered')
        rows.append({'label': lab, 'class': label_class(lab, vo or vn), 'kind': (vo or vn)['kind'],
                     'number_8f051a2': vo['number'] if vo else None, 'number_edd7a63': vn['number'] if vn else None,
                     'line_8f051a2': vo['line'] if vo else None, 'line_edd7a63': vn['line'] if vn else None,
                     'status': st, 'desc_8f051a2': label_desc(adj_o, io, lab) if vo else None,
                     'desc_edd7a63': label_desc(adj_n, inn, lab) if vn else None})
    cl_o, cl_n = typeset(lo), typeset(ln_)
    hm = heading_map(adj_o, adj_n, cl_o, cl_n)
    for h in hm.values():
        if h['number_new'] is None:
            rx = re.compile(r'^\s*%+\s*\\' + h['kind'] + r'\*?\{.*' + re.escape(h['title_old']))
            h['heading_line_commented_at_edd7a63'] = [i for i, x in enumerate(ln_, 1) if rx.search(x)]
            h['title_typeset_anywhere_at_edd7a63'] = any(h['title_old'] in x for x in cl_n)
    eo, en = eq_bodies(adj_o, cl_o), eq_bodies(adj_n, cl_n)
    inv_n = {}
    for k, v in en.items():
        inv_n.setdefault(v, []).append(k)
    eq_unlabelled = [{'number_8f051a2': k, 'number_edd7a63': (inv_n.get(v) or [None])[0] if len(inv_n.get(v, [])) <= 1 else None,
                      'body_start': v[0][:90]} for k, v in eo.items()]
    eq_unlabelled_new = [{'number_edd7a63': k, 'in_8f051a2': any(v == w for w in eo.values()), 'body_start': v[0][:90]}
                         for k, v in en.items()]
    unlabelled_floats = {rev: [{'kind': o['kind'], 'number': o['number'], 'line': o['line'], 'caption': _clean_desc(o['desc'])}
                               for o in ix.values() if o['kind'] in FLOAT_KINDS and not o['labels']]
                         for rev, ix in ((OLD, io), (DECLARED_HEAD, inn))}
    same_7 = {k: adj_7['labels'].get(k, {}).get('number') == v['number'] for k, v in adj_o['labels'].items()}
    same_w = {k: adj_w['labels'].get(k, {}).get('number') == v['number'] for k, v in adj_o['labels'].items()}
    counts = Counter(r['status'] for r in rows)
    by_class = {}
    for r in rows:
        by_class.setdefault(r['class'], Counter())[r['status']] += 1
    out = {'rows': rows, 'counts': dict(counts), 'by_class': {k: dict(v) for k, v in by_class.items()},
           'n_labels_8f051a2': len(adj_o['labels']), 'n_labels_edd7a63': len(adj_n['labels']),
           'removed': [r for r in rows if r['status'] == 'removed'], 'added': [r for r in rows if r['status'] == 'added'],
           'heading_map': list(hm.values()),
           'heading_counts': dict(Counter(h['status'] for h in hm.values())),
           'equations_unlabelled_8f051a2': eq_unlabelled, 'equations_unlabelled_edd7a63': eq_unlabelled_new,
           'unlabelled_floats': unlabelled_floats,
           'object_counts': {OLD: adj_o['counts'], DECLARED_HEAD: adj_n['counts']},
           'counter_adjustments': {OLD: adj_o['counter_adjustments'], DECLARED_HEAD: adj_n['counter_adjustments']},
           'numbering_warnings': {OLD: adj_o['warnings'], DECLARED_HEAD: adj_n['warnings']},
           'cross_check_8f051a2_equals_cd57efc_all_labels': all(same_7.values()) and set(adj_7['labels']) == set(adj_o['labels']),
           'cross_check_8f051a2_equals_e5322f8_all_labels': all(same_w.values()) and set(adj_w['labels']) == set(adj_o['labels']),
           'elsarticle': B.task_elsarticle(),
           'static_numbering_note': 'static numbering (no TeX on this machine); nothing compiled'}
    if not out['cross_check_8f051a2_equals_cd57efc_all_labels'] or not out['cross_check_8f051a2_equals_e5322f8_all_labels']:
        _fail('8f051a2 numbering differs from cd57efc / e5322f8 (Addendum 81 says identical to round 7)')
    if adj_o['warnings'] or adj_n['warnings']:
        _log(f'numbering warnings: {out["numbering_warnings"]}')
    ctx = {'adj_o': adj_o, 'adj_n': adj_n, 'raw_o': raw_o, 'raw_n': raw_n, 'io': io, 'inn': inn, 'hm': hm,
           'eo': eo, 'en': en, 'cl_o': cl_o, 'cl_n': cl_n, 'lo': lo, 'ln': ln_,
           'cites_o': R.cite_order(lo), 'cites_n': R.cite_order(ln_)}
    out['citations'] = {'n_keys_8f051a2': len(ctx['cites_o']), 'n_keys_edd7a63': len(ctx['cites_n']),
                        'order_identical': ctx['cites_o'] == ctx['cites_n'],
                        'renumbered': [{'key': k, 'number_8f051a2': i + 1,
                                        'number_edd7a63': ctx['cites_n'].index(k) + 1 if k in ctx['cites_n'] else None}
                                       for i, k in enumerate(ctx['cites_o'])
                                       if k not in ctx['cites_n'] or ctx['cites_n'].index(k) != i],
                        'new_keys': [k for k in ctx['cites_n'] if k not in ctx['cites_o']]}
    return out, ctx


# ======================================================================================================================
#  3. A3 pointers
# ======================================================================================================================
def old_objects_of(atom_kind, ctx):
    """objects of the old document a pointer of this kind can name: {number: identity}."""
    io = ctx['io']
    if atom_kind in ('section', 'new'):
        return {o['number']: ('heading', k) for k, o in io.items() if o['kind'] in HEADING_KINDS and not o['appendix']}
    if atom_kind == 'appendix':
        return {o['number']: ('heading', k) for k, o in io.items() if o['kind'] in HEADING_KINDS and o['appendix']}
    if atom_kind in FLOAT_KINDS or atom_kind == 'equation':
        return {o['number']: (atom_kind, k) for k, o in io.items() if o['kind'] == atom_kind}
    return {}


def resolve_identity(ident, ctx, old=True):
    """identity -> {'key': canonical key, 'number_new', 'how', 'desc_old', 'desc_new'}."""
    t, k = ident
    adj_n, inn = ctx['adj_n'], ctx['inn']
    if t == 'cite':
        n = ctx['cites_n'].index(k) + 1 if k in ctx['cites_n'] else None
        return {'key': f'cite:{k}', 'number_new': None if n is None else str(n), 'how': 'citation order', 'desc_old': k,
                'desc_new': k if n else None}
    o = ctx['io'][k]
    if t == 'heading':
        h = ctx['hm'][k]
        key = o['labels'][0] if o['labels'] else f'heading:{o["kind"]}:{o["desc"]}'
        return {'key': key, 'number_new': h['number_new'], 'how': h['method'], 'desc_old': o['desc'],
                'desc_new': h['title_new'], 'kind_new': h['kind_new']}
    if o['labels']:
        lab = o['labels'][0]
        v = adj_n['labels'].get(lab)
        n = v['number'] if v else None
        on = inn.get((t, n)) if n else None
        return {'key': lab, 'number_new': n, 'how': f'label {lab}', 'desc_old': _clean_desc(o['desc']),
                'desc_new': _clean_desc(on['desc']) if on else None}
    if t == 'equation':
        body = ctx['eo'].get(k[1])
        hits = [n for n, b in ctx['en'].items() if b == body]
        return {'key': f'eqbody:{k[1]}', 'number_new': hits[0] if len(hits) == 1 else None,
                'how': 'unlabelled equation, body match' if len(hits) == 1 else 'unlabelled equation, body not unique/absent',
                'desc_old': (body or ('',))[0][:80], 'desc_new': None}
    hits = [n for n in inn.values() if n['kind'] == t and n['desc'] == o['desc']]
    return {'key': f'{t}:caption:{_clean_desc(o["desc"])[:40]}', 'number_new': hits[0]['number'] if len(hits) == 1 else None,
            'how': 'unlabelled float, caption match', 'desc_old': _clean_desc(o['desc']),
            'desc_new': _clean_desc(hits[0]['desc']) if len(hits) == 1 else None}


def now_denotes(atom_kind, number, ctx):
    """What a printed number names at edd7a63 (same pointer kind)."""
    inn = ctx['inn']
    if atom_kind == 'reference':
        i = int(number) - 1
        return f'[{number}] = {ctx["cites_n"][i]}' if 0 <= i < len(ctx['cites_n']) else 'no such citation'
    kinds = (HEADING_KINDS if atom_kind in ('section', 'new', 'appendix') else (atom_kind,))
    hits = [o for o in inn.values() if o['kind'] in kinds and o['number'] == number]
    if not hits:
        return 'nothing'
    return _obj_str(hits[0])


def _obj_str(o):
    parts = [o['kind'], o['number']] + ([o['labels'][0]] if o['labels'] else []) + \
            ([f'"{_clean_desc(o["desc"])[:70]}"'] if o.get('desc') else [])
    return ' '.join(parts)


def range_members(atom, ctx):
    a, b = atom['numbers']
    if atom['kind'] == 'reference':
        return [('cite', ctx['cites_o'][i - 1]) for i in range(int(a), int(b) + 1) if 0 < i <= len(ctx['cites_o'])], None
    objs = old_objects_of(atom['kind'], ctx)
    pa, pb = a.rsplit('.', 1) if '.' in a else ('', a), b.rsplit('.', 1) if '.' in b else ('', b)
    if pa[0] != pb[0] or not pa[1].isdigit() or not pb[1].isdigit():
        return None, f'range endpoints not in one series: {a}, {b}'
    out = []
    for i in range(int(pa[1]), int(pb[1]) + 1):
        num = f'{pa[0]}.{i}' if pa[0] else str(i)
        if num not in objs:
            return None, f'range member {num} has no object at 8f051a2'
        out.append(objs[num])
    return out, None


def series_between(atom_kind, lo, hi, ctx):
    inn = ctx['inn']
    if atom_kind == 'reference':
        return [str(i) for i in range(int(lo), int(hi) + 1)]
    kinds = HEADING_KINDS if atom_kind in ('section', 'new', 'appendix') else (atom_kind,)
    pre = lo.rsplit('.', 1)[0] if '.' in lo else ''
    nums = [o['number'] for o in inn.values() if o['kind'] in kinds
            and ((o['number'].rsplit('.', 1)[0] if '.' in o['number'] else '') == pre)
            and natkey(lo) <= natkey(o['number']) <= natkey(hi)]
    return sorted(set(nums), key=natkey)


def canonical_w180c_target(t, ctx):
    """W180c's declared target (label / title: / titlek: / cite: / list) -> canonical key at 8f051a2."""
    if isinstance(t, list):
        return [canonical_w180c_target(x, ctx) for x in t]
    if t is None:
        return None
    if t.startswith('cite:'):
        return t
    if t.startswith('title:') or t.startswith('titlek:'):
        num = B.resolve_v6(t, ctx['adj_o'], ctx['cites_o'])
        if t.startswith('titlek:'):
            kind = t.split(':', 2)[1]
        else:
            h = R.heading_key(ctx['adj_o'], t[6:])
            kind = h['kind'] if h else None
        o = ctx['io'].get((kind, num))
        if o is None:
            return f'UNRESOLVED:{t}'
        return o['labels'][0] if o['labels'] else f'heading:{o["kind"]}:{o["desc"]}'
    return t


def letter_line_map(old_text, new_text):
    a, b = old_text.split('\n'), new_text.split('\n')
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    mp = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                mp[j1 + k + 1] = i1 + k + 1
    return mp


def pointer_rows(letter_path, ctx, w180c_rows, letter_w180c_text, letter_text):
    atoms = R.letter_pointers(letter_path)
    mp = letter_line_map(letter_w180c_text, letter_text)        # v9 line -> v7 line (equal lines only)
    prior = {}
    for r_ in w180c_rows:
        prior.setdefault((r_['line'], r_['kind'], tuple(r_['numbers']), r_['text']), r_)
    rows = []
    for a in atoms:
        row = {k: a[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range')}
        v7 = mp.get(a['line'])
        pr = prior.get((v7, a['kind'], tuple(a['numbers']), a['text'])) if v7 else None
        row['line_v7'] = v7
        row['w180c_verdict'] = pr['verdict'] if pr else None
        row['w180c_target'] = pr['target'] if pr else None
        printed = a['numbers']
        if pr and pr['verdict'] == 'SUBMITTED':
            row['verdict'], row['note'] = 'SUBMITTED', pr.get('note')
            rows.append(row)
            continue
        if a['kind'] == 'former':
            row['verdict'], row['note'] = 'AMBIGUOUS', 'a "former" pointer not declared SUBMITTED by W180c'
            rows.append(row)
            continue
        # intended target at 8f051a2
        if a['range']:
            members, why = range_members(a, ctx)
            if members is None:
                row['verdict'], row['note'] = 'AMBIGUOUS', why
                rows.append(row)
                continue
        else:
            if a['kind'] == 'reference':
                i = int(printed[0]) - 1
                members = [('cite', ctx['cites_o'][i])] if 0 <= i < len(ctx['cites_o']) else None
            else:
                objs = old_objects_of(a['kind'], ctx)
                members = [objs[printed[0]]] if printed[0] in objs else None
            if members is None:
                row['verdict'], row['note'] = 'AMBIGUOUS', f'no {a["kind"]} {printed[0]} at 8f051a2'
                rows.append(row)
                continue
        res = [resolve_identity(m, ctx) for m in members]
        row['target_8f051a2'] = [r_['key'] for r_ in res] if a['range'] else res[0]['key']
        row['target_desc_8f051a2'] = res[0]['desc_old'] if not a['range'] else [r_['desc_old'] for r_ in res]
        row['resolution'] = [r_['how'] for r_ in res]
        new_nums = [r_['number_new'] for r_ in res]
        row['numbers_edd7a63'] = new_nums
        if pr:   # cross-check against W180c's declared target
            mine = row['target_8f051a2']
            theirs = canonical_w180c_target(pr['target'], ctx)
            if isinstance(theirs, list) and len(theirs) == 1 and isinstance(mine, str):
                theirs = theirs[0]
            row['w180c_target_canonical'] = theirs
            row['agrees_with_w180c_target'] = (mine == theirs)
        if a['range']:
            alive = [n for n in new_nums if n is not None]
            removed = [res[i]['key'] for i, n in enumerate(new_nums) if n is None]
            row['members_removed'] = removed
            if not alive:
                row['verdict'], row['subclass'] = 'MISMATCH', 'all range members removed'
                rows.append(row)
                continue
            alive_sorted = sorted(alive, key=natkey)
            got = [alive_sorted[0], alive_sorted[-1]]
            span = series_between(a['kind'], got[0], got[1], ctx)
            row['inserted_in_new_span'] = [n for n in span if n not in alive]
            row['should_be'] = '--'.join(got)
            if removed:
                row['verdict'], row['subclass'] = 'MISMATCH', 'range member removed'
            elif got != printed:
                row['verdict'], row['subclass'] = 'MISMATCH', 'renumbered'
            else:
                row['verdict'] = 'MATCH'
            if row['verdict'] == 'MATCH' and row['inserted_in_new_span']:
                row['note'] = 'endpoints unchanged but the span now includes inserted objects'
        else:
            n = new_nums[0]
            if n is None and a['kind'] in ('section', 'new', 'appendix') and 'not located' in res[0]['how']:
                # a removed unlabelled heading: decided by W180c's content needles for this pointer, if any
                needles = list((pr or {}).get('content_needles') or {})
                where = {nd: sorted({heading_at(ctx['adj_n'], i) for i, ln in enumerate(ctx['cl_n'], 1) if nd in ln}
                                    - {None}) for nd in needles}
                row['needle_headings_edd7a63'] = {nd: [f'{k[0]} {k[1]}' for k in v] for nd, v in where.items()}
                found = [v for v in where.values() if v]
                cand = {k for v in found for k in v}
                if needles and len(found) == len(needles) and len(cand) == 1:
                    k = cand.pop()
                    row['via_needle'] = f'{k[0]} {k[1]}'
                    if k[1] == printed[0]:
                        row['verdict'], row['subclass'] = 'MATCH', 'heading removed; needle text now under the printed number'
                    else:
                        row['verdict'], row['subclass'], row['should_be'] = ('MISMATCH', 'heading removed; needle text '
                                                                             'now under another number', k[1])
                elif needles and not found:
                    row['verdict'], row['subclass'] = 'MISMATCH', 'target removed (heading and needle text no longer typeset)'
                else:
                    row['verdict'], row['subclass'] = 'AMBIGUOUS', 'heading removed; content not located'
            elif n is None:
                row['verdict'], row['subclass'] = 'MISMATCH', 'target removed'
            elif n == printed[0]:
                row['verdict'] = 'MATCH'
                if res[0].get('kind_new') and str(res[0]['how']).startswith('content'):
                    row['note'] = f'heading removed or retitled; its content is now under {res[0]["kind_new"]} {n} ({res[0]["how"]})'
            else:
                row['verdict'], row['subclass'], row['should_be'] = 'MISMATCH', 'renumbered', n
                if str(res[0]['how']).startswith('content'):
                    row['note'] = f'heading removed or retitled; its content is now under {res[0]["kind_new"]} {n} ({res[0]["how"]})'
        row['printed_now_denotes'] = [now_denotes(a['kind'], p, ctx) for p in printed]
        # W180c content needles, re-read at edd7a63 (informational)
        if pr and pr.get('content_needles'):
            needles = list(pr['content_needles'].keys())
            row['needles'] = needle_check(needles, row, res, ctx)
        rows.append(row)
    return rows, atoms, mp


def needle_check(needles, row, res, ctx):
    cl = ctx['cl_n']
    full = '\n'.join(cl)
    out = {}
    for nd in needles:
        if row['kind'] == 'reference' or (res and res[0]['key'].startswith('cite:')):
            out[nd] = {'in_target_region': None, 'in_document': nd in full}
            continue
        regs = []
        for r_ in res:
            if r_['number_new'] is None:
                continue
            key = r_['key']
            if key.startswith('heading:'):
                _, kind, title = key.split(':', 2)
                h = next((h for h in ctx['hm'].values() if h['title_old'] == title and h['kind'] == kind), None)
                if h and h['number_new'] is not None:
                    o = ctx['inn'].get((h['kind_new'], h['number_new']))
                    if o:
                        regs.append(B.region_v6(cl, f'titlek:{o["kind"]}:^{re.escape(o["desc"])}$'))
            elif key.startswith(('eqbody:', 'table:', 'figure:', 'algorithm:')):
                continue
            else:
                regs.append(R._region_text(cl, key))
        reg = '\n'.join(regs)
        where = [i for i, ln in enumerate(cl, 1) if nd in ln]
        out[nd] = {'in_target_region': (nd in reg) if regs else None, 'in_document': bool(where), 'lines': where[:5]}
    return out


def task_pointers(ctx, texts):
    w180c = json.load(open(W180C_JSON, encoding='utf-8'))
    rows, atoms, mp = pointer_rows(LETTER, ctx, w180c['pointers']['pointers'], texts['letter_w180c'], texts['letter_head'])
    counts = Counter(r_['verdict'] for r_ in rows)
    by_kind = {}
    for r_ in rows:
        by_kind.setdefault(r_['kind'], Counter())[r_['verdict']] += 1
    sub = Counter(r_.get('subclass') for r_ in rows if r_['verdict'] == 'MISMATCH')
    xc = [r_ for r_ in rows if 'agrees_with_w180c_target' in r_]
    disagree = [{k: r_.get(k) for k in ('line', 'text', 'numbers', 'target_8f051a2', 'w180c_target', 'w180c_target_canonical')}
                for r_ in xc if not r_['agrees_with_w180c_target']]
    if disagree:
        _fail(f'derived targets disagree with W180c declared targets: {disagree}')
    n_sub = counts.get('SUBMITTED', 0)
    if n_sub != 5:
        _fail(f'{n_sub} SUBMITTED pointers carried from W180c (expected its 5)')
    new_lines = sorted({r_['line'] for r_ in rows if r_['line_v7'] is None})
    not_in_w180c = [{k: r_.get(k) for k in ('line', 'text', 'numbers', 'verdict')} for r_ in rows if r_['w180c_verdict'] is None]
    needles_absent = [{k: r_.get(k) for k in ('line', 'text', 'verdict', 'needles')} for r_ in rows
                      if r_.get('needles') and not all(v['in_document'] for v in r_['needles'].values())]
    needles_moved = [{k: r_.get(k) for k in ('line', 'text', 'verdict', 'needles')} for r_ in rows
                     if r_.get('needles') and any(v['in_target_region'] is False and v['in_document'] for v in r_['needles'].values())]
    distinct_mismatch = sorted({(r_['line'], r_['text'], '--'.join(r_['numbers'])) for r_ in rows if r_['verdict'] == 'MISMATCH'})
    return {'n_pointers': len(rows), 'counts': dict(counts), 'by_kind': {k: dict(v) for k, v in by_kind.items()},
            'mismatch_subclasses': dict(sub), 'pointers': rows,
            'mismatch': [r_ for r_ in rows if r_['verdict'] == 'MISMATCH'],
            'ambiguous': [r_ for r_ in rows if r_['verdict'] == 'AMBIGUOUS'],
            'n_mismatch_distinct_line_text_number': len(distinct_mismatch),
            'w180c_cross_check': {'n_compared': len(xc), 'n_agree': len(xc) - len(disagree), 'disagree': disagree},
            'letter_lines_not_mapped_to_v7': new_lines, 'pointers_not_in_w180c': not_in_w180c,
            'needles_absent_from_document': needles_absent, 'needles_outside_target_region': needles_moved,
            'line_map_v9_to_v7_equal_lines': len(mp),
            'n_atoms_extracted': len(atoms)}


# ======================================================================================================================
#  4. A4: dangling references, kind words, retitled labels, literal pointers, prose mentions
# ======================================================================================================================
REF_RX = re.compile(r'\\(eqref|ref|autoref|cref|Cref|pageref|nameref)\{([^}]*)\}')
WORD_REF_RX = re.compile(r'([A-Za-z]+\.?)(?:~|\s+)(?:\(?)\\(?:eq)?ref\{([^}]*)\}')


def task_dangling(ctx, label_out, letter_text):
    adj_n, cl_n, cl_o = ctx['adj_n'], ctx['cl_n'], ctx['cl_o']
    refs_n = C.task_refs(ctx['ln'], adj_n)
    refs_o = C.task_refs(ctx['lo'], ctx['adj_o'])
    uses = []
    for i, ln in enumerate(cl_n, 1):
        for m in REF_RX.finditer(ln):
            for k in m.group(2).split(','):
                k = k.strip()
                if k not in adj_n['labels']:
                    uses.append({'line': i, 'cmd': m.group(1), 'label': k, 'in_8f051a2_as': (ctx['adj_o']['labels'].get(k) or {}).get('number'),
                                 'context': ws(ln[max(0, m.start() - 90):m.end() + 40])})
    commented = []
    for i, ln in enumerate(ctx['ln'], 1):
        if ln.lstrip().startswith('%') and REF_RX.search(ln):
            for m in REF_RX.finditer(ln):
                if m.group(2) in ('tab:res_flex_ladder', 'fig:search_generic'):
                    commented.append({'line': i, 'label': m.group(2), 'context': ws(ln)[:160]})
    # kind word before \ref vs the label's kind at edd7a63
    kind_mismatch = []
    for i, ln in enumerate(cl_n, 1):
        for m in WORD_REF_RX.finditer(ln):
            w, lab = m.group(1).lower(), m.group(2).split(',')[0].strip()
            want = KIND_WORDS.get(w)
            v = adj_n['labels'].get(lab)
            if not want or v is None:
                continue
            have = 'heading' if v['kind'] in HEADING_KINDS else v['kind']
            if want != have:
                kind_mismatch.append({'line': i, 'word': m.group(1), 'label': lab, 'label_kind': v['kind'],
                                      'context': ws(ln[max(0, m.start() - 60):m.end() + 30])})
    # labels kept whose caption / title changed
    retitled = []
    for r_ in label_out['rows']:
        if r_['status'] in ('unchanged', 'renumbered') and r_['desc_8f051a2'] and r_['desc_edd7a63'] \
                and r_['desc_8f051a2'] != r_['desc_edd7a63']:
            ratio = difflib.SequenceMatcher(a=r_['desc_8f051a2'], b=r_['desc_edd7a63'], autojunk=False).ratio()
            n_uses = sum(1 for ln in cl_n for m in REF_RX.finditer(ln) if r_['label'] in [k.strip() for k in m.group(2).split(',')])
            retitled.append({'label': r_['label'], 'number_8f051a2': r_['number_8f051a2'], 'number_edd7a63': r_['number_edd7a63'],
                             'old': r_['desc_8f051a2'][:90], 'new': r_['desc_edd7a63'][:90], 'similarity': round(ratio, 2),
                             'ref_uses_edd7a63': n_uses})
    # literal numbered pointers typed in main.tex (outside \ref)
    literal = []
    for i, ln in enumerate(cl_n, 1):
        s = REF_RX.sub(lambda m: ' ' * len(m.group(0)), ln)
        for rx in (R.PTR, R.PTR_APP):
            for m in rx.finditer(s):
                kind = R.KW.get(m.group(1), m.group(1))
                nums = re.findall(r'[A-E]\.\d+|[A-E](?![\w])|\d+(?:\.\d+)*', m.group(2))
                literal.append({'line': i, 'text': m.group(0), 'kind': kind, 'numbers': nums,
                                'denotes_edd7a63': [now_denotes(kind, n, ctx) for n in nums],
                                'denotes_8f051a2': [old_denotes(kind, n, ctx) for n in nums],
                                'context': ws(ln[max(0, m.start() - 80):m.end() + 40])})
    # prose mentions
    prose = {}
    for name, rx in PROSE_PATTERNS:
        hits = []
        for src, lines in (('main.tex', cl_n), ('letter', R._blank_cmd(letter_text, r'\rcomment').split('\n'))):
            for i, ln in enumerate(lines, 1):
                if src == 'letter':
                    ln = re.sub(r'(?<!\\)%.*', '', ln)
                for m in re.finditer(rx, ln, re.I):
                    hits.append({'file': src, 'line': i, 'match': m.group(0), 'context': ws(ln[max(0, m.start() - 90):m.end() + 60])})
        prose[name] = hits
    removed_labels = [r_['label'] for r_ in label_out['removed']]
    ref_to_removed = [u for u in uses if u['label'] in removed_labels]
    return {'refs_static_edd7a63': refs_n, 'refs_static_8f051a2': {k: refs_o[k] for k in ('n_ref_uses', 'n_ref_keys', 'undefined_refs', 'duplicate_labels')},
            'undefined_ref_uses_edd7a63': uses, 'undefined_ref_uses_to_removed_labels': ref_to_removed,
            'commented_lines_naming_removed_labels': commented,
            'kind_word_mismatch_edd7a63': kind_mismatch, 'labels_kept_caption_or_title_changed': retitled,
            'literal_pointers_main_tex_edd7a63': literal, 'prose_mentions': prose}


def old_denotes(kind, number, ctx):
    io = ctx['io']
    kinds = HEADING_KINDS if kind in ('section', 'new', 'appendix') else (kind,)
    hits = [o for o in io.values() if o['kind'] in kinds and o['number'] == number]
    if not hits:
        return 'nothing'
    return _obj_str(hits[0])


# ======================================================================================================================
#  5. A5: dropped blue
# ======================================================================================================================
BLUE_OPEN = re.compile(r'\\textcolor\{blue\}\{')


def blue_mask_text(text):
    """(clean text with every \\textcolor{blue}{ opener and its matching } removed, blue mask per clean char)."""
    n = len(text)
    remove = bytearray(n)
    blue = bytearray(n)
    for m in BLUE_OPEN.finditer(text):
        depth, j = 1, m.end()
        while j < n and depth:
            c = text[j]
            if c == '\\':
                j += 2
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
            j += 1
        if depth:
            continue
        close = j - 1
        for p in range(m.start(), m.end()):
            remove[p] = 1
        remove[close] = 1
        for p in range(m.end(), close):
            blue[p] = 1
    keep = [i for i in range(n) if not remove[i]]
    return ''.join(text[i] for i in keep), bytes(blue[i] for i in keep)


_BLUE_INDEX = {}


def blue_index(text):
    """(clean text, blue mask, kept-prefix) with kept_prefix[i] = number of kept characters before original position i."""
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    if key not in _BLUE_INDEX:
        clean, mask = blue_mask_text(text)
        rem = bytearray(len(text))
        # recompute the removal map exactly as blue_mask_text does
        n = len(text)
        for m in BLUE_OPEN.finditer(text):
            depth, j = 1, m.end()
            while j < n and depth:
                c = text[j]
                if c == '\\':
                    j += 2
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                j += 1
            if depth:
                continue
            for q in range(m.start(), m.end()):
                rem[q] = 1
            rem[j - 1] = 1
        pref, c_ = [0] * (n + 1), 0
        for i in range(n):
            pref[i] = c_
            c_ += 0 if rem[i] else 1
        pref[n] = c_
        if c_ != len(clean):
            raise RuntimeError('blue_index: removal map disagrees with blue_mask_text')
        _BLUE_INDEX[key] = (clean, mask, pref)
    return _BLUE_INDEX[key]


def blue_tokens(lines):
    """comment-stripped lines -> per line a list of (token, is_blue)."""
    text = '\n'.join(lines)
    clean, mask = blue_mask_text(text)
    per_line, pos = [], 0
    for ln in clean.split('\n'):
        toks = []
        for m in re.finditer(r'\S+', ln):
            seg = mask[pos + m.start(): pos + m.end()]
            toks.append((m.group(0), sum(seg) * 2 > len(seg)))
        per_line.append(toks)
        pos += len(ln) + 1
    return per_line


def blue_compare(old_lines, new_lines, k_moved=8):
    """Per-token status at the new revision: kept / unwrapped / rewritten / moved_unwrapped / new / new_blue / newly_blue."""
    to, tn = blue_tokens(old_lines), blue_tokens(new_lines)
    fo = [(t, b, i + 1) for i, ts in enumerate(to) for t, b in ts]
    fn = [(t, b, j + 1) for j, ts in enumerate(tn) for t, b in ts]
    so, sn = [0], [0]
    for ts in to:
        so.append(so[-1] + len(ts))
    for ts in tn:
        sn.append(sn[-1] + len(ts))
    status, old_of = ['kept'] * len(fn), [None] * len(fn)

    def pair(io_, jn):
        ob, nb = fo[io_][1], fn[jn][1]
        old_of[jn] = io_
        status[jn] = 'unwrapped' if ob and not nb else 'newly_blue' if nb and not ob else 'kept'
    lo = [' '.join(t for t, _ in ts) for ts in to]
    ln_ = [' '.join(t for t, _ in ts) for ts in tn]
    sm = difflib.SequenceMatcher(a=lo, b=ln_, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1):
                for p in range(len(to[i1 + k])):
                    pair(so[i1 + k] + p, sn[j1 + k] + p)
            continue
        a0, a1, b0, b1 = so[i1], so[i2], sn[j1], sn[j2]
        sm2 = difflib.SequenceMatcher(a=[x[0] for x in fo[a0:a1]], b=[x[0] for x in fn[b0:b1]], autojunk=False)
        for op2, x1, x2, y1, y2 in sm2.get_opcodes():
            if op2 == 'equal':
                for d in range(x2 - x1):
                    pair(a0 + x1 + d, b0 + y1 + d)
            elif op2 in ('replace', 'insert'):
                ob = sum(1 for x in fo[a0 + x1:a0 + x2] if x[1]) / max(1, x2 - x1) if op2 == 'replace' else 0.0
                nb = sum(1 for y in fn[b0 + y1:b0 + y2] if y[1]) / max(1, y2 - y1)
                for jn in range(b0 + y1, b0 + y2):
                    if fn[jn][1]:
                        status[jn] = 'new_blue'
                    elif op2 == 'replace' and ob >= 0.5 and nb < 0.5:
                        status[jn], old_of[jn] = 'rewritten', a0 + x1
                    else:
                        status[jn] = 'new'
    # moved text: a run of k new, non-blue tokens that occurs as k consecutive blue tokens at the old revision
    old_blue_sh = {tuple(x[0] for x in fo[i:i + k_moved]) for i in range(len(fo) - k_moved + 1)
                   if all(x[1] for x in fo[i:i + k_moved])}
    moved = set()
    for j in range(len(fn) - k_moved + 1):
        if all(status[j + d] == 'new' for d in range(k_moved)) and tuple(x[0] for x in fn[j:j + k_moved]) in old_blue_sh:
            moved.update(range(j, j + k_moved))
    for j in moved:
        status[j] = 'moved_unwrapped'
    return {'fo': fo, 'fn': fn, 'status': status, 'old_of': old_of, 'k_moved': k_moved,
            'counts': dict(Counter(status)), 'n_tokens_old': len(fo), 'n_tokens_new': len(fn),
            'n_blue_tokens_old': sum(1 for x in fo if x[1]), 'n_blue_tokens_new': sum(1 for x in fn if x[1])}


DROPPED = ('unwrapped', 'rewritten', 'moved_unwrapped')


def dropped_by_line(cmp_, new_raw, cl_new):
    """One row per edd7a63 line carrying dropped-blue tokens: counts by class, the line's first words, blue share now."""
    fn, st, oo, fo = cmp_['fn'], cmp_['status'], cmp_['old_of'], cmp_['fo']
    by = {}
    for j, (t, b, ln_no) in enumerate(fn):
        by.setdefault(ln_no, []).append(j)
    out = []
    for ln_no in sorted(by):
        js = by[ln_no]
        c = Counter(st[j] for j in js)
        if not any(c.get(k) for k in DROPPED):
            continue
        raw = new_raw[ln_no - 1] if ln_no - 1 < len(new_raw) else ''
        out.append({'line': ln_no, 'n_tokens_line': len(js), 'unwrapped': c.get('unwrapped', 0),
                    'rewritten': c.get('rewritten', 0), 'moved_unwrapped': c.get('moved_unwrapped', 0),
                    'blue_share_now': round(sum(1 for j in js if fn[j][1]) / len(js), 2),
                    'old_lines': sorted({fo[oo[j]][2] for j in js if oo[j] is not None and st[j] in DROPPED})[:4],
                    'first_words': ' '.join(fn[j][0] for j in js[:12])[:150],
                    'dropped_words_first': ' '.join(fn[j][0] for j in js if st[j] in DROPPED)[:150],
                    'line_is_caption': '\\caption' in raw, 'where': caption_or_env(cl_new, ln_no)})
    return out


def caption_blue(cl, label, adj):
    """(blue share of the \\caption argument of the float carrying `label`, its unwrapped text) or (None, None)."""
    v = adj['labels'].get(label)
    if v is None:
        return None, None
    text = '\n'.join(cl)
    pos = sum(len(x) + 1 for x in cl[:v['line'] - 1])
    m = re.compile(r'\\caption\{').search(text, pos)
    if not m:
        return None, None
    depth, j = 1, m.end()
    while j < len(text) and depth:
        if text[j] == '\\':
            j += 2
            continue
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        j += 1
    arg = text[m.end():j - 1]
    clean, mask, pref = blue_index(text)
    k0, k1 = pref[m.end()], pref[j - 1]
    seg = [b for b, ch in zip(mask[k0:k1], clean[k0:k1]) if not ch.isspace()]
    share = (sum(seg) / len(seg)) if seg else 0.0
    return round(share, 2), ws(strip_blue(arg))[:90]


def caption_or_env(cl, ln_no):
    """A short location hint: caption / table / equation / figure / algorithm / prose, and the heading above."""
    i = ln_no - 1
    stack = []
    for k in range(i, -1, -1):
        m = re.search(r'\\(begin|end)\{(table\*?|longtable|figure\*?|equation\*?|align\*?|algorithm|tabular\*?)\}', cl[k])
        if m:
            if m.group(1) == 'end' and k != i:
                break
            if m.group(1) == 'begin':
                stack.append(m.group(2))
                break
    head = next((re.sub(r'\\textcolor\{[^}]*\}\{(.*)\}', r'\1', re.match(r'\s*\\\w+\*?\{(.*)\}', cl[k]).group(1))
                 for k in range(i, -1, -1) if re.match(r'\s*\\(section|subsection|subsubsection)\*?\{', cl[k])), 'front matter')
    return (stack[0] if stack else 'prose') + ' / ' + head[:60]


def task_blue(ctx):
    cmp_ = blue_compare(ctx['cl_o'], ctx['cl_n'])
    rows = dropped_by_line(cmp_, ctx['ln'], ctx['cl_n'])
    n_wr_o = sum(len(BLUE_OPEN.findall(ln)) for ln in ctx['cl_o'])
    n_wr_n = sum(len(BLUE_OPEN.findall(ln)) for ln in ctx['cl_n'])
    caps = []
    for lab, v in ctx['adj_o']['labels'].items():
        if v['kind'] not in FLOAT_KINDS or lab not in ctx['adj_n']['labels']:
            continue
        bo, co = caption_blue(ctx['cl_o'], lab, ctx['adj_o'])
        bn, cn = caption_blue(ctx['cl_n'], lab, ctx['adj_n'])
        if bo is None or bn is None:
            continue
        caps.append({'label': lab, 'number_8f051a2': v['number'], 'number_edd7a63': ctx['adj_n']['labels'][lab]['number'],
                     'line_edd7a63': ctx['adj_n']['labels'][lab]['line'], 'blue_8f051a2': bo, 'blue_edd7a63': bn,
                     'caption_edd7a63': cn, 'dropped': bo >= 0.5 and bn < 0.5})
    heads = []
    for h in ctx['hm'].values():
        if h['number_new'] is None:
            continue
        bo = bool(BLUE_OPEN.search(ctx['cl_o'][h['line_old'] - 1]))
        bn = bool(BLUE_OPEN.search(ctx['cl_n'][h['line_new'] - 1]))
        heads.append({'heading_8f051a2': f'{h["kind"]} {h["number_old"]}', 'heading_edd7a63': f'{h["kind_new"]} {h["number_new"]}',
                      'line_edd7a63': h['line_new'], 'title_edd7a63': h['title_new'], 'blue_8f051a2': bo, 'blue_edd7a63': bn,
                      'dropped': bo and not bn})
    return {'n_wrappers_8f051a2_typeset': n_wr_o, 'n_wrappers_edd7a63_typeset': n_wr_n,
            'n_wrappers_8f051a2_raw': sum(len(BLUE_OPEN.findall(ln)) for ln in ctx['lo']),
            'n_wrappers_edd7a63_raw': sum(len(BLUE_OPEN.findall(ln)) for ln in ctx['ln']),
            'token_counts': {k: cmp_[k] for k in ('counts', 'n_tokens_old', 'n_tokens_new', 'n_blue_tokens_old', 'n_blue_tokens_new')},
            'dropped_by_line': rows, 'n_lines_dropped': len(rows),
            'n_lines_dropped_captions': sum(1 for r_ in rows if r_['line_is_caption']),
            'captions_by_label': caps, 'captions_dropped': [c for c in caps if c['dropped']],
            'heading_titles': heads, 'heading_titles_dropped': [h for h in heads if h['dropped']],
            'new_text_not_blue_lines': sorted({x[2] for x, st in zip(cmp_['fn'], cmp_['status']) if st == 'new'}),
            'method': ('comment-stripped source; each \\textcolor{blue}{ opener brace-matched to its closing brace (escaped '
                       'braces skipped); wrapper syntax removed; tokens = whitespace-separated words of the remaining text, '
                       'blue if more than half their characters lie inside a wrapper; lines aligned (difflib, on the '
                       'unwrapped text), tokens aligned inside changed line blocks. Classes of a token not blue at '
                       'edd7a63: "unwrapped" = the same token, aligned, blue at 8f051a2; "rewritten" = in a replaced run at '
                       'least half blue at 8f051a2; "moved_unwrapped" = new at its place but part of a run of 8 tokens that '
                       'occurs as 8 consecutive blue tokens at 8f051a2 (moved text). Captions are also compared label by '
                       'label (blue share of the non-space characters of the caption argument), and numbered heading '
                       'titles heading by heading (a wrapper on the heading line).')}


# ======================================================================================================================
#  6. A6: Appendix E
# ======================================================================================================================
def task_appendix_e(out_dir, texts):
    gen_full = os.path.join(out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir), GEN_NAME)
    gen_bytes, gen_stdout = A.run_generator(gen_full)
    ver = A.run_verifier()
    with open(os.path.join(os.path.dirname(gen_full), VERIFY_NAME), 'x', encoding='utf-8') as fh:
        fh.write(ver['output'])
    main_n = _git_bytes(['show', 'HEAD:main.tex'], cwd=CLONE)
    main_o = _git_bytes(['show', f'{OLD}:main.tex'], cwd=CLONE)
    with open(APPX_REL, 'rb') as fh:
        committed = fh.read()
    w181a = None
    if os.path.isfile(W181A_GEN):
        with open(W181A_GEN, 'rb') as fh:
            w181a = fh.read()
    loc_n, loc_o = A.locate_bytes(main_n, gen_bytes), A.locate_bytes(main_o, gen_bytes)
    ln_ = main_n.decode('utf-8').split('\n')
    lo = main_o.decode('utf-8').split('\n')
    gl = gen_bytes.decode('utf-8').split('\n')
    if gl and gl[-1] == '':
        gl = gl[:-1]
    adj_e = _num(ln_)[1]
    b0, b1 = A.appendix_block(ln_)
    o0, o1 = A.appendix_block(lo)
    blk_n, blk_o = ln_[b0:b1], lo[o0:o1]
    hunks = A.block_diff(gl, 0, blk_n, b0, ln_)
    kinds = Counter(h['kind'] for h in hunks)
    tables = {}
    try:
        raw_g = A.raw_tables(gl)
        raw_n = A.raw_tables(ln_)
        dt = A.diff_tables(raw_g, raw_n)
        tables = {'n_cells_compared': dt['n_cells_compared'], 'n_cells_changed': len(dt['cells']),
                  'cells_changed': dt['cells'], 'row_set_changes': dt['row_set_changes'], 'captions': dt['captions'],
                  'notes': dt['notes'], 'group_headers': dt['group_headers'], 'other_lines': dt['other_lines'],
                  'regions_edd7a63': {t: v['region'] for t, v in raw_n.items()}}
    except Exception as e:  # recorded, never absorbed: an integrity failure
        tables = {'error': repr(e)}
        _fail(f'A6 table parse failed: {e!r}')
    numbering_e = {lab: (adj_e['labels'].get(lab) or {}).get('number')
                   for lab in ('tab:appE_claims', 'tab:appE_evaluations', 'tab:appE_neighbours', 'tab:appE_benchmark',
                               'tab:appE_gap')}
    bitwise = len(loc_n) == 1 and loc_n[0]['starts_at_line_start'] and loc_n[0]['ends_at_line_end']
    gen_refs = [{'line': i, 'label': k.strip(), 'context': ws(x)[:140]} for i, x in enumerate(gl, 1)
                if not x.lstrip().startswith('%') for m in REF_RX.finditer(x) for k in m.group(2).split(',')
                if k.strip() not in adj_e['labels']]
    out = {'generator': {'path': GEN_REL, 'sha256': _sha(GEN_REL), 'output': os.path.relpath(gen_full, REPO) if gen_full.startswith(REPO) else gen_full,
                         'output_sha256': hashlib.sha256(gen_bytes).hexdigest(), 'output_lines': len(gl),
                         'stdout': gen_stdout[:500],
                         'equals_committed_appendixE_evaluation_record': gen_bytes == committed,
                         'equals_w181a_generated': (gen_bytes == w181a) if w181a is not None else None},
           'verifier': {'path': VERIFY_REL, 'sha256': _sha(VERIFY_REL), 'n_pass': ver['n_pass'], 'n_fail': ver['n_fail'],
                        'fail_lines': ver['fail_lines'], 'exception': ver['exception'],
                        'tempdir_output_sha256': ver['tempdir_output_sha256'],
                        'tempdir_output_equals_generated': ver['tempdir_output_sha256'] == hashlib.sha256(gen_bytes).hexdigest(),
                        'reads_main_tex': False,
                        'note': 'verify_appE.py re-derives the Appendix E prose counts from export_W160 through gen_appE.py; '
                                'it never reads main.tex, so its PASS count is independent of the manuscript'},
           'located_in_main_edd7a63': loc_n, 'located_in_main_8f051a2': loc_o,
           'bitwise_in_main_edd7a63': bitwise,
           'generated_refs_undefined_at_edd7a63': gen_refs,
           'bitwise_in_main_8f051a2_positive_control': len(loc_o) == 1,
           'block_edd7a63': {'lines': [b0 + 1, b1], 'n_lines': b1 - b0}, 'block_8f051a2': {'lines': [o0 + 1, o1], 'n_lines': o1 - o0},
           'block_8f051a2_equals_generated': blk_o == gl,
           'n_hunks_generated_vs_edd7a63_block': len(hunks), 'hunk_kinds': dict(kinds),
           'hunks': [{k: h[k] for k in ('op', 'lines_old', 'lines_new', 'section', 'kind')} |
                     {'n_old': len(h['old']), 'n_new': len(h['new']),
                      'old_first': ws(' '.join(h['old']))[:160], 'new_first': ws(' '.join(h['new']))[:160]} for h in hunks],
           'hunks_full': hunks,
           'tables_entry_diff_generated_vs_edd7a63': tables, 'appendix_E_table_numbers_edd7a63': numbering_e}
    if ver['exception'] or not out['bitwise_in_main_8f051a2_positive_control']:
        _fail(f'A6: verifier exception {ver["exception"]} / positive control at 8f051a2 {len(loc_o)} hits')
    return out


# ======================================================================================================================
#  7. negative controls
# ======================================================================================================================
def task_negative_controls(ctx, texts, label_out, ptr_out, appe):
    ctl = []
    # (1) label map: a captioned table inserted before tab:res_value_ladder at edd7a63 -> it and every later body table shift
    ln_ = list(ctx['ln'])
    i = next(k for k, x in enumerate(ln_) if '\\label{tab:res_value_ladder}' in x)
    i0 = max(k for k in range(i + 1) if '\\begin{table' in ln_[k])
    ins = ['\\begin{table}[h]', '\\caption{Injected control table.}', '\\label{tab:w182c_control}', '\\end{table}']
    mut = ln_[:i0] + ins + ln_[i0:]
    _, adj_m = _num(mut)
    shifted = [k for k, v in ctx['adj_n']['labels'].items() if v['kind'] == 'table' and k in adj_m['labels']
               and adj_m['labels'][k]['number'] != v['number']]
    ok1 = ('tab:res_value_ladder' in shifted and adj_m['labels']['tab:w182c_control']['number'] == '4'
           and adj_m['labels']['tab:res_value_ladder']['number'] == '5')
    ctl.append({'id': 'label map: injected table before Table 4 shifts tab:res_value_ladder 4 -> 5',
                'n_shifted': len(shifted), 'detected': ok1})
    # (2) label map: a deleted \label is reported removed
    mut2 = [x for x in ln_ if '\\label{tab:res_benchmark}' not in x]
    _, adj_m2 = _num(mut2)
    ok2 = 'tab:res_benchmark' not in adj_m2['labels'] and 'tab:res_benchmark' in ctx['adj_n']['labels']
    ctl.append({'id': 'label map: deleted \\label{tab:res_benchmark} is absent', 'detected': ok2})
    # (3) pointers: a synthetic letter with known answers, checked by the same pointer code
    tmp = tempfile.mkdtemp(prefix='w182c_ctl_')
    body = ('\\begin{document}\n'
            'See Table~4 and Section~3.1 and Table~9 and Figure~2 and Section~2.2.7 and [44] and Tables~7--15.\n'
            '\\end{document}\n')
    p = os.path.join(tmp, 'letter_ctl.tex')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write(body)
    rows, _, _ = pointer_rows(p, ctx, [], body, body)
    got = {(r_['kind'], '--'.join(r_['numbers'])): (r_['verdict'], r_.get('subclass'), r_.get('should_be')) for r_ in rows}
    expect = {('table', '4'): ('MISMATCH', 'renumbered', 'D.18'), ('section', '3.1'): ('MATCH', None, None),
              ('table', '9'): ('MISMATCH', 'target removed', None), ('figure', '2'): ('MISMATCH', 'target removed', None),
              ('section', '2.2.7'): ('MISMATCH', 'renumbered', '2.2.6'),
              ('table', '7--15'): ('MISMATCH', 'range member removed', '4--11')}
    ok3 = all(got.get(k) == v for k, v in expect.items())
    ref44 = got.get(('reference', '44'))
    ctl.append({'id': 'pointers: synthetic letter (Table 4 -> D.18, Section 3.1 MATCH, Table 9 removed, Figure 2 removed, '
                      '2.2.7 -> 2.2.6, Tables 7--15 member removed -> 4--11)', 'got': {f'{k[0]} {k[1]}': v for k, v in got.items()},
                'reference_44': ref44, 'detected': ok3})
    # (4) pointers: letter v9 against 8f051a2 itself -> no MISMATCH
    ctx_self = dict(ctx)
    ctx_self.update({'adj_n': ctx['adj_o'], 'inn': ctx['io'], 'cl_n': ctx['cl_o'], 'ln': ctx['lo'], 'en': ctx['eo'],
                     'cites_n': ctx['cites_o']})
    ctx_self['hm'] = heading_map(ctx['adj_o'], ctx['adj_o'], ctx['cl_o'], ctx['cl_o'])
    w180c = json.load(open(W180C_JSON, encoding='utf-8'))
    rows4, _, _ = pointer_rows(LETTER, ctx_self, w180c['pointers']['pointers'], texts['letter_w180c'], texts['letter_head'])
    c4 = Counter(r_['verdict'] for r_ in rows4)
    ok4 = c4.get('MISMATCH', 0) == 0 and c4.get('AMBIGUOUS', 0) == 0
    ctl.append({'id': 'pointers: letter v9 against 8f051a2 numbering itself gives 0 MISMATCH, 0 AMBIGUOUS', 'counts': dict(c4),
                'detected': ok4})
    # (5) A4: an injected \ref to a removed label and a typed "Table 12" are caught
    mut5 = list(ctx['ln'])
    k5 = next(k for k, x in enumerate(mut5) if '\\label{tab:res_value_ladder}' in x)
    mut5.insert(k5 + 2, 'Control: see Table~\\ref{tab:res_flex_ladder} and Table 12.')
    refs5 = C.task_refs(mut5, _num(mut5)[1])
    s5 = REF_RX.sub(lambda m: ' ' * len(m.group(0)), 'Control: see Table~\\ref{tab:res_flex_ladder} and Table 12.')
    lit5 = [m.group(0) for m in R.PTR.finditer(s5)]
    ok5 = 'tab:res_flex_ladder' in refs5['undefined_refs'] and lit5 == ['Table 12']
    ctl.append({'id': 'A4: injected \\ref{tab:res_flex_ladder} undefined, typed "Table 12" literal', 'undefined': refs5['undefined_refs'],
                'literal': lit5, 'detected': ok5})
    # (6) A5: one caption unwrapped in a copy of 8f051a2 -> exactly that caption's words; self-compare -> nothing
    lo = list(ctx['cl_o'])
    k6 = next(k for k, x in enumerate(lo) if re.match(r'\s*\\caption\{\\textcolor\{blue\}\{.*\}\}\s*$', x))
    m6 = re.match(r'(\s*)\\caption\{\\textcolor\{blue\}\{(.*)\}\}\s*$', lo[k6])
    mut6 = list(lo)
    mut6[k6] = f'{m6.group(1)}\\caption{{{m6.group(2)}}}'
    c6 = blue_compare(lo, mut6)
    un6 = [j for j, st in enumerate(c6['status']) if st == 'unwrapped']
    n_cap = len(blue_mask_text(m6.group(2))[0].split())
    self6 = blue_compare(lo, lo)
    ok6 = (len(un6) == n_cap and {c6['fn'][j][2] for j in un6} == {k6 + 1}
           and set(self6['counts']) == {'kept'})
    ctl.append({'id': f'A5: caption at 8f051a2 l. {k6 + 1} unwrapped in a copy -> {n_cap} tokens on that line only; self all kept',
                'got': len(un6), 'self_counts': self6['counts'], 'detected': ok6})
    # (7) A5: a paragraph wrapper whose opener and closer sit on their own lines is removed -> the paragraph is reported
    k7 = next(k for k, x in enumerate(lo) if x.strip() == '\\textcolor{blue}{' and k > len(lo) // 2)
    k7e = next(k for k in range(k7 + 1, len(lo)) if lo[k].strip() == '}')
    mut7 = lo[:k7] + lo[k7 + 1:k7e] + lo[k7e + 1:]
    c7 = blue_compare(lo, mut7)
    n7 = sum(len(blue_mask_text(x)[0].split()) for x in lo[k7 + 1:k7e])
    got7 = sum(1 for st in c7['status'] if st == 'unwrapped')
    ok7 = got7 == n7 and n7 > 0
    ctl.append({'id': f'A5: wrapper lines {k7 + 1}/{k7e + 1} of 8f051a2 removed -> {n7} tokens reported', 'got': got7,
                'detected': ok7})
    # (7b) A5: the same paragraph moved to the end of the document without its wrapper -> moved_unwrapped
    mut7b = lo[:k7] + lo[k7e + 1:] + lo[k7 + 1:k7e]
    c7b = blue_compare(lo, mut7b)
    got7b = sum(1 for st in c7b['status'] if st in ('moved_unwrapped', 'unwrapped'))
    ok7b = got7b >= n7 - 7 and got7b <= n7
    ctl.append({'id': f'A5: the same {n7}-token paragraph moved to the end without its wrapper -> moved/unwrapped tokens',
                'got': got7b, 'counts': c7b['counts'], 'detected': ok7b})
    # (8) A6: a one-byte change in the generated block is not located in 8f051a2 (where the original is)
    main_o = _git_bytes(['show', f'{OLD}:main.tex'], cwd=CLONE)
    gl = appe['_gen_bytes']
    j = gl.find(b'Comparisons')
    mut8 = gl[:j] + b'X' + gl[j + 1:]
    ok8 = len(A.locate_bytes(main_o, gl)) == 1 and len(A.locate_bytes(main_o, mut8)) == 0
    ctl.append({'id': 'A6: generated block located once in 8f051a2; one byte changed -> not located', 'detected': ok8})
    for c in ctl:
        if c['detected'] is not True:
            _fail(f'negative control not detected: {c["id"]}')
    return {'controls': ctl, 'tempdir': tmp}


# ======================================================================================================================
#  guards, output
# ======================================================================================================================
def verify_guards():
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(B.FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    g = {'own': GUARD.verify(0), 'w181a': A.GUARD.verify(0), 'w179b': B.GUARD.verify(0), 'w178c': C.GUARD.verify(0),
         'w177b3r': R.GUARD.verify(0), 'w177b3': P3.GUARD.verify(0), 'w167': W167.GUARD.verify(0),
         'phase_b_record': PB.PARENT_GUARD.verify(0)}
    pk = {'own': dict(PICKLE_COUNTS), 'w181a': dict(A.PICKLE_COUNTS), 'w179b': dict(B.PICKLE_COUNTS),
          'w178c': dict(C.PICKLE_COUNTS), 'w177b3r': dict(R.PICKLE_COUNTS), 'w177b3': dict(P3.PICKLE_COUNTS),
          'w167': dict(W167.PICKLE_COUNTS)}
    out = {'verify_0_failures': g,
           'counts': {'own': GUARD.counts, 'w181a': A.GUARD.counts, 'w179b': B.GUARD.counts, 'w178c': C.GUARD.counts,
                      'w177b3r': R.GUARD.counts, 'w177b3': P3.GUARD.counts, 'w167': W167.GUARD.counts,
                      'phase_b_record': PB.PARENT_GUARD.counts},
           'pickle_counts': pk, 'pickle_blocked_at_end': pickle.load is _blocked_load and pickle.loads is _blocked_loads,
           'production_modules_imported': forbidden}
    bad = any(g.values()) or any(v for d in pk.values() for v in d.values()) or forbidden or not out['pickle_blocked_at_end']
    if bad:
        _fail(f'guards: {out}')
    out['ok'] = not bad
    return out


def _c(s, n=110):
    s = str(s).replace('|', '/').replace('\n', ' ')
    return s if len(s) <= n else s[:n - 1] + '…'


def write_md(out, path):
    I = out['identity']
    L = [f'# W182c — label map, letter pointers, dropped blue, Appendix E at Overleaf {I["clone_head"]} '
         f'(generated by {SCRIPT_NAME})', '',
         f'Clone HEAD {I["clone_head_full"]} ({I["clone_head_date"]}); {I["commits_after_8f051a2"]} commits after 8f051a2 '
         f'({I["commits_after_d2ebb0d"]} after d2ebb0d); main.tex {I["main_sha256_working_copy"][:16]} ({I["main_lines"]} lines); '
         f'letter {I["letter_sha256_working_copy"][:16]} (v9, = d2ebb0d blob). Static numbering (no TeX installed; nothing '
         f'compiled). Check failures: {len(out["check_failures"])}.', '', '## Identity checks', '', '| check | ok |', '|---|---|']
    L += [f'| {k} | {v} |' for k, v in I['checks'].items()]
    LM = out['label_map']
    L += ['', '## A3 label map 8f051a2 → edd7a63', '',
          f'Labels: 8f051a2 {LM["n_labels_8f051a2"]}, edd7a63 {LM["n_labels_edd7a63"]}; {LM["counts"]}. By class: {LM["by_class"]}. '
          f'8f051a2 numbering = cd57efc: {LM["cross_check_8f051a2_equals_cd57efc_all_labels"]}; = e5322f8: '
          f'{LM["cross_check_8f051a2_equals_e5322f8_all_labels"]}. Citations: order identical {LM["citations"]["order_identical"]}, '
          f'renumbered {len(LM["citations"]["renumbered"])}, new keys {LM["citations"]["new_keys"]}.', '',
          '| label | class | 8f051a2 | edd7a63 | status | object at edd7a63 (or 8f051a2) |', '|---|---|---|---|---|---|']
    for r_ in LM['rows']:
        L.append(f'| {r_["label"]} | {r_["class"]} | {r_["number_8f051a2"] or "—"} | {r_["number_edd7a63"] or "—"} | {r_["status"]} | '
                 f'{_c(r_["desc_edd7a63"] or r_["desc_8f051a2"] or "", 70)} |')
    L += ['', '### Headings (labelled and unlabelled) old → new', '', '| kind | 8f051a2 | title 8f051a2 | edd7a63 | title edd7a63 | method | status |',
          '|---|---|---|---|---|---|---|']
    for h in LM['heading_map']:
        L.append(f'| {h["kind"]} | {h["number_old"]} | {_c(h["title_old"], 50)} | {h["number_new"] or "—"} | {_c(h["title_new"] or "", 50)} | '
                 f'{_c(h["method"], 40)} | {h["status"]} |')
    L += ['', f'Unlabelled equations 8f051a2: {[(e["number_8f051a2"], e["number_edd7a63"]) for e in LM["equations_unlabelled_8f051a2"]]}; '
          f'edd7a63: {[(e["number_edd7a63"], e["in_8f051a2"]) for e in LM["equations_unlabelled_edd7a63"]]}. Unlabelled floats: '
          f'{LM["unlabelled_floats"]}.']
    P = out['pointers']
    L += ['', '## A3 letter v9 pointers at edd7a63', '',
          f'{P["n_pointers"]} pointers: {P["counts"]}; MISMATCH subclasses {P["mismatch_subclasses"]}; distinct (line, text, number) '
          f'MISMATCH {P["n_mismatch_distinct_line_text_number"]}. W180c target cross-check: {P["w180c_cross_check"]["n_agree"]}/'
          f'{P["w180c_cross_check"]["n_compared"]} agree. Letter lines not mapped to v7: {P["letter_lines_not_mapped_to_v7"]}.', '',
          '| l. | text | printed | target (8f051a2) | edd7a63 | verdict | printed number now denotes | note |', '|---|---|---|---|---|---|---|---|']
    for r_ in P['pointers']:
        tgt = r_.get('target_8f051a2')
        tgt = ', '.join(tgt) if isinstance(tgt, list) else (tgt or '')
        L.append(f'| {r_["line"]} | {_c(r_["text"], 40)} | {"--".join(r_["numbers"])} | {_c(tgt, 50)} | '
                 f'{_c(r_.get("should_be") or ("--".join(x or "∅" for x in r_.get("numbers_edd7a63") or [])), 20)} | '
                 f'{r_["verdict"]}{(" (" + r_["subclass"] + ")") if r_.get("subclass") else ""} | '
                 f'{_c("; ".join(r_.get("printed_now_denotes") or []), 60)} | {_c(r_.get("note") or "", 70)} |')
    D = out['dangling']
    L += ['', '## A4 dangling / re-pointed references and prose mentions', '',
          f'Undefined refs at edd7a63: {D["refs_static_edd7a63"]["undefined_refs"]} (8f051a2: {D["refs_static_8f051a2"]["undefined_refs"]}); '
          f'duplicates {D["refs_static_edd7a63"]["duplicate_labels"]}.', '']
    for u in D['undefined_ref_uses_edd7a63']:
        L.append(f'- l. {u["line"]}: \\{u["cmd"]}{{{u["label"]}}} (8f051a2: {u["in_8f051a2_as"]}) — {_c(u["context"], 150)}')
    L += ['', 'Commented lines naming removed labels: ' + '; '.join(f'l. {c["line"]} {c["label"]}' for c in D['commented_lines_naming_removed_labels']), '',
          'Kind word before \\ref differing from the label kind:']
    L += [f'- l. {k["line"]}: "{k["word"]}" + {k["label"]} ({k["label_kind"]}) — {_c(k["context"], 120)}' for k in D['kind_word_mismatch_edd7a63']] or ['- none']
    L += ['', 'Labels kept whose caption or title changed (similarity of old/new text):', '',
          '| label | 8f051a2 → edd7a63 | sim. | old | new |', '|---|---|---|---|---|']
    for r_ in D['labels_kept_caption_or_title_changed']:
        L.append(f'| {r_["label"]} | {r_["number_8f051a2"]} → {r_["number_edd7a63"]} | {r_["similarity"]} | {_c(r_["old"], 60)} | {_c(r_["new"], 60)} |')
    L += ['', 'Literal (typed) numbered pointers in main.tex at edd7a63:', '']
    L += [f'- l. {x["line"]}: "{x["text"]}" — edd7a63: {"; ".join(x["denotes_edd7a63"])}; 8f051a2: {"; ".join(x["denotes_8f051a2"])} — {_c(x["context"], 110)}'
          for x in D['literal_pointers_main_tex_edd7a63']] or ['- none']
    L += ['', 'Prose mentions:', '']
    for name, hits in D['prose_mentions'].items():
        L.append(f'- {name}: {len(hits)}')
        L += [f'  - {h["file"]} l. {h["line"]}: "{h["match"]}" — {_c(h["context"], 130)}' for h in hits]
    BL = out['blue']
    L += ['', '## A5 dropped \\textcolor{blue} (list only)', '',
          f'Wrappers (typeset): 8f051a2 {BL["n_wrappers_8f051a2_typeset"]}, edd7a63 {BL["n_wrappers_edd7a63_typeset"]} (raw '
          f'{BL["n_wrappers_8f051a2_raw"]} / {BL["n_wrappers_edd7a63_raw"]}). Token classes at edd7a63: {BL["token_counts"]["counts"]}. '
          f'Lines with dropped blue: {BL["n_lines_dropped"]} ({BL["n_lines_dropped_captions"]} caption lines). Captions by label '
          f'dropped: {len(BL["captions_dropped"])}; heading titles dropped: {len(BL["heading_titles_dropped"])}.', '',
          '### Captions whose blue was dropped (label by label)', '', '| label | 8f051a2 → edd7a63 | edd7a63 l. | blue share old/new | caption at edd7a63 |',
          '|---|---|---|---|---|']
    L += [f'| {c["label"]} | {c["number_8f051a2"]} → {c["number_edd7a63"]} | {c["line_edd7a63"]} | {c["blue_8f051a2"]}/{c["blue_edd7a63"]} | '
          f'{_c(c["caption_edd7a63"], 80)} |' for c in BL['captions_dropped']] or ['| none | | | | |']
    L += ['', '### Heading titles whose blue was dropped', '']
    L += [f'- {h["heading_edd7a63"]} (l. {h["line_edd7a63"]}, was {h["heading_8f051a2"]}): {h["title_edd7a63"]}' for h in BL['heading_titles_dropped']] or ['- none']
    L += ['', '### Lines carrying text that was blue at 8f051a2 and is not blue at edd7a63', '',
          '| edd7a63 l. | 8f051a2 l. | unwr. | rewr. | moved | blue now | where | line starts |', '|---|---|---|---|---|---|---|---|']
    L += [f'| {x["line"]} | {x["old_lines"]} | {x["unwrapped"]} | {x["rewritten"]} | {x["moved_unwrapped"]} | {x["blue_share_now"]} | '
          f'{_c(x["where"], 45)} | {_c(x["first_words"], 90)} |' for x in BL['dropped_by_line']]
    L += ['', f'New text not blue (not A5; line numbers in the JSON only): {len(BL["new_text_not_blue_lines"])} lines.']
    E = out['appendix_e']
    L += ['', '## A6 Appendix E', '',
          f'Generator {E["generator"]["output"]} sha256 {E["generator"]["output_sha256"][:16]}, {E["generator"]["output_lines"]} lines; '
          f'= committed appendixE_evaluation_record.tex: {E["generator"]["equals_committed_appendixE_evaluation_record"]}; = W181a output: '
          f'{E["generator"]["equals_w181a_generated"]}. Located in main.tex at edd7a63: {len(E["located_in_main_edd7a63"])} '
          f'(bitwise {E["bitwise_in_main_edd7a63"]}); at 8f051a2: {len(E["located_in_main_8f051a2"])} (positive control). '
          f'Block edd7a63 l. {E["block_edd7a63"]["lines"]} ({E["block_edd7a63"]["n_lines"]} lines; generated {E["generator"]["output_lines"]}). '
          f'verify_appE: {E["verifier"]["n_pass"]} PASS, {E["verifier"]["n_fail"]} FAIL. References in the generated block undefined '
          f'at edd7a63: {[(g["line"], g["label"]) for g in E["generated_refs_undefined_at_edd7a63"]]}.', '',
          f'Hunks generated → edd7a63 block: {E["n_hunks_generated_vs_edd7a63_block"]} {E["hunk_kinds"]}', '',
          '| op | gen l. | main l. | section | kind | −/+ | new text starts |', '|---|---|---|---|---|---|---|']
    for h in E['hunks']:
        L.append(f'| {h["op"]} | {h["lines_old"]} | {h["lines_new"]} | {_c(h["section"], 45)} | {h["kind"]} | {h["n_old"]}/{h["n_new"]} | '
                 f'{_c(h["new_first"] or h["old_first"], 90)} |')
    T = E['tables_entry_diff_generated_vs_edd7a63']
    if 'error' not in T:
        L += ['', f'Tables E.20–E.24 entries: {T["n_cells_compared"]} compared, {T["n_cells_changed"]} changed; row-set changes '
              f'{len(T["row_set_changes"])}; captions {len(T["captions"])}, notes {len(T["notes"])}, group headers '
              f'{len(T["group_headers"])}, other lines {len(T["other_lines"])}. Table numbers at edd7a63: {E["appendix_E_table_numbers_edd7a63"]}']
        for g in T['group_headers'] + T['captions'] + T['notes'] + T['other_lines']:
            L.append(f'- {g["table"]} {g["op"]} l. {g["lines_old"]} → {g["lines_new"]}: {_c(g["token_changes"], 200)}')
    L += ['', '## Negative controls', ''] + [f'- {c["id"]}: detected {c["detected"]}' for c in out['negative_controls']['controls']]
    L += ['', '## Predictions', ''] + [f'- {k}: {v}' for k, v in out['predictions'].items()]
    L += ['', f'Guards ok: {out["guards"]["ok"]}; pickle {out["guards"]["pickle_counts"]["own"]}; production modules imported '
          f'{out["guards"]["production_modules_imported"]}.']
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run(out_dir):
    if not os.path.isdir(out_dir):
        print(f'REFUSED: {out_dir} does not exist', flush=True)
        sys.exit(1)
    p = os.path.join(out_dir, POST_NAME)
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = ['launch.log', MANIFEST_NAME, MD_NAME, OUT_NAME, GEN_NAME, VERIFY_NAME] + list(TYPING_NAMES)
    missing = [n for n in names if not os.path.isfile(os.path.join(out_dir, n))]
    if missing:
        print(f'REFUSED: post-run inputs missing: {missing}', flush=True)
        sys.exit(1)
    man = {'files': {n: _sha(os.path.join(out_dir, n)) for n in names}, 'script_sha256': _sha(__file__),
           'written_utc': datetime.now(timezone.utc).isoformat()}
    with open(p, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    print(f'written {p}', flush=True)


def main():
    out_dir = OUT_DIR
    if '--trial-out-dir' in sys.argv:
        out_dir = sys.argv[sys.argv.index('--trial-out-dir') + 1]
        if not os.path.isabs(out_dir) or os.path.abspath(out_dir).startswith(REPO + os.sep):
            print('REFUSED: --trial-out-dir is for trial runs, an absolute path outside the repository', flush=True)
            sys.exit(1)
    if '--post-run' in sys.argv:
        post_run(out_dir)
        return
    if not os.path.isdir(out_dir):
        print(f'REFUSED: {out_dir} does not exist (create it with mkdir; it must hold only launch.log)', flush=True)
        sys.exit(1)
    extra = sorted(set(os.listdir(out_dir)) - {'launch.log'})
    if extra:
        print(f'REFUSED: {out_dir} already holds {extra} (write-once; fresh directory required)', flush=True)
        sys.exit(1)
    refuse_concurrent()
    t0 = datetime.now(timezone.utc)
    _log(f'W182c: start (out {out_dir})')
    out = {'schema': 'p515_s53_w182c_labels_pointers_appE',
           'task': 'W182c (Addendum 82 v2, W182 items A3-A6): label map 8f051a2 -> edd7a63 and letter v9 pointers; dangling '
                   'references; dropped blue; Appendix E bitwise + verify_appE',
           'objective_convention': 'no objective value is computed here; Q = gross_operational_cost, settlement excluded, '
                                   'wherever the manuscript quotes one',
           'static_numbering': 'no TeX installation (pdflatex/latex/latexmk/lualatex/xelatex absent); numbering is static',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(__file__)},
           'imported': {m: _sha(m) for m in ('p515_s53_w179b_checks.py', 'p515_s53_w178c_letter_checks.py',
                                             'p515_s53_w177b3r_checks.py', 'p515_s53_w177b_part3_checks.py',
                                             'p515_s53_w167_nomenclature_years.py', 'p515_s47_phase_b_record.py',
                                             'p515_s53_w181a_identity_appendixE.py', 'gate_result_io.py',
                                             'p513_solve_profile_guard.py')}}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    out['identity'], texts = identity()
    I = out['identity']
    _log(f"identity: head {I['clone_head']} ({I['commits_after_8f051a2']} after 8f051a2, {I['commits_after_d2ebb0d']} after d2ebb0d); "
         f"checks {sum(I['checks'].values())}/{len(I['checks'])}; main hunks {I['main_diff_8f051a2_head_hunks_U3']}; letter v8->HEAD "
         f"hunks U3 {I['letter_diff_cd57efc_head_hunks_U3']} U0 {I['letter_diff_cd57efc_head_hunks_U0']}")
    out['label_map'], ctx = task_label_map(texts)
    LM = out['label_map']
    _log(f"(A3) labels {LM['n_labels_8f051a2']} -> {LM['n_labels_edd7a63']}: {LM['counts']}; headings {LM['heading_counts']}; "
         f"= cd57efc {LM['cross_check_8f051a2_equals_cd57efc_all_labels']}, = e5322f8 {LM['cross_check_8f051a2_equals_e5322f8_all_labels']}")
    out['pointers'] = task_pointers(ctx, texts)
    P = out['pointers']
    _log(f"(A3) pointers {P['n_pointers']}: {P['counts']} {P['mismatch_subclasses']}; W180c cross-check "
         f"{P['w180c_cross_check']['n_agree']}/{P['w180c_cross_check']['n_compared']}")
    out['dangling'] = task_dangling(ctx, LM, texts['letter_head'])
    D = out['dangling']
    _log(f"(A4) undefined {D['refs_static_edd7a63']['undefined_refs']}; kind-word mismatches {len(D['kind_word_mismatch_edd7a63'])}; "
         f"retitled labels {len(D['labels_kept_caption_or_title_changed'])}; literal pointers {len(D['literal_pointers_main_tex_edd7a63'])}; "
         f"prose {[(k, len(v)) for k, v in D['prose_mentions'].items()]}")
    out['blue'] = task_blue(ctx)
    BL = out['blue']
    _log(f"(A5) wrappers {BL['n_wrappers_8f051a2_typeset']} -> {BL['n_wrappers_edd7a63_typeset']}; token classes "
         f"{BL['token_counts']['counts']}; lines with dropped blue {BL['n_lines_dropped']}; captions dropped "
         f"{len(BL['captions_dropped'])}; heading titles dropped {len(BL['heading_titles_dropped'])}")
    appe = task_appendix_e(out_dir, texts)
    appe['_gen_bytes'] = open(os.path.join(out_dir, GEN_NAME), 'rb').read()
    out['negative_controls'] = task_negative_controls(ctx, texts, LM, P, appe)
    del appe['_gen_bytes']
    out['appendix_e'] = appe
    _log(f"(A6) bitwise at edd7a63 {appe['bitwise_in_main_edd7a63']} ({len(appe['located_in_main_edd7a63'])} hits); at 8f051a2 "
         f"{len(appe['located_in_main_8f051a2'])}; hunks {appe['n_hunks_generated_vs_edd7a63_block']} {appe['hunk_kinds']}; "
         f"verify {appe['verifier']['n_pass']} PASS {appe['verifier']['n_fail']} FAIL")
    _log(f"negative controls: {[(c['id'][:40], c['detected']) for c in out['negative_controls']['controls']]}")
    n_mm = P['counts'].get('MISMATCH', 0)
    out['predictions'] = {
        'A3 (expert): >= 10 letter pointers MISMATCH': {'outcome': P['counts'], 'n_mismatch': n_mm,
                                                        'n_mismatch_distinct_line_text_number': P['n_mismatch_distinct_line_text_number'],
                                                        'held': n_mm >= 10},
        'A6 (expert): Appendix E bitwise equal to gen_appE.py output': {'hits_edd7a63': len(appe['located_in_main_edd7a63']),
                                                                          'n_hunks': appe['n_hunks_generated_vs_edd7a63_block'],
                                                                          'held': appe['bitwise_in_main_edd7a63']},
        'A6 (expert): verify_appE.py 33 PASS': {'n_pass': appe['verifier']['n_pass'], 'n_fail': appe['verifier']['n_fail'],
                                                'held': appe['verifier']['n_pass'] == 33 and appe['verifier']['n_fail'] == 0}}
    out['guards'] = verify_guards()
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    out_json, out_md, out_man = (os.path.join(out_dir, n) for n in (OUT_NAME, MD_NAME, MANIFEST_NAME))
    with open(out_json, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, out_md)
    inputs = sorted({MAIN_TEX, LETTER, DTX, LETTER_V8, GEN_REL, VERIFY_REL, APPX_REL, W180C_JSON, BRIEF, REVIEW, SCRIPT_NAME,
                     'p515_s53_w179b_checks.py', 'p515_s53_w178c_letter_checks.py', 'p515_s53_w177b3r_checks.py',
                     'p515_s53_w177b_part3_checks.py', 'p515_s53_w167_nomenclature_years.py', 'p515_s47_phase_b_record.py',
                     'p515_s53_w181a_identity_appendixE.py', 'gate_result_io.py', 'p513_solve_profile_guard.py'}
                    | {os.path.join(MR, 'export_W160', f) for f in os.listdir(os.path.join(MR, 'export_W160'))})
    manifest = {'outputs': {n: _sha(os.path.join(out_dir, n)) for n in (OUT_NAME, MD_NAME, GEN_NAME, VERIFY_NAME)},
                'out_dir': out_dir,
                'git_show_blobs_sha256': {f'{c}:{p}': sha_text(show(c, p)) for c in (OLD, ROUND7, W180C_HEAD, LETTER_LAST, 'HEAD')
                                          for p in ('main.tex', LETTER_NAME)},
                'clone_head_full': I['clone_head_full'],
                'inputs': {p: _sha(p) for p in inputs}}
    with open(out_man, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {out_json}, {out_md}, {out_man}; check failures {len(CHECK_FAILURES)}; guards ok {out["guards"]["ok"]}; '
         f'pickle {PICKLE_COUNTS}; predictions {[(k, v["held"]) for k, v in out["predictions"].items()]}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
