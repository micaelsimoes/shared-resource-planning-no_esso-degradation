"""
P5.15 Addendum 78, W179 items (b) and (c), Planner task W179b: the main-text and letter changes of Addendum 78 at the
Overleaf HEAD 48099ab (main.tex = manuscript_review/main_e926fb7_appE.tex; letter = response_to_reviewers_v6.tex), the
pointer check of the whole letter v6, and the G1-G12 fixes and letter v5 edits (applied at e926fb7, still present at
48099ab) against the W178 rows they address.
ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT; read-only on the clone, manuscript_review/, the frozen JSON
(590088fe), every committed artifact and every earlier script (p515_s53_w178c_letter_checks.py is IMPORTED, not edited;
it imports W177b-3' and its chain, also not edited).

What it does (static reading only):
  0. Identity: clone HEAD (declared 48099ab), main.tex / letter sha256 and line counts, byte-identity with their
     manuscript_review/ copies; e926fb7 blobs = main_9fd19e7_fixed.tex / letter v5; 9fd19e7 blobs = the W178 hashes;
     e926fb7 -> 48099ab is one commit; the +/- lines of the clone diffs 9fd19e7 -> e926fb7 -> 48099ab equal those of the
     committed main_9fd19e7_followup.diff, main_e926fb7_appE.diff and letter_v5_v6.diff (and v4 -> v5 for the letter).
  1. Capture-path checklist, asserted BEFORE any task runs (exit 4 naming the missing item).
  2. Static numbering at 48099ab (W178c's numbering_adjusted, imported) and at e926fb7: every label outside Appendix E
     keeps its number; Appendix E tables E.20-E.24; elsarticle.dtx \\appendix read (table counter not reset); 0
     undefined refs, 0 duplicate labels; every remaining "<word>~\\ref{app:...}" prefix; every CONFIRM comment.
  3. (b) Section 4.6 / 4.7 / 4.8 as edited: every numeric token of the changed paragraphs declared and checked against
     the frozen JSON / a named record (undeclared tokens fail); the moved items (absent from 4.7/4.8, present in E.5/E.6)
     and the kept items; claims rated on the W174b scale (H / M / L / consistent) with located evidence.
  4. (b) Letter: R2.2, R3.1 and the two "Further changes" bullets (numbers in the spans inserted since v4, the rest
     carried from W178c's committed JSON; claims rated); then every numbered pointer of the letter v6 against the static
     numbering at 48099ab: MATCH / MISMATCH / SUBMITTED / AMBIGUOUS, with content needles.
  5. (c) G1-G12 (14 hunks of main_9fd19e7_followup.diff) and the letter v5 edits: applied as worded (verbatim / with
     changes / absent) at e926fb7 and at 48099ab; the W178 rows they address re-scored (closed / still / new
     difference, rated), plus the rows Addendum 78 says Appendix E closes (LM-9, 4.6-2, R3.7(a)).

History: drafted by a Worker interrupted by a session limit (uncommitted, no outputs); reviewed, fixed and trial-run by the
resuming Worker before this commit. Fixes: the letter's "Section~3.3" pointer resolved through a level-qualified title
('titlek:'), since "Transmission Network" is the title of both 3.3 and Appendix C (it was AMBIGUOUS); G9's "one another"
declared as language; G10's "three" derived from the T6 export; G4 checked against both instance files; the CONFIRM row
read on raw (comment) lines; two carried letter tokens W178c has no row for declared; a vacuous W153 fallback removed.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at exit, with the
guards of the imported W178c / W177b-3' / W177b-3 / W167 / Phase B record modules; pickle.load / loads blocked and
counted (0 expected); no production module imported (asserted). Refuses to run concurrently with another copy of
itself. Output JSON written through gate_result_io (string flags refused).

Output (write-once): the directory must exist and hold nothing but launch.log; writes w179b_checks.json,
w179b_checks.md and manifest_sha256.json. `--post-run` (after the run and the W100 typing test) writes
manifest_post_run_sha256.json once. `--out-dir DIR`: trial runs only (absolute path outside the repository).

Commands (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w179b_text_letter && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w179b_checks.py \\
      > data/SRP1/Results/P515S53/w179b_text_letter/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
      --out data/SRP1/Results/P515S53/w179b_text_letter/w179b_bool_typing_test.json \\
      > data/SRP1/Results/P515S53/w179b_text_letter/w179b_bool_typing_test.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w179b_checks.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label="P5.15 W179b text + letter checks (never solves)").install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W179b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W179b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s53_w178c_letter_checks as C  # noqa: E402  (W178c; installs its own permitted=() guard; NOT edited)
import gate_result_io as GRIO  # noqa: E402

R, P3, PB, W167 = C.R, C.P3, C.PB, C.W167
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w179b_text_letter')
OUT_NAME = 'w179b_checks.json'
MD_NAME = 'w179b_checks.md'
MANIFEST_NAME = 'manifest_sha256.json'
POST_NAME = 'manifest_post_run_sha256.json'
TYPING_NAMES = ('w179b_bool_typing_test.json', 'w179b_bool_typing_test.log')
CLONE = C.CLONE
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
DTX = os.path.join(CLONE, 'elsarticle.dtx')
MR = 'manuscript_review'
MAIN_APPE = os.path.join(MR, 'main_e926fb7_appE.tex')
MAIN_9FD_FIXED = os.path.join(MR, 'main_9fd19e7_fixed.tex')
LETTER_V4 = os.path.join(MR, 'response_to_reviewers_v4.tex')
LETTER_V5 = os.path.join(MR, 'response_to_reviewers_v5.tex')
LETTER_V6 = os.path.join(MR, 'response_to_reviewers_v6.tex')
DIFF_FOLLOWUP = os.path.join(MR, 'main_9fd19e7_followup.diff')
DIFF_APPE = os.path.join(MR, 'main_e926fb7_appE.diff')
DIFF_LETTER = os.path.join(MR, 'letter_v5_v6.diff')
CHANGE_LOG = os.path.join(MR, 'W177B_CHANGE_LOG.md')
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
DECLARED_HEAD = '48099ab'
DECLARED_MAIN_SHA = '28008ab0d79b477c4e8d86a0c3a4d1acf9c95ed9c4921f4757f1600cda1c60af'
DECLARED_MAIN_LINES = 2717
DECLARED_LETTER_SHA = '6ecd0ee7f2d5de7a528d84f6ee4fecf71905dc9a887388e66c0d78ca466c036e'
DECLARED_LETTER_LINES = 292
G_HEAD = 'e926fb7'
W178_HEAD = '9fd19e7'
W178_MAIN_SHA = C.DECLARED_MAIN_SHA
W178_LETTER_SHA = C.DECLARED_LETTER_SHA
FZ, FZ_SHA = C.FZ, C.FZ_SHA
W178C_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w178c_letter', 'w178c_letter_checks.json')
W178C_JSON_SHA_PREFIX = '102b16c7'
W178B_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w178b_number_check', 'overleaf_9fd19e7', 'w178b_number_check.json')
W178C_SCRIPT, W178C_SCRIPT_SHA_PREFIX = 'p515_s53_w178c_letter_checks.py', '48e5d8b9'
W178B_SCRIPT = 'p515_s53_w178b_number_check.py'
W153_CERT, W153_SALV = C.W153_CERT, C.W153_SALV
W163 = C.W163
W155_SUM = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w155_a64_cells', 'w155_summary_after_04_e_soh050.json')
SRP1_JSON = C.SRP1_JSON
INSTANCE_3X3 = C.INSTANCE_3X3
EXPORT_T6 = C.EXPORT_T6
F2_SPEC = C.F2_SPEC
SUBMITTED = C.SUBMITTED
HOOKS_W101 = 'p515_s53_w101_settling_continuation_hooks.py'
FORBIDDEN_MODULE_PREFIXES = C.FORBIDDEN_MODULE_PREFIXES
CHECK_FAILURES = []
SCRIPT_NAME = os.path.basename(__file__)
WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
         'fourteen': 14, 'seventeen': 17, 'fifty-five': 55, 'sixty-one': 61, 'single': 1, 'twice': 2, 'doubled': 2}


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


_sha, _git, _read = C._sha, C._git, C._read


def _json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def show(commit, path='main.tex'):
    return _git(['show', f'{commit}:{path}'], cwd=CLONE)


def lines_of(text):
    return text.split('\n')


def norm(s, ci=False):
    s = re.sub(r'\s+', ' ', s.replace('~', ' ')).strip()
    return s.lower() if ci else s


def typeset(lines):
    return W167.strip_comments(lines)


# ======================================================================================================================
#  0. identity, concurrency
# ======================================================================================================================
def refuse_concurrent():
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me = os.getpid()
    others = []
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(os.path.basename(a) == SCRIPT_NAME for a in argv[1:]):
            others.append(ln.strip())
    if others:
        print(f'REFUSED: another copy of {SCRIPT_NAME} is running: {others}', flush=True)
        sys.exit(1)


def _pm_lines(diff_text):
    """The '-' and '+' content lines of a unified diff (headers excluded), as two lists in order."""
    minus, plus = [], []
    for ln in diff_text.split('\n'):
        if ln.startswith(('---', '+++', '@@')):
            continue
        if ln.startswith('-'):
            minus.append(ln[1:])
        elif ln.startswith('+'):
            plus.append(ln[1:])
    return minus, plus


def _unified(a_text, b_text):
    return '\n'.join(difflib.unified_diff(a_text.split('\n'), b_text.split('\n'), lineterm='', n=0))


def identity():
    head = _git(['rev-parse', '--short', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    main_text, letter_text = _read(MAIN_TEX), _read(LETTER)
    revs = _git(['rev-list', f'{G_HEAD}..HEAD'], cwd=CLONE).split()
    m9, me, mh = show(W178_HEAD), show(G_HEAD), show('HEAD')
    l9, le, lh = (show(c, 'response_to_reviewers_draft.tex') for c in (W178_HEAD, G_HEAD, 'HEAD'))
    sha = lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest()  # noqa: E731
    d = {'clone_head': head, 'clone_head_full': _git(['rev-parse', 'HEAD'], cwd=CLONE).strip(),
         'clone_head_subject': _git(['log', '-1', '--format=%s'], cwd=CLONE).strip(),
         'clone_head_date': _git(['log', '-1', '--format=%cI'], cwd=CLONE).strip(),
         'commits_after_e926fb7': revs, 'clone_tracked_modified': status,
         'main_sha256': _sha(MAIN_TEX), 'main_lines': main_text.count('\n'),
         'letter_sha256': _sha(LETTER), 'letter_lines': letter_text.count('\n'),
         'blob_sha256': {f'{c}:main.tex': sha(t) for c, t in ((W178_HEAD, m9), (G_HEAD, me), ('HEAD', mh))}
         | {f'{c}:letter': sha(t) for c, t in ((W178_HEAD, l9), (G_HEAD, le), ('HEAD', lh))},
         'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip()}
    diffs = {}
    for name, a, b, committed in (('main 9fd19e7 -> e926fb7 vs main_9fd19e7_followup.diff', m9, me, DIFF_FOLLOWUP),
                                  ('main e926fb7 -> HEAD vs main_e926fb7_appE.diff', me, mh, DIFF_APPE),
                                  ('letter e926fb7 -> HEAD vs letter_v5_v6.diff', le, lh, DIFF_LETTER)):
        got = _pm_lines(_unified(a, b))
        want = _pm_lines(_read(committed))
        # net multisets (a line both removed and re-added is alignment, not change: difflib and diff(1) align '}' / ''
        # lines differently)
        net = lambda pm: (Counter(pm[0]) - Counter(pm[1]), Counter(pm[1]) - Counter(pm[0]))  # noqa: E731
        diffs[name] = {'minus_equal': net(got)[0] == net(want)[0], 'plus_equal': net(got)[1] == net(want)[1],
                       'raw_sorted_equal': sorted(got[0]) == sorted(want[0]) and sorted(got[1]) == sorted(want[1]),
                       'n_minus': len(got[0]), 'n_plus': len(got[1]),
                       'n_hunks_committed': sum(1 for ln in _read(committed).split('\n') if ln.startswith('@@'))}
    v45 = _pm_lines(_unified(_read(LETTER_V4), _read(LETTER_V5)))
    diffs['letter v4 -> v5 (manuscript_review copies)'] = {'n_minus': len(v45[0]), 'n_plus': len(v45[1])}
    d['diffs'] = diffs
    checks = {'head': head == DECLARED_HEAD, 'main_sha': d['main_sha256'] == DECLARED_MAIN_SHA,
              'main_lines': d['main_lines'] == DECLARED_MAIN_LINES, 'letter_sha': d['letter_sha256'] == DECLARED_LETTER_SHA,
              'letter_lines': d['letter_lines'] == DECLARED_LETTER_LINES,
              'main_equals_main_e926fb7_appE': _sha(MAIN_TEX) == _sha(MAIN_APPE),
              'letter_equals_v6': _sha(LETTER) == _sha(LETTER_V6),
              'e926fb7_main_equals_main_9fd19e7_fixed': sha(me) == _sha(MAIN_9FD_FIXED),
              'e926fb7_letter_equals_v5': sha(le) == _sha(LETTER_V5),
              '9fd19e7_main_is_W178s': sha(m9) == W178_MAIN_SHA, '9fd19e7_letter_is_W178s': sha(l9) == W178_LETTER_SHA,
              '9fd19e7_letter_equals_v4': sha(l9) == _sha(LETTER_V4),
              'head_blobs_equal_working_copy': sha(mh) == _sha(MAIN_TEX) and sha(lh) == _sha(LETTER),
              'one_commit_after_e926fb7': len(revs) == 1 and revs[0].startswith(DECLARED_HEAD),
              'clone_clean': status == '', 'frozen_json': _sha(FZ) == FZ_SHA,
              'w178c_json': _sha(W178C_JSON).startswith(W178C_JSON_SHA_PREFIX),
              'w178c_script': _sha(W178C_SCRIPT).startswith(W178C_SCRIPT_SHA_PREFIX)}
    for name, v in diffs.items():
        if 'minus_equal' in v:
            checks[f'diff: {name}'] = v['minus_equal'] and v['plus_equal']
    d['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'identity check {k} failed')
    return d


# ======================================================================================================================
#  1. capture-path checklist (asserted before executing)
# ======================================================================================================================
REQUIRED_CAPTURES = {
    'static numbering at 48099ab and e926fb7; labels outside Appendix E unchanged; E.20-E.24': 'task_numbering',
    'elsarticle \\appendix reading (table counter)': 'task_elsarticle',
    'remaining appendix-pointer prefixes and CONFIRM comments': 'task_prefixes_confirm',
    '(b) 4.6/4.7/4.8 numbers (every token declared)': 'task_main_numbers',
    '(b) 4.7/4.8 moved and kept items': 'task_moved_kept',
    '(b) 4.6/4.7/4.8 claims rated': 'task_main_claims',
    '(b) letter R2.2, R3.1, Further-changes bullets: numbers and claims': 'task_letter_changes',
    '(b) every letter v6 pointer: verdict, number at HEAD, number at 9fd19e7, needles': 'task_pointers',
    '(c) G1-G12 hunks applied at e926fb7 and 48099ab': 'task_g_hunks',
    '(c) letter v5 edits applied': 'task_letter_v5',
    '(c) W178 rows re-scored': 'task_w178_rows',
    'guards (solve 0, pickle 0, no production import)': 'verify_guards',
}
REQUIRED_C_SYMBOLS = ('numbering_adjusted', '_target', 'GUARD', 'PICKLE_COUNTS', 'EXPORT_T6', 'F2_SPEC')
REQUIRED_R_SYMBOLS = ('numbering', 'letter_pointers', '_resolve', '_region_text', 'cite_order', '_blank_cmd', 'PTR', 'PTR_APP')


def capture_checklist():
    missing = []
    g = globals()
    for q, fn in REQUIRED_CAPTURES.items():
        if not callable(g.get(fn)):
            missing.append(f'capture function {fn} for "{q}"')
    for s in REQUIRED_C_SYMBOLS:
        if not hasattr(C, s):
            missing.append(f'W178c symbol C.{s}')
    for s in REQUIRED_R_SYMBOLS:
        if not hasattr(R, s):
            missing.append(f'W177b-3\' symbol R.{s}')
    for p in [MAIN_TEX, LETTER, DTX, MAIN_APPE, MAIN_9FD_FIXED, LETTER_V4, LETTER_V5, LETTER_V6, DIFF_FOLLOWUP, DIFF_APPE,
              DIFF_LETTER, CHANGE_LOG, BRIEF, FZ, W178C_JSON, W178B_JSON, W178C_SCRIPT, W178B_SCRIPT, W153_CERT, W153_SALV,
              W163, W155_SUM, SRP1_JSON, INSTANCE_3X3, EXPORT_T6, F2_SPEC, SUBMITTED, HOOKS_W101]:
        if not os.path.isfile(p):
            missing.append(f'input {p}')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'c_symbols': list(REQUIRED_C_SYMBOLS), 'r_symbols': list(REQUIRED_R_SYMBOLS),
            'all_present': True}


# ======================================================================================================================
#  sources
# ======================================================================================================================
class Src:
    def __init__(self):
        self.fz = _json(FZ)
        self.t = self.fz['tables']
        self.claims = {c['claim_id']: c for c in self.t['claims']}
        self.yl = self.t['year_ladder']
        self.w153 = _json(W153_CERT)['result']
        self.w153s = _json(W153_SALV)
        self.w163 = _json(W163)
        self.w163_checks = {r['id']: r for r in self.w163['checks_in_v4']}
        self.w155 = _json(W155_SUM)
        self.srp1 = _json(SRP1_JSON)
        self.i33 = _json(INSTANCE_3X3)
        self.f2 = _json(F2_SPEC)
        self.w178c = _json(W178C_JSON)
        self.w178b = _json(W178B_JSON)

    def w163_ok(self, cid):
        r = self.w163_checks.get(cid)
        return r is not None and r['status'] == 'match'

    def flex_multiplier_f2(self):
        hits = []

        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == 'flex_price_multiplier':
                        hits.append(v)
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(self.f2)
        return sorted(set(hits))


def fmt_like(tok, value):
    """value formatted at the written precision of tok (decimals after '.', thousands separators ignored)."""
    t = tok.replace(',', '')
    dp = len(t.split('.')[1]) if '.' in t else 0
    return f'{value:.{dp}f}'


def eq_written(tok, value):
    t = tok.replace(',', '')
    try:
        return float(fmt_like(tok, value)) == float(t)
    except ValueError:
        return False


# ======================================================================================================================
#  2. numbering, elsarticle, prefixes, CONFIRM comments
# ======================================================================================================================
APPE_LABEL = re.compile(r'^(app:results|tab:appE_)')


def task_numbering(main_lines, g_lines):
    before = list(C.CHECK_FAILURES)
    raw, adj = C.numbering_adjusted(main_lines)
    raw_g, adj_g = C.numbering_adjusted(g_lines)
    if C.CHECK_FAILURES != before:
        _fail(f'W178c numbering_adjusted reported: {C.CHECK_FAILURES[len(before):]}')
    changed, gone, new = [], [], []
    for k, v in adj_g['labels'].items():
        if k not in adj['labels']:
            gone.append({'label': k, 'number_e926fb7': v['number']})
        elif adj['labels'][k]['number'] != v['number'] and not APPE_LABEL.match(k):
            changed.append({'label': k, 'e926fb7': v['number'], 'head': adj['labels'][k]['number']})
    for k, v in adj['labels'].items():
        if k not in adj_g['labels']:
            new.append({'label': k, 'kind': v['kind'], 'number': v['number'], 'line': v['line']})
    outside_e = [k for k in adj_g['labels'] if k in adj['labels'] and not APPE_LABEL.match(k)]
    appe_tables = {k: adj['labels'][k]['number'] for k in adj['labels'] if k.startswith('tab:appE_')}
    appe_sections = {k: adj['labels'][k]['number'] for k in adj['labels'] if k.startswith('app:results')}
    refs = C.task_refs(main_lines, adj)
    out = {'n_labels_head': len(adj['labels']), 'n_labels_e926fb7': len(adj_g['labels']),
           'n_labels_outside_appendix_E_compared': len(outside_e), 'labels_changed_outside_appendix_E': changed,
           'labels_removed': gone, 'labels_new': new, 'appendix_E_tables': appe_tables,
           'appendix_E_sections': appe_sections, 'counter_adjustments': adj['counter_adjustments'],
           'refs_static_head': refs,
           'appendix_tables_all': [{'number': o['number'], 'line': o['line'], 'kind': o['kind']} for o in adj['objects']
                                   if o['kind'].startswith('table') and o['number'] and o['number'][0].isalpha()]}
    expect_tabs = {'tab:appE_claims': 'E.20', 'tab:appE_evaluations': 'E.21', 'tab:appE_neighbours': 'E.22',
                   'tab:appE_benchmark': 'E.23', 'tab:appE_gap': 'E.24'}
    out['appendix_E_tables_are_E20_E24'] = appe_tables == expect_tabs
    if changed or not out['appendix_E_tables_are_E20_E24'] or refs['undefined_refs'] or refs['duplicate_labels']:
        _fail(f'numbering: changed {changed}; E tables {appe_tables}; undefined {refs["undefined_refs"]}; '
              f'duplicates {refs["duplicate_labels"]}')
    return raw, adj, out


def task_elsarticle():
    t = _read(DTX).split('\n')
    i = next(k for k, ln in enumerate(t) if ln.startswith(r'\renewcommand\appendix{'))
    block = t[i:i + 12]
    reset_table = [k + 1 for k, ln in enumerate(t) if re.search(r'\\@addtoreset\{table\}', ln)]
    out = {'dtx_appendix_block_first_line': i + 1, 'dtx_appendix_block': block,
           'thetable_redefined_as_section_letter_dot_arabic': any(r'\gdef\thetable{\@Alph\c@section.\arabic{table}}' in ln for ln in block),
           'table_counter_reset_anywhere_in_dtx': bool(reset_table), 'addtoreset_table_lines': reset_table,
           'thesubsection_redefined_in_dtx': any(r'\thesubsection' in ln for ln in t),
           'reading': ('\\appendix sets \\thetable to <section letter>.<table counter> and resets section, subsection and '
                       'equation only; the table counter is never reset (no \\@addtoreset{table}), so the appendix tables '
                       'continue the global count: D.16-D.19, then E.20-E.24. \\thesubsection is not redefined, so it '
                       'is article\'s \\thesection.\\arabic{subsection} = "Appendix A.4": a "Subsection~\\ref{app:...}" '
                       'prints "Subsection Appendix A.4" (static reading; nothing compiled)')}
    if not out['thetable_redefined_as_section_letter_dot_arabic'] or out['table_counter_reset_anywhere_in_dtx']:
        _fail(f'elsarticle reading not as expected: {out}')
    return out


PREFIX = re.compile(r'(?<![\w])(Appendix|Appendices|Section|Sections|Subsection|Subsections)(~|\s+)\\ref\{(app:[^}]*)\}')


def task_prefixes_confirm(main_lines, g_lines, w_lines):
    def prefixes(lines):
        cl = typeset(lines)
        return [{'line': i, 'text': m.group(0), 'label': m.group(3)} for i, ln in enumerate(cl, 1) for m in PREFIX.finditer(ln)]

    def confirms(lines):
        return [{'line': i, 'tag': (re.search(r'CONFIRM\s*(?:—|--|-)\s*(W\d+)', ln) or re.search(r'(CONFIRM)', ln)).group(1),
                 'text': ln.strip()[:160]} for i, ln in enumerate(lines, 1) if 'CONFIRM' in ln]
    out = {'appendix_ref_prefixes': {'9fd19e7': prefixes(w_lines), 'e926fb7': prefixes(g_lines), 'head': prefixes(main_lines)},
           'confirm_comments': {'9fd19e7': confirms(w_lines), 'e926fb7': confirms(g_lines), 'head': confirms(main_lines)}}
    hc = out['confirm_comments']['head']
    out['confirm_head_counts'] = dict(Counter(c['tag'] for c in hc))
    out['confirm_head_typeset'] = [c for c in hc if not main_lines[c['line'] - 1].lstrip().startswith('%')]
    if out['confirm_head_typeset']:
        _fail(f'a CONFIRM marker is typeset: {out["confirm_head_typeset"]}')
    return out


# ======================================================================================================================
#  3. (b) main text: numbers
# ======================================================================================================================
NUMW = r'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|seventeen|twenty|' \
       r'twenty-five|forty-two|fifty-five|sixty-one|single|twice|doubled'
TOK = re.compile(r'(?<![\w.\\])(-?\d+(?:[.,]\d+)*)(?![\w])|(?<![\w-])(' + NUMW + r')(?![\w-])', re.I)


def clean_main(s):
    s = re.sub(r'(?<!\\)%.*', '', s)
    s = re.sub(r'\\(?:eq|auto|c|C)?ref\{[^}]*\}', ' ', s)
    s = re.sub(r'\\(?:label|cite|textcolor|includegraphics)\{[^}]*\}', ' ', s)
    s = s.replace('{,}', ',').replace('$-$', '-').replace('--', ' \u2013 ').replace('\\,', ' ').replace('~', ' ')
    return s


def tokens_of(s):
    return [(m.group(1) or m.group(2)) for m in TOK.finditer(s)]


def find_line(lines, frag):
    hits = [i for i, ln in enumerate(lines, 1) if frag in ln]
    if len(hits) != 1:
        _fail(f'fragment {frag!r} found on {len(hits)} lines (expected 1)')
        return None
    return hits[0]


def _decl(tok, status, expected=None, value=None, source=None, note=None):
    return {'token': tok, 'status': status, 'expected': expected, 'value': value, 'source': source, 'note': note}


def V(tok, value, source, note=None):
    """a value at the written precision"""
    return _decl(tok, 'match' if eq_written(tok, value) else 'MISMATCH', fmt_like(tok, value), value, source, note)


def N(tok, n, source, note=None):
    """a count or an integer label"""
    got = WORDS.get(tok.lower(), None)
    got = got if got is not None else (int(tok) if re.fullmatch(r'-?\d+', tok) else None)
    return _decl(tok, 'match' if got == n else 'MISMATCH', str(n), n, source, note)


def Dv(tok, value, source, note=None):
    st = 'derived, holds' if eq_written(tok, value) else 'MISMATCH'
    return _decl(tok, st, fmt_like(tok, value), value, source, note)


def K(S, tok, cid, note=None):
    """carried: unchanged text, checked by W163 against the frozen tables"""
    return _decl(tok, 'carried, W163 match' if S.w163_ok(cid) else 'MISMATCH', None, None, f'W163 check {cid}', note)


def L_(tok, why):
    return _decl(tok, 'language (not a quantity)', None, None, why)


def main_number_declarations(S):
    cl = S.claims
    yl = S.yl
    py = yl['per_year']
    head = cl['CHECK:headline_V_minus_I_settled']
    g_ids = ['G:n7_2h_e1_y2030:gross', 'G:n7_2h_e1_y2035:gross', 'G:n7_4h_e2_y2030:gross', 'G:n7_4h_e2_y2035:gross']
    g_vals = [v for g in g_ids for v in (cl[g]['d_gross'], cl[g]['d_net'])]
    g_plans = sorted({tuple(cl[g]['instance'][cl[g]['other_cell']]['candidate_canonical']['nodes']['7']) for g in g_ids})
    g_years = sorted({cl[g]['instance'][cl[g]['other_cell']]['candidate_canonical']['investment_year'] for g in g_ids})
    m_net = {'2025': -head['d_net'], '2030': py['2030']['M_net_w153_form'], '2035': py['2035']['M_net_w153_form']}
    f2 = cl['L:y2030__n5_p0.25_e0.5__n7_p1_e3']
    tot = S.w153['totals']
    cells = S.w153['cells']
    ats = S.t['at_or_above_0_95_tau']
    reg = ats['registry']
    n46 = sum(1 for r_ in reg if not r_.get('superseded') and not r_.get('twin_of'))
    pcm = S.w163['post_certification_movement']['over_tau']
    reb = S.w163['reference_evaluations_beside']
    s_ref = [reb['x0']['s_signed'], reb['n7_4h_e1']['s_signed']]
    nc_cells = [c for c in cells if c['non_clean_cycles_after_N']]
    nc_blocks = Counter(b.split('|', 1)[1] for c in cells for d_ in c['non_clean_after_N_detail'] for b in d_['blocks'])
    dz = {c['cell']: c for c in S.t['dead_zone']['cells']}
    dz_m2 = ['j_5f3cccb4', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5', 'l_0ee93aca', 'ref:5ca4f86c', 'ref:e28de4ac']
    t_m2 = [dz[c]['t_sum_at_cap'] for c in dz_m2]
    wall = sum(c['wall_s_parent'] for c in cells)
    cyc = sum(c['cycles_recorded'] for c in cells)
    med = statistics.median(c['wall_s_per_cycle'] for c in cells)
    gsrc = 'frozen claims ' + ', '.join(g_ids) + ' (d_gross and d_net)'
    decl = {
        'S46a': ('Deferring the reference unit does not make it pay', [
            N('2030', 2030, 'frozen year_ladder.per_year'), V('68.5', py['2030']['M_gross_w153_form'] / 1e3, 'frozen year_ladder.per_year.2030.M_gross_w153_form'),
            N('2035', 2035, 'frozen year_ladder.per_year'), V('111.0', py['2035']['M_gross_w153_form'] / 1e3, 'frozen year_ladder.per_year.2035.M_gross_w153_form'),
            V('11.8', py['2030']['salvage_last'] / 1e3, 'frozen year_ladder.per_year.2030.salvage_last'), N('2030', 2030, 'idem'),
            V('56.5', py['2035']['salvage_last'] / 1e3, 'frozen year_ladder.per_year.2035.salvage_last'), N('2035', 2035, 'idem'),
            N('two', 2, 'the two deferral years 2030, 2035'), V('42.5', yl['D_gross'] / 1e3, 'frozen year_ladder.D_gross'),
            V('3.46', yl['gross_v6']['multiple'], 'frozen year_ladder.gross_v6.multiple'),
            V('12.3', yl['gross_v6']['threshold'] / 1e3, 'frozen year_ladder.gross_v6.threshold'),
            V('-2.3', yl['D_net'] / 1e3, 'frozen year_ladder.D_net'), V('0.19', yl['net_v6']['multiple'], 'frozen year_ladder.net_v6.multiple'),
            V('54.5', min(m_net.values()) / 1e3, 'min M net over 2025 (CHECK headline, salvage 0), 2030, 2035 (frozen per_year M_net_w153_form)'),
            V('64.4', max(m_net.values()) / 1e3, 'max of the same three'),
            N('two', len(g_plans), f'distinct node-7 plans of the four G claims: {g_plans}', 'NEW (A78 G-rows sentence)'),
            N('7', 7, 'G claims: node 7 only', 'NEW'), N('2030', g_years[0], f'G claim years {g_years}', 'NEW'),
            N('2035', g_years[-1], 'idem', 'NEW'),
            V('0.5', g_plans[0][0], 'G plan P (MVA) of n7_2h_e1', 'NEW'), V('1', g_plans[0][1], 'G plan E (MWh) of n7_2h_e1', 'NEW'),
            V('0.5', g_plans[1][0], 'G plan P of n7_4h_e2', 'NEW'), V('2', g_plans[1][1], 'G plan E of n7_4h_e2', 'NEW'),
            V('85.1', min(g_vals) / 1e3, gsrc + ': minimum', 'NEW'), V('223.6', max(g_vals) / 1e3, gsrc + ': maximum', 'NEW'),
            N('17', 17, 'Table E.20 row of G:n7_2h_e1_y2030 (parsed; see task_main_claims C46-1)', 'NEW'),
            N('20', 20, 'Table E.20 row of G:n7_4h_e2_y2035 (parsed)', 'NEW')]),
        'S46b': ('Across the 55 comparisons of Table', [
            N('55', 55, 'frozen claims: 60 less the four G :net duplicates and CHECK:headline (= distinct comparisons with a net figure); '
                        'Table E.20 rows with a net Delta (parsed)', 'NEW'),
            N('one', 1, 'verdict changes gross -> net over the 55 (computed in task_main_claims C46-2)'),
            N('twice', int(S.flex_multiplier_f2()[-1]), f'{F2_SPEC}: flex_price_multiplier'),
            L_('one', 'one of its unit neighbours (a claim, C46-3)'),
            N('34', 34, 'Table E.20 row of L:y2030__n5_p0.25_e0.5__n7_p1_e3 (parsed)', 'NEW'),
            V('0.95', f2['gross_multiple'], 'frozen claims L:y2030__n5_p0.25_e0.5__n7_p1_e3 gross_multiple'),
            V('1.22', f2['net_multiple'], 'idem net_multiple'),
            V('42.5', yl['D_gross'] / 1e3, 'frozen year_ladder.D_gross', 'NEW'), V('-2.3', yl['D_net'] / 1e3, 'frozen year_ladder.D_net', 'NEW')]),
        'S47b': ('Of the 42 single-scenario evaluations run under the certification rule', [
            N('42', tot['n_cells'], 'W153 totals n_cells'), N('32', tot['n_certified'], 'W153 totals n_certified'),
            N('10', tot['n_uncertified'], 'W153 totals n_uncertified'),
            N('5', tot['uncertified_by_cause']['gap clause']['n'], 'W153 uncertified_by_cause gap clause'),
            N('2', tot['uncertified_by_cause']['lapse reset']['n'], 'W153 lapse reset'),
            N('3', tot['uncertified_by_cause']['growth test']['n'], 'W153 growth test'),
            N('Ten', ats['n_counted'], 'frozen at_or_above_0_95_tau.n_counted'),
            N('46', n46, 'frozen at_or_above_0_95_tau.registry: entries neither superseded nor a twin'),
            N('5', 5, 'the 0.95 tau flag threshold = within 5 % of tau (frozen constants.FLAG_RANGE_OVER_TAU 0.95)'),
            V('0.9', max(abs(v) for v in pcm.values()), 'W163 post_certification_movement.over_tau, largest |movement|/tau (0.882; "at most 0.9")'),
            V('15', min(s_ref) / 1e3, 'W163 reference_evaluations_beside x0.s_signed (k EUR)'),
            V('21', max(s_ref) / 1e3, 'W163 reference_evaluations_beside n7_4h_e1.s_signed (k EUR)')]),
        'S47d': ('The 42 evaluations run under the rule took', [
            N('42', tot['n_cells'], 'W153 totals n_cells'), V('220,876', wall, 'W153 cells: sum wall_s_parent'),
            V('8,570', cyc, 'W153 cells: sum cycles_recorded'), V('25.6', med, 'W153 cells: median wall_s_per_cycle'),
            L_('one', 'one evaluation at a time (W163 N44 attestation)'), L_('single', 'single thread (W163 N45; A.8)')]),
        'S48a': ('Degenerate interface duals.', [
            L_('0', 'lower end of the set-valued interval [0, c^flex]'),
            L_('one', 'one fingerprint (a claim, C48-1)'),
            V('-8.8', max(t_m2) / 1e3, 'frozen dead_zone t_sum_at_cap, the seven m = 2 evaluations (5 gap-refused + l_0ee93aca + 2 F2 refs): max'),
            V('-9.4', min(t_m2) / 1e3, 'idem: min'),
            N('three', len(dz['j_5f3cccb4']['share_by_node']), 'interface nodes: frozen dead_zone j_5f3cccb4 share_by_node keys (5, 7, 9)'),
            N('5', 5, 'the 5 MWh evaluation at the base price (d_36686489, n7 1.25/5)'),
            V('1.1', dz['d_36686489']['t_sum_at_cap'] / 1e3, 'frozen dead_zone d_36686489 t_sum_at_cap (+)')]),
        'S48b': ('Solver recoveries and numerical slacks.', [
            N('21', sum(len(c['non_clean_cycles_after_N']) for c in cells), 'W153 cells: non-clean cycles after N, summed'),
            N('10', len(nc_cells), 'W153 cells with non-clean cycles after N'),
            N('2035', 2035, f'W153 non-clean blocks after N, most common: {nc_blocks.most_common(3)}'),
            K(S, '10', 'N33', '10^{-5} p.u.'), K(S, '-5', 'N33', 'exponent of 10^{-5}'), K(S, '62', 'N35'),
            K(S, '3', 'N36', 'about 3 tau'), K(S, '44', 'N37')]),
    }
    checks = {'2035_is_most_common_nonclean_year': nc_blocks.most_common(1)[0][0].startswith('2035|Spring') if nc_blocks else False,
              'nonclean_all_tso': all(b.startswith('TSO') for c in cells for d_ in c['non_clean_after_N_detail'] for b in d_['blocks']),
              'w153_totals_agree': tot.get('non_clean_cycles_after_N_total') == 21 and tot.get('cycles_total', cyc) == cyc}
    return decl, checks


def task_main_numbers(S, main_lines):
    cl = typeset(main_lines)
    decl, aux = main_number_declarations(S)
    rows, counts = [], Counter()
    for seg, (frag, dl) in decl.items():
        ln = find_line(cl, frag)
        if ln is None:
            continue
        toks = tokens_of(clean_main(cl[ln - 1]))
        if [t for t in toks] != [d['token'] for d in dl]:
            _fail(f'{seg} l. {ln}: tokens {toks} != declared {[d["token"] for d in dl]}')
        for k, d in enumerate(dl):
            r_ = dict(d, segment=seg, line=ln, occ=k + 1)
            rows.append(r_)
            counts[d['status']] += 1
            if d['status'] == 'MISMATCH':
                _fail(f'{seg} l. {ln} token {d["token"]}: MISMATCH (expected {d["expected"]}, {d["source"]})')
    for k, v in aux.items():
        if v is not True:
            _fail(f'number auxiliary check {k} failed')
    return {'rows': rows, 'counts': dict(counts), 'n_tokens': len(rows), 'aux_checks': aux,
            'mismatch': [r_ for r_ in rows if r_['status'] == 'MISMATCH']}


# ======================================================================================================================
#  3b. moved and kept items
# ======================================================================================================================
def region(cl, start_label, end_regex):
    s = next(i for i, ln in enumerate(cl) if f'\\label{{{start_label}}}' in ln)
    e = next(i for i in range(s + 1, len(cl)) if re.search(end_regex, cl[i]))
    return '\n'.join(cl[s:e]), s + 1, e


MOVED = [   # (id, item, old fragment (e926fb7, in 4.7/4.8), destination, new fragment at HEAD, how)
    ('M1', 'branch split 24 / 8', '(24 on the oscillatory branch, 8 on the monotone branch)', 'E.6',
     '32 certified, 24 on the oscillatory branch and 8 on the monotone branch', 'reworded'),
    ('M2', 'median certification cycle 174', 'The median certification cycle was 174', 'E.6', 'The median certification cycle was 174', 'verbatim'),
    ('M3', 'median k* - k0 65 (21-89)', 'certification came a median 65 cycles after the first residual pass (range 21--89)', 'E.6',
     'certification came a median 65 cycles after the first residual pass (range 21--89)', 'verbatim'),
    ('M4', 'median range/tau 0.86', 'The median ratio of the window range to $\\tau$ at certification was 0.86', 'E.6',
     'the median ratio of the window range to $\\tau$ at certification was 0.86', 'verbatim (case of the first word)'),
    ('M5', 'the four further evaluations', 'The four further evaluations all certified', 'E.6',
     'The four evaluations added later (32, 33, 60 and 61) all certified', 'reworded (IDs added)'),
    ('M6', 'machine and memory', 'on an Apple M2~Max with 32~GiB of memory under macOS~27.0', 'E.6',
     'on an Apple M2~Max with 32~GiB of memory under macOS~27.0', 'verbatim'),
    ('M7', 'thread caps', 'OMP, MKL, OpenBLAS, vecLib and NumExpr were capped at one thread in every evaluation\'s recorded environment', 'E.6',
     'OMP, MKL, OpenBLAS, vecLib and NumExpr were capped at one thread in every evaluation\'s recorded environment', 'verbatim'),
    ('M8', 'HSL without OpenMP', 'the HSL library is built without OpenMP, so MA97 ran serially', 'E.6',
     'the HSL library is built without OpenMP, so MA97 ran serially', 'verbatim'),
    ('M9', 'node split 55 / 31 / 14 %', 'split about 55\\,\\% at node~5, 31\\,\\% at node~9 and 14\\,\\% at node~7', 'E.5',
     'split about 55\\,\\%, 14\\,\\% and 31\\,\\% among nodes~5, 7 and~9', 'reworded (node order 5, 7, 9)'),
    ('M10', 'power-flow residual ratio', 'stuck near 0.69', 'E.5', 'with the power-flow primal residual at 0.68--0.70 of its tolerance in the last cycle',
     'restated as a range'),
]
KEPT = [   # (id, item, fragment at HEAD in 4.7 / 4.8)
    ('K1', 'counts by cause', '32 certified and 10 did not: 5 were refused by the gap clause, 2 reset by residual lapses and 3 failed by the growth test', '4.7'),
    ('K2', 'the tau statement', 'Ten of the 46 certificates the tables use stop within 5\\,\\% of $\\tau$; the rule therefore bounds each evaluation\'s stopping error at about $\\tau$, not well inside it', '4.7'),
    ('K3', '0.9 tau movement', 'Evaluations continued past a certificate under this rule moved by at most $0.9\\tau$', '4.7'),
    ('K4', '15-21 k EUR movement', 'on the reference evaluations the objective moved by a further 15--21~k\\euro{} after the residual-based stopping rule had certified them', '4.7'),
    ('K5', 'Fig. 9 paragraph', 'Figure~\\ref{fig:res_q_cycle} shows the two outcomes on recorded trajectories', '4.7'),
    ('K6', 'search paragraph', 'The three planning searches of Algorithm~\\ref{alg:shared_ess_planning_mads} made 41 new recourse evaluations in all', '4.7'),
    ('K7', 'wall time', 'The 42 evaluations run under the rule took 220{,}876~s of wall time over 8{,}570 ADMM cycles, a median of 25.6~s per cycle', '4.7'),
    ('K8', 'the fingerprint', 'a priced gap of $-8.8$ to $-9.4$~k\\euro{}', '4.8'),
    ('K9', 'the 5 MWh evaluation, +1.1 k EUR', 'its consensus gap is small ($+1.1$~k\\euro) and of the opposite sign', '4.8'),
]


def task_moved_kept(main_lines, g_lines):
    cl, gl = typeset(main_lines), typeset(g_lines)
    r47, a47, b47 = region(cl, 'subsec:res_certification', r'\\subsection\{\\textcolor\{blue\}\{Limitations\}\}')
    r48, a48, b48 = region(cl, 'subsec:res_limitations', r'^\\section\{')
    g47, _, _ = region(gl, 'subsec:res_certification', r'\\subsection\{\\textcolor\{blue\}\{Limitations\}\}')
    g48, _, _ = region(gl, 'subsec:res_limitations', r'^\\section\{')
    e5, _, _ = region(cl, 'app:results_gap', r'\\subsection\{')
    e6, a6, _ = region(cl, 'app:results_statistics', r'\\section\*\{Declaration')
    dest = {'E.5': e5, 'E.6': e6}
    old_reg = {'E.6': g47, 'E.5': g48}
    new_src = {'E.6': r47, 'E.5': r48}
    moved = []
    for mid, item, old, d, new, how in MOVED:
        row = {'id': mid, 'item': item, 'old_fragment': old, 'destination': d, 'new_fragment': new, 'how': how,
               'old_present_at_e926fb7': norm(old) in norm(old_reg[d]),
               'old_absent_from_head_section': norm(old, True) not in norm(new_src[d], True),
               'new_present_in_destination': norm(new) in norm(dest[d])}
        row['ok'] = row['old_present_at_e926fb7'] and row['old_absent_from_head_section'] and row['new_present_in_destination']
        if not row['ok']:
            _fail(f'moved item {mid}: {row}')
        moved.append(row)
    # the reproducibility paragraph: every sentence of the e926fb7 paragraph present in E.6, absent from 4.7
    gpar = next(ln for ln in gl if '\\textbf{Reproducibility.}' in ln)
    body = norm(gpar.split('\\textbf{Reproducibility.}', 1)[1])
    sents = [s.strip() for s in re.split(r'(?<=[a-z0-9\}\)])\. (?=[A-Z])', body) if s.strip()]
    rep = []
    for s in sents:
        s0 = s.rstrip('.')
        rep.append({'sentence': s0[:120], 'in_E6': norm(s0) in norm(e6), 'in_head_4_7': norm(s0) in norm(r47)})
    rep_ok = all(r_['in_E6'] and not r_['in_head_4_7'] for r_ in rep)
    if not rep_ok:
        _fail(f'reproducibility paragraph not moved whole: {[r_ for r_ in rep if not r_["in_E6"] or r_["in_head_4_7"]]}')
    kept = []
    for kid, item, frag, where in KEPT:
        reg = r47 if where == '4.7' else r48
        row = {'id': kid, 'item': item, 'fragment': frag, 'section': where, 'present_at_head': norm(frag) in norm(reg),
               'verbatim_at_e926fb7': norm(frag) in norm(g47 if where == '4.7' else g48)}
        row['ok'] = row['present_at_head']
        if not row['ok']:
            _fail(f'kept item {kid}: {row}')
        kept.append(row)
    return {'regions_head': {'4.7': [a47, b47], '4.8': [a48, b48], 'E.6_first_line': a6}, 'moved': moved,
            'reproducibility_sentences': rep, 'reproducibility_moved_whole': rep_ok, 'kept': kept,
            'n_moved_ok': sum(r_['ok'] for r_ in moved), 'n_kept_ok': sum(r_['ok'] for r_ in kept)}


# ======================================================================================================================
#  3c. (b) claims, with Table E.20 / E.21 / E.24 parsed
# ======================================================================================================================
def parse_longtable_rows(cl, label, ncols):
    s = next(i for i, ln in enumerate(cl) if f'\\label{{{label}}}' in ln)
    e = next(i for i in range(s, len(cl)) if '\\end{longtable}' in cl[i] or '\\end{table}' in cl[i])
    rows = {}
    for i in range(s, e):
        ln = cl[i].strip()
        m = re.match(r'^(\d+) & ', ln)
        if m and ln.endswith('\\\\'):
            cells = [c.strip() for c in ln[:-2].split(' & ')]
            if len(cells) != ncols:
                _fail(f'{label} l. {i + 1}: {len(cells)} cells, expected {ncols}')
            rows[int(m.group(1))] = {'line': i + 1, 'cells': cells}
    return rows


def kx(v):
    return float(v.replace('$-$', '-').replace('\\,', '').replace(',', ''))


def task_main_claims(S, main_lines):
    cl = typeset(main_lines)
    e20 = parse_longtable_rows(cl, 'tab:appE_claims', 10)
    e21 = parse_longtable_rows(cl, 'tab:appE_evaluations', 14)
    claims = S.claims
    distinct = [c for cid, c in claims.items() if not cid.endswith(':net') and not cid.startswith('CHECK:')]
    isdet = lambda v: v == 'determinate'  # noqa: E731
    sign_changes = [c['claim_id'] for c in distinct if (c['d_gross'] > 0) != (c['d_net'] > 0)]
    verdict_changes = [c['claim_id'] for c in distinct if isdet(c['gross_verdict']) != isdet(c['net_verdict_recomputed_here'])]
    verdict_label_changes = [c['claim_id'] for c in distinct
                             if c['gross_verdict'] != c['net_verdict_recomputed_here']]
    w153_disagree = [c['claim_id'] for c in distinct if c['net_verdict_recomputed_here'] != c['net_verdict_W153']]
    e20_net_rows = [n for n, r_ in e20.items() if r_['cells'][7] != '---']
    e20_gross_only = [n for n, r_ in e20.items() if r_['cells'][7] == '---']

    def row_matches(n, cid):
        c = claims[cid]
        cells = e20[n]['cells']
        exp = [fmt_like('0.0', c['d_gross'] / 1e3), fmt_like('0.0', c['gross_threshold_or_bar'] / 1e3),
               fmt_like('0.00', c['gross_multiple']), fmt_like('0.0', c['d_net'] / 1e3), fmt_like('0.00', c['net_multiple'])]
        got = [f'{kx(cells[3]):.1f}', f'{kx(cells[4]):.1f}', f'{kx(cells[5]):.2f}', f'{kx(cells[7]):.1f}', f'{kx(cells[8]):.2f}']
        return {'row': n, 'claim': cid, 'line': e20[n]['line'], 'label': cells[1], 'eval': cells[2], 'expected': exp, 'printed': got,
                'verdicts': [cells[6], cells[9]], 'match': exp == got}
    g_map = {17: 'G:n7_2h_e1_y2030:gross', 18: 'G:n7_2h_e1_y2035:gross', 19: 'G:n7_4h_e2_y2030:gross', 20: 'G:n7_4h_e2_y2035:gross'}
    g_rows = [row_matches(n, cid) for n, cid in g_map.items()]
    r34 = row_matches(34, 'L:y2030__n5_p0.25_e0.5__n7_p1_e3')
    g_labels_ok = [r_['label'] for r_ in g_rows] == ['7: 0.5/1, 2030', '7: 0.5/1, 2035', '7: 0.5/2, 2030', '7: 0.5/2, 2035']
    g_det = all(r_['verdicts'] == ['det.', 'det.'] for r_ in g_rows)
    yl_rows = [n for n, r_ in e20.items() if re.search(r'0\.25/1, 20(30|35)', r_['cells'][1])]
    nc = {n: r_['cells'][11] for n, r_ in e21.items()}
    nc_pos = {n: int(v) for n, v in nc.items() if v not in ('---', '0')}
    k_end = {n: r_['cells'][6] for n, r_ in e21.items()}
    dz = {c['cell']: c for c in S.t['dead_zone']['cells']}
    e24 = parse_longtable_rows(cl, 'tab:appE_gap', 11)  # a tabular; the same row parser applies
    share_missing = [n for n, r_ in e24.items() if r_['cells'][6] == '---']
    sec47 = norm(region(cl, 'subsec:res_certification', r'\\subsection\{\\textcolor\{blue\}\{Limitations\}\}')[0])
    e6 = norm(region(cl, 'app:results_statistics', r'\\section\*\{Declaration')[0])
    rows = []

    def add(cid, where, fragment, rating, basis, evidence, expect_true=True):
        ok = all(v is True for v in evidence.values()) if expect_true else True
        if expect_true and not ok:
            _fail(f'claim {cid}: evidence failed {evidence}')
        rows.append({'id': cid, 'where': where, 'fragment': fragment, 'rating': rating, 'basis': basis,
                     'evidence': evidence, 'evidence_ok': ok})
    add('C46-1', '4.6 l. 1637', 'The same holds for two larger plans at node~7 ... cost 85.1--223.6 k EUR more ..., gross or net of '
        'salvage, and every difference is determinate (Table E.20, rows 17--20)', 'consistent',
        'Rows 17-20 of Table E.20 are the four G claims (node 7, 0.5/1 and 0.5/2, 2030 and 2035); printed gross / net / multiples '
        'equal the frozen claims at 1 dp / 2 dp; all eight verdicts determinate; range 85.1 (2035 0.5/1 net) to 223.6 (2035 0.5/2 '
        'gross). "The same holds" refers to the previous sentence (net of salvage the unit costs more in every year): the G '
        'plans also cost more, gross and net.',
        {'rows_17_20_are_G_plans': g_labels_ok, 'rows_17_20_match_frozen': all(r_['match'] for r_ in g_rows),
         'rows_17_20_all_det': g_det,
         'frozen_G_all_determinate': all(claims[c]['gross_verdict'] == 'determinate' and claims[c]['net_verdict_recomputed_here'] == 'determinate'
                                         for c in g_map.values()),
         'G_CONFIRM_comment_removed': not any('G rows of the frozen T1' in ln for ln in main_lines)})
    add('C46-2', '4.6 l. 1681', 'Across the 55 comparisons of Table E.20 reported both gross and net of salvage, applying the salvage '
        'credit changes no sign and one verdict', 'consistent',
        f'55 distinct frozen comparisons with a net figure (60 less 4 G :net duplicates less CHECK:headline); Table E.20 prints a net '
        f'Delta on {len(e20_net_rows)} rows (gross only: rows {e20_gross_only}); sign changes {sign_changes}; determinacy changes '
        f'{verdict_changes}; W153 net verdicts agree with the recomputed ones on every row ({w153_disagree} differ).',
        {'n_distinct_55': len(distinct) == 55, 'e20_net_rows_55': len(e20_net_rows) == 55, 'e20_gross_only_rows_22_52_58': e20_gross_only == [22, 52, 58],
         'no_sign_change': sign_changes == [], 'one_verdict_change': verdict_changes == ['L:y2030__n5_p0.25_e0.5__n7_p1_e3'],
         'w153_n_claims_60': S.w153s['result']['totals']['n_claims'] == 60,
         'w153_n_sign_differs_0': S.w153s['result']['totals']['n_sign_differs'] == 0,
         'w153_verdict_differs_is_row34': S.w153s['result']['totals']['claims_verdict_differs'] == ['L:y2030__n5_p0.25_e0.5__n7_p1_e3']})
    add('C46-3', '4.6 l. 1681', '... with one of its unit neighbours (row 34), within the uncertified bar in gross terms (0.95 times) and '
        'determinate net of salvage (1.22 times), in the returned plan\'s favour', 'consistent',
        'Row 34 = frozen L:y2030__n5_p0.25_e0.5__n7_p1_e3 (returned plan ref:5ca4f86c = 2030 n5 0.25/0.5 + n7 1/3.5; neighbour n7 1/3, '
        'one lattice step in E): gross 26.4 k EUR, 0.947x, "within the uncertified bar"; net 34.1 k EUR, 1.223x, determinate; Delta = '
        'F(neighbour) - F(returned) > 0, so the returned plan costs less.',
        {'row34_match_frozen': r34['match'], 'row34_label': r34['label'] == '5: 0.25/0.5; 7: 1/3',
         'row34_verdicts': r34['verdicts'] == ['w.b.', 'det.'],
         'gross_verdict_within_bar': claims['L:y2030__n5_p0.25_e0.5__n7_p1_e3']['gross_verdict'] == 'within the uncertified bar',
         'positive_delta': claims['L:y2030__n5_p0.25_e0.5__n7_p1_e3']['d_gross'] > 0})
    add('C46-4', '4.6 l. 1681', 'The investment-year comparison above is not among them; on it the credit changes both the sign and the '
        'verdict, from a determinate 42.5 k EUR in gross terms to -2.3 k EUR, within resolution, net of salvage', 'consistent',
        'No row of Table E.20 is the reference unit in 2030 or 2035 (the year pair is T4, Table 14); frozen year_ladder D_gross '
        '+42,458.65 (3.46x, determinate), D_net -2,286.25 (0.19x, within resolution): sign and verdict both change. Resolves the '
        'W178b note on 4.6-2 ("its sign also flips ... not stated").',
        {'year_pair_not_in_E20': yl_rows == [], 'gross_det': S.yl['gross_v6']['verdict'] == 'determinate',
         'net_within': S.yl['net_v6']['verdict'] == 'within resolution', 'sign_flip': S.yl['D_gross'] > 0 > S.yl['D_net']})
    add('C47-1', '4.7 l. 1690', 'Every comparison and every evaluation behind this section is listed in Appendix E, together with the '
        'full certification statistics and the reproducibility record', 'L',
        'Table E.20 lists 58 comparisons, Table E.21 the 61 evaluations behind them and the Section 4 tables, E.6 the statistics and '
        'the reproducibility paragraph. Not listed: the 41 search evaluations of the third 4.7 paragraph (production exit, compared '
        'at sigma_Q) - Appendix E names no search evaluation. Wording; no number or verdict changes.',
        {'E21_has_61_rows': sorted(e21) == list(range(1, 62)), 'E20_has_58_rows': sorted(e20) == list(range(1, 59)),
         'E6_statistics_and_reproducibility': ('certification statistics' in e6.lower() and 'reproducibility' in e6.lower()),
         '41_search_evaluations_in_4_7': '41 new recourse evaluations' in sec47})
    add('C47-2', '4.7 l. 1707', 'one evaluation at a time on a single thread of an Apple M2 Max (E.6)', 'consistent',
        'E.6 now carries the machine, memory, OS, thread caps and the serial MA97 statement verbatim (moved items M6-M8).',
        {'E6_machine': 'apple m2 max' in e6.lower(), 'E6_single_threaded': 'single-threaded' in e6})
    add('C48-1', '4.8 l. 1716', 'they share one fingerprint, a priced gap of -8.8 to -9.4 k EUR split in the same proportions among the '
        'three interface nodes (Table E.24)', 'L',
        f'Table E.24 prints the split for evaluations 39, 50, 51, 45 and 42 (55.2-55.3 / 13.7 / 31.0-31.1 %) and "---" for '
        f'the two F2 references (evaluations 40 and 41; frozen dead_zone ref:5ca4f86c, '
        f'ref:e28de4ac carry no share_by_node). So "the same proportions" is shown for 5 of the 7 m = 2 evaluations whose gap '
        'is quoted. Unchanged in substance from e926fb7 (which quoted 55/31/14 % without a table). Wording.',
        {'F2_refs_no_share_in_frozen': dz['ref:5ca4f86c'].get('share_by_node') is None and dz['ref:e28de4ac'].get('share_by_node') is None,
         'E24_rows_without_share_are_40_41': sorted(int(e24[n]['cells'][0]) for n in share_missing) == [40, 41] if share_missing else False})
    add('C48-2', '4.8 l. 1720', '21 cycles in 10 evaluations ... excluded from certification evidence (Table E.21)', 'consistent',
        f'Column N.c. of Table E.21: positive on {len(nc_pos)} rows {sorted(nc_pos)}, sum {sum(nc_pos.values())}; every one is a '
        'campaign evaluation (E.6: 3-11, 15, 22, 23, 26-31, 34-39, 42-59).',
        {'nc_sum_21': sum(nc_pos.values()) == 21, 'nc_rows_10': len(nc_pos) == 10})
    add('C-R22-cycles', 'letter R2.2 l. 125', 'the cycles of every evaluation in Appendix E', 'L',
        f'Table E.21 gives k*/end for all 61 evaluations ({sum(1 for v in k_end.values() if v != "---")} of 61 printed) and k0 for 57 '
        '(not for the continued evaluations 1, 2, 40, 41; the note gives k0 = 123 for evaluation 1). The 41 search evaluations '
        '(production exit) and the multi-scenario pair (cycle 72, Table 13) are not in Appendix E. Wording; the reply is about the '
        'outer loop, whose candidate count is in 4.7.',
        {'k_end_all_61': all(v != '---' for v in k_end.values()) and len(k_end) == 61,
         'k0_blank_only_1_2_40_41': sorted(n for n, r_ in e21.items() if r_['cells'][5] == '---') == [1, 2, 40, 41]})
    return {'rows': rows, 'counts': dict(Counter(r_['rating'] for r_ in rows)),
            'e20_rows_17_20': g_rows, 'e20_row_34': r34, 'sign_changes': sign_changes, 'verdict_changes': verdict_changes,
            'verdict_label_changes': verdict_label_changes, 'e20_net_rows': len(e20_net_rows), 'e20_gross_only_rows': e20_gross_only,
            'e21_nonclean_rows': nc_pos, 'e24_rows_without_share': [e24[n]['cells'][0] for n in share_missing]}


# ======================================================================================================================
#  4. (b) letter: changed replies
# ======================================================================================================================
def mask_letter_line(x):
    x = re.sub(r'(?<!\\)%.*', '', x)
    for pat in (R.PTR, R.PTR_APP):
        x = pat.sub(lambda m: ' ' * len(m.group(0)), x)
    x = x.replace('--', ' \u2013')
    x = re.sub(r'R\d\.\d+', lambda m: ' ' * len(m.group(0)), x)
    x = re.sub(r'\$3\\times\s*3\$', lambda m: '3x3' + ' ' * (len(m.group(0)) - 3), x)
    x = re.sub(r'\b(former|new) \d+\.\d+\.\d+', lambda m: ' ' * len(m.group(0)), x)
    return x


def word_tokens(x):
    """[(word_index, token)] over whitespace-split words."""
    out = []
    for wi, w in enumerate(x.split()):
        for m in TOK.finditer(w):
            out.append((wi, m.group(1) or m.group(2)))
    return out


def line_map(old, new):
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    mp = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                mp[j1 + k + 1] = i1 + k + 1
    return mp, sm.get_opcodes()


def letter_line_tokens(new_line, old_line):
    """tokens of new_line; each marked inserted (not in an equal word span vs old_line) or carried (old word index, occ)."""
    nw, ow = mask_letter_line(new_line).split(), mask_letter_line(old_line).split() if old_line is not None else []
    sm = difflib.SequenceMatcher(a=ow, b=nw, autojunk=False)
    eqmap = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(j2 - j1):
                eqmap[j1 + k] = i1 + k
    old_tok = word_tokens(' '.join(ow))
    old_occ, seen = {}, Counter()
    for wi, t in old_tok:
        seen[t.lower()] += 1
        old_occ.setdefault(wi, []).append((t, seen[t.lower()]))
    out = []
    for wi, t in word_tokens(' '.join(nw)):
        if wi in eqmap:
            occ = next((o for tt, o in old_occ.get(eqmap[wi], []) if tt == t), None)
            out.append({'token': t, 'inserted': False, 'old_occ': occ})
        else:
            out.append({'token': t, 'inserted': True, 'old_occ': None})
    return out


def task_letter_changes(S, v6, v5, v4):
    """Numbers: every token on the letter lines changed v4 -> v6; inserted tokens declared here, carried ones looked up in
    W178c's committed Part 1 rows (by reply, v4 line, token and occurrence). Claims: the changed sentences rated."""
    mp64, ops = line_map(v4, v6)
    changed = [j for j in range(1, len(v6) + 1) if j not in mp64]
    w178c_rows = {(r_['line'], r_['token'].lower(), r_['occ']): r_ for r_ in S.w178c['part1_numbers']['rows']}
    # v4 line for each changed v6 line (equal-length replace blocks map line by line; v6 l. 285 is a pure insertion)
    rep = {}
    for tag, i1, i2, j1, j2 in ops:
        if tag == 'replace' and i2 - i1 == j2 - j1:
            for k in range(j2 - j1):
                rep[j1 + k + 1] = i1 + k + 1
    inserted_decl = {   # (v6 line, token, occurrence among inserted tokens) -> declaration
        (221, '55', 1): N('55', 55, 'distinct frozen comparisons with a net figure (see C46-2); W153 n_claims 60 less 4 G :net less CHECK'),
    }
    carried_decl = {   # carried tokens W178c's Part 1 has no row for (its extractor masked them)
        (58, '1'): L_('1', '"(Reviewer~1)": a reviewer label'), (58, '3'): L_('3', '"(Reviewer~3, ...)": a reviewer label')}
    new_claim_tokens = {(284, 'single', 1): L_('single', 'single machine (claim FC-1; E.6 "Apple M2 Max")'),
                        (284, 'single', 2): L_('single', 'single thread (claim FC-1; E.6 "single-threaded")')}
    rows = []
    for j in changed:
        old = v4[rep[j] - 1] if j in rep else None
        occ_ins = Counter()
        for t in letter_line_tokens(v6[j - 1], old):
            row = {'line_v6': j, 'line_v4': rep.get(j), 'token': t['token'], 'inserted_since_v4': t['inserted']}
            if t['inserted']:
                occ_ins[t['token'].lower()] += 1
                key = (j, t['token'].lower(), occ_ins[t['token'].lower()])
                d = inserted_decl.get(key) or new_claim_tokens.get(key)
                if d is None:
                    _fail(f'letter l. {j}: inserted token {t["token"]} undeclared')
                    row['status'] = 'UNDECLARED'
                else:
                    row.update({k: d[k] for k in ('status', 'expected', 'source')})
            else:
                pr = w178c_rows.get((rep.get(j), t['token'].lower(), t['old_occ']))
                if pr is None:
                    d = carried_decl.get((j, t['token']))
                    if d is None:
                        _fail(f'letter l. {j}: carried token {t["token"]} has no W178c row and no declaration')
                        row['status'] = 'UNDECLARED'
                    else:
                        row.update({k: d[k] for k in ('status', 'expected', 'source')})
                else:
                    row['status'] = f'carried, W178c {pr["status"]}'
                    row['source'] = pr['source']
            rows.append(row)
    for r_ in rows:
        if r_['status'] in ('MISMATCH', 'UNDECLARED') or 'MISMATCH' in r_['status']:
            _fail(f'letter number {r_}')
    cl_main = typeset(lines_of(_read(MAIN_TEX)))
    main_t = '\n'.join(cl_main)
    e_all = norm(region(cl_main, 'app:results', r'\\section\*\{Declaration')[0])
    a8 = norm(R._region_text(cl_main, 'app:admm_updated_settings'))
    e6 = norm(region(cl_main, 'app:results_statistics', r'\\section\*\{Declaration')[0])
    sub = _read(SUBMITTED)
    sub_app = sub[sub.find('\\label{app:results}'):]
    claims = []

    def add(cid, line, fragment, rating, basis, evidence, expect=True):
        frag_hits = sum(1 for ln in v6 if fragment in ln)
        evidence = dict(evidence, fragment_once_in_letter=frag_hits == 1)
        ok = all(v is True for v in evidence.values())
        if expect and not ok:
            _fail(f'letter claim {cid}: {evidence}')
        claims.append({'id': cid, 'line': line, 'fragment': fragment, 'rating': rating, 'basis': basis, 'evidence': evidence,
                       'evidence_ok': ok})
    add('R22-a', 125, 'The number of candidates evaluated is reported in Section~4.7', 'consistent',
        '4.7: "made 41 new recourse evaluations in all (14, 20 and 7 against budgets of 20, 20 and 60)".',
        {'4_7_count': '41 new recourse evaluations in all (14, 20 and 7' in main_t})
    add('R22-b', 125, 'and the cycles of every evaluation in Appendix~E', 'L',
        'See C-R22-cycles: E.21 carries k*/end for the 61 evaluations behind the comparisons; the search and multi-scenario '
        'evaluations are not in Appendix E.', {'E21_present': 'tab:appe_evaluations' in e_all.lower() or 'label{tab:appE_evaluations}' in main_t})
    add('R22-c', 126, 'Section~4.7; Appendix~E.}', 'consistent', 'rchanges now lists Appendix E, which carries the cycle columns.', {})
    add('R31-a', 221, 'Among the 55 comparisons of the full claims table that are reported both gross and net, it changes no sign and one verdict',
        'consistent', 'As C46-2 (55 distinct comparisons with a net figure; 0 sign changes; 1 verdict change). "the full claims '
        'table" (of Appendix E, l. 220) is Table E.20.', {'E20_present': 'label{tab:appE_claims}' in main_t,
                                                          'E_salvage': 'salvage' in e_all})
    add('FC-1', 284, 'A reproducibility note is new: single machine, single thread, bitwise replays, solver and library versions (Appendices~E.6 and~A.8)',
        'consistent', 'E.6: Apple M2 Max (one machine), single-threaded, bitwise replays; A.8: IPOPT 3.14.18, Pyomo 6.9.5, Python '
        '3.11.11, MA97 / MA57. No HSL version is printed (A.8 names the linear solvers only); "library versions" holds for Pyomo / '
        'Python.', {'E6_machine': 'apple m2 max' in e6.lower(), 'E6_single_threaded': 'single-threaded' in e6,
                    'E6_bitwise': 'bitwise' in e6, 'A8_ipopt': 'IPOPT 3.14.18' in a8, 'A8_pyomo': 'Pyomo 6.9.5' in a8,
                    'A8_python': 'Python 3.11.11' in a8})
    add('FC-2a', 285, 'In the submitted version it held per-block operating results of the submitted instance, which the revision does not use',
        'consistent', 'manuscript_submitted/main.tex: \\section{Results} \\label{app:results} with '
        'tab:cs3_results_summary_years_days (daily cost, generation, curtailment by representative year and day); the revision '
        'evaluates the three-year single-scenario instance and the 3 x 3 instance, not the submitted 5-year 25-scenario one (4.8).',
        {'submitted_app_table': 'cs3_results_summary_years_days' in sub_app,
         'head_table_gone': 'cs3_results_summary_years_days' not in main_t})
    add('FC-2b', 285, 'every pairwise comparison, with its threshold or bar and its verdict, gross and, where evaluated, net of salvage',
        'consistent', 'Table E.20: Thr. column, verdicts gross and net; rows 22, 52, 58 gross only ("---: not evaluated net").',
        {'E20': 'label{tab:appE_claims}' in main_t, 'not_evaluated_net_note': 'not evaluated net' in e_all})
    add('FC-2c', 285, 'every recourse evaluation, with its stopping status, band, gap and slack', 'L',
        'Table E.21 lists the 61 evaluations behind the comparisons with status, band, gap and slack; the 41 search evaluations and '
        'the multi-scenario pair (Table 13, named in the E intro) are not in it. Same wording point as C47-1 / R22-b.',
        {'E21': 'label{tab:appE_evaluations}' in main_t})
    add('FC-2d', 285, 'the neighbours of the plan without storage re-evaluated under the certification rule; the solver starts of the static '
        'arrangements; the evaluations refused by the gap clause; and the certification statistics', 'consistent',
        'Tables E.22, E.23, E.24 and E.6, in that order.',
        {'E22': 'label{tab:appE_neighbours}' in main_t, 'E23': 'label{tab:appE_benchmark}' in main_t,
         'E24': 'label{tab:appE_gap}' in main_t, 'E6': 'label{app:results_statistics}' in main_t})
    return {'changed_lines_v4_to_v6': changed, 'number_rows': rows,
            'number_counts': dict(Counter(r_['status'] for r_ in rows)), 'claims': claims,
            'claim_counts': dict(Counter(c['rating'] for c in claims))}


# ======================================================================================================================
#  4b. pointers of the whole letter v6
# ======================================================================================================================
SEC_EXTRA = {'3.3': 'titlek:subsection:^Transmission Network$',   # Appendix C has the same title (section level)
             '2.2.6': 'title:^Coupling Between Planning and Operational Recourse$',
             'E.6': 'app:results_statistics'}
EQS_EXTRA = {'26-28': ['eq:storage_active_power_limit', 'eq:converter_capability', 'eq:network_complementarity']}
NEEDLES_V6 = {   # pointers on lines changed since v4 (and added needles on unchanged lines); (v6 line, kind, first number)
    (58, 'section', '4.8'): ['the value is affine in size', 'could not be evaluated within the memory available'],
    (58, 'section', '4.5'): ['The multi-scenario instance combines three market scenarios'],
    (97, 'section', '3.3'): ['Conventional generation in the TN is priced at the wholesale energy price'],
    (97, 'table', '15'): ['rates other than 2\\,\\% re-weight the 2\\,\\% evaluations'],
    (97, 'section', '4.6'): ['re-weight the per-year costs of the 2\\,\\% evaluations'],
    (97, 'appendix', 'A.1'): ['(1+r)^{-(y-y_0)}'],
    (125, 'equation', '13'): ['\\frac{Y_y}{(1+r)^{y-y_0}}', '\\omega_m \\, \\omega_o'],
    (125, 'section', '4.7'): ['41 new recourse evaluations in all'],
    (125, 'appendix', 'E'): ['$k^{*}$/end'],
    (126, 'appendix', 'E'): ['Recourse evaluations.'],
    (136, 'section', '2.2.6'): ['are fixed parameters of the operational subproblem'],
    (136, 'section', '2.3.2'): ['whose ratings are the fixed additions'],
    (160, 'appendix', 'E'): ['settling slack', 'bitwise'],
    (172, 'section', '4.8'): ['a different penalty schedule or starting point could reach a different local solution'],
    (193, 'equation', '26'): ['\\le S^{\\text{Av}}_{e,y}', '\\left(S^{\\text{Av}}_{e,y}\\right)^2', '{S^{\\text{Av}}_{e,y}}'],
    (220, 'appendix', 'E'): ['Every pairwise comparison of the record', 'Net: net of the terminal salvage credit'],
    (221, 'section', '4.6'): ['The investment-year comparison above is not among them'],
    (221, 'section', '3.5'): ['10--90'],
    (221, 'equation', '22'): ['SoC^{\\text{Min}}_e E^{\\text{Av}}_{e,y}'],
    (222, 'appendix', 'E'): ['Every pairwise comparison of the record'],
    (284, 'appendix', 'E.6'): ['bitwise', 'single-threaded', 'Apple M2~Max'],
    (284, 'appendix', 'A.8'): ['IPOPT~3.14.18', 'Pyomo~6.9.5'],
    (285, 'appendix', 'E'): ['Every pairwise comparison of the record', 'Neighbours of the Plan without Storage',
                             'Evaluations Refused by the Gap Clause', 'Certification Statistics and Reproducibility'],
}


def _titlek(t):
    """'titlek:<kind>:<regex>' -> (kind, regex): a heading title qualified by its level (W179b; the W177b-3' 'title:'
    form requires a unique title, and 'Transmission Network' is both 3.3 and Appendix C)."""
    _, kind, rx = t.split(':', 2)
    return kind, rx


def resolve_v6(t, numb, cites):
    if t.startswith('titlek:'):
        kind, rx = _titlek(t)
        hits = [o for o in numb['objects'] if o['kind'] == kind and re.search(rx, o['title'])]
        return hits[0]['number'] if len(hits) == 1 else None
    return R._resolve(t, numb, cites)


def region_v6(cl, t):
    if t.startswith('titlek:'):
        kind, rx = _titlek(t)
        idx = [k for k, ln in enumerate(cl) if re.match(r'\s*\\' + kind + r'\{', ln)
               and re.search(rx, re.sub(r'\\textcolor\{[^}]*\}\{(.*)\}', r'\1', re.sub(r'^\s*\\\w+\{(.*)\}\s*$', r'\1', ln)).strip())]
        if len(idx) != 1:
            return ''
        stop = {'section': ('section',), 'subsection': ('section', 'subsection'),
                'subsubsection': ('section', 'subsection', 'subsubsection')}[kind]
        e = next((k for k in range(idx[0] + 1, len(cl)) if re.match(r'\s*\\(' + '|'.join(stop) + r')\*?\{', cl[k])), len(cl))
        return '\n'.join(cl[idx[0]:e])
    return R._region_text(cl, t)


def target_v6(a):
    t = C._target(a)
    if t is not None:
        return t
    k, nums = a['kind'], a['numbers']
    if k in ('section', 'appendix') and not a['range']:
        return SEC_EXTRA.get(nums[0])
    if k == 'equation':
        return EQS_EXTRA.get('-'.join(nums))
    return None


def task_pointers(S, main_lines, adj, raw, v6, v4):
    cites = R.cite_order(main_lines)
    cl = typeset(main_lines)
    atoms = R.letter_pointers(LETTER)
    mp64, _ = line_map(v4, v6)
    prior = {}
    for r_ in S.w178c['pointers']['pointers']:
        prior.setdefault((r_['line'], r_['kind'], r_['numbers'][0], r_['text']), r_)
    needles = {}
    for (ln, kind, num, text), r_ in prior.items():   # carried from W178c (unchanged lines only)
        if r_.get('content_needles'):
            for j, i in mp64.items():
                if i == ln:
                    needles.setdefault((j, kind, num), [])
                    for nd in r_['content_needles']:
                        if nd not in needles[(j, kind, num)]:
                            needles[(j, kind, num)].append(nd)
    for k, v in NEEDLES_V6.items():
        if k[0] in mp64:                              # an unchanged line: add, keep the carried ones
            needles.setdefault(k, [])
            for nd in v:
                if nd not in needles[k]:
                    needles[k].append(nd)
        else:
            needles[k] = list(v)
    rows = []
    for a in atoms:
        tgt = target_v6(a)
        row = {k: a[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range')}
        row['target'] = tgt
        row['v4_line'] = mp64.get(a['line'])
        pr = prior.get((mp64.get(a['line']), a['kind'], a['numbers'][0], a['text'])) if a['line'] in mp64 else None
        row['verdict_9fd19e7'] = pr['verdict'] if pr else None
        row['number_9fd19e7'] = pr.get('number_9fd19e7') if pr else None
        if tgt is None:
            row['verdict'] = 'UNDECLARED'
            rows.append(row)
            continue
        if isinstance(tgt, str) and tgt.startswith('SUBMITTED:'):
            row['verdict'], row['note'] = 'SUBMITTED', tgt[10:]
            rows.append(row)
            continue
        tl = tgt if isinstance(tgt, list) else [tgt]
        n_adj = [resolve_v6(t, adj, cites) for t in tl]
        n_raw = [resolve_v6(t, raw, cites) for t in tl]
        if any(v is None for v in n_adj):
            row['verdict'], row['note'] = 'AMBIGUOUS', f'target not resolvable: {n_adj}'
            rows.append(row)
            continue
        if a['range']:
            got, got_raw = [n_adj[0], n_adj[-1]], [n_raw[0], n_raw[-1]]
            row['members'], row['members_head'] = tl, n_adj
            if not C._consecutive(n_adj):
                row['verdict'], row['note'] = 'AMBIGUOUS', f'range members not consecutive: {n_adj}'
                rows.append(row)
                continue
        else:
            got, got_raw = n_adj, n_raw
        row['number_head'] = '--'.join(got)
        row['number_head_unadjusted'] = '--'.join(got_raw)
        row['verdict'] = 'MATCH' if got == a['numbers'] else 'MISMATCH'
        if row['verdict'] == 'MISMATCH':
            row['should_be'] = row['number_head']
        nds = needles.get((a['line'], a['kind'], a['numbers'][0]))
        if nds:
            if tl[0].startswith('cite:'):
                row['content_needles'] = {nd: (tl[0][5:] in cites) for nd in nds}
            else:
                reg = '\n'.join(region_v6(cl, t) for t in tl)
                row['content_needles'] = {nd: (nd in reg) for nd in nds}
        rows.append(row)
    used = {(r_['line'], r_['kind'], r_['numbers'][0]) for r_ in rows}
    for k in NEEDLES_V6:
        if k not in used:
            _fail(f'needle declaration {k} matches no extracted pointer')
    und = [r_ for r_ in rows if r_['verdict'] == 'UNDECLARED']
    if und:
        _fail(f'{len(und)} pointer(s) without a declared target: {[(r_["line"], r_["text"]) for r_ in und]}')
    counts = Counter(r_['verdict'] for r_ in rows)
    by_kind = {}
    for r_ in rows:
        by_kind.setdefault(r_['kind'], Counter())[r_['verdict']] += 1
    changed_lines = sorted({r_['line'] for r_ in rows if r_['line'] not in mp64})
    absent = [{k: r_.get(k) for k in ('line', 'text', 'target', 'content_needles')}
              for r_ in rows if r_.get('content_needles') and not all(r_['content_needles'].values())]
    return {'n_pointers': len(rows), 'counts': dict(counts), 'by_kind': {k: dict(v) for k, v in by_kind.items()},
            'pointers': rows, 'mismatch': [r_ for r_ in rows if r_['verdict'] == 'MISMATCH'],
            'ambiguous': [r_ for r_ in rows if r_['verdict'] == 'AMBIGUOUS'],
            'pointer_lines_changed_since_v4': changed_lines, 'n_with_needles': sum(1 for r_ in rows if r_.get('content_needles')),
            'content_absent': absent,
            'verdict_changed_vs_9fd19e7': [{k: r_.get(k) for k in ('line', 'text', 'verdict_9fd19e7', 'verdict')}
                                           for r_ in rows if r_['verdict_9fd19e7'] and r_['verdict_9fd19e7'] != r_['verdict']],
            'new_pointers_since_v4': [{k: r_.get(k) for k in ('line', 'text', 'target', 'number_head', 'verdict')}
                                      for r_ in rows if r_['line'] not in mp64]}


# ======================================================================================================================
#  5. (c) G1-G12 hunks and letter v5
# ======================================================================================================================
G_OF_HUNK = ['G12', 'G1', 'G4', 'G9', 'G11', 'G7', 'G3', 'G3+G8', 'G2', 'G11', 'G10', 'G5', 'G6', 'G6']


def parse_hunks(diff_text):
    hunks, cur = [], None
    for ln in diff_text.split('\n'):
        if ln.startswith('@@'):
            cur = {'header': ln, 'minus': [], 'plus': []}
            hunks.append(cur)
        elif cur is not None and ln.startswith('-') and not ln.startswith('---'):
            cur['minus'].append(ln[1:])
        elif cur is not None and ln.startswith('+') and not ln.startswith('+++'):
            cur['plus'].append(ln[1:])
    return hunks


def inserted_words(old, new):
    ow, nw = old.split(), new.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ow, b=nw, autojunk=False).get_opcodes():
        if tag in ('insert', 'replace'):
            out.append(' '.join(nw[j1:j2]))
    return out


G_TOKENS = {   # inserted numeric tokens of the G hunks -> (expected, source)
    ('G3', '0.778'): (lambda S: S.w155['reports']['e_soh050']['AE'], 'W155 summary reports.e_soh050.AE (T8 3 dp)'),
    ('G3', '1.08'): (lambda S: S.w155['reports']['e_soh050']['EFC'], 'W155 summary reports.e_soh050.EFC (T8 2 dp)'),
    ('G4', '92'): (lambda S: S.srp1['Days']['Spring'], 'SRP1.json Days.Spring'),
    ('G4', '91'): (lambda S: S.srp1['Days']['Summer'], 'SRP1.json Days.Summer / Autumn / Winter'),
    ('G7', '2025'): (lambda S: 2025, 'tab:res_energy_ladder_m2 caption "node 7 in 2025"'),
    ('G10', 'three'): (lambda S: t6_starts_per_arrangement(), f'{EXPORT_T6}: recorded starts per NRF arrangement (each arrangement)'),
}
G_LANG = {('G9', 'one'): '"or with one another" (language, not a quantity)'}


def t6_starts_per_arrangement():
    """number of recorded solver starts per NRF arrangement in the T6 export; one value required for all arrangements"""
    import csv
    with open(EXPORT_T6, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    per = Counter(r_['arrangement'] for r_ in rows if r_['arrangement'].endswith('NRF') and '\u2212' not in r_['arrangement']
                  and r_['figure label'] == 'recorded'
                  and not r_['start'].startswith('best of'))
    vals = set(per.values())
    return vals.pop() if len(vals) == 1 else None


def task_g_hunks(S, main_lines, g_lines, w_lines):
    hunks = parse_hunks(_read(DIFF_FOLLOWUP))
    if len(hunks) != len(G_OF_HUNK):
        _fail(f'followup diff has {len(hunks)} hunks, expected {len(G_OF_HUNK)}')
    head_set = {ln.strip() for ln in main_lines}
    g_set = {ln.strip() for ln in g_lines}
    w_set = {ln.strip() for ln in w_lines}
    head_norm = norm('\n'.join(typeset(main_lines)))
    rows = []
    for k, h in enumerate(hunks):
        gid = G_OF_HUNK[k] if k < len(G_OF_HUNK) else '?'
        plus = [p for p in h['plus'] if p.strip()]
        minus = [m for m in h['minus'] if m.strip()]
        ins = []
        for m_, p_ in zip(minus, plus):
            ins += inserted_words(m_, p_)
        if len(plus) > len(minus):
            ins += plus[len(minus):]
        row = {'hunk': k + 1, 'header': h['header'], 'g': gid, 'n_minus': len(minus), 'n_plus': len(plus),
               'plus_all_verbatim_at_e926fb7': all(p.strip() in g_set for p in plus),
               'plus_all_verbatim_at_head': all(p.strip() in head_set for p in plus),
               'minus_all_present_at_9fd19e7': all(m.strip() in w_set for m in minus),
               'minus_all_absent_at_head': all(m.strip() not in head_set for m in minus),
               'inserted_spans': ins,
               'inserted_spans_present_at_head': all(norm(re.sub(r'(?<!\\)%.*', '', s)) in head_norm for s in ins if not s.lstrip().startswith('%'))}
        if row['plus_all_verbatim_at_head'] and row['minus_all_absent_at_head']:
            row['applied'] = 'verbatim'
        elif row['inserted_spans_present_at_head']:
            row['applied'] = 'with changes'
        else:
            row['applied'] = 'absent'
        toks = []
        for s in ins:
            for t in tokens_of(clean_main(s)):
                key = (gid.split('+')[0], t)
                fn = G_TOKENS.get(key) or G_TOKENS.get((gid.split('+')[-1], t))
                if fn is None and key in G_LANG:
                    toks.append({'token': t, 'status': 'language (not a quantity)', 'source': G_LANG[key]})
                    continue
                if fn is None:
                    toks.append({'token': t, 'status': 'UNDECLARED'})
                    _fail(f'G hunk {k + 1} ({gid}): inserted token {t} undeclared')
                    continue
                val = fn[0](S)
                ok = eq_written(t, val) if not t.isalpha() else WORDS.get(t.lower()) == val
                toks.append({'token': t, 'expected': val, 'source': fn[1], 'status': 'match' if ok else 'MISMATCH'})
                if not ok:
                    _fail(f'G hunk {k + 1} token {t}: MISMATCH ({val})')
        row['inserted_tokens'] = toks
        if not (row['plus_all_verbatim_at_e926fb7'] and row['minus_all_present_at_9fd19e7']):
            _fail(f'G hunk {k + 1} not as committed at e926fb7 / 9fd19e7: {row}')
        rows.append(row)
    days = S.srp1['Days']
    d33 = S.i33['Days']
    same = days == d33
    if not same:
        _fail(f'G4: the two instances have different day counts: SRP1 {days}, 3x3 {d33}')
    return {'n_hunks': len(rows), 'rows': rows, 'applied_counts': dict(Counter(r_['applied'] for r_ in rows)),
            'srp1_days_in_order': [days['Spring'], days['Summer'], days['Autumn'], days['Winter']],
            'instance_3x3_days_in_order': [d33['Spring'], d33['Summer'], d33['Autumn'], d33['Winter']],
            'both_instances_same_day_counts': same}


def task_letter_v5(v6, v5, v4):
    old, new = _pm_lines(_unified('\n'.join(v4), '\n'.join(v5)))
    items = [
        ('R1.4 fuel', 'fuel costs, through the wholesale prices at which conventional generation is costed (Section~3.3)'),
        ('R1.4 A.1', 'the ageing chain couples the years and the rate enters the coordination weights (Appendix~A.1); the paper says so (Section~4.6)'),
        ('R2.10 converter limits', 'entering only the converter limits (Eqs.~(26)--(28)), not the degradation'),
        ('R2.3 2.2.6', 'the investment being a fixed parameter of the recourse (Sections~2.2.6 and~2.3.2)'),
        ('LM-3 inference', 'which we state as an inference, not a result (Section~4.8)'),
        ('LL-8 pointer', 'state this limitation explicitly (Section~4.8).'),
    ]
    v6t = '\n'.join(v6)
    rows = []
    for name, frag in items:
        r_ = {'edit': name, 'fragment': frag, 'in_v5_plus_lines': any(frag in ln for ln in new),
              'present_at_head': v6t.count(frag) == 1}
        r_['applied'] = 'verbatim' if r_['present_at_head'] else 'absent'
        if not (r_['in_v5_plus_lines'] and r_['present_at_head']):
            _fail(f'letter v5 edit {name}: {r_}')
        rows.append(r_)
    gone = {'assumption (Section~4.8)': 'which we state as an assumption (Section~4.8)' not in v6t,
            'only through the P-Q-S capability inequality': 'only through the $P$--$Q$--$S$ capability inequality' not in v6t}
    for k, v in gone.items():
        if not v:
            _fail(f'letter v5: old text still present: {k}')
    return {'n_v4_v5_changed_lines': len(new), 'rows': rows, 'old_text_gone': gone}


# ======================================================================================================================
#  5b. W178 rows re-scored
# ======================================================================================================================
def task_w178_rows(S, main_lines, g_lines, w_lines, v6):
    ht = '\n'.join(typeset(main_lines))
    wt = '\n'.join(typeset(w_lines))
    lt = '\n'.join(v6)
    cl = typeset(main_lines)
    hooks = _read(HOOKS_W101)
    t11 = next(ln for ln in cl if 'Baseline, 0.50 floor &' in ln)
    t10_rows = [ln for ln in R._region_text(cl, 'tab:res_energy_ladder_m2').split('\n') if re.match(r'\s*\d\$?\^?\{?\w?\}?\$?\s*&', ln)]
    t10_vmi = [kx(re.split(r'\s*&\s*', ln.strip())[1]) for ln in t10_rows]
    fuel_hits = len(re.findall(r'\bfuel', ht, re.I))
    intro = R._region_text(cl, 'title:^Introduction$')
    rows = []

    def add(rid, source, prior, g, old_frag, new_frag, status, rating, basis, extra=None):
        ev = {'old_fragment_at_9fd19e7': (old_frag in wt or old_frag in lt or old_frag in '\n'.join(_read(LETTER_V4).split('\n')))
              if old_frag else None,
              'old_fragment_absent_at_head': (old_frag not in ht and old_frag not in lt) if old_frag else None,
              'new_fragment_at_head': (new_frag in ht or new_frag in lt) if new_frag else None}
        if extra:
            ev.update(extra)
        bad = [k for k, v in ev.items() if v is False]
        if bad:
            _fail(f'W178 row {rid}: evidence {bad} False')
        rows.append({'row': rid, 'source': source, 'prior_rating': prior, 'addressed_by': g, 'status': status,
                     'rating_now': rating, 'basis': basis, 'evidence': ev})
    add('W178a R2 (l. 755)', 'P5_15_W178A_SECTION2.md / W178 Decision 1', 'L', 'G1',
        'Appendix~\\ref{app:admm_updated_implementation}', 'detailed in \\ref{app:admm_updated_implementation}', 'closed', None,
        'Both prefixes dropped (l. 756: "detailed in \\ref{...implementation}", "(\\ref{...algorithm})").',
        {'no_Appendix_ref_prefix_at_head': not re.search(r'Appendix~\\ref', ht)})
    add('W178b A11 (l. 1603)', 'P5_15_W178B_FIXES.md Blocked 2 / W178 Decision 1', 'L', 'G2',
        'Section~\\ref{app:admm_updated_settings}', 'every local solve successful; \\ref{app:admm_updated_settings})', 'closed', None,
        '4.5 l. 1601 now reads "...; \\ref{app:admm_updated_settings})".',
        {'no_Section_app_prefix_at_head': not re.search(r'Sections?~\\ref\{app:', ht)})
    add('A-prefix, A.3 l. 1899 (not in W178)', 'this audit (same class as G1-G2)', None, '-',
        None, 'their tolerances (Subsection~\\ref{app:admm_updated_stopping})', 'new difference (pre-existing, missed by W178)', 'L',
        '"Subsection~\\ref{app:admm_updated_stopping}" prints "Subsection Appendix A.4" (elsarticle reading, static). Present '
        'since at least 9fd19e7 (l. 1908); W178a/b listed l. 755 and l. 1603 only. Fix: drop "Subsection~".',
        {'present_at_9fd19e7': 'Subsection~\\ref{app:admm_updated_stopping}' in wt})
    add('P0-C5 / W178 Decision 2 (0.50-floor AE, EFC)', 'P5_15_W178B_FIXES.md (c)', 'author item', 'G3',
        'never & --- & ---', 'never & 0.778 & 1.08', 'closed', None,
        f'T11 row now "{t11.strip()[:80]}"; W155 AE 0.77811 -> 0.778, EFC 1.07630 -> 1.08; the "---: not reported" legend is '
        'removed with it. Letter R3.6 ("Table 11 gives AE and EFC/day per calibration") now holds for every row.',
        {'legend_removed': '---: not reported' not in ht})
    dline = next(i for i, ln in enumerate(main_lines, 1) if 'weighted by the numbers of days they stand for' in ln)
    add('R37-4 (day weights unstated)', 'P5_15_W178C_LETTER.md Blocked 1(b)', 'M', 'G4',
        None, 'In both instances each year is represented by four seasonal representative days, weighted by the numbers of days they stand for (92, 91, 91 and 91)', 'closed', None,
        f'Section 3 l. {dline} states 92, 91, 91 and 91 "in both instances" (SRP1.json and the 3 x 3 instance file both: Spring 92, '
        'Summer 91, Autumn 91, Winter 91; the seasons are not named beside the counts, which is immaterial as three are equal). '
        'The letter R3.7 "(Section~3)" now holds.',
        {'srp1_days': [S.srp1['Days'][k] for k in ('Spring', 'Summer', 'Autumn', 'Winter')] == [92, 91, 91, 91],
         'i33_days': [S.i33['Days'][k] for k in ('Spring', 'Summer', 'Autumn', 'Winter')] == [92, 91, 91, 91],
         'letter_R37_unchanged': 'the representative-day weights and the block length of each representative year are stated (Section~3)' in lt})
    add('C-1 (conclusions stopping status)', 'P5_15_W178B_FIXES.md / W178 Decision 4', 'L', 'G5',
        'either certified by an explicit stopping rule or reported in an uncertified form with its error bar',
        'or, at the multi-scenario instance, stopped by the production exit with its remaining movement measured',
        'still (narrowed)', 'L',
        'The 3 x 3 pair is now covered. Residual: (i) the search evaluations behind 4.1 "none reduced the total discounted cost '
        'beyond the search resolution" and 4.8 Scope (production exit at sigma_Q) are still not named; (ii) "its remaining '
        'movement measured": only the x = 0 evaluation was continued, and its further descent (6.2 k EUR) is a post-hoc fit '
        'estimate (4.5 "A subsequent fit ... estimated"; Table 13 "damped-cosine fit"); the unit evaluation\'s was not measured.',
        {'search_resolution_sentence_in_4_1': 'none reduced the total discounted cost beyond the search resolution' in ht,
         'x0_only_continued': 'the evaluation of the plan without storage was continued beyond its stopping point' in ht})
    add('A-1 (hold origin, continued evaluations)', 'P5_15_W178B_FIXES.md / W178 Decision 4', 'L', 'G6',
        "or, in the three reference evaluations continued from an earlier run, from that run's stopping cycle",
        "from the cycle after that run's stopping cycle", 'closed', None,
        'A.3 l. 1899 and A.5 l. 1923 both say "from the cycle after that run\'s stopping cycle"; the hooks hold "for every cycle '
        '> N" (p515_s53_w101_settling_continuation_hooks.py docstring).',
        {'hooks_for_every_cycle_gt_N': 'for every cycle > N the certifying regime HELD' in hooks,
         'two_occurrences_at_head': ht.count("from the cycle after that run's stopping cycle") == 2})
    add('4.2-1', 'P5_15_W177B_PART2_E10C104.md (carried by W178b)', 'L', 'G7',
        'the value of every evaluated 4~h configuration at node~7 exceeds its investment cost',
        'the value of each 4~h configuration at node~7 installed in 2025 exceeds its investment cost', 'closed', None,
        f'Table 10 (node 7, 2025, m = 2) value - I: {t10_vmi} k EUR, all positive.',
        {'t10_all_positive': len(t10_vmi) == 5 and all(v > 0 for v in t10_vmi)})
    add('4.3-1', 'P5_15_W177B_PART2_E10C104.md (carried by W178b)', 'L', 'G8',
        'persists across the tested ageing calibrations and both floor settings',
        'persists across the tested ageing calibrations, and under the baseline calibration at both floor settings', 'closed', None,
        'The 0.50 floor was evaluated at the baseline calibration only (one 0.50 row in Table 11); the sentence now says so.')
    add('S3-12', 'P5_15_W177B_PART2_E10C104.md (carried by W178b)', 'L', 'G9',
        'Every candidate plan is compared with the plan without shared storage',
        'Storage plans are compared under coordinated operation with the plan without shared storage, $\\boldsymbol{x} = \\boldsymbol{0}$, which isolates the value of the storage investment, or with one another',
        'closed', None, 'Table E.20 (b) increments, (c) neighbours against the returned plan and (d) calibration differences are '
        'plan-to-plan comparisons; "or with one another" covers them.')
    add('LL-8 (manuscript and letter R2.5 l. 172)', 'P5_15_W178B_FIXES.md / P5_15_W178C_LETTER.md (R25-7)', 'L', 'G10 + letter v5',
        None, 'a different penalty schedule or starting point could reach a different local solution', 'closed', None,
        '4.8 Scope now states the local-solution limitation and points to the three-start spread (Table 12 prints the spread in '
        'parentheses: 25.8 and 0.0 k EUR); the letter points to it ("(Section~4.8)").',
        {'letter_pointer': 'state this limitation explicitly (Section~4.8).' in lt,
         'table12_spread': 'the spread of each arrangement over its starts' in ht})
    add('A8 CONFIRM comment + two others (W178b unexpected 1)', 'P5_15_W178B_FIXES.md', 'note (not typeset)', 'G11 + A78 Appendix E',
        None, None, 'closed', None,
        'No "[CONFIRM — W177]" comment remains at 48099ab: salvage and W173 removed at e926fb7 (G11), the G-rows one at 48099ab '
        '(Addendum 78). Four "[CONFIRM -- W179]" comments are new (l. 2379, 2590, 2625, 2694), all in Appendix E and not typeset '
        '(W179a scope).',
        {'no_W177_confirm_at_head': not any('CONFIRM — W177' in ln for ln in main_lines),
         'w177_confirms_at_9fd19e7': sum('CONFIRM — W177' in ln for ln in w_lines) == 3,
         'salvage_confirm_raw_at_9fd19e7': any('[CONFIRM — W177] eq:salvage_credit' in ln for ln in w_lines),
         'w177_confirms_at_e926fb7_only_G_rows': [ln.strip().startswith('% [CONFIRM — W177] G rows of the frozen T1') for ln in g_lines if 'CONFIRM — W177' in ln] == [True]})
    add('Nomenclature hat-S / hat-E (W178 Decision 5)', 'P5_15_W178_REPORT.md', 'L', 'G12',
        None, '$\\hat{S}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$, $\\hat{E}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$', 'closed; new difference', 'L',
        'The entry exists ("The same additions as fixed parameters of the operational subproblem") and the symbols are used in '
        '2.3.2 (l. 771, 776, 783). New (wording): 2.2.6 l. 697 calls the un-hatted S^Inv, E^Inv "fixed parameters of the '
        'operational subproblem", which the nomenclature reserves for the hatted symbols.',
        {'hat_used_in_2_3_2': ht.count('\\hat{S}^{\\text{Inv}}') >= 2,
         'unhatted_called_fixed_parameters_in_2_2_6': 'the scenario-independent capacity additions $S^{\\text{Inv}}_{e,y^{\\text{Inv}}}$ and $E^{\\text{Inv}}_{e,y^{\\text{Inv}}}$ are fixed parameters of the operational subproblem' in ht})
    add('R14-2 (fuel)', 'P5_15_W178C_LETTER.md Blocked 3', 'L', 'letter v5', None,
        'fuel costs, through the wholesale prices at which conventional generation is costed (Section~3.3)', 'still (narrowed)', 'L',
        f'3.3 l. 1183 states the mechanism ("Conventional generation in the TN is priced at the wholesale energy price of the '
        f'scenario"), so the parenthetical holds; but the reply frames it as what "the introduction now states", and "fuel" '
        f'occurs {fuel_hits} times in main.tex (none in the Introduction).',
        {'sec_3_3_mechanism': 'Conventional generation in the TN is priced at the wholesale energy price of the scenario' in ht,
         'fuel_absent_from_intro': 'fuel' not in intro.lower()})
    add('R14-6 (rate in coordination weights)', 'P5_15_W178C_LETTER.md Blocked 3', 'L', 'letter v5', None,
        'the rate enters the coordination weights (Appendix~A.1); the paper says so (Section~4.6)', 'closed', None,
        'A.1 l. 1771 prints the block weight w_b with (1+r)^{-(y-y_0)}; the "(Section~4.6)" pointer now attaches to the '
        're-weighting statement 4.6 makes.', {'A1_weight': '(1+r)^{-(y-y_0)}' in R._region_text(cl, 'app:admm_updated_agents')})
    add('R210-2 (converter rating "only through")', 'P5_15_W178C_LETTER.md Blocked 3', 'L', 'letter v5', None,
        'entering only the converter limits (Eqs.~(26)--(28)), not the degradation', 'closed', None,
        '(26) P^Ch + P^Dch <= S^Av, (27) the P-Q-S circle and (28) the complementarity normalised by S^Av all carry S^Av; the agent '
        'bound of A.2 (0 <= P <= S^Rated,Unit) is also a converter limit, not cited, which does not make the statement false.')
    add('R2.3 pointer (2.3.2 sentence moved)', 'P5_15_W178C_LETTER.md Blocked 4', 'L', 'letter v5', None,
        'the investment being a fixed parameter of the recourse (Sections~2.2.6 and~2.3.2)', 'closed', None,
        '2.2.6 l. 697 "are fixed parameters of the operational subproblem"; 2.3.2 "whose ratings are the fixed additions".')
    add('LM-3 ("assumption" vs 4.8)', 'P5_15_W178C_LETTER.md Blocked 2', 'M', 'letter v5',
        'which we state as an assumption (Section~4.8)', 'which we state as an inference, not a result (Section~4.8)', 'closed', None,
        '4.8 Scope: "the plan without storage is the search\'s answer under each of them, but this was not run" - an inference, '
        'not a result, as the letter now says.', {'4_8_not_run': 'is the search\'s answer under each of them, but this was not run' in ht})
    add('LM-9 (Appendix E cited for content it lacked)', 'P5_15_W178C_LETTER.md Blocked 2 / A78 ruling 3', 'M', 'A78 Appendix E',
        None, None, 'closed', None,
        'R2.5 l. 159-160 (statistics, post-certification movement, bitwise replays -> E.6, column Replay, "settling slack" in '
        'E.1/E.21) and R3.1 l. 220/222 (full claims table with salvage -> Table E.20) now land on content.',
        {'E_settling_slack': 'settling slack' in R._region_text(cl, 'app:results'),
         'E_bitwise': 'bitwise' in R._region_text(cl, 'app:results'),
         'E_salvage_table': 'Net: net of the terminal salvage credit' in R._region_text(cl, 'app:results')})
    add('4.6-2 (claims table cited before it existed)', 'P5_15_W178B_FIXES.md Blocked 4 / A78 ruling 3', 'M', 'A78 Appendix E',
        'Across the 60 pairwise comparisons of the full claims table', 'Across the 55 comparisons of Table~\\ref{tab:appE_claims}',
        'closed', None, 'See C46-2 / C46-4: the cited table exists, the count is restated (55 with a net figure), the sign flip of '
        'the year pair is stated.')
    add('R37-3 ("no annual figures")', 'P5_15_W178C_LETTER.md Blocked 1(a) / A78 ruling 2', 'M', 'A78 Appendix E',
        None, None, 'closed (L residue accepted)', 'L',
        'The submitted per-block table is gone (tab:cs3_results_summary_years_days absent). Non-monetary per-year figures remain '
        '(4.3 "0.18, 0.20 and 0.27 equivalent full cycles per day in 2025, 2030 and 2035"; Table 13 none), the L residue '
        'Addendum 78 ruling 2 accepted.',
        {'per_block_table_gone': 'cs3_results_summary_years_days' not in ht,
         'per_year_efc_residue': '0.18, 0.20 and 0.27 equivalent full cycles per day in 2025, 2030 and 2035' in ht})
    return {'rows': rows, 'status_counts': dict(Counter(r_['status'] for r_ in rows))}


# ======================================================================================================================
#  guards, output
# ======================================================================================================================
def verify_guards():
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    g = {'own': GUARD.verify(0), 'w178c': C.GUARD.verify(0), 'w177b3r': R.GUARD.verify(0), 'w177b3': P3.GUARD.verify(0),
         'w167': W167.GUARD.verify(0), 'phase_b_record': PB.PARENT_GUARD.verify(0)}
    pk = {'own': dict(PICKLE_COUNTS), 'w178c': dict(C.PICKLE_COUNTS), 'w177b3r': dict(R.PICKLE_COUNTS),
          'w177b3': dict(P3.PICKLE_COUNTS), 'w167': dict(W167.PICKLE_COUNTS)}
    out = {'verify_0_failures': g,
           'counts': {'own': GUARD.counts, 'w178c': C.GUARD.counts, 'w177b3r': R.GUARD.counts, 'w177b3': P3.GUARD.counts,
                      'w167': W167.GUARD.counts, 'phase_b_record': PB.PARENT_GUARD.counts},
           'pickle_counts': pk, 'production_modules_imported': forbidden}
    bad = any(g.values()) or any(v for d in pk.values() for v in d.values()) or forbidden
    if bad:
        _fail(f'guards: {out}')
    out['ok'] = not bad
    return out


def write_md(out, path):
    L = [f'# W179b checks (generated by {SCRIPT_NAME})', '',
         f'Clone HEAD {out["identity"]["clone_head"]}; main.tex {out["identity"]["main_sha256"][:8]}; letter '
         f'{out["identity"]["letter_sha256"][:8]}. Check failures: {len(out["check_failures"])}.', '',
         '## (b) numbers, 4.6-4.8', '', f'{out["b_main_numbers"]["counts"]}', '',
         '| seg | l. | token | status | expected | source | note |', '|---|---|---|---|---|---|---|']
    for r_ in out['b_main_numbers']['rows']:
        L.append(f'| {r_["segment"]} | {r_["line"]} | {r_["token"]} | {r_["status"]} | {r_["expected"]} | {str(r_["source"])[:110]} | {r_["note"] or ""} |')
    L += ['', '## (b) moved / kept', '', '| id | item | how | ok |', '|---|---|---|---|']
    for r_ in out['b_moved_kept']['moved']:
        L.append(f'| {r_["id"]} | {r_["item"]} -> {r_["destination"]} | {r_["how"]} | {r_["ok"]} |')
    for r_ in out['b_moved_kept']['kept']:
        L.append(f'| {r_["id"]} | {r_["item"]} (kept, {r_["section"]}) | - | {r_["ok"]} |')
    L += ['', '## (b) claims', '', '| id | where | rating | basis |', '|---|---|---|---|']
    for r_ in out['b_main_claims']['rows'] + out['b_letter']['claims']:
        L.append(f'| {r_["id"]} | {r_.get("where", r_.get("line"))} | {r_["rating"]} | {r_["basis"][:220]} |')
    L += ['', '## (b) letter numbers on changed lines', '', '| l. v6 | l. v4 | token | inserted | status |', '|---|---|---|---|---|']
    for r_ in out['b_letter']['number_rows']:
        L.append(f'| {r_["line_v6"]} | {r_["line_v4"]} | {r_["token"]} | {r_["inserted_since_v4"]} | {r_["status"]} |')
    P = out['b_pointers']
    L += ['', '## (b) pointers', '', f'{P["n_pointers"]} pointers: {P["counts"]}', '',
          '| l. | text | kind | target | HEAD | 9fd19e7 | verdict | needles |', '|---|---|---|---|---|---|---|---|']
    for r_ in P['pointers']:
        tgt = r_['target'] if isinstance(r_['target'], str) else ', '.join(r_['target'] or [])
        nd = '' if not r_.get('content_needles') else ('all' if all(r_['content_needles'].values()) else
                                                        'ABSENT: ' + '; '.join(k for k, v in r_['content_needles'].items() if not v))
        L.append(f'| {r_["line"]} | {r_["text"].replace("|", "/")} | {r_["kind"]} | {tgt[:55]} | {r_.get("number_head", "")} | '
                 f'{r_.get("number_9fd19e7") or ""} | {r_["verdict"]} | {nd} |')
    L += ['', '## (c) G hunks', '', '| hunk | G | applied | e926fb7 verbatim | HEAD verbatim | tokens |', '|---|---|---|---|---|---|']
    for r_ in out['c_g_hunks']['rows']:
        L.append(f'| {r_["hunk"]} | {r_["g"]} | {r_["applied"]} | {r_["plus_all_verbatim_at_e926fb7"]} | {r_["plus_all_verbatim_at_head"]} | '
                 f'{[(t["token"], t["status"]) for t in r_["inserted_tokens"]]} |')
    L += ['', '## (c) W178 rows', '', '| row | prior | by | status | now | basis |', '|---|---|---|---|---|---|']
    for r_ in out['c_w178_rows']['rows']:
        L.append(f'| {r_["row"]} | {r_["prior_rating"]} | {r_["addressed_by"]} | {r_["status"]} | {r_["rating_now"] or ""} | {r_["basis"][:200]} |')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run():
    if not os.path.isdir(OUT_DIR):
        print(f'REFUSED: {OUT_DIR} does not exist', flush=True)
        sys.exit(1)
    p = os.path.join(OUT_DIR, POST_NAME)
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = ['launch.log', MANIFEST_NAME, MD_NAME, OUT_NAME] + list(TYPING_NAMES)
    missing = [n for n in names if not os.path.isfile(os.path.join(OUT_DIR, n))]
    if missing:
        print(f'REFUSED: post-run inputs missing: {missing}', flush=True)
        sys.exit(1)
    man = {'files': {n: _sha(os.path.join(OUT_DIR, n)) for n in names}, 'script_sha256': _sha(__file__),
           'written_utc': datetime.now(timezone.utc).isoformat()}
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write('\n')
    print(f'written {p}', flush=True)


def main():
    global OUT_DIR
    if '--post-run' in sys.argv:
        post_run()
        return
    if '--out-dir' in sys.argv:
        OUT_DIR = sys.argv[sys.argv.index('--out-dir') + 1]
        if os.path.abspath(OUT_DIR).startswith(REPO + os.sep):
            print('REFUSED: --out-dir is for trial runs outside the repository', flush=True)
            sys.exit(1)
    if not os.path.isdir(OUT_DIR):
        print(f'REFUSED: {OUT_DIR} does not exist (create it with mkdir; it must hold only launch.log)', flush=True)
        sys.exit(1)
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        print(f'REFUSED: {OUT_DIR} already holds {extra}', flush=True)
        sys.exit(1)
    refuse_concurrent()
    out_json, out_md, out_man = (os.path.join(OUT_DIR, n) for n in (OUT_NAME, MD_NAME, MANIFEST_NAME))
    t0 = datetime.now(timezone.utc)
    _log('W179b checks: start')
    out = {'schema': 'p515_s53_w179b_checks', 'task': 'W179b (Addendum 78, W179 items (b) and (c)): 4.6/4.7/4.8 and the letter '
           'changes of Addendum 78; pointer check of the letter v6; G1-G12 and letter v5 against the W178 rows',
           'objective_convention': 'Q = certified gross_operational_cost, settlement excluded; net = Q - terminal salvage credit; '
                                   'value = Q(0) - Q(x); F = Q + I; M = I + Q(x) - Q(0)',
           'rating_scale_W174b': {'H': 'a referee re-deriving from the text gets a different model, number or verdict',
                                  'M': 'imprecise, but not a different result', 'L': 'wording'},
           'started_utc': t0.isoformat(), 'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'script': {'path': SCRIPT_NAME, 'sha256': _sha(__file__)},
           'imported': {m: _sha(m) for m in (W178C_SCRIPT, 'p515_s53_w177b3r_checks.py', 'p515_s53_w177b_part3_checks.py',
                                             'p515_s53_w167_nomenclature_years.py', 'p515_s47_phase_b_record.py',
                                             'gate_result_io.py', 'p513_solve_profile_guard.py')}}
    out['capture_checklist'] = capture_checklist()
    _log('capture-path checklist: all present')
    out['identity'] = identity()
    _log(f"identity: head {out['identity']['clone_head']}, checks {sum(out['identity']['checks'].values())}/{len(out['identity']['checks'])}")
    main_lines = lines_of(_read(MAIN_TEX))
    g_lines = lines_of(show(G_HEAD))
    w_lines = lines_of(show(W178_HEAD))
    v6 = lines_of(_read(LETTER))
    v5 = lines_of(_read(LETTER_V5))
    v4 = lines_of(_read(LETTER_V4))
    S = Src()
    raw, adj, out['numbering'] = task_numbering(main_lines, g_lines)
    _log(f"numbering: {out['numbering']['n_labels_outside_appendix_E_compared']} labels outside E compared, changed "
         f"{len(out['numbering']['labels_changed_outside_appendix_E'])}; E tables {out['numbering']['appendix_E_tables']}")
    out['elsarticle'] = task_elsarticle()
    out['prefixes_confirm'] = task_prefixes_confirm(main_lines, g_lines, w_lines)
    _log(f"prefixes at head: {[(p['line'], p['text']) for p in out['prefixes_confirm']['appendix_ref_prefixes']['head']]}; "
         f"CONFIRM at head {out['prefixes_confirm']['confirm_head_counts']}")
    out['b_main_numbers'] = task_main_numbers(S, main_lines)
    _log(f"(b) 4.6-4.8 numbers: {out['b_main_numbers']['n_tokens']} tokens {out['b_main_numbers']['counts']}")
    out['b_moved_kept'] = task_moved_kept(main_lines, g_lines)
    _log(f"(b) moved {out['b_moved_kept']['n_moved_ok']}/{len(MOVED)}, kept {out['b_moved_kept']['n_kept_ok']}/{len(KEPT)}, "
         f"reproducibility moved whole {out['b_moved_kept']['reproducibility_moved_whole']}")
    out['b_main_claims'] = task_main_claims(S, main_lines)
    _log(f"(b) main claims: {out['b_main_claims']['counts']}")
    out['b_letter'] = task_letter_changes(S, v6, v5, v4)
    _log(f"(b) letter: changed lines {out['b_letter']['changed_lines_v4_to_v6']}; numbers {out['b_letter']['number_counts']}; "
         f"claims {out['b_letter']['claim_counts']}")
    out['b_pointers'] = task_pointers(S, main_lines, adj, raw, v6, v4)
    _log(f"(b) pointers: {out['b_pointers']['n_pointers']} {out['b_pointers']['counts']}; needles absent "
         f"{len(out['b_pointers']['content_absent'])}")
    out['c_g_hunks'] = task_g_hunks(S, main_lines, g_lines, w_lines)
    _log(f"(c) G hunks: {out['c_g_hunks']['applied_counts']}")
    out['c_letter_v5'] = task_letter_v5(v6, v5, v4)
    out['c_w178_rows'] = task_w178_rows(S, main_lines, g_lines, w_lines, v6)
    _log(f"(c) W178 rows: {out['c_w178_rows']['status_counts']}")
    out['frozen_json_unchanged_at_end'] = _sha(FZ) == FZ_SHA
    if not out['frozen_json_unchanged_at_end']:
        _fail('frozen JSON changed during the run')
    out['guards'] = verify_guards()
    nmis = out['b_pointers']['counts'].get('MISMATCH', 0)
    addressed = [r_ for r_ in out['c_w178_rows']['rows'] if r_['addressed_by'] not in ('-',)]
    not_closed = [r_['row'] for r_ in addressed if not r_['status'].startswith('closed')]
    out['predictions'] = {
        '(b) 0 pointer MISMATCH': {'outcome_mismatch': nmis, 'held': nmis == 0},
        '(c) all W178 rows addressed close': {'n_addressed': len(addressed), 'not_closed': not_closed, 'held': not not_closed}}
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'w', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, out_md)
    inputs = sorted({MAIN_TEX, LETTER, DTX, MAIN_APPE, MAIN_9FD_FIXED, LETTER_V4, LETTER_V5, LETTER_V6, DIFF_FOLLOWUP, DIFF_APPE,
                     DIFF_LETTER, CHANGE_LOG, BRIEF, FZ, W178C_JSON, W178B_JSON, W178C_SCRIPT, W178B_SCRIPT, W153_CERT, W153_SALV,
                     W163, W155_SUM, SRP1_JSON, INSTANCE_3X3, EXPORT_T6, F2_SPEC, SUBMITTED, HOOKS_W101, 'p515_s53_w177b3r_checks.py',
                     'p515_s53_w177b_part3_checks.py', 'p515_s53_w167_nomenclature_years.py', 'p515_s47_phase_b_record.py',
                     'gate_result_io.py', 'p513_solve_profile_guard.py', SCRIPT_NAME})
    manifest = {'outputs': {OUT_NAME: _sha(out_json), MD_NAME: _sha(out_md)}, 'out_dir': OUT_DIR,
                'git_show_blobs_sha256': {f'{c}:{p}': hashlib.sha256(show(c, p).encode('utf-8')).hexdigest()
                                          for c in (W178_HEAD, G_HEAD, 'HEAD') for p in ('main.tex', 'response_to_reviewers_draft.tex')},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(out_man, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'written {out_json}, {out_md}, {out_man}; check failures: {len(CHECK_FAILURES)}; guards ok: {out["guards"]["ok"]}; '
         f'pickle {PICKLE_COUNTS}; predictions {[(k, v["held"]) for k, v in out["predictions"].items()]}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
