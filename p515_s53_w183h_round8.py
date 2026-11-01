"""P5.15 Addendum 84 (re-ordered unchanged by Addendum 85 W183g (e)) -- Planner task W183h: ROUND 8 AT THE OVERLEAF HEAD
a87e195. ZERO SOLVES, NO MODEL, NO PICKLE. Read-only on the Overleaf clone (git rev-parse / show / diff / status / log
only; no fetch, pull or checkout), on manuscript_review/, on every committed artifact and on every earlier script.

ITEMS (Addendum 84 W183h; predictions in brackets).
  (a) identity: main.tex = manuscript_review/main_edd7a63_round8.tex, response_to_reviewers_draft.tex = v10,
      highlights.tex = v3, cover_letter.tex = v2 (bytes; the Addendum 85 md5 fingerprints); the clone diff
      edd7a63 -> HEAD hunk by hunk against the expert's four diffs (W181a parse_udiff / compare_hunks) [identical].
  (b) Appendix E: manuscript_review/gen_appE.py run IN-PROCESS with an explicit output path INSIDE this stage's output
      directory (never the cwd default appE.tex, never manuscript_review/appendixE_evaluation_record.tex); bitwise
      location in main.tex at HEAD; verify_appE.py 33 PASS (it regenerates into its own tempfile.mkdtemp()); the entry
      check of W180a/W181a (table_check) against frozen_step6_tables_v1_590088fe.json on the HEAD tables with the
      round-8 over-budget mark $^{b}$ removed from the E.20 / E.21 rows only (each removal recorded; E.23's own $^{b}$,
      "the start reported in Table 8", is left in place); cells changed edd7a63 -> HEAD (W181a diff_tables) raw and
      with the marks removed; the marks on exactly E.21 rows 5, 6, 9, 10, 11, 38, 39 and E.20 rows 4, 5, 8, 9, 10, 26,
      27, 30, 31, each on a plan that is over the budget (W182d's committed I(x) table), every over-budget evaluation
      marked; the two caption legends [as stated: 33 PASS, 0 MISMATCH, 0 entries changed, the marks as listed].
  (c) letter v10 pointers (W182c method, static numbering): the label map 8f051a2 -> HEAD and the v9 pointer rows are
      recomputed at HEAD by W182c's own functions (imported, not edited); every v10 pointer (W177b-3' extractor, plus
      a lettered-number extractor for "Fig.~E.19" / "Eqs.~(E.1)--(E.2)", which the W177b-3' regex does not read) is
      paired with its v9 pointer (letter line map, then the pointer whose expected HEAD number equals the printed one,
      then in order); its verdict is MATCH when the printed number is the HEAD number of the v9 pointer's target;
      a v10 pointer with no v9 counterpart, or whose v9 target was removed, takes a DECLARED target (DECLARED_TARGETS,
      each from STEP6_ROUND8_CORRECTIONS.md section C) or is AMBIGUOUS; SUBMITTED carried from W180c through v9.
      Citation order [42]/[43] [0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED].
  (d) re-rating at HEAD of W182d's 16 listed A7 sentences (original quote present? declared defect phrases absent,
      declared fix phrases present at the mapped place) and of W182d's 34 A8 places (mark / probe text at the place,
      in its paragraph, covered by the Section 3.5 sentence, withdrawn) [0 of 16 remaining; every over-budget point
      labelled at every table / figure occurrence].
  (e) numbers on the main.tex '+' lines of git diff -U0 edd7a63 HEAD outside the Appendix E block (the 21 edits), by
      the W182b method (W182b imported, not edited: its tokenizer, spec constructors, evaluator, negative control;
      DECLARATIONS VERSION W183h, keyed by the HEAD line; specs of unchanged tokens re-used from the W182b / W177b
      declarations of the same sentence and RE-EVALUATED here) against the frozen JSON and the named records:
      growth 1.00 / 2.50 % (case9_operational_data.xlsx), 1.15-1.91 MEUR (W182d's I(x) table), "about 3 %" and 2037
      (W182g's committed records). Supplement: E.1's 2040 / 2035 / 2037 [0 MISMATCH].
  (f) 0 "[CONFIRM"; "<word>~\\ref{app:" (W180c's general pattern, label-word prefixes separated); blue on the 23 A5
      lines of W182c (mapped edd7a63 -> HEAD) and on the five A5 captions [hold].
  Also: the letter's prediction-record bullet, quoted only (Addendum 84 ruling 1: the author's call).

IMPORTED, NOT EDITED: p515_s53_w182c_labels_pointers_appE.py (W) and its chain (W179b, W178c, W177b-3', W177b-3, W167,
S47 phase-B record, W181a); p515_s53_w182b_number_check.py (NB) and its chain (W178b, W177b-1', W177b-1, W177a, ...,
W160); p515_s53_w180c_text_letter.py (GENERAL_PREFIX, LABEL_WORDS). None of their output directories is written.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import; every guard of the imported chains
verified at exactly 0 at the end; pickle.load / pickle.loads blocked for the whole run (counters verified 0); refuses to
run concurrently with another copy of itself; refuses unless the output directory exists and holds nothing but
launch.log. JSON written only through gate_result_io. Capture-path checklist asserted before any task runs (exit 4).

NEGATIVE CONTROLS (in memory, or a tempfile.mkdtemp() directory outside the repository): an altered '+' line of each
expert diff; an altered table entry; a removed and an added $^{b}$ mark; a one-byte change of the generated block;
altered letter pointers; a planted A7 quote; every rated number perturbed (W182b method); a planted [CONFIRM and an
unwrapped caption. Each must be refused (exit 3 otherwise).

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p data/SRP1/Results/P515S55/w183h_round8 && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183h_round8.py \\
      > data/SRP1/Results/P515S55/w183h_round8/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
      --out data/SRP1/Results/P515S55/w183h_round8/w183h_bool_typing_test.json \\
      > data/SRP1/Results/P515S55/w183h_round8/w183h_bool_typing_test.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w183h_round8.py --post-run
  (trial runs: --trial-out-dir <absolute path outside the repository>)
Exit: 0 = written, every integrity check holds, guards at 0 (a MISMATCH or a failed prediction is a FINDING and never
changes the exit code); 3 = written, an integrity check failed (listed); 1 / 4 = precondition / capture refusal.
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183h round 8 at Overleaf a87e195 (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W183h: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W183h: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import collections  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s53_w182c_labels_pointers_appE as W  # noqa: E402  (W182c; its chain arms its own guards; NOT edited)
import p515_s53_w182b_number_check as NB  # noqa: E402  (W182b; its chain arms its own guards; NOT edited)
import p515_s53_w180c_text_letter as W180C  # noqa: E402  (W180c; GENERAL_PREFIX / LABEL_WORDS; NOT edited)
import gate_result_io as GRIO  # noqa: E402

A, B, C_, R = W.A, W.B, W.C, W.R
W177A, W164, W174, B1, B1R, W178B = NB.W177A, NB.W164, NB.W174, NB.B1, NB.B1R, NB.W178B
os.chdir(REPO)
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s53_w183h_round8'
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
S55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
OUT_DIR = os.path.join(S55, 'w183h_round8')
OUT_NAMES = {'json': 'w183h_round8.json', 'md': 'w183h_round8.md', 'man': 'manifest_sha256.json',
             'gen': 'appE_generated_w183h.tex', 'verify': 'verify_appE_output.txt', 'log': 'launch.log',
             'typing_json': 'w183h_bool_typing_test.json', 'typing_log': 'w183h_bool_typing_test.log',
             'post': 'manifest_post_run_sha256.json'}
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
HEAD_FULL = 'a87e1950fbdbb3667f37e64a43c298ba675e6cff'
PREV = 'edd7a63'
PREV_FULL = 'edd7a63a6c2ea06c1db17e3b6941bde8aa6d6838'
OLD = W.OLD                      # 8f051a2: the numbering letter v9 was written against (W182c)
MR = 'manuscript_review'
LETTER_NAME = 'response_to_reviewers_draft.tex'
FILES = ('main.tex', LETTER_NAME, 'highlights.tex', 'cover_letter.tex')
REVIEW_COPY = {'main.tex': os.path.join(MR, 'main_edd7a63_round8.tex'),
               LETTER_NAME: os.path.join(MR, 'response_to_reviewers_v10.tex'),
               'highlights.tex': os.path.join(MR, 'highlights_v3.tex'),
               'cover_letter.tex': os.path.join(MR, 'cover_letter_v2.tex')}
EXPERT_DIFF = {'main.tex': os.path.join(MR, 'main_edd7a63_round8.diff'),
               LETTER_NAME: os.path.join(MR, 'letter_v9_v10.diff'),
               'highlights.tex': os.path.join(MR, 'highlights_v2_v3.diff'),
               'cover_letter.tex': os.path.join(MR, 'cover_letter_v1_v2.diff')}
EXPECTED_HUNKS = {'main.tex': 31, LETTER_NAME: 12, 'highlights.tex': 1, 'cover_letter.tex': 2}
MD5_ADDENDUM_85 = {'main.tex': '39f6fd86', LETTER_NAME: '2aeb33b2', 'highlights.tex': '4cedd45c',
                   'cover_letter.tex': '88eee8ce'}
SHA_HEAD = {'main.tex': 'cec2e2982dfa1509e6e066a208521fb1eea40d5c71f1cdf71a9bf00a12b3520b',
            LETTER_NAME: '982482a1e0705aa1fd20e5861214e8bab9b828f2fe83b0b830ce5b3009d23c60',
            'highlights.tex': '13e37f6d452de934839042cf470e51787a1eb1a1b31417e9e16315843e1bf7af',
            'cover_letter.tex': 'de7a91748074b3ce221c89eee40845ff4d0c2f0995480f36b6593d0806aecbea'}
LETTER_V9 = os.path.join(MR, 'response_to_reviewers_v9.tex')
SUBMITTED_COVER = os.path.join('manuscript_submitted', 'cover_letter.tex')
GEN_REL, VERIFY_REL, APPX_REL = A.GEN_REL, A.VERIFY_REL, A.APPX_REL
FZ_REL, FZ_SHA = A.FZ_REL, A.FZ_SHA
W182C_JSON = os.path.join(S53, 'w182c_labels_pointers_appE', 'w182c_labels_pointers_appE.json')
W182D_JSON = os.path.join(S53, 'w182d_claims', 'w182d_claims.json')
W182G_JSON = os.path.join(S53, 'w182g_cost_horizon_budget', 'w182g_cost_horizon_budget.json')
W180C_JSON = W.W180C_JSON
STEP6_R8 = 'STEP6_ROUND8_CORRECTIONS.md'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
MARK = '$^{b}$'
E21_MARKED_EXPECTED = [5, 6, 9, 10, 11, 38, 39]
E20_MARKED_EXPECTED = [4, 5, 8, 9, 10, 26, 27, 30, 31]
CHECK_FAILURES = []
IMPORTED_SCRIPTS = ('p515_s53_w182c_labels_pointers_appE.py', 'p515_s53_w182b_number_check.py',
                    'p515_s53_w180c_text_letter.py', 'p515_s53_w181a_identity_appendixE.py', 'p515_s53_w179b_checks.py',
                    'p515_s53_w178c_letter_checks.py', 'p515_s53_w177b3r_checks.py', 'p515_s53_w178b_number_check.py',
                    'p515_s53_w177b1r_number_check.py', 'p515_s53_w177b_number_check.py',
                    'p515_s53_w177a_section4_frame_check.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


def _sha(rel):
    with open(os.path.join(REPO, rel), 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _shab(b):
    return hashlib.sha256(b).hexdigest()


def _rd(rel):
    with open(os.path.join(REPO, rel), 'rb') as fh:
        return fh.read()


def _git(args, cwd=REPO, binary=False):
    r = subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, text=not binary, check=True)
    return r.stdout


def show(rev, path):
    return _git(['show', f'{rev}:{path}'], cwd=CLONE)


def ws(s):
    return re.sub(r'\s+', ' ', s).strip()


def line_map(old_text, new_text):
    """{old line: new line} over equal blocks and equal-size replace blocks (1-based)."""
    a, b = old_text.split('\n'), new_text.split('\n')
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    mp = {}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal' or (op == 'replace' and i2 - i1 == j2 - j1):
            for k in range(i2 - i1):
                mp[i1 + k + 1] = j1 + k + 1
    return mp


def other_copies():
    out = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me, found = os.getpid(), []
    for ln in out.splitlines():
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or not parts[0].isdigit() or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(os.path.basename(a) == SCRIPT_NAME for a in argv[1:]) \
                and '--post-run' not in argv:
            found.append(ln.strip())
    return found


# ======================================================================================================================
#  (a) identity
# ======================================================================================================================
def task_identity():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    parent = _git(['rev-parse', 'HEAD^'], cwd=CLONE).strip()
    prev_full = _git(['rev-parse', PREV], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    untracked = _git(['status', '--porcelain'], cwd=CLONE).strip()
    files_changed = [x for x in _git(['diff', '--name-only', PREV_FULL, head], cwd=CLONE).split('\n') if x]
    out = {'clone_head': head, 'clone_head_parent': parent, 'clone_head_date': _git(['log', '-1', '--format=%cI'], cwd=CLONE).strip(),
           'clone_log_top3': _git(['log', '--format=%h %cI %s', '-3'], cwd=CLONE).strip().split('\n'),
           'clone_tracked_modified': status, 'clone_porcelain_all': untracked, 'files_changed_edd7a63_head': files_changed,
           'files': {}, 'diffs': {}, 'diff_negative_controls': {}}
    checks = {'head_is_a87e195': head == HEAD_FULL, 'head_parent_is_edd7a63': parent == PREV_FULL,
              'edd7a63_resolves': prev_full == PREV_FULL, 'clone_tracked_clean': status == '',
              'files_changed_are_the_four': sorted(files_changed) == sorted(FILES)}
    texts = {}
    for f in FILES:
        wc = _rd(os.path.join(CLONE, f))
        hb = _git(['show', f'HEAD:{f}'], cwd=CLONE, binary=True)
        rc = _rd(REVIEW_COPY[f])
        md5 = hashlib.md5(wc).hexdigest()
        out['files'][f] = {'sha256': _shab(wc), 'md5': md5, 'lines': wc.count(b'\n'), 'working_copy_equals_HEAD_blob': wc == hb,
                           'review_copy': REVIEW_COPY[f], 'review_copy_sha256': _shab(rc), 'byte_identical_to_review_copy': wc == rc,
                           'md5_prefix_addendum_85': MD5_ADDENDUM_85[f], 'md5_matches_addendum_85': md5.startswith(MD5_ADDENDUM_85[f]),
                           'sha256_declared': SHA_HEAD[f], 'sha256_matches_declared': _shab(wc) == SHA_HEAD[f]}
        texts[f] = wc.decode('utf-8')
        clone_diff_text = _git(['diff', PREV_FULL, head, '--', f], cwd=CLONE)
        got = A.parse_udiff(clone_diff_text).get(f, [])
        exp_all = A.parse_udiff(_rd(EXPERT_DIFF[f]).decode('utf-8'), default_file=f)
        exp = exp_all.get(f, [])
        cmp_ = A.compare_hunks(got, exp)
        cmp_['expert_diff'] = EXPERT_DIFF[f]
        cmp_['expert_diff_sha256'] = _sha(EXPERT_DIFF[f])
        cmp_['expert_diff_files_parsed'] = sorted(exp_all)
        cmp_['clone_diff_sha256'] = _shab(clone_diff_text.encode('utf-8'))
        cmp_['expected_hunks'] = EXPECTED_HUNKS[f]
        cmp_['counts_consistent'] = all(h['counts_consistent'] for h in got + exp)
        out['diffs'][f] = cmp_
        # negative control: one '+' line of the expert diff altered in memory -> exactly that hunk is flagged
        mut = json.loads(json.dumps(exp))
        hk = next(k for k, h in enumerate(mut) if any(x[0] == '+' for x in h['lines']))
        j = next(j for j, x in enumerate(mut[hk]['lines']) if x[0] == '+')
        mut[hk]['lines'][j] = ['+', mut[hk]['lines'][j][1] + 'X']
        mut = [dict(h, lines=[tuple(x) for x in h['lines']]) for h in mut]
        cm = A.compare_hunks(got, mut)
        out['diff_negative_controls'][f] = {'hunk_mutated': hk + 1,
                                            'refused': (not cm['every_hunk_pm_equal'])
                                            and [p['hunk'] for p in cm['per_hunk'] if not p['pm_lines_equal']] == [hk + 1]}
    # the cover letter's expert diff is against the submitted cover letter; edd7a63's cover letter is that file
    out['cover_letter_edd7a63_equals_manuscript_submitted'] = (show(PREV_FULL, 'cover_letter.tex').encode('utf-8')
                                                               == _rd(SUBMITTED_COVER))
    out['letter_v9_equals_edd7a63_letter'] = show(PREV_FULL, LETTER_NAME).encode('utf-8') == _rd(LETTER_V9)
    out['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'(a) identity check {k} failed')
    for f in FILES:
        if not out['files'][f]['working_copy_equals_HEAD_blob']:
            _fail(f'(a) {f}: working copy differs from the HEAD blob')
        if not out['diffs'][f]['counts_consistent']:
            _fail(f'(a) {f}: a diff hunk is not count-consistent')
        if not out['diff_negative_controls'][f]['refused']:
            _fail(f'(a) {f}: the diff negative control was not refused')
    out['outcome'] = {f: {'byte_identical': out['files'][f]['byte_identical_to_review_copy'],
                          'md5_addendum_85': out['files'][f]['md5_matches_addendum_85'],
                          'hunks_clone_expert': [out['diffs'][f]['n_hunks_clone'], out['diffs'][f]['n_hunks_expert']],
                          'every_hunk_pm_equal': out['diffs'][f]['every_hunk_pm_equal'],
                          'ranges_equal': out['diffs'][f]['every_hunk_range_equal'],
                          'context_equal': out['diffs'][f]['every_hunk_context_equal']} for f in FILES}
    out['prediction_held'] = all(v['byte_identical'] and v['md5_addendum_85'] and v['every_hunk_pm_equal']
                                 and v['hunks_clone_expert'] == [EXPECTED_HUNKS[f]] * 2 for f, v in out['outcome'].items())
    return out, texts


# ======================================================================================================================
#  (b) Appendix E
# ======================================================================================================================
def strip_marks(lines, raw, tables=('E.20', 'E.21')):
    """the HEAD lines with $^{b}$ removed from the rows of the named tables only; every removal recorded."""
    L, removed = list(lines), []
    for t in tables:
        for k, r in raw[t]['rows'].items():
            i = r['line'] - 1
            n = L[i].count(MARK)
            if n:
                removed.append({'table': t, 'row': k, 'line': r['line'], 'n_marks': n,
                                'cells': [c for c in r['cells'] if MARK in c]})
                L[i] = L[i].replace(MARK, '')
    return L, removed


def _ints(s):
    return [int(x) for x in re.findall(r'\d+', A.strip_markers(s)[0])]


def mark_check(raw, W182D):
    """marks on E.20 / E.21 rows: the set, their placement on over-budget plans, completeness (W182d's I(x) table)."""
    ob = {e['eval']: e for e in W182D['A8_1_budget']['over_budget_evaluations']}
    evals = {e['eval']: e for e in W182D['A8_1_budget']['evaluations']}
    ob_plans = set()
    for e in ob.values():
        p = A.decode_plan(e['plan_text'])
        ob_plans |= {(n, v[0], v[1], e['year']) for n, v in p.items()}
    e21 = {}
    for k, r in raw['E.21']['rows'].items():
        plan_cell = r['cells'][1]
        e21[int(k)] = {'marked': MARK in plan_cell, 'n_marks_row': sum(c.count(MARK) for c in r['cells']),
                       'plan_cell': plan_cell, 'over_budget_w182d': int(k) in ob,
                       'w182d_over_budget_flag': (evals.get(int(k)) or {}).get('over_budget')}
    e21_marked = sorted(k for k, v in e21.items() if v['n_marks_row'])
    e21_bad = [k for k, v in e21.items() if v['marked'] != v['over_budget_w182d'] or v['n_marks_row'] not in (0, 1)
               or (v['marked'] and MARK not in v['plan_cell'])]
    e20 = {}
    for k, r in raw['E.20']['rows'].items():
        comp, ev = r['cells'][1], r['cells'][2]
        ev_ids = _ints(ev)
        n_ob = len({e for e in ev_ids if e in ob})
        frags = re.findall(r'([0-9.]+)/([0-9.]+)\$\^\{b\}\$', comp)
        node = re.match(r'\s*(\d+):', A.strip_markers(comp)[0])
        placed_ok = all(node and (node.group(1), float(s), float(e_), 2025) in ob_plans for s, e_ in frags)
        e20[int(k)] = {'comparison': comp, 'evals': ev_ids, 'n_marks': comp.count(MARK),
                       'n_marks_other_cells': sum(c.count(MARK) for i, c in enumerate(r['cells']) if i != 1),
                       'n_over_budget_evals': n_ob, 'marked_plans': frags, 'marks_on_over_budget_plans': placed_ok}
    e20_marked = sorted(k for k, v in e20.items() if v['n_marks'])
    e20_bad = [k for k, v in e20.items() if v['n_marks'] != v['n_over_budget_evals'] or not v['marks_on_over_budget_plans']
               or v['n_marks_other_cells']]
    cap20 = ' '.join(x['text'] if isinstance(x, dict) else str(x) for x in raw['E.20']['caption'])
    cap21 = ' '.join(x['text'] if isinstance(x, dict) else str(x) for x in raw['E.21']['caption'])
    legend20 = '$^{b}$: investment above the 1~M\\euro{} budget, a valuation probe outside the admissible set' in cap20
    legend21 = '$^{b}$: investment above the 1~M\\euro{} budget (valuation probe outside the admissible set)' in cap21
    other_tables = {t: sum(c.count(MARK) for r in raw[t]['rows'].values() for c in r['cells']) for t in ('E.22', 'E.23', 'E.24')}
    return {'over_budget_evaluations_w182d': sorted(ob), 'e21_marked_rows': e21_marked, 'e20_marked_rows': e20_marked,
            'e21_expected': E21_MARKED_EXPECTED, 'e20_expected': E20_MARKED_EXPECTED,
            'e21_exact': e21_marked == E21_MARKED_EXPECTED, 'e20_exact': e20_marked == E20_MARKED_EXPECTED,
            'e21_rows_inconsistent': e21_bad, 'e20_rows_inconsistent': e20_bad,
            'e21_marked_detail': {k: v for k, v in e21.items() if v['n_marks_row'] or v['over_budget_w182d']},
            'e20_marked_detail': {k: v for k, v in e20.items() if v['n_marks'] or v['n_over_budget_evals']},
            'legend_e20_caption': legend20, 'legend_e21_caption': legend21,
            'marks_in_other_appendix_tables': other_tables,
            'note_e23': 'E.23 carries its own $^{b}$ ("the start reported in Table~\\ref{tab:res_benchmark}", its note), '
                        'present before round 8; it is not the over-budget mark and is not removed or counted here',
            'holds': (e21_marked == E21_MARKED_EXPECTED and e20_marked == E20_MARKED_EXPECTED and not e21_bad and not e20_bad
                      and legend20 and legend21)}


def frozen_ctx():
    J = {k: A._json(v) for k, v in A.INPUTS.items() if v.endswith('.json')}
    T = J['FROZEN_JSON']['tables']
    ALL = dict(T['cells'])
    ALL.update(T['cells_appended_w160'])
    for k, c in ALL.items():
        c.setdefault('cell', k)
    return J, T, ALL, A.cell_conditions(J, ALL)


def entry_totals(R_):
    return {'entries': sum(s['entries'] for s in R_['summary'].values()),
            'MATCH': sum(s['MATCH'] for s in R_['summary'].values()),
            'MISMATCH': sum(s['MISMATCH'] for s in R_['summary'].values()),
            'not_frozen_agrees': sum(s['NOT_FROZEN_AGREES'] for s in R_['summary'].values()),
            'not_frozen_disagrees': sum(s['NOT_FROZEN_DISAGREES'] for s in R_['summary'].values())}


def task_appendix_e(out_dir, texts, W182D):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    gen_full = os.path.join(od, OUT_NAMES['gen'])
    gen_bytes, gen_stdout = A.run_generator(gen_full)
    ver = A.run_verifier()
    with open(os.path.join(od, OUT_NAMES['verify']), 'x', encoding='utf-8') as fh:
        fh.write(ver['output'])
    main_b = texts['main.tex'].encode('utf-8')
    prev_t = show(PREV_FULL, 'main.tex')
    lines, plines = texts['main.tex'].split('\n'), prev_t.split('\n')
    occ = A.locate_bytes(main_b, gen_bytes)
    blk, pblk = A.appendix_block(lines), A.appendix_block(plines)
    committed = _rd(APPX_REL)
    bit = {'generator': {'path': GEN_REL, 'sha256': _sha(GEN_REL), 'output': os.path.join(out_dir, OUT_NAMES['gen']),
                         'output_sha256': _shab(gen_bytes), 'output_lines': gen_bytes.count(b'\n'), 'stdout': gen_stdout.strip()},
           'equals_committed_appendix': gen_bytes == committed, 'committed_appendix_sha256': _shab(committed),
           'equals_verifier_regeneration': ver['tempdir_output_sha256'] == _shab(gen_bytes),
           'occurrences_in_main_head': occ, 'n_occurrences': len(occ),
           'block_head_lines': [blk[0] + 1, blk[1]], 'block_edd7a63_lines': [pblk[0] + 1, pblk[1]]}
    if occ:
        o0 = occ[0]
        bit['block_lines_outside_generated_range'] = [{'line': i + 1, 'text': lines[i]} for i in range(blk[0], blk[1])
                                                      if not (o0['start_line'] <= i + 1 <= o0['end_line'])]
    bit['bitwise'] = (len(occ) == 1 and occ[0]['starts_at_line_start'] and occ[0]['ends_at_line_end']
                      and bit['equals_committed_appendix'] and bit['equals_verifier_regeneration'])
    verifier = {'path': VERIFY_REL, 'sha256': _sha(VERIFY_REL), 'n_pass': ver['n_pass'], 'n_fail': ver['n_fail'],
                'fail_lines': ver['fail_lines'], 'exception': ver['exception'],
                'tempdir_output_sha256': ver['tempdir_output_sha256'],
                'note': 'verify_appE.py regenerates through gen_appE.py into its own tempfile.mkdtemp(); it never reads main.tex'}
    # entry check (W180a / W181a table_check) on the HEAD tables with the over-budget marks removed (E.20 / E.21 only)
    J, T, ALL, COND = frozen_ctx()
    raw_head, raw_prev = A.raw_tables(lines), A.raw_tables(plines)
    stripped, removed = strip_marks(lines, raw_head)
    R_ = A.table_check(stripped, ALL, COND, J, T)
    RP = A.table_check(plines, ALL, COND, J, T)
    marks = mark_check(raw_head, W182D)
    td_raw = A.diff_tables(raw_prev, raw_head)
    td_str = A.diff_tables(raw_prev, A.raw_tables(stripped))
    mark_only = all(c['new'].replace(MARK, '') == c['old'] and MARK in c['new'] for c in td_raw['cells'])
    cells_raw = [{k: c[k] for k in ('table', 'row', 'col', 'column', 'old', 'new')} for c in td_raw['cells']]
    BD = A.block_diff(plines[pblk[0]:pblk[1]], pblk[0], lines[blk[0]:blk[1]], blk[0], lines)
    cls_old, cls_new = A.line_classes(raw_prev, plines), A.line_classes(raw_head, lines)
    for h in BD:
        h['class'] = A.classify_hunk(h, plines, lines, cls_old, cls_new)
    entries = {'head_marks_removed': {'summary': R_['summary'], 'totals': entry_totals(R_), 'bad': R_['bad'],
                                      'mapping_problems': R_['mapping_problems']},
               'edd7a63': {'summary': RP['summary'], 'totals': entry_totals(RP), 'bad': RP['bad'],
                           'mapping_problems': RP['mapping_problems']},
               'marks_removed_for_the_entry_check': removed, 'n_rows_marks_removed': len(removed),
               'e21_rows_numbered_1_61': [r['no'] for r in R_['E21']] == list(range(1, 62)),
               'e20_rows_numbered_1_58': [r['no'] for r in R_['E20']] == list(range(1, 59)),
               'e21_map_one_to_one_onto_frozen_cells': (len(set(R_['idmap'].values())) == len(ALL)
                                                        and set(R_['idmap'].values()) == set(ALL)),
               'e21_map_same_as_edd7a63': R_['idmap'] == RP['idmap']}
    diffs = {'raw': {'n_cells_compared': td_raw['n_cells_compared'], 'n_cells_changed': len(td_raw['cells']),
                     'cells_changed': cells_raw, 'every_changed_cell_is_a_mark_insertion_only': mark_only,
                     'row_set_changes': td_raw['row_set_changes'], 'captions': td_raw['captions'], 'notes': td_raw['notes'],
                     'group_headers': td_raw['group_headers'], 'other_lines': td_raw['other_lines']},
             'marks_removed': {'n_cells_compared': td_str['n_cells_compared'], 'n_cells_changed': len(td_str['cells']),
                               'cells_changed': [{k: c[k] for k in ('table', 'row', 'col', 'column', 'old', 'new')}
                                                 for c in td_str['cells']]},
             'block_hunks_edd7a63_to_head': [{k: h[k] for k in ('op', 'lines_old', 'lines_new', 'section', 'kind', 'class')}
                                             | {'n_old': len(h['old']), 'n_new': len(h['new']),
                                                'old_first': ws(' '.join(h['old']))[:200], 'new_first': ws(' '.join(h['new']))[:200]}
                                             for h in BD]}
    out = {'bitwise': bit, 'verifier': verifier, 'entries': entries, 'marks': marks, 'table_diffs': diffs}
    tot = entries['head_marks_removed']['totals']
    out['outcome'] = {'bitwise': bit['bitwise'], 'verify_33_pass': ver['n_pass'] == 33 and ver['n_fail'] == 0 and not ver['exception'],
                      'entries': tot['entries'], 'mismatch': tot['MISMATCH'], 'not_frozen_disagrees': tot['not_frozen_disagrees'],
                      'entries_changed_vs_edd7a63_marks_removed': len(td_str['cells']),
                      'cells_changed_vs_edd7a63_raw': len(td_raw['cells']), 'raw_changes_are_mark_insertions_only': mark_only,
                      'marks_as_stated': marks['holds']}
    out['prediction_held'] = (bit['bitwise'] and out['outcome']['verify_33_pass'] and tot['MISMATCH'] == 0
                              and tot['not_frozen_disagrees'] == 0 and len(td_str['cells']) == 0 and marks['holds'])
    for k in ('e21_rows_numbered_1_61', 'e20_rows_numbered_1_58', 'e21_map_one_to_one_onto_frozen_cells',
              'e21_map_same_as_edd7a63'):
        if entries[k] is not True:
            _fail(f'(b) entry-check integrity {k} failed')
    if entries['edd7a63']['totals']['MISMATCH'] or entries['edd7a63']['totals']['not_frozen_disagrees'] \
            or entries['edd7a63']['mapping_problems'] or entries['head_marks_removed']['mapping_problems']:
        _fail('(b) the entry check does not reproduce 0 MISMATCH at edd7a63, or has mapping problems')
    if ver['exception'] or not gen_stdout.startswith('ok '):
        _fail(f'(b) generator / verifier did not run cleanly: {gen_stdout!r} {ver["exception"]}')
    ctx = {'gen_bytes': gen_bytes, 'lines': lines, 'raw_head': raw_head, 'stripped': stripped, 'J': J, 'T': T, 'ALL': ALL,
           'COND': COND, 'blk': blk}
    return out, ctx


def controls_b(bctx, W182D):
    out = []
    lines, raw = bctx['lines'], bctx['raw_head']
    # an altered table entry (E.21 row 1, band) -> MISMATCH at exactly that entry
    raw_s = A.raw_tables(bctx['stripped'])
    band_row = next(k for k, r in raw_s['E.21']['rows'].items() if r['cells'][8] != '---')
    L2, old, new = A.mutate_cell(bctx['stripped'], raw_s, 'E.21', band_row, 8,
                                 lambda s: f"{A.parse_num(s) + 0.1:.{A.dec_places(s)}f}")
    R2 = A.table_check(L2, bctx['ALL'], bctx['COND'], bctx['J'], bctx['T'])
    out.append({'id': f'b1 entry: E.21 row {band_row} band altered', 'old': old, 'new': new,
                'bad': [{k: b.get(k) for k in ('table', 'row', 'key', 'status')} for b in R2['bad']],
                'refused': len(R2['bad']) == 1 and R2['bad'][0]['table'] == 'E.21' and str(R2['bad'][0]['row']) == str(band_row)})
    # a removed mark (E.21 row 39) and an added mark (E.21 row 4, 7: 0.75/3, admissible)
    L3 = list(lines)
    i39 = raw['E.21']['rows']['39']['line'] - 1
    L3[i39] = L3[i39].replace(MARK, '')
    m3 = mark_check(A.raw_tables(L3), W182D)
    out.append({'id': 'b2 mark removed from E.21 row 39', 'e21_marked': m3['e21_marked_rows'], 'refused': not m3['holds']})
    L4 = list(lines)
    i4 = raw['E.21']['rows']['4']['line'] - 1
    cells = A.row_cells(L4[i4])
    cells[1] = cells[1] + MARK
    L4[i4] = re.match(r'^\s*', L4[i4]).group(0) + ' & '.join(cells) + ' \\\\'
    m4 = mark_check(A.raw_tables(L4), W182D)
    out.append({'id': 'b3 mark added to E.21 row 4 (an admissible plan)', 'e21_marked': m4['e21_marked_rows'],
                'inconsistent': m4['e21_rows_inconsistent'], 'refused': not m4['holds']})
    L5 = list(lines)
    i26 = raw['E.20']['rows']['26']['line'] - 1
    L5[i26] = L5[i26].replace('1/4$^{b}$', '1/4')
    i30 = raw['E.20']['rows']['30']['line'] - 1
    L5[i30] = L5[i30].replace('0.75/3 $\\to$', '0.75/3$^{b}$ $\\to$')
    m5 = mark_check(A.raw_tables(L5), W182D)
    out.append({'id': 'b4 E.20: mark dropped from row 26 and misplaced on the admissible 0.75/3 of row 30',
                'inconsistent': m5['e20_rows_inconsistent'], 'refused': not m5['holds'] and {26, 30} <= set(m5['e20_rows_inconsistent'])})
    gb = bctx['gen_bytes']
    j = gb.find(b'Comparisons')
    j = j if j >= 0 else len(gb) // 2
    mut = gb[:j] + (b'X' if gb[j:j + 1] != b'X' else b'Y') + gb[j + 1:]
    main_b = '\n'.join(lines).encode('utf-8')
    out.append({'id': 'b5 generated block with one byte changed is not located in main.tex',
                'refused': len(A.locate_bytes(main_b, gb)) == 1 and len(A.locate_bytes(main_b, mut)) == 0})
    return out


# ======================================================================================================================
#  (c) letter v10 pointers
# ======================================================================================================================
NUM_L = r'\(?[A-E]\.\d+\)?'
LET_PTR = re.compile(r'(?<![\w.])(Tables?|Figures?|Figs?\.|Eqs?\.|Equations?)(?:~|\s)\s*(' + NUM_L
                     + r'(?:\s*(?:--|,\s*and|,|and)\s*~?' + NUM_L + r')*)')

# v10 pointers with no v9 counterpart, or whose v9 target was removed: the target is declared from
# STEP6_ROUND8_CORRECTIONS.md section C (the expert's retargeting) or from the letter's own sentence.
DECLARED_TARGETS = {
    (58, 'section', '4.3'): {'label': 'subsec:res_degradation',
                             'why': 'R3.6 scope (STEP6 C): "the ageing comparison of Section 4.3 re-evaluates a fixed plan" '
                                    '-- the degradation results subsection (the ageing table, Table 7)'},
    (136, 'section', '2.2'): {'label': 'subsec:master',
                              'why': 'STEP6 C: "Sections 2.2.6 and 2.3.2" -> "Sections 2.2 and 2.3.1" (the coupling '
                                     'subsubsection is gone); 2.2 is the subsection that held it'},
    (229, 'section', '2.2.6'): {'label': 'subsubsec:certification',
                                'why': 'L-09 rewritten (STEP6 C): "under the acceptance rule of Section 2.2.6" -- the '
                                       'recourse-evaluation and numerical-acceptance subsubsection'},
    (231, 'appendix', 'A.5'): {'heading_of_label': 'eq:acceptance_tolerance',
                               'why': 'L-10 rewritten (STEP6 C): "The tolerance is set by tau = delta_R V/4 ... '
                                      '(Appendix A.5)" -- the appendix subsection holding eq:acceptance_tolerance'},
}


def lettered_pointers(path):
    t = R._read(path)
    s = R._blank_cmd(t, r'\rcomment')
    s = '\n'.join(re.sub(r'(?<!\\)%.*', '', ln) for ln in s.split('\n'))
    doc = s.find(r'\begin{document}')
    atoms = []
    for m in LET_PTR.finditer(s, doc):
        kw, lst = m.group(1), m.group(2)
        for p in re.split(r'\s*(?:,\s*and|,|and)\s*~?', lst):
            p = p.strip()
            if '--' in p:
                a, b = [x.strip('() ~') for x in p.split('--')]
                nums, rng = [a, b], True
            else:
                nums, rng = [p.strip('() ~')], False
            atoms.append({'line': s.count('\n', 0, m.start()) + 1, 'keyword': kw, 'kind': R.KW.get(kw, kw),
                          'text': m.group(0), 'numbers': nums, 'range': rng, 'extractor': 'W183h lettered'})
    return atoms


def all_pointers(path):
    base = R.letter_pointers(path)
    for a in base:
        a['extractor'] = 'W177b-3prime letter_pointers'
    return base + lettered_pointers(path)


def expected_of(r9):
    v = r9['verdict']
    if v in ('MATCH', 'SUBMITTED'):
        return '--'.join(r9['numbers'])
    if v == 'MISMATCH' and r9.get('should_be'):
        return r9['should_be']
    return None


def label_number(ctx, label):
    v = ctx['adj_n']['labels'].get(label)
    return v['number'] if v else None


def declared_number(ctx, dec):
    if 'label' in dec:
        return label_number(ctx, dec['label']), dec['label']
    v = ctx['adj_n']['labels'].get(dec['heading_of_label'])
    if not v:
        return None, None
    h = W.heading_at(ctx['adj_n'], v['line'])
    return (h[1] if h else None), f'heading containing {dec["heading_of_label"]}'


def pair_and_rate(atoms10, rows9, ctx, mp10to9, declared):
    by9 = collections.defaultdict(list)
    for r in rows9:
        by9[(r['line'], r['kind'])].append(r)
    by10 = collections.defaultdict(list)
    for a in atoms10:
        by10[(a['line'], a['kind'])].append(a)
    used9, used_decl, out = set(), set(), []
    for (ln, kind), lst in sorted(by10.items()):
        l9 = mp10to9.get(ln)
        cand = [r for r in by9.get((l9, kind), [])]
        pairs = {}
        for i, a in enumerate(lst):                       # step 1: the v9 pointer whose expected HEAD number is printed
            printed = '--'.join(a['numbers'])
            for r in cand:
                if id(r) not in used9 and expected_of(r) == printed:
                    pairs[i] = r
                    used9.add(id(r))
                    break
        for i, a in enumerate(lst):                       # step 2: the rest in order, v9 pointers with a HEAD target first
            if i in pairs:
                continue
            for r in sorted(cand, key=lambda r_: expected_of(r_) is None):
                if id(r) not in used9:
                    pairs[i] = r
                    used9.add(id(r))
                    break
        for i, a in enumerate(lst):
            printed = '--'.join(a['numbers'])
            row = {k: a[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range', 'extractor')}
            row['line_v9'] = l9
            r = pairs.get(i)
            key = (ln, kind, printed)
            if r is not None:
                row['v9'] = {k: r.get(k) for k in ('line', 'text', 'numbers', 'verdict', 'subclass', 'should_be',
                                                   'target_8f051a2', 'numbers_edd7a63')}
                exp = expected_of(r)
                row['expected_at_head'] = exp
                if r['verdict'] == 'SUBMITTED':
                    row['verdict'] = 'SUBMITTED' if printed == exp else 'MISMATCH'
                    row['note'] = r.get('note')
                elif exp is not None:
                    row['verdict'] = 'MATCH' if printed == exp else 'MISMATCH'
                    row['target'] = r.get('target_8f051a2')
                    if r.get('members_removed'):
                        row['note'] = f'range member(s) removed since 8f051a2: {r["members_removed"]}'
                    if r.get('subclass', '') and 'needle' in str(r.get('subclass')):
                        row['note'] = r['subclass']
                    if r.get('via_needle'):
                        row['note'] = f'v9 target heading removed; resolved by its content needle ({r["via_needle"]})'
                elif key in declared:
                    num, what = declared_number(ctx, declared[key])
                    used_decl.add(key)
                    row.update({'verdict': 'MATCH' if num == printed else 'MISMATCH', 'expected_at_head': num,
                                'target': what, 'declared_target': True, 'why': declared[key]['why'],
                                'note': f'v9 target {r.get("subclass") or r["verdict"]}; declared target'})
                else:
                    row['verdict'], row['note'] = 'AMBIGUOUS', f'v9 pointer {r["verdict"]} {r.get("subclass")}: no target'
            elif key in declared:
                num, what = declared_number(ctx, declared[key])
                used_decl.add(key)
                row.update({'verdict': 'MATCH' if num == printed else 'MISMATCH', 'expected_at_head': num, 'target': what,
                            'declared_target': True, 'why': declared[key]['why'], 'note': 'new in v10 (no v9 counterpart)'})
            else:
                row['verdict'], row['note'] = 'AMBIGUOUS', 'new in v10 (no v9 counterpart) and no declared target'
            row['printed_now_denotes'] = (None if a['kind'] == 'former'
                                          else [W.now_denotes(a['kind'], p, ctx) for p in a['numbers']])
            out.append(row)
    dropped = [{k: r.get(k) for k in ('line', 'text', 'numbers', 'verdict', 'subclass', 'should_be')}
               for r in rows9 if id(r) not in used9]
    return out, dropped, used_decl


def task_pointers(texts):
    t_head = texts['main.tex']
    tx = {'main_old': W.show(OLD), 'main_head': t_head, 'main_r7': W.show(W.ROUND7), 'main_w180c': W.show(W.W180C_HEAD),
          'letter_w180c': W.show(W.W180C_HEAD, W.LETTER_NAME)}
    n_fail0 = len(W.CHECK_FAILURES)
    lab, ctx = W.task_label_map(tx)
    _, adj_prev = W._num(show(PREV_FULL, 'main.tex').split('\n'))
    lab_diff = sorted(k for k in set(adj_prev['labels']) | set(ctx['adj_n']['labels'])
                      if (adj_prev['labels'].get(k) or {}).get('number') != (ctx['adj_n']['labels'].get(k) or {}).get('number'))
    w180c = json.load(open(W180C_JSON, encoding='utf-8'))
    v9_text = _rd(LETTER_V9).decode('utf-8')
    rows9, atoms9, _ = W.pointer_rows(LETTER_V9, ctx, w180c['pointers']['pointers'], tx['letter_w180c'], v9_text)
    c9 = Counter(r['verdict'] for r in rows9)
    w182c = json.load(open(W182C_JSON, encoding='utf-8'))['pointers']['pointers']
    agree = [(a['line'], a['text'], a['verdict'], a.get('should_be')) == (b['line'], b['text'], b['verdict'], b.get('should_be'))
             for a, b in zip(rows9, w182c)]
    differs = [{'line': a['line'], 'text': a['text'], 'head': [a['verdict'], a.get('should_be'), a.get('subclass'), a.get('via_needle')],
                'w182c_edd7a63': [b['verdict'], b.get('should_be'), b.get('subclass')]}
               for a, b, ok in zip(rows9, w182c, agree) if not ok]
    let9 = lettered_pointers(LETTER_V9)
    letter_path = os.path.join(CLONE, LETTER_NAME)
    atoms10 = all_pointers(letter_path)
    mp = {v: k for k, v in line_map(v9_text, texts[LETTER_NAME]).items()}        # v10 line -> v9 line
    rows10, dropped, used_decl = pair_and_rate(atoms10, rows9, ctx, mp, DECLARED_TARGETS)
    cnt = Counter(r['verdict'] for r in rows10)
    unused_decl = sorted(set(DECLARED_TARGETS) - used_decl)
    if unused_decl:
        _fail(f'(c) declared targets not used (stale): {unused_decl}')
    if len(W.CHECK_FAILURES) > n_fail0:
        _fail(f'(c) W182c functions reported: {W.CHECK_FAILURES[n_fail0:]}')
    cites = ctx['cites_n']
    cite_no = {k: (cites.index(k) + 1 if k in cites else None) for k in ('naumann_2018', 'eve_mb31_datasheet', 'boyd_admm_2011')}
    letter_ws = ws(texts[LETTER_NAME])
    cite_sent = {'one of them read from a manufacturer datasheet [43]': 'one of them read from a manufacturer datasheet [43]' in letter_ws,
                 'in the range reported for LFP cells [42]': 'in the range reported for LFP cells [42]' in letter_ws}
    ref_rows = [{k: r[k] for k in ('line', 'text', 'verdict', 'expected_at_head')} | {'denotes': r['printed_now_denotes']}
                for r in rows10 if r['kind'] == 'reference']
    out = {'label_map': {'counts': lab['counts'], 'n_labels_8f051a2': lab['n_labels_8f051a2'],
                         'n_labels_head': lab['n_labels_edd7a63'], 'cross_check_8f051a2_equals_cd57efc': lab['cross_check_8f051a2_equals_cd57efc_all_labels'],
                         'labels_numbered_differently_edd7a63_vs_head': lab_diff,
                         'n_labels_edd7a63': len(adj_prev['labels']),
                         'note': 'static numbering (no TeX installation); the key "edd7a63" in W182c\'s output names the '
                                 'revision passed to it, which here is the HEAD a87e195'},
           'citations': {'numbers_at_head': cite_no, 'order_identical_8f051a2': lab['citations']['order_identical'],
                         'letter_sentences_present': cite_sent, 'letter_reference_pointers': ref_rows},
           'v9_at_head': {'counts': dict(c9), 'agree_with_committed_w182c': sum(agree), 'n': len(rows9),
                          'differ_from_committed_w182c': differs, 'lettered_pointers_in_v9': let9},
           'v10': {'n_pointers': len(rows10), 'counts': dict(cnt), 'by_extractor': dict(Counter(r['extractor'] for r in rows10)),
                   'mismatch': [r for r in rows10 if r['verdict'] == 'MISMATCH'],
                   'ambiguous': [r for r in rows10 if r['verdict'] == 'AMBIGUOUS'],
                   'submitted': [r for r in rows10 if r['verdict'] == 'SUBMITTED'],
                   'declared_target_rows': [r for r in rows10 if r.get('declared_target')],
                   'pointers': rows10, 'v9_pointers_dropped_in_v10': dropped},
           'declared_targets': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in DECLARED_TARGETS.items()}}
    out['prediction_held'] = (cnt.get('MISMATCH', 0) == 0 and cnt.get('AMBIGUOUS', 0) == 0 and cnt.get('SUBMITTED', 0) == 5
                              and cite_no['naumann_2018'] == 42 and cite_no['eve_mb31_datasheet'] == 43
                              and all(cite_sent.values()) and all(r['verdict'] == 'MATCH' for r in ref_rows))
    return out, ctx, rows9, mp


def controls_c(ctx, rows9, texts):
    out = []
    v10 = texts[LETTER_NAME]
    tmp = tempfile.mkdtemp(prefix='w183h_ctl_')
    cases = [('c1 l. 45 "Section~3.4" -> "Section~3.5"', 45, 'Section~3.4', 'Section~3.5', ('section', '3.5'), 'MISMATCH'),
             ('c2 l. 256 "[42]" -> "[43]"', 256, '[42]', '[43]', ('reference', '43'), 'MISMATCH'),
             ('c3 l. 288 "Fig.~E.19" -> "Fig.~E.18"', 288, 'Fig.~E.19', 'Fig.~E.18', ('figure', 'E.18'), 'MISMATCH'),
             ('c4 l. 231 "Appendix~A.5" -> "Appendix~A.4" (declared target)', 231, 'Appendix~A.5', 'Appendix~A.4',
              ('appendix', 'A.4'), 'MISMATCH')]
    for cid, ln, a, b, (kind, num), want in cases:
        L = v10.split('\n')
        ok_src = a in L[ln - 1]
        L[ln - 1] = L[ln - 1].replace(a, b, 1)
        p = os.path.join(tmp, f'letter_{len(out)}.tex')
        with open(p, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(L))
        atoms = all_pointers(p)
        mp = {v: k for k, v in line_map(_rd(LETTER_V9).decode('utf-8'), '\n'.join(L)).items()}
        decl = dict(DECLARED_TARGETS)
        if (ln, kind, num) not in decl and kind == 'appendix':
            decl[(ln, kind, num)] = DECLARED_TARGETS[(231, 'appendix', 'A.5')]
        rows, _, _ = pair_and_rate(atoms, rows9, ctx, mp, decl)
        hit = [r for r in rows if r['line'] == ln and r['kind'] == kind and '--'.join(r['numbers']) == num]
        out.append({'id': cid, 'source_text_present': ok_src, 'verdicts': [r['verdict'] for r in hit], 'expected_verdict': want,
                    'refused': ok_src and bool(hit) and all(r['verdict'] != 'MATCH' for r in hit)})
    return out


# ======================================================================================================================
#  (d) A7 re-rate and A8 places
# ======================================================================================================================
# per listed A7 entry: phrases of the defect that must be absent from the HEAD file, phrases of the correction that must
# be present at the mapped place (from STEP6_ROUND8_CORRECTIONS.md B / C / D)
A7_DECL = {
    'M-ABS-1': {'defect': ['two-stage stochastic framework for shared battery ESS siting, sizing and installation timing'],
                'fix': ['formulated as a two-stage stochastic program in which one scenario-independent plan',
                        'probability-weighted cost trajectories']},
    'M-ABS-2': {'defect': ['evaluates their operation under load, renewable-generation and market-price uncertainty'],
                'fix': ['explores discrete investment plans in a single operating scenario',
                        'fixed plans are then re-evaluated under load, renewable-generation and market-price scenarios']},
    'M-HL-1': {'defect': ['A stochastic framework plans shared storage'],
               'fix': ['A two-stage framework plans shared lithium-ion storage at TN--DN interface nodes']},
    'M-CON-1': {'defect': ['determines a multi-year investment plan and evaluates it against scenario-dependent'],
                'fix': ['the planning searches run on a single operating scenario and fixed plans are evaluated under market and operation scenarios']},
    'H-3': {'defect': ['Every recourse evaluation is certified', 'certif'], 'fix': []},
    'H-4': {'defect': ['breaks even'], 'fix': []},
    'L-01': {'defect': ['under an explicit stopping and certification rule'],
             'fix': ['during the search under a residual-based stopping rule', 'under an explicit numerical acceptance rule']},
    'L-04': {'defect': ['loses determinately to the plan without storage', '0.933 predicted'],
             'fix': ['by more than the reported numerical resolution', 'stopped by the production exit and are not certified']},
    'L-05': {'defect': ['driving certified recourse evaluations'],
             'fix': ['driving recourse evaluations under a declared acceptance rule']},
    'L-07': {'defect': ['what it certifies is each evaluation', 'pairwise comparison of certified evaluations'],
             'fix': ['carries a declared acceptance status']},
    'L-08': {'defect': ['the certification bounds the'], 'fix': ['neither bounds the']},
    'L-09': {'defect': ['what is certified is each recourse evaluation'], 'fix': ['what is assessed is each recourse evaluation']},
    'L-10': {'defect': ['so that a ratio of two values is resolved to'], 'fix': ['not a bound on the error of a ratio']},
    'L-12': {'defect': ['within resolution, i.e.\\ break-even'],
             'fix': ['within the numerical resolution, so its sign is not resolved']},
    'L-13': {'defect': ['the unit is at break-even and the plan is indeterminate'],
             'fix': ["margin is within the numerical resolution of zero, so its sign is not resolved"]},
    'C-1': {'defect': ['bi-level stochastic optimization model', 'staged investment decisions'],
            'fix': ['a two-stage stochastic program selecting one scenario-independent investment plan']},
}


def quote_present(text, quote):
    t = ws(text)
    frags = [ws(f) for f in re.split(r'…|\.\.\.', quote) if len(ws(f)) >= 12]
    return bool(frags) and all(f in t for f in frags)


def region(lines, centre, span=2):
    lo, hi = max(1, centre - span), min(len(lines), centre + span)
    return ws('\n'.join(lines[lo - 1:hi]))


def task_a7(texts, W182D):
    prev = {f: show(PREV_FULL, f) for f in FILES}
    maps = {f: line_map(prev[f], texts[f]) for f in FILES}
    listed = [e for e in W182D['A7']['entries'] if e['status'] == 'listed']
    rows = []
    for e in listed:
        f = e['file']
        d = A7_DECL[e['id']]
        l0, l1 = e['lines'][0], e['lines'][-1]
        new_l = [maps[f].get(x) for x in range(l0, l1 + 1)]
        new_lines = [x for x in new_l if x]
        hl = texts[f].split('\n')
        centre = new_lines[0] if new_lines else None
        reg = ws(' '.join(region(hl, x, 1) for x in new_lines)) if new_lines else ws(texts[f])
        full = ws(texts[f])
        r = {'id': e['id'], 'file': f, 'lines_edd7a63': [l0, l1], 'lines_head': new_l, 'cat': e['cat'],
             'quote': e['quote'], 'quote_present_edd7a63': quote_present(prev[f], e['quote']),
             'quote_present_head': quote_present(texts[f], e['quote']),
             'defects_present_head': {p: (p.lower() in full.lower()) for p in d['defect']},
             'defects_present_edd7a63': {p: (p.lower() in ws(prev[f]).lower()) for p in d['defect']},
             'fix_present_at_place': {p: (p in reg) for p in d['fix']},
             'place_mapped': bool(new_lines), 'fix_searched_in': 'mapped lines +-1' if new_lines else 'whole file (place not mapped)',
             'head_text_at_place': (ws(hl[centre - 1])[:600] if centre else None)}
        if r['quote_present_head']:
            r['rating'] = 'REMAINS'
        elif any(r['defects_present_head'].values()):
            r['rating'] = 'CHANGED, DEFECT PHRASE STILL PRESENT'
        elif all(r['fix_present_at_place'].values()):
            r['rating'] = 'RESOLVED'
        else:
            r['rating'] = 'CHANGED, CORRECTION NOT LOCATED'
        rows.append(r)
        if not r['quote_present_edd7a63']:
            _fail(f'(d) A7 {e["id"]}: the W182d quote is not found at edd7a63 (method broken)')
    border = [{'id': e['id'], 'file': e['file'], 'quote': e['quote'][:200],
               'present_head': quote_present(texts[e['file']], e['quote'])}
              for e in W182D['A7']['entries'] if e['status'] == 'borderline']
    cnt = Counter(r['rating'] for r in rows)
    # negative control: the L-05 quote planted back into a copy of the letter
    planted = texts[LETTER_NAME].replace('driving recourse evaluations under a declared acceptance rule, with no cuts.',
                                         'driving certified recourse evaluations, with no cuts.', 1)
    l05 = next(e for e in listed if e['id'] == 'L-05')
    ctl = {'id': 'd1 the L-05 quote planted back into the letter', 'planted': planted != texts[LETTER_NAME],
           'refused': planted != texts[LETTER_NAME] and quote_present(planted, l05['quote'])
           and not quote_present(texts[LETTER_NAME], l05['quote'])}
    return {'n_listed': len(listed), 'ratings': dict(cnt), 'rows': rows, 'borderline_at_head': border,
            'remaining': [r['id'] for r in rows if r['rating'] != 'RESOLVED'],
            'prediction_held': len(listed) == 16 and cnt.get('RESOLVED', 0) == 16,
            'ruling_addendum_84_2': 'M-CON-3, M-45-1/2 and M-5-1 (borderline) stand; listed for information only'}, ctl


PROBE_RX = re.compile(r'valuation probe|exceeds? the (?:investment |1~M\\euro\{\} )?budget|above the 1~M\\euro\{\} budget|'
                      r'over the budget|outside the admissible', re.I)


def paragraph_of(lines, ln):
    i0, i1 = ln - 1, ln - 1
    while i0 > 0 and lines[i0 - 1].strip() and not lines[i0 - 1].strip().startswith(('\\begin{table', '\\begin{figure', '}')):
        i0 -= 1
    while i1 + 1 < len(lines) and lines[i1 + 1].strip() and not lines[i1 + 1].strip().startswith(('\\end{', '}')):
        i1 += 1
    return '\n'.join(lines[i0:i1 + 1])


def section_region(lines, label):
    i = next(k for k, x in enumerate(lines) if f'\\label{{{label}}}' in x)
    j = next((k for k in range(i + 1, len(lines)) if re.match(r'\s*\\(sub)?section\*?\{', lines[k])), len(lines))
    return i + 1, j


def task_a8(texts, W182D):
    prev = {f: show(PREV_FULL, f) for f in FILES}
    maps = {f: line_map(prev[f], texts[f]) for f in FILES}
    ml = texts['main.tex'].split('\n')
    be0, be1 = section_region(ml, 'subsec:res_breakeven')
    s35 = [i + 1 for i, x in enumerate(ml) if x.strip().startswith('Five single-node plans at node~7 evaluated in 2025')]
    s35_text = ws(ml[s35[0] - 1]) if s35 else ''
    s35_names_fit = 'break-even fit of Subsection~\\ref{subsec:res_breakeven}' in s35_text
    s35_says_marked = 'marked $^{b}$ in the tables' in s35_text
    rows = []
    for p in W182D['A8_1_places']['places']:
        f = p['file']
        L = texts[f].split('\n')
        nl = maps[f].get(p['line'])
        txt = L[nl - 1] if nl else ''
        mark = bool(re.search(r'\^\{(?:[a-z],)*b(?:,[a-z])*\}', txt))
        probe = bool(PROBE_RX.search(txt))
        para = bool(PROBE_RX.search(paragraph_of(L, nl))) if nl else False
        in_be = f == 'main.tex' and nl is not None and be0 <= nl <= be1
        is_table_or_fig = bool(re.search(r'Table|Figure|tab:|row|caption|fig', p['element']))
        affine = 'affine' in txt if f == LETTER_NAME else None
        legend_in_float = False
        if f == 'main.tex' and nl and 'caption' in p['element']:
            i0 = max((k for k in range(nl) if re.match(r'\s*\\begin\{(table|figure|longtable)', L[k])), default=None)
            i1 = next((k for k in range(nl - 1, len(L)) if re.match(r'\s*\\end\{(table|figure|longtable)', L[k])), None)
            if i0 is not None and i1 is not None:
                env = '\n'.join(L[i0:i1 + 1])
                legend_in_float = ('$^{b}$:' in env) and bool(re.search(r'\^\{(?:[a-z],)*b(?:,[a-z])*\}\$?\s*&', env))
        if p['id'] == 'P11a':     # the E.21 group-(a) header: its over-budget rows
            hdr_rows = [x for x in W182D['A8_1_places']['places'] if x['id'] in ('P11d', 'P11e', 'P11f', 'P11g', 'P11h')]
            rows_marked = all('$^{b}$' in texts['main.tex'].split('\n')[maps['main.tex'][x['line']] - 1] for x in hdr_rows)
        else:
            rows_marked = None
        if mark:
            rating = 'LABELLED: $^{b}$ at the place'
        elif probe:
            rating = 'LABELLED: probe / budget text at the place'
        elif rows_marked:
            rating = 'LABELLED: every over-budget row under the header carries $^{b}$'
        elif legend_in_float:
            rating = 'LABELLED: the table marks its over-budget rows $^{b}$ and carries the legend'
        elif para:
            rating = 'LABELLED: probe / budget text in the same paragraph'
        elif f == LETTER_NAME and affine is False:
            rating = 'WITHDRAWN: the affine-ladder argument is no longer in the letter line'
        elif in_be and s35_names_fit:
            rating = 'COVERED BY THE SECTION 3.5 SENTENCE (it names the break-even fit of this subsection); no mark at the place'
        elif p['id'] == 'P14':
            rating = 'PARTIAL (unchanged): the lattice figure legend marks over-budget lattice points'
        elif f == 'main.tex' and is_table_or_fig:
            rating = 'NOT MARKED: a table row without $^{b}$; only the Section 3.5 sentence names the plan'
        else:
            rating = 'NOT LABELLED AT THE PLACE (text)'
        rows.append({'id': p['id'], 'file': f, 'line_edd7a63': p['line'], 'line_head': nl, 'element': p['element'],
                     'label_edd7a63': p['label'], 'mark_at_place': mark, 'probe_text_at_place': probe,
                     'probe_text_in_paragraph': para, 'in_subsec_res_breakeven': in_be, 'rows_under_header_marked': rows_marked,
                     'affine_in_letter_line': affine, 'legend_in_float': legend_in_float,
                     'table_or_figure_occurrence': is_table_or_fig, 'rating': rating,
                     'head_text': ws(txt)[:300]})
    # new occurrences at HEAD not among the 34: the five plan tokens of 2025 rows (W182d's token scan) and plan words
    toks = W182D['A8_1_places']['token_scan_main_tex']['tokens']
    mapped = {r['line_head'] for r in rows if r['file'] == 'main.tex'}
    raw = A.raw_tables(ml)
    e21_year = {int(k): r['cells'][2] for k, r in raw['E.21']['rows'].items()}
    row_year = {r['line']: [r['cells'][2]] for r in raw['E.21']['rows'].values()}
    for r in raw['E.20']['rows'].values():
        row_year[r['line']] = sorted({e21_year.get(e) for e in _ints(r['cells'][2]) if e21_year.get(e)})
    new_hits = []
    for i, x in enumerate(ml, 1):
        for t in toks:
            if re.search(re.escape(t) + r'(?![0-9./])', x) and i not in mapped:
                yrs = row_year.get(i)
                new_hits.append({'line': i, 'token': t, 'text': ws(x)[:200], 'plan_years_from_E21': yrs,
                                 'is_a_2025_over_budget_occurrence': (None if yrs is None else '2025' in yrs)})
    phrase_hits = [{'line': i, 'text': ws(x)[:240]} for i, x in enumerate(ml, 1) if PROBE_RX.search(x) and i not in mapped]
    tf = [r for r in rows if r['table_or_figure_occurrence']]
    tf_ok = [r for r in tf if r['rating'].startswith(('LABELLED', 'COVERED', 'PARTIAL'))]
    return {'n_places': len(rows), 'ratings': dict(Counter(r['rating'] for r in rows)), 'rows': rows,
            'section_3_5_sentence': {'lines': s35, 'text': s35_text, 'names_the_break_even_fit': s35_names_fit,
                                     'says_marked_in_the_tables': s35_says_marked},
            'subsec_res_breakeven_lines': [be0, be1],
            'table_or_figure_occurrences': len(tf), 'table_or_figure_labelled_or_covered': len(tf_ok),
            'table_or_figure_not_labelled': [{k: r[k] for k in ('id', 'line_head', 'element', 'rating', 'head_text')}
                                             for r in tf if r not in tf_ok],
            'text_places_not_labelled': [{k: r[k] for k in ('id', 'file', 'line_head', 'element', 'rating', 'head_text')}
                                         for r in rows if not r['table_or_figure_occurrence'] and r['rating'].startswith('NOT')],
            'head_plan_token_lines_outside_the_34': new_hits, 'head_probe_phrase_lines_outside_the_34': phrase_hits,
            'prediction_held': len(tf_ok) == len(tf)}


# ======================================================================================================================
#  (e) numbers in the 21 edits (W182b method, declarations version W183h)
# ======================================================================================================================
EXTRA_INPUTS_E = {'W182D_JSON': W182D_JSON, 'W182G_JSON': W182G_JSON}


def _w182d(X):
    return X.j['W182D_JSON']['A8_1_budget']


def _ob(X, cond):
    out = []
    for e in _w182d(X)['over_budget_evaluations']:
        if e['condition'] != cond:
            continue
        p = A.decode_plan(e['plan_text'])
        (n, (s, en)), = p.items()
        out.append({'eval': e['eval'], 'node': int(n), 'S': s, 'E': en, 'dur': en / s, 'year': e['year'], 'I': e['I_eur']})
    return out


def _all_same(vals):
    vals = list(vals)
    return vals[0] if vals and all(v == vals[0] for v in vals) else None


def ob_count(cond, evals=None):
    def f(X):
        return len([e for e in _ob(X, cond) if evals is None or e['eval'] in evals])
    return f


def ob_field(cond, field):
    return lambda X: _all_same(e[field] for e in _ob(X, cond))


def ob_E(cond, dur, idx):
    def f(X):
        es = sorted(e['E'] for e in _ob(X, cond) if abs(e['dur'] - dur) < 1e-9)
        return es[idx] if -len(es) <= idx < len(es) else None
    return f


def ob_dur(cond, which):
    def f(X):
        ds = sorted({e['dur'] for e in _ob(X, cond)}, reverse=True)
        return ds[which] if which < len(ds) else None
    return f


def ob_I(which):
    def f(X):
        vals = [e['I'] for e in _ob(X, 'base')]
        b = X.j['ESS_PARAMS']['budget']
        if not vals or not all(v > b for v in vals):
            return None
        return (min(vals) if which == 'min' else max(vals)) / 1e6
    return f


def der_dev_premium_share(X):
    r = X.j['W182G_JSON']['C1']['records']['3x3_x0']
    q, dv, pm = r['gross_operational_cost'], r['deviation_total'], r['row18_charge_weighted']
    v = 100.0 * (dv + pm) / q
    return {'value': v, 'arithmetic': f"W182g records 3x3_x0 (eval {r['eval_key'][:8]}, cycle {r['cycles_run']}): (deviation_total "
                                      f"{dv:,.2f} + row18_charge_weighted {pm:,.2f}) / gross_operational_cost {q:,.2f} x 100 = "
                                      f"{v:.4f} % (premium alone {100 * pm / q:.4f} %); 'about 3 %' read at 0 dp",
            'consistent': abs(r['including_settlement'] - r['contracted_total'] - r['voltage_pin_total'] - q) < 1e-6 * q}


def der_floor_2037(X):
    c = X.j['W182G_JSON']['C2']['numeric']['cells']['3x3_n7_4h_e1']
    first = next((b['block'] for b in c['chain'] if b['floor_active']), None)
    before = [b['soh_recorded'] for b in c['chain'] if first is not None and b['block'] < first]
    return {'value': first, 'arithmetic': f"W182g C2 cell 3x3_n7_4h_e1 ({c['plan']}, eval {c['eval_key'][:8]}, cycle {c['cycle']}): "
                                          f"first block with the 0.70 floor active = {first}; SoH before "
                                          f"{[round(x, 4) for x in before]}, at {first} {c['terminal_soh']:.10f}",
            'consistent': all(x > 0.70 for x in before) and abs(c['terminal_soh'] - 0.70) < 1e-6}


def _c2(cfg, path):
    def f(X):
        d = X.j['W182G_JSON']['C2']['horizon'][cfg]
        for k in path.split('.'):
            d = d[k]
        return d
    return f


def tn_growth_pct(which):
    return lambda X: 100.0 * NB.tn_growth(X)[which]


def carried(decls_by_line, line, written):
    d = decls_by_line.get(line)
    if d is None:
        raise RuntimeError(f'no earlier declaration at line {line}')
    ws_ = [s['w'] for s in d['specs']]
    if ws_ != list(written):
        raise RuntimeError(f'earlier declaration at line {line} is {ws_}, expected {list(written)}')
    return [dict(s) for s in d['specs']]


def declarations_w183h():
    S, U, SF, DER, DL, SINGLE, LANG = NB.S, NB.U, NB.SF, NB.DER, NB.DL, NB.SINGLE, NB.LANG
    b182 = {d['line']: d for d in NB.declarations()}
    b177 = {d['line']: d for d in B1.declarations()}
    budget = SF('1', 'raw', lambda X: X.j['ESS_PARAMS']['budget'] / 1e6, 'SRP1_ESS_Params.json budget (EUR) / 1e6',
                conv='investment budget (expected discounted investment cost, EUR)')
    four_h = SF('4', 'raw', ob_dur('base', 0), 'W182d over-budget base evaluations: the longer duration E/S (h)')
    two_h = SF('2', 'raw', ob_dur('base', 1), 'W182d over-budget base evaluations: the shorter duration E/S (h)')
    e4 = [SF(w, 'raw', ob_E('base', 4.0, i), f'W182d over-budget base evaluations at 4 h: E (MWh), {i + 1}. smallest')
          for i, w in enumerate(('4', '5'))]
    e2 = [SF(w, 'raw', ob_E('base', 2.0, i), f'W182d over-budget base evaluations at 2 h: E (MWh), {i + 1}. smallest')
          for i, w in enumerate(('3', '4', '5'))]
    one_scen = SF('single', 'int', lambda X: (1 if X.j[NB.SR]['NumMarketScenarios'] == 1 and B1R._op_scen(X.j[NB.SR]) == 1
                                              else None),
                  'SRP1.json (the instance of the planning searches): 1 market scenario x 1 operation scenario')
    L = []
    add = L.append
    add(DL(145, U('two', 'compound adjective', '"two-stage" (the model class, not a count)'),
           U('one', 'model structure', '"one scenario-independent plan" (the formulation, not a count)'), dict(one_scen)))
    add(DL(163, U('two', 'compound adjective', '"two-stage" (the model class, not a count)')))
    add(DL(387, U('two', 'compound adjective', '"two-stage" (the model class, not a count)'),
           U('one', 'model structure', '"one scenario-independent investment plan" (the formulation, not a count)'),
           dict(one_scen)))
    add(DL(390, DER('108', B1.der_108, 'buses of the integrated system')))
    add(DL(1291, *carried(b182, 1290, ('32', '1.25', '3.00')),
           SF('1.00', 'raw', tn_growth_pct('Load'), 'case9_operational_data.xlsx sheet Growth Factors, Load (x 100)'),
           SF('2.50', 'raw', tn_growth_pct('Flexible Load'), 'case9_operational_data.xlsx, Flexible Load (x 100)')))
    add(DL(1347, SF('Five', 'int', ob_count('base'), 'W182d A8 table: over-budget evaluations at the base price (2025)'),
           U('single', 'compound adjective', '"single-node" (one node per plan; not a count)'),
           SF('7', 'int', ob_field('base', 'node'), 'W182d: the node of every over-budget base evaluation'),
           SF('2025', 'int', ob_field('base', 'year'), 'W182d: the installation year of every over-budget base evaluation'),
           four_h, *e4, two_h, *e2,
           SF('1.15', 'raw', ob_I('min'), 'W182d A8 table: smallest I(x) of the five, MEUR (each above the budget)',
              conv='investment cost I (expected discounted, MEUR)'),
           SF('1.91', 'raw', ob_I('max'), 'W182d A8 table: largest I(x) of the five, MEUR', conv='investment cost I (expected discounted, MEUR)'),
           budget))
    add(DL(1352, *carried(b182, 1350, ('three',))))
    add(DL(1382, *carried(b182, 1380, ('single',))))
    c1425 = carried(b182, 1425, ('ten', '171,310', '192,560', '24', '33', '253,880'))
    add(DL(1427, *c1425, SF('five', 'int', ob_count('base', set(range(2, 12))),
                            'W182d: over-budget evaluations among the fit points (evaluations 2-11)'),
           dict(c1425[0]), *e4, four_h, *e2, two_h))
    c1431 = carried(b182, 1431, ('7', '2025', '4', '2', '2'))
    add(DL(1433, *c1431, budget, *e4, four_h, e2[0], e2[2], two_h))
    c1467 = carried(b182, 1467, ('five', '4', '7', '2025', '1', '41.0', '51.8'))
    add(DL(1469, *c1467[:4], SF('4', 'raw', ob_E('$m=2$', 4.0, 0), 'W182d over-budget evaluations at m = 2 (4 h): E, smallest'),
           SF('5', 'raw', ob_E('$m=2$', 4.0, 1), 'W182d over-budget evaluations at m = 2 (4 h): E, largest'), *c1467[4:]))
    add(DL(1474, *carried(b182, 1472, ('4', '7', '2025'))))
    add(DL(1489, *carried(b182, 1487, ('4', '176.6', '15.9', '42.8', '27.7'))))
    add(DL(1490, *carried(b182, 1488, ('5', '228.4', '26.5', '51.8', '26.5'))))
    add(DL(1502, budget))
    add(DL(1568, *carried(b182, 1565, ('318.0',))))
    c1360 = carried(b177, 1360, ('0.70', '2035', '50', '8,000', '0.70', '0.50', '0.18', '0.20', '0.27', '2025', '2030', '2035',
                                 '4.9', '59.5'))
    add(DL(1576, SINGLE('single'), *c1360[:4],
           SF('three', 'int', lambda X: _all_same(X.j[NB.C3X]['Years'].values()), 'SRP1__s53_3x3.json Years: block length (each)'),
           DER('2037', der_floor_2037, 'the first 3 x 3 block in which the reference unit\'s 0.70 floor binds (W182g)'),
           *c1360[4:]))
    add(DL(1585, *carried(b182, 1582, ('three',))))
    c1636 = carried(b182, 1636, ('nine', 'three', 'three', 'five', 'three', '236.7', '318.0', '81.3', '18.0'))
    add(DL(1639, *c1636[:5], DER('3', der_dev_premium_share, '(deviation settlement + commitment premium) / Q at x = 0, 3 x 3, %'),
           SINGLE('single'), *c1636[5:]))
    add(DL(1835, U('Three', 'model structure', 'three families of consensus variables (the decomposition, not a figure)'),
           U('one', 'model structure', 'one copy per agent (the decomposition)'), U('one', 'model structure', 'one dual per agent'),
           U('three', 'model structure', '"three of each" (three agents on the storage channel)'),
           U('one', 'model structure', '"one per side" (interface channels)')))
    # supplement (Appendix E prose, generator-owned; E.1 salvage paragraph)
    add(DL(2601, SF('2040', 'int', _c2('SRP1', 'terminal_date_code'), 'W182g C2 horizon SRP1: y0 + sum Y = terminal date'),
           U('1', 'date', '"1 January" (calendar day of the terminal date)'),
           SF('2040', 'int', lambda X: _all_same([_c2('SRP1', 'terminal_date_code')(X), _c2('3x3', 'terminal_date_code')(X)]),
              'W182g C2: terminal date of both configurations (equal)'),
           SF('2035', 'int', _c2('SRP1', 'last_block.year'), 'W182g C2: last representative block, single-scenario'),
           SF('2037', 'int', _c2('3x3', 'last_block.year'), 'W182g C2: last representative block, 3 x 3')))
    return L


def build_X(head_lines):
    all_inputs = dict(W177A.INPUTS)
    all_inputs.update(B1.EXTRA_INPUTS)
    all_inputs.update(B1R.EXTRA_INPUTS)
    all_inputs.update(W178B.EXTRA_INPUTS)
    all_inputs.update(NB.EXTRA_INPUTS)
    all_inputs.update(EXTRA_INPUTS_E)
    inputs, problems = {}, []
    for key, rel in all_inputs.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            problems.append(f'capture path missing: {key} {rel}')
            continue
        in_clone = rel.startswith(CLONE + os.sep)
        clean = (_git(['status', '--porcelain', '--', os.path.relpath(rel, CLONE)], cwd=CLONE).strip() == '' if in_clone
                 else NB.W160.W157.L132._committed_clean(rel))
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': clean}
        if not clean:
            problems.append(f'{rel} not committed clean')
    if inputs.get('FROZEN_JSON', {}).get('sha256') != FZ_SHA:
        problems.append('frozen JSON sha256 differs')
    if problems:
        return None, inputs, problems
    X = W177A.Ctx(inputs)
    for k in B1.EXTRA_JSON + B1R.EXTRA_JSON + W178B.EXTRA_JSON + NB.EXTRA_JSON + tuple(EXTRA_INPUTS_E):
        X.j[k] = W177A._json(inputs[k]['path'])
    for k in B1R.EXTRA_TEXT + W178B.EXTRA_TEXT + NB.EXTRA_TEXT:
        X.texts[k] = W177A._text(inputs[k]['path'])
    for k in ('CODE_DEFS', 'CODE_SES', 'CODE_SESD'):
        X.texts[k] = W177A._text(inputs[k]['path'])
    X.texts['BE_CONV'] = X.t['break_even']['objective_convention']
    X.def_src = {'BE_CONV': f'{FZ_REL} tables.break_even.objective_convention'}
    X.main_lines = head_lines
    X.e21_rows_unchanged = None
    X.cells = dict(X.t['cells'])
    X.cells.update(X.t['cells_appended_w160'])
    return X, inputs, []


def task_numbers(texts, appe_block):
    lines = texts['main.tex'].split('\n')
    X, inputs, problems = build_X(lines)
    if problems:
        for p in problems:
            _fail(f'(e) {p}')
        return {'error': problems}, None
    d0 = _git(['diff', '-U0', PREV_FULL, HEAD_FULL, '--', 'main.tex'], cwd=CLONE)
    plus = NB.added_lines(d0)
    b0, b1 = appe_block
    scope = {ln for ln in plus if not (b0 + 1 <= ln <= b1)}
    supplement = {2601}
    doc = W164.Doc('main.tex', texts['main.tex'])
    toks = []
    for t in W174.tokenize_v3(doc):
        if t['line'] not in scope | supplement:
            continue
        t['file'] = 'main.tex'
        t['status'] = None
        toks.append(t)
    n_split = NB.split_coefficients(toks)
    for t in toks:
        if t['excluded']:
            t.update(status='excluded', category=f"structure: {t['excluded']}")
        elif t['type'] in W177A.IDENT_TYPES:
            t.update(status='excluded', category=f"identifier ({t['type']})")
    decls = declarations_w183h()
    stale, claimed_twice, decl_out, dup = NB.apply_declarations(X, toks, decls, lines)
    unassigned = [{k: t[k] for k in ('line', 'col', 'written')} for t in toks if t['status'] is None]
    for t in toks:
        t['rating'] = NB.RATING.get(t['status'], 'UNRATED')
        t['convention'] = NB.convention(t)
        t['scope'] = 'supplement (E.1, generator prose)' if t['line'] in supplement else '21 main-text edits'
    controls = NB.negative_control(X, toks, lines)
    keep = ('line', 'col', 'written', 'type', 'status', 'rating', 'category', 'counterpart', 'counterpart_value',
            'value_at_written_precision', 'rule', 'note', 'convention', 'scope')
    trows = [{k: (t.get(k) if not isinstance(t.get(k), float) or t.get(k) == t.get(k) else None) for k in keep} for t in toks]
    for r, t in zip(trows, toks):
        r['spec_desc'] = (t.get('spec') or {}).get('desc') or (t.get('spec') or {}).get('why') or str((t.get('spec') or {}).get('src'))
        r['line_text'] = ws(lines[t['line'] - 1])[:160]
    main_rows = [r for r in trows if r['scope'] == '21 main-text edits']
    st = {
        'S1 the five plans named = the over-budget base-price set (W182d)': {
            'named': '4 h: 4 and 5 MWh; 2 h: 3, 4 and 5 MWh; node 7; 2025',
            'w182d_set': [(e['node'], e['S'], e['E'], e['dur'], e['year'], round(e['I'], 2)) for e in _ob(X, 'base')],
            'holds': sorted((e['dur'], e['E']) for e in _ob(X, 'base')) == [(2.0, 3.0), (2.0, 4.0), (2.0, 5.0), (4.0, 4.0), (4.0, 5.0)]
                     and all(e['node'] == 7 and e['year'] == 2025 for e in _ob(X, 'base'))},
        'S2 "no returned plan ... is among them" (W182g inventory: returned / prose-anchor plans all admissible)': {
            'inadmissible_plans': [r['plan'] for r in X.j['W182G_JSON']['C3']['inventory'] if r['verdict'] == 'INADMISSIBLE'],
            'prose_anchor_plans_inadmissible': sorted({pl for a in X.j['W182G_JSON']['C3']['prose_anchors'].values()
                                                       for pl in a['plans']}
                                                      & {r['plan'] for r in X.j['W182G_JSON']['C3']['inventory']
                                                         if r['verdict'] == 'INADMISSIBLE'}),
            'returned_plan_m2': X.j['W182G_JSON']['C3']['prose_anchors'].get('returned_plan_m2', {}).get('plans'),
            'scope': 'planning statements read as the prose anchors of W182g (abstract, 4.1, 4.2, 4.3, 4.5, 4.6, 5, E.3); a '
                     'comparison row of Tables E.20 / 6 that values an over-budget plan is the probe itself, not a comparator'},
        'S3 "the contracted settlement is excluded" at 3 x 3 (W182g records: gross = including - contracted - pin)': {
            'records': {k: abs(r['including_settlement'] - r['contracted_total'] - r['voltage_pin_total'] - r['gross_operational_cost'])
                        for k, r in X.j['W182G_JSON']['C1']['records'].items()}},
    }
    st['S2 "no returned plan ... is among them" (W182g inventory: returned / prose-anchor plans all admissible)']['holds'] = \
        not st['S2 "no returned plan ... is among them" (W182g inventory: returned / prose-anchor plans all admissible)']['prose_anchor_plans_inadmissible']
    st['S3 "the contracted settlement is excluded" at 3 x 3 (W182g records: gross = including - contracted - pin)']['holds'] = all(
        v < 1e-3 for v in st['S3 "the contracted settlement is excluded" at 3 x 3 (W182g records: gross = including - contracted - pin)']['records'].values())
    out = {'scope': {'plus_lines_total': sorted(plus), 'appendix_e_block': [b0 + 1, b1], 'lines_in_scope': sorted(scope),
                     'supplement_lines': sorted(supplement)},
           'n_tokens': len(toks), 'n_split_coefficients': n_split,
           'by_rating_main': dict(Counter(r['rating'] for r in main_rows)),
           'by_rating_supplement': dict(Counter(r['rating'] for r in trows if r['scope'] != '21 main-text edits')),
           'not_rated_categories': dict(Counter(r['category'] or 'excluded' for r in trows if r['rating'] == 'NOT RATED')),
           'tokens': trows, 'declarations': decl_out, 'stale_declarations': stale, 'claimed_twice': claimed_twice,
           'duplicate_declaration_lines': dup, 'unassigned': unassigned,
           'negative_control': {'n': len(controls), 'n_refused': sum(1 for c in controls if c['refused']),
                                'not_refused': [c for c in controls if not c['refused']]},
           'statement_checks': st,
           'inputs': {k: {'path': v['path'], 'sha256': v['sha256']} for k, v in inputs.items()}}
    mm = [r for r in trows if r['rating'] in ('MISMATCH', 'UNSOURCED')]
    out['non_match'] = mm
    out['prediction_held'] = not [r for r in mm if r['scope'] == '21 main-text edits']
    if stale or claimed_twice or dup or unassigned:
        _fail(f'(e) declarations: stale {stale} claimed twice {claimed_twice} dup {dup} unassigned {unassigned}')
    if out['negative_control']['n_refused'] != out['negative_control']['n']:
        _fail(f"(e) negative control: {out['negative_control']['not_refused']}")
    return out, X


# ======================================================================================================================
#  (f) CONFIRM, prefixes, blue
# ======================================================================================================================
def alnum_blue_share(clean, mask, s, e):
    """share of the alphanumeric characters of clean[s:e] inside a blue wrapper, LaTeX control-word names excluded."""
    seg = clean[s:e]
    skip = set()
    for m in re.finditer(r'\\[A-Za-z@]+', seg):
        skip.update(range(m.start(), m.end()))
    idx = [k for k, ch in enumerate(seg) if ch.isalnum() and k not in skip]
    if not idx:
        return None
    return sum(1 for k in idx if mask[s + k]) / len(idx)


def blue_lines_share(lines):
    text = '\n'.join(lines)
    clean, mask = W.blue_mask_text(text)
    out, pos = [], 0
    for ln in clean.split('\n'):
        out.append(alnum_blue_share(clean, mask, pos, pos + len(ln)))
        pos += len(ln) + 1
    return out


def task_confirm_blue(texts):
    ml = texts['main.tex'].split('\n')
    cl = W.typeset(ml)
    confirm = {f: [{'line': i, 'text': ws(x)[:160]} for i, x in enumerate(texts[f].split('\n'), 1) if '[CONFIRM' in x]
               for f in FILES}
    gen = [{'line': i, 'word': m.group(1), 'sep': m.group(2), 'label': m.group(3)} for i, x in enumerate(cl, 1)
           for m in W180C.GENERAL_PREFIX.finditer(x)]
    gen_tilde = [g for g in gen if g['sep'] == '~']
    label_word = [g for g in gen if g['word'].lower() in W180C.LABEL_WORDS]
    prev_lines = show(PREV_FULL, 'main.tex').split('\n')
    clp = W.typeset(prev_lines)
    gen_prev = [{'line': i, 'word': m.group(1), 'sep': m.group(2), 'label': m.group(3)} for i, x in enumerate(clp, 1)
                for m in W180C.GENERAL_PREFIX.finditer(x)]
    label_word_prev = [g for g in gen_prev if g['word'].lower() in W180C.LABEL_WORDS]
    plus = NB.added_lines(_git(['diff', '-U0', PREV_FULL, HEAD_FULL, '--', 'main.tex'], cwd=CLONE))
    for g in label_word:
        g['on_round8_plus_line'] = g['line'] in plus
        g['context'] = ws(cl[g['line'] - 1])[max(0, cl[g['line'] - 1].find('ref{' + g['label']) - 90):][:160]
    hist = {}
    for rev in ('e5322f8', 'cd57efc', '8f051a2', PREV_FULL):
        cr = W.typeset(show(rev, 'main.tex').split('\n'))
        hist[rev[:7]] = sum(1 for x in cr for m in W180C.GENERAL_PREFIX.finditer(x) if m.group(1).lower() in W180C.LABEL_WORDS)
    w182c = json.load(open(W182C_JSON, encoding='utf-8'))['blue']
    a5 = w182c['dropped_by_line']
    mp = line_map(show(PREV_FULL, 'main.tex'), texts['main.tex'])
    share = blue_lines_share(cl)
    rows = []
    for r in a5:
        nl = mp.get(r['line'])
        rows.append({'line_edd7a63': r['line'], 'line_head': nl, 'where': r['where'],
                     'blue_share_alnum_head': share[nl - 1] if nl else None,
                     'head_text': ws(ml[nl - 1])[:140] if nl else None})
    caps = []
    for c in w182c['captions_dropped']:
        b, txt = W.caption_blue(cl, c['label'], W._num(ml)[1])
        caps.append({'label': c['label'], 'blue_share_head': b, 'caption': txt})
    a5_ok = len(rows) == 23 and all(r['blue_share_alnum_head'] == 1.0 for r in rows)
    caps_ok = len(caps) == 5 and all(c['blue_share_head'] is not None and c['blue_share_head'] >= 0.99 for c in caps)
    n_conf = sum(len(v) for v in confirm.values())
    # negative controls: a planted [CONFIRM; one A5 caption unwrapped in a copy
    planted = list(ml)
    planted.insert(10, '% [CONFIRM: control]')
    ctl1 = {'id': 'f1 a planted "[CONFIRM" is counted', 'refused': sum('[CONFIRM' in x for x in planted) == 1}
    k = rows[0]['line_head'] - 1
    unwrapped = list(cl)
    unwrapped[k] = unwrapped[k].replace('\\textcolor{blue}{', '', 1).replace('}}', '}', 1)
    ctl2 = {'id': f'f2 the A5 caption at HEAD l. {k + 1} unwrapped in a copy is detected',
            'share_after': blue_lines_share(unwrapped)[k], 'refused': (blue_lines_share(unwrapped)[k] or 0) < 1.0}
    out = {'confirm': confirm, 'n_confirm': n_conf,
           'general_word_tilde_ref_app_typeset': {'n': len(gen_tilde), 'by_word': dict(Counter(g['word'] for g in gen_tilde)),
                                                  'hits': gen_tilde},
           'label_word_prefixes': label_word, 'n_label_word_prefixes': len(label_word),
           'n_label_word_prefixes_edd7a63': len(label_word_prev), 'label_word_prefixes_by_revision': hist,
           'label_word_prefix_note': 'elsarticle prints \\ref{app:...} with its "Appendix" prefix (W179b PREFIX, W180c), so '
                                     '"Appendix~\\ref{app:x}" prints "Appendix Appendix A.4"; prepositions ("in", "of", '
                                     '"and") are the general pattern only',
           'a5_lines': rows, 'a5_lines_all_blue': a5_ok, 'a5_captions': caps, 'a5_captions_all_blue': caps_ok,
           'method_blue': 'comment-stripped HEAD source; W182c blue_mask_text (each \\textcolor{blue}{ opener brace-matched); per '
                          'line, the share of alphanumeric characters (LaTeX control-word names excluded) inside a wrapper; '
                          'the 23 A5 lines are W182c\'s dropped_by_line rows, mapped edd7a63 -> HEAD by the line diff; captions '
                          'by W182c caption_blue',
           'prediction_held': n_conf == 0 and len(label_word) == 0 and a5_ok and caps_ok}
    return out, [ctl1, ctl2]


# ======================================================================================================================
#  checklist, guards, output
# ======================================================================================================================
REQUIRED_CAPTURES = {
    '(a) identity of the four files and the clone diff vs the expert diffs': 'task_identity',
    '(b) Appendix E bitwise, verify_appE, entry check, b-marks': 'task_appendix_e',
    '(c) letter v10 pointers and citation order': 'task_pointers',
    '(d) A7 re-rate (16 listed sentences)': 'task_a7',
    '(d) A8 places at HEAD': 'task_a8',
    '(e) numbers in the 21 edits (W182b method)': 'task_numbers',
    '(f) CONFIRM, prefixes, blue on the A5 lines': 'task_confirm_blue',
    'prediction-record bullet (quoted only)': 'task_prediction_bullet',
    'negative controls': 'controls_b',
    'guards (solve 0, pickle 0)': 'verify_guards',
}
REQUIRED_SYMBOLS = {
    'W182c': (W, ('task_label_map', 'pointer_rows', 'now_denotes', 'heading_at', '_num', 'blue_mask_text', 'caption_blue',
                  'typeset', 'show', 'OLD', 'ROUND7', 'W180C_HEAD', 'LETTER_NAME', 'CHECK_FAILURES')),
    'W181a': (A, ('run_generator', 'run_verifier', 'locate_bytes', 'appendix_block', 'raw_tables', 'diff_tables', 'block_diff',
                  'table_check', 'cell_conditions', 'parse_udiff', 'compare_hunks', 'strip_markers', 'decode_plan',
                  'mutate_cell', 'line_classes', 'classify_hunk', 'INPUTS')),
    'W177b-3prime': (R, ('letter_pointers', '_blank_cmd', 'KW')),
    'W182b': (NB, ('declarations', 'apply_declarations', 'negative_control', 'split_coefficients', 'added_lines', 'convention',
                   'RATING', 'S', 'U', 'SF', 'DER', 'DL', 'SINGLE', 'tn_growth', 'C3X', 'EXTRA_INPUTS', 'EXTRA_JSON', 'EXTRA_TEXT',
                   'GUARDS')),
    'W177b-1': (B1, ('declarations', 'der_108', 'EXTRA_INPUTS', 'EXTRA_JSON')),
    'W180c': (W180C, ('GENERAL_PREFIX', 'LABEL_WORDS', 'GUARD')),
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
    need = [os.path.join(CLONE, f) for f in FILES] + list(REVIEW_COPY.values()) + list(EXPERT_DIFF.values()) + [
        LETTER_V9, SUBMITTED_COVER, GEN_REL, VERIFY_REL, APPX_REL, FZ_REL, W182C_JSON, W182D_JSON, W182G_JSON, W180C_JSON,
        STEP6_R8, BRIEF]
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    gen_out = os.path.realpath(os.path.join(od, OUT_NAMES['gen']))
    if gen_out in (os.path.realpath(os.path.join(REPO, APPX_REL)), os.path.realpath(os.path.join(REPO, MR, 'appE.tex')),
                   os.path.realpath(os.path.join(REPO, 'appE.tex'))):
        missing.append('generator target is a manuscript_review / cwd file')
    if not gen_out.startswith(os.path.realpath(od) + os.sep):
        missing.append('generator target is not inside the output directory')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def task_prediction_bullet(texts):
    L = texts[LETTER_NAME].split('\n')
    hits = [{'line': i, 'text': x.strip()} for i, x in enumerate(L, 1) if 'prediction record' in x.lower()]
    return {'hits': hits, 'note': 'quoted only (Addendum 84 ruling 1: the author\'s call); not rated'}


def all_guards():
    gs = [('w183h', GUARD), ('w182c', W.GUARD), ('w181a', A.GUARD), ('w179b', B.GUARD), ('w178c', C_.GUARD),
          ('w177b3r', R.GUARD), ('w177b3', W.P3.GUARD), ('w167', W.W167.GUARD), ('phase_b_record', W.PB.PARENT_GUARD),
          ('w180c', W180C.GUARD)] + list(NB.GUARDS)
    seen, out = set(), []
    for nm, g in gs:
        if id(g) in seen:
            continue
        seen.add(id(g))
        out.append((nm, g))
    return out


def verify_guards():
    g = {nm: {'counts': dict(gd.counts), 'verify_0_failures': gd.verify(0)} for nm, gd in all_guards()}
    pk = {'w183h': dict(PICKLE_COUNTS)}
    for nm, mod in (('w182c', W), ('w181a', A), ('w179b', B), ('w178c', C_), ('w177b3r', R), ('w180c', W180C), ('w182b', NB),
                    ('w178b', W178B), ('w177b1r', B1R)):
        if hasattr(mod, 'PICKLE_COUNTS'):
            pk[nm] = dict(mod.PICKLE_COUNTS)
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(B.FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    ok = all(not v['verify_0_failures'] for v in g.values()) and blocked and all(v == {'load': 0, 'loads': 0} for v in pk.values())
    if not ok:
        _fail(f'guards: {g} {pk} blocked {blocked}')
    return {'guards': g, 'n_guards': len(g), 'pickle_counts': pk, 'pickle_blocked_at_end': blocked,
            'production_modules_imported': forbidden, 'ok': ok}


def write_md(out, path):
    I, Bx, P, D7, D8, E, F = (out['a_identity'], out['b_appendix_e'], out['c_pointers'], out['d_a7'], out['d_a8'],
                              out['e_numbers'], out['f_confirm_blue'])
    L = [f'# W183h — round 8 at Overleaf {I["clone_head"][:7]} (zero solves)', '',
         f'HEAD `{I["clone_head"]}` (parent `{I["clone_head_parent"]}`), {I["clone_head_date"]}. Objective convention: no '
         'objective is computed; numbers are rated against the frozen JSON and named records (Q = gross_operational_cost, '
         'contracted interface settlement excluded).', '', '## Predictions', '']
    for k, v in out['predictions'].items():
        L.append(f'- {k}: **{"HELD" if v["held"] else "NOT HELD"}** — {json.dumps(v["outcome"], ensure_ascii=False)[:600]}')
    L += ['', '## (a) identity', '', '| file | sha256 | md5 | = review copy | hunks clone/expert | +/- equal | ranges | context |',
          '|---|---|---|---|---|---|---|---|']
    for f, v in I['outcome'].items():
        fi = I['files'][f]
        L.append(f'| {f} | `{fi["sha256"][:12]}` | `{fi["md5"][:8]}` | {v["byte_identical"]} | {v["hunks_clone_expert"]} | '
                 f'{v["every_hunk_pm_equal"]} | {v["ranges_equal"]} | {v["context_equal"]} |')
    L += ['', '## (b) Appendix E', '', f'- outcome: `{json.dumps(Bx["outcome"])}`',
          f'- E.21 marked rows {Bx["marks"]["e21_marked_rows"]}; E.20 marked rows {Bx["marks"]["e20_marked_rows"]}; legends '
          f'{Bx["marks"]["legend_e20_caption"]}/{Bx["marks"]["legend_e21_caption"]}',
          f'- block hunks edd7a63 -> HEAD: {len(Bx["table_diffs"]["block_hunks_edd7a63_to_head"])}']
    for h in Bx['table_diffs']['block_hunks_edd7a63_to_head']:
        L.append(f'  - {h["class"]} [{h["section"][:50]}] {h["lines_old"]} -> {h["lines_new"]}: {h["new_first"][:140]}')
    L += ['', '## (c) letter v10 pointers', '', f'- counts {P["v10"]["counts"]}; by extractor {P["v10"]["by_extractor"]}',
          f'- citations {P["citations"]["numbers_at_head"]}; v9 at HEAD {P["v9_at_head"]["counts"]}',
          f'- v9 pointers dropped in v10: {[(d["line"], d["text"]) for d in P["v10"]["v9_pointers_dropped_in_v10"]]}', '',
          '| line | pointer | verdict | expected at HEAD | note |', '|---|---|---|---|---|']
    for r in P['v10']['pointers']:
        if r['verdict'] != 'MATCH' or r.get('declared_target') or r.get('note') or (r.get('v9') or {}).get('verdict') != 'MATCH':
            L.append(f'| {r["line"]} | {r["text"][:40]} | {r["verdict"]} | {r.get("expected_at_head")} | {str(r.get("note") or "")[:90]} |')
    L += ['', '## (d) A7', '', f'- ratings {D7["ratings"]}; remaining {D7["remaining"]}', '', '| id | file:line HEAD | rating |',
          '|---|---|---|']
    for r in D7['rows']:
        L.append(f'| {r["id"]} | {r["file"]}:{r["lines_head"]} | {r["rating"]} |')
    L += ['', '## (d) A8', '', f'- ratings {D8["ratings"]}', f'- table/figure occurrences {D8["table_or_figure_occurrences"]}, '
          f'labelled or covered {D8["table_or_figure_labelled_or_covered"]}', '', '| id | line HEAD | element | rating |',
          '|---|---|---|---|']
    for r in D8['rows']:
        L.append(f'| {r["id"]} | {r["line_head"]} | {r["element"][:50]} | {r["rating"]} |')
    L += ['', '## (e) numbers', '', f'- main: {E["by_rating_main"]}; supplement {E["by_rating_supplement"]}; negative control '
          f'{E["negative_control"]["n_refused"]}/{E["negative_control"]["n"]} refused', '',
          '| line | written | rating | source |', '|---|---|---|---|']
    for r in E['tokens']:
        L.append(f'| {r["line"]} | {r["written"]} | {r["rating"]} | {str(r.get("counterpart") or r.get("category") or "")[:120].replace("|", "/")} |')
    L += ['', '## (f)', '', f'- CONFIRM {F["n_confirm"]}; general "<word>~\\ref{{app:" {F["general_word_tilde_ref_app_typeset"]["by_word"]}; '
          f'label-word prefixes {F["n_label_word_prefixes"]}; A5 lines all blue {F["a5_lines_all_blue"]}; A5 captions all blue '
          f'{F["a5_captions_all_blue"]}', '', '## Prediction-record bullet (quoted only)', '']
    for h in out['prediction_record_bullet']['hits']:
        L.append(f'- l. {h["line"]}: "{h["text"]}"')
    L += ['', f'Negative controls: {[(c["id"][:50], c["refused"]) for c in out["negative_controls"]]}', '',
          f'Guards ok {out["guards"]["ok"]} ({out["guards"]["n_guards"]} guards); check failures {out["check_failures"]}']
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run(out_dir):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    p = os.path.join(od, OUT_NAMES['post'])
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = [OUT_NAMES[k] for k in ('log', 'man', 'json', 'md', 'gen', 'verify', 'typing_json', 'typing_log')]
    missing = [n for n in names if not os.path.isfile(os.path.join(od, n))]
    if missing:
        print(f'REFUSED: post-run inputs missing: {missing}', flush=True)
        sys.exit(1)
    man = {'files': {n: _sha(os.path.join(od, n)) for n in names}, 'script_sha256': _sha(SCRIPT_NAME),
           'written_utc': datetime.now(timezone.utc).isoformat()}
    with open(p, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    print(f'written {p}', flush=True)
    sys.exit(0)


def main():
    out_dir = OUT_DIR
    if '--trial-out-dir' in sys.argv:
        out_dir = sys.argv[sys.argv.index('--trial-out-dir') + 1]
        if not os.path.isabs(out_dir) or os.path.abspath(out_dir).startswith(REPO + os.sep):
            print('REFUSED: --trial-out-dir is for trial runs, an absolute path outside the repository', flush=True)
            sys.exit(1)
    if '--post-run' in sys.argv:
        post_run(out_dir)
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    if not os.path.isdir(od):
        print(f'REFUSED: {out_dir} does not exist (create it with mkdir; it must hold only launch.log)', flush=True)
        sys.exit(1)
    extra = sorted(set(os.listdir(od)) - {'launch.log'})
    if extra:
        print(f'REFUSED: {out_dir} already holds {extra} (write-once; fresh directory required)', flush=True)
        sys.exit(1)
    others = other_copies()
    if others:
        print(f'REFUSED: another copy of {SCRIPT_NAME} is running: {others}', flush=True)
        sys.exit(1)
    t0 = datetime.now(timezone.utc)
    _log(f'W183h: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W183h (Addendum 84; Addendum 85 W183g (e)): round 8 at the Overleaf HEAD a87e195',
           'objective_convention': 'no objective value is computed; rated numbers carry their own convention '
                                   '(Q = gross_operational_cost, contracted interface settlement excluded)',
           'static_numbering': 'no TeX installation on this machine; numbering is static (W182c method); nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'script_committed_clean': NB.W160.W157.L132._committed_clean(SCRIPT_NAME),
           'imported': {m: _sha(m) for m in IMPORTED_SCRIPTS}}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    W182D = json.load(open(W182D_JSON, encoding='utf-8'))
    out['a_identity'], texts = task_identity()
    _log(f"(a) HEAD {out['a_identity']['clone_head'][:8]} parent {out['a_identity']['clone_head_parent'][:8]}; outcome "
         f"{out['a_identity']['outcome']}")
    out['b_appendix_e'], bctx = task_appendix_e(out_dir, texts, W182D)
    _log(f"(b) {out['b_appendix_e']['outcome']}")
    out['c_pointers'], pctx, rows9, _ = task_pointers(texts)
    _log(f"(c) v10 {out['c_pointers']['v10']['counts']}; citations {out['c_pointers']['citations']['numbers_at_head']}; v9 at HEAD "
         f"{out['c_pointers']['v9_at_head']['counts']}; dropped {len(out['c_pointers']['v10']['v9_pointers_dropped_in_v10'])}")
    out['d_a7'], ctl_d = task_a7(texts, W182D)
    out['d_a8'] = task_a8(texts, W182D)
    _log(f"(d) A7 {out['d_a7']['ratings']}; A8 {out['d_a8']['ratings']}")
    out['e_numbers'], _X = task_numbers(texts, bctx['blk'])
    if 'error' not in out['e_numbers']:
        _log(f"(e) main {out['e_numbers']['by_rating_main']}; supplement {out['e_numbers']['by_rating_supplement']}; "
             f"control {out['e_numbers']['negative_control']['n_refused']}/{out['e_numbers']['negative_control']['n']}")
    out['f_confirm_blue'], ctl_f = task_confirm_blue(texts)
    _log(f"(f) CONFIRM {out['f_confirm_blue']['n_confirm']}; label-word prefixes {out['f_confirm_blue']['n_label_word_prefixes']}; "
         f"A5 lines {out['f_confirm_blue']['a5_lines_all_blue']}; captions {out['f_confirm_blue']['a5_captions_all_blue']}")
    out['prediction_record_bullet'] = task_prediction_bullet(texts)
    ctl = ([{'id': f'a {f}: expert diff + line altered', 'refused': v['refused']}
            for f, v in out['a_identity']['diff_negative_controls'].items()]
           + controls_b(bctx, W182D) + controls_c(pctx, rows9, texts) + [ctl_d] + ctl_f
           + [{'id': f"e every rated number perturbed ({out['e_numbers'].get('negative_control', {}).get('n')})",
               'refused': out['e_numbers'].get('negative_control', {}).get('n_refused') == out['e_numbers'].get('negative_control', {}).get('n')
               and bool(out['e_numbers'].get('negative_control', {}).get('n'))}])
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f'negative control not refused: {c["id"]}')
    _log(f"negative controls: {[(c['id'][:44], c['refused']) for c in ctl]}")
    E = out['e_numbers']
    out['predictions'] = {
        '(a) the four files identical to the round-8 deliverables; clone diff = expert diffs': {
            'outcome': out['a_identity']['outcome'], 'held': out['a_identity']['prediction_held']},
        '(b) Appendix E bitwise; verify 33 PASS; 0 MISMATCH; 0 entries changed; b-marks exactly as listed': {
            'outcome': out['b_appendix_e']['outcome'], 'held': out['b_appendix_e']['prediction_held']},
        '(c) letter v10 pointers: 0 MISMATCH, 0 AMBIGUOUS, 5 SUBMITTED; [42]/[43]': {
            'outcome': {'counts': out['c_pointers']['v10']['counts'], 'citations': out['c_pointers']['citations']['numbers_at_head']},
            'held': out['c_pointers']['prediction_held']},
        '(d) 0 of the 16 A7 sentences remaining': {'outcome': out['d_a7']['ratings'], 'held': out['d_a7']['prediction_held']},
        '(d) every over-budget point labelled at every table / figure occurrence': {
            'outcome': {'table_or_figure': out['d_a8']['table_or_figure_occurrences'],
                        'labelled_or_covered': out['d_a8']['table_or_figure_labelled_or_covered'],
                        'not': [r['id'] for r in out['d_a8']['table_or_figure_not_labelled']]},
            'held': out['d_a8']['prediction_held']},
        '(e) 0 MISMATCH in the numbers of the 21 edits': {
            'outcome': E.get('by_rating_main'), 'held': bool(E.get('prediction_held'))},
        '(f) 0 [CONFIRM, 0 label-word prefixes, blue on the 23 A5 lines': {
            'outcome': {'confirm': out['f_confirm_blue']['n_confirm'], 'label_word_prefixes': out['f_confirm_blue']['n_label_word_prefixes'],
                        'a5_lines_blue': out['f_confirm_blue']['a5_lines_all_blue'],
                        'a5_captions_blue': out['f_confirm_blue']['a5_captions_all_blue']},
            'held': out['f_confirm_blue']['prediction_held']}}
    out['guards'] = verify_guards()
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    pj, pm, pn = (os.path.join(od, OUT_NAMES[k]) for k in ('json', 'md', 'man'))
    with open(pj, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, pm)
    inputs = sorted(set([os.path.join(CLONE, f) for f in FILES] + list(REVIEW_COPY.values()) + list(EXPERT_DIFF.values())
                        + [LETTER_V9, SUBMITTED_COVER, GEN_REL, VERIFY_REL, APPX_REL, FZ_REL, W182C_JSON, W182D_JSON,
                           W182G_JSON, W180C_JSON, STEP6_R8, BRIEF, SCRIPT_NAME] + list(IMPORTED_SCRIPTS)
                        + [os.path.join(MR, 'export_W160', f) for f in os.listdir(os.path.join(REPO, MR, 'export_W160'))]
                        + [v['path'] for v in (E.get('inputs') or {}).values()]))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md', 'gen', 'verify')},
                'out_dir': out_dir, 'clone_head_full': out['a_identity']['clone_head'],
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(show(c, f).encode('utf-8')) for c in (PREV_FULL, HEAD_FULL) for f in FILES},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(CHECK_FAILURES)}; guards ok {out["guards"]["ok"]}; '
         f'pickle {PICKLE_COUNTS}; predictions {[(k[:40], v["held"]) for k, v in out["predictions"].items()]}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
