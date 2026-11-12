"""P5.15 Addendum 91 (Planner order, W190a line; rulings 4 and 5) -- Planner task W190a PART 2: THE ROUND-9c AUDIT AT THE
OVERLEAF HEAD 3c379fa -- IDENTITY, LABELS / NUMBERING, LETTER v13 POINTERS (incl. W189's one MISMATCH), BLUE, STRUCTURE
AND COUNTS. ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT. Read-only on the Overleaf clone (git rev-parse / show /
diff / status / log / ls-tree only; no fetch, pull or checkout), on manuscript_review/, on manuscript_submitted/, on every
committed artifact and on every earlier script. Part 1 (generator gate, verifier, numbers, ruling-4 wording) is
p515_s55_w190a_numbers.py.

NUMBERING IS STATIC (W182c method, as W189 part 2). No TeX installation exists on this machine; nothing is compiled.

REUSE, NOT REIMPLEMENTATION. The W189 part-2 harness p515_s55_w189_p2_structure.py (P2) is IMPORTED, NOT EDITED and NOT
RUN (its main() is never called; nothing is written into its output directory). Its functions are called on two static
contexts built with P2.build_ctx:
  ctx_A = (a87e195 -> 3c379fa): the W189 pointer chain (letter v10 at a87e195, re-rated at HEAD, paired with the current
          letter) is applied unchanged to letter v13, with W189's DECLARED_TARGETS_V12 / DECLARED_SUBMITTED_V12 (the v13 line
          structure equals v12's: one line replaced by one line, twice); dangling references and removed-label uses; the
          blue check of every round-9b AND round-9c change against a87e195 (so W189's five unwrapped lines are re-tested);
  ctx_B = (46c9d77 -> 3c379fa): the label map with the stated expectation "every label identical to 9b" (STATED_NUMBERING
          re-bound to {}; the expert's compile claim "numbering identical to 9b"); the blue check of the lines changed
          9b -> 9c; the counts at 9c against 9b and against the committed W189 counts.
Identity (a) is composed of W181a parse_udiff / compare_hunks and P2's apply_udiff / rotation_equivalent / diff_control,
applied to the expert's 9b -> 9c diffs (main_round9b_round9c.diff, letter_v12_v13.diff, cover_letter_v3_v4.diff).
P2 module constants are re-bound only for the call that needs them (PREV_FULL / HEAD_FULL for the '+' lines of the blue
check; STATED_* for the ctx_B label map) and restored after it.

ITEMS (Addendum 91 W190a line; ruling 5).
  (a) identity: HEAD 3c379fa = origin/main; parent 52a607d whose tree equals 46c9d77's; the four files byte-identical to
      main_46c9d77_round9c.tex / response_to_reviewers_v13.tex / cover_letter_v4.tex / highlights_v3.tex (md5 cd1433d7 /
      74e88c78 / 63b7ddee / 4cedd45c); the 46c9d77 blobs = the round-9b deliverables; the clone diff 46c9d77..3c379fa = the
      expert's diffs hunk for hunk; every other changed file listed.
  (e) letter v13 pointers by the W189 P2 method: expected 0 MISMATCH; W189's one MISMATCH named from the committed W189 P2
      JSON and its v13 line rated.
  (f) 0 [CONFIRM; 0 \\b(Appendix|Section)~\\ref{app: prefixes; 0 undefined refs, 0 duplicate labels (label count stated);
      blue on every line changed 9b -> 9c; floats / equations / words against W189 (17, 41, 6,692); numbering 9b = 9c.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0 together with
P2's and every guard of its chain (P2.verify_guards: pickle.load / loads blocked, counters 0, no production module);
refuses to run concurrently with another copy of itself; refuses unless the output directory exists and holds nothing but
launch.log; capture-path checklist asserted before any task (exit 4); JSON written only through gate_result_io.

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir w190a_audit/structure && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_structure.py \\
      > w190a_audit/structure/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_structure.py --post-run
  (trial runs: --trial-out-dir <absolute path outside the repository>)
Exit: 0 = written, every integrity check holds, guards at 0 (a MISMATCH or a failed expectation is a FINDING and never
changes the exit code); 3 = written, an integrity check failed (listed); 1 / 4 = precondition / capture refusal.
"""
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W190a part 2 structure audit at Overleaf 3c379fa '
                                               '(never solves)').install()

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s55_w189_p2_structure as P2  # noqa: E402  (W189 part 2; imports W183h and its chain; NOT edited, main() not run)
import gate_result_io as GRIO  # noqa: E402

W, A, H = P2.W, P2.A, P2.H
os.chdir(REPO)

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s55_w190a_structure'
OUT_DIR = os.path.join('w190a_audit', 'structure')
OUT_NAMES = {'json': 'w190a_structure.json', 'md': 'w190a_structure.md', 'man': 'manifest_sha256.json', 'log': 'launch.log',
             'post': 'manifest_post_run_sha256.json'}
CLONE = P2.CLONE
HEAD_FULL = '3c379fa44f3755dbdc6fd540df566b55527d921c'
MID_FULL = '52a607d6ea4225154fa78047f5438fada03bb98a'
BASE_FULL = '46c9d779743b40a42da73c0050d8923e6af6fb55'
A87_FULL = P2.PREV_FULL
MR = 'manuscript_review'
LETTER_NAME = P2.LETTER_NAME
FIG6 = P2.FIG6
FILES = P2.FILES
REVIEW_COPY = {'main.tex': os.path.join(MR, 'main_46c9d77_round9c.tex'),
               LETTER_NAME: os.path.join(MR, 'response_to_reviewers_v13.tex'),
               'highlights.tex': os.path.join(MR, 'highlights_v3.tex'),
               'cover_letter.tex': os.path.join(MR, 'cover_letter_v4.tex')}
PREV_COPY = dict(P2.REVIEW_COPY)              # the round-9b deliverables = the 46c9d77 blobs (W189 part 2 (a))
EXPERT_DIFF = {'main.tex': os.path.join(MR, 'main_round9b_round9c.diff'),
               LETTER_NAME: os.path.join(MR, 'letter_v12_v13.diff'),
               'cover_letter.tex': os.path.join(MR, 'cover_letter_v3_v4.diff')}
MD5_DECLARED = {'main.tex': ('cd1433d7', 'Planner (W190a task) / Addendum 91 deliverable main_46c9d77_round9c.tex'),
                LETTER_NAME: ('74e88c78', 'Planner (W190a task) / letter v13'),
                'highlights.tex': ('4cedd45c', 'highlights v3 (unchanged)'),
                'cover_letter.tex': ('63b7ddee', 'Planner (W190a task) / cover letter v4')}
W189P2_JSON = os.path.join(P2.S55, 'w189_p2_structure', 'w189_p2_structure.json')
W189_MISMATCH_EXPECTED = {'line': 285, 'kind': 'figure', 'printed': 'B.8', 'expected': 'B.9', 'label': 'fig:investment_lattice'}
NINE_B_UNBLUE = (1250, 1685, 1695, 1709, 1710)     # W189 part 2 Found 5 (line numbers at 46c9d77 = at 3c379fa; checked)
IMPORTED_SCRIPTS = ('p515_s55_w189_p2_structure.py',) + P2.IMPORTED_SCRIPTS
CHECK_FAILURES = []
EXPECTED_P2_FAILURES = {}      # P2 check failures that are declared not applicable to a context (listed, never silent)


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


_sha, _shab, _rd, _git, ws, committed_clean = P2._sha, P2._shab, P2._rd, P2._git, P2.ws, P2.committed_clean


def show(rev, path):
    return _git(['show', f'{rev}:{path}'], cwd=CLONE)


def showb(rev, path):
    return _git(['show', f'{rev}:{path}'], cwd=CLONE, binary=True)


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


@contextlib.contextmanager
def p2_bound(**kw):
    """Re-bind P2 module constants for one call; restore them after it."""
    old = {k: getattr(P2, k) for k in kw}
    for k, v in kw.items():
        setattr(P2, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(P2, k, v)


def p2_failures_since(n0, context, declared=()):
    """P2 check failures raised in a call: those in `declared` are listed as not applicable to the context, any other is a
    check failure of this stage."""
    new = P2.CHECK_FAILURES[n0:]
    for m in new:
        if m in declared:
            EXPECTED_P2_FAILURES.setdefault(context, []).append(m)
            _log(f'note: P2 reported "{m}" in context {context}: declared not applicable (see the JSON)')
        else:
            _fail(f'P2 ({context}): {m}')


def rename(obj, pairs):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str):
                for a, b in pairs:
                    k = k.replace(a, b)
            out[k] = rename(v, pairs)
        return out
    if isinstance(obj, list):
        return [rename(v, pairs) for v in obj]
    return obj


RENAME_B = (('46c9d77', '@9c_3c379fa'), ('a87e195', '9b_46c9d77'), ('@9c_3c379fa', '9c_3c379fa'))
RENAME_V13 = (('v12', 'v13'),)

# ======================================================================================================================
#  (a) identity
# ======================================================================================================================
def task_identity():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    om = _git(['rev-parse', 'origin/main'], cwd=CLONE).strip()
    parent = _git(['rev-parse', 'HEAD^'], cwd=CLONE).strip()
    mid_parent = _git(['rev-parse', f'{MID_FULL}^'], cwd=CLONE).strip()
    trees = {r: _git(['rev-parse', f'{r}^{{tree}}'], cwd=CLONE).strip() for r in (BASE_FULL, MID_FULL, HEAD_FULL)}
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    allst = _git(['status', '--porcelain'], cwd=CLONE).strip()
    changed = [x for x in _git(['diff', '--name-only', BASE_FULL, HEAD_FULL], cwd=CLONE).split('\n') if x]
    changed_mid = [x for x in _git(['diff', '--name-only', BASE_FULL, MID_FULL], cwd=CLONE).split('\n') if x]
    out = {'clone_head': head, 'origin_main': om, 'clone_head_parent': parent, 'mid_parent': mid_parent, 'trees': trees,
           'clone_log_top4': _git(['log', '--format=%h %cI %s', '-4'], cwd=CLONE).strip().split('\n'),
           'clone_tracked_modified': status, 'clone_porcelain_all': allst, 'files_changed_46c9d77_head': changed,
           'files_changed_46c9d77_52a607d': changed_mid, 'files': {}, 'diffs': {}, 'diff_negative_controls': {}}
    expected_changed = sorted(['main.tex', LETTER_NAME, 'cover_letter.tex'])
    checks = {'head_is_3c379fa': head == HEAD_FULL, 'origin_main_equals_head': om == HEAD_FULL,
              'head_parent_is_52a607d': parent == MID_FULL, '52a607d_parent_is_46c9d77': mid_parent == BASE_FULL,
              '52a607d_tree_equals_46c9d77_tree': trees[BASE_FULL] == trees[MID_FULL],
              'clone_tracked_clean': status == '', 'untracked_only_DS_Store': allst in ('', '?? .DS_Store'),
              'files_changed_are_main_letter_cover': sorted(changed) == expected_changed}
    out['other_changed_files'] = sorted(set(changed) - set(expected_changed))
    texts = {}
    for f in FILES:
        wc = _rd(os.path.join(CLONE, f))
        hb, pb = showb('HEAD', f), showb(BASE_FULL, f)
        rc, pc = _rd(REVIEW_COPY[f]), _rd(PREV_COPY[f])
        md5 = hashlib.md5(wc).hexdigest()
        out['files'][f] = {'sha256': _shab(wc), 'md5': md5, 'lines': wc.count(b'\n'), 'working_copy_equals_HEAD_blob': wc == hb,
                           'review_copy': REVIEW_COPY[f], 'review_copy_sha256': _shab(rc), 'review_copy_md5': hashlib.md5(rc).hexdigest(),
                           'byte_identical_to_review_copy': wc == rc, 'md5_declared': MD5_DECLARED[f][0],
                           'md5_declared_source': MD5_DECLARED[f][1], 'md5_matches_declared': md5.startswith(MD5_DECLARED[f][0]),
                           '46c9d77_blob_sha256': _shab(pb), '46c9d77_blob_equals': PREV_COPY[f],
                           '46c9d77_blob_byte_identical_to_that_copy': pb == pc, 'unchanged_since_46c9d77': pb == hb}
        texts[f] = wc.decode('utf-8')
        clone_diff_text = _git(['diff', BASE_FULL, HEAD_FULL, '--', f], cwd=CLONE)
        got = A.parse_udiff(clone_diff_text).get(f, [])
        if f in EXPERT_DIFF:
            exp_all = A.parse_udiff(_rd(EXPERT_DIFF[f]).decode('utf-8'), default_file=f)
            exp = exp_all.get(f, [])
            cmp_ = A.compare_hunks(got, exp)
            old_t, new_t = pb.decode('utf-8'), hb.decode('utf-8')
            applied, err = P2.apply_udiff(old_t, exp)
            diffpm = [p['hunk'] for p in cmp_['per_hunk'] if not p['pm_lines_equal']]
            cmp_.update({'expert_diff': EXPERT_DIFF[f], 'expert_diff_sha256': _sha(EXPERT_DIFF[f]),
                         'expert_diff_files_parsed': sorted(exp_all),
                         'counts_consistent': all(h['counts_consistent'] for h in got + exp), 'hunks_pm_differ': diffpm,
                         'hunks_pm_differ_rotation_equivalent': {str(k): P2.rotation_equivalent(got[k - 1], exp[k - 1])
                                                                for k in diffpm if k <= min(len(got), len(exp))},
                         'expert_diff_applied_to_46c9d77_equals_head': applied == new_t, 'apply_error': err,
                         'plus_minus_line_counts_expert': {'plus': sum(1 for h in exp for x in h['lines'] if x[0] == '+'),
                                                           'minus': sum(1 for h in exp for x in h['lines'] if x[0] == '-')}})
            cmp_['pm_equal_up_to_equivalent_placement'] = (
                len(got) == len(exp) and applied == new_t
                and all(cmp_['hunks_pm_differ_rotation_equivalent'].get(str(k)) for k in diffpm))
            out['diff_negative_controls'][f] = P2.diff_control(got, exp, old_t, new_t)
        else:
            cmp_ = {'n_hunks_clone': len(got), 'n_hunks_expert': 0, 'every_hunk_pm_equal': len(got) == 0,
                    'every_hunk_range_equal': len(got) == 0, 'every_hunk_context_equal': len(got) == 0,
                    'counts_consistent': all(h['counts_consistent'] for h in got),
                    'note': 'no expert diff this round (highlights v3 unchanged); the clone diff must be empty'}
        cmp_['clone_diff_sha256'] = _shab(clone_diff_text.encode('utf-8'))
        cmp_['plus_minus_line_counts_clone'] = {'plus': sum(1 for h in got for x in h['lines'] if x[0] == '+'),
                                                'minus': sum(1 for h in got for x in h['lines'] if x[0] == '-')}
        out['diffs'][f] = cmp_
    fw, fr = _rd(os.path.join(CLONE, FIG6)), _rd(os.path.join(MR, FIG6))
    out['figure6_pdf'] = {'clone_sha256': _shab(fw), 'review_copy_sha256': _shab(fr), 'byte_identical': fw == fr,
                          'unchanged_since_46c9d77': fw == showb(BASE_FULL, FIG6)}
    out['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'(a) identity check {k} failed')
    for f in FILES:
        if not out['files'][f]['working_copy_equals_HEAD_blob']:
            _fail(f'(a) {f}: working copy differs from the HEAD blob')
        if not out['diffs'][f]['counts_consistent']:
            _fail(f'(a) {f}: a diff hunk is not count-consistent')
    for f, c in out['diff_negative_controls'].items():
        if not c['refused']:
            _fail(f'(a) {f}: the diff negative control was not refused')
    out['outcome'] = {f: {'byte_identical_to_review_copy': out['files'][f]['byte_identical_to_review_copy'],
                          'md5': out['files'][f]['md5'], 'md5_matches_declared': out['files'][f]['md5_matches_declared'],
                          'hunks_clone_expert': [out['diffs'][f]['n_hunks_clone'], out['diffs'][f]['n_hunks_expert']],
                          'every_hunk_pm_equal': out['diffs'][f]['every_hunk_pm_equal'],
                          'pm_equal_up_to_equivalent_placement': out['diffs'][f].get('pm_equal_up_to_equivalent_placement',
                                                                                     out['diffs'][f]['every_hunk_pm_equal']),
                          'ranges_equal': out['diffs'][f]['every_hunk_range_equal'],
                          'context_equal': out['diffs'][f]['every_hunk_context_equal'],
                          'expert_diff_applied_equals_head': out['diffs'][f].get('expert_diff_applied_to_46c9d77_equals_head'),
                          '46c9d77_blob_is_round9b_deliverable': out['files'][f]['46c9d77_blob_byte_identical_to_that_copy']}
                      for f in FILES}
    out['held'] = (all(v['byte_identical_to_review_copy'] and v['md5_matches_declared'] and v['every_hunk_pm_equal']
                       and v['ranges_equal'] and v['46c9d77_blob_is_round9b_deliverable'] for v in out['outcome'].values())
                   and out['figure6_pdf']['byte_identical'] and out['figure6_pdf']['unchanged_since_46c9d77'] and all(checks.values()))
    return out, texts


# ======================================================================================================================
#  numbering 9b vs 9c (ctx_B) and against the committed W189 label map
# ======================================================================================================================
def numbering_compare(ctxB, labB, w189):
    ao, an = ctxB['adj_o'], ctxB['adj_n']
    obj = lambda adj: [(o['kind'], o['number']) for o in adj['objects']]  # noqa: E731
    w189_rows = {r['label']: r['number_46c9d77'] for r in w189['2_label_map']['rows'] if r['number_46c9d77'] is not None}
    now = {lab: v['number'] for lab, v in an['labels'].items()}
    by_kind = {}
    for adj, rev in ((ao, '9b_46c9d77'), (an, '9c_3c379fa')):
        c = Counter(o['kind'] for o in adj['objects'] if o['number'])
        by_kind[rev] = dict(c)
    out = {'objects_sequence_equal': obj(ao) == obj(an), 'n_objects_9b': len(ao['objects']), 'n_objects_9c': len(an['objects']),
           'numbered_objects_by_kind': by_kind,
           'labels_9b': len(ao['labels']), 'labels_9c': len(an['labels']),
           'label_map_counts': labB['counts'], 'n_disagree_with_identical_numbering': labB['n_disagree_with_stated'],
           'equal_to_committed_w189_label_numbers_at_46c9d77': now == w189_rows,
           'differences_vs_committed_w189': sorted(k for k in set(now) | set(w189_rows) if now.get(k) != w189_rows.get(k)),
           'citations_order_identical': labB['citations']['order_identical'],
           'unlabelled_equations_identical': all(e.get('in_a87e195') for e in labB['equations_unlabelled_46c9d77']),
           'heading_map_counts': labB['heading_counts'],
           'counter_adjustments_equal': ao['counter_adjustments'] == an['counter_adjustments']}
    out['identical'] = (out['objects_sequence_equal'] and labB['counts'] == {'unchanged': len(an['labels'])}
                        and out['n_disagree_with_identical_numbering'] == 0 and out['equal_to_committed_w189_label_numbers_at_46c9d77']
                        and out['citations_order_identical'] and out['unlabelled_equations_identical'])
    return out


def heading_title(adj, number):
    hit = [o for o in adj['objects'] if o['kind'] in W.HEADING_KINDS and o['number'] == number]
    return [{'kind': o['kind'], 'line': o['line'], 'title': o.get('title')} for o in hit]


# ======================================================================================================================
#  (e) the W189 MISMATCH named, and the v13 line rated
# ======================================================================================================================
def task_w189_mismatch(P, w189):
    mm = w189['3_pointers']['v12_mismatch']
    exp = W189_MISMATCH_EXPECTED
    named = [{'line': r['line'], 'kind': r['kind'], 'text': r['text'], 'printed': r['numbers'], 'expected_at_head': r.get('expected_at_head'),
              'target': r.get('target'), 'v10_line': r.get('line_v10'), 'v10_text': (r.get('v10_at_head') or {}).get('text')} for r in mm]
    is_b8 = (len(mm) == 1 and mm[0]['line'] == exp['line'] and mm[0]['kind'] == exp['kind'] and mm[0]['numbers'] == [exp['printed']]
             and mm[0].get('expected_at_head') == exp['expected'] and mm[0].get('target') == exp['label'])
    v13 = [r for r in P['v12'] if r['line'] == exp['line'] and r['kind'] == exp['kind']]
    v13_line = _rd(os.path.join(CLONE, LETTER_NAME)).decode('utf-8').split('\n')[exp['line'] - 1]
    out = {'w189_json': W189P2_JSON, 'w189_json_sha256': _sha(W189P2_JSON), 'w189_mismatch': named,
           'w189_mismatch_is_the_l285_Fig_B8_pointer': is_b8,
           'v13_l285_pointers': [{'text': r['text'], 'numbers': r['numbers'], 'verdict': r['verdict'],
                                  'expected_at_head': r.get('expected_at_head'), 'target': r.get('target')} for r in v13],
           'v13_l285_prints_Fig_B9': 'Fig.~B.9' in v13_line and 'Fig.~B.8' not in v13_line,
           'v13_l285_text': ws(v13_line)[:400]}
    out['closed'] = (is_b8 and out['v13_l285_prints_Fig_B9'] and len(v13) == 1 and v13[0]['verdict'] == 'MATCH'
                     and v13[0].get('expected_at_head') == 'B.9')
    return out


def control_b8(ctxA, texts, sub_ctx):
    """v13 l. 285 with Fig.~B.9 put back to Fig.~B.8 in a copy (tempfile.mkdtemp, outside the repository) is MISMATCH."""
    tmp = tempfile.mkdtemp(prefix='w190a_ctl_')
    L = texts[LETTER_NAME].split('\n')
    ok = 'Fig.~B.9' in L[284]
    L[284] = L[284].replace('Fig.~B.9', 'Fig.~B.8', 1)
    p = os.path.join(tmp, 'letter_b8.tex')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    res = P2.task_pointers(ctxA, texts, sub_ctx, letter_path=p)
    hit = [r for r in res['v12'] if r['line'] == 285 and r['kind'] == 'figure']
    return [{'id': 'e-1 v13 l. 285 "Fig.~B.9" -> "Fig.~B.8" in a copy is rated MISMATCH', 'tempdir': tmp,
             'verdicts': [r['verdict'] for r in hit], 'refused': ok and bool(hit) and all(r['verdict'] == 'MISMATCH' for r in hit)}]


# ======================================================================================================================
#  (f) blue on W189's five lines and on every 9b -> 9c changed line
# ======================================================================================================================
def strip_wrappers(s):
    return re.sub(r'\\textcolor\{blue\}\{', '', s).replace('}', '').strip()


def line_blue(ctx):
    """Per line of ctx's new revision: share of content tokens (P2.content_token) at least half blue (P2.token_content_blue),
    the W189 P2 token rule."""
    cmp_ = W.blue_compare(ctx['cl_o'], ctx['cl_n'])
    fn = cmp_['fn']
    share = P2.token_content_blue(ctx['cl_n'])
    by = {}
    for j, (t, _b, ln) in enumerate(fn):
        if P2.content_token(t):
            by.setdefault(ln, []).append((t, share[j] if j < len(share) else None, cmp_['status'][j]))
    return by, len(share) == len(fn)


def multicolumn_wrapped(raw):
    """A \\multicolumn{n}{spec}{text} line: is the typeset third argument exactly one \\textcolor{blue}{...} wrapper? (P2's
    brace matcher.) The W189 token rule reads the whitespace token "\\multicolumn{4}{l}{(a)" as content "4", "l", "a" and so
    counts the two non-typeset arguments as unwrapped text; this check reads the arguments instead."""
    m = re.search(r'\\multicolumn\{', raw)
    if not m:
        return None
    i = m.end() - 1
    e1 = P2.brace_arg(raw, i)
    e2 = P2.brace_arg(raw, e1) if raw[e1:e1 + 1] == '{' else None
    e3 = P2.brace_arg(raw, e2) if e2 is not None and raw[e2:e2 + 1] == '{' else None
    if e3 is None:
        return {'parsed': False}
    a1, a2, a3 = raw[i + 1:e1 - 1], raw[e1 + 1:e2 - 1], raw[e2 + 1:e3 - 1]
    head = '\\textcolor{blue}{'
    wrapped = a3.startswith(head) and P2.brace_arg(a3, len(head) - 1) == len(a3)
    return {'parsed': True, 'arg1_columns': a1, 'arg2_spec': a2, 'arg3_text': a3, 'arg3_entirely_one_blue_wrapper': wrapped}


def controls_multicolumn(ctxB):
    raw9b = ctxB['lo'][1684]
    part = ctxB['ln'][1684].replace('\\textcolor{blue}{(a) Installation year, 2\\,\\% discount rate}}',
                                    '\\textcolor{blue}{(a) Installation year}, 2\\,\\% discount rate}', 1)
    a, b = multicolumn_wrapped(raw9b), multicolumn_wrapped(part)
    return [{'id': 'f-1 the 9b panel heading l. 1685 (no wrapper) fails the multicolumn check', 'refused': bool(a and a['parsed'])
             and a['arg3_entirely_one_blue_wrapper'] is False},
            {'id': 'f-2 a partly wrapped panel heading fails the multicolumn check', 'changed': part != ctxB['ln'][1684],
             'refused': part != ctxB['ln'][1684] and bool(b and b['parsed']) and b['arg3_entirely_one_blue_wrapper'] is False}]


def task_blue(ctxA, ctxB, hyA, hyB):
    byB, alignedB = line_blue(ctxB)
    # the 9b shares of the same lines: blue_compare(9c, 9b), i.e. 9b as the "new" revision
    cl9b, cl9c = ctxB['cl_o'], ctxB['cl_n']
    rev = {'cl_o': cl9c, 'cl_n': cl9b}
    by9b, aligned9b = line_blue(rev)
    rows = []
    for ln in NINE_B_UNBLUE:
        same = strip_wrappers(ctxB['lo'][ln - 1]) == strip_wrappers(ctxB['ln'][ln - 1])
        t9c, t9b = byB.get(ln, []), by9b.get(ln, [])
        rows.append({'line': ln, 'text_unchanged_apart_from_wrappers': same,
                     'content_tokens': len(t9c),
                     'blue_share_9b': round(sum(1 for _t, s, _st in t9b if (s or 0) >= 0.5) / len(t9b), 3) if t9b else None,
                     'blue_share_9c': round(sum(1 for _t, s, _st in t9c if (s or 0) >= 0.5) / len(t9c), 3) if t9c else None,
                     'non_blue_tokens_9c': ' '.join(t for t, s, _st in t9c if (s or 0) < 0.5)[:200],
                     'listed_by_W189_method_vs_a87e195_at_9c': ln in {r['line'] for r in hyA['blue_changed_lines_lacking_blue']},
                     'multicolumn_check_9c': multicolumn_wrapped(ctxB['ln'][ln - 1]),
                     'text_9c': ws(ctxB['ln'][ln - 1])[:200]})
    for r in rows:
        mc = r['multicolumn_check_9c']
        r['typeset_text_blue_at_9c'] = (bool(mc and mc.get('arg3_entirely_one_blue_wrapper')) if mc
                                        else (r['blue_share_9c'] == 1.0 or not r['listed_by_W189_method_vs_a87e195_at_9c']))
    plusB = sorted(P2.plus_lines(_git(['diff', '-U0', BASE_FULL, HEAD_FULL, '--', 'main.tex'], cwd=CLONE)))
    changed = []
    for ln in plusB:
        t9c = byB.get(ln, [])
        changed_tokens = [(t, s) for t, s, st in t9c if st in P2.NONBLUE_CHANGED + ('new_blue', 'newly_blue')]
        changed.append({'line': ln, 'n_content_tokens': len(t9c),
                        'n_changed_tokens': len(changed_tokens),
                        'n_changed_tokens_not_blue': sum(1 for _t, s in changed_tokens if (s or 0) < 0.5),
                        'inside_or_carrying_a_wrapper': all((s or 0) >= 0.5 for _t, s in changed_tokens),
                        'text': ws(ctxB['ln'][ln - 1])[:160]})
    out = {'w189_five_lines': rows,
           'all_five_now_blue': all(r['typeset_text_blue_at_9c'] for r in rows),
           'all_five_pass_the_w189_token_rule_vs_a87e195': all(not r['listed_by_W189_method_vs_a87e195_at_9c'] for r in rows),
           'w189_method_vs_a87e195_at_9c_lines_lacking_blue': hyA['blue_changed_lines_lacking_blue'],
           'w189_method_vs_9b_at_9c_lines_lacking_blue': hyB['blue_changed_lines_lacking_blue'],
           'changed_lines_9b_9c': changed,
           'changed_lines_without_blue': [r['line'] for r in changed if not r['inside_or_carrying_a_wrapper']],
           'token_alignment_ok': alignedB and aligned9b,
           'note_1250': 'l. 1250 is an original (submitted) paragraph; only its changed sentence ("Seasonal profiles are provided '
                        'in the supplementary material (Section~S1).") is new text, and that sentence is now wrapped; the '
                        'rest of the line is not blue at 9b or 9c (unchanged since the submission)'}
    listedA = [r['line'] for r in hyA['blue_changed_lines_lacking_blue']]
    artefact = [r['line'] for r in rows if r['line'] in listedA and r['multicolumn_check_9c']
                and r['multicolumn_check_9c'].get('arg3_entirely_one_blue_wrapper')]
    out['w189_rule_listings_vs_a87e195_explained_as_multicolumn_argument_artefact'] = artefact
    out['w189_rule_listings_vs_a87e195_unexplained'] = sorted(set(listedA) - set(artefact))
    out['held'] = (not out['changed_lines_without_blue'] and not hyB['blue_changed_lines_lacking_blue']
                   and out['all_five_now_blue'] and not out['w189_rule_listings_vs_a87e195_unexplained'])
    return out


# ======================================================================================================================
#  counts against W189
# ======================================================================================================================
def counts_compare(cn, w189):
    h, p = cn['head_46c9d77'], cn['a87e195']           # here: head = 9c (3c379fa), prev = 9b (46c9d77)
    w = w189['6_counts']['head_46c9d77']
    keys = ('prose_words', 'heading_words', 'caption_words', 'n_float_blocks', 'n_display_math_blocks', 'source_lines')
    regions = [r for r in ('abstract', 'main', 'appendix A', 'appendix B') if r in h]
    cmp_ = {r: {k: {'w189_46c9d77': w.get(r, {}).get(k), '9b_46c9d77_here': p.get(r, {}).get(k), '9c_3c379fa': h[r].get(k),
                    'change_9b_9c': (h[r].get(k) - p.get(r, {}).get(k)) if isinstance(h[r].get(k), int) and
                    isinstance(p.get(r, {}).get(k), int) else None} for k in keys} for r in regions}
    for r in regions:
        cmp_[r]['floats'] = {'w189': w.get(r, {}).get('floats'), '9c': h[r].get('floats')}
        cmp_[r]['equations'] = {'w189': w.get(r, {}).get('equations'), '9c': h[r].get('equations')}
    rep = all(p.get(r, {}).get(k) == w.get(r, {}).get(k) for r in regions for k in keys)
    out = {'by_region': cmp_, 'method_reproduces_w189_at_46c9d77': rep,
           'main_text_floats_9c': h['main_text_floats_tables_plus_figures'], 'main_text_floats_w189': w['main_text_floats_tables_plus_figures'],
           'main_text_equations_9c': h['main']['equations'], 'main_text_equations_w189': w['main']['equations'],
           'main_text_prose_words_9c': h['main']['prose_words'], 'main_text_prose_words_w189': w['main']['prose_words'],
           'appendix_B_prose_words_9c': h.get('appendix B', {}).get('prose_words'),
           'appendix_B_prose_words_w189': w.get('appendix B', {}).get('prose_words'),
           'n_appendices_9c': h['n_appendices']}
    if not rep:
        _fail('(f) the W189 counting method on 46c9d77 does not reproduce the committed W189 counts')
    return out


# ======================================================================================================================
#  checklist, guards, md, post-run, main
# ======================================================================================================================
REQUIRED_CAPTURES = {'(a) identity': 'task_identity', '(e) W189 mismatch named': 'task_w189_mismatch',
                     '(f) numbering 9b = 9c': 'numbering_compare', '(f) blue': 'task_blue', '(f) counts': 'counts_compare',
                     'guards': 'verify_guards'}
REQUIRED_SYMBOLS = {'W189p2': (P2, ('build_ctx', 'task_label_map', 'task_pointers', 'table7_check', 'controls_pointers',
                                    'task_dangling', 'task_hygiene', 'task_counts', 'controls_label', 'apply_udiff',
                                    'rotation_equivalent', 'diff_control', 'plus_lines', 'token_content_blue', 'content_token',
                                    'NONBLUE_CHANGED', 'verify_guards', 'DECLARED_TARGETS_V12', 'DECLARED_SUBMITTED_V12',
                                    'STATED_NUMBERING', 'STATED_REMOVED_APPENDICES', 'CHECK_FAILURES', '_num')),
                    'W181a': (A, ('parse_udiff', 'compare_hunks')), 'W182c': (W, ('blue_compare', 'objects_index', 'HEADING_KINDS'))}


def capture_checklist(out_dir):
    missing = []
    g = globals()
    for q, fn in REQUIRED_CAPTURES.items():
        if not callable(g.get(fn)):
            missing.append(f'capture function {fn} for "{q}"')
    for name, (mod, syms) in REQUIRED_SYMBOLS.items():
        for s_ in syms:
            if not hasattr(mod, s_):
                missing.append(f'symbol {name}.{s_}')
    need = ([os.path.join(CLONE, f) for f in FILES] + [os.path.join(CLONE, FIG6), os.path.join(MR, FIG6)]
            + list(REVIEW_COPY.values()) + list(PREV_COPY.values()) + list(EXPERT_DIFF.values())
            + [W189P2_JSON, P2.LETTER_V10, P2.LETTER_V9, P2.SUBMITTED_MAIN, P2.W183H_JSON, P2.W182C_JSON, P2.W180C_JSON])
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    od = os.path.realpath(out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir))
    if od.startswith(os.path.realpath(os.path.join(REPO, 'data')) + os.sep) or \
            od.startswith(os.path.realpath(os.path.join(REPO, MR)) + os.sep):
        missing.append('output directory under data/ or manuscript_review/ (not permitted for W190a)')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def verify_guards():
    n0 = len(P2.CHECK_FAILURES)
    g = P2.verify_guards()
    p2_failures_since(n0, 'guards')
    mine = GUARD.verify(0)
    ok = g['ok'] and not mine
    if not ok:
        _fail(f'guards: W189p2 chain ok {g["ok"]}; this script {mine}')
    return {'w189p2_chain': g, 'w190a': {'counts': dict(GUARD.counts), 'verify_0_failures': mine}, 'ok': ok}


def _md_row(*cells):
    return '| ' + ' | '.join(str(c).replace('|', '/').replace('\n', ' ') for c in cells) + ' |'


def write_md(out, path):
    I, N, P, M, D, Hy, B, C = (out['a_identity'], out['f_numbering_9b_9c'], out['e_pointers_v13'], out['e_w189_mismatch'],
                               out['f_dangling'], out['f_hygiene_9b_9c'], out['f_blue'], out['f_counts'])
    L = [f'# W190a part 2 -- round 9c structure audit at Overleaf {I["clone_head"][:7]} (generated by {SCRIPT_NAME}; do not edit)',
         '', 'Static numbering (no TeX). No objective value is computed in this stage.', '', '## (a) identity', '',
         _md_row('file', 'md5', '= review copy', 'hunks clone/expert', '+/- equal', 'ranges', 'context', '46c9d77 = 9b copy'),
         '|---|---|---|---|---|---|---|---|']
    for f, v in I['outcome'].items():
        L.append(_md_row(f, v['md5'], v['byte_identical_to_review_copy'], v['hunks_clone_expert'], v['every_hunk_pm_equal'],
                         v['ranges_equal'], v['context_equal'], v['46c9d77_blob_is_round9b_deliverable']))
    L += ['', f"Checks: {I['checks']}; other changed files: {I['other_changed_files']}", '',
          '## (e) letter v13 pointers', '', f"Counts: {P['v13_counts']}", '',
          _md_row('line', 'printed', 'verdict', 'expected', 'target / note'), '|---:|---|---|---|---|']
    L += [_md_row(r['line'], r['text'], r['verdict'], r.get('expected_at_head'), r.get('target') or r.get('note'))
          for r in P['v13'] if r['verdict'] != 'MATCH']
    L += ['', f"W189 MISMATCH: {M['w189_mismatch']}; is the l. 285 Fig. B.8 pointer: {M['w189_mismatch_is_the_l285_Fig_B8_pointer']}; "
              f"v13 l. 285: {M['v13_l285_pointers']}; closed {M['closed']}", '',
          '## (f) numbering 9b vs 9c', '', f"{ {k: v for k, v in N.items() if k != 'label_map'} }", '',
          '## (f) dangling', '', f"undefined {D['n_undefined']}, duplicate labels {D['n_duplicate_labels']}, labels "
                                 f"{out['f_label_count_9c']}, removed-label uses {len(D['uses_of_removed_labels_typeset'])}", '',
          '## (f) hygiene', '', f"[CONFIRM {Hy['n_confirm']}; tightened prefixes {Hy['n_tightened_raw']} (typeset "
                                f"{Hy['n_tightened_typeset']}); other files {Hy['tightened_in_other_files']}", '',
          '## (f) blue', '', _md_row('line', '9b share', '9c share', 'listed (W189 method vs a87e195)', 'non-blue tokens 9c'),
          '|---:|---|---|---|---|']
    L += [_md_row(r['line'], r['blue_share_9b'], r['blue_share_9c'], r['listed_by_W189_method_vs_a87e195_at_9c'],
                  r['non_blue_tokens_9c'][:100]) for r in B['w189_five_lines']]
    L += ['', _md_row('9b->9c changed line', 'changed tokens', 'not blue', 'text'), '|---:|---|---|---|']
    L += [_md_row(r['line'], r['n_changed_tokens'], r['n_changed_tokens_not_blue'], r['text'][:100]) for r in B['changed_lines_9b_9c']]
    L += ['', '## (f) counts', '', f"main floats 9c {C['main_text_floats_9c']} (W189 {C['main_text_floats_w189']}); equations 9c "
                                   f"{C['main_text_equations_9c']} (W189 {C['main_text_equations_w189']}); main prose words 9c "
                                   f"{C['main_text_prose_words_9c']} (W189 {C['main_text_prose_words_w189']}); Appendix B prose "
                                   f"words 9c {C['appendix_B_prose_words_9c']} (W189 {C['appendix_B_prose_words_w189']})", '']
    for r, v in C['by_region'].items():
        L.append(f"- {r}: " + '; '.join(f"{k} {x['9b_46c9d77_here']} -> {x['9c_3c379fa']}" for k, x in v.items()
                                         if isinstance(x, dict) and '9c_3c379fa' in x))
    L += ['', '## Negative controls', ''] + [f"- {c['id']}: refused {c['refused']}" for c in out['negative_controls']]
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run(out_dir):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    p = os.path.join(od, OUT_NAMES['post'])
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = [OUT_NAMES[k] for k in ('log', 'man', 'json', 'md')]
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
    _log(f'W190a part 2: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W190a part 2 (Addendum 91 W190a line, rulings 4-5): round-9c audit of identity, pointers, '
                                    'numbering, blue, structure and counts at the Overleaf HEAD 3c379fa',
           'objective_convention': 'no objective value is computed in this stage',
           'static_numbering': 'no TeX installation on this machine; numbering is static (W182c method); nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'script_committed_clean': committed_clean(SCRIPT_NAME), 'imported': {m: _sha(m) for m in IMPORTED_SCRIPTS},
           'revisions': {'head_3c379fa': HEAD_FULL, 'mid_52a607d': MID_FULL, 'base_9b_46c9d77': BASE_FULL,
                         'round8_a87e195': A87_FULL},
           'key_names': 'outputs of W189 P2 functions on ctx_B are renamed a87e195 -> 9b_46c9d77 and 46c9d77 -> 9c_3c379fa (keys '
                        'only; value strings such as stated_source keep W189 wording, the expectation on ctx_B being "every '
                        'label identical to 9b"); the pointer outputs are renamed v12 -> v13 (they rate the letter at the '
                        'clone HEAD, v13)'}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    w189 = json.load(open(os.path.join(REPO, W189P2_JSON), encoding='utf-8'))
    out['a_identity'], texts = task_identity()
    _log(f"(a) identity held {out['a_identity']['held']}; other changed files {out['a_identity']['other_changed_files']}")
    n0 = len(P2.CHECK_FAILURES)
    ctxA = P2.build_ctx(show(A87_FULL, 'main.tex').split('\n'), texts['main.tex'].split('\n'))
    ctxB = P2.build_ctx(show(BASE_FULL, 'main.tex').split('\n'), texts['main.tex'].split('\n'))
    p2_failures_since(n0, 'build_ctx')
    # label maps: ctx_A with W189's stated numbering (for the removed-label set of the dangling check); ctx_B "identical"
    n0 = len(P2.CHECK_FAILURES)
    with p2_bound(HEAD_FULL=HEAD_FULL):
        labA = P2.task_label_map(ctxA)
    with p2_bound(STATED_NUMBERING={}, STATED_REMOVED_APPENDICES=()):
        labB = P2.task_label_map(ctxB)
    p2_failures_since(n0, 'label maps')
    out['f_label_map_vs_a87e195_w189_stated'] = {'counts': labA['counts'], 'n_labels_a87e195': labA['n_labels_a87e195'],
                                                 'n_labels_3c379fa': labA['n_labels_46c9d77'],
                                                 'disagree_with_stated': [(r['label'], r['number_46c9d77'], r['stated_46c9d77'])
                                                                          for r in labA['disagree_with_stated']]}
    out['f_label_map_9b_9c'] = rename(labB, RENAME_B)
    out['f_numbering_9b_9c'] = numbering_compare(ctxB, labB, w189)
    out['f_label_count_9c'] = len(ctxB['adj_n']['labels'])
    _log(f"(f) numbering 9b = 9c: {out['f_numbering_9b_9c']['identical']}; labels {out['f_label_count_9c']}; "
         f"label map {labB['counts']}; disagree {labB['n_disagree_with_stated']}")
    # pointers: the W189 chain on ctx_A with the v13 letter
    sub_lines = _rd(P2.SUBMITTED_MAIN).decode('utf-8').split('\n')
    n0 = len(P2.CHECK_FAILURES)
    _, adj_s = P2._num(sub_lines)
    sub_ctx = {'inn': W.objects_index(adj_s), 'n_labels': len(adj_s['labels'])}
    P = P2.task_pointers(ctxA, texts, sub_ctx)
    t7 = P2.table7_check(P['v12'], texts)
    p2_failures_since(n0, 'pointers')
    out['e_pointers_v13'] = rename(P, RENAME_V13)
    out['e_table7'] = t7
    out['e_w189_mismatch'] = task_w189_mismatch(P, w189)
    _log(f"(e) v13 {P['v12_counts']}; stale {P['stale_declarations']}; declaration checks {P['declaration_checks_all_hold']}; "
         f"W189 mismatch is l. 285 B.8: {out['e_w189_mismatch']['w189_mismatch_is_the_l285_Fig_B8_pointer']}; closed "
         f"{out['e_w189_mismatch']['closed']}")
    for r in P['v12_mismatch'] + P['v12_ambiguous']:
        _log(f"   {r['verdict']} l. {r['line']} {r['text']!r} kind {r['kind']} printed {r['numbers']} expected "
             f"{r.get('expected_at_head')}")
    if P['stale_declarations']:
        _fail(f'(e) stale declarations: {P["stale_declarations"]}')
    if not P['declaration_checks_all_hold']:
        _fail(f'(e) declaration checks: {[k for k, v in P["declaration_checks"].items() if not v["holds"]]}')
    # the cover letter's pointer (inside the W187 bracket), static, for information
    out['cover_letter_section_4_3'] = {'static_heading_4_3_at_9c': heading_title(ctxB['adj_n'], '4.3'),
                                       'note': 'the cover letter prints "(Section~4.3)" inside the W187 bracket; informational'}
    # dangling (W189 method on ctx_A: removed labels since a87e195)
    n0 = len(P2.CHECK_FAILURES)
    with p2_bound(HEAD_FULL=HEAD_FULL):
        dang, ctl_d = P2.task_dangling(ctxA, labA, texts)
    p2_failures_since(n0, 'dangling')
    out['f_dangling'] = dang
    _log(f"(f) undefined {dang['n_undefined']}; duplicates {dang['n_duplicate_labels']}; removed-label uses "
         f"{len(dang['uses_of_removed_labels_typeset'])}")
    # hygiene: ctx_B (lines changed 9b -> 9c) and ctx_A (every change since a87e195)
    no_cap = '5 no blue changed caption to build the unwrap control'
    n0 = len(P2.CHECK_FAILURES)
    with p2_bound(PREV_FULL=BASE_FULL, HEAD_FULL=HEAD_FULL):
        hyB, ctl_hB = P2.task_hygiene(ctxB, texts)
    p2_failures_since(n0, 'hygiene ctx_B (9b -> 9c)', declared=(no_cap,))
    n0 = len(P2.CHECK_FAILURES)
    with p2_bound(PREV_FULL=A87_FULL, HEAD_FULL=HEAD_FULL):
        hyA, ctl_hA = P2.task_hygiene(ctxA, texts)
    p2_failures_since(n0, 'hygiene ctx_A (a87e195 -> 9c)')
    out['f_hygiene_9b_9c'] = rename(hyB, RENAME_B)
    out['f_hygiene_a87e195_9c'] = hyA
    out['f_blue'] = task_blue(ctxA, ctxB, hyA, hyB)
    _log(f"(f) CONFIRM {hyB['n_confirm']}; tightened {hyB['n_tightened_raw']}; 9b->9c lines lacking blue "
         f"{hyB['n_lines_lacking_blue']}; vs a87e195 lines lacking blue {hyA['n_lines_lacking_blue']} "
         f"{[r['line'] for r in hyA['blue_changed_lines_lacking_blue']]}; changed 9b->9c without blue "
         f"{out['f_blue']['changed_lines_without_blue']}")
    # counts (ctx_B: head = 9c, prev = 9b)
    n0 = len(P2.CHECK_FAILURES)
    cn, ctl_n = P2.task_counts(ctxB)
    p2_failures_since(n0, 'counts')
    out['f_counts_raw'] = rename(cn, RENAME_B)
    out['f_counts'] = counts_compare(cn, w189)
    _log(f"(f) floats {out['f_counts']['main_text_floats_9c']}; equations {out['f_counts']['main_text_equations_9c']}; words "
         f"{out['f_counts']['main_text_prose_words_9c']} (W189 {out['f_counts']['main_text_prose_words_w189']}); method "
         f"reproduces W189 {out['f_counts']['method_reproduces_w189_at_46c9d77']}")
    # negative controls
    n0 = len(P2.CHECK_FAILURES)
    with p2_bound(STATED_NUMBERING={}, STATED_REMOVED_APPENDICES=()):
        ctl_l = P2.controls_label(ctxB)
    ctl_p = P2.controls_pointers(ctxA, texts, sub_ctx, P['v12'])
    p2_failures_since(n0, 'controls')
    ctl = ([{'id': f'a {f}: expert diff + line altered', 'refused': v['refused']}
            for f, v in out['a_identity']['diff_negative_controls'].items()]
           + ctl_l + ctl_p + control_b8(ctxA, texts, sub_ctx) + controls_multicolumn(ctxB) + ctl_d
           + [dict(c, id='ctx_B ' + c['id']) for c in ctl_hB] + [dict(c, id='ctx_A ' + c['id']) for c in ctl_hA] + ctl_n)
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f'negative control not refused: {c["id"]}')
    out['p2_failures_declared_not_applicable'] = {
        k: {'messages': v, 'why': ('no float caption changed between 9b and 9c (W189 P2 builds its caption-unwrap control from a '
                                   'changed caption); the same control is built on ctx_A (a87e195 -> 9c) and is listed '
                                   'under ctx_A' if 'ctx_B' in k else 'declared')} for k, v in EXPECTED_P2_FAILURES.items()}
    out['guards'] = verify_guards()
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    pj, pm, pn = (os.path.join(od, OUT_NAMES[k]) for k in ('json', 'md', 'man'))
    with open(pj, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, pm)
    inputs = sorted(set([os.path.join(CLONE, f) for f in FILES] + [os.path.join(CLONE, FIG6), os.path.join(MR, FIG6),
                         os.path.join(CLONE, 'elsarticle.dtx')] + list(REVIEW_COPY.values()) + list(PREV_COPY.values())
                        + list(EXPERT_DIFF.values()) + [W189P2_JSON, P2.LETTER_V9, P2.LETTER_V10, P2.SUBMITTED_MAIN,
                                                        P2.W183H_JSON, P2.W182C_JSON, P2.W180C_JSON, SCRIPT_NAME]
                        + list(IMPORTED_SCRIPTS)))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md')},
                'out_dir': out_dir, 'clone_head_full': out['a_identity']['clone_head'],
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(showb(c, f)) for c in (A87_FULL, BASE_FULL, HEAD_FULL) for f in FILES},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(CHECK_FAILURES)}; guards ok {out["guards"]["ok"]}; '
         f'pickle {P2.PICKLE_COUNTS}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
