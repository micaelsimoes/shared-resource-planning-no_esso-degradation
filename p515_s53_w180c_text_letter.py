"""
P5.15 Addendum 79, W180 item (c), Planner task W180c: the nine main-text edits of round 6 (STEP6_ROUND6_CORRECTIONS.md
A.1-A.9; A.10 = Appendix E belongs to W180a and is only located here) and the letter v7 edits (section B), verbatim at
the Overleaf HEAD e5322f8 (main.tex = manuscript_review/main_48099ab_round6.tex; letter = response_to_reviewers_v7.tex);
the pointer check of the whole letter v7 at that HEAD; the re-score of the W179b rows C-1, R14-2, C47-1, C48-1, R22-b,
FC-2c, the G12 follow-on and the A.3 prefix; and the CONFIRM / appendix-prefix / AUTHOR counts.
ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT; read-only on the clone (git show / diff / rev-parse / status
only), manuscript_review/, the frozen JSON, every committed artifact and every earlier script
(p515_s53_w179b_checks.py is IMPORTED, not edited; it imports W178c and its chain, also not edited; W179b's committed
JSON is read, never written).

What it does (static reading only):
  0. Identity: clone HEAD (declared e5322f8, full hash checked), one commit after 48099ab, clone clean; main.tex /
     letter sha256 and line counts; byte-identity with main_48099ab_round6.tex / response_to_reviewers_v7.tex; the
     48099ab blobs are W179b's (main_e926fb7_appE.tex, letter v6); the +/- lines of the clone's own `git diff
     48099ab HEAD` (and of a difflib diff) equal those of main_48099ab_round6.diff (16 hunks) and letter_v6_v7.diff
     (5 hunks).
  1. Capture-path checklist, asserted BEFORE any task runs (exit 4 naming the missing item).
  2. Main text: each of A.1-A.9 stated as exact (old -> new) replacement pairs copied from STEP6 section A (or, where
     STEP6 gives only the new text, the old text from the expert's diff, so marked); each old string occurs exactly
     once outside Appendix E at 48099ab (its line compared with STEP6's), each new string exactly once at HEAD; the
     A.1 label line under the 2.3.2 heading; then RECONSTRUCTION: applying the declared pairs (and the label insertion)
     to 48099ab outside Appendix E must give HEAD outside Appendix E exactly (so no other main-text change exists).
     Appendix E: every hunk of the clone diff that is not an A.1-A.9 hunk lies wholly inside the Appendix E block, at
     48099ab and at HEAD.
  3. Letter v7: the section-B edits as (old -> new) pairs; RECONSTRUCTION on whitespace-collapsed text (the letter
     wraps lines and R2.5 crosses a line break); changed lines; the pointer multiset of v6 and v7 differ only by
     "Appendix~E" -> "Appendix~E.6" on the R2.5 line.
  4. Static numbering at e5322f8 and at 48099ab (W178c's numbering_adjusted): every label's number compared; the only
     added label must be subsubsec:capacity at 2.3.2; 0 removed, 0 changed; 0 undefined refs, 0 duplicates. Every
     numbered pointer of letter v7 against the static numbering at e5322f8 (MATCH / MISMATCH / SUBMITTED / AMBIGUOUS),
     content needles carried from W179b's committed rows plus one for "Appendix E.6"; verdicts compared with W179b's.
  5. The eight W179b rows re-scored (closed / still / new), each with located, checked evidence.
  6. Counts at HEAD: CONFIRM (main.tex and letter, comment lines included), every "<word>~\\ref{app:" (general pattern;
     label-word prefixes separated from prepositions), "% [AUTHOR]" comments.
  7. Negative controls (in memory / a temporary directory outside the repository): an altered number in an edit, an
     altered word outside every edit, a section inserted before 3.3, a removed needle sentence and an undeclared letter
     pointer number must each be refused.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at exit, with
the guards of the imported W179b / W178c / W177b-3' / W177b-3 / W167 / Phase B record modules; pickle.load / loads
blocked and counted (0 expected); no production module imported (asserted). Refuses to run concurrently with another
copy of itself. Output JSON written through gate_result_io (string flags refused).

Output (write-once): data/SRP1/Results/P515S53/w180c_text_letter/ must exist and hold nothing but launch.log; writes
w180c_text_letter.json, w180c_text_letter.md and manifest_sha256.json. `--post-run` (after the run and the W100 typing
test) writes manifest_post_run_sha256.json once. `--out-dir DIR`: trial runs only (absolute path outside the repository).

Commands (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w180c_text_letter && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180c_text_letter.py \\
      > data/SRP1/Results/P515S53/w180c_text_letter/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
      --out data/SRP1/Results/P515S53/w180c_text_letter/w180c_bool_typing_test.json \\
      > data/SRP1/Results/P515S53/w180c_text_letter/w180c_bool_typing_test.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180c_text_letter.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label="P5.15 W180c text + letter checks (never solves)").install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W180c: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W180c: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s53_w179b_checks as B  # noqa: E402  (W179b; installs its own permitted=() guard; NOT edited)
import gate_result_io as GRIO  # noqa: E402

C, R, P3, PB, W167 = B.C, B.R, B.P3, B.PB, B.W167
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w180c_text_letter')
OUT_NAME = 'w180c_text_letter.json'
MD_NAME = 'w180c_text_letter.md'
MANIFEST_NAME = 'manifest_sha256.json'
POST_NAME = 'manifest_post_run_sha256.json'
TYPING_NAMES = ('w180c_bool_typing_test.json', 'w180c_bool_typing_test.log')
CLONE = B.CLONE
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
LETTER_NAME = 'response_to_reviewers_draft.tex'
DTX = os.path.join(CLONE, 'elsarticle.dtx')
MR = 'manuscript_review'
MAIN_R6 = os.path.join(MR, 'main_48099ab_round6.tex')
MAIN_PREV_COPY = os.path.join(MR, 'main_e926fb7_appE.tex')
LETTER_V6 = os.path.join(MR, 'response_to_reviewers_v6.tex')
LETTER_V7 = os.path.join(MR, 'response_to_reviewers_v7.tex')
DIFF_MAIN = os.path.join(MR, 'main_48099ab_round6.diff')
DIFF_LETTER = os.path.join(MR, 'letter_v6_v7.diff')
STEP6 = 'STEP6_ROUND6_CORRECTIONS.md'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
DECLARED_HEAD = 'e5322f8'
DECLARED_HEAD_FULL = 'e5322f85dced8034312c9afb97c19001bce6bfa0'
DECLARED_MAIN_SHA = 'b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492'
DECLARED_MAIN_LINES = 2710
DECLARED_LETTER_SHA = '34a5754ba6ccb53e29d64fad38fddc921c3048479fc5c31eef39664d9a0efa38'
DECLARED_LETTER_LINES = 292
PREV = '48099ab'
PREV_MAIN_SHA = B.DECLARED_MAIN_SHA          # W179b's HEAD hashes
PREV_LETTER_SHA = B.DECLARED_LETTER_SHA
N_HUNKS_MAIN, N_HUNKS_LETTER = 16, 5
W179B_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w179b_text_letter', 'w179b_checks.json')
W179B_JSON_SHA_PREFIX = '88221b8d'
W179B_SCRIPT, W179B_SCRIPT_SHA_PREFIX = 'p515_s53_w179b_checks.py', '69e24631'
FZ, FZ_SHA = B.FZ, B.FZ_SHA
FORBIDDEN_MODULE_PREFIXES = B.FORBIDDEN_MODULE_PREFIXES
CHECK_FAILURES = []
SCRIPT_NAME = os.path.basename(__file__)
APPE_START_MARK = '% APPENDIX E -- EVALUATION RECORD'
DECL_MARK = r'\section*{Declaration of generative AI'
# STEP6 section A.10: "the block l. 2374-2701 of 48099ab is replaced whole"; deliverables table: "= lines 2375-2694 of
# the new main.tex"
EXPERT_APPE_PREV, EXPERT_APPE_HEAD = (2374, 2701), (2375, 2694)


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


def sha_text(t):
    return hashlib.sha256(t.encode('utf-8')).hexdigest()


def ws(s):
    """Collapse whitespace runs to one space (the only normalisation used; '~' and all other characters kept)."""
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
        if argv and 'python' in os.path.basename(argv[0]) and any(os.path.basename(a) == SCRIPT_NAME for a in argv[1:]):
            others.append(ln.strip())
    if others:
        print(f'REFUSED: another copy of {SCRIPT_NAME} is running: {others}', flush=True)
        sys.exit(1)


# ======================================================================================================================
#  0. identity
# ======================================================================================================================
def _pm_lines(diff_text):
    minus, plus = [], []
    for ln in diff_text.split('\n'):
        if ln.startswith(('---', '+++', '@@', 'diff --git', 'index ')):
            continue
        if ln.startswith('-'):
            minus.append(ln[1:])
        elif ln.startswith('+'):
            plus.append(ln[1:])
    return minus, plus


def _unified(a_text, b_text):
    return '\n'.join(difflib.unified_diff(a_text.split('\n'), b_text.split('\n'), lineterm='', n=0))


def git_hunks(path):
    """Hunks of the clone's own `git diff -U0 PREV HEAD -- path`: [(old_start, old_len, new_start, new_len)]."""
    d = _git(['diff', '--no-color', '--no-ext-diff', '-U0', PREV, 'HEAD', '--', path], cwd=CLONE)
    hunks = []
    for ln in d.split('\n'):
        m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', ln)
        if m:
            hunks.append((int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)))
    return d, hunks


def identity():
    head = _git(['rev-parse', '--short', 'HEAD'], cwd=CLONE).strip()
    head_full = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    revs = _git(['rev-list', f'{PREV}..HEAD'], cwd=CLONE).split()
    main_text, letter_text = _read(MAIN_TEX), _read(LETTER)
    mp, mh = show(PREV), show('HEAD')
    lp, lh = show(PREV, LETTER_NAME), show('HEAD', LETTER_NAME)
    d = {'clone_head': head, 'clone_head_full': head_full,
         'clone_head_subject': _git(['log', '-1', '--format=%s'], cwd=CLONE).strip(),
         'clone_head_date': _git(['log', '-1', '--format=%cI'], cwd=CLONE).strip(),
         'commits_after_48099ab': revs, 'clone_tracked_modified': status,
         'main_sha256': _sha(MAIN_TEX), 'main_lines': main_text.count('\n'),
         'letter_sha256': _sha(LETTER), 'letter_lines': letter_text.count('\n'),
         'blob_sha256': {f'{PREV}:main.tex': sha_text(mp), 'HEAD:main.tex': sha_text(mh),
                         f'{PREV}:letter': sha_text(lp), 'HEAD:letter': sha_text(lh)},
         'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip()}
    diffs = {}
    for name, a, b, committed, path in (('main 48099ab -> HEAD vs main_48099ab_round6.diff', mp, mh, DIFF_MAIN, 'main.tex'),
                                        ('letter 48099ab -> HEAD vs letter_v6_v7.diff', lp, lh, DIFF_LETTER, LETTER_NAME)):
        want = _pm_lines(_read(committed))
        gtext, ghunks = git_hunks(path)
        row = {'n_hunks_committed': sum(1 for ln in _read(committed).split('\n') if ln.startswith('@@')),
               'n_hunks_git_U0': len(ghunks), 'n_minus_committed': len(want[0]), 'n_plus_committed': len(want[1])}
        for tool, got in (('git_diff', _pm_lines(gtext)), ('difflib', _pm_lines(_unified(a, b)))):
            net = lambda pm: (Counter(pm[0]) - Counter(pm[1]), Counter(pm[1]) - Counter(pm[0]))  # noqa: E731
            row[tool] = {'ordered_equal': got == want, 'sorted_equal': sorted(got[0]) == sorted(want[0]) and sorted(got[1]) == sorted(want[1]),
                         'net_equal': net(got) == net(want), 'n_minus': len(got[0]), 'n_plus': len(got[1])}
        diffs[name] = row
    d['diffs'] = diffs
    checks = {'head': head == DECLARED_HEAD, 'head_full': head_full == DECLARED_HEAD_FULL,
              'main_sha': d['main_sha256'] == DECLARED_MAIN_SHA, 'main_lines': d['main_lines'] == DECLARED_MAIN_LINES,
              'letter_sha': d['letter_sha256'] == DECLARED_LETTER_SHA, 'letter_lines': d['letter_lines'] == DECLARED_LETTER_LINES,
              'main_equals_main_48099ab_round6': _sha(MAIN_TEX) == _sha(MAIN_R6),
              'letter_equals_v7': _sha(LETTER) == _sha(LETTER_V7),
              'head_blobs_equal_working_copy': sha_text(mh) == _sha(MAIN_TEX) and sha_text(lh) == _sha(LETTER),
              '48099ab_main_is_W179bs': sha_text(mp) == PREV_MAIN_SHA and sha_text(mp) == _sha(MAIN_PREV_COPY),
              '48099ab_letter_is_W179bs_v6': sha_text(lp) == PREV_LETTER_SHA and sha_text(lp) == _sha(LETTER_V6),
              'one_commit_after_48099ab': len(revs) == 1 and revs[0] == DECLARED_HEAD_FULL,
              'clone_clean': status == '', 'frozen_json': _sha(FZ) == FZ_SHA,
              'w179b_json': _sha(W179B_JSON).startswith(W179B_JSON_SHA_PREFIX),
              'w179b_script': _sha(W179B_SCRIPT).startswith(W179B_SCRIPT_SHA_PREFIX),
              'main_hunks_16': diffs['main 48099ab -> HEAD vs main_48099ab_round6.diff']['n_hunks_committed'] == N_HUNKS_MAIN,
              'letter_hunks_5': diffs['letter 48099ab -> HEAD vs letter_v6_v7.diff']['n_hunks_committed'] == N_HUNKS_LETTER}
    for name, v in diffs.items():
        checks[f'git diff +/- lines equal (as multisets): {name}'] = v['git_diff']['sorted_equal']
        checks[f'difflib +/- lines equal (net): {name}'] = v['difflib']['net_equal']
    d['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'identity check {k} failed')
    return d


# ======================================================================================================================
#  1. capture-path checklist
# ======================================================================================================================
REQUIRED_CAPTURES = {
    'identity (HEAD, sha256, lines, byte-identity, diff +/- lines)': 'identity',
    '(2) A.1-A.9 verbatim, located; reconstruction outside Appendix E; Appendix E hunks inside the block': 'task_main_edits',
    '(3) letter v7 section-B edits verbatim; reconstruction; pointer multiset v6 -> v7': 'task_letter_edits',
    '(4) static numbering e5322f8 vs 48099ab; one label added': 'task_numbering',
    '(4) every letter v7 pointer: verdict, number at HEAD, W179b verdict, needles': 'task_pointers',
    '(5) eight W179b rows re-scored': 'task_rows',
    '(6) CONFIRM / <word>~\\ref{app: / [AUTHOR] counts': 'task_counts',
    'negative controls': 'task_negative_controls',
    'guards (solve 0, pickle 0, no production import)': 'verify_guards',
}
REQUIRED_B_SYMBOLS = ('target_v6', 'resolve_v6', 'region_v6', 'line_map', 'task_elsarticle', 'GUARD', 'PICKLE_COUNTS', 'SEC_EXTRA')
REQUIRED_C_SYMBOLS = ('numbering_adjusted', 'task_refs', '_consecutive', 'GUARD', 'PICKLE_COUNTS')
REQUIRED_R_SYMBOLS = ('letter_pointers', 'cite_order', 'GUARD', 'PICKLE_COUNTS')


def capture_checklist():
    missing = []
    g = globals()
    for q, fn in REQUIRED_CAPTURES.items():
        if not callable(g.get(fn)):
            missing.append(f'capture function {fn} for "{q}"')
    for mod, name, syms in ((B, 'B', REQUIRED_B_SYMBOLS), (C, 'C', REQUIRED_C_SYMBOLS), (R, 'R', REQUIRED_R_SYMBOLS)):
        for s in syms:
            if not hasattr(mod, s):
                missing.append(f'symbol {name}.{s}')
    for p in [MAIN_TEX, LETTER, DTX, MAIN_R6, MAIN_PREV_COPY, LETTER_V6, LETTER_V7, DIFF_MAIN, DIFF_LETTER, STEP6, BRIEF,
              W179B_JSON, W179B_SCRIPT, FZ]:
        if not os.path.isfile(p):
            missing.append(f'input {p}')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'b_symbols': list(REQUIRED_B_SYMBOLS), 'c_symbols': list(REQUIRED_C_SYMBOLS),
            'r_symbols': list(REQUIRED_R_SYMBOLS), 'all_present': True}


# ======================================================================================================================
#  2. main text: A.1-A.9
# ======================================================================================================================
# (id, where, STEP6 line at 48099ab, [(old, new)], source of old, source of new). Every string is copied from
# STEP6_ROUND6_CORRECTIONS.md section A (markdown line wraps joined by one space) unless marked 'expert diff'.
MAIN_EDITS = [
    ('A.1', '2.2.6', 697, [
        (r'are fixed parameters of the operational subproblem. The resulting evaluated operating cost',
         r'are held fixed in the operational subproblem, where they enter as the parameters '
         r'$\hat{S}^{\text{Inv}}_{e,y^{\text{Inv}}}$ and $\hat{E}^{\text{Inv}}_{e,y^{\text{Inv}}}$ of '
         r'Subsection~\ref{subsubsec:capacity}. The resulting evaluated operating cost')], 'STEP6 A.1', 'STEP6 A.1'),
    ('A.2', '2.2.7 Determinacy', 743, [
        (r'every cell it proposed was re-evaluated under this rule before being reported',
         r'every incumbent it proposed was re-evaluated under this rule before being reported'),
        (r'the bands, gaps and slacks of every evaluation used in a reported comparison are tabulated in~\ref{app:results}.',
         r'the bands, gaps and slacks of every single-scenario evaluation used in a reported comparison are tabulated '
         r'in~\ref{app:results}; the multi-scenario evaluations, stopped by the production exit, carry the resolution '
         r'stated with them in Table~\ref{tab:res_multiscenario}.')],
     'STEP6 A.2', 'STEP6 A.2 (first pair: "re-evaluated ..." completed by the unchanged words of the old text)'),
    ('A.3', '2.3.3 closure sentence', 968, [
        (r'Both daily closure slacks were zero, within solver tolerance, at every certified evaluation reported in this study.',
         r'At every certificate used in a reported comparison, and at both multi-scenario evaluations, both daily closure '
         r"slacks sat at their lower bound to within the solver's bound relaxation ($10^{-6}$~MWh per slack); in the "
         r'superseded evaluation~16 of Table~\ref{tab:appE_evaluations} their sum reached $1.8 \times 10^{-6}$~MWh, '
         r'$5 \times 10^{-6}$ of the available energy. Without storage the closure row is not built.')],
     'expert diff (STEP6 A.3 says "Replace the sentence by")', 'STEP6 A.3 (code block)'),
    ('A.4', '4.7 first sentence', 1689, [
        (r'Every certificate reported in this paper was decided, or re-decided from its committed record, under the rule',
         r'Every certificate used in a reported comparison was decided, or re-decided from its committed record, under the rule'),
        (r'and certifies at the cycle reported; the single-scenario tables carry these certificates.',
         r'and certifies at the cycle reported; the single-scenario tables carry these certificates. One earlier '
         r'certificate, issued under a previous version of the rule and since superseded, is recorded in '
         r'\ref{app:results} and used in no comparison.')],
     'STEP6 A.4 (its "..." spans filled by the words common to its old and new text)', 'STEP6 A.4'),
    ('A.5', '4.7 last sentence', 1690, [
        (r'Every comparison and every evaluation behind this section is listed in~\ref{app:results}, together with the '
         r'full certification statistics and the reproducibility record.',
         r'Every comparison of this section, and every single-scenario evaluation behind it, is listed '
         r'in~\ref{app:results}, together with the full certification statistics and the reproducibility record; the '
         r"search's own evaluations, stopped at the production exit, are reported by their counts, and the multi-scenario "
         r'pair in Table~\ref{tab:res_multiscenario}.')], 'STEP6 A.5', 'STEP6 A.5'),
    ('A.6', '4.8 degenerate duals', 1716, [
        (r'split in the same proportions among the three interface nodes (Table~\ref{tab:appE_gap})',
         r'split in the same proportions among the three interface nodes wherever the split was recorded '
         r'(Table~\ref{tab:appE_gap})')], 'STEP6 A.6', 'STEP6 A.6'),
    ('A.7', '4.8 Scope', 1724, [
        (r'with every evaluated unit neighbour of the plan without storage determinately worse.',
         r'with every evaluated unit neighbour of the plan without storage worse, determinately for the eight '
         r're-evaluated under the certification rule (Table~\ref{tab:appE_neighbours}) and beyond the search resolution '
         r'for the other six.')], 'STEP6 A.7', 'STEP6 A.7'),
    ('A.8', 'Appendix A.3', 1899, [
        (r'(Subsection~\ref{app:admm_updated_stopping})', r'(\ref{app:admm_updated_stopping})')], 'STEP6 A.8', 'STEP6 A.8'),
    ('A.9', 'Conclusions', 1736, [
        (r'or, at the multi-scenario instance, stopped by the production exit with its remaining movement measured --- '
         r'and every planning comparison is reported with the resolution at which it is decided.',
         r'or, at the multi-scenario instance, stopped by the production exit, with the remaining movement of the plan '
         r'without storage estimated from a continued run --- and every planning comparison is reported with the '
         r"resolution at which it is decided, the search's own evaluations at the search resolution.")],
     'STEP6 A.9', 'STEP6 A.9'),
]
LABEL_HEADING = r'\subsubsection{\textcolor{blue}{Installed and Available Capacity}}'
LABEL_LINE = r'\label{subsubsec:capacity}'
LABEL_STEP6_LINE = 768


def appe_block(lines):
    """(first, last) 1-based lines of the Appendix E block: from the '% ====' rule above the APPENDIX E banner to the
    line before the Declaration heading; 'last_nonblank_minus' = the Declaration line - 2 (the expert's convention)."""
    i = next(k for k, ln in enumerate(lines) if ln.startswith(APPE_START_MARK))
    s = i if lines[i - 1].startswith('% ====') else i + 1           # 0-based index of the rule line -> 1-based i
    e = next(k for k, ln in enumerate(lines) if ln.startswith(DECL_MARK))
    return s, e, e - 1         # first (1-based), last (1-based, line before Declaration), expert-style last


def _find_lines(lines, s, lo=1, hi=None):
    hi = hi or len(lines)
    return [k for k in range(lo, hi + 1) if s in lines[k - 1]]


def main_edit_check(prev_lines, head_lines):
    """Pure: returns the per-edit rows and the reconstruction result (no _fail)."""
    ps, pe, pe_x = appe_block(prev_lines)
    hs, he, he_x = appe_block(head_lines)
    prev_out = prev_lines[:ps - 1]
    head_out = head_lines[:hs - 1]
    prev_tail, head_tail = prev_lines[pe:], head_lines[he:]
    rows = []
    rec = list(prev_out)
    for eid, where, l48, pairs, osrc, nsrc in MAIN_EDITS:
        prow = []
        for old, new in pairs:
            old_hits = _find_lines(prev_out, old)
            old_count = sum(ln.count(old) for ln in prev_out)
            new_hits = _find_lines(head_out, new)
            new_count = sum(ln.count(new) for ln in head_out)
            old_left = sum(ln.count(old) for ln in head_out) if old not in new else None
            ok = old_count == 1 and new_count == 1 and (old_left in (0, None))
            prow.append({'old': old, 'new': new, 'old_lines_48099ab': old_hits, 'old_count_48099ab': old_count,
                         'new_lines_head': new_hits, 'new_count_head': new_count,
                         'old_count_head': old_left, 'ok': ok})
            if old_count == 1:
                k = old_hits[0] - 1
                rec[k] = rec[k].replace(old, new)
        line_ok = all(p['old_lines_48099ab'] == [l48] for p in prow)
        rows.append({'id': eid, 'where': where, 'step6_line_48099ab': l48, 'old_source': osrc, 'new_source': nsrc,
                     'pairs': prow, 'verbatim_at_head': all(p['ok'] for p in prow),
                     'line_head': sorted({x for p in prow for x in p['new_lines_head']}),
                     'step6_line_matches_48099ab': line_ok})
    # A.1b label line
    h_prev = _find_lines(prev_out, LABEL_HEADING)
    h_head = _find_lines(head_out, LABEL_HEADING)
    lab_head = _find_lines(head_out, LABEL_LINE)
    lab_prev = _find_lines(prev_out, LABEL_LINE)
    label_ok = (len(h_prev) == 1 and len(h_head) == 1 and lab_prev == [] and lab_head == [h_head[0] + 1]
                and head_out[h_head[0]] == LABEL_LINE and h_prev[0] == LABEL_STEP6_LINE)
    if len(h_prev) == 1:
        rec.insert(h_prev[0], LABEL_LINE)
    rows.insert(1, {'id': 'A.1b', 'where': '2.3.2 heading', 'step6_line_48099ab': LABEL_STEP6_LINE,
                    'heading_line_48099ab': h_prev, 'heading_line_head': h_head, 'label_line_head': lab_head,
                    'label_line_48099ab': lab_prev, 'label_text_exact': lab_head != [] and head_out[lab_head[0] - 1] == LABEL_LINE,
                    'verbatim_at_head': label_ok, 'step6_line_matches_48099ab': h_prev == [LABEL_STEP6_LINE],
                    'line_head': lab_head})
    recon_equal = rec == head_out
    first_diff = None
    if not recon_equal:
        for k in range(max(len(rec), len(head_out))):
            a = rec[k] if k < len(rec) else None
            b = head_out[k] if k < len(head_out) else None
            if a != b:
                first_diff = {'line': k + 1, 'reconstructed': (a or '')[:300], 'head': (b or '')[:300]}
                break
    tail_equal = prev_tail == head_tail
    # hunks of the clone's own git diff
    _, gh = git_hunks('main.tex')
    a_lines = {r_['step6_line_48099ab'] for r_ in rows}
    hunk_rows = []
    for os_, on, ns, nn in gh:
        o_rng = (os_, os_ + on - 1) if on else (os_, os_)       # pure insertion: after line os_
        n_rng = (ns, ns + nn - 1) if nn else (ns, ns)
        in_e = ps <= o_rng[0] and o_rng[1] <= pe and hs <= n_rng[0] and n_rng[1] <= he
        if on == 0:
            a_edit = [r_['id'] for r_ in rows if r_['id'] == 'A.1b' and os_ == r_['step6_line_48099ab']]
        else:
            a_edit = sorted({r_['id'] for r_ in rows if r_['id'] != 'A.1b' and o_rng[0] <= r_['step6_line_48099ab'] <= o_rng[1]})
        hunk_rows.append({'old': [os_, on], 'new': [ns, nn], 'inside_appendix_E': in_e, 'a_edits': a_edit,
                          'classified': bool(a_edit) != in_e})
    covered = {x for h in hunk_rows for x in h['a_edits']}
    out = {'appendix_E_block_48099ab': [ps, pe], 'appendix_E_block_head': [hs, he],
           'appendix_E_block_expert_convention': {'48099ab': [ps, pe_x], 'head': [hs, he_x],
                                                  'expert_48099ab': list(EXPERT_APPE_PREV), 'expert_head': list(EXPERT_APPE_HEAD),
                                                  'agree': (ps, pe_x) == EXPERT_APPE_PREV and (hs, he_x) == EXPERT_APPE_HEAD},
           'rows': rows, 'reconstruction_outside_appendix_E_equal': recon_equal, 'reconstruction_first_difference': first_diff,
           'after_appendix_E_equal': tail_equal, 'git_hunks': hunk_rows,
           'n_hunks_main_text': sum(1 for h in hunk_rows if h['a_edits']), 'n_hunks_appendix_E': sum(1 for h in hunk_rows if h['inside_appendix_E']),
           'every_hunk_classified': all(h['classified'] for h in hunk_rows),
           'every_edit_has_a_hunk': covered == {r_['id'] for r_ in rows}}
    out['ok'] = (all(r_['verbatim_at_head'] for r_ in rows) and recon_equal and tail_equal and out['every_hunk_classified']
                 and out['every_edit_has_a_hunk'])
    return out


def task_main_edits(prev_lines, head_lines):
    out = main_edit_check(prev_lines, head_lines)
    for r_ in out['rows']:
        if not r_['verbatim_at_head']:
            _fail(f'main edit {r_["id"]} not verbatim at HEAD: {r_}')
        if not r_['step6_line_matches_48099ab']:
            _fail(f'main edit {r_["id"]}: STEP6 line {r_["step6_line_48099ab"]} is not where the old text is at 48099ab')
    if not out['reconstruction_outside_appendix_E_equal']:
        _fail(f'reconstruction outside Appendix E differs: {out["reconstruction_first_difference"]}')
    if not out['after_appendix_E_equal']:
        _fail('the text after Appendix E changed')
    if not out['every_hunk_classified'] or not out['every_edit_has_a_hunk']:
        _fail(f'hunk classification: {out["git_hunks"]}')
    if not out['appendix_E_block_expert_convention']['agree']:
        _fail(f'Appendix E block lines differ from STEP6: {out["appendix_E_block_expert_convention"]}')
    return out


# ======================================================================================================================
#  3. letter v7
# ======================================================================================================================
LETTER_EDITS = [   # (id, [(old, new)], source); compared on whitespace-collapsed text
    ('R1.4 fuel out of the clause', [
        (r'technology maturation, as trajectories of future storage investment cost; fuel costs, through the wholesale '
         r'prices at which conventional generation is costed (Section~3.3), and market prices',
         r'technology maturation, as trajectories of future storage investment cost, and market prices')],
     'expert diff (STEP6 B: "Fuel taken out of the \'introduction now states\' clause")'),
    ('R1.4 closing sentence', [
        (r'left to future work. On (ii):',
         r'left to future work. Fuel costs are not represented separately: conventional generation in the transmission '
         r'network is priced at the wholesale energy price of the scenario (Section~3.3), so fuel-cost uncertainty enters '
         r'only through the market-price scenarios. On (ii):')], 'STEP6 B R1.4 (sentence verbatim; placement from the diff)'),
    ('R2.2', [(r'and the cycles of every evaluation in Appendix~E',
               r'and the cycles of every evaluation behind a reported comparison in Appendix~E')], 'STEP6 B R2.2'),
    ('R2.5', [(r'bitwise replays of every gated evaluation (Section~4.7 and Appendix~E)',
               r'bitwise replays of every evaluation repeated from an earlier record (Section~4.7 and Appendix~E.6)')],
     'STEP6 B R2.5'),
    ('Further changes, Appendix E bullet', [
        (r'every recourse evaluation, with its stopping status',
         r'every single-scenario recourse evaluation those comparisons use, with its stopping status')],
     'STEP6 B Further changes (its "..." = the unchanged rest)'),
    ('header', [(r"% Response to reviewers — DRAFT for the author (expert's skeleton, 2026-10-06).",
                 r"% Response to reviewers — v7 (expert, 2026-10-15; round 6 / Addendum 79). DRAFT for the author "
                 r"(expert's skeleton, 2026-10-06).")], 'expert diff (STEP6 B: "Header comment dated v7")'),
]


def letter_edit_check(v6_text, v7_text):
    a, b = ws(v6_text), ws(v7_text)
    rows, rec = [], a
    for eid, pairs, src in LETTER_EDITS:
        pr = []
        for old, new in pairs:
            o, n = ws(old), ws(new)
            oc, nc = a.count(o), b.count(n)
            left = b.count(o) if o not in n else None
            pr.append({'old': old, 'new': new, 'old_count_v6': oc, 'new_count_v7': nc, 'old_count_v7': left,
                       'ok': oc == 1 and nc == 1 and left in (0, None)})
            if oc == 1:
                rec = rec.replace(o, n)
        rows.append({'id': eid, 'source': src, 'pairs': pr, 'verbatim_at_head': all(p['ok'] for p in pr),
                     'lines_v7': sorted({i for p in pr for i, ln in enumerate(v7_text.split('\n'), 1)
                                         if ws(p['new'])[:60] in ws(ln) or ws(p['new'])[-60:] in ws(ln)})})
    v6l, v7l = v6_text.split('\n'), v7_text.split('\n')
    mp, ops = B.line_map(v6l, v7l)
    changed = [j for j in range(1, len(v7l) + 1) if j not in mp]
    # the "introduction now states" clause of R1.4 at HEAD carries no "fuel"
    r14 = next(ln for ln in v7l if ln.startswith(r'\rresponse{On (i): the introduction now states'))
    clause = r14[:r14.index('left to future work')]
    out = {'rows': rows, 'reconstruction_equal': rec == b, 'changed_lines_v7': changed,
           'line_count_v6_v7': [len(v6l), len(v7l)], 'fuel_in_introduction_clause': 'fuel' in clause.lower(),
           'introduction_clause': clause}
    if rec != b:
        sm = difflib.SequenceMatcher(a=rec, b=b, autojunk=False)
        out['reconstruction_differences'] = [(t, rec[i1:i2][:200], b[j1:j2][:200]) for t, i1, i2, j1, j2 in sm.get_opcodes()
                                             if t != 'equal'][:5]
    out['ok'] = all(r_['verbatim_at_head'] for r_ in rows) and rec == b and not out['fuel_in_introduction_clause']
    return out


def pointer_multiset_change(v6_path, v7_path):
    def ms(path):
        return Counter((a['line'], a['kind'], a['text'], tuple(a['numbers'])) for a in R.letter_pointers(path))
    p6, p7 = ms(v6_path), ms(v7_path)
    return {'n_v6': sum(p6.values()), 'n_v7': sum(p7.values()),
            'only_in_v6': [list(k) for k in (p6 - p7)], 'only_in_v7': [list(k) for k in (p7 - p6)]}


def task_letter_edits(v6_text, v7_text):
    out = letter_edit_check(v6_text, v7_text)
    out['pointer_change'] = pointer_multiset_change(LETTER_V6, LETTER)
    expect = {'only_in_v6': [[160, 'appendix', 'Appendix~E)', ('E',)]], 'only_in_v7': [[160, 'appendix', 'Appendix~E.6)', ('E.6',)]]}
    pc = out['pointer_change']
    out['pointer_change_is_only_E_to_E6'] = (pc['only_in_v6'] == expect['only_in_v6'] and pc['only_in_v7'] == expect['only_in_v7'])
    for r_ in out['rows']:
        if not r_['verbatim_at_head']:
            _fail(f'letter edit {r_["id"]} not verbatim at HEAD: {r_["pairs"]}')
    if not out['reconstruction_equal']:
        _fail(f'letter reconstruction differs: {out.get("reconstruction_differences")}')
    if out['fuel_in_introduction_clause']:
        _fail('"fuel" still in the R1.4 introduction clause')
    if not out['pointer_change_is_only_E_to_E6']:
        _fail(f'pointer multiset change v6 -> v7: {pc}')
    return out


# ======================================================================================================================
#  4. numbering and pointers
# ======================================================================================================================
def numbering_compare(head_lines, prev_lines):
    raw, adj = C.numbering_adjusted(head_lines)
    _, adj_p = C.numbering_adjusted(prev_lines)
    changed, gone, new = [], [], []
    for k, v in adj_p['labels'].items():
        if k not in adj['labels']:
            gone.append({'label': k, 'number_48099ab': v['number']})
        elif adj['labels'][k]['number'] != v['number']:
            changed.append({'label': k, '48099ab': v['number'], 'head': adj['labels'][k]['number']})
    for k, v in adj['labels'].items():
        if k not in adj_p['labels']:
            new.append({'label': k, 'kind': v['kind'], 'number': v['number'], 'line': v['line']})
    n_e = sum(1 for k in adj_p['labels'] if B.APPE_LABEL.match(k))
    return raw, adj, {'n_labels_head': len(adj['labels']), 'n_labels_48099ab': len(adj_p['labels']),
                      'n_compared': sum(1 for k in adj_p['labels'] if k in adj['labels']),
                      'n_compared_outside_appendix_E': sum(1 for k in adj_p['labels'] if k in adj['labels'] and not B.APPE_LABEL.match(k)),
                      'n_compared_appendix_E': n_e, 'labels_changed': changed, 'labels_removed': gone, 'labels_new': new}


def task_numbering(head_lines, prev_lines):
    before = list(C.CHECK_FAILURES)
    raw, adj, out = numbering_compare(head_lines, prev_lines)
    if C.CHECK_FAILURES != before:
        _fail(f'W178c numbering_adjusted reported: {C.CHECK_FAILURES[len(before):]}')
    out['refs_static_head'] = C.task_refs(head_lines, adj)
    heading = [o for o in adj['objects'] if o['kind'] == 'subsubsection' and 'Installed and Available Capacity' in o['title']]
    out['heading_2_3_2'] = [{'number': o['number'], 'line': o['line'], 'title': o['title']} for o in heading]
    out['only_new_label_is_subsubsec_capacity_at_2_3_2'] = (
        [n['label'] for n in out['labels_new']] == ['subsubsec:capacity'] and out['labels_new'][0]['number'] == '2.3.2'
        and len(heading) == 1 and heading[0]['number'] == '2.3.2')
    out['elsarticle'] = B.task_elsarticle()
    if (out['labels_changed'] or out['labels_removed'] or not out['only_new_label_is_subsubsec_capacity_at_2_3_2']
            or out['refs_static_head']['undefined_refs'] or out['refs_static_head']['duplicate_labels']):
        _fail(f'numbering: changed {out["labels_changed"]}, removed {out["labels_removed"]}, new {out["labels_new"]}, '
              f'undefined {out["refs_static_head"]["undefined_refs"]}, dup {out["refs_static_head"]["duplicate_labels"]}')
    return raw, adj, out


NEEDLES_REPLACED = {   # W179b needles whose manuscript sentence a round-6 edit rewrote: (line, kind, first number) -> (old, new, why)
    (136, 'section', '2.2.6'): ('are fixed parameters of the operational subproblem', 'are held fixed in the operational subproblem',
                                'A.1 rewrote the 2.2.6 sentence the W179b needle quoted; the letter (l. 136: "the investment being a '
                                'fixed parameter of the recourse") is checked against the new wording'),
}
NEEDLES_V7 = {   # added for v7 (line, kind, first number) -> needles
    (160, 'appendix', 'E.6'): ['every evaluation repeated from an earlier record replayed that record bitwise'],
    (97, 'section', '3.3'): ['no separate generator cost curves are used'],
}


def pointer_rows(main_lines, adj, raw, letter_path, w179b_rows):
    cites = R.cite_order(main_lines)
    cl = typeset(main_lines)
    atoms = R.letter_pointers(letter_path)
    prior = {}
    for r_ in w179b_rows:
        prior.setdefault((r_['line'], r_['kind'], r_['numbers'][0], r_['text']), r_)
    rows = []
    for a in atoms:
        tgt = B.target_v6(a)
        row = {k: a[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range')}
        row['target'] = tgt
        pr = prior.get((a['line'], a['kind'], a['numbers'][0], a['text']))
        row['verdict_w179b_48099ab'] = pr['verdict'] if pr else None
        row['number_w179b_48099ab'] = pr.get('number_head') if pr else None
        nds = list(pr.get('content_needles', {}).keys()) if pr and pr.get('content_needles') else []
        rep = NEEDLES_REPLACED.get((a['line'], a['kind'], a['numbers'][0]))
        if rep and rep[0] in nds:
            nds = [rep[1] if nd == rep[0] else nd for nd in nds]
            row['needle_replaced'] = {'w179b_needle': rep[0], 'w180c_needle': rep[1], 'why': rep[2],
                                      'w179b_needle_present_at_head': rep[0] in '\n'.join(cl)}
        for nd in NEEDLES_V7.get((a['line'], a['kind'], a['numbers'][0]), []):
            if nd not in nds:
                nds.append(nd)
        if tgt is None:
            row['verdict'] = 'UNDECLARED'
            rows.append(row)
            continue
        if isinstance(tgt, str) and tgt.startswith('SUBMITTED:'):
            row['verdict'], row['note'] = 'SUBMITTED', tgt[10:]
            rows.append(row)
            continue
        tl = tgt if isinstance(tgt, list) else [tgt]
        n_adj = [B.resolve_v6(t, adj, cites) for t in tl]
        n_raw = [B.resolve_v6(t, raw, cites) for t in tl]
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
        if nds:
            if tl[0].startswith('cite:'):
                row['content_needles'] = {nd: (tl[0][5:] in cites) for nd in nds}
            else:
                reg = '\n'.join(B.region_v6(cl, t) for t in tl)
                row['content_needles'] = {nd: (nd in reg) for nd in nds}
        rows.append(row)
    return rows


def task_pointers(main_lines, adj, raw, w179b):
    rows = pointer_rows(main_lines, adj, raw, LETTER, w179b['b_pointers']['pointers'])
    used = {(r_['line'], r_['kind'], r_['numbers'][0]) for r_ in rows}
    for k in NEEDLES_REPLACED:
        if not any('needle_replaced' in r_ for r_ in rows if (r_['line'], r_['kind'], r_['numbers'][0]) == k):
            _fail(f'needle replacement {k} matched no carried needle')
    for k in NEEDLES_V7:
        if k not in used:
            _fail(f'needle declaration {k} matches no extracted pointer')
    und = [r_ for r_ in rows if r_['verdict'] == 'UNDECLARED']
    if und:
        _fail(f'{len(und)} pointer(s) without a declared target: {[(r_["line"], r_["text"]) for r_ in und]}')
    counts = Counter(r_['verdict'] for r_ in rows)
    by_kind = {}
    for r_ in rows:
        by_kind.setdefault(r_['kind'], Counter())[r_['verdict']] += 1
    absent = [{k: r_.get(k) for k in ('line', 'text', 'target', 'content_needles')}
              for r_ in rows if r_.get('content_needles') and not all(r_['content_needles'].values())]
    if absent:
        _fail(f'content needles absent: {absent}')
    changed = [{k: r_.get(k) for k in ('line', 'text', 'verdict_w179b_48099ab', 'verdict', 'number_w179b_48099ab', 'number_head')}
               for r_ in rows if r_['verdict_w179b_48099ab'] is None or r_['verdict_w179b_48099ab'] != r_['verdict']
               or (r_.get('number_head') and r_['number_w179b_48099ab'] and r_['number_head'] != r_['number_w179b_48099ab'])]
    return {'n_pointers': len(rows), 'counts': dict(counts), 'by_kind': {k: dict(v) for k, v in by_kind.items()},
            'pointers': rows, 'mismatch': [r_ for r_ in rows if r_['verdict'] == 'MISMATCH'],
            'ambiguous': [r_ for r_ in rows if r_['verdict'] == 'AMBIGUOUS'],
            'submitted': [{k: r_.get(k) for k in ('line', 'text', 'note')} for r_ in rows if r_['verdict'] == 'SUBMITTED'],
            'n_with_needles': sum(1 for r_ in rows if r_.get('content_needles')), 'content_absent': absent,
            'needles_replaced': [{k: r_.get(k) for k in ('line', 'text', 'needle_replaced')} for r_ in rows if 'needle_replaced' in r_],
            'differs_from_w179b': changed}


# ======================================================================================================================
#  5. the eight W179b rows
# ======================================================================================================================
def _ev(desc, ok, where=None):
    return {'check': desc, 'ok': bool(ok), 'where': where}


def task_rows(main_lines, letter_text, adj, w179b):
    cl = typeset(main_lines)
    T = '\n'.join(cl)
    vl = letter_text.split('\n')

    def at(frag, lines=cl):
        return [k for k, ln in enumerate(lines, 1) if frag in ln]

    def has(frag, lines=cl):
        return bool(at(frag, lines))
    prior = {r_['id']: r_ for r_ in w179b['b_main_claims']['rows'] + w179b['b_letter']['claims']}
    prior_g = {r_['row']: r_ for r_ in w179b['c_w178_rows']['rows']}
    lab = lambda k: adj['labels'].get(k, {}).get('number')  # noqa: E731
    rows = []
    # C-1 -------------------------------------------------------------------------------------------------------------
    ev = [_ev('conclusions: "stopped by the production exit, with the remaining movement of the plan without storage '
              'estimated from a continued run"', has('with the remaining movement of the plan without storage estimated from a continued run'),
              at('with the remaining movement of the plan without storage estimated from a continued run')),
          _ev('conclusions: "the search\'s own evaluations at the search resolution"', has("the search's own evaluations at the search resolution"),
              at("the search's own evaluations at the search resolution")),
          _ev('"with its remaining movement measured" gone', not has('with its remaining movement measured')),
          _ev('4.5: only the x = 0 evaluation was continued, its movement a fit', has('the evaluation of the plan without storage was continued beyond its stopping point')
              and has('A subsequent fit to the objective trajectory estimated a further cost decrease of 6.2~k\\euro{}'),
              at('the evaluation of the plan without storage was continued beyond its stopping point')),
          _ev('Table 13: D "damped-cosine fit", "post-hoc"', has('damped-cosine fit') and has('Post-hoc settled descent $D$'), at('damped-cosine fit')),
          _ev('4.1: the neighbours compared "beyond the search resolution"', has('none reduced the total discounted cost beyond the search resolution'),
              at('none reduced the total discounted cost beyond the search resolution')),
          _ev('4.7: the search evaluations "each stopped by the production exit ... compared at the search resolution"',
              has('each stopped by the production exit of the coordination procedure and compared at the search resolution'),
              at('each stopped by the production exit of the coordination procedure and compared at the search resolution'))]
    rows.append({'row': 'C-1', 'where': 'Conclusions (A.9)', 'w179b': 'still (narrowed), L: search evaluations not covered; '
                 '"movement measured" is a fit on x = 0 only', 'status': 'closed', 'rating_now': 'consistent',
                 'basis': 'Both W179b residues are answered by the text: the search evaluations are now named with their '
                          'resolution ("the search\'s own evaluations at the search resolution"; their stopping status, the '
                          'production exit, is stated in 4.7), and the multi-scenario clause now says the movement of the plan '
                          'without storage was "estimated from a continued run", which is what 4.5 and Table 13 report (one '
                          'continued run, x = 0; D = 6.2 k EUR by a post-hoc damped-cosine fit).', 'evidence': ev})
    # R14-2 -----------------------------------------------------------------------------------------------------------
    r14 = next(ln for ln in vl if ln.startswith(r'\rresponse{On (i): the introduction now states'))
    intro = '\n'.join(cl[379:383])
    ev = [_ev('letter: "fuel" absent from the "introduction now states" clause', 'fuel' not in r14[:r14.index('left to future work')].lower()),
          _ev('letter: new sentence present', 'Fuel costs are not represented separately: conventional generation in the transmission network is priced '
              'at the wholesale energy price of the scenario (Section~3.3), so fuel-cost uncertainty enters only through the market-price scenarios.' in r14),
          _ev('main.tex: "fuel" occurs 0 times (typeset and comments)', not any('fuel' in ln.lower() for ln in main_lines)),
          _ev('3.3 (Transmission Network): "Conventional generation in the TN is priced at the wholesale energy price of the scenario; '
              'no separate generator cost curves are used."', has('Conventional generation in the TN is priced at the wholesale energy price of the scenario; no separate generator cost curves are used.'),
              at('Conventional generation in the TN is priced at the wholesale energy price of the scenario')),
          _ev('3.3 is the subsection "Transmission Network" (static number)', any(o['kind'] == 'subsection' and o['title'] == 'Transmission Network'
                                                                                 and o['number'] == '3.3' for o in adj['objects'])),
          _ev('intro l. 381: investment-cost trajectories; operational scenarios of load, RES generation and market prices',
              'alternative trajectories of future ESS investment costs and operational scenarios of load, RES generation and market prices' in intro),
          _ev('intro l. 382: demand growth and RES capacity predefined; policy changes, infrastructure retirements, cost of capital '
              'outside, future work', 'Demand growth and RES capacity follow predefined trajectories, while policy changes, infrastructure '
              'retirements and the cost of capital as a stochastic quantity are outside the present formulation and left to future work.' in intro),
          _ev('"technology maturation" is the letter\'s gloss: the intro says "future ESS investment costs"; 3.2 ties the trajectories '
              'to "technological progress"', has('reflecting plausible cost ranges associated with technological progress'),
              at('reflecting plausible cost ranges associated with technological progress'))]
    rows.append({'row': 'R14-2', 'where': 'letter R1.4 l. 97', 'w179b': 'still (narrowed), L: the reply said the introduction '
                 'states fuel; "fuel" occurs 0 times in main.tex', 'status': 'closed', 'rating_now': 'consistent',
                 'basis': 'Fuel is no longer attributed to the introduction. Every source the clause attributes to the '
                          'introduction is in l. 381-382 as listed. The new sentence restates 3.3 l. 1184; "so fuel-cost '
                          'uncertainty enters only through the market-price scenarios" is the letter\'s inference from '
                          '"no separate generator cost curves are used", not a manuscript claim, and is consistent with it.',
                 'evidence': ev})
    # C47-1 -----------------------------------------------------------------------------------------------------------
    s47 = ("Every comparison of this section, and every single-scenario evaluation behind it, is listed in~\\ref{app:results}, "
           "together with the full certification statistics and the reproducibility record; the search's own evaluations, "
           "stopped at the production exit, are reported by their counts, and the multi-scenario pair in Table~\\ref{tab:res_multiscenario}.")
    ev = [_ev('4.7 sentence as edited', has(s47), at(s47)),
          _ev('4.7: the counts "41 new recourse evaluations in all (14, 20 and 7 against budgets of 20, 20 and 60)"',
              has('made 41 new recourse evaluations in all (14, 20 and 7 against budgets of 20, 20 and 60)'),
              at('made 41 new recourse evaluations in all')),
          _ev('Table 13 (tab:res_multiscenario) holds the 3x3 pair at cycle 72', lab('tab:res_multiscenario') == '13'
              and has('Value $Q(\\boldsymbol{0}) - Q(\\text{unit})$ at operational stopping (cycle 72)'), at('at operational stopping (cycle 72)')),
          _ev('Appendix E intro: E.21 = "every recourse evaluation those comparisons use"', has('every recourse evaluation those comparisons use'))]
    rows.append({'row': 'C47-1', 'where': '4.7 last sentence (A.5)', 'w179b': 'L: the 41 search evaluations named in 4.7 are not in Appendix E',
                 'status': 'closed', 'rating_now': 'consistent',
                 'basis': 'The sentence now scopes Appendix E to the single-scenario evaluations behind the comparisons and '
                          'sends the search evaluations to their counts (4.7) and the 3x3 pair to Table 13; each destination '
                          'holds what it is cited for.', 'evidence': ev})
    # C48-1 -----------------------------------------------------------------------------------------------------------
    b_before = list(B.CHECK_FAILURES)
    e24 = B.parse_longtable_rows(cl, 'tab:appE_gap', 11)
    if B.CHECK_FAILURES != b_before:
        _fail(f'W179b parse_longtable_rows reported: {B.CHECK_FAILURES[len(b_before):]}')
    e24_rows = {str(k): v['cells'] for k, v in e24.items()}
    no_share = sorted(k for k, c in e24_rows.items() if c[6:9] == ['---', '---', '---'])
    ev = [_ev('4.8: "split in the same proportions among the three interface nodes wherever the split was recorded"',
              has('split in the same proportions among the three interface nodes wherever the split was recorded (Table~\\ref{tab:appE_gap})'),
              at('wherever the split was recorded')),
          _ev('Table E.24 prints no split ("---") exactly for evaluations 40 and 41', no_share == ['40', '41'], no_share),
          _ev('E.5: "(for evaluations~40 and~41 the split and the residual were not recorded)"',
              has('(for evaluations~40 and~41 the split and the residual were not recorded)'), at('the split and the residual were not recorded')),
          _ev('the recorded splits (39, 50, 51, 45, 42) are 55.2-55.3 / 13.7 / 31.0-31.1', all(
              e24_rows.get(k, [None] * 9)[6] in ('55.2', '55.3') and e24_rows.get(k, [None] * 9)[7] == '13.7'
              and e24_rows.get(k, [None] * 9)[8] in ('31.0', '31.1') for k in ('39', '50', '51', '45', '42')))]
    rows.append({'row': 'C48-1', 'where': '4.8 degenerate duals (A.6)', 'w179b': 'L: E.24 prints the split for 5 of the 7 m = 2 evaluations',
                 'status': 'closed', 'rating_now': 'consistent',
                 'basis': '"wherever the split was recorded" covers the two "---" rows (40, 41), and E.5 now says so; the five '
                          'recorded splits agree to 0.1 point.', 'evidence': ev, 'e24_rows': e24_rows})
    # R22-b -----------------------------------------------------------------------------------------------------------
    r22 = next(ln for ln in vl if ln.startswith(r'\rresponse{Algorithm~1 and Section~2.1 now state the outer loop'))
    ev = [_ev('letter R2.2: "and the cycles of every evaluation behind a reported comparison in Appendix~E"',
              'and the cycles of every evaluation behind a reported comparison in Appendix~E.' in r22),
          _ev('letter R2.2 defines its classes: "ends at the production exit during the search, or, for the evaluations behind the '
              'reported comparisons, when the certification rule ... holds or its cycle cap is reached"',
              'the evaluation ends at the production exit during the search, or, for the evaluations behind the reported comparisons, '
              'when the certification rule' in r22),
          _ev('letter R2.2: the cycles clause carries no "single-scenario" qualifier (the FC bullet, 4.7 and 2.2.7 do; R2.2 uses '
              '"single-scenario" only for the instance, "at the single-scenario instance, 48 blocks")',
              'every single-scenario evaluation behind' not in r22 and 'at the single-scenario instance, 48 blocks' in r22),
          _ev('Appendix E intro sends the multi-scenario evaluations to Table 13 (not in E.21)',
              has('The multi-scenario evaluations, stopped by the production exit rather than certified, are reported in Table~\\ref{tab:res_multiscenario}.')),
          _ev('the 3x3 pair is behind a reported comparison: Table 13 "Value $- I$ ... determinate", "(cycle 72)"',
              has('at operational stopping (cycle 72)') and bool(re.search(r'Value \$- I\$\s*&\s*\$-\$81\.3~k\\euro\s*&\s*determinate', T))),
          _ev('E.6 gives the multi-scenario x = 0 evaluation\'s cycles ("replayed its 72 recorded cycles bitwise"), not the unit\'s',
              has('the multi-scenario evaluation of the plan without storage replayed its 72 recorded cycles bitwise')),
          _ev('E.3 / 4.8 Scope: six neighbours "evaluated by the search only" and compared "beyond the search resolution" (no cycles '
              'printed for them)', has('were evaluated by the search only, on an earlier state of the code and stopped by the production exit')
              and has('beyond the search resolution for the other six'))]
    rows.append({'row': 'R22-b', 'where': 'letter R2.2 l. 125', 'w179b': 'L: Appendix E does not hold the 41 search evaluations or the 3x3 pair (Table 13, cycle 72)',
                 'status': 'still (narrowed)', 'rating_now': 'L',
                 'basis': 'Narrowed: under R2.2\'s own two classes (search: production exit; behind the reported comparisons: '
                          'certification rule or cap) the search evaluations are excluded, and every certification-rule '
                          'evaluation has k*/end in E.21. What remains: the 3x3 pair is behind a reported comparison (Table 13, '
                          '-81.3 k EUR determinate) and fits neither class of R2.2; its stopping cycle (72) is in Table 13, which '
                          'Appendix E\'s introduction points to, and E.6 gives the x = 0 evaluation\'s 72 replayed cycles but not '
                          'the unit\'s. Secondary, arguable: the six search-only neighbours enter the 4.1 / 4.8 Scope / E.3 '
                          'comparison at the search resolution and have no cycles anywhere. R2.2 is the only one of the four '
                          'scoped sentences (2.2.7, 4.7, FC bullet, R2.2) without "single-scenario". Wording only; no number '
                          'or verdict changes.', 'evidence': ev})
    # FC-2c -----------------------------------------------------------------------------------------------------------
    fc = next(ln for ln in vl if 'Appendix~E was rebuilt.' in ln)
    ev = [_ev('FC bullet: "every single-scenario recourse evaluation those comparisons use, with its stopping status, band, gap and slack"',
              'every single-scenario recourse evaluation those comparisons use, with its stopping status, band, gap and slack' in fc),
          _ev('Appendix E intro: "every recourse evaluation those comparisons use, with its stopping status, band, gap and slack (Table E.21)"',
              has('every recourse evaluation those comparisons use, with its stopping status, band, gap and slack (Table~\\ref{tab:appE_evaluations})')),
          _ev('"those comparisons" = "every pairwise comparison" of the same bullet = Table E.20, whose Eval. column indexes E.21',
              'it now holds the record behind Section~4: every pairwise comparison' in fc and has('Eval.: the two evaluations of Table~\\ref{tab:appE_evaluations}')),
          _ev('the multi-scenario evaluations are outside E.21 (Appendix E intro, Table 13)',
              has('The multi-scenario evaluations, stopped by the production exit rather than certified, are reported in Table~\\ref{tab:res_multiscenario}.'))]
    rows.append({'row': 'FC-2c', 'where': 'letter Further changes, Appendix E bullet l. 285', 'w179b': 'L: the bullet dropped the "those comparisons use" qualifier',
                 'status': 'closed', 'rating_now': 'consistent',
                 'basis': 'The bullet now carries the qualifier of the Appendix E introduction, plus "single-scenario"; E.21 '
                          'is that set (it also lists the superseded evaluation 16, which "every ... those comparisons use" does '
                          'not exclude being exceeded).', 'evidence': ev})
    # G12 follow-on ---------------------------------------------------------------------------------------------------
    ev = [_ev('2.2.6: un-hatted additions "held fixed in the operational subproblem, where they enter as the parameters Ŝ, Ê of '
              'Subsection 2.3.2"', has('are held fixed in the operational subproblem, where they enter as the parameters $\\hat{S}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$'),
              at('are held fixed in the operational subproblem')),
          _ev('"are fixed parameters of the operational subproblem" gone from 2.2.6', not has('are fixed parameters of the operational subproblem')),
          _ev('nomenclature: Ŝ/Ê "The same additions as fixed parameters of the operational subproblem"',
              has('The same additions as fixed parameters of the operational subproblem'), at('The same additions as fixed parameters of the operational subproblem')),
          _ev('nomenclature: S^Inv/E^Inv "Converter rating and energy capacity installed at $e$ in $y^{\\text{Inv}}$"',
              has('Converter rating and energy capacity installed at $e$ in $y^{\\text{Inv}}$')),
          _ev('subsubsec:capacity resolves to 2.3.2 "Installed and Available Capacity"', lab('subsubsec:capacity') == '2.3.2'),
          _ev('2.3.2 uses the hats: "the fixed additions $\\hat{S}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$ and $\\hat{E}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$ of the candidate plan"',
              has('whose ratings are the fixed additions $\\hat{S}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$ and $\\hat{E}^{\\text{Inv}}_{e,y^{\\text{Inv}}}$ of the candidate plan'))]
    rows.append({'row': 'G12 follow-on', 'where': '2.2.6 l. 697 (A.1)', 'w179b': 'new L: 2.2.6 called the un-hatted S^Inv, E^Inv "fixed parameters of the operational subproblem"',
                 'status': 'closed', 'rating_now': 'consistent',
                 'basis': '2.2.6 now uses the nomenclature\'s split: the un-hatted additions are the plan, entering the '
                          'subproblem as the hatted parameters of 2.3.2 (label added, resolves to 2.3.2).', 'evidence': ev})
    # A.3 prefix ------------------------------------------------------------------------------------------------------
    ev = [_ev('A.3: "their tolerances (\\ref{app:admm_updated_stopping})"', has('their tolerances (\\ref{app:admm_updated_stopping})'),
              at('their tolerances (\\ref{app:admm_updated_stopping})')),
          _ev('no "Subsection~\\ref{app:" anywhere', not re.search(r'Subsections?(~|\s+)\\ref\{app:', T)),
          _ev('no label-word prefix before \\ref{app: (W179b PREFIX regex)', not any(B.PREFIX.search(ln) for ln in cl))]
    rows.append({'row': 'A.3 prefix', 'where': 'Appendix A.3 l. 1900 (A.8)', 'w179b': 'L: "Subsection~\\ref{app:admm_updated_stopping}" prints "Subsection Appendix A.4"',
                 'status': 'closed', 'rating_now': 'consistent', 'basis': 'The prefix is dropped; 0 label-word prefixes remain (part 6).',
                 'evidence': ev})
    for r_ in rows:
        key = {'C-1': 'C-1 (conclusions stopping status)', 'R14-2': 'R14-2 (fuel)',
               'G12 follow-on': 'Nomenclature hat-S / hat-E (W178 Decision 5)', 'A.3 prefix': 'A-prefix, A.3 l. 1899 (not in W178)'}.get(r_['row'])
        r_['w179b_row'] = ({'id': r_['row'], 'rating': prior[r_['row']]['rating']} if r_['row'] in prior else
                           {'id': key, 'status': prior_g[key]['status'], 'rating_now': prior_g[key]['rating_now']} if key in prior_g else None)
        if r_['w179b_row'] is None:
            _fail(f'row {r_["row"]}: no W179b row found')
        r_['evidence_ok'] = all(e['ok'] for e in r_['evidence'])
        if not r_['evidence_ok']:
            _fail(f'row {r_["row"]}: evidence check failed: {[e for e in r_["evidence"] if not e["ok"]]}')
    return {'rows': rows, 'status_counts': dict(Counter(r_['status'] for r_ in rows))}


# ======================================================================================================================
#  6. counts
# ======================================================================================================================
GENERAL_PREFIX = re.compile(r'([A-Za-z]+)(~|\s+)\\ref\{(app:[^}]*)\}')
LABEL_WORDS = {'appendix', 'appendices', 'section', 'sections', 'subsection', 'subsections', 'subsubsection', 'table', 'tables',
               'figure', 'figures', 'fig', 'eq', 'eqs', 'equation', 'equations', 'algorithm', 'part', 'chapter', 'paragraph'}


def task_counts(main_lines, letter_lines):
    cl = typeset(main_lines)

    def gen(lines, tilde_only):
        out = []
        for i, ln in enumerate(lines, 1):
            for m in GENERAL_PREFIX.finditer(ln):
                if tilde_only and m.group(2) != '~':
                    continue
                out.append({'line': i, 'word': m.group(1), 'sep': m.group(2), 'label': m.group(3)})
        return out
    g_raw_t, g_ts_t, g_ts_any = gen(main_lines, True), gen(cl, True), gen(cl, False)
    label_word = [h for h in g_ts_any if h['word'].lower() in LABEL_WORDS]
    out = {'confirm': {'main_tex': [{'line': i, 'text': ln.strip()[:160]} for i, ln in enumerate(main_lines, 1) if 'CONFIRM' in ln],
                       'letter': [{'line': i, 'text': ln.strip()[:160]} for i, ln in enumerate(letter_lines, 1) if 'CONFIRM' in ln]},
           'word_tilde_ref_app_general': {'pattern': r'[A-Za-z]+~\\ref\{app:', 'raw_lines': g_raw_t, 'typeset_lines': g_ts_t,
                                          'by_word_typeset': dict(Counter(h['word'] for h in g_ts_t))},
           'word_space_or_tilde_ref_app_typeset_by_word': dict(Counter(h['word'] for h in g_ts_any)),
           'label_word_prefixes': label_word,
           'appendix_word_before_any_ref': [{'line': i, 'text': m.group(0)} for i, ln in enumerate(cl, 1)
                                            for m in re.finditer(r'(Appendix|Appendices)(~|\s+)\\ref\{', ln)],
           'author_comments': {'main_tex_percent_AUTHOR': [{'line': i, 'text': ln.strip()[:160]} for i, ln in enumerate(main_lines, 1)
                                                           if re.search(r'%\s*\[AUTHOR', ln)],
                               'main_tex_any_AUTHOR_bracket': [i for i, ln in enumerate(main_lines, 1) if '[AUTHOR' in ln],
                               'letter_any_AUTHOR_bracket': [i for i, ln in enumerate(letter_lines, 1) if '[AUTHOR' in ln],
                               'expected_by_STEP6_and_Addendum_79': 1}}
    out['n_confirm_main'] = len(out['confirm']['main_tex'])
    out['n_confirm_letter'] = len(out['confirm']['letter'])
    out['n_general_pattern_hits_typeset'] = len(g_ts_t)
    out['n_label_word_prefixes'] = len(label_word)
    out['n_author_comments_main'] = len(out['author_comments']['main_tex_percent_AUTHOR'])
    # history of the [AUTHOR] marker in the clone (read-only log)
    out['author_marker_last_change_in_clone'] = _git(['log', '-1', '--format=%h %cI', '-S[AUTHOR', '--', 'main.tex'], cwd=CLONE).strip()
    out['author_marker_at_48099ab'] = sum(1 for ln in show(PREV).split('\n') if '[AUTHOR' in ln)
    return out


# ======================================================================================================================
#  7. negative controls
# ======================================================================================================================
def task_negative_controls(prev_lines, head_lines, v6_text, v7_text, adj_unused, w179b):
    res = []
    # NC1: an altered number inside an edit
    m1 = [ln.replace(r'$1.8 \times 10^{-6}$', r'$1.9 \times 10^{-6}$') for ln in head_lines]
    assert m1 != head_lines
    o1 = main_edit_check(prev_lines, m1)
    a3 = next(r_ for r_ in o1['rows'] if r_['id'] == 'A.3')
    res.append({'id': 'NC1 number 1.8 -> 1.9 in A.3', 'refused': (not a3['verbatim_at_head']) and not o1['reconstruction_outside_appendix_E_equal'] and not o1['ok']})
    # NC2: an altered word outside every edit (Introduction)
    k = next(i for i, ln in enumerate(head_lines) if 'formulates shared ESS planning as a two-stage stochastic program' in ln)
    m2 = list(head_lines)
    m2[k] = m2[k].replace('two-stage stochastic program', 'two-stage stochastic programme')
    o2 = main_edit_check(prev_lines, m2)
    res.append({'id': 'NC2 word altered outside the edits (l. %d)' % (k + 1),
                'refused': all(r_['verbatim_at_head'] for r_ in o2['rows']) and not o2['reconstruction_outside_appendix_E_equal']
                and o2['reconstruction_first_difference'] is not None and o2['reconstruction_first_difference']['line'] == k + 1})
    # NC3: a subsection inserted before 3.3 -> the letter's "Section 3.3" pointers must be MISMATCH
    j = next(i for i, ln in enumerate(head_lines) if ln.strip() == r'\subsection{Transmission Network}')
    m3 = head_lines[:j] + [r'\subsection{Inserted by the W180c negative control}', ''] + head_lines[j:]
    raw3, adj3 = C.numbering_adjusted(m3)
    rows3 = pointer_rows(m3, adj3, raw3, LETTER, w179b['b_pointers']['pointers'])
    p33 = [r_ for r_ in rows3 if r_['kind'] == 'section' and r_['numbers'] == ['3.3']]
    res.append({'id': 'NC3 subsection inserted before 3.3', 'n_mismatch': sum(1 for r_ in rows3 if r_['verdict'] == 'MISMATCH'),
                'section_3_3_verdicts': [(r_['line'], r_['verdict'], r_.get('number_head')) for r_ in p33],
                'refused': bool(p33) and all(r_['verdict'] == 'MISMATCH' for r_ in p33)})
    # NC4: the needle sentence of 3.3 removed -> needle absent on l. 97
    m4 = [ln.replace('no separate generator cost curves are used', 'generator costs follow') for ln in head_lines]
    raw4, adj4 = C.numbering_adjusted(m4)
    rows4 = pointer_rows(m4, adj4, raw4, LETTER, w179b['b_pointers']['pointers'])
    r97 = [r_ for r_ in rows4 if r_['line'] == 97 and r_['numbers'] == ['3.3']]
    res.append({'id': 'NC4 3.3 needle sentence removed', 'refused': bool(r97) and not all(r97[0].get('content_needles', {True: True}).values())})
    # NC5: an undeclared letter pointer number ("Appendix~E.6" -> "Appendix~E.9") must not MATCH
    with tempfile.TemporaryDirectory(prefix='w180c_nc_') as td:
        p5 = os.path.join(td, 'letter_nc5.tex')
        with open(p5, 'w', encoding='utf-8') as fh:
            fh.write(v7_text.replace('Appendix~E.6)', 'Appendix~E.9)'))
        raw5, adj5 = C.numbering_adjusted(head_lines)
        rows5 = pointer_rows(head_lines, adj5, raw5, p5, w179b['b_pointers']['pointers'])
    r160 = [r_ for r_ in rows5 if r_['line'] == 160 and r_['kind'] == 'appendix']
    res.append({'id': 'NC5 letter Appendix E.6 -> E.9', 'verdicts': [(r_['text'], r_['verdict']) for r_ in r160],
                'refused': any(r_['numbers'] == ['E.9'] and r_['verdict'] != 'MATCH' for r_ in r160)})
    # NC6: an altered word in a letter edit
    o6 = letter_edit_check(v6_text, v7_text.replace('so fuel-cost uncertainty enters only', 'so fuel-cost uncertainty enters mainly'))
    res.append({'id': 'NC6 letter R1.4 sentence altered', 'refused': not o6['ok']})
    out = {'controls': res, 'all_refused': all(r_['refused'] for r_ in res)}
    if not out['all_refused']:
        _fail(f'negative control not refused: {[r_ for r_ in res if not r_["refused"]]}')
    return out


# ======================================================================================================================
#  guards, output
# ======================================================================================================================
def verify_guards():
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    g = {'own': GUARD.verify(0), 'w179b': B.GUARD.verify(0), 'w178c': C.GUARD.verify(0), 'w177b3r': R.GUARD.verify(0),
         'w177b3': P3.GUARD.verify(0), 'w167': W167.GUARD.verify(0), 'phase_b_record': PB.PARENT_GUARD.verify(0)}
    pk = {'own': dict(PICKLE_COUNTS), 'w179b': dict(B.PICKLE_COUNTS), 'w178c': dict(C.PICKLE_COUNTS),
          'w177b3r': dict(R.PICKLE_COUNTS), 'w177b3': dict(P3.PICKLE_COUNTS), 'w167': dict(W167.PICKLE_COUNTS)}
    out = {'verify_0_failures': g,
           'counts': {'own': GUARD.counts, 'w179b': B.GUARD.counts, 'w178c': C.GUARD.counts, 'w177b3r': R.GUARD.counts,
                      'w177b3': P3.GUARD.counts, 'w167': W167.GUARD.counts, 'phase_b_record': PB.PARENT_GUARD.counts},
           'pickle_counts': pk, 'production_modules_imported': forbidden}
    bad = any(g.values()) or any(v for d in pk.values() for v in d.values()) or forbidden
    if bad:
        _fail(f'guards: {out}')
    out['ok'] = not bad
    return out


def write_md(out, path):
    I = out['identity']
    L = [f'# W180c text + letter checks (generated by {SCRIPT_NAME})', '',
         f'Clone HEAD {I["clone_head"]} ({I["clone_head_full"]}); main.tex {I["main_sha256"][:8]} ({I["main_lines"]} lines); '
         f'letter {I["letter_sha256"][:8]} ({I["letter_lines"]} lines). Check failures: {len(out["check_failures"])}.', '',
         '## Identity checks', '', '| check | ok |', '|---|---|']
    L += [f'| {k} | {v} |' for k, v in I['checks'].items()]
    M = out['main_edits']
    L += ['', '## Main-text edits A.1-A.9', '', f'Appendix E block: 48099ab {M["appendix_E_block_48099ab"]}, HEAD {M["appendix_E_block_head"]}; '
          f'reconstruction outside Appendix E equal: {M["reconstruction_outside_appendix_E_equal"]}; hunks main text '
          f'{M["n_hunks_main_text"]}, inside Appendix E {M["n_hunks_appendix_E"]}; every hunk classified {M["every_hunk_classified"]}', '',
          '| edit | where | STEP6 l. (48099ab) | l. at HEAD | verbatim | STEP6 line agrees |', '|---|---|---|---|---|---|']
    for r_ in M['rows']:
        L.append(f'| {r_["id"]} | {r_["where"]} | {r_["step6_line_48099ab"]} | {r_["line_head"]} | {r_["verbatim_at_head"]} | {r_["step6_line_matches_48099ab"]} |')
    L += ['', '| hunk old | hunk new | A edit | inside Appendix E |', '|---|---|---|---|']
    for h in M['git_hunks']:
        L.append(f'| {h["old"]} | {h["new"]} | {h["a_edits"]} | {h["inside_appendix_E"]} |')
    LE = out['letter_edits']
    L += ['', '## Letter v7 edits', '', f'reconstruction equal: {LE["reconstruction_equal"]}; changed lines {LE["changed_lines_v7"]}; '
          f'pointer change only E -> E.6: {LE["pointer_change_is_only_E_to_E6"]}', '', '| edit | lines v7 | verbatim |', '|---|---|---|']
    for r_ in LE['rows']:
        L.append(f'| {r_["id"]} | {r_["lines_v7"]} | {r_["verbatim_at_head"]} |')
    N = out['numbering']
    L += ['', '## Numbering', '', f'labels HEAD {N["n_labels_head"]}, 48099ab {N["n_labels_48099ab"]}, compared {N["n_compared"]} '
          f'({N["n_compared_outside_appendix_E"]} outside Appendix E); changed {N["labels_changed"]}; removed {N["labels_removed"]}; '
          f'new {N["labels_new"]}; undefined refs {N["refs_static_head"]["undefined_refs"]}; duplicates {N["refs_static_head"]["duplicate_labels"]}']
    P = out['pointers']
    L += ['', '## Pointers', '', f'{P["n_pointers"]} pointers: {P["counts"]}; with needles {P["n_with_needles"]}; absent {len(P["content_absent"])}', '',
          '| l. | text | kind | target | HEAD | W179b (48099ab) | verdict | needles |', '|---|---|---|---|---|---|---|---|']
    for r_ in P['pointers']:
        tgt = r_['target'] if isinstance(r_['target'], str) else ', '.join(r_['target'] or [])
        nd = '' if not r_.get('content_needles') else ('all' if all(r_['content_needles'].values()) else
                                                        'ABSENT: ' + '; '.join(k for k, v in r_['content_needles'].items() if not v))
        L.append(f'| {r_["line"]} | {r_["text"].replace("|", "/")} | {r_["kind"]} | {tgt[:55]} | {r_.get("number_head", "")} | '
                 f'{r_.get("number_w179b_48099ab") or ""} {r_.get("verdict_w179b_48099ab") or ""} | {r_["verdict"]} | {nd} |')
    L += ['', '## Rows', '', '| row | W179b | status | now | basis |', '|---|---|---|---|---|']
    for r_ in out['rows']['rows']:
        L.append(f'| {r_["row"]} | {r_["w179b"]} | {r_["status"]} | {r_["rating_now"]} | {r_["basis"]} |')
    K = out['counts']
    L += ['', '## Counts', '', f'CONFIRM main {K["n_confirm_main"]}, letter {K["n_confirm_letter"]}; general "<word>~\\ref{{app:" hits '
          f'(typeset) {K["n_general_pattern_hits_typeset"]} {K["word_tilde_ref_app_general"]["by_word_typeset"]}; label-word prefixes '
          f'{K["n_label_word_prefixes"]}; "% [AUTHOR]" comments in main.tex {K["n_author_comments_main"]} (expected 1)']
    L += ['', '## Negative controls', '']
    L += [f'- {c["id"]}: refused {c["refused"]}' for c in out['negative_controls']['controls']]
    L += ['', '## Predictions', ''] + [f'- {k}: {v}' for k, v in out['predictions'].items()]
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
    if os.path.abspath(OUT_DIR) == os.path.abspath(B.OUT_DIR):
        print('REFUSED: never write into the W179b output directory', flush=True)
        sys.exit(1)
    refuse_concurrent()
    out_json, out_md, out_man = (os.path.join(OUT_DIR, n) for n in (OUT_NAME, MD_NAME, MANIFEST_NAME))
    t0 = datetime.now(timezone.utc)
    _log('W180c checks: start')
    out = {'schema': 'p515_s53_w180c_text_letter',
           'task': 'W180c (Addendum 79, W180 item (c)): A.1-A.9 and letter v7 verbatim at e5322f8; letter pointer check; '
                   'eight W179b rows re-scored; CONFIRM / prefix / AUTHOR counts',
           'objective_convention': 'Q = gross_operational_cost, settlement excluded; net = Q - terminal salvage credit; F = I + Q '
                                   '(no objective value is computed here; quoted values are the manuscript\'s)',
           'rating_scale_W174b': {'H': 'a referee re-deriving from the text gets a different model, number or verdict',
                                  'M': 'imprecise, but not a different result', 'L': 'wording'},
           'started_utc': t0.isoformat(), 'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'script': {'path': SCRIPT_NAME, 'sha256': _sha(__file__)},
           'imported': {m: _sha(m) for m in (W179B_SCRIPT, 'p515_s53_w178c_letter_checks.py', 'p515_s53_w177b3r_checks.py',
                                             'p515_s53_w177b_part3_checks.py', 'p515_s53_w167_nomenclature_years.py',
                                             'p515_s47_phase_b_record.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}}
    out['capture_checklist'] = capture_checklist()
    _log('capture-path checklist: all present')
    out['identity'] = identity()
    _log(f"identity: head {out['identity']['clone_head']}, checks {sum(out['identity']['checks'].values())}/{len(out['identity']['checks'])}")
    head_text, prev_text = _read(MAIN_TEX), show(PREV)
    head_lines, prev_lines = head_text.split('\n'), prev_text.split('\n')
    v7_text, v6_text = _read(LETTER), show(PREV, LETTER_NAME)
    w179b = _json(W179B_JSON)
    out['main_edits'] = task_main_edits(prev_lines, head_lines)
    _log(f"(2) main edits verbatim {sum(r_['verbatim_at_head'] for r_ in out['main_edits']['rows'])}/{len(out['main_edits']['rows'])}; "
         f"reconstruction {out['main_edits']['reconstruction_outside_appendix_E_equal']}; hunks main {out['main_edits']['n_hunks_main_text']} "
         f"App E {out['main_edits']['n_hunks_appendix_E']}")
    out['letter_edits'] = task_letter_edits(v6_text, v7_text)
    _log(f"(3) letter edits verbatim {sum(r_['verbatim_at_head'] for r_ in out['letter_edits']['rows'])}/{len(out['letter_edits']['rows'])}; "
         f"reconstruction {out['letter_edits']['reconstruction_equal']}; changed lines {out['letter_edits']['changed_lines_v7']}")
    raw, adj, out['numbering'] = task_numbering(head_lines, prev_lines)
    _log(f"(4) numbering: compared {out['numbering']['n_compared']}, changed {len(out['numbering']['labels_changed'])}, "
         f"removed {len(out['numbering']['labels_removed'])}, new {out['numbering']['labels_new']}")
    out['pointers'] = task_pointers(head_lines, adj, raw, w179b)
    _log(f"(4) pointers: {out['pointers']['n_pointers']} {out['pointers']['counts']}; needles {out['pointers']['n_with_needles']}, "
         f"absent {len(out['pointers']['content_absent'])}; differ from W179b {len(out['pointers']['differs_from_w179b'])}")
    out['rows'] = task_rows(head_lines, v7_text, adj, w179b)
    _log(f"(5) rows: {[(r_['row'], r_['status']) for r_ in out['rows']['rows']]}")
    out['counts'] = task_counts(head_lines, v7_text.split('\n'))
    _log(f"(6) CONFIRM main {out['counts']['n_confirm_main']} letter {out['counts']['n_confirm_letter']}; general pattern "
         f"{out['counts']['word_tilde_ref_app_general']['by_word_typeset']}; label-word prefixes {out['counts']['n_label_word_prefixes']}; "
         f"[AUTHOR] comments {out['counts']['n_author_comments_main']}")
    out['negative_controls'] = task_negative_controls(prev_lines, head_lines, v6_text, v7_text, adj, w179b)
    _log(f"(7) negative controls: {[(c['id'], c['refused']) for c in out['negative_controls']['controls']]}")
    out['frozen_json_unchanged_at_end'] = _sha(FZ) == FZ_SHA
    if not out['frozen_json_unchanged_at_end']:
        _fail('frozen JSON changed during the run')
    out['guards'] = verify_guards()
    pc = out['pointers']['counts']
    not_closed = [r_['row'] for r_ in out['rows']['rows'] if r_['status'] != 'closed']
    out['predictions'] = {
        '(c) 0 pointer MISMATCH': {'outcome': pc, 'held': pc.get('MISMATCH', 0) == 0},
        '(c) all eight re-scored rows close': {'not_closed': not_closed, 'held': not not_closed},
        '(c) 0 CONFIRM': {'main': out['counts']['n_confirm_main'], 'letter': out['counts']['n_confirm_letter'],
                          'held': out['counts']['n_confirm_main'] == 0 and out['counts']['n_confirm_letter'] == 0},
        '(c) 0 <word>~\\ref{app: prefixes': {'label_word_prefixes': out['counts']['n_label_word_prefixes'],
                                              'general_pattern_hits_typeset': out['counts']['n_general_pattern_hits_typeset'],
                                              'held': out['counts']['n_label_word_prefixes'] == 0},
        'task: one % [AUTHOR] comment (expected)': {'found': out['counts']['n_author_comments_main'],
                                                    'held': out['counts']['n_author_comments_main'] == 1}}
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'w', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, out_md)
    inputs = sorted({MAIN_TEX, LETTER, DTX, MAIN_R6, MAIN_PREV_COPY, LETTER_V6, LETTER_V7, DIFF_MAIN, DIFF_LETTER, STEP6, BRIEF,
                     W179B_JSON, W179B_SCRIPT, FZ, 'p515_s53_w178c_letter_checks.py', 'p515_s53_w177b3r_checks.py',
                     'p515_s53_w177b_part3_checks.py', 'p515_s53_w167_nomenclature_years.py', 'p515_s47_phase_b_record.py',
                     'gate_result_io.py', 'p513_solve_profile_guard.py', SCRIPT_NAME})
    manifest = {'outputs': {OUT_NAME: _sha(out_json), MD_NAME: _sha(out_md)}, 'out_dir': OUT_DIR,
                'git_show_blobs_sha256': {f'{c}:{p}': sha_text(show(c, p)) for c in (PREV, 'HEAD') for p in ('main.tex', LETTER_NAME)},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(out_man, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'written {out_json}, {out_md}, {out_man}; check failures: {len(CHECK_FAILURES)}; guards ok: {out["guards"]["ok"]}; '
         f'pickle {PICKLE_COUNTS}; predictions {[(k, v["held"]) for k, v in out["predictions"].items()]}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
