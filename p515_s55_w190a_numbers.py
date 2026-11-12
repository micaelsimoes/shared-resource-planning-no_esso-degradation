"""P5.15 Addendum 91 (Planner order, W190a line; ruling 4) -- Planner task W190a PART 1: THE ROUND-9c AUDIT AT THE OVERLEAF
HEAD 3c379fa -- GENERATOR GATE, VERIFIER, NUMBERS AND THE RULING-4 WORDING ITEMS. ZERO SOLVES, NO MODEL, NO PICKLE.
Read-only on the Overleaf clone (git rev-parse / show / diff / status only; no fetch, pull or checkout), on
manuscript_review/, on every committed artifact and on every earlier script. Part 2 (identity, pointers, labels, blue,
structure) is p515_s55_w190a_structure.py.

REUSE, NOT REIMPLEMENTATION. The W189 part-1 harness p515_s55_w189_p1_numbers.py (P1) is IMPORTED, NOT EDITED and NOT RUN
(its main() is never called; nothing is written into its output directory). Its task functions are called with its
module constants re-bound to this round's revisions:
  * P1.task_appendix (the full-mode generator run, verify_appE.py with its tempfile.mkdtemp() redirected into this
    stage's output directory, the W180a / W181a entry check of the printed tables against the frozen JSON) -- called with
    P1.PREV_FULL = a87e195 (as in W189: its integrity checks locate the full generator output once in a87e195) and
    P1.OUT_NAMES re-bound to the W190a file names;
  * P1.task_numbers (the W182b method: tokenizer, spec kinds, evaluator, perturbation) -- called with P1.PREV_FULL =
    46c9d77 and P1.HEAD_FULL = 3c379fa, so the '+' lines are those of git diff -U0 46c9d77 3c379fa, and with
    P1.declarations_w189p1 re-bound to declarations_w190a below, which CARRIES the W189p1 declaration of every '+' line
    whose tokens are unchanged (the 9b -> 9c diff is line-preserving: every hunk replaces k lines by k lines, checked) and
    adds new specs only for the new tokens (Figure 6 paragraph: 90 / 10 order, six, eight, 14, 15; the cover-letter
    paragraph; the manuscript number);
  * P1.controls_appendix and the per-token perturbation of task_numbers (negative controls).

ITEMS (Addendum 91 ruling 4 and the W190a line).
  (b) generator gate: manuscript_review/gen_appE.py --reduced run IN-PROCESS (runpy, as W181a run_generator, with the flag)
      into this stage's output directory; its output located BITWISE in the round-9c main.tex (expected: exactly once,
      215 / 215 lines, per the expert); the full mode bitwise equal to manuscript_review/appendixE_evaluation_record.tex
      (P1.task_appendix); the verifier's own full and reduced regenerations compared by sha256.
  (c) verify_appE.py: expected 40 PASS (33 + 7 reduced-mode checks), 0 FAIL.
  (d) numbers on every '+' line of 46c9d77 -> 3c379fa (main.tex, letter, cover letter): expected 0 MISMATCH; every token
      listed with its source record and marked new / carried against the 9b '-' lines of its hunk; the five ruling-4
      wording items (i)-(iv) checked against W188 (a) and the W188 (d) CSVs ((v), blue, is part 2's); the cover letter v4:
      the filled paragraph against the A89 claims JSON, against P5_15_W184_REPORT.md / P5_15_W185_REPORT.md as printed
      and against the cell records (campaign_results.json Q_end); exactly one bracket left; EST-D-26-06323; the delay
      sentence.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import; P1's guard and every guard of its
imported chain verified at exactly 0 (P1.verify_guards) together with this script's; pickle.load / pickle.loads /
pickle.Unpickler blocked for the whole run by P1's blockers (counters verified 0); refuses to run concurrently with another
copy of itself; refuses unless the output directory exists and holds nothing but launch.log; capture-path checklist
asserted before any task (exit 4). JSON written only through gate_result_io. verify_appE.py itself launches gen_appE.py
twice as a subprocess (its reduced-mode checks); gen_appE.py imports csv / sys / os / re only (checked by ast below).

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p w190a_audit && mkdir w190a_audit/numbers && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_numbers.py \\
      > w190a_audit/numbers/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190a_numbers.py --post-run
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W190a part 1 generator gate and numbers at Overleaf 3c379fa '
                                               '(never solves)').install()

import ast  # noqa: E402
import collections  # noqa: E402
import contextlib  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s55_w189_p1_numbers as P1  # noqa: E402  (W189 part 1; imports W183h and its chain; NOT edited, main() not run)
import gate_result_io as GRIO  # noqa: E402

A, NB, H, W177A, B1R = P1.A, P1.NB, P1.H, P1.W177A, P1.B1R
os.chdir(REPO)
ORIG_DECL = P1.declarations_w189p1          # the W189p1 declarations, kept before P1's name is re-bound

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s55_w190a_numbers'
OUT_DIR = os.path.join('w190a_audit', 'numbers')
OUT_NAMES = {'json': 'w190a_numbers.json', 'md': 'w190a_numbers.md', 'man': 'manifest_sha256.json', 'log': 'launch.log',
             'post': 'manifest_post_run_sha256.json', 'gen': 'appE_generated_full_w190a.tex',
             'red': 'appE_generated_reduced_w190a.tex', 'verify': 'verify_appE_output_w190a.txt',
             'verify_tmp': 'verify_appE_tmp'}
CLONE = P1.CLONE
HEAD_FULL = '3c379fa44f3755dbdc6fd540df566b55527d921c'
MID_FULL = '52a607d6ea4225154fa78047f5438fada03bb98a'       # between 46c9d77 and HEAD (same tree as 46c9d77; checked)
BASE_FULL = '46c9d779743b40a42da73c0050d8923e6af6fb55'      # round 9b
A87_FULL = 'a87e1950fbdbb3667f37e64a43c298ba675e6cff'       # round 8 (W189's PREV)
MAIN, LETTER, COVER, HL = P1.MAIN, P1.LETTER, P1.COVER, P1.HL
FILES = (MAIN, LETTER, COVER, HL)
MR = 'manuscript_review'
REVIEW_COPY = {MAIN: os.path.join(MR, 'main_46c9d77_round9c.tex'), LETTER: os.path.join(MR, 'response_to_reviewers_v13.tex'),
               COVER: os.path.join(MR, 'cover_letter_v4.tex'), HL: os.path.join(MR, 'highlights_v3.tex')}
MD5_DECLARED = {MAIN: 'cd1433d7', LETTER: '74e88c78', COVER: '63b7ddee', HL: '4cedd45c'}
GEN_REL, VERIFY_REL, APPX_REL = P1.GEN_REL, P1.VERIFY_REL, P1.APPX_REL
EXPERT_REDUCED_LINES = 215                      # Addendum 91 ruling 2: "gate run by the expert: 215/215 lines"
EXPECTED_VERIFY_PASS = 40                       # 33 + 7 reduced-mode checks
W184_REPORT, W185_REPORT = 'P5_15_W184_REPORT.md', 'P5_15_W185_REPORT.md'
E3_RESULTS = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e3_cnrf', 'campaign_s55_step7_e3_cnrf_x0_1x',
                          'campaign_results.json')
E2_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e2_rho')
A91 = os.path.join(MR, 'ADDENDUM91_2026-11-12.md')
BRIEF = P1.BRIEF
W189P1_JSON = os.path.join(P1.S55, 'w189_p1_numbers', 'w189_p1_numbers.json')
IMPORTED_SCRIPTS = ('p515_s55_w189_p1_numbers.py',) + P1.IMPORTED_SCRIPTS
CHECK_FAILURES = []


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log(f'CHECK FAILURE: {msg}')


_sha, _shab, _rd, _git, ws, committed_clean = P1._sha, P1._shab, P1._rd, P1._git, P1.ws, P1.committed_clean


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


# ======================================================================================================================
#  state (provenance of this part; the full identity check is part 2's)
# ======================================================================================================================
def task_state():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    om = _git(['rev-parse', 'origin/main'], cwd=CLONE).strip()
    parent = _git(['rev-parse', 'HEAD^'], cwd=CLONE).strip()
    trees = {r: _git(['rev-parse', f'{r}^{{tree}}'], cwd=CLONE).strip() for r in (BASE_FULL, MID_FULL, HEAD_FULL)}
    tracked = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    files = {}
    for f in FILES:
        wb, hb, rc = _rd(os.path.join(CLONE, f)), showb(HEAD_FULL, f), _rd(REVIEW_COPY[f])
        md5 = hashlib.md5(wb).hexdigest()
        files[f] = {'sha256': _shab(wb), 'md5': md5, 'md5_declared_prefix': MD5_DECLARED[f],
                    'md5_matches_declared': md5.startswith(MD5_DECLARED[f]), 'working_equals_head_blob': wb == hb,
                    'review_copy': REVIEW_COPY[f], 'equals_review_copy': wb == rc}
    d0 = {f: _git(['diff', '-U0', BASE_FULL, HEAD_FULL, '--', f], cwd=CLONE) for f in (MAIN, LETTER, COVER)}
    line_preserving = {}
    for f, d in d0.items():
        hs = [(int(m.group(2) or 1), int(m.group(4) or 1)) for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',
                                                                               d, re.M)]
        line_preserving[f] = {'hunks_U0': len(hs), 'every_hunk_k_for_k': all(a == b for a, b in hs)}
    out = {'clone_head': head, 'origin_main': om, 'clone_head_parent': parent, 'trees': trees, 'tracked_modified': tracked,
           'files': files, 'diff_U0_line_preserving': line_preserving}
    out['holds'] = (head == HEAD_FULL and om == HEAD_FULL and parent == MID_FULL and trees[BASE_FULL] == trees[MID_FULL]
                    and tracked == '' and all(v['md5_matches_declared'] and v['working_equals_head_blob'] and v['equals_review_copy']
                                              for v in files.values()))
    out['line_preserving_main_letter'] = line_preserving[MAIN]['every_hunk_k_for_k'] and line_preserving[LETTER]['every_hunk_k_for_k']
    if not out['holds']:
        _fail(f'(a) state: {json.dumps({k: v for k, v in out.items() if k != "files"})[:600]}')
    if not out['line_preserving_main_letter']:
        _fail('(d) the 9b -> 9c diff of main.tex / the letter is not line-preserving: the carried W189p1 line keys do not apply')
    return out


# ======================================================================================================================
#  (b) generator gate and (c) verifier
# ======================================================================================================================
def gen_imports():
    tree = ast.parse(_rd(GEN_REL).decode('utf-8'))
    return sorted({(n.module if isinstance(n, ast.ImportFrom) else a.name).split('.')[0]
                   for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                   for a in (n.names if isinstance(n, ast.Import) else [n])})


def run_generator_reduced(out_path):
    """W181a run_generator (runpy, in-process, stdout captured), with the --reduced flag of round 9 (Addendum 91 ruling 2)."""
    if os.path.exists(out_path):
        raise RuntimeError(f'{out_path} exists: the generator would overwrite it')
    argv0 = list(sys.argv)
    buf = io.StringIO()
    sys.argv = [os.path.join(REPO, GEN_REL), out_path, '--reduced']
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(REPO, GEN_REL), run_name='__main__')
    finally:
        sys.argv = argv0
    gc.collect()
    with open(out_path, 'rb') as h:
        return h.read(), buf.getvalue()


def task_generator(out_dir, texts, apx, actx):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    red_path = os.path.join(od, OUT_NAMES['red'])
    red, red_stdout = run_generator_reduced(red_path)
    full = actx['gen_bytes']
    main_b = texts[MAIN].encode('utf-8')
    hl = texts[MAIN].split('\n')
    occ = A.locate_bytes(main_b, red)
    occ_9b = A.locate_bytes(showb(BASE_FULL, MAIN), red)
    occ_full = A.locate_bytes(main_b, full)
    rl = red.decode('utf-8').split('\n')
    if rl and rl[-1] == '':
        rl = rl[:-1]
    b0, b1 = A.appendix_block(hl)
    blk = hl[b0:b1]
    line_eq = None
    if len(occ) == 1:
        s = occ[0]['start_line'] - 1
        line_eq = sum(1 for k, x in enumerate(rl) if s + k < len(hl) and hl[s + k] == x)
        outside = [{'line': i + 1, 'text': ws(hl[i])[:160]} for i in range(b0, b1) if not (s <= i < s + len(rl))]
    else:
        outside = None
    # the verifier's own regenerations (redirected into verify_appE_tmp by P1.task_appendix)
    vt = os.path.join(od, OUT_NAMES['verify_tmp'])
    regen = {}
    for dp, _d, fs in os.walk(vt):
        for f in fs:
            p = os.path.join(dp, f)
            b = _rd(p)
            regen[os.path.relpath(p, od)] = {'sha256': _shab(b), 'equals_reduced_output': b == red, 'equals_full_output': b == full}
    out = {'generator': {'path': GEN_REL, 'sha256': _sha(GEN_REL), 'committed_clean': committed_clean(GEN_REL),
                         'last_commit': _git(['log', '-1', '--format=%h', '--', GEN_REL]).strip(), 'imports': gen_imports()},
           'reduced': {'output': os.path.join(out_dir, OUT_NAMES['red']), 'sha256': _shab(red), 'stdout': red_stdout.strip(),
                       'n_lines': len(rl),
                       'line_count_variants': {'all': len(rl), 'non_blank': sum(1 for x in rl if x.strip()),
                                               'comment': sum(1 for x in rl if x.strip().startswith('%')),
                                               'non_blank_non_comment': sum(1 for x in rl if x.strip() and
                                                                            not x.strip().startswith('%')),
                                               'without_trailing_blank': len(rl) - next(
                                                   (k for k, x in enumerate(reversed(rl)) if x.strip()), 0),
                                               'note': 'recorded because the expert reports 215 / 215 lines'},
                       'occurrences_in_9c_main': occ, 'n_occurrences_in_9c_main': len(occ),
                       'n_occurrences_in_9b_main': len(occ_9b), 'lines_equal_at_located_position': line_eq,
                       'expert_lines': EXPERT_REDUCED_LINES,
                       'appendix_block_9c_lines': [b0 + 1, b1], 'appendix_block_n_lines': b1 - b0,
                       'block_lines_outside_the_generated_text': outside},
           'full': {'output': os.path.join(out_dir, OUT_NAMES['gen']), 'sha256': _shab(full),
                    'equals_committed_appendixE_evaluation_record': apx['bitwise']['equals_committed_appendix'],
                    'committed_appendix_sha256': apx['bitwise']['committed_appendix_sha256'],
                    'n_occurrences_in_9c_main': len(occ_full), 'stdout': apx['bitwise']['generator']['stdout']},
           'verifier_regenerations': regen}
    out['gate_bitwise_reduced_in_main'] = (len(occ) == 1 and occ[0]['starts_at_line_start'] and occ[0]['ends_at_line_end']
                                           and line_eq == len(rl))
    out['matches_expert_215'] = out['gate_bitwise_reduced_in_main'] and len(rl) == EXPERT_REDUCED_LINES
    out['full_mode_bitwise_committed'] = apx['bitwise']['equals_committed_appendix']
    out['verifier_reduced_regeneration_equals'] = any(v['equals_reduced_output'] for v in regen.values())
    if not red_stdout.startswith('ok ') or 'reduced' not in red_stdout:
        _fail(f'(b) the reduced generator did not run cleanly: {red_stdout!r}')
    if not set(out['generator']['imports']) <= {'csv', 'sys', 'os', 're'}:
        _fail(f"(b) gen_appE.py imports {out['generator']['imports']} (expected csv / sys / os / re only)")
    # negative control: one byte changed in the reduced block is not located
    j = red.find(b'Certification Statistics')
    mut = red[:j] + (b'X' if red[j:j + 1] != b'X' else b'Y') + red[j + 1:]
    ctl = [{'id': 'b-1 the reduced block with one byte changed is not located in the 9c main.tex',
            'refused': len(occ) == 1 and len(A.locate_bytes(main_b, mut)) == 0}]
    return out, ctl, red


def verifier_summary(apx):
    v = dict(apx['verifier'])
    v.pop('note', None)
    v['note_w190a'] = ('verify_appE.py was extended in 441722c9 (Addendum 91 ruling 2): after its 33 full-mode checks it runs '
                       'gen_appE.py twice as a subprocess (full and --reduced, into its own tempfile.mkdtemp() directories, '
                       'redirected here) and checks the reduced output (7 checks). It does not read main.tex; the link from the '
                       'reduced output to the manuscript is item (b) of this stage. W189\'s static note on this field is '
                       'superseded and removed here.')
    v['expected_pass'] = EXPECTED_VERIFY_PASS
    lines = apx_verify_lines(v)
    v['pass_lines'] = [x for x in lines if x.startswith('PASS ')]
    v['reduced_mode_pass_lines'] = [x for x in lines if x.startswith('PASS reduced:')]
    v['outcome_40_pass_0_fail'] = v['n_pass'] == EXPECTED_VERIFY_PASS and v['n_fail'] == 0 and not v['exception']
    return v


_VERIFY_TEXT = {}


def apx_verify_lines(_v):
    return _VERIFY_TEXT.get('text', '').split('\n')


# ======================================================================================================================
#  (d) numbers: declarations W190a (carried W189p1 + new), new / carried marking
# ======================================================================================================================
def _soc_sum():
    return P1.EX['SOCSUM']


def n_lower_contact_days(X):
    return sum(1 for r in _soc_sum() if int(r['n_contact_lower_tso']) > 0)


def n_upper_contact_days(X):
    return sum(1 for r in _soc_sum() if int(r['n_contact_upper_tso']) > 0)


def n_soc_days(X):
    blocks = {(r['year'], r['day']) for r in P1.EX['SOC']}
    return len(_soc_sum()) if blocks == {(r['year'], r['day']) for r in _soc_sum()} else None


def winter_min_pct(y):
    def f(X):
        r = next(r for r in _soc_sum() if r['year'] == y and r['day'] == 'Winter')
        return 100.0 * float(r['soc_min_tso_mwh']) / float(P1.EX['LIM'][y]['e_available_record_mwh'])
    return f


def n_reviewers(letter_lines):
    def f(X):
        return len([x for x in letter_lines if re.match(r'\\section\*\{Reviewer \d+\}', x)])
    return f


def n_checks_named(X):
    # the experiment parts of the A89 claims JSON (part1_w184_..., part2_w185_...); part3_code_heads is not an experiment
    return len([k for k in P1.EX['A89'] if re.match(r'part\d+_w18[45]_', k)])


def declarations_w190a(X, actx, letter_lines):
    S, SF, U, LANG = NB.S, NB.SF, NB.U, NB.LANG
    DLF, by_w = P1.DLF, P1.by_w
    base = ORIG_DECL(X, actx, letter_lines)
    bm, bl = {d['line']: d for d in base[MAIN]}, {d['line']: d for d in base[LETTER]}

    def carry(ln, why):
        d = dict(bm[ln])
        d['specs'] = [dict(s) for s in d['specs']]
        d['ctx'] = f'W189p1 l. {ln} (46c9d77; same line at 3c379fa), carried unchanged: {why}'
        return d
    M, L_, C = [], [], []
    M.append(carry(1250, 'blue wrapper added around the supplement sentence; numbers unchanged'))
    M.append(carry(1353, '"Section~S5" -> "Table~S5"; numbers unchanged'))
    s = by_w(bm[1565]['specs'], 'four', '2025', '2035', '90', '10', '2035', '0.70', '0.70', '0.56', 'seven', '50', '2035', '0.50',
             '0.68')
    new = [SF('six', 'int', n_lower_contact_days, 'W188d SoC block summary CSV: (year, day) blocks with n_contact_lower_tso > 0 '
                                                  '(W188d contact rule, within 1e-4 MWh of the 10 % limit)'),
           SF('eight', 'int', n_soc_days, 'W188d SoC CSVs: (year, day) blocks plotted (2 years x 4 days)'),
           SF('14', 'raw', winter_min_pct('2025'), 'W188d: soc_min_tso_mwh(2025, Winter) / e_available_record_mwh(2025) x 100'),
           SF('15', 'raw', winter_min_pct('2035'), 'W188d: soc_min_tso_mwh(2035, Winter) / e_available_record_mwh(2035) x 100')]
    M.append(DLF(MAIN, 1565, *s[:5], *new, *s[5:],
                 ctx='W189p1 l. 1565 specs re-used for the carried tokens (90 / 10 now in that order); new: six, eight, 14, 15 '
                     '(ruling 4 (ii))'))
    M.append(carry(1627, '"Section~S5" -> "Table~S5"; numbers unchanged'))
    M.append(carry(1685, 'blue wrapper added (panel heading)'))
    M.append(carry(1695, 'blue wrapper added (panel heading)'))
    M.append(carry(1710, 'wrapper closed at the end of the note'))
    M.append(carry(2310, '"in Q" parenthetical and the evaluation-16 clause reworded; the seven numbers unchanged (171 is now '
                         'stated as the row total, "slack content ... almost all of it voltage slack")'))
    nrf_cost, nrf_share, meur = by_w(bm[1627]['specs'], '20.5', '0.02', '90.9')
    pct122 = by_w(bl[279]['specs'], '12.2')[0]
    n_prices = by_w(bl[170]['specs'], 'two')[0]
    C.append(DLF(COVER, 13, U('26', 'identifier', 'manuscript number EST-D-26-06323 (string-checked in the cover-letter checks)'),
                 U('06323', 'identifier', 'manuscript number EST-D-26-06323'),
                 SF('three', 'int', n_reviewers(letter_lines), 'letter v13: \\section*{Reviewer k} headings'),
                 ctx='JES-D-[number] -> EST-D-26-06323; "three Reviewers" unchanged'))
    C.append(DLF(COVER, 21, SF('two', 'int', n_checks_named, 'A89 claims JSON experiment parts part1_w184_section_4_4 (W184, '
                                                             'E3 NRF) and part2_w185_spread_table (W185, E2 penalty), named in '
                                                             'the next two sentences'),
                 dict(nrf_cost), dict(nrf_share), dict(meur), dict(pct122), dict(n_prices),
                 ctx='new (round 9c): the bracketed placeholder filled with W184 / W185; the specs are the W189p1 specs of the '
                     'same quantities (section 4.4 l. 1627; letter l. 279 / 170)'))
    C.append(DLF(COVER, 23, LANG('one', '"a prompt one" (pronoun)'), ctx='new (round 9c): the delay paragraph'))
    return {MAIN: M, LETTER: L_, COVER: C}


def _hunks_U0(d):
    out = []
    for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', d, re.M):
        a, b, c, e = int(m.group(1)), int(m.group(2) or 1), int(m.group(3)), int(m.group(4) or 1)
        out.append({'minus': list(range(a, a + b)) if b else [], 'plus': list(range(c, c + e)) if e else []})
    return out


def mark_new(numbers, texts):
    """Each '+' token marked NEW or CARRIED: carried when the written string occurs at least as often on the 9b '-' lines of
    its hunk (W189's tokenizer on the 46c9d77 text) as on the hunk's '+' lines up to and including it."""
    rows = numbers['tokens']
    for f in (MAIN, LETTER, COVER):
        d = _git(['diff', '-U0', BASE_FULL, HEAD_FULL, '--', f], cwd=CLONE)
        old = show(BASE_FULL, f)
        for h in _hunks_U0(d):
            ot = P1.tokens_of(f, old, set(h['minus'])) if h['minus'] else []
            have = Counter(t['written'] for t in ot)
            seen = Counter()
            for r in rows:
                if r['file'] == f and r['line'] in h['plus']:
                    seen[r['written']] += 1
                    r['new_vs_9b'] = 'carried' if seen[r['written']] <= have[r['written']] else 'NEW'
                    r['hunk_9b_lines'] = h['minus']
    for r in rows:
        r.setdefault('new_vs_9b', None)
    return [r for r in rows if r['new_vs_9b'] == 'NEW']


# ======================================================================================================================
#  (d) ruling 4 (i)-(iv) and the cover letter
# ======================================================================================================================
def texnorm(s):
    return P1.texnorm(s)


def task_ruling4(texts):
    main = texts[MAIN].split('\n')
    old = show(BASE_FULL, MAIN).split('\n')
    chk = collections.OrderedDict()
    val = {}
    # (i) B.4
    ib4 = [i for i, x in enumerate(main) if 'More generally, at every certified evaluation behind the tables' in x]
    tb4 = texnorm(main[ib4[0]]) if len(ib4) == 1 else ''
    chk['(i) B.4 sentence located once'] = len(ib4) == 1
    chk['(i) "every penalised slack in Q (daily closure and network feasibility)"'] = (
        'every penalised slack in q (daily closure and network feasibility) sits at its zero bound' in tb4)
    chk['(i) the net-power pair "penalised in the agent\'s own objective", at its bound'] = (
        "and the agent's net-power pair, penalised in the agent's own objective, is at its bound as well" in tb4)
    paren = re.findall(r'in q \(([^)]*)\)', tb4)
    chk['(i) no "in Q" parenthetical names the net-power pair'] = bool(paren) and not any('net-power' in p for p in paren)
    chk['(i) the net-power pair is not placed "in Q" anywhere in the sentence'] = not re.search(r"net-power pair[^.;]*in q\b", tb4)
    chk['(i) W188 (a) code anchors for the ESSO pair present (W188 (a): its penalty is never summed into Q)'] = any(
        k.startswith('esso') for k in P1.EX['W188A']['C_code_anchors'].keys())
    for p in P1.RULING3_PARTS[2:]:
        chk[f'(i) ruling-3 part still present: "{p[:50]}..."'] = p in tb4
    val['(i)'] = {'line': ib4[0] + 1 if len(ib4) == 1 else None, 'in_Q_parentheticals': paren,
                  'text': ws(main[ib4[0]])[:1600] if len(ib4) == 1 else None}
    # (ii) Figure 6 prose
    i6 = [i for i, x in enumerate(main) if 'shows how degradation reaches the dispatch' in x]
    t6 = texnorm(main[i6[0]]) if len(i6) == 1 else ''
    soc = _soc_sum()
    up = {f"{r['year']} {r['day']}": int(r['n_contact_upper_tso']) for r in soc}
    lo = {f"{r['year']} {r['day']}": int(r['n_contact_lower_tso']) for r in soc}
    no_lo = sorted(k for k, v in lo.items() if v == 0)
    pct = {y: winter_min_pct(y)(None) for y in ('2025', '2035')}
    chk['(ii) Figure 6 paragraph located once'] = len(i6) == 1
    chk['(ii) wording: 90 % limit "on every day", 10 % limit "on six of the eight", Winter "14--15 %"'] = (
        'reaching the 90 % limit of its available energy on every day and the 10 % limit on six of the eight (the winter days '
        'stop at 14--15 %)' in t6)
    chk['(ii) the old "between the 10 % and 90 % limits" is gone'] = 'between the 10 % and 90 % limits' not in t6
    chk['(ii) W188 (d): the upper (90 %) limit is touched on all 8 days'] = len(up) == 8 and all(v > 0 for v in up.values())
    chk['(ii) W188 (d): the lower (10 %) limit is touched on 6 of 8 days'] = sum(1 for v in lo.values() if v > 0) == 6
    chk['(ii) W188 (d): the two days missing the lower limit are the two Winter days'] = no_lo == ['2025 Winter', '2035 Winter']
    chk['(ii) W188 (d): Winter minima 13.9 % / 15.4 % of E^Av round to 14 and 15'] = (
        f"{pct['2025']:.0f}" == '14' and f"{pct['2035']:.0f}" == '15')
    sums = {(r['year'], r['day']): r for r in soc}
    cyc = {f'{y} {d}': float(r['energy_charged_tso_mwh']) / (float(P1.EX['LIM'][y]['soc_upper_90pct_record_mwh'])
                                                             - float(P1.EX['LIM'][y]['soc_lower_10pct_record_mwh']))
           for (y, d), r in sums.items()}
    chk['(ii) "cycles once or twice a day across the usable band": charged energy / band in [1, 2] (W189 measure)'] = all(
        0.95 <= v <= 2.05 for v in cyc.values())
    val['(ii)'] = {'line': i6[0] + 1 if len(i6) == 1 else None, 'n_contact_upper_by_day': up, 'n_contact_lower_by_day': lo,
                   'days_without_lower_contact': no_lo, 'winter_min_pct_of_E_av': pct,
                   'winter_min_soc_mwh': {y: float(sums[(y, 'Winter')]['soc_min_tso_mwh']) for y in ('2025', '2035')},
                   'charged_over_band': {k: round(v, 3) for k, v in cyc.items()},
                   'rule': 'W188 (d) contact rule: within 1e-4 MWh of the limit (n_contact_* columns of '
                           'soc_block_summary_reference_unit.csv, the TSO copy, as W189 part 1)',
                   'text': ws(main[i6[0]])[:1600] if len(i6) == 1 else None}
    # (iii) evaluation 16
    c = P1.EX['W188A']['cells']['pb_y2025_n5']
    tot = c['score']['positive_content_bound_eur_in_Q_refined']
    v = c['families']['voltage_slack']['positive_part_bound_eur_weighted']
    cl = c['families']['shared_ess_day_balance_slack']['positive_part_bound_eur_weighted']
    chk['(iii) "the superseded evaluation 16 a positive slack content of at most 171 €, almost all of it voltage slack"'] = (
        'the superseded evaluation 16 a positive slack content of at most 171 €, almost all of it voltage slack' in tb4)
    chk['(iii) the old "voltage-slack content of at most 171" is gone'] = 'voltage-slack content of at most 171' not in tb4
    chk['(iii) W188 (a): row total 170.69 <= 171 and rounds up to 171'] = tot <= 171 and math.ceil(tot) == 171
    chk['(iii) W188 (a): voltage part + closure part = row total'] = abs(v + cl - tot) < 1e-9 * max(1.0, tot)
    chk['(iii) W188 (a): "almost all of it voltage slack" (voltage share >= 0.99)'] = v / tot >= 0.99
    val['(iii)'] = {'row_total_bound_eur': tot, 'voltage_part_eur': v, 'closure_part_eur': cl, 'voltage_share': v / tot,
                    'source': f'{P1.W188A_JSON} cells[pb_y2025_n5] (sha256 pinned {P1.W188A_SHA[:8]})'}
    # (iv) Section S5 -> Table S5
    rx_sec, rx_tab = re.compile(r'Section\s*~?\s*S5\b'), re.compile(r'Table\s*~?\s*S5\b')
    old_sec = [i + 1 for i, x in enumerate(old) if rx_sec.search(x)]
    tx = dict(texts)
    tx.setdefault(HL, _rd(os.path.join(CLONE, HL)).decode('utf-8'))
    new_sec = {f: [i + 1 for i, x in enumerate(tx[f].split('\n')) if rx_sec.search(x)] for f in (MAIN, LETTER, COVER, HL)}
    new_tab = [i + 1 for i, x in enumerate(main) if rx_tab.search(x)]
    chk['(iv) 9b had "Section S5" at exactly two places'] = len(old_sec) == 2
    chk['(iv) both former places now read "Table S5"'] = len(old_sec) == 2 and all(rx_tab.search(main[i - 1]) for i in old_sec)
    chk['(iv) 0 "Section S5" remaining in main.tex'] = new_sec[MAIN] == []
    chk['(iv) 0 "Section S5" in the letter, cover letter and highlights'] = not any(new_sec[f] for f in (LETTER, COVER, HL))
    val['(iv)'] = {'section_S5_lines_9b': old_sec, 'section_S5_lines_9c': new_sec, 'table_S5_lines_9c_main': new_tab,
                   'contexts_9c': {i: ws(main[i - 1][max(0, rx_tab.search(main[i - 1]).start() - 80):
                                                     rx_tab.search(main[i - 1]).end() + 40]) for i in new_tab}}
    out = {'checks': dict(chk), 'values': val, 'holds': all(chk.values()),
           '(v)': 'blue wrappers are checked in part 2 (p515_s55_w190a_structure.py, W189 P2 method)'}
    return out


def _num(s):
    return float(s.replace(',', '').replace('+', '').replace('−', '-'))


def task_cover(texts):
    cl = texts[COVER].split('\n')
    ct = texts[COVER]
    chk = collections.OrderedDict()
    val = {}
    # brackets (optional arguments of commands removed first)
    body = re.sub(r'\\[A-Za-z]+\*?\[[^\]]*\]', ' ', ct)
    body = re.sub(r'(?<!\\)%.*', '', body)
    br = [(i + 1, m.group(0)) for i, ln in enumerate(body.split('\n')) for m in re.finditer(r'\[[^\]]*\]', ln)]
    chk['exactly one bracket left'] = len(br) == 1
    chk['the bracket left is the W187 one'] = len(br) == 1 and br[0][1].startswith('[If W187 landed')
    val['brackets'] = [{'line': a, 'text': b[:200]} for a, b in br]
    # the manuscript number
    a91 = _rd(A91).decode('utf-8')
    chk['manuscript number EST-D-26-06323 printed once'] = ct.count('EST-D-26-06323') == 1
    chk['the placeholder JES-D-[number] is gone'] = 'JES-D-' not in ct and '[number]' not in ct
    chk['EST-D-26-06323 is the number of the editorial notice (Addendum 91 / brief)'] = (
        'EST-D-26-06323' in a91 and 'EST-D-26-06323' in _rd(BRIEF).decode('utf-8'))
    val['manuscript_number_line'] = next((i + 1 for i, x in enumerate(cl) if 'EST-D-26-06323' in x), None)
    # the delay sentence
    dl = [i for i, x in enumerate(cl) if 'apologise for the delay' in x]
    sentences = re.split(r'(?<=[.])\s+(?=[A-Z])', ws(cl[dl[0]])) if len(dl) == 1 else []
    chk['the delay paragraph present once'] = len(dl) == 1
    val['delay'] = {'line': dl[0] + 1 if len(dl) == 1 else None, 'n_sentences': len(sentences), 'sentences': sentences}
    # the filled paragraph against the A89 JSON, the reports and the cell records
    ip = [i for i, x in enumerate(cl) if 'Beyond the Reviewers\' requests, the revision adds two robustness checks' in x]
    tp = texnorm(cl[ip[0]]) if len(ip) == 1 else ''
    p1, p2 = P1.EX['A89']['part1_w184_section_4_4'], P1.EX['A89']['part2_w185_spread_table']
    chk['filled paragraph located once'] = len(ip) == 1
    chk['NRF: "the restriction costs 20.5 k€, 0.02 % of the 90.9 m€ saving, so the reported 12.2 % benefit does not depend"'] = (
        'the restriction costs 20.5 k€, 0.02 % of the 90.9 m€ saving, so the reported 12.2 % benefit does not depend on the '
        'interface rule' in tp)
    chk['penalty: "halved and doubled: every sign and verdict is unchanged"'] = (
        'the two decisive storage comparisons were re-evaluated with the coordination penalties halved and doubled: every sign '
        'and verdict is unchanged' in tp)
    ratios = {pp: p2['spread'][pp]['spread_over_threshold']['baseline (rho0 x 1)'] for pp in ('1x', '2x')}
    chk['"about a tenth of their determinacy threshold": baseline spread / threshold at 1 significant figure = 0.1'] = (
        'move by about a tenth of their determinacy threshold' in tp and all(f'{v:.1g}' == '0.1' for v in ratios.values()))
    chk['A89: sign_kept and class_kept at both prices'] = all(p2['spread'][pp]['sign_kept'] is True and
                                                             p2['spread'][pp]['class_kept'] is True for pp in ('1x', '2x'))
    chk['A89: the eight cells certified, rho0 scale {0.5, 2.0}'] = (
        len(p2['new_cells']) == 8 and all(v['status'] == 'certified' for v in p2['new_cells'].values())
        and {v['rho0_scale'] for v in p2['new_cells'].values()} == {0.5, 2.0})
    val['spread_over_threshold_baseline'] = ratios
    val['a_tenth_note'] = ('"a tenth" is not a token of the W182b / W174 tokenizer, so it is checked here as a statement: '
                           f"baseline ratios {ratios['1x']:.4f} (1x) and {ratios['2x']:.4f} (2x); the main text (section 4.7) and "
                           'the letter print "about 0.12"')
    # W184 / W185 reports as printed, against the A89 JSON
    r84, r85 = _rd(W184_REPORT).decode('utf-8'), _rd(W185_REPORT).decode('utf-8')
    rep = collections.OrderedDict()
    m = re.search(r'\+(20,463\.88) €', r84)
    rep['W184 report Q(C_NRF) - Q(C_OPEN) +20,463.88 € = A89 at 2 dp'] = bool(m) and \
        f"{p1['differences_eur']['Q_C_NRF_minus_Q_C_OPEN']:,.2f}" == m.group(1)
    m = re.search(r'max\{3 × 4,209\.17, 2τ\} = (12,627\.51) €', r84)
    rep['W184 report threshold 12,627.51 € = A89 at 2 dp'] = bool(m) and \
        f"{p1['resolution']['share_numerator_rule_v6']['threshold']:,.2f}" == m.group(1)
    rep['W184 report "determinate, positive, 1.62×" = A89'] = ('determinate, positive, 1.62×' in r84 and f"{p1['resolution']['share_numerator_rule_v6']['margin_over_threshold']:.2f}" == '1.62'
                                                              and p1['resolution']['share_numerator_rule_v6']['sign'] == 1)
    m = re.search(r'(90,876,144\.52) €', r84)
    rep['W184 report Q(P_NRF) - Q(C_NRF) 90,876,144.52 € = A89 at 2 dp'] = bool(m) and \
        f"{p1['differences_eur']['Q_P_NRF_minus_Q_C_NRF']:,.2f}" == m.group(1)
    rows85 = re.findall(r'^\| (1×|2×), ρ₀ × (0\.5|2\.0) \| ([−+0-9,.]+) \| ([0-9,.]+) \| ([0-9.]+) \| (determinate, \w+) \|', r85, re.M)
    a89rows = {(pp, a['arm']): a for pp in ('1x', '2x') for a in p2['spread'][pp]['rows']}
    ok85 = len(rows85) == 4
    for pr, sc, mg, th, _r, verd in rows85:
        a = a89rows[(pr.replace('×', 'x'), f'rho0 x {sc}')]
        ok85 = ok85 and f"{a['margin']:,.2f}" == mg.replace('−', '-').replace('+', '') and f"{a['threshold']:,.2f}" == th \
            and verd == f"determinate, {'negative' if a['sign'] < 0 else 'positive'}"
    rep['W185 report pair table (4 rows: margin, threshold, verdict) = A89 at 2 dp'] = ok85
    b1 = re.search(r'\*\*1×:\*\* ([−+0-9,.]+) €, threshold ([0-9,.]+) €', r85)
    b2 = re.search(r'\*\*2×:\*\* ([−+0-9,.]+) €, threshold ([0-9,.]+) €', r85)
    rep['W185 report baselines (1x / 2x margin and threshold) = A89 at 2 dp'] = bool(b1 and b2) and all(
        f"{a89rows[(pp, 'baseline (rho0 x 1)')]['margin']:,.2f}" == b.group(1).replace('−', '-').replace('+', '')
        and f"{a89rows[(pp, 'baseline (rho0 x 1)')]['threshold']:,.2f}" == b.group(2) for pp, b in (('1x', b1), ('2x', b2)))
    sp_rep = {}
    for pp, b, prr in (('1x', b1, '1×'), ('2x', b2, '2×')):
        ms = [_num(r[2]) for r in rows85 if r[0] == prr] + ([_num(b.group(1))] if b else [])
        sp_rep[pp] = {'margins_printed': ms, 'spread': max(ms) - min(ms) if ms else None,
                      'threshold_baseline': _num(b.group(2)) if b else None}
        sp_rep[pp]['spread_over_threshold'] = (sp_rep[pp]['spread'] / sp_rep[pp]['threshold_baseline']
                                               if ms and b else None)
    rep['W185 report: spread / baseline threshold from the printed table is about a tenth at both prices'] = all(
        v['spread_over_threshold'] is not None and f"{v['spread_over_threshold']:.1g}" == '0.1' for v in sp_rep.values())
    rep['W185 report: every verdict "determinate", signs negative at 1x and positive at 2x'] = len(rows85) == 4 and all(
        r[5] == ('determinate, negative' if r[0] == '1×' else 'determinate, positive') for r in rows85)
    # the cell records
    rec = collections.OrderedDict()
    e3 = json.loads(_rd(E3_RESULTS))
    q3 = W177A.jget(e3, 'outcome.cell_report.Q_end')
    rec['E3 campaign_results Q_end = A89 instances.C_NRF.Q (bitwise)'] = q3 == p1['instances']['C_NRF']['Q']
    e2 = {}
    for cell, v in p2['new_cells'].items():
        p = os.path.join(E2_DIR, f'campaign_s55_step7_{cell}', 'campaign_results.json')
        q = W177A.jget(json.loads(_rd(p)), 'outcome.cell_report.Q_end')
        e2[cell] = {'path': p, 'sha256': _sha(p), 'Q_end': q, 'A89_Q': v['Q'], 'equal': q == v['Q']}
    rec['E2 campaign_results Q_end = A89 new_cells Q, 8 / 8 (bitwise)'] = len(e2) == 8 and all(v['equal'] for v in e2.values())
    val['records'] = {'E3': {'path': E3_RESULTS, 'sha256': _sha(E3_RESULTS), 'Q_end': q3, 'A89_Q': p1['instances']['C_NRF']['Q']},
                      'E2': e2, 'W185_report_spread': sp_rep}
    val['text'] = ws(cl[ip[0]])[:1600] if len(ip) == 1 else None
    out = {'checks': dict(chk), 'report_checks': dict(rep), 'record_checks': dict(rec), 'values': val,
           'holds': all(chk.values()) and all(rep.values()) and all(rec.values())}
    return out


def controls_w190a(texts):
    out = []
    t = dict(texts)
    t[MAIN] = texts[MAIN].replace('detailed in~Table~S5 of', 'detailed in~Section~S5 of', 1)
    r = task_ruling4_probe(t)
    out.append({'id': 'd-1 a planted "Section~S5" in main.tex is counted', 'refused': r == 1})
    t = dict(texts)
    t[MAIN] = texts[MAIN].replace('(daily closure and network feasibility) sits', "(daily closure, network feasibility and the "
                                                                                 "agent's net-power pair) sits", 1)
    tb4 = texnorm(next(x for x in t[MAIN].split('\n') if 'More generally, at every certified evaluation' in x))
    paren = re.findall(r'in q \(([^)]*)\)', tb4)
    out.append({'id': 'd-2 the 9b parenthetical (net-power pair "in Q") planted back is detected',
                'refused': any('net-power' in p for p in paren)})
    c2 = texts[COVER].replace('We apologise for the delay', '[Second placeholder.] We apologise for the delay', 1)
    body = re.sub(r'\\[A-Za-z]+\*?\[[^\]]*\]', ' ', c2)
    nbr = sum(1 for ln in body.split('\n') for _ in re.finditer(r'\[[^\]]*\]', re.sub(r'(?<!\\)%.*', '', ln)))
    out.append({'id': 'd-3 a planted second bracket in the cover letter is counted', 'refused': nbr == 2})
    return out


def task_ruling4_probe(t):
    rx_sec = re.compile(r'Section\s*~?\s*S5\b')
    return sum(1 for x in t[MAIN].split('\n') if rx_sec.search(x))


# ======================================================================================================================
#  checklist, guards, md, post-run, main
# ======================================================================================================================
REQUIRED_CAPTURES = {'(a) state / provenance': 'task_state', '(b) generator gate': 'task_generator',
                     '(c) verifier': 'verifier_summary', '(d) numbers': 'declarations_w190a', '(d) new tokens': 'mark_new',
                     '(d) ruling 4 (i)-(iv)': 'task_ruling4', '(d) cover letter v4': 'task_cover', 'guards': 'verify_guards'}
REQUIRED_SYMBOLS = {'W189p1': (P1, ('task_appendix', 'task_numbers', 'controls_appendix', 'declarations_w189p1', 'load_extras',
                                    'verify_guards', 'tokens_of', 'DLF', 'by_w', 'texnorm', 'RULING3_PARTS', 'EX', 'OUT_NAMES',
                                    'PREV_FULL', 'HEAD_FULL', 'CHECK_FAILURES', 'W188A_JSON', 'W188A_SHA')),
                    'W181a': (A, ('run_generator', 'run_verifier', 'locate_bytes', 'appendix_block')),
                    'W182b': (NB, ('S', 'SF', 'U', 'LANG', 'added_lines')), 'W177a': (W177A, ('jget',))}


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
    need = ([os.path.join(CLONE, f) for f in FILES] + list(REVIEW_COPY.values()) + [GEN_REL, VERIFY_REL, APPX_REL, W184_REPORT,
            W185_REPORT, E3_RESULTS, A91, BRIEF, W189P1_JSON, P1.A89_JSON, P1.W188A_JSON, P1.W188D_MAN])
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    for k in ('gen', 'red', 'verify_tmp', 'verify'):
        tgt = os.path.realpath(os.path.join(od, OUT_NAMES[k]))
        if not tgt.startswith(os.path.realpath(od) + os.sep):
            missing.append(f'{k} target not inside the output directory')
    if os.path.realpath(od).startswith(os.path.realpath(os.path.join(REPO, 'data')) + os.sep) or \
            os.path.realpath(od).startswith(os.path.realpath(os.path.join(REPO, MR)) + os.sep):
        missing.append('output directory under data/ or manuscript_review/ (not permitted for W190a)')
    if missing:
        for x in missing:
            print(f'REFUSED (capture-path checklist): missing {x}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def verify_guards():
    g = P1.verify_guards()
    mine = GUARD.verify(0)
    ok = g['ok'] and not mine
    if not ok:
        _fail(f'guards: W189p1 chain ok {g["ok"]}; this script {mine}')
    return {'w189p1_chain': g, 'w190a': {'counts': dict(GUARD.counts), 'verify_0_failures': mine}, 'ok': ok}


def _cell(s, n=200):
    return str(s).replace('|', '\\|').replace('\n', ' ')[:n]


def write_md(out, path):
    N, G, V, R4, CV = out['numbers'], out['generator'], out['verifier'], out['ruling4'], out['cover_letter']
    L = [f'# W190a part 1 -- generator gate, verifier and numbers at Overleaf {HEAD_FULL[:7]} (generated by {SCRIPT_NAME}; '
         'do not edit)', '', 'Objective convention: no objective value is computed; every rated monetary token carries its '
         'convention (Q = gross_operational_cost, contracted interface settlement excluded, unless stated).', '',
         f"Check failures: {len(out['check_failures'])}; guards ok {out['guards']['ok']}.", '',
         '## (b) generator gate', '',
         f"- reduced output {G['reduced']['n_lines']} lines, sha256 `{G['reduced']['sha256']}`; occurrences in the 9c main.tex "
         f"{G['reduced']['n_occurrences_in_9c_main']} (at lines {[(o['start_line'], o['end_line']) for o in G['reduced']['occurrences_in_9c_main']]}); "
         f"lines equal {G['reduced']['lines_equal_at_located_position']}; bitwise gate {G['gate_bitwise_reduced_in_main']}",
         f"- appendix block l. {G['reduced']['appendix_block_9c_lines']} ({G['reduced']['appendix_block_n_lines']} lines); lines "
         f"outside the generated text: {G['reduced']['block_lines_outside_the_generated_text']}",
         f"- full output sha256 `{G['full']['sha256']}` = appendixE_evaluation_record.tex: {G['full_mode_bitwise_committed']}",
         f"- verifier regenerations: {G['verifier_regenerations']}", '',
         '## (c) verifier', '', f"PASS {V['n_pass']}, FAIL {V['n_fail']}, exception {V['exception']}; reduced-mode PASS lines "
                                f"{len(V['reduced_mode_pass_lines'])}", '']
    L += [f'- {x}' for x in V['pass_lines']]
    L += ['', '## (d) numbers', '', f"By rating: {N.get('by_rating_total')}", '']
    for f, v in N.get('files', {}).items():
        L.append(f"- {f}: '+' lines {v['plus_lines']}, {v['n_tokens']} tokens, {v['by_rating']}; NOT RATED {v['not_rated_categories']}")
    L += ['', f"Negative control: {N.get('negative_control', {}).get('n_refused')} / {N.get('negative_control', {}).get('n')} refused.",
          '', '### Every token on a 9c \'+\' line', '',
          '| file | line | written | new vs 9b | rating | counterpart | at precision | convention |', '|---|---:|---|---|---|---|---|---|']
    L += [f"| {r['file']} | {r['line']} | {r['written']} | {r['new_vs_9b']} | {r['rating']} | {_cell(r['counterpart'] or r.get('category'), 180)} | "
          f"{_cell(r['value_at_written_precision'], 50)} | {_cell(r['convention'], 40)} |" for r in N.get('tokens', [])]
    L += ['', '## (d) ruling 4', ''] + [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in R4['checks'].items()]
    L += ['', '## (d) cover letter v4', ''] + [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in CV['checks'].items()]
    L += [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in CV['report_checks'].items()]
    L += [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in CV['record_checks'].items()]
    L += ['', '## Negative controls', ''] + [f"- {c['id']}: refused {c['refused']}" for c in out['negative_controls']]
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run(out_dir):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    p = os.path.join(od, OUT_NAMES['post'])
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = [OUT_NAMES[k] for k in ('log', 'man', 'json', 'md', 'gen', 'red', 'verify')]
    missing = [n for n in names if not os.path.isfile(os.path.join(od, n))]
    if missing:
        print(f'REFUSED: post-run inputs missing: {missing}', flush=True)
        sys.exit(1)
    extra = [os.path.relpath(os.path.join(dp, f), od) for dp, _d, fs in os.walk(os.path.join(od, OUT_NAMES['verify_tmp']))
             for f in fs]
    man = {'files': {n: _sha(os.path.join(od, n)) for n in names + sorted(extra)}, 'script_sha256': _sha(SCRIPT_NAME),
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
    _log(f'W190a part 1: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W190a part 1 (Addendum 91 ruling 4, W190a line): generator gate, verifier, numbers and '
                                    'ruling-4 wording at Overleaf 3c379fa',
           'objective_convention': 'no objective value is computed; rated numbers carry their own convention (Q = '
                                   'gross_operational_cost, contracted interface settlement excluded)',
           'static_reading': 'no TeX installation on this machine; nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME),
                                                      'committed_clean': committed_clean(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'interpreter': sys.executable, 'imported': {s_: _sha(s_) for s_ in IMPORTED_SCRIPTS},
           'revisions': {'head_3c379fa': HEAD_FULL, 'mid_52a607d': MID_FULL, 'base_9b_46c9d77': BASE_FULL,
                         'round8_a87e195': A87_FULL}}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    probs = P1.load_extras()
    out['extra_input_pins'] = {'problems': probs}
    for p in probs:
        _fail(f'(inputs) {p}')
    out['state'] = task_state()
    _log(f"(a) state holds {out['state']['holds']}; line-preserving {out['state']['line_preserving_main_letter']}")
    texts = {f: _rd(os.path.join(CLONE, f)).decode('utf-8') for f in (MAIN, LETTER, COVER)}
    # (c) and the full mode of (b): W189p1 task_appendix, PREV = a87e195 as in W189, outputs under W190a names
    P1.OUT_NAMES = dict(P1.OUT_NAMES, gen=OUT_NAMES['gen'], verify=OUT_NAMES['verify'], verify_tmp=OUT_NAMES['verify_tmp'])
    P1.PREV_FULL = A87_FULL
    P1.HEAD_FULL = HEAD_FULL
    apx, actx = P1.task_appendix(out_dir, texts)
    _VERIFY_TEXT['text'] = _rd(os.path.join(od, OUT_NAMES['verify'])).decode('utf-8')
    out['appendix_w189p1_method'] = apx
    out['appendix_w189p1_method_note'] = ('W189p1 task_appendix output: its "bitwise" / "head_block_is_generator_output_bitwise" '
                                          'fields refer to the FULL generator output (W189\'s question; 0 occurrences in the '
                                          'reduced manuscript by construction); the round-9c gate on the --reduced output is '
                                          'the "generator" item of this JSON. Its entry check rates the printed tables of the '
                                          '9c main.tex against the frozen JSON.')
    out['verifier'] = verifier_summary(apx)
    _log(f"(c) verify_appE: PASS {out['verifier']['n_pass']} FAIL {out['verifier']['n_fail']} exception "
         f"{out['verifier']['exception']}; entry check {apx['outcome']}")
    out['generator'], ctl_b, _red = task_generator(out_dir, texts, apx, actx)
    _log(f"(b) reduced {out['generator']['reduced']['n_lines']} lines, occurrences in 9c main "
         f"{out['generator']['reduced']['n_occurrences_in_9c_main']}, lines equal "
         f"{out['generator']['reduced']['lines_equal_at_located_position']}; gate {out['generator']['gate_bitwise_reduced_in_main']}; "
         f"full = committed {out['generator']['full_mode_bitwise_committed']}")
    ctl = list(P1.controls_appendix(actx)) + ctl_b
    # (d) numbers: W189p1 task_numbers on the 46c9d77 -> 3c379fa '+' lines with the W190a declarations
    P1.PREV_FULL = BASE_FULL
    P1.declarations_w189p1 = declarations_w190a
    out['numbers'], X = P1.task_numbers(texts, actx)
    P1.declarations_w189p1 = ORIG_DECL
    out['new_tokens_vs_9b'] = mark_new(out['numbers'], texts) if 'tokens' in out['numbers'] else None
    _log(f"(d) {out['numbers'].get('by_rating_total')}; control {out['numbers'].get('negative_control', {}).get('n_refused')}/"
         f"{out['numbers'].get('negative_control', {}).get('n')}; new tokens {len(out['new_tokens_vs_9b'] or [])}")
    out['ruling4'] = task_ruling4(texts)
    _log(f"(d) ruling 4 holds {out['ruling4']['holds']}: {[k for k, v in out['ruling4']['checks'].items() if not v]}")
    out['cover_letter'] = task_cover(texts)
    _log(f"(d) cover letter holds {out['cover_letter']['holds']}: "
         f"{[k for d in ('checks', 'report_checks', 'record_checks') for k, v in out['cover_letter'][d].items() if not v]}")
    ctl += controls_w190a(texts)
    nc = out['numbers'].get('negative_control', {})
    ctl.append({'id': f"d every rated number perturbed ({nc.get('n')})",
                'refused': bool(nc.get('n')) and nc.get('n_refused') == nc.get('n')})
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f"negative control not refused: {c['id']}")
    out['guards'] = verify_guards()
    out['check_failures'] = list(P1.CHECK_FAILURES) + list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    pj, pm, pn = (os.path.join(od, OUT_NAMES[k]) for k in ('json', 'md', 'man'))
    with open(pj, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, pm)
    e2 = [v['path'] for v in out['cover_letter']['values']['records']['E2'].values()]
    inputs = sorted(set([os.path.join(CLONE, f) for f in FILES] + list(REVIEW_COPY.values())
                        + [GEN_REL, VERIFY_REL, APPX_REL, W184_REPORT, W185_REPORT, E3_RESULTS, A91, BRIEF, W189P1_JSON,
                           P1.A89_JSON, P1.W188A_JSON, P1.W188D_MAN, SCRIPT_NAME] + e2 + list(IMPORTED_SCRIPTS)
                        + [os.path.join(P1.W188D_DIR, f) for f in P1.FIG_CSVS + (P1.SOC_SUMMARY_CSV,)]
                        + [os.path.join(MR, 'export_W160', f) for f in os.listdir(os.path.join(REPO, MR, 'export_W160'))]
                        + [v['path'] for v in (out['numbers'].get('inputs') or {}).values()]))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md', 'gen', 'red', 'verify')},
                'out_dir': out_dir, 'clone_head_full': out['state']['clone_head'],
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(showb(c, f)) for c in (BASE_FULL, HEAD_FULL) for f in FILES},
                'inputs': {p: {'sha256': _sha(p), 'committed_clean': committed_clean(p)} for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(out["check_failures"])}; guards ok {out["guards"]["ok"]}; '
         f'pickle {P1.PICKLE_COUNTS}')
    sys.exit(3 if out['check_failures'] else 0)


if __name__ == '__main__':
    main()
