"""P5.15 Addendum 89 (W189 line) and Addendum 90 (W189 additions) -- Planner task W189 PART 1: THE ROUND-9b AUDIT AT THE
OVERLEAF HEAD 46c9d77 -- NUMBERS AND CLAIMS. ZERO SOLVES, NO MODEL, NO PICKLE. Read-only on the Overleaf clone (git
rev-parse / show / diff / status only; no fetch, pull or checkout), on manuscript_review/, on every committed artifact and
on every earlier script. W189 part 2 (pointers / labels / structure) and W189a (supplement v2) are other tasks.

ITEMS (Planner order W189 part 1).
  (0) state: clone HEAD 46c9d779 (parent a87e195), tracked tree clean, working files = HEAD blobs; main.tex md5 49139cc2 =
      manuscript_review/main_a87e195_round9b.tex; letter md5 ed8f5b13 = response_to_reviewers_v12.tex; cover letter =
      cover_letter_v3.tex; fig_soc_soh_reference_unit.pdf = manuscript_review copy (bytes).
  (1) numbers on every '+' line of git diff -U0 a87e195 46c9d77 for main.tex, the letter and the cover letter, by the
      W182b method (W182b / W183h imported, not edited: tokenizer, spec constructors, evaluator, perturbation), with a NEW
      DECLARATIONS VERSION W189p1 keyed by the HEAD line; specs of carried sentences taken from the W182b (edd7a63) /
      W183h (a87e195) / W177b-1 declarations of the same sentence and RE-EVALUATED here; new specs against the frozen
      JSON 590088fe, the A89 claims JSON (W184 / W185), the W188 (a) slack inventory JSON, the W188 (d) CSVs, the case /
      market / params files. Letter pointers (Section~4.7, Table~7, ...) and citations are NOT RATED here (W189 part 2).
      The REDUCED APPENDIX (now Appendix B): gen_appE.py run IN-PROCESS into this stage's output directory (never onto
      manuscript_review/appendixE_evaluation_record.tex); located bitwise in main.tex at HEAD or not; line-level diff of the
      HEAD block against the generator output; verify_appE.py (its tempfile.mkdtemp() redirected into the output
      directory); the W180a / W181a entry check of the two printed tables (tab:appE_claims, tab:appE_neighbours) against the
      frozen JSON, the evaluation numbers mapped through the generator's E.21 (= supplementary Table S4) numbering.
  (2) the W184 / W185 / G10 sentences (section 4.4 NRF sentence, section 4.7 penalty paragraph, Appendix B.4 slack
      sentence; letter benchmark bullet and R2.6): numbers (item 1) and wording against Addendum 90 rulings 1-3.
  (3) Figure 6 prose against the W188 (d) CSVs; the PDF in the clone = the manuscript_review PDF; the figure script re-run
      (subprocess, canonical interpreter, -I, cwd = a fresh directory INSIDE the output directory holding copies of the
      three CSVs, MPLCONFIGDIR inside it) and compared (bytes; axes-clip-relative polylines); the clone PDF's data polylines
      compared directly with the CSV values.
  (4) Addendum 90 ruling 6: the letter discloses the Appendix D +-250 / +-100 correction; case files, a87e195 / submitted
      tables, supplement v1 CSV.
  (5) every occurrence of "predict" in main.tex, the letter, the cover letter and highlights.tex, with context; 0 applied to
      P1 / P2 / P5.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import; every guard of the imported chains and
this script's verified at exactly 0; pickle.load / pickle.loads / pickle.Unpickler blocked for the whole run (counters 0);
refuses to run concurrently with another copy of itself; refuses unless the output directory exists and holds nothing but
launch.log. JSON written only through gate_result_io. Capture-path checklist asserted before any task (exit 4).

NEGATIVE CONTROLS (in memory): every rated number perturbed (W182b perturbation); an altered E.20 entry; a one-byte change
of the generated block; a planted "prediction" beside P2; a planted three-decimal NRF percentage; a removed disclosure line;
a perturbed CSV value against the clone PDF polylines. Each must be refused (exit 3 otherwise).

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S55/w189_p1_numbers && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p1_numbers.py \\
      > data/SRP1/Results/P515S55/w189_p1_numbers/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p1_numbers.py --post-run
  (trial runs: --trial-out-dir <absolute path outside the repository>)
Exit: 0 = written, every integrity check holds, guards at 0 (a MISMATCH is a FINDING and never changes the exit code);
3 = written, an integrity check failed (listed); 1 / 4 = precondition / capture refusal.
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W189 part 1 numbers and claims at Overleaf 46c9d77 (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0, 'Unpickler': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W189p1: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W189p1: pickle.loads called -- no model loads are permitted')


class _BlockedUnpickler:
    def __init__(self, *_a, **_k):
        PICKLE_COUNTS['Unpickler'] += 1
        raise RuntimeError('W189p1: pickle.Unpickler called -- no model loads are permitted')


def _block_pickle():
    pickle.load, pickle.loads, pickle.Unpickler = _blocked_load, _blocked_loads, _BlockedUnpickler


_block_pickle()

import ast  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import difflib  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import zipfile  # noqa: E402
import zlib  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import openpyxl  # noqa: E402 -- reads the committed market-data workbook (Growth Factors), read only

import p515_s53_w183h_round8 as H  # noqa: E402  (W183h; imports W182c, W182b and their chains; NOT edited, not run)
import gate_result_io as GRIO  # noqa: E402

NB, A, W177A, W174, W164, B1, B1R = H.NB, H.A, H.W177A, H.W174, H.W164, H.B1, H.B1R
os.chdir(REPO)
_block_pickle()   # the imported modules re-bound pickle to their own blockers; this script's are re-installed

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s55_w189_p1_numbers'
S55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
OUT_DIR = os.path.join(S55, 'w189_p1_numbers')
OUT_NAMES = {'json': 'w189_p1_numbers.json', 'md': 'w189_p1_numbers.md', 'man': 'manifest_sha256.json',
             'gen': 'appE_generated_w189p1.tex', 'verify': 'verify_appE_output.txt', 'verify_tmp': 'verify_appE_tmp',
             'fig': 'fig_rerun', 'log': 'launch.log', 'post': 'manifest_post_run_sha256.json'}
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
HEAD_FULL = '46c9d779743b40a42da73c0050d8923e6af6fb55'
PREV_FULL = 'a87e1950fbdbb3667f37e64a43c298ba675e6cff'
MAIN, LETTER, COVER, HL = 'main.tex', 'response_to_reviewers_draft.tex', 'cover_letter.tex', 'highlights.tex'
FIGPDF = 'fig_soc_soh_reference_unit.pdf'
DIFF_FILES = (MAIN, LETTER, COVER)
MR = 'manuscript_review'
REVIEW_COPY = {MAIN: os.path.join(MR, 'main_a87e195_round9b.tex'), LETTER: os.path.join(MR, 'response_to_reviewers_v12.tex'),
               COVER: os.path.join(MR, 'cover_letter_v3.tex'), FIGPDF: os.path.join(MR, FIGPDF)}
MD5_DECLARED = {MAIN: '49139cc2', LETTER: 'ed8f5b13', COVER: '0b9e4c21'}
EXPECTED_HUNKS_MAIN = 34
GEN_REL, VERIFY_REL, APPX_REL = A.GEN_REL, A.VERIFY_REL, A.APPX_REL
GEN_COMMIT_EXPECTED = '22e19b56'     # the last commit touching gen_appE.py / verify_appE.py (round 8)
FIG_PY = os.path.join(MR, 'fig_soc_soh_reference_unit.py')
FZ_REL, FZ_SHA = A.FZ_REL, A.FZ_SHA
A89_JSON = os.path.join(S55, 'a89_w184_w185_claims', 'a89_w184_w185_claims.json')
A89_SHA = '6aa72591c889626d2589a19b30ebcad705278c7650b822e54f0cb24a623759df'
W188A_JSON = os.path.join(S55, 'w188a_slack_inventory', 'w188a_slack_inventory.json')
W188A_SHA = 'fdcb1ba18b88fceaf4c8ee85f44820e3888910a46450b8e46adfd1c37b033b21'
W188D_DIR = os.path.join(S55, 'w188d_operational_figure')
W188D_MAN = os.path.join(W188D_DIR, 'manifest_sha256.json')
FIG_CSVS = ('soc_hourly_reference_unit_2025_2035.csv', 'soc_limits_per_year_reference_unit.csv',
            'soh_per_block_calibrations.csv')
SOC_SUMMARY_CSV = 'soc_block_summary_reference_unit.csv'
MARKET_XLSX = os.path.join('data', 'SRP1', 'MarketData', 'SRP1_market_data.xlsx')
CODE_SRP = 'shared_resources_planning.py'
SUPP_ZIP = os.path.join(MR, 'supplement', 'SRP1_supplement_v1_a94d8193.zip')
SUBMITTED_MAIN = os.path.join('manuscript_submitted', 'main.tex')
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
STEP6_R9 = 'STEP6_ROUND9_CORRECTIONS.md'
REPORTS = ('P5_15_A89_W184_W185_CLAIMS.md', 'P5_15_W188A_SLACK_INVENTORY.md', 'P5_15_W188D_OPERATIONAL_FIGURE.md',
           'P5_15_W182B_NUMBER_CHECK.md', 'P5_15_W183H_ROUND8.md')
IMPORTED_SCRIPTS = ('p515_s53_w183h_round8.py', 'p515_s53_w182c_labels_pointers_appE.py', 'p515_s53_w182b_number_check.py',
                    'p515_s53_w181a_identity_appendixE.py', 'p515_s53_w177a_section4_frame_check.py',
                    'p515_s53_w177b_number_check.py', 'p515_s53_w177b1r_number_check.py', 'p515_s53_w178b_number_check.py')
MARK = '$^{b}$'
NAN = float('nan')
CHECK_FAILURES = []

# ======================================================================================================================
#  helpers
# ======================================================================================================================


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log(f'CHECK FAILURE: {msg}')


def _shab(b):
    return hashlib.sha256(b).hexdigest()


def _sha(rel):
    p = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
    with open(p, 'rb') as h:
        return _shab(h.read())


def _md5b(b):
    return hashlib.md5(b).hexdigest()


def _rd(rel):
    p = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
    with open(p, 'rb') as h:
        return h.read()


def _git(args, cwd=REPO, binary=False):
    r = subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, text=not binary, check=True)
    return r.stdout


def show(rev, path):
    return _git(['show', f'{rev}:{path}'], cwd=CLONE)


def showb(rev, path):
    return _git(['show', f'{rev}:{path}'], cwd=CLONE, binary=True)


def ws(s):
    return re.sub(r'\s+', ' ', s).strip()


def committed_clean(rel):
    if rel.startswith(CLONE + os.sep):
        return _git(['status', '--porcelain', '--', os.path.relpath(rel, CLONE)], cwd=CLONE).strip() == ''
    tracked = _git(['ls-files', '--', rel]).strip() != ''
    clean = _git(['status', '--porcelain', '--', rel]).strip() == ''
    return tracked and clean


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


def _csv_rows(path):
    with open(os.path.join(REPO, path), encoding='utf-8', newline='') as h:
        return list(csv.DictReader(h))


def _all_same(vals):
    vals = list(vals)
    return vals[0] if vals and all(v == vals[0] for v in vals) else None


def _dp(s):
    s = str(s).replace(',', '').lstrip('+-')
    return len(s.split('.')[1]) if '.' in s else 0


# ======================================================================================================================
#  extra inputs (loaded once, pinned)
# ======================================================================================================================
EX = {}


def load_extras():
    problems = []
    if _sha(A89_JSON) != A89_SHA:
        problems.append(f'{A89_JSON} sha256 differs from the A89 report pin')
    if _sha(W188A_JSON) != W188A_SHA:
        problems.append(f'{W188A_JSON} sha256 differs from the W188a report pin')
    EX['A89'] = json.loads(_rd(A89_JSON))
    EX['W188A'] = json.loads(_rd(W188A_JSON))
    man = json.loads(_rd(W188D_MAN))
    EX['W188D_MAN_OUT'] = man['outputs']
    for f in FIG_CSVS + (SOC_SUMMARY_CSV,):
        p = os.path.join(W188D_DIR, f)
        if man['outputs'].get(p) != _sha(p):
            problems.append(f'{p} sha256 differs from the W188d manifest')
    EX['SOC'] = _csv_rows(os.path.join(W188D_DIR, FIG_CSVS[0]))
    EX['LIM'] = {r['year']: r for r in _csv_rows(os.path.join(W188D_DIR, FIG_CSVS[1]))}
    EX['SOH'] = _csv_rows(os.path.join(W188D_DIR, FIG_CSVS[2]))
    EX['SOCSUM'] = _csv_rows(os.path.join(W188D_DIR, SOC_SUMMARY_CSV))
    wb = openpyxl.load_workbook(os.path.join(REPO, MARKET_XLSX), read_only=True, data_only=True)
    EX['MKT_GROWTH'] = {r[0]: r[1] for r in wb['Growth Factors'].iter_rows(values_only=True) if r and r[0]}
    wb.close()
    EX['CODE_SRP'] = _rd(CODE_SRP).decode('utf-8')
    cases = {}
    for net in ('case33_1', 'case33_2', 'case33_3'):
        for f in sorted(glob.glob(os.path.join(REPO, 'data', 'SRP1', net, f'{net}_20*.json'))):
            rel = os.path.relpath(f, REPO)
            d = json.loads(_rd(rel))
            g = [x for x in d['generators'] if x.get('type') == 'REF']
            cases[rel] = {'net': net, 'year': rel[-9:-5], 'ref_generators': g}
    EX['CASE33_REF'] = cases
    EX['SUBMITTED'] = _rd(SUBMITTED_MAIN).decode('utf-8')
    return problems


def a89(path):
    return W177A.jget(EX['A89'], path)


def w188a_named_certified():
    P5 = EX['W188A']['P5']
    rows = {r['cell']: r for r in EX['W188A']['summary_rows']}
    return [rows[c] for c in P5['named_certified']]


# ======================================================================================================================
#  (0) state
# ======================================================================================================================
def task_state():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    parent = _git(['rev-parse', 'HEAD^'], cwd=CLONE).strip()
    tracked_mod = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    untracked = _git(['status', '--porcelain'], cwd=CLONE).strip()
    files = {}
    for f in DIFF_FILES + (FIGPDF,):
        wb = _rd(os.path.join(CLONE, f))
        hb = showb(HEAD_FULL, f)
        rc = _rd(REVIEW_COPY[f])
        files[f] = {'sha256': _shab(wb), 'md5': _md5b(wb), 'working_equals_head_blob': wb == hb,
                    'equals_review_copy': wb == rc, 'review_copy': REVIEW_COPY[f], 'review_copy_sha256': _shab(rc),
                    'md5_declared_prefix': MD5_DECLARED.get(f), 'md5_matches_declared': (
                        _md5b(wb).startswith(MD5_DECLARED[f]) if f in MD5_DECLARED else None),
                    'lines': wb.count(b'\n') if f.endswith('.tex') else None}
    dstat = _git(['diff', '--stat', PREV_FULL, HEAD_FULL], cwd=CLONE)
    hunks = len(re.findall(r'^@@ ', _git(['diff', PREV_FULL, HEAD_FULL, '--', MAIN], cwd=CLONE), re.M))
    out = {'clone_head': head, 'clone_head_parent': parent, 'tracked_modified': tracked_mod, 'status_porcelain': untracked,
           'files': files, 'diff_stat': dstat.strip().split('\n'), 'main_hunks_default_context': hunks,
           'gen_appE_last_commit': _git(['log', '-1', '--format=%h %ci %s', '--', GEN_REL]).strip()[:160],
           'verify_appE_last_commit': _git(['log', '-1', '--format=%h %ci %s', '--', VERIFY_REL]).strip()[:160]}
    ok = (head == HEAD_FULL and parent == PREV_FULL and tracked_mod == '' and all(v['working_equals_head_blob'] for v in files.values())
          and all(v['equals_review_copy'] for v in files.values())
          and all(v['md5_matches_declared'] in (True, None) for v in files.values()) and hunks == EXPECTED_HUNKS_MAIN)
    out['holds'] = ok
    if not ok:
        _fail(f'(0) state: HEAD {head[:8]} parent {parent[:8]} tracked {tracked_mod!r} hunks {hunks} '
              f'files {[(f, v["working_equals_head_blob"], v["equals_review_copy"], v["md5_matches_declared"]) for f, v in files.items()]}')
    return out


# ======================================================================================================================
#  (1b) the reduced appendix
# ======================================================================================================================
def _strip_rows(lines, row_idx):
    L, removed = list(lines), []
    for i in row_idx:
        n = L[i].count(MARK)
        if n:
            removed.append({'line': i + 1, 'n_marks': n})
            L[i] = L[i].replace(MARK, '')
    return L, removed


def _e20_idx(lines):
    i0, i1 = A.table_body(lines, 'tab:appE_claims', '\\end{longtable}')
    return [i for i in range(i0, i1) if re.match(r'^\d+ & ', lines[i].strip())]


def _e21_idx(lines):
    i0, i1 = A.table_body(lines, 'tab:appE_evaluations', '\\end{longtable}')
    return [i for i in range(i0, i1) if re.match(r'^\d+ & ', lines[i].strip())]


def _summ(rows_out, extra=None):
    s = {'rows': len(rows_out), 'entries': 0, 'MATCH': 0, 'MISMATCH': 0, 'NOT_FROZEN_AGREES': 0, 'NOT_FROZEN_DISAGREES': 0}
    flat = []
    items = [(r.get('no') or r.get('eval'), k, e) for r in rows_out for k, e in r.get('entries', {}).items()]
    items += [('note', k, e) for k, e in (extra or {}).items()]
    for rid, k, e in items:
        s['entries'] += 1
        s[e['status']] = s.get(e['status'], 0) + 1
        flat.append({'row': rid, 'key': k, 'status': e['status'], 'printed': e.get('printed'),
                     'frozen_or_derived': e.get('frozen', e.get('derived'))})
    return s, flat


def entry_check(head_lines, gen_lines, ALL, COND, J, T):
    """E.21 of the generator (= supplementary Table S4 numbering) -> idmap; E.20 / E.22 as printed at HEAD."""
    g_str, g_rem = _strip_rows(gen_lines, _e21_idx(gen_lines) + _e20_idx(gen_lines))
    e21_rows, _ = A.parse_e21(g_str)
    e21_out, idmap, p21 = A.check_e21(e21_rows, ALL, COND, J)
    h_str, h_rem = _strip_rows(head_lines, _e20_idx(head_lines))
    e20_rows, _n, _nl = A.parse_e20(h_str)
    e20_out, used, p20 = A.check_e20(e20_rows, idmap, ALL, COND, T)
    e22_out, e22_note = A.check_e22(h_str, idmap, ALL, T)
    s20, f20 = _summ(e20_out)
    s22, f22 = _summ(e22_out, {'note_superseded': e22_note})
    s21, f21 = _summ(e21_out)
    bad = [dict(x, table='E.20 (HEAD)') for x in f20 if x['status'] in ('MISMATCH', 'NOT_FROZEN_DISAGREES')] + \
          [dict(x, table='E.22 (HEAD)') for x in f22 if x['status'] in ('MISMATCH', 'NOT_FROZEN_DISAGREES')]
    bad21 = [x for x in f21 if x['status'] in ('MISMATCH', 'NOT_FROZEN_DISAGREES')]
    return {'idmap': idmap, 'summary': {'E.20 (HEAD, printed)': s20, 'E.22 (HEAD, printed)': s22,
                                        'E.21 (generator = Table S4 numbering; not printed)': s21},
            'bad': bad, 'bad_e21_generator': bad21, 'mapping_problems': p20 + p21,
            'marks_removed_head_e20': h_rem, 'marks_removed_generator': len(g_rem),
            'e20_rows_numbered': [r['no'] for r in e20_out], 'e21_rows_numbered_1_61': [r['no'] for r in e21_out] == list(range(1, 62)),
            'e20_out': e20_out, 'used_claims': used}


def _block(lines):
    return A.appendix_block(lines)


def _rows_cells(lines, idx):
    return {A.row_cells(lines[i])[0]: A.row_cells(lines[i]) for i in idx}


def _tab_rows(lines, label):
    rows, _ = A.tabular_rows(lines, label)
    return {r['cells'][2]: r['cells'] for r in rows}


def task_appendix(out_dir, texts):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    gen_full = os.path.join(od, OUT_NAMES['gen'])
    gen_bytes, gen_stdout = A.run_generator(gen_full)
    vt = os.path.join(od, OUT_NAMES['verify_tmp'])
    os.mkdir(vt)
    old_tmp = tempfile.tempdir
    tempfile.tempdir = vt
    try:
        ver = A.run_verifier()
    finally:
        tempfile.tempdir = old_tmp
    with open(os.path.join(od, OUT_NAMES['verify']), 'x', encoding='utf-8') as fh:
        fh.write(ver['output'])
    head = texts[MAIN]
    prev = show(PREV_FULL, MAIN)
    hl, pl = head.split('\n'), prev.split('\n')
    gl = gen_bytes.decode('utf-8').split('\n')
    if gl and gl[-1] == '':
        gl = gl[:-1]
    committed = _rd(APPX_REL)
    hb0, hb1 = _block(hl)
    pb0, pb1 = _block(pl)
    occ_head = A.locate_bytes(head.encode('utf-8'), gen_bytes)
    occ_prev = A.locate_bytes(prev.encode('utf-8'), gen_bytes)
    blk_head = hl[hb0:hb1]
    sm = difflib.SequenceMatcher(a=gl, b=blk_head, autojunk=False)
    ops = []
    n_equal = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            n_equal += i2 - i1
            continue
        ops.append({'op': op, 'generator_lines': [i1 + 1, i2], 'head_lines': [hb0 + j1 + 1, hb0 + j2],
                    'n_generator': i2 - i1, 'n_head': j2 - j1,
                    'generator_first': ws(' '.join(gl[i1:i2]))[:240], 'head_first': ws(' '.join(blk_head[j1:j2]))[:240]})
    gen_info = {'path': GEN_REL, 'sha256': _sha(GEN_REL), 'committed_clean': committed_clean(GEN_REL),
                'last_commit': _git(['log', '-1', '--format=%h', '--', GEN_REL]).strip(),
                'output': os.path.join(out_dir, OUT_NAMES['gen']), 'output_sha256': _shab(gen_bytes),
                'output_lines': gen_bytes.count(b'\n'), 'stdout': gen_stdout.strip()}
    bit = {'generator': gen_info, 'equals_committed_appendix': gen_bytes == committed,
           'committed_appendix_sha256': _shab(committed),
           'equals_verifier_regeneration': ver['tempdir_output_sha256'] == _shab(gen_bytes),
           'occurrences_in_head_main': len(occ_head), 'occurrences_in_a87e195_main': len(occ_prev),
           'a87e195_block_lines': [pb0 + 1, pb1], 'head_block_lines': [hb0 + 1, hb1], 'head_block_n_lines': hb1 - hb0,
           'generator_n_lines': len(gl), 'lines_equal_in_order': n_equal, 'diff_hunks_generator_to_head_block': ops,
           'n_diff_hunks': len(ops),
           'head_block_is_generator_output_bitwise': len(occ_head) == 1}
    if not (bit['equals_committed_appendix'] and bit['equals_verifier_regeneration'] and len(occ_prev) == 1):
        _fail(f"(1b) generator integrity: committed {bit['equals_committed_appendix']} verifier {bit['equals_verifier_regeneration']} "
              f"a87e195 occurrences {len(occ_prev)}")
    if ver['exception'] or not gen_stdout.startswith('ok '):
        _fail(f'(1b) generator / verifier did not run cleanly: {gen_stdout!r} {ver["exception"]}')
    verifier = {'path': VERIFY_REL, 'sha256': _sha(VERIFY_REL), 'last_commit': _git(['log', '-1', '--format=%h', '--', VERIFY_REL]).strip(),
                'n_pass': ver['n_pass'], 'n_fail': ver['n_fail'], 'fail_lines': ver['fail_lines'], 'exception': ver['exception'],
                'tempdir_output': os.path.relpath(ver['tempdir_output'], REPO) if ver['tempdir_output'] else None,
                'tempdir_output_sha256': ver['tempdir_output_sha256'],
                'reads_main_tex': 'main.tex' in _rd(VERIFY_REL).decode('utf-8'),
                'mentions_reduced_table': bool(re.search(r'reduced|Table~?B|appendixB|46c9d77', _rd(VERIFY_REL).decode('utf-8'))),
                'extended_to_the_reduced_table': False,
                'note': 'verify_appE.py is unchanged since its last commit (see last_commit); it regenerates through gen_appE.py into its own '
                        'tempfile.mkdtemp() (redirected here into the output directory) and never reads main.tex, so it checks '
                        'the full generator output, not the reduced table printed at HEAD'}
    verifier['extended_to_the_reduced_table'] = bool(verifier['reads_main_tex'] and verifier['mentions_reduced_table'])
    J, T, ALL, COND = H.frozen_ctx()
    ec = entry_check(hl, gl, ALL, COND, J, T)
    ec_prev = entry_check(pl, gl, ALL, COND, J, T)
    # printed rows HEAD vs a87e195 vs generator
    h20, p20, g20 = (_rows_cells(x, _e20_idx(x)) for x in (hl, pl, gl))
    h22, p22, g22 = (_tab_rows(x, 'tab:appE_neighbours') for x in (hl, pl, gl))
    rows_cmp = {'E.20 rows HEAD': len(h20), 'E.20 rows a87e195': len(p20), 'E.20 rows generator': len(g20),
                'E.20 HEAD == a87e195 (cells, every row)': h20 == p20, 'E.20 HEAD == generator (cells, every row)': h20 == g20,
                'E.22 rows HEAD': len(h22), 'E.22 HEAD == a87e195': h22 == p22, 'E.22 HEAD == generator': h22 == g22,
                'E.20 rows differing from the generator': sorted(k for k in set(h20) | set(g20) if h20.get(k) != g20.get(k)),
                'tables printed at HEAD': sorted(lbl for t, lbl in A.TABLES if any(f'\\label{{{lbl}}}' in x for x in hl))}
    # rows of tab:appE_claims the main text cites
    cites = []
    for i, ln in enumerate(hl[:hb0], 1):
        for m in re.finditer(r'Table~\\ref\{tab:appE_claims\}(,?\s*rows?~?([0-9]+(?:--[0-9]+)?))?', ln):
            cites.append({'line': i, 'text': m.group(0), 'rows': m.group(2)})
    tot = {k: sum(v[k] for nm, v in ec['summary'].items() if 'HEAD' in nm) for k in ('entries', 'MATCH', 'MISMATCH',
                                                                                     'NOT_FROZEN_AGREES', 'NOT_FROZEN_DISAGREES')}
    out = {'bitwise': bit, 'verifier': verifier,
           'entry_check_head': {k: v for k, v in ec.items() if k not in ('idmap', 'e20_out', 'used_claims')},
           'entry_check_head_totals': tot,
           'entry_check_a87e195_same_method_totals': {nm: v for nm, v in ec_prev['summary'].items()},
           'idmap_generator_e21': {str(k): v for k, v in sorted(ec['idmap'].items())},
           'rows_compared': rows_cmp, 'main_text_citations_of_tab_appE_claims': cites}
    out['outcome'] = {'bitwise': bit['head_block_is_generator_output_bitwise'], 'generator_unchanged_since': gen_info['last_commit'],
                      'verify_n_pass': ver['n_pass'], 'verify_n_fail': ver['n_fail'],
                      'verify_extended_to_reduced_table': verifier['extended_to_the_reduced_table'],
                      'entries': tot['entries'], 'mismatch': tot['MISMATCH'], 'not_frozen_disagrees': tot['NOT_FROZEN_DISAGREES'],
                      'mapping_problems': len(ec['mapping_problems']),
                      'e20_rows_printed': len(h20), 'e20_unchanged_vs_a87e195': h20 == p20, 'e22_unchanged_vs_a87e195': h22 == p22}
    if ec['mapping_problems'] or ec['bad_e21_generator'] or not ec['e21_rows_numbered_1_61']:
        _fail(f"(1b) entry-check integrity: mapping {ec['mapping_problems']} gen E.21 bad {ec['bad_e21_generator'][:3]}")
    if sum(v['MISMATCH'] + v['NOT_FROZEN_DISAGREES'] for v in ec_prev['summary'].values()):
        _fail('(1b) the entry check does not reproduce 0 MISMATCH on the a87e195 tables (method control)')
    ctx = {'gen_bytes': gen_bytes, 'gen_lines': gl, 'head_lines': hl, 'idmap': ec['idmap'], 'J': J, 'T': T, 'ALL': ALL,
           'COND': COND, 'block': (hb0, hb1)}
    return out, ctx


def controls_appendix(actx):
    out = []
    hl = list(actx['head_lines'])
    idx = _e20_idx(hl)
    i = next(j for j in idx if re.fullmatch(r'\d+\.\d', A.row_cells(hl[j])[3]))
    cells = A.row_cells(hl[i])
    old = cells[3]
    cells[3] = f"{A.parse_num(old) + 0.1:.{A.dec_places(old)}f}"
    hl[i] = re.match(r'^\s*', hl[i]).group(0) + ' & '.join(cells) + ' \\\\'
    ec = entry_check(hl, actx['gen_lines'], actx['ALL'], actx['COND'], actx['J'], actx['T'])
    out.append({'id': f'1b-1 E.20 row {cells[0]} Delta gross altered {old} -> {cells[3]}',
                'bad': [(b['row'], b['key']) for b in ec['bad']], 'refused': len(ec['bad']) == 1 and ec['bad'][0]['key'] == 'd_gross_k'})
    gb = actx['gen_bytes']
    j = gb.find(b'Comparisons')
    mut = gb[:j] + (b'X' if gb[j:j + 1] != b'X' else b'Y') + gb[j + 1:]
    prev = show(PREV_FULL, MAIN).encode('utf-8')
    out.append({'id': '1b-2 generated block with one byte changed is not located in main.tex at a87e195',
                'refused': len(A.locate_bytes(prev, gb)) == 1 and len(A.locate_bytes(prev, mut)) == 0})
    return out


# ======================================================================================================================
#  (1) numbers: tokens, exclusions, declarations version W189p1
# ======================================================================================================================
PTR_NUM = r'\(?[A-Z]?\.?\d+(?:\.\d+)*\)?(?:\([ab]\))?'
POINTER_RX = re.compile(r'(?:Sections?|Subsections?|Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Appendix|Appendices|Algorithm|'
                        r'Reviewer|\\S)(?:~|\s)+' + PTR_NUM + r'(?:\s*(?:--|,|~and~|\s+and~|~and\s+|\s+and\s+)\s*~?' + PTR_NUM + r')*')
CITE_RX = re.compile(r'\[\d+(?:\s*,\s*\d+)*\]')


def pointer_spans(line):
    return [m.span() for m in POINTER_RX.finditer(line)] + [m.span() for m in CITE_RX.finditer(line)]


def comment_start(line):
    m = re.search(r'(?<!\\)%', line)
    return m.start() if m else None


def tokens_of(fname, text, plus):
    doc = W164.Doc(fname, text)
    lines = text.split('\n')
    toks = []
    for t in W174.tokenize_v3(doc):
        if t['line'] not in plus:
            continue
        t['file'] = fname
        t['status'] = None
        toks.append(t)
    NB.split_coefficients(toks)
    for t in toks:
        ln = lines[t['line'] - 1]
        cs = comment_start(ln)
        if t['excluded']:
            t.update(status='excluded', category=f"structure: {t['excluded']}")
        elif t['type'] in W177A.IDENT_TYPES:
            t.update(status='excluded', category=f"identifier ({t['type']})")
        elif cs is not None and t['col0'] >= cs:
            t.update(status='excluded', category='comment (not typeset)')
        elif fname != MAIN and any(a <= t['col0'] < b for a, b in pointer_spans(ln)):
            t.update(status='excluded', category='pointer / citation number (W189 part 2 scope; not rated here)')
    return toks


def DLF(fname, line, *specs, note=None, ctx=None):
    return {'file': fname, 'line': line, 'frag': None, 'specs': [dict(s) for s in specs], 'note': note, 'ctx': ctx}


def by_w(specs, *ws_):
    """pick specs by written token (first unused match each), from a list of earlier specs."""
    pool, out = list(specs), []
    for w in ws_:
        k = next(i for i, s in enumerate(pool) if s['w'] == w)
        out.append(dict(pool.pop(k)))
    return out


def UBR(w, fn, desc, conv='n/a'):
    """a written upper bound rounded UP at its written precision: fn(X) -> (recorded bound, source); MATCH when the
    recorded bound <= written and ceil(recorded at the written precision) == written."""
    return dict(w=w, kind='ubr', fn=fn, desc=desc, conv=conv)


def my_evaluate(X, spec, t, line):
    if spec['kind'] == 'ubr':
        try:
            rec, src = spec['fn'](X)
        except Exception as e:  # noqa: BLE001
            rec, src = NAN, f'error {type(e).__name__}: {e}'
        wv = W177A._written_value(t, line)
        dp = _dp(t['written']) if t['type'] != 'word' else 0
        up = math.ceil(round(rec * 10 ** dp, 9)) / 10 ** dp if rec == rec else NAN
        ok = rec == rec and rec <= abs(wv) and abs(up - abs(wv)) < 1e-12
        return {'status': 'match' if ok else 'MISMATCH', 'counterpart': src, 'counterpart_value': rec,
                'value_at_written_precision': f'{up:.{dp}f} (recorded bound rounded up)' if rec == rec else None,
                'rule': 'written upper bound = the recorded bound rounded up at the written precision', 'note': spec.get('desc')}
    return NB.evaluate(X, spec, t, line)


def my_apply(X, toks, decls, lines):
    stale, claimed_twice, decl_out = [], [], []
    by_line = collections.defaultdict(list)
    for t in toks:
        by_line[t['line']].append(t)
    seen = Counter(d['line'] for d in decls)
    dup = [ln for ln, n in seen.items() if n > 1]
    for d in decls:
        text = lines[d['line'] - 1]
        cand = [t for t in by_line[d['line']] if t['status'] != 'excluded']
        if len(cand) != len(d['specs']):
            stale.append({'file': d['file'], 'line': d['line'], 'why': f"{len(d['specs'])} specs for {len(cand)} tokens",
                          'tokens': [t['written'] for t in cand], 'specs': [s['w'] for s in d['specs']]})
            continue
        bad = [(t['written'], s['w']) for t, s in zip(cand, d['specs']) if t['written'].lower() != s['w'].lower()]
        if bad:
            stale.append({'file': d['file'], 'line': d['line'], 'why': 'written differs from the declaration', 'pairs': bad})
            continue
        res = []
        for t, s in zip(cand, d['specs']):
            if t['status'] is not None:
                claimed_twice.append(t['key'])
                continue
            try:
                r = my_evaluate(X, s, t, text)
            except Exception as e:  # noqa: BLE001 -- listed, a run failure
                stale.append({'file': d['file'], 'line': d['line'], 'why': f'evaluation error {type(e).__name__}: {e}',
                              'written': t['written']})
                continue
            t.update(r)
            t['spec'] = s
            res.append({'written': t['written'], 'col': t['col'], 'status': r['status']})
        decl_out.append({'file': d['file'], 'line': d['line'], 'tokens': res, 'note': d.get('note'), 'context': d.get('ctx')})
    return stale, claimed_twice, decl_out, dup


def my_negative_control(X, toks, lines):
    out = []
    for t in toks:
        if t['status'] not in ('match', 'approximate', 'derived') or 'spec' not in t:
            continue
        s = t['spec']
        try:
            u = NB.perturb(t, down=(s['kind'] in ('bound', 'ubr')))
        except KeyError:   # a number word with no neighbouring word in W182b's table (e.g. twenty-five): a numeral neighbour
            v = W174.wv3(t['text'])
            u = dict(t, written=str(int(v) + 1), text=str(int(v) + 1), type='num', value=float(v + 1))
        try:
            st = my_evaluate(X, s, u, lines[t['line'] - 1])['status']
        except Exception as e:  # noqa: BLE001
            st = f'error {type(e).__name__}'
        out.append({'file': t['file'], 'line': t['line'], 'col': t['col'], 'written': t['written'], 'altered': u['written'],
                    'status_altered': st, 'refused': st not in ('match', 'approximate', 'derived')})
    return out


# ---- source functions for the new specs ------------------------------------------------------------------------------
def _dval(v, arith, consistent=True):
    return {'value': v, 'arithmetic': arith, 'consistent': consistent}


def mkt_growth(name):
    def f(X):
        anchor = "sheet_name='Growth Factors'" in EX['CODE_SRP']
        return 100.0 * EX['MKT_GROWTH'][name] if anchor else None
    return f


def a89f(path, scale=1.0):
    return lambda X: a89(path) * scale


def spread(price):
    return lambda X: a89(f'part2_w185_spread_table.spread.{price}.spread')


def der_spread_ratio(X):
    s = a89('part2_w185_spread_table.spread')
    r = {p: s[p]['spread_over_threshold'] for p in ('1x', '2x')}
    base = {p: r[p]['baseline (rho0 x 1)'] for p in r}
    allv = [v for p in r for v in r[p].values()]
    return _dval(base['1x'], f"spread / rule-v6 threshold, baseline arm: 1x {base['1x']:.4f}, 2x {base['2x']:.4f}; all six arms "
                             f"{[round(v, 3) for v in allv]} (A89 claims JSON part2 spread_over_threshold); 'about 0.12 ... in "
                             f"each case' read as the baseline ratio of each price at 2 dp",
                 consistent=all(f'{v:.2f}' == '0.12' for v in base.values()) and all(0.105 <= v < 0.135 for v in allv))


def n_new_cells(X):
    nc = a89('part2_w185_spread_table.new_cells')
    return len(nc) if all(v['status'] == 'certified' for v in nc.values()) else None


def n_arms(X):
    s = a89('part2_w185_spread_table.spread')
    return _all_same([len(s[p]['rows']) for p in ('1x', '2x')])


def n_roles(X):
    return len({v['role'] for v in a89('part2_w185_spread_table.new_cells').values()})


def n_prices(X):
    return len({v['price'] for v in a89('part2_w185_spread_table.new_cells').values()})


def n_inside_window(X):
    return sum(1 for v in a89('part2_w185_spread_table.P2_cells').values() if v['inside_frozen_window'] is True)


def n_p2_cells(X):
    return len(a89('part2_w185_spread_table.P2_cells'))


def max_outside_distance(X):
    d = [v['distance_to_frozen_window'] for v in a89('part2_w185_spread_table.P2_cells').values() if not v['inside_frozen_window']]
    return d[0] if len(d) == 1 else None


def g10_offset(X):
    rows = w188a_named_certified()
    v = [r['signed_eur_in_Q_total'] for r in rows]
    med = statistics.median(v)
    return _dval(med / 1e3, f'W188a P5.named_certified (20 rows): signed EUR content in Q, median {med:,.2f}, range '
                            f'[{min(v):,.2f}, {max(v):,.2f}] (spread {max(v) - min(v):,.2f}); rounded at 1 dp: '
                            f'{sorted(Counter(f"{x / 1e3:.1f}" for x in v).items())}',
                 consistent=len(rows) == 20 and med < 0 and max(v) - min(v) <= 40.0)


def g10_rel(X):
    rows = w188a_named_certified()
    cells = EX['W188A']['cells']
    rel = [r['signed_eur_in_Q_total'] / cells[r['cell']]['gross_operational_cost_at_P'] for r in rows]
    med = statistics.median(rel)
    mant = [x / 10 ** math.floor(math.log10(abs(x))) for x in rel]
    return {'rel': rel, 'median': med, 'mant': mant, 'exp': 10.0 ** math.floor(math.log10(abs(med)))}


def g10_rel_mant(X):
    g = g10_rel(X)
    m = g['median'] / g['exp']
    return _dval(m, f"signed content / gross_operational_cost_at_P over the 20 certified rows: median {g['median']:.4e}, range "
                    f"[{min(g['rel']):.3e}, {max(g['rel']):.3e}]; mantissas at 1 significant figure "
                    f"{sorted(Counter(f'{x:.0f}' for x in g['mant']).items())}",
                 consistent=all(f'{x:.0f}' == '-2' for x in g['mant']))


def g10_rel_exp(X):
    g = g10_rel(X)
    return g['exp'] if _all_same([10.0 ** math.floor(math.log10(abs(x))) for x in g['rel']]) is not None else None


def g10_spread(X):
    v = [r['signed_eur_in_Q_total'] for r in w188a_named_certified()]
    return max(v) - min(v), f'W188a 20 certified rows: max - min signed content = {max(v) - min(v):.2f} EUR'


def g10_3x3_unit(X):
    r = next(r for r in EX['W188A']['summary_rows'] if r['cell'] == 't11_3x3_unit')
    return r['positive_content_bound_eur_in_Q_refined'], (
        f"W188a summary_rows[t11_3x3_unit].positive_content_bound_eur_in_Q_refined (flexibility P day-balance, not "
        f"established <= 0; eval {r['eval_key'][:8]}, cycle {r['point']}) = {r['positive_content_bound_eur_in_Q_refined']:.4f} EUR")


def g10_e16(X):
    c = EX['W188A']['cells']['pb_y2025_n5']
    tot = c['score']['positive_content_bound_eur_in_Q_refined']
    v = c['families']['voltage_slack']['positive_part_bound_eur_weighted']
    cl = c['families']['shared_ess_day_balance_slack']['positive_part_bound_eur_weighted']
    return tot, (f'W188a cells[pb_y2025_n5].score.positive_content_bound_eur_in_Q_refined = {tot:.2f} EUR (the row total: '
                 f'voltage {v:.2f} + closure {cl:.2f} EUR); the voltage-slack part alone is {v:.2f} EUR')


def e16_row(actx):
    def f(X):
        inv = {c: r for r, c in actx['idmap'].items()}
        return inv.get('pb_y2025_n5')
    return f


def fig_days(X):
    d = sorted({r['day'] for r in EX['SOC']})
    return len(d) if len(d) == len(X.j['SRP1_JSON']['Days']) and set(d) == set(X.j['SRP1_JSON']['Days']) else None


def fig_year(y):
    def f(X):
        ys = {r['year'] for r in EX['SOC']}
        return int(y) if y in ys and y in X.j['SRP1_JSON']['Years'] else None
    return f


def usable_band(y):
    def f(X):
        r = EX['LIM'][y]
        return float(r['soc_upper_90pct_record_mwh']) - float(r['soc_lower_10pct_record_mwh'])
    return f


def soh_rows(arm):
    return sorted([r for r in EX['SOH'] if r['arm'] == arm], key=lambda r: int(r['block_year']))


def first_floor_year(arms):
    def f(X):
        ys = []
        for a in arms:
            fl = [int(r['block_year']) for r in soh_rows(a) if r['floor_row_active_at_terminal_flag'] == '1']
            ys.append(fl[0] if fl else None)
        return _all_same(ys)
    return f


def n_calibrations(X):
    return len({r['calibration_table6'] for r in EX['SOH']})


def soh050_end(X):
    return float(soh_rows('soh050')[-1]['soh_end_of_block'])


def der_48(X):
    y, d, n = len(X.j['SRP1_JSON']['Years']), len(X.j['SRP1_JSON']['Days']), 1 + len(X.j['SRP1_JSON']['DistributionNetworks'])
    return _dval(y * d * n, f'SRP1.json: {y} years x {d} days x {n} network operators (TN + {n - 1} DNs) = {y * d * n} blocks')


def der_we_did_not(lines, ln):
    def f(X):
        n = len(re.findall(r'We did not ', lines[ln - 1]))
        return _dval(n, f'count of "We did not" sentences in the paragraph (letter l. {ln})')
    return f


def tn_conn_250(X):
    sub = re.findall(r'1 & 1 & ([0-9.]+) & -([0-9.]+) & ([0-9.]+) & -([0-9.]+) & 1\.00', EX['SUBMITTED'])
    prev = re.findall(r'1 & 1 & ([0-9.]+) & -([0-9.]+) & ([0-9.]+) & -([0-9.]+) & 1\.00', show(PREV_FULL, MAIN))
    v = {float(x) for t in sub + prev for x in t}
    return v.pop() if len(sub) == 1 and len(prev) == 1 and len(v) == 1 else None


def case33_2_node(X):
    m = NB._dn_by_node(X)
    k = [n for n, nm in m.items() if nm == 'case33_2']
    return int(k[0]) if len(k) == 1 else None


def case33_2_limit(X):
    vals = set()
    for rel, c in EX['CASE33_REF'].items():
        if c['net'] != 'case33_2':
            continue
        for g in c['ref_generators']:
            vals |= {abs(g['Pmax']), abs(g['Pmin']), abs(g['Qmax']), abs(g['Qmin'])}
    return vals.pop() if len(vals) == 1 else None


# ---- the declarations ------------------------------------------------------------------------------------------------
def declarations_w189p1(X, actx, letter_lines):
    S, U, SF, DER, SINGLE, LANG, NOT, P, W3 = NB.S, NB.U, NB.SF, NB.DER, NB.SINGLE, NB.LANG, NB.NOT, NB.P, NB.W3
    CONV, BOUND, SUBF = NB.CONV, NB.BOUND, B1R.SUBF
    car = H.carried
    b182 = {d['line']: d for d in NB.declarations()}
    b183 = {d['line']: d for d in H.declarations_w183h()}
    b177 = {d['line']: d for d in B1.declarations()}
    UNIT, REF4, BM, AF, C57 = B1.UNIT, B1.REF4, B1.BM, B1.AF, B1.C57
    soh070 = S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh'), 'SRP1_ESS_Params.json minimum_soh')
    soh050 = S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh')
    two_pct = S('2', 'pct0', NB.TWO_PCT, 'SRP1.json DiscountFactor (x 100)')
    pct122 = DER('12.2', NB.der_pct_price_taker, 'coordination saving relative to the price-taker benchmark (gross Q)')
    meur = DER('90.9', B1.der_meur(f'{BM}.benefit', 'benefit'), 'coordination saving vs the price-taker arrangement')
    budget = SF('1', 'raw', lambda X: X.j['ESS_PARAMS']['budget'] / 1e6, 'SRP1_ESS_Params.json budget (EUR) / 1e6',
                conv='investment budget (expected discounted investment cost, EUR)')
    cyc10, cyc90 = by_w(b182[1299]['specs'], '10', '90')
    ret50 = by_w(b182[1549]['specs'], '50')[0]
    c8000 = by_w(b182[1553]['specs'], '8,000')[0]
    four_days = SF('four', 'int', fig_days, 'W188d SoC CSV: distinct representative days = SRP1.json Days (count)')
    yr2025, yr2035 = SF('2025', 'int', fig_year('2025'), 'W188d SoC CSV year, in SRP1.json Years'), \
        SF('2035', 'int', fig_year('2035'), 'W188d SoC CSV year, in SRP1.json Years')
    nrf_cost = SF('20.5', 'k', a89f('part1_w184_section_4_4.differences_eur.Q_C_NRF_minus_Q_C_OPEN'),
                  'A89 claims JSON: Q(C_NRF) - Q(C_OPEN), the NRF restriction on the consensus exchange (W184 E3)',
                  conv='gross (Q = gross_operational_cost, settlement excluded)')
    nrf_x = SF('1.62', 'raw', a89f('part1_w184_section_4_4.resolution.share_numerator_rule_v6.margin_over_threshold'),
               'A89 claims JSON: |Q(C_NRF) - Q(C_OPEN)| / rule-v6 threshold 12,627.51 EUR (determinate)')
    nrf_share = SF('0.02', 'pct', a89f('part1_w184_section_4_4.ratios.share_of_gap'),
                   'A89 claims JSON: (Q(C_NRF) - Q(C_OPEN)) / (Q(P_NRF) - Q(C_OPEN)) = 0.0225 %',
                   conv='gross (ratio of gross Q differences)')
    enabled = SF('12.2', 'pct', a89f('part1_w184_section_4_4.ratios.enabled'),
                 'A89 claims JSON: enabled (Q(P_NRF) - Q(C_OPEN)) / Q(P_NRF) = 12.205 %', conv='gross (price-taker denominator)')
    prohib = SF('12.2', 'pct', a89f('part1_w184_section_4_4.ratios.X_deconfounded'),
                'A89 claims JSON: X (Q(P_NRF) - Q(C_NRF)) / Q(P_NRF) = 12.202 %', conv='gross (price-taker denominator)')
    sp1 = SF('1.6', 'k', spread('1x'), 'A89 claims JSON: spread of the decisive difference across the three rho0 arms, 1x',
             conv='gross (difference of Q, settlement excluded)')
    sp2 = SF('1.5', 'k', spread('2x'), 'A89 claims JSON: spread of the decisive difference across the three rho0 arms, 2x',
             conv='gross (difference of Q, settlement excluded)')
    ratio012 = DER('0.12', der_spread_ratio, 'spread / determinacy threshold')
    n8 = SF('eight', 'int', n_new_cells, 'A89 claims JSON part2 new_cells: count, every one certified')
    n3arms = SF('three', 'int', n_arms, 'A89 claims JSON part2 spread rows per price (rho0 x 1 / 0.5 / 2)')
    n_starts = SF('three', 'int', lambda X: _all_same([X.fzp(f'{AF}.passive.n_starts'), X.fzp(f'{AF}.price_taker.n_starts')]),
                  'T6 n_starts (both arms)')
    M, L_, C = [], [], []
    m, l_, c = M.append, L_.append, C.append
    # ================================ main.tex (HEAD line numbers) ================================
    m(DLF(MAIN, 525, *car(b182, 525, ('ten',)), ctx='W182b l. 525 (edd7a63), same sentence; a clause added after it'))
    m(DLF(MAIN, 950, U('2', 'notation', '(S^Av)^2: the square in the product-form complementarity row')))
    m(DLF(MAIN, 997, *car(b182, 996, ('one',)), ctx='W182b l. 996; a sentence inserted before it (ruling 5 wording)'))
    m(DLF(MAIN, 1204, SINGLE('single')))
    m(DLF(MAIN, 1250, SF('2.50', 'raw', mkt_growth('Energy'), 'SRP1_market_data.xlsx sheet Growth Factors, Energy (x 100); the '
                                                             'sheet production reads (shared_resources_planning.py, '
                                                             "sheet_name='Growth Factors')"),
          SF('2.00', 'raw', mkt_growth('Flexibility'), 'SRP1_market_data.xlsx sheet Growth Factors, Flexibility (x 100)'),
          ctx='unchanged numbers; the pointer to Appendix B replaced by "supplementary material (Section S1)"'))
    m(DLF(MAIN, 1292, *car(b183, 1291, ('32', '1.25', '3.00', '1.00', '2.50')), ctx='W183h l. 1291 (a87e195); pointer changed'))
    m(DLF(MAIN, 1293, *car(b182, 1291, ('2025',)), ctx='W182b l. 1291; pointers changed'))
    m(DLF(MAIN, 1353, *car(b183, 1352, ('three',)), ctx='W183h l. 1352; pointer changed'))
    m(DLF(MAIN, 1376, *car(b182, 1373, ('fourteen',)), ctx='W182b l. 1373; the global-optimality sentence rewritten'))
    m(DLF(MAIN, 1426, *car(b182, 1423, ('7',)), ctx='W182b l. 1423; the pointer to the removed Table 5 dropped'))
    c1427 = car(b183, 1427, ('ten', '171,310', '192,560', '24', '33', '253,880', 'five', 'ten', '4', '5', '4', '3', '4', '5', '2'))
    m(DLF(MAIN, 1428, *c1427[:3],
          CONV(S('182,700', 'e10', 'break_even.committed_W145.certified_only.breakeven_marginal_4h_energy_cost'),
               'break-even unit energy cost (EUR/MWh; fit to gross values)'),
          CONV(S('173,490', 'e10', 'break_even.committed_W145.breakeven_range[0]'), 'break-even unit energy cost (gross fit)'),
          CONV(S('190,380', 'e10', 'break_even.committed_W145.breakeven_range[1]'), 'break-even unit energy cost (gross fit)'),
          *c1427[3:], ctx='W183h l. 1427 + the "flagged certificate kept" row of the removed Table 5 moved into the prose '
                          '(certified-only fit 182,700; banded range 173,490-190,380, the committed_W145 fit)'))
    m(DLF(MAIN, 1505, *car(b182, 1517, ('4.1', '31.6', '64.4')), ctx='W182b l. 1517; "break-even" wording changed'))
    m(DLF(MAIN, 1565, four_days, yr2025, yr2035, cyc10, cyc90,
          SF('2035', 'int', first_floor_year(['C2_calfade']), 'W188d SoH CSV: first block with the floor row active, baseline'),
          dict(soh070),
          SF('0.70', 'raw', usable_band('2025'), 'W188d limits CSV 2025: 90 % - 10 % of E^Av (MWh) = 0.7044'),
          SF('0.56', 'raw', usable_band('2035'), 'W188d limits CSV 2035: 90 % - 10 % of E^Av (MWh) = 0.5600'),
          SF('seven', 'int', n_calibrations, 'W188d SoH CSV: calibrations'),
          dict(ret50),
          SF('2035', 'int', first_floor_year(['C2_calfade', 'C3_unit', 'C3_midblock']),
             'W188d SoH CSV: first floor block of the baseline and both 50 % retention arms (equal)'),
          dict(soh050),
          SF('0.68', 'raw', soh050_end, 'W188d SoH CSV: baseline 0.50 floor, SoH at the end of the 2035 block (0.677290)'),
          ctx='new (Figure 6 paragraph, section 4.3)'))
    m(DLF(MAIN, 1571, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('2025', 'year', UNIT),
          dict(four_days), dict(yr2025), dict(yr2035), dict(cyc10), dict(cyc90), dict(soh070), dict(soh050),
          ctx='new (Figure 6 caption)'))
    m(DLF(MAIN, 1627, *car(b182, 1628, ('four', 'one', 'twelve', 'eight')), LANG('two', '"the two effects"'),
          nrf_cost, nrf_x, nrf_share, dict(meur), enabled, prohib,
          ctx='W182b l. 1628 for the first four; the NRF sentence (W184) new'))
    m(DLF(MAIN, 1635, *car(b183, 1639, ('nine', 'three', 'three', 'five', 'three', '3', 'single', '236.7', '318.0', '81.3',
                                         '18.0')), ctx='W183h l. 1639; pointer changed'))
    m(DLF(MAIN, 1676, *car(b182, 1677, ('54.5', '64.4', 'three', '2030', '2035', '7', '0.5', '1', '0.5', '2', '17', '20')),
          ctx='W182b l. 1677; table reference now "(a)"'))
    m(DLF(MAIN, 1681, *car(b182, 1682, ('7', '0.25', '1')), ctx='W182b l. 1682 (caption); merged two-panel table'))
    m(DLF(MAIN, 1685, dict(two_pct), ctx='new panel header "(a) Installation year, 2 % discount rate"'))
    m(DLF(MAIN, 1695, P('2025', 'year', UNIT), ctx='new panel header "(b) Discount rate, installation in 2025"'))
    m(DLF(MAIN, 1709, NOT('0'), *car(b182, 1701, ('2035', '2030', '12.3')), ctx='W182b l. 1701 (table note), merged note'))
    m(DLF(MAIN, 1710, *car(b182, 1728, ('318.0', '2025')), *car(b182, 1730, ('2', '2')),
          ctx='W182b l. 1728 and 1730 (the removed discount table notes), merged'))
    m(DLF(MAIN, 1715, *car(b182, 1706, ('four', '40.8', '114.6', '8', '2025', '2', '2')), ctx='W182b l. 1706; pointer "(b)"'))
    m(DLF(MAIN, 1724, *car(b182, 1739, ('single', '32', '42', 'ten', '15', '21', 'three', '0.9')), ctx='W182b l. 1739'))
    m(DLF(MAIN, 1728, dict(n8), dict(n3arms), SF('two', 'int', n_roles, 'A89 claims JSON part2 new_cells roles {x0, unit}'),
          sp1, sp2, ratio012, SF('seven', 'int', n_inside_window, 'A89 claims JSON part2 P2_cells inside_frozen_window'),
          SF('eight', 'int', n_p2_cells, 'A89 claims JSON part2 P2_cells (count)'),
          SF('0.7', 'k', max_outside_distance, 'A89 claims JSON part2 P2_cells: the one distance to the frozen window '
                                                '(e2_x0_2x_rho0p5, 682.59 EUR)', conv='gross (distance of Q to a window)'),
          ctx='new (penalty-sensitivity paragraph, W185)'))
    m(DLF(MAIN, 1749, *car(b182, 1760, ('three', '41', 'single', '42', '61.4', '8,570', '25.6', 'single')), ctx='W182b l. 1760'))
    m(DLF(MAIN, 1758, *car(b182, 1773, ('10^{-5}', '62')), ctx='W182b l. 1773; pointer "(Table S4)"'))
    m(DLF(MAIN, 1788, *car(b182, 1803, ('4', '7', '1.75', '2')), ctx='W182b l. 1803; "break-even" wording changed'))
    m(DLF(MAIN, 1821, U('one', 'model structure', '"the storage schedule is one scenario-free variable"'),
          LANG('single', '"At a single scenario"')))
    m(DLF(MAIN, 1902, U('one', 'model structure', '"one per channel"')))
    pu = 'admm.penalty_update.'
    m(DLF(MAIN, 1953, U('one', 'model structure', '"with one value for all agents"'),
          S('1.5', 'raw', ('rec', 'SRP1_PARAMS', pu + 'increase_factor'), 'SRP1_params.json increase_factor'),
          S('1.5', 'raw', ('rec', 'SRP1_PARAMS', pu + 'decrease_factor'), 'SRP1_params.json decrease_factor'),
          S('5', 'raw', ('rec', 'SRP1_PARAMS', pu + 'residual_balance_ratio'), 'SRP1_params.json residual_balance_ratio (mu)'),
          S('5', 'raw', ('rec', 'SRP1_PARAMS', pu + 'residual_balance_ratio'), "the same (mu')"),
          S('3', 'raw', ('rec', 'SRP1_PARAMS', pu + 'residual_balance_ratio_pf_decrease'), 'SRP1_params.json (mu\' of PF)'),
          S('10^{-4}', 'sci', ('rec', 'SRP1_PARAMS', pu + 'min'), 'SRP1_params.json min'),
          S('10^{4}', 'sci', ('rec', 'SRP1_PARAMS', pu + 'max'), 'SRP1_params.json max'),
          S('one', 'raw', ('rec', 'SRP1_PARAMS', pu + 'balancing_exempt_until.ess.dual_ratio_below'), 'dual_ratio_below'),
          S('five', 'int', ('rec', 'SRP1_PARAMS', pu + 'balancing_exempt_until.ess.consecutive_cycles'), 'consecutive_cycles'),
          U('two', 'compound adjective', '"two-phase schedule"'),
          S('ten', 'int', ('rec', 'SRP1_PARAMS', pu + 'freeze_after_unchanged_cycles'), 'freeze_after_unchanged_cycles'),
          S('200', 'int', ('rec', 'SRP1_PARAMS', pu + 'freeze_backstop_cycle'), 'freeze_backstop_cycle'),
          ctx='numbers unchanged since a87e195 (one clause removed); rated against the SRP1 case file (W174 A301-A304 '
              'source)'))
    m(DLF(MAIN, 1974, U('two', 'compound adjective', '"two-block nonconvex problems" (a problem class)')))
    m(DLF(MAIN, 2120, NOT('0'), by_w(b182[2919]['specs'], '4,539.07')[0],
          DER('0.1', lambda X: fmt_check(actx, 'amounts'), 'amounts in the printed appendix tables at 1 dp (kEUR)'),
          DER('two', lambda X: fmt_check(actx, 'ratios'), 'ratio columns of the printed appendix tables at 2 dp'),
          DER('three', lambda X: fmt_check(actx, 'resid'), 'the residual ratio of Table S6 (generator E.24 Resid.) at 3 dp'),
          ctx='the appendix opening paragraph rewritten (generator prose at a87e195)'))
    m(DLF(MAIN, 2162, LANG('two', '"the two evaluations of Table S4" (a pair)'), dict(budget),
          ctx='E.20 caption: "of Table E.21" -> "of Table S4"'))
    m(DLF(MAIN, 2308, *car(b182, 2929, ('10^{-6}', '16', '1.8', '10^{-6}', '5', '10^{-6}')),
          ctx='W182b l. 2929; "evaluation 16 of Table S4" (the generator E.21 numbering)'))
    m(DLF(MAIN, 2310, DER('-13.0', g10_offset, 'common relaxation offset in Q (kEUR)'),
          DER('-2', g10_rel_mant, 'offset / Q, mantissa'), SF('10^{-5}', 'sci', g10_rel_exp, 'offset / Q, order of magnitude'),
          UBR('40', g10_spread, 'variation of the offset across the 20 certified rows (EUR)', conv='gross'),
          UBR('2.2', g10_3x3_unit, 'positive flexibility-balance content bound, 3 x 3 reference unit (EUR)', conv='gross'),
          SF('16', 'int', e16_row(actx), 'generator E.21 (= Table S4) row of pb_y2025_n5'),
          UBR('171', g10_e16, 'positive content bound of the superseded evaluation 16 (EUR)', conv='gross'),
          ctx='new (G10 sentence, W188 (a))'))
    m(DLF(MAIN, 2314, *car(b182, 2934, ('42', 'single', '3', '11', '15', '22', '23', '26', '31', '34', '39', '42', '59', '32',
                                         '24', '8', '10', '5', '2', '3', '174', '65', '21', '89', '0.86', 'four', '32', '33',
                                         '60', '61', 'Ten', '46', '5', '2', '3', '5', '10', '11', '12', '14', '17', '20', '23',
                                         '36', '0.94', 'three', '12', '14', '3', '0.004', '0.74', '0.78', '0.88', '21', '10',
                                         '119', 'ten')), ctx='W182b l. 2934; "of Table S4"'))
    # ================================ the letter (HEAD line numbers) ================================
    sr = X.j['SRP1_JSON']
    l_(DLF(LETTER, 58, DER('Two', der_we_did_not(letter_lines, 58), 'requests not met in full'),
           SUBF('five', 'five years (2025, 2028, 2031, 2034, and 2037)', 'five', 'the submitted instance: five years'),
           SUBF('twenty-five', 'resulting in 25 operating scenarios per representative day', '25', 'the submitted instance'),
           SF('3', 'int', lambda X: X.j[NB.C3X]['NumMarketScenarios'], 'SRP1__s53_3x3.json NumMarketScenarios'),
           SF('3', 'int', lambda X: B1R._op_scen(X.j[NB.C3X]), 'SRP1__s53_3x3.json num_operation_scenarios')))
    l_(DLF(LETTER, 81, SF('three', 'int', lambda X: len(X.j['SRP1_JSON']['Years']), 'SRP1.json Years (count)'),
           SF('five', 'int', lambda X: _all_same(X.j['SRP1_JSON']['Years'].values()), 'SRP1.json block length (each)'),
           SINGLE('single'),
           SF('five', 'int', lambda X: len(X.j[NB.C3X]['Years']), '3 x 3 Years (count)'),
           SF('three', 'int', lambda X: _all_same(X.j[NB.C3X]['Years'].values()), '3 x 3 block length (each)')))
    l_(DLF(LETTER, 97, *[S(w, 'pct0', f'discount[{r}].rate') for w, r in (('0', '0.0'), ('2', '0.02'), ('5', '0.05'), ('8', '0.08'))],
           *[S(w, 'k', f'discount[{r}].value_minus_I') for w, r in (('-40.8', '0.0'), ('-64.4', '0.02'), ('-92.7', '0.05'),
                                                                       ('-114.6', '0.08'))],
           dict(two_pct), dict(two_pct), dict(two_pct)))
    lat = by_w(b182[1301]['specs'], '0.25', '0.5', '2', '4')
    l_(DLF(LETTER, 125, *lat, U('One', 'model structure', '"One recourse evaluation is made per candidate plan"'),
           U('one', 'model structure', '"one local problem per network block"'),
           SF('three', 'int', lambda X: len(X.j['SRP1_JSON']['Years']), 'SRP1.json Years (count)'),
           SF('four', 'int', lambda X: len(X.j['SRP1_JSON']['Days']), 'SRP1.json Days (count)'),
           SF('four', 'int', lambda X: 1 + len(X.j['SRP1_JSON']['DistributionNetworks']), 'network operators: TN + DNs'),
           SINGLE('single'), DER('48', der_48, 'network blocks per cycle'),
           U('one', 'model structure', '"one per shared-ESS agent"'),
           SF('three', 'int', lambda X: len(X.j['SRP1_JSON']['DistributionNetworks']), 'shared-ESS agents = interface nodes'),
           U('one', 'model structure', '"one per interface node"'), SINGLE('single')))
    l_(DLF(LETTER, 170, SF('two', 'int', n_prices, 'A89 claims JSON part2: prices {1x, 2x} (the two decisive comparisons)'),
           dict(n8), LANG('two', '"the two plans"'), dict(sp1), dict(sp2), dict(n3arms), dict(ratio012), dict(n_starts)))
    l_(DLF(LETTER, 218, P('two', 'nnodes', C57)))
    l_(DLF(LETTER, 231, LANG('one', '"in one figure"'), dict(four_days), dict(yr2025), dict(yr2035)))
    l_(DLF(LETTER, 253, S('0.985', 'raw', ('rec', 'ESS_PARAMS', 'ageing.calendar_retention_per_year'), 'SRP1_ESS_Params.json'),
           S('31.6', 'k', 'ageing.rows[C2].value_minus_I'), S('64.4', 'k', 'ageing.rows[C2_calfade].value_minus_I')))
    l_(DLF(LETTER, 260, P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('7', 'node', UNIT, [7]),
           S('-4.1', 'k', 'ageing.rows[no_ageing].value_minus_I'), S('-31.6', 'k', 'ageing.rows[C2].value_minus_I'),
           S('-64.4', 'k', 'ageing.rows[C2_calfade].value_minus_I'), dict(c8000), S('-45.2', 'k', 'ageing.rows[C4].value_minus_I'),
           dict(ret50), S('-73.7', 'k', 'ageing.rows[C3_unit].value_minus_I'),
           S('-65.9', 'k', 'ageing.rows[C3_midblock].value_minus_I')))
    c1360 = car(b177, 1360, ('0.70', '2035', '50', '8,000', '0.70', '0.50', '0.18', '0.20', '0.27', '2025', '2030', '2035',
                             '4.9', '59.5'))
    l_(DLF(LETTER, 261, dict(soh070), dict(soh050), dict(c1360[12], w='+4.9'),
           S('2035', 'int', 'ageing.rows[C2_calfade].floor_year_070'), dict(ret50)))
    l_(DLF(LETTER, 279, S('1', 'int', f'{BM}.sweep.sweep_passive_cold.n_blocks'), SF('12', 'int', B1._twelve, '3 years x 4 days'),
           S('8', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_blocks'), SF('12', 'int', B1._twelve, '3 years x 4 days'),
           dict(nrf_cost), dict(nrf_share), dict(meur), dict(pct122)))
    l_(DLF(LETTER, 280, by_w(b182[1773]['specs'], '10^{-5}')[0]))
    l_(DLF(LETTER, 281, LANG('single', '"single machine"'), by_w(b182[1760]['specs'], 'single', 'single')[1]))
    l_(DLF(LETTER, 282, SINGLE('single')))
    l_(DLF(LETTER, 283, SF('250', 'int', tn_conn_250, 'the TN-connection row printed in the submitted and a87e195 Appendix D '
                                                     '(one row, +-250)'),
           SF('7', 'int', case33_2_node, 'SRP1.json DistributionNetworks: case33_2 connection node'),
           SF('100', 'int', case33_2_limit, 'case33_2_*.json REF generator |Pmax| = |Pmin| = |Qmax| = |Qmin| (every year file)'),
           U('one', 'model structure', '"one row per network"')))
    # ================================ the cover letter ================================
    c(DLF(COVER, 19, S('64.4', 'k', W177A._cl(REF4, 'd_gross')), dict(pct122),
          SUBF('18.25', '-18.25\\%', '18.25', 'the submitted version figure (withdrawn)'),
          SUBF('92.16', '-92.16\\%', '92.16', 'the submitted version figure (withdrawn)')))
    c(DLF(COVER, 21, U('10', 'identifier', '"round 10": the revision-round number in the bracketed placeholder')))
    return {MAIN: M, LETTER: L_, COVER: C}


def fmt_check(actx, what):
    hl, gl = actx['head_lines'], actx['gen_lines']
    if what in ('amounts', 'ratios'):
        e20 = [A.row_cells(hl[i]) for i in _e20_idx(hl)]
        e22 = [r['cells'] for r in A.tabular_rows(hl, 'tab:appE_neighbours')[0]]
        num = lambda s: A.strip_markers(s)[0].replace('$-$', '-').replace('$', '')  # noqa: E731
        if what == 'amounts':
            cells = [num(r[k]) for r in e20 for k in (3, 4, 7)] + [num(r[k]) for r in e22 for k in (3, 4)]
            want = 1
        else:
            cells = [num(r[k]) for r in e20 for k in (5, 8)] + [num(r[5]) for r in e22]
            want = 2
        cells = [x for x in cells if x not in ('---', '')]
        dps = Counter(_dp(x) for x in cells)
        val = 0.1 if what == 'amounts' else 2
        return _dval(val, f'{len(cells)} printed numeric cells of tab:appE_claims / tab:appE_neighbours: decimal places '
                          f'{dict(dps)}', consistent=set(dps) == {want})
    i0, i1 = A.tabular_rows(gl, 'tab:appE_gap')[1]
    rows = [r['cells'] for r in A.tabular_rows(gl, 'tab:appE_gap')[0]]
    col = A.COLS['E.24'].index('Resid.')
    cells = [r[col] for r in rows if r[col] not in ('---', '')]
    dps = Counter(_dp(x) for x in cells)
    return _dval(3, f'generator E.24 (= Table S6) Resid. column, {len(cells)} cells: decimal places {dict(dps)}',
                 consistent=set(dps) == {3})


def build_X(head_lines):
    X, inputs, problems = H.build_X(head_lines)
    return X, inputs, problems


def task_numbers(texts, actx):
    lines_main = texts[MAIN].split('\n')
    X, inputs, problems = build_X(lines_main)
    if problems:
        for p in problems:
            _fail(f'(1) {p}')
        return {'error': problems}, None
    gen_e21 = NB.e21_rows(actx['gen_lines'])
    w181 = NB.e21_rows(show(NB.W181A_COMMIT, MAIN).split('\n'))
    X.e21_rows_unchanged = bool(gen_e21) and [r.replace(MARK, '') for r in gen_e21] == w181
    e21_note = ('the evaluation numbers now refer to supplementary Table S4; E.21 is no longer printed. W182b\'s e21-based specs '
                'read W181a\'s committed e21_map, valid only while the E.21 rows equal cd57efc\'s: checked on the generator\'s E.21 '
                'rows (= Table S4) with the round-8 $^{b}$ marks removed -> ' + str(X.e21_rows_unchanged))
    letter_lines = texts[LETTER].split('\n')
    decls = declarations_w189p1(X, actx, letter_lines)
    per_file, all_rows, controls = {}, [], []
    stale_all, twice_all, dup_all, unassigned_all = [], [], [], []
    for f in DIFF_FILES:
        d0 = _git(['diff', '-U0', PREV_FULL, HEAD_FULL, '--', f], cwd=CLONE)
        plus = NB.added_lines(d0)
        toks = tokens_of(f, texts[f], plus)
        lines = texts[f].split('\n')
        stale, twice, decl_out, dup = my_apply(X, toks, decls[f], lines)
        unassigned = [{'file': f, 'line': t['line'], 'col': t['col'], 'written': t['written']} for t in toks if t['status'] is None]
        stale_all += stale
        twice_all += twice
        dup_all += [(f, x) for x in dup]
        unassigned_all += unassigned
        for t in toks:
            t['rating'] = NB.RATING.get(t['status'], 'UNRATED')
            t['convention'] = NB.convention(t)
        ctl = my_negative_control(X, toks, lines)
        controls += ctl
        keep = ('file', 'line', 'col', 'written', 'type', 'status', 'rating', 'category', 'counterpart', 'counterpart_value',
                'value_at_written_precision', 'rule', 'note', 'convention')
        rows = []
        for t in toks:
            r = {k: (t.get(k) if not isinstance(t.get(k), float) or t.get(k) == t.get(k) else None) for k in keep}
            sp = t.get('spec') or {}
            r['spec_desc'] = sp.get('desc') or sp.get('why') or (str(sp.get('src')) if sp else None)
            r['line_text'] = ws(lines[t['line'] - 1])[:160]
            rows.append(r)
        all_rows += rows
        per_file[f] = {'plus_lines': sorted(plus), 'n_plus_lines': len(plus), 'n_tokens': len(toks),
                       'by_rating': dict(Counter(r['rating'] for r in rows)),
                       'not_rated_categories': dict(Counter(r['category'] or 'excluded' for r in rows if r['rating'] == 'NOT RATED')),
                       'declarations': decl_out}
    non_match = [r for r in all_rows if r['rating'] in ('MISMATCH', 'UNSOURCED', 'UNRATED')]
    derived = [r for r in all_rows if r['rating'] == 'DERIVED']
    out = {'e21_numbering_note': e21_note, 'e21_rows_unchanged_marks_removed': X.e21_rows_unchanged, 'files': per_file,
           'by_rating_total': dict(Counter(r['rating'] for r in all_rows)), 'tokens': all_rows,
           'non_match': non_match, 'derived': derived,
           'stale_declarations': stale_all, 'claimed_twice': twice_all, 'duplicate_declaration_lines': dup_all,
           'unassigned': unassigned_all,
           'negative_control': {'n': len(controls), 'n_refused': sum(1 for c in controls if c['refused']),
                                'not_refused': [c for c in controls if not c['refused']]},
           'inputs': {k: {'path': v['path'], 'sha256': v['sha256']} for k, v in inputs.items()}}
    if not X.e21_rows_unchanged:
        _fail('(1) the generator E.21 rows (marks removed) differ from cd57efc: the W181a e21_map does not apply')
    if stale_all or twice_all or dup_all or unassigned_all:
        _fail(f'(1) declarations: stale {stale_all} twice {twice_all} dup {dup_all} unassigned {unassigned_all}')
    if out['negative_control']['n_refused'] != out['negative_control']['n'] or not controls:
        _fail(f"(1) negative control: {out['negative_control']['not_refused']}")
    return out, X


# ======================================================================================================================
#  (2) the W184 / W185 / G10 sentences: wording against Addendum 90 rulings 1-3
# ======================================================================================================================
def texnorm(s):
    s = s.replace('$\\pm$', '±').replace('\\pm', '±')
    s = s.replace('\\euro{}', '€').replace('\\euro', '€').replace('\\,', ' ').replace('~', ' ').replace('\\%', '%')
    s = re.sub(r'\$\s*(-?[0-9.]+)\s*\\times\s*10\^\{(-?\d+)\}\s*\$', r'\1 × 10^\2', s)
    s = s.replace('$', '').replace('{,}', ',')
    s = re.sub(r'\\textcolor\{blue\}\{', '', s)
    return ws(s).lower()


RULING1 = ('the restriction raises the coordinated cost by 20.5 k€, a determinate increase at 1.62 times its threshold, which '
           'is 0.02 % of the 90.9 m€ saving: the saving relative to the price-taker arrangement is 12.2 % with reverse flow '
           'enabled and 12.2 % with it prohibited. the coordination benefit reported here does not depend on the interface '
           'restriction')
RULING3_PARTS = ('at every certified evaluation behind the tables', 'every penalised slack in q',
                 "sits at its zero bound to within the solver's bound relaxation, so no penalty is paid in q",
                 'the relaxation itself leaves a common offset of -13.0 k€ in q (-2 × 10^-5 of q), identical across evaluations '
                 'to within 40 €, which cancels in every reported difference')
THREE_DP = (r'12\.20[0-9]', r'0\.022[0-9]?', r'20[,{}]*463', r'20\.46', r'1,572', r'1\{,\}572', r'1,452', r'1\{,\}452',
            r'682\.5', r'1\.57[0-9]?~?k', r'1\.45[0-9]?~?k')


def find_paragraph(lines, needle):
    hits = [i for i, x in enumerate(lines) if needle in x]
    return hits


def task_sentences(texts, X):
    main = texts[MAIN].split('\n')
    letter = texts[LETTER].split('\n')
    out = {}
    i44 = find_paragraph(main, 'The restriction raises the coordinated cost')
    i47 = find_paragraph(main, 'The sensitivity of the decisive comparisons to the coordination penalties')
    ib4 = find_paragraph(main, 'More generally, at every certified evaluation behind the tables')
    ibb = find_paragraph(letter, 'we also re-evaluated it with the same no-reverse-flow rule imposed on the consensus exchange')
    ir26 = find_paragraph(letter, 'were re-evaluated with every initial penalty parameter halved and doubled')
    loc = {'section 4.4 NRF sentence (main.tex)': i44, 'section 4.7 penalty paragraph (main.tex)': i47,
           'Appendix B.4 slack sentence (main.tex)': ib4, 'letter benchmark bullet': ibb, 'letter R2.6': ir26}
    out['located_lines'] = {k: [i + 1 for i in v] for k, v in loc.items()}
    if any(len(v) != 1 for v in loc.values()):
        _fail(f'(2) sentences not located exactly once: {out["located_lines"]}')
        return out
    t44, t47, tb4 = (texnorm(main[v[0]]) for v in (i44, i47, ib4))
    tbb, t26 = texnorm(letter[ibb[0]]), texnorm(letter[ir26[0]])
    A89p1, A89p2 = EX['A89']['part1_w184_section_4_4'], EX['A89']['part2_w185_spread_table']
    sub_lines = {MAIN: [main[i44[0]], main[i47[0]], main[ib4[0]]], LETTER: [letter[ibb[0]], letter[ir26[0]]]}
    three = {f: sorted({m.group(0) for ln in L for rx in THREE_DP for m in re.finditer(rx, ln)}) for f, L in sub_lines.items()}
    chk = collections.OrderedDict()
    chk['R1 section 4.4 sentence = the ruling-1 text (normalized)'] = RULING1 in t44
    chk['R1 one decimal: no three-decimal / unrounded variant in the five sentences'] = not any(three.values())
    chk['R1 NRF cost printed in kEUR (20.5 k€) in main and letter'] = '20.5 k€' in t44 and '20.5 k€' in tbb
    chk['R1 "determinate" with the record verdict determinate, sign +'] = (
        'determinate increase' in t44 and A89p1['resolution']['share_numerator_rule_v6']['verdict'] == 'determinate'
        and A89p1['resolution']['share_numerator_rule_v6']['sign'] == 1)
    chk['R1 the 90.9 M€ "saving" is the enabled gap Q(P_NRF) - Q(C_OPEN) = frozen benchmark.benefit'] = (
        A89p1['cross_checks']['gap Q(P_NRF) - Q(C_OPEN) == frozen benchmark.benefit (|diff| <= 1e-6 EUR)'] is True)
    chk['R1 order: "12.2 % with reverse flow enabled and 12.2 % with it prohibited" (enabled = 12.205, X = 12.202)'] = (
        'is 12.2 % with reverse flow enabled and 12.2 % with it prohibited' in t44)
    chk['R1 letter bullet: 20.5 k€, 0.02 % of the 90.9 m€ saving, 12.2 % benefit independent of the restriction'] = (
        'raises its cost by 20.5 k€, 0.02 % of the 90.9 m€ saving, so the 12.2 % benefit does not depend on the interface '
        'restriction' in tbb)
    newc = A89p2['new_cells']
    chk['R2 "All eight re-evaluations certified" (8 new cells, every one certified)'] = (
        'all eight re-evaluations certified' in t47 and len(newc) == 8 and all(v['status'] == 'certified' for v in newc.values()))
    chk['R2 "every initial penalty parameter halved and doubled" (rho0_scale {0.5, 2.0})'] = (
        'every initial penalty parameter halved and doubled' in t47 and {v['rho0_scale'] for v in newc.values()} == {0.5, 2.0})
    chk['R2 "at the base and at twice the base flexibility price" (prices {1x, 2x})'] = (
        'at the base and at twice the base flexibility price' in t47 and {v['price'] for v in newc.values()} == {'1x', '2x'})
    chk['R2 "every sign and verdict was unchanged" (sign_kept and class_kept at both prices)'] = (
        'every sign and verdict was unchanged' in t47 and all(A89p2['spread'][p]['sign_kept'] is True and
                                                             A89p2['spread'][p]['class_kept'] is True for p in ('1x', '2x')))
    chk['R2 "seven of the eight ... inside the acceptance window ... the eighth lay 0.7 k€ above it"'] = (
        'seven of the eight re-evaluations fell inside the acceptance window of the corresponding frozen evaluation and the '
        'eighth lay 0.7 k€ above it' in t47 and
        sorted(v['inside_frozen_window'] for v in A89p2['P2_cells'].values()) == [False] + [True] * 7)
    out['p2_outside_cell'] = [k for k, v in A89p2['P2_cells'].items() if not v['inside_frozen_window']]
    out['p2_outside_above'] = [v['Q_new'] > v['window_frozen'][1] for v in A89p2['P2_cells'].values() if not v['inside_frozen_window']]
    chk['R2 the eighth lies ABOVE its window (Q_new > hi)'] = out['p2_outside_above'] == [True]
    chk['R2 the "operating-cost difference" spread = the V spread (I constant)'] = all(
        abs(A89p2['spread'][p]['V_spread'] - A89p2['spread'][p]['spread']) < 1e-6 for p in ('1x', '2x'))
    chk['R2 initialisation caveat in section 4.7 (penalty schedule, not the start; initialisation; limitation stands)'] = (
        'the check covers the penalty schedule, not the start of the coordination' in t47 and 'initialisation' in t47
        and 'the limitation below stands' in t47)
    chk['R2 letter R2.6: eight evaluations, all certified; 1.6 / 1.5 k€; about 0.12; signs and verdicts unchanged; '
        'penalty schedule not initialisation'] = (
        'eight evaluations, all certified' in t26 and 'moved by 1.6 k€ and 1.5 k€ respectively across the three penalty '
                                                      'settings, about 0.12 of its determinacy threshold' in t26
        and 'with every sign and verdict unchanged' in t26 and 'the check covers the penalty schedule and not the '
                                                               'initialisation of the coordination' in t26)
    chk['R2 the penalty paragraph does not mention the monotone-branch cell (ruling 2: no main-text sentence)'] = \
        'monotone' not in t47
    for p in RULING3_PARTS:
        chk[f'R3 G10 wording part present: "{p[:60]}..."'] = p in tb4
    chk['R3 exception 1: the multi-scenario reference unit, at most 2.2 €'] = (
        'the multi-scenario evaluation of the reference unit carries a positive flexibility-balance content of at most 2.2 €'
        in tb4)
    chk['R3 exception 2: the superseded evaluation 16, at most 171 € of voltage-slack content'] = (
        'the superseded evaluation 16 a positive voltage-slack content of at most 171 €' in tb4)
    out['checks'] = dict(chk)
    # statements in the G10 parenthetical against W188a's code reading (not a ruling-3 part)
    fam = 'daily closure, network feasibility and the agent\'s net-power pair'
    esso_in_q = [r for r in EX['W188A']['C_code_anchors'].keys() if r.startswith('esso')]
    paren = {'printed': fam, 'present': fam in tb4,
             'w188a_reading': 'W188a "Which penalised slacks enter Q": ESSO sigma+- = slack_es_pnet_up/down is penalised in the '
                              'ESSO feasibility_penalty, which is never summed into Q ("In Q: no"; "sigma+- has 0 EUR in Q by '
                              'construction"); closure s+- and the network-feasibility slacks (voltage, flexibility P '
                              'day-balance) are in Q',
             'w188a_code_anchors_esso': esso_in_q,
             'agent_pair_listed_as_a_slack_in_Q': fam in tb4,
             'rating': 'STATEMENT DIFFERS FROM W188a' if fam in tb4 else 'n/a'}
    out['g10_parenthetical'] = paren
    v = EX['W188A']['cells']['pb_y2025_n5']
    out['g10_171_attribution'] = {
        'printed': 'a positive voltage-slack content of at most 171 €',
        'row_total_bound_eur': v['score']['positive_content_bound_eur_in_Q_refined'],
        'voltage_part_bound_eur': v['families']['voltage_slack']['positive_part_bound_eur_weighted'],
        'closure_part_bound_eur': v['families']['shared_ess_day_balance_slack']['positive_part_bound_eur_weighted'],
        'reading': '171 € is the row total (voltage + closure) rounded up; the voltage-slack part alone is <= 169.27 € (170 € '
                   'rounded up). The printed bound is true for the voltage content; it is the same figure as Addendum 90 ruling 3 '
                   '("superseded evaluation 16 <= 171 € of voltage-slack content")'}
    out['g10_minus13_rounding'] = {'per_row_1dp': dict(Counter(f"{r['signed_eur_in_Q_total'] / 1e3:.1f}" for r in w188a_named_certified())),
                                   'note': 'one of the 20 certified rows (c_156ce2d1, -13,052.36 EUR) reads -13.1 kEUR at 1 dp; the '
                                           'sentence states the offset with "to within 40 EUR"'}
    out['three_decimal_hits'] = three
    out['holds'] = all(chk.values())
    out['texts'] = {'section_4_4': ws(main[i44[0]])[:2000], 'section_4_7': ws(main[i47[0]])[:2000],
                    'appendix_B4': ws(main[ib4[0]])[:2000], 'letter_benchmark_bullet': ws(letter[ibb[0]])[:2000],
                    'letter_R2_6': ws(letter[ir26[0]])[:2400]}
    return out


def controls_sentences(texts):
    out = []
    main = texts[MAIN]
    t2 = main.replace('All eight re-evaluations certified.', 'All eight re-evaluations certified, as P2 predicted.')
    hits = predict_scan({MAIN: t2})
    out.append({'id': '2-1 a planted "predicted" beside the penalty paragraph is found and attributed to P1/P2/P5',
                'refused': any(h['applied_to_P1_P2_P5'] for h in hits)})
    t3 = main.replace('is 12.2\\,\\% with reverse flow enabled', 'is 12.205\\,\\% with reverse flow enabled')
    three = sorted({m.group(0) for rx in THREE_DP for m in re.finditer(rx, t3)})
    out.append({'id': '2-2 a planted 12.205 % is caught by the one-decimal check', 'hits': three,
                'refused': any('12.20' in x for x in three)})
    return out


# ======================================================================================================================
#  (3) Figure 6
# ======================================================================================================================
def pdf_streams(b):
    objs = {int(mm.group(1)): mm.group(2) for mm in re.finditer(rb'(\d+) 0 obj\s*(\d+)\s*endobj', b)}
    out = []
    for mm in re.finditer(rb'<<(.*?)>>\s*stream\r?\n', b, re.S):
        d, s = mm.group(1), mm.end()
        lm = re.search(rb'/Length (\d+)( 0 R)?', d)
        if not lm:
            continue
        n = int(objs[int(lm.group(1))]) if lm.group(2) else int(lm.group(1))
        raw = b[s:s + n]
        if b'/FlateDecode' in d:
            raw = zlib.decompress(raw)
        out.append((d, raw))
    return out


NUMR = r'-?\d+(?:\.\d+)?'


def pdf_paths(b):
    c = max(pdf_streams(b), key=lambda x: x[1].count(b' l\n'))[1].decode('latin1')
    clip, out, cur = None, [], None
    for ln in c.split('\n'):
        mm = re.search('q (' + NUMR + ') (' + NUMR + ') (' + NUMR + ') (' + NUMR + ') re W n', ln)
        if mm:
            clip = tuple(float(x) for x in mm.groups())
        m1 = re.fullmatch('(' + NUMR + ') (' + NUMR + ') m', ln.strip())
        m2 = re.fullmatch('(' + NUMR + ') (' + NUMR + ') l', ln.strip())
        if m1:
            if cur and len(cur) > 1:
                out.append((clip, cur))
            cur = [(float(m1.group(1)), float(m1.group(2)))]
        elif m2 and cur is not None:
            cur.append((float(m2.group(1)), float(m2.group(2))))
        else:
            if cur and len(cur) > 1:
                out.append((clip, cur))
            cur = None
    return out


def rel_paths(paths):
    cnt = Counter()
    for clip, pts in paths:
        if clip is None:
            continue
        cnt[(round(clip[2], 4), round(clip[3], 4), tuple((round(x - clip[0], 4), round(y - clip[1], 4)) for x, y in pts))] += 1
    return cnt


def pdf_data_check(pdf_bytes, soc, soh, tol=2e-5):
    """panel (a): eight 24-point polylines = the TSO-copy SoC of (year, day), x = (h-1)/23 * w, y = soc/0.85 * h_ax;
    panel (b): seven 4-point polylines = [1] + SoH end-of-block of each calibration, x over [2024.5, 2050], y over [0.45, 1.03]."""
    paths = pdf_paths(pdf_bytes)
    a_clips = sorted({cl for cl, pts in paths if cl and len(pts) == 24}, key=lambda c: c[0])
    res = {'panel_a_axes': len(a_clips), 'panel_a_matched': [], 'panel_b_matched': [], 'unmatched': []}
    days = ['Spring', 'Summer', 'Autumn', 'Winter']
    for k, cl in enumerate(a_clips):
        x0, y0, w, h = cl
        for clp, pts in paths:
            if clp != cl or len(pts) != 24:
                continue
            best = None
            for yr in ('2025', '2035'):
                rows = sorted([r for r in soc if r['day'] == days[k] and r['year'] == yr], key=lambda r: int(r['hour_ending']))
                exp = [((int(r['hour_ending']) - 1) / 23 * w, float(r['soc_tso_copy_mwh']) / 0.85 * h) for r in rows]
                err = max(max(abs(px - x0 - ex), abs(py - y0 - ey)) for (px, py), (ex, ey) in zip(pts, exp))
                if best is None or err < best[1]:
                    best = ((days[k], yr), err)
            (res['panel_a_matched'] if best[1] <= tol * max(w, h) * 10 else res['unmatched']).append(
                {'series': best[0], 'max_abs_err_pt': best[1]})
    b_clips = sorted({cl for cl, pts in paths if cl and len(pts) == 4 and cl not in a_clips}, key=lambda c: -c[2])
    cal = sorted({r['calibration_table6'] for r in soh})
    if b_clips:
        x0, y0, w, h = b_clips[0]
        xs = [2025, 2030, 2035, 2040]
        for clp, pts in paths:
            if clp != b_clips[0] or len(pts) != 4:
                continue
            best = None
            for nm in cal:
                rr = sorted([r for r in soh if r['calibration_table6'] == nm], key=lambda r: int(r['block_year']))
                ys = [1.0] + [float(r['soh_end_of_block']) for r in rr]
                exp = [((x - 2024.5) / (2050 - 2024.5) * w, (y - 0.45) / (1.03 - 0.45) * h) for x, y in zip(xs, ys)]
                err = max(max(abs(px - x0 - ex), abs(py - y0 - ey)) for (px, py), (ex, ey) in zip(pts, exp))
                if best is None or err < best[1]:
                    best = (nm, err)
            (res['panel_b_matched'] if best[1] <= 1e-3 else res['unmatched']).append({'series': best[0], 'max_abs_err_pt': best[1]})
    res['panel_a_series'] = sorted(tuple(x['series']) for x in res['panel_a_matched'])
    res['panel_b_series'] = sorted(x['series'] for x in res['panel_b_matched'])
    res['holds'] = (len(res['panel_a_matched']) == 8 and len(set(res['panel_a_series'])) == 8 and
                    len(res['panel_b_matched']) == 7 and len(set(res['panel_b_series'])) == 7 and not res['unmatched'])
    return res


def task_figure(out_dir, texts):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    clone_pdf = _rd(os.path.join(CLONE, FIGPDF))
    rev_pdf = _rd(REVIEW_COPY[FIGPDF])
    out = {'clone_pdf_sha256': _shab(clone_pdf), 'review_pdf_sha256': _shab(rev_pdf), 'clone_equals_review_copy': clone_pdf == rev_pdf,
           'clone_pdf_equals_head_blob': clone_pdf == showb(HEAD_FULL, FIGPDF),
           'pdf_metadata': {k: (re.search(rb'/' + k.encode() + rb' \((.*?)\)', clone_pdf).group(1).decode('latin1')
                                if re.search(rb'/' + k.encode() + rb' \((.*?)\)', clone_pdf) else None)
                            for k in ('Creator', 'Producer', 'CreationDate')}}
    # the script: static inspection, then the re-run into the output directory
    src = _rd(FIG_PY).decode('utf-8')
    tree = ast.parse(src)
    imports = sorted({(n.module if isinstance(n, ast.ImportFrom) else a.name).split('.')[0]
                      for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                      for a in (n.names if isinstance(n, ast.Import) else [n])})
    opens = re.findall(r"open\('([^']+)'\)", src)
    saves = re.findall(r"savefig\('([^']+)'", src)
    out['script'] = {'path': FIG_PY, 'sha256': _sha(FIG_PY), 'committed_clean': committed_clean(FIG_PY), 'imports': imports,
                     'reads': opens, 'writes': saves, 'zero_solve_by_inspection': set(imports) <= {'csv', 'collections', 'matplotlib'},
                     'reads_are_the_three_csvs': sorted(opens) == sorted(FIG_CSVS),
                     'writes_relative_to_cwd_only': all('/' not in s for s in saves)}
    fd = os.path.join(od, OUT_NAMES['fig'])
    os.mkdir(fd)
    for f in FIG_CSVS:
        shutil.copyfile(os.path.join(REPO, W188D_DIR, f), os.path.join(fd, f))
    mpl = os.path.join(fd, 'mplconfig')
    os.mkdir(mpl)
    watch = [os.path.join(REPO, d, FIGPDF[:-4] + e) for d in ('', MR, CLONE, W188D_DIR) for e in ('.pdf', '.png')]
    snap = lambda: {os.path.relpath(w, REPO): (_sha(w) if os.path.isfile(w) else None) for w in watch}  # noqa: E731
    before = snap()
    rev_before = (_sha(REVIEW_COPY[FIGPDF]), _sha(FIG_PY))
    ok_script = out['script']['zero_solve_by_inspection'] and out['script']['reads_are_the_three_csvs'] and \
        out['script']['writes_relative_to_cwd_only']
    if ok_script:
        env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        env['MPLCONFIGDIR'] = mpl
        r = subprocess.run([sys.executable, '-I', os.path.join(REPO, FIG_PY)], cwd=fd, env=env, capture_output=True, text=True)
        out['rerun'] = {'cwd': os.path.relpath(fd, REPO) if fd.startswith(REPO) else fd, 'interpreter': sys.executable,
                        'returncode': r.returncode, 'stdout': r.stdout.strip()[:600], 'stderr': r.stderr.strip()[:600]}
    else:
        out['rerun'] = {'skipped': 'the script is not zero-solve / CSV-only / cwd-only by inspection'}
        _fail('(3) figure script not run: inspection failed')
    after = snap()
    out['watched_paths_before_after'] = {'before': before, 'after': after}
    out['repo_status_unchanged_by_rerun'] = before == after
    out['manuscript_review_unchanged'] = rev_before == (_sha(REVIEW_COPY[FIGPDF]), _sha(FIG_PY))
    new_files = sorted(os.path.relpath(os.path.join(dp, f), fd) for dp, _d, fs in os.walk(fd) for f in fs)
    out['rerun_files'] = [x for x in new_files if not x.startswith('mplconfig')]
    rp = os.path.join(fd, FIGPDF)
    if os.path.isfile(rp):
        rr = _rd(rp)
        a, b = rel_paths(pdf_paths(clone_pdf)), rel_paths(pdf_paths(rr))
        diff_a, diff_b = a - b, b - a
        out['rerun_pdf_sha256'] = _shab(rr)
        out['rerun_pdf_bitwise_equal_clone'] = rr == clone_pdf
        out['rerun_pdf_metadata'] = {k: (re.search(rb'/' + k.encode() + rb' \((.*?)\)', rr).group(1).decode('latin1')
                                         if re.search(rb'/' + k.encode() + rb' \((.*?)\)', rr) else None)
                                     for k in ('Creator', 'Producer', 'CreationDate')}
        mx = 0.0
        for ka in diff_a:
            cands = [kb for kb in diff_b if kb[:2] == ka[:2] and len(kb[2]) == len(ka[2])]
            if cands:
                mx = max(mx, min(max(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p, q in zip(ka[2], kb[2])) for kb in cands))
        out['polylines_axes_relative'] = {'n_clone': sum(a.values()), 'n_rerun': sum(b.values()),
                                          'n_only_clone': sum(diff_a.values()), 'n_only_rerun': sum(diff_b.values()),
                                          'only_clone': [{'axes': list(k[:2]), 'n_points': len(k[2]), 'first': k[2][:3]} for k in diff_a],
                                          'only_rerun': [{'axes': list(k[:2]), 'n_points': len(k[2]), 'first': k[2][:3]} for k in diff_b],
                                          'max_offset_pt_of_the_differing': mx}
        out['rerun_data_check'] = pdf_data_check(rr, EX['SOC'], EX['SOH'])
    else:
        _fail('(3) the figure re-run produced no PDF')
    out['clone_pdf_data_check'] = pdf_data_check(clone_pdf, EX['SOC'], EX['SOH'])
    # statements of the paragraph / caption
    sums = {(r['year'], r['day']): r for r in EX['SOCSUM']}
    cyc = {f'{y} {d}': float(r['energy_charged_tso_mwh']) / (float(EX['LIM'][y]['soc_upper_90pct_record_mwh'])
                                                           - float(EX['LIM'][y]['soc_lower_10pct_record_mwh']))
           for (y, d), r in sums.items()}
    contacts = {f'{y} {d}': (int(r['n_contact_lower_tso']), int(r['n_contact_upper_tso']),
                             float(r['soc_min_tso_mwh']), float(r['soc_max_tso_mwh'])) for (y, d), r in sums.items()}
    floor = {a: [r['block_year'] for r in soh_rows(a) if r['floor_row_active_at_terminal_flag'] == '1']
             for a in ('C2_calfade', 'C2', 'C3_unit', 'C3_midblock', 'C4', 'no_ageing', 'soh050')}
    st = collections.OrderedDict()
    st['"cycles once or twice a day": charged energy / usable band per block in [1, 2]'] = {
        'value': {k: round(v, 3) for k, v in cyc.items()}, 'holds': all(0.95 <= v <= 2.05 for v in cyc.values())}
    st['"between the 10 % and 90 % limits": every block touches both limits'] = {
        'value': contacts, 'holds': all(v[0] > 0 and v[1] > 0 for v in contacts.values()),
        'note': 'contacts within 1e-4 MWh (W188d rule); the two Winter days touch only the upper limit (minimum SoC 0.1222 MWh '
                'in 2025 = 13.9 % of E^Av, 0.1078 MWh in 2035 = 15.4 %)'}
    base = {r['block_year']: float(r['e_available_mwh']) for r in soh_rows('C2_calfade')}
    lim_ok = {y: (EX['LIM'][y]['row'], float(EX['LIM'][y]['e_available_record_mwh']), base.get(y)) for y in ('2025', '2035')}
    st['panel (a) is the baseline reference unit: the SoC export (E.21 row 2) has the E^Av of the baseline calibration '
       '(e_c2_calfade) in 2025 and 2035, bitwise'] = {
        'value': lim_ok, 'holds': all(v[0] == '2' and v[1] == v[2] for v in lim_ok.values()),
        'note': 'W188d also reports the e_c2_calfade SoH trajectory = W101 row 2 (3f084f2f) exactly (not re-checked here)'}
    st['"the baseline and both 50 % retention calibrations reach the floor in the 2035 block"'] = {
        'value': floor, 'holds': floor['C2_calfade'] == floor['C3_unit'] == floor['C3_midblock'] == ['2035']}
    st['"the cycling-only and datasheet calibrations do not"'] = {'value': {k: floor[k] for k in ('C2', 'C4')},
                                                                  'holds': floor['C2'] == [] and floor['C4'] == []}
    st['"the 0.50 floor lets the baseline calibration continue to 0.68" (no floor row active)'] = {
        'value': {'soh_end_2035': soh050_end(None), 'floor': floor['soh050']}, 'holds': floor['soh050'] == [] and
        f'{soh050_end(None):.2f}' == '0.68'}
    ml = texts[MAIN].split('\n')
    il = next(i for i, x in enumerate(ml) if '\\label{tab:ageing_calibrations}' in x)
    ib = max(i for i in range(il) if '\\begin{table}' in ml[i])
    tab3 = [x for x in ml[ib:il] if ' & ' in x and x.rstrip().endswith('\\\\') and not x.strip().startswith('Calibration')]
    st['"the seven calibrations" = the seven rows of Table tab:ageing_calibrations; the last is the 0.50 floor'] = {
        'value': {'csv_calibrations': n_calibrations(None), 'table_rows': len(tab3[:7]),
                  'last_row': ws(tab3[6])[:80] if len(tab3) >= 7 else None},
        'holds': n_calibrations(None) == 7 and len(tab3) >= 7 and tab3[6].strip().startswith('Baseline, 0.50 floor')}
    st['"(b) state of health at the end of each representative block" (the plotted series = soh_end_of_block)'] = {
        'value': 'clone PDF panel (b) polylines matched to [1] + soh_end_of_block of each calibration',
        'holds': out['clone_pdf_data_check']['holds']}
    out['statements'] = dict(st)
    out['holds'] = (out['clone_equals_review_copy'] and out['clone_pdf_data_check']['holds'] and out['repo_status_unchanged_by_rerun']
                    and out['manuscript_review_unchanged'])
    if not (out['repo_status_unchanged_by_rerun'] and out['manuscript_review_unchanged']):
        _fail('(3) the figure re-run changed something outside the output directory')
    return out


def controls_figure(texts):
    soc = [dict(r) for r in EX['SOC']]
    k = next(i for i, r in enumerate(soc) if r['year'] == '2035' and r['day'] == 'Summer' and r['hour_ending'] == '12')
    soc[k]['soc_tso_copy_mwh'] = str(float(soc[k]['soc_tso_copy_mwh']) + 0.01)
    res = pdf_data_check(_rd(os.path.join(CLONE, FIGPDF)), soc, EX['SOH'])
    return [{'id': '3-1 one SoC value perturbed by 0.01 MWh: the clone PDF no longer matches the CSV',
             'unmatched': res['unmatched'], 'refused': not res['holds']}]


# ======================================================================================================================
#  (4) ruling 6 disclosure
# ======================================================================================================================
def disclosure_scan(letter_text):
    L = letter_text.split('\n')
    hits = [i for i, x in enumerate(L, 1) if '250' in x and '100' in x and 'node~7' in x and 'case file' in x]
    return hits, L


def task_disclosure(texts):
    hits, L = disclosure_scan(texts[LETTER])
    zb = _rd(SUPP_ZIP)
    with zipfile.ZipFile(io.BytesIO(zb)) as z:
        rows = list(csv.DictReader(io.StringIO(z.read('appendices_B_D/D_ieee33_tn_connection.csv').decode('utf-8'))))
    per_net = collections.defaultdict(set)
    for rel, c in EX['CASE33_REF'].items():
        for g in c['ref_generators']:
            per_net[c['net']].add((g['Pmax'], g['Pmin'], g['Qmax'], g['Qmin'], g['bus']))
    out = {'letter_lines': hits, 'letter_text': ws(L[hits[0] - 1]) if hits else None,
           'case_files': {k: sorted(v) for k, v in per_net.items()},
           'case_files_years': sorted({c['year'] for c in EX['CASE33_REF'].values()}),
           'n_case_files': len(EX['CASE33_REF']),
           'a87e195_table_row': re.findall(r'1 & 1 & 250\.00 & -250\.00 & 250\.00 & -250\.00 & 1\.00', show(PREV_FULL, MAIN)),
           'submitted_table_row': re.findall(r'1 & 1 & 250\.00 & -250\.00 & 250\.00 & -250\.00 & 1\.00', EX['SUBMITTED']),
           'head_main_has_tn_connection_table': 'TN connection data' in texts[MAIN],
           'supplement_v1_zip': {'path': SUPP_ZIP, 'sha256': _shab(zb), 'D_ieee33_tn_connection.csv': rows}}
    t = texnorm(L[hits[0] - 1]) if hits else ''
    chk = {'letter v12 states the submitted table printed +-250 for every network': 'printed ±250 mw/mvar for every network' in t,
           'letter v12 states node 7 at +-100 in the case file every reported run used': '±100 mw/mvar in the case file that every '
           'reported run used' in t,
           'case33_2 (node 7) REF generator +-100 in every year file': per_net['case33_2'] == {(100, -100, 100, -100, 1)},
           'case33_1 and case33_3 REF generator +-250 in every year file': per_net['case33_1'] == per_net['case33_3'] == {(250, -250, 250, -250, 1)},
           'a87e195 and submitted Appendix D printed one +-250 row': len(out['a87e195_table_row']) == 1 and len(out['submitted_table_row']) == 1,
           'the table has left the paper at HEAD': not out['head_main_has_tn_connection_table'],
           'supplement v1 CSV: one row per ADN, node 7 at +-100, nodes 5 and 9 at +-250': sorted(
               (r['adn_node'], r['p_max_MW'], r['p_min_MW'], r['q_max_MVAr'], r['q_min_MVAr']) for r in rows) == [
               ('5', '250', '-250', '250', '-250'), ('7', '100', '-100', '100', '-100'), ('9', '250', '-250', '250', '-250')]}
    out['raw_letter_line'] = L[hits[0] - 1].strip() if hits else None
    out['checks'] = chk
    out['holds'] = all(chk.values()) and len(hits) == 1
    return out


def controls_disclosure(texts):
    L = texts[LETTER].split('\n')
    hits, _ = disclosure_scan(texts[LETTER])
    L2 = [x for i, x in enumerate(L, 1) if i not in hits]
    h2, _ = disclosure_scan('\n'.join(L2))
    return [{'id': '4-1 the disclosure line removed from a copy of the letter is not found', 'refused': h2 == [] and len(hits) == 1}]


# ======================================================================================================================
#  (5) "predict"
# ======================================================================================================================
P_CTX = re.compile(r'\bP[125]\b|penalt|reverse flow|reverse-flow|no-reverse|NRF|penalised slack|slack|interface restriction|'
                   r'initial penalty|re-evaluat', re.I)


def predict_scan(texts):
    hits = []
    for f, t in texts.items():
        for i, ln in enumerate(t.split('\n'), 1):
            for mm in re.finditer(r'predict\w*', ln, re.I):
                a, b = max(0, mm.start() - 160), min(len(ln), mm.end() + 160)
                ctx = ln[a:b]
                comment = comment_start(ln) is not None and mm.start() >= comment_start(ln)
                hits.append({'file': f, 'line': i, 'word': mm.group(0), 'context': ws(ctx), 'in_comment': comment,
                             'applied_to_P1_P2_P5': bool(re.search(r'\bP[125]\b', ctx)) or (
                                 bool(P_CTX.search(ctx)) and bool(re.search(r'penalt|reverse|NRF|slack', ctx, re.I))
                                 and not comment)})
    return hits


def task_predict(texts):
    t = dict(texts)
    t[HL] = _rd(os.path.join(CLONE, HL)).decode('utf-8')
    hits = predict_scan(t)
    out = {'files_scanned': sorted(t), 'n_hits': len(hits), 'hits': hits,
           'n_applied_to_P1_P2_P5': sum(1 for h in hits if h['applied_to_P1_P2_P5']),
           'rule': 'a hit is attributed to P1 / P2 / P5 when "P1"/"P2"/"P5" is within 160 characters, or when the context names '
                   'the penalty pilot, the reverse-flow restriction or the slacks; every hit is listed with its context for '
                   'reading'}
    out['holds'] = out['n_applied_to_P1_P2_P5'] == 0
    return out


# ======================================================================================================================
#  capture checklist, guards, md, post-run, main
# ======================================================================================================================
REQUIRED_CAPTURES = {'(0) state': 'task_state', '(1) numbers': 'task_numbers', '(1b) reduced appendix': 'task_appendix',
                     '(2) sentences': 'task_sentences', '(3) figure 6': 'task_figure', '(4) disclosure': 'task_disclosure',
                     '(5) predict': 'task_predict', 'guards': 'verify_guards'}
REQUIRED_SYMBOLS = {'W181a': (A, ('run_generator', 'run_verifier', 'locate_bytes', 'appendix_block', 'parse_e20', 'parse_e21',
                                  'check_e20', 'check_e21', 'check_e22', 'tabular_rows', 'row_cells', 'table_body', 'strip_markers',
                                  'parse_num', 'dec_places', 'TABLES', 'COLS')),
                    'W183h': (H, ('build_X', 'carried', 'declarations_w183h', 'frozen_ctx', 'all_guards')),
                    'W182b': (NB, ('declarations', 'added_lines', 'split_coefficients', 'evaluate', 'perturb', 'RATING', 'convention',
                                   'e21_rows', 'W181A_COMMIT', 'S', 'U', 'SF', 'DER', 'SINGLE', 'LANG', 'NOT', 'P', 'W3', 'CONV',
                                   'TWO_PCT', 'C3X', 'der_pct_price_taker', '_dn_by_node')),
                    'W177b-1': (B1, ('declarations', 'der_meur', 'UNIT', 'REF4', 'BM', 'AF', 'C57', '_twelve')),
                    'W177b-1prime': (B1R, ('SUBF', '_op_scen')),
                    'W177a': (W177A, ('jget', '_cl', '_written_value', 'IDENT_TYPES'))}


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
    need = [os.path.join(CLONE, f) for f in DIFF_FILES + (FIGPDF, HL)] + list(REVIEW_COPY.values()) + [
        GEN_REL, VERIFY_REL, APPX_REL, FIG_PY, FZ_REL, A89_JSON, W188A_JSON, W188D_MAN, MARKET_XLSX, SUPP_ZIP, SUBMITTED_MAIN,
        CODE_SRP, BRIEF, STEP6_R9] + [os.path.join(W188D_DIR, f) for f in FIG_CSVS + (SOC_SUMMARY_CSV,)] + list(REPORTS)
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    for k in ('gen', 'verify_tmp', 'fig'):
        tgt = os.path.realpath(os.path.join(od, OUT_NAMES[k]))
        if not tgt.startswith(os.path.realpath(od) + os.sep):
            missing.append(f'{k} target not inside the output directory')
    if os.path.realpath(os.path.join(od, OUT_NAMES['gen'])) in (os.path.realpath(os.path.join(REPO, APPX_REL)),
                                                                os.path.realpath(os.path.join(REPO, MR, 'appE.tex'))):
        missing.append('generator target is a manuscript_review file')
    if missing:
        for x in missing:
            print(f'REFUSED (capture-path checklist): missing {x}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def verify_guards():
    gs = [('w189p1', GUARD)] + [(nm, g) for nm, g in H.all_guards()]
    seen, lst = set(), []
    for nm, g in gs:
        if id(g) not in seen:
            seen.add(id(g))
            lst.append((nm, g))
    g = {nm: {'counts': dict(gd.counts), 'verify_0_failures': gd.verify(0)} for nm, gd in lst}
    pk = {'w189p1': dict(PICKLE_COUNTS)}
    for nm, mod in (('w183h', H), ('w182c', H.W), ('w181a', A), ('w182b', NB), ('w178b', H.W178B), ('w177b1r', B1R)):
        if hasattr(mod, 'PICKLE_COUNTS'):
            pk[nm] = dict(mod.PICKLE_COUNTS)
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads and pickle.Unpickler is _BlockedUnpickler
    ok = all(not v['verify_0_failures'] for v in g.values()) and blocked and all(not any(v.values()) for v in pk.values())
    if not ok:
        _fail(f'guards: {g} {pk} blocked {blocked}')
    return {'guards': g, 'n_guards': len(g), 'pickle_counts': pk, 'pickle_blocked_at_end': blocked, 'ok': ok}


def _cell(s, n=200):
    return str(s).replace('|', '\\|').replace('\n', ' ')[:n]


def write_md(out, path):
    N, Ap, Se, Fi, Di, Pr = (out['numbers'], out['appendix'], out['sentences'], out['figure'], out['disclosure'], out['predict'])
    L = [f'# W189 part 1 -- numbers and claims at Overleaf {HEAD_FULL[:7]} (generated by {SCRIPT_NAME}; do not edit)', '',
         'Objective convention: no objective value is computed; every rated monetary token carries its convention (Q = '
         'gross_operational_cost, contracted interface settlement excluded, unless stated).', '',
         f"Check failures: {len(out['check_failures'])}; guards ok {out['guards']['ok']}.", '',
         '## (1) numbers', '', f"By rating: {N.get('by_rating_total')}", '']
    for f, v in N.get('files', {}).items():
        L.append(f"- {f}: {v['n_plus_lines']} '+' lines, {v['n_tokens']} tokens, {v['by_rating']}; NOT RATED {v['not_rated_categories']}")
    L += ['', f"Negative control: {N.get('negative_control', {}).get('n_refused')} / {N.get('negative_control', {}).get('n')} refused.",
          '', '### Non-MATCH (MISMATCH / UNSOURCED / UNRATED)', '']
    if N.get('non_match'):
        L += ['| file | line | written | rating | counterpart | value at precision |', '|---|---:|---|---|---|---|']
        L += [f"| {r['file']} | {r['line']} | {r['written']} | {r['rating']} | {_cell(r['counterpart'])} | "
              f"{_cell(r['value_at_written_precision'], 60)} |" for r in N['non_match']]
    else:
        L.append('none')
    L += ['', '### DERIVED', '', '| file | line | written | arithmetic |', '|---|---:|---|---|']
    L += [f"| {r['file']} | {r['line']} | {r['written']} | {_cell(r['counterpart'], 300)} |" for r in N.get('derived', [])]
    L += ['', '### Every rated token', '', '| file | line | written | rating | counterpart | at precision | convention |',
          '|---|---:|---|---|---|---|---|']
    L += [f"| {r['file']} | {r['line']} | {r['written']} | {r['rating']} | {_cell(r['counterpart'], 160)} | "
          f"{_cell(r['value_at_written_precision'], 50)} | {_cell(r['convention'], 40)} |"
          for r in N.get('tokens', []) if r['rating'] != 'NOT RATED']
    L += ['', '### NOT RATED', '', '| file | line | written | category |', '|---|---:|---|---|']
    L += [f"| {r['file']} | {r['line']} | {r['written']} | {_cell(r['category'] or r.get('note'), 100)} |"
          for r in N.get('tokens', []) if r['rating'] == 'NOT RATED']
    L += ['', '## (1b) reduced appendix', '', f"Outcome: {Ap['outcome']}", '', '| generator lines | HEAD lines | op | generator | HEAD |',
          '|---|---|---|---|---|']
    L += [f"| {h['generator_lines']} | {h['head_lines']} | {h['op']} | {_cell(h['generator_first'], 140)} | {_cell(h['head_first'], 140)} |"
          for h in Ap['bitwise']['diff_hunks_generator_to_head_block']]
    L += ['', f"Entry check: {Ap['entry_check_head']['summary']}", '', f"Rows compared: {Ap['rows_compared']}", '',
          f"Main-text citations of tab:appE_claims: {Ap['main_text_citations_of_tab_appE_claims']}", '',
          '## (2) sentences', '']
    L += [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in Se.get('checks', {}).items()]
    L += ['', f"G10 parenthetical: {Se.get('g10_parenthetical')}", '', f"171 attribution: {Se.get('g10_171_attribution')}", '',
          '## (3) figure 6', '', f"clone = review copy {Fi['clone_equals_review_copy']}; clone data check {Fi['clone_pdf_data_check']['holds']}; "
          f"re-run bitwise {Fi.get('rerun_pdf_bitwise_equal_clone')}; polylines {Fi.get('polylines_axes_relative', {}).get('n_only_clone')} "
          f"differing (max offset {Fi.get('polylines_axes_relative', {}).get('max_offset_pt_of_the_differing')} pt)", '']
    L += [f"- {'HOLDS' if v['holds'] else 'DOES NOT HOLD'}: {k} -- {_cell(v['value'], 300)}" for k, v in Fi['statements'].items()]
    L += ['', '## (4) disclosure', ''] + [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in Di['checks'].items()]
    L += ['', '## (5) predict', '', '| file | line | word | applied to P1/P2/P5 | context |', '|---|---:|---|---|---|']
    L += [f"| {h['file']} | {h['line']} | {h['word']} | {h['applied_to_P1_P2_P5']} | {_cell(h['context'], 300)} |" for h in Pr['hits']]
    L += ['', '## Negative controls', ''] + [f"- {c['id']}: refused {c['refused']}" for c in out['negative_controls']]
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run(out_dir):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    p = os.path.join(od, OUT_NAMES['post'])
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = [OUT_NAMES[k] for k in ('log', 'man', 'json', 'md', 'gen', 'verify')]
    missing = [n for n in names if not os.path.isfile(os.path.join(od, n))]
    if missing:
        print(f'REFUSED: post-run inputs missing: {missing}', flush=True)
        sys.exit(1)
    extra = []
    for d in (OUT_NAMES['verify_tmp'], OUT_NAMES['fig']):
        for dp, _dd, fs in os.walk(os.path.join(od, d)):
            if 'mplconfig' in os.path.relpath(dp, od).split(os.sep):
                continue
            extra += [os.path.relpath(os.path.join(dp, f), od) for f in fs]
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
    _log(f'W189p1: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W189 part 1 (Addendum 89 W189; Addendum 90 W189 additions): numbers and claims at Overleaf '
                                    '46c9d77', 'objective_convention': 'no objective value is computed; rated numbers carry '
                                                                       'their own convention (Q = gross_operational_cost, '
                                                                       'contracted interface settlement excluded)',
           'static_reading': 'no TeX installation on this machine; nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME),
                                                      'committed_clean': committed_clean(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'interpreter': sys.executable, 'imported': {s: _sha(s) for s in IMPORTED_SCRIPTS}}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    probs = load_extras()
    out['extra_input_pins'] = {'problems': probs}
    for p in probs:
        _fail(f'(inputs) {p}')
    out['state'] = task_state()
    _log(f"(0) state holds {out['state']['holds']}")
    texts = {f: _rd(os.path.join(CLONE, f)).decode('utf-8') for f in DIFF_FILES}
    out['appendix'], actx = task_appendix(out_dir, texts)
    _log(f"(1b) {out['appendix']['outcome']}")
    out['numbers'], X = task_numbers(texts, actx)
    _log(f"(1) {out['numbers'].get('by_rating_total')}; control {out['numbers'].get('negative_control', {}).get('n_refused')}/"
         f"{out['numbers'].get('negative_control', {}).get('n')}")
    out['sentences'] = task_sentences(texts, X)
    _log(f"(2) holds {out['sentences'].get('holds')}")
    out['figure'] = task_figure(out_dir, texts)
    _log(f"(3) holds {out['figure']['holds']}")
    out['disclosure'] = task_disclosure(texts)
    _log(f"(4) holds {out['disclosure']['holds']}")
    out['predict'] = task_predict(texts)
    _log(f"(5) hits {out['predict']['n_hits']}; applied to P1/P2/P5 {out['predict']['n_applied_to_P1_P2_P5']}")
    ctl = (controls_appendix(actx) + controls_sentences(texts) + controls_figure(texts) + controls_disclosure(texts)
           + [{'id': f"1 every rated number perturbed ({out['numbers'].get('negative_control', {}).get('n')})",
               'refused': out['numbers'].get('negative_control', {}).get('n_refused') == out['numbers'].get('negative_control', {}).get('n')
               and bool(out['numbers'].get('negative_control', {}).get('n'))}])
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f"negative control not refused: {c['id']}")
    out['guards'] = verify_guards()
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    pj, pm, pn = (os.path.join(od, OUT_NAMES[k]) for k in ('json', 'md', 'man'))
    with open(pj, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, pm)
    inputs = sorted(set([os.path.join(CLONE, f) for f in DIFF_FILES + (FIGPDF, HL)] + list(REVIEW_COPY.values())
                        + [GEN_REL, VERIFY_REL, APPX_REL, FIG_PY, FZ_REL, A89_JSON, W188A_JSON, W188D_MAN, MARKET_XLSX, SUPP_ZIP,
                           SUBMITTED_MAIN, CODE_SRP, BRIEF, STEP6_R9, SCRIPT_NAME] + list(IMPORTED_SCRIPTS) + list(REPORTS)
                        + [os.path.join(W188D_DIR, f) for f in FIG_CSVS + (SOC_SUMMARY_CSV,)] + sorted(EX['CASE33_REF'])
                        + [os.path.join(MR, 'export_W160', f) for f in os.listdir(os.path.join(REPO, MR, 'export_W160'))]
                        + [v['path'] for v in (out['numbers'].get('inputs') or {}).values()]))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md', 'gen', 'verify')},
                'out_dir': out_dir, 'clone_head_full': out['state']['clone_head'],
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(showb(c, f)) for c in (PREV_FULL, HEAD_FULL) for f in DIFF_FILES}
                | {f'{HEAD_FULL}:{FIGPDF}': _shab(showb(HEAD_FULL, FIGPDF))},
                'inputs': {p: {'sha256': _sha(p), 'committed_clean': committed_clean(p)} for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(CHECK_FAILURES)}; guards ok {out["guards"]["ok"]}; pickle {PICKLE_COUNTS}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
