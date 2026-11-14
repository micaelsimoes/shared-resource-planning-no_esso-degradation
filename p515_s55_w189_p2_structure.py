"""P5.15 Addendum 89 (W189 line) and Addendum 90 (W189 additions) -- Planner task W189 PART 2: THE ROUND-9b AUDIT OF
IDENTITY, LABELS, POINTERS, STRUCTURE AND COUNTS AT THE OVERLEAF HEAD 46c9d77. ZERO SOLVES, NO MODEL, NO PICKLE, NO
PRODUCTION IMPORT. Read-only on the Overleaf clone (git rev-parse / show / diff / status / log only; no fetch, pull or
checkout), on manuscript_review/, on manuscript_submitted/, on every committed artifact and on every earlier script.
(Part 1 -- numbers and claim sentences -- and W189a -- supplement v2 -- are other Workers' tasks and are not done here.)

NUMBERING IS STATIC (W182c method). No TeX installation exists on this machine. Every number is the static elsarticle
numbering of W177b-3'/W178c (`numbering_adjusted`) as read by W182c (`_num`, `objects_index`, `heading_map`): \\appendix
resets section, subsection and equation; the figure and table counters are not reset and print <letter>.<n>.

ITEMS.
  1. Identity: main.tex = manuscript_review/main_a87e195_round9b.tex (md5 49139cc2), response_to_reviewers_draft.tex =
     response_to_reviewers_v12.tex (md5 ed8f5b13), highlights.tex = highlights_v3.tex (md5 4cedd45c, unchanged since
     a87e195), cover_letter.tex = cover_letter_v3.tex (md5 0b9e4c21, STEP6_ROUND9_CORRECTIONS.md); the clone diff
     a87e195 -> HEAD per file against main_a87e195_round9b.diff (34 hunks), letter_v10_v12.diff (10) and
     cover_letter_v2_v3.diff (1), hunk by hunk in +/- lines (W181a parse_udiff / compare_hunks); the a87e195 blobs equal
     the round-8 deliverables (so the expert diffs are clone diffs); the new Figure 6 PDF equals the review copy.
  2. Label map a87e195 -> 46c9d77 of every table, figure, equation, section and appendix label (W182c functions),
     the numbered headings mapped (W182c heading_map), compared with the numbering STATED in STEP6_ROUND9_CORRECTIONS.md
     and Addendum 90 (encoded below as STATED_NUMBERING; "everything else identical").
  3. Letter v12 pointers. Chain: (i) the W183h rating of letter v10 at a87e195 is re-derived in memory with W183h's /
     W182c's own functions and checked against the committed W183h JSON; (ii) each v10 pointer is re-rated at 46c9d77
     with target = the a87e195 object carrying its printed number (W182c pointer_rows, plus the lettered pointers), the
     SUBMITTED five carried, and the merges STEP6 declares (4.8 -> 4.7, A.7 -> A.1, Table 11 -> Table 9(b)) applied as
     declared successors; (iii) every v12 pointer is paired with its v10 pointer (W183h pair_and_rate; letter line map)
     and rated MATCH / MISMATCH / SUBMITTED / AMBIGUOUS; v12 pointers with no v10 counterpart (or whose v10 target left
     the paper) take a DECLARED target (DECLARED_TARGETS_V12, each with its reason) or a DECLARED SUBMITTED reading,
     else AMBIGUOUS. Every SUBMITTED pointer is also resolved in manuscript_submitted/main.tex (static), for information.
  4. Dangling references at HEAD: undefined \\ref / \\eqref keys, duplicate labels (W178c task_refs), uses of labels
     removed since a87e195 (typeset and commented lines), kind word before \\ref vs label kind, typed (literal) numbered
     pointers (W182c task_dangling), and prose mentions of the objects round 9b removed (Appendices B-D, Tables E.21 /
     E.23 / E.24, sections E.3 / E.5 / E.6, Table 5 break-even, Table 11 discount, Section 4.8, Appendix A.7) in main.tex
     and in the letter, plus the inventory of supplementary pointers (S<n>).
  5. Hygiene: "[CONFIRM" in the four files; the tightened pattern \\b(Appendix|Section)~\\ref\\{app: (Addendum 86 ruling 4
     (i)) in main.tex (raw and typeset); the general [A-Za-z]+~\\ref\\{app: matches by word; blue wrappers on the text
     changed in round 9b: W182c blue_compare a87e195 -> HEAD, every changed line whose content tokens are not blue,
     classed (caption / table note / prose / heading) and split into newly written text and text moved verbatim from
     a87e195; every float caption new or changed in round 9b with its blue share.
  6. Counts (Addendum 89 decision 4): main-text floats (tables, figures; algorithms reported apart), numbered equations,
     prose words of the main text / Appendix A / Appendix B (= the reduced former Appendix E), headings and caption words
     reported apart, at HEAD and at a87e195; the number of appendices; pages are NOT estimable statically (stated).
     COUNTING RULE (also written into the JSON): comment-stripped source; region = main text from \\section{Introduction}
     to \\appendix, Appendix A from its \\section to the next, Appendix B from its \\section to the unnumbered declaration
     section; float environments (table, figure, longtable, algorithm and starred forms) removed (caption words counted
     apart), display-math environments removed, inline math ($..$, \\(..\\), \\[..\\]) removed, heading titles removed
     (counted apart), the arguments of \\ref / \\eqref / \\cite* / \\label / \\url / \\includegraphics / \\begin / \\end
     and the colour argument of \\textcolor removed, every other control word removed with its argument TEXT kept,
     braces / ~ / escaped characters turned into spaces; a word is a whitespace-separated token holding a letter or digit.

IMPORTED, NOT EDITED: p515_s53_w183h_round8.py (H) and through it W182c (W), W181a (A), W179b, W178c, W177b-3' (R),
W177b-3, W167, the S47 record, W182b and its chain, W180c. None of their output directories is written; none is run.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0, together
with every guard of the imported chains; pickle.load / pickle.loads blocked for the whole run (counters verified 0); no
production module imported (asserted); refuses to run concurrently with another copy of itself; refuses unless the
output directory exists and holds nothing but launch.log; capture-path checklist asserted before any task runs (exit 4);
JSON written only through gate_result_io.

NEGATIVE CONTROLS (in memory, or a tempfile.mkdtemp() directory outside the repository): an altered '+' line of each
expert diff; an injected table shifting a label; altered letter pointers (a paired one, a declared one, the R2.6 Table 7);
an injected \\ref to a removed label and a duplicated \\label; a planted [CONFIRM and a planted Appendix~\\ref{app:; an
unwrapped changed caption; a known sentence and an inline formula added to the word count; an added numbered equation.
Each must be refused (exit 3 otherwise).

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S55/w189_p2_structure && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p2_structure.py \\
      > data/SRP1/Results/P515S55/w189_p2_structure/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
      --out data/SRP1/Results/P515S55/w189_p2_structure/w189p2_bool_typing_test.json \\
      > data/SRP1/Results/P515S55/w189_p2_structure/w189p2_bool_typing_test.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w189_p2_structure.py --post-run
  (trial runs: --trial-out-dir <absolute path outside the repository>)
Exit: 0 = written, every integrity check holds, guards at 0 (a MISMATCH or a failed expectation is a FINDING and never
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W189 part 2 structure audit at Overleaf 46c9d77 '
                                               '(never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W189p2: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W189p2: pickle.loads called -- no model loads are permitted')


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

import p515_s53_w183h_round8 as H  # noqa: E402  (W183h; its chain arms its own guards; NOT edited, not run)
import gate_result_io as GRIO  # noqa: E402

W, A, R, C_, B = H.W, H.A, H.R, H.C_, H.B
os.chdir(REPO)
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s55_w189_p2_structure'
S55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
OUT_DIR = os.path.join(S55, 'w189_p2_structure')
OUT_NAMES = {'json': 'w189_p2_structure.json', 'md': 'w189_p2_structure.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'typing_json': 'w189p2_bool_typing_test.json',
             'typing_log': 'w189p2_bool_typing_test.log', 'post': 'manifest_post_run_sha256.json'}
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
HEAD_FULL = '46c9d779743b40a42da73c0050d8923e6af6fb55'
PREV = 'a87e195'
PREV_FULL = 'a87e1950fbdbb3667f37e64a43c298ba675e6cff'
MR = 'manuscript_review'
LETTER_NAME = 'response_to_reviewers_draft.tex'
FIG6 = 'fig_soc_soh_reference_unit.pdf'
FILES = ('main.tex', LETTER_NAME, 'highlights.tex', 'cover_letter.tex')
REVIEW_COPY = {'main.tex': os.path.join(MR, 'main_a87e195_round9b.tex'),
               LETTER_NAME: os.path.join(MR, 'response_to_reviewers_v12.tex'),
               'highlights.tex': os.path.join(MR, 'highlights_v3.tex'),
               'cover_letter.tex': os.path.join(MR, 'cover_letter_v3.tex')}
PREV_COPY = {'main.tex': os.path.join(MR, 'main_edd7a63_round8.tex'),            # the a87e195 blobs (W183h (a))
             LETTER_NAME: os.path.join(MR, 'response_to_reviewers_v10.tex'),
             'highlights.tex': os.path.join(MR, 'highlights_v3.tex'),
             'cover_letter.tex': os.path.join(MR, 'cover_letter_v2.tex')}
EXPERT_DIFF = {'main.tex': os.path.join(MR, 'main_a87e195_round9b.diff'),
               LETTER_NAME: os.path.join(MR, 'letter_v10_v12.diff'),
               'cover_letter.tex': os.path.join(MR, 'cover_letter_v2_v3.diff')}
EXPECTED_HUNKS = {'main.tex': 34, LETTER_NAME: 10, 'cover_letter.tex': 1, 'highlights.tex': 0}
MD5_DECLARED = {'main.tex': ('49139cc2', 'Addendum 90 / STEP6_ROUND9_CORRECTIONS.md'),
                LETTER_NAME: ('ed8f5b13', 'Addendum 90 / STEP6_ROUND9_CORRECTIONS.md'),
                'highlights.tex': ('4cedd45c', 'Addendum 85 (highlights v3; Addendum 90: unchanged)'),
                'cover_letter.tex': ('0b9e4c21', 'STEP6_ROUND9_CORRECTIONS.md deliverables table')}
LETTER_V10 = os.path.join(MR, 'response_to_reviewers_v10.tex')
LETTER_V9 = H.LETTER_V9
SUBMITTED_MAIN = os.path.join('manuscript_submitted', 'main.tex')
W183H_JSON = os.path.join(S55, 'w183h_round8', 'w183h_round8.json')
W182C_JSON = H.W182C_JSON
W180C_JSON = H.W180C_JSON
STEP6_R9 = 'STEP6_ROUND9_CORRECTIONS.md'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
CHECK_FAILURES = []
IMPORTED_SCRIPTS = H.IMPORTED_SCRIPTS + ('p515_s53_w183h_round8.py',)

# ---- the numbering STATED by STEP6_ROUND9_CORRECTIONS.md ("Numbering changes") and Addendum 90 (Figures) -------------
# label -> (number at a87e195, number at round 9b; None = removed). Every other label: "identical to a87e195", except
# the labels under Appendices B, C, D at a87e195 ("removed -> supplementary material S1-S3").
STATED_NUMBERING = {
    'tab:res_breakeven': ('5', None), 'tab:res_energy_ladder_m2': ('6', '5'), 'tab:res_ageing': ('7', '6'),
    'tab:res_benchmark': ('8', '7'), 'tab:res_multiscenario': ('9', '8'), 'tab:res_year_ladder': ('10', '9'),
    'tab:res_discount': ('11', None),
    'fig:res_soc_soh': (None, '6'), 'fig:res_benchmark': ('6', '7'), 'fig:res_q_cycle': ('7', '8'),
    'subsec:res_limitations': ('4.8', None),
    'app:admm_updated_scenarios': ('A.7', None), 'app:admm_updated_settings': ('A.8', 'A.7'),
    'app:results': ('E', 'B'), 'app:salvage_credit': ('E.1', 'B.1'), 'eq:salvage_credit': ('E.1', 'B.1'),
    'eq:salvage_remaining_life': ('E.2', 'B.2'), 'app:results_claims': ('E.2', 'B.2'), 'tab:appE_claims': ('E.20', 'B.10'),
    'app:results_neighbours': ('E.4', 'B.3'), 'fig:investment_lattice': ('E.19', 'B.8'),
    'tab:appE_neighbours': ('E.22', 'B.11'), 'app:results_statistics': ('E.7', 'B.4'),
    'tab:appE_evaluations': ('E.21', None), 'tab:appE_benchmark': ('E.23', None), 'tab:appE_gap': ('E.24', None),
    'app:results_evaluations': ('E.3', None), 'app:results_benchmark': ('E.5', None), 'app:results_gap': ('E.6', None),
}
STATED_REMOVED_APPENDICES = ('B', 'C', 'D')
STATED_TITLE = {'subsec:res_certification': 'Numerical Performance, Scope and Limitations'}
STATED_EQ_MAIN = '(1)-(40) unchanged (STEP6 "Equations (1)-(40) of the main text: unchanged"; "40 equations")'
STATED_COUNTS = {'STEP6': 'main text 6.5 k prose words, 9 tables + 9 figures, 40 equations; 63 pages (was 76)',
                 'Addendum 89 decision 4': 'main text <= 10 floats; Appendix A compact; Appendix E to one page'}

# merges STEP6 declares: a pointer to the removed object now points to the object it was merged into
DECLARED_SUCCESSORS = {
    'subsec:res_limitations': ('subsec:res_certification', 'STEP6: "Section 4.8 Limitations merged into Section 4.7 '
                                                           '\'Numerical Performance, Scope and Limitations\'"'),
    'app:admm_updated_scenarios': ('app:admm_updated_agents', 'STEP6: "Appendix A.7 Scenario mechanics folded into A.1 '
                                                              '(one paragraph)"'),
    'tab:res_discount': ('tab:res_year_ladder', 'STEP6: "tab:res_discount removed (merged into Table 9(b); refs '
                                                'retargeted)"'),
}

# v12 pointers with no v10 counterpart, or whose v10 target left the paper: (v12 line, kind, printed) -> target.
# Each target from STEP6_ROUND9_CORRECTIONS.md (hunk numbers) or from the letter's own sentence.  Filled from the trial
# runs (the first trial lists every pointer that needs one); a declaration that is never used is a check failure.
# 'v10_line': the v10 line holding the counterpart (cross-checked: a v10 pointer of the same kind on that line resolves
# at HEAD to the declared label, directly or through a declared successor); 'needle': text that must occur in the
# declared target's region at HEAD (heading region, or the float environment) -- both are checks, not assertions.
DECLARED_TARGETS_V12 = {
    (152, 'appendix', 'A.4'): {'label': 'app:admm_updated_stopping', 'needle': 'KKT',
                               'why': 'R2.5: "What can be said about a point the procedure accepts is now stated (Appendix~A.4)" -- '
                                      'STEP6 hunk 3: the first-order characterization is a "new paragraph at the end of Appendix A.4"'},
    (170, 'section', '4.7'): {'label': 'subsec:res_certification', 'needle': 'halved and doubled', 'v10_line': 172,
                              'why': 'R2.6: the penalty re-evaluations "(Section~4.7)" -- STEP6 hunk 15 (W185 paragraph in 4.7); '
                                     'and "the limitation is stated (Section~4.7)" -- the limitations merged into 4.7 (STEP6 '
                                     'hunk 9), v10 l. 172 "Section~4.8"'},
    (170, 'table', '7'): {'label': 'tab:res_benchmark', 'needle': 'three solver starts', 'v10_line': 171,
                          'why': 'R2.6: "three-start bands, Table~7" -- the static benchmark arms (Addendum 90: v11\'s pointer '
                                 'was wrong, v12 prints Table 7); v10 l. 171 "the three-start bands of the benchmark arms, Table~8"'},
    (171, 'appendix', 'A.3--A.5'): {'range_labels': ('app:admm_updated_updates', 'app:admm_updated_certification'), 'v10_line': 173,
                                    'why': 'R2.6 \\rchanges; v10 l. 173 "Appendix~A.3--A.5 and A.8"'},
    (171, 'appendix', 'A.7'): {'label': 'app:admm_updated_settings', 'v10_line': 173,
                               'why': 'R2.6 \\rchanges; v10 l. 173 "and A.8" (settings; STEP6: A.8 -> A.7)'},
    (171, 'appendix', 'B'): {'label': 'app:results', 'v10_line': 173,
                             'why': 'R2.6 \\rchanges; v10 l. 173 "Appendix~E" (the evaluation record; STEP6: E -> B)'},
    (171, 'section', '2.2.6'): {'label': 'subsubsec:certification', 'v10_line': 173,
                                'why': 'R2.6 \\rchanges; v10 l. 173 "Section~2.2.6"'},
    (171, 'section', '4.7'): {'label': 'subsec:res_certification', 'v10_line': 173,
                              'why': 'R2.6 \\rchanges; v10 l. 173 "Section~4.7; Section~4.8" (STEP6: 4.8 merged into 4.7)'},
    (227, 'appendix', 'A.4'): {'label': 'app:admm_updated_stopping', 'needle': 'KKT',
                               'why': 'R3.2: "an approximate first-order stationary point ... (Section~2.2.6 and Appendix~A.4)" -- '
                                      'STEP6 hunk 3'},
    (227, 'section', '2.2.6'): {'label': 'subsubsec:certification', 'needle': 'first-order stationary',
                                'why': 'R3.2: "(Section~2.2.6 and Appendix~A.4)" -- STEP6 hunk 3: "One sentence in 2.2.7 '
                                       '[2.2.6 at a87e195] points to it"'},
    (231, 'figure', '6'): {'label': 'fig:res_soc_soh', 'needle': 'Storage operation and degradation of the reference unit',
                           'why': 'R3.2: "in one figure (Figure~6): the stored energy of the reference unit ... and the '
                                  'state-of-health trajectory" -- Addendum 90 ruling 8; STEP6 hunk 17'},
    (232, 'appendix', 'A.4--A.5'): {'range_labels': ('app:admm_updated_stopping', 'app:admm_updated_certification'),
                                    'v10_line': 235,
                                    'why': 'R3.2 \\rchanges: the characterization (A.4) and the acceptance rule (A.5); v10 l. 235 '
                                           '"Appendix~A.5"'},
    (232, 'section', '2.2.6'): {'label': 'subsubsec:certification', 'v10_line': 235,
                                'why': 'R3.2 \\rchanges; v10 l. 235 "Section~2.2.6"'},
    (232, 'section', '4.7'): {'label': 'subsec:res_certification', 'v10_line': 235,
                              'why': 'R3.2 \\rchanges; v10 l. 235 "Section~4.7 (replaces the former Computational performance)"'},
    (232, 'section', '4'): {'label': 'sec:results', 'v10_line': 235, 'why': 'R3.2 \\rchanges; v10 l. 235 "Section~4"'},
    (279, 'section', '4.4'): {'label': 'subsec:res_coordination', 'needle': '20.5~k',
                              'why': 'benchmark bullet: the coordinated arrangement re-evaluated with the no-reverse-flow rule '
                                     '"(Section~4.4)" -- STEP6 hunk 14 (W184 sentence in 4.4)'},
}
# v12 pointers that name the SUBMITTED version: (v12 line, kind, printed) -> why
DECLARED_SUBMITTED_V12 = {
    (282, 'appendix', 'E'): 'further changes: "The submitted Appendix~E held per-block operating results of the submitted instance"',
    (282, 'appendix', 'B--D'): 'further changes: "The market, network and profile data of the submitted Appendices~B--D"',
    (283, 'appendix', 'D'): 'further changes: "The table of the distribution networks\' transmission connection in the submitted '
                            'Appendix~D printed +-250" (Addendum 90 ruling 6)',
}


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


def committed_clean(rel):
    try:
        _git(['ls-files', '--error-unmatch', rel])
    except subprocess.CalledProcessError:
        return False
    return _git(['status', '--porcelain', '--', rel]).strip() == ''


def rename_keys(obj):
    """W182c's output keys name its revisions 8f051a2 / edd7a63; here they are a87e195 / 46c9d77."""
    if isinstance(obj, dict):
        return {(k.replace('8f051a2', PREV).replace('edd7a63', HEAD_FULL[:7]) if isinstance(k, str) else k): rename_keys(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename_keys(v) for v in obj]
    return obj


# ======================================================================================================================
#  1. identity
# ======================================================================================================================
def apply_udiff(old_text, hunks):
    """The expert's unified diff applied to the old text (context and '-' lines verified) -> (new text, None) or
    (None, the first disagreement). Independent of the diff algorithm's placement of equivalent insertions."""
    a = old_text.split('\n')
    out, pos = [], 0
    for k, h in enumerate(hunks):
        start = h['old_start'] - 1 if h['old_len'] > 0 else h['old_start']
        if start < pos:
            return None, f'hunk {k + 1} overlaps the previous one'
        out.extend(a[pos:start])
        pos = start
        for tag, t in h['lines']:
            if tag == '\\':
                continue
            if tag in (' ', '-'):
                if pos >= len(a) or a[pos] != t:
                    return None, f'hunk {k + 1}: old line {pos + 1} differs from the diff\'s {tag!r} line'
                if tag == ' ':
                    out.append(t)
                pos += 1
            else:
                out.append(t)
    out.extend(a[pos:])
    return '\n'.join(out), None


def rotation_equivalent(g, e):
    """Two hunks that differ only by where an identical-ending insertion is placed: same '-' lines, and the '+' lines of
    one are a rotation of the other's with the old ranges shifted by the rotation."""
    gp, ep = [x for x in g['lines'] if x[0] in '+-'], [x for x in e['lines'] if x[0] in '+-']
    if [x for x in gp if x[0] == '-'] or [x for x in ep if x[0] == '-'] or len(gp) != len(ep):
        return False
    gs, es = [x[1] for x in gp], [x[1] for x in ep]
    return any(gs[r:] + gs[:r] == es for r in range(1, len(gs))) and g['old_len'] == e['old_len']


def diff_control(got, exp, old_text, new_text):
    base = A.compare_hunks(got, exp)
    flagged0 = {p['hunk'] for p in base['per_hunk'] if not p['pm_lines_equal']}
    mut = json.loads(json.dumps(exp))
    hk = next(k for k, h in enumerate(mut) if any(x[0] == '+' for x in h['lines']) and k + 1 not in flagged0)
    j = next(j for j, x in enumerate(mut[hk]['lines']) if x[0] == '+')
    mut[hk]['lines'][j] = ['+', mut[hk]['lines'][j][1] + 'X']
    mut = [dict(h, lines=[tuple(x) for x in h['lines']]) for h in mut]
    cm = A.compare_hunks(got, mut)
    applied, _err = apply_udiff(old_text, mut)
    return {'hunk_mutated': hk + 1, 'flagged_before': sorted(flagged0),
            'flagged_after': [p['hunk'] for p in cm['per_hunk'] if not p['pm_lines_equal']],
            'applied_equals_head_after': applied == new_text,
            'refused': {p['hunk'] for p in cm['per_hunk'] if not p['pm_lines_equal']} == flagged0 | {hk + 1}
            and applied != new_text}


def task_identity():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    parent = _git(['rev-parse', 'HEAD^'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    allst = _git(['status', '--porcelain'], cwd=CLONE).strip()
    changed = [x for x in _git(['diff', '--name-only', PREV_FULL, head], cwd=CLONE).split('\n') if x]
    out = {'clone_head': head, 'clone_head_parent': parent, 'clone_head_date': _git(['log', '-1', '--format=%cI'], cwd=CLONE).strip(),
           'clone_log_top3': _git(['log', '--format=%h %cI %s', '-3'], cwd=CLONE).strip().split('\n'),
           'clone_tracked_modified': status, 'clone_porcelain_all': allst, 'files_changed_a87e195_head': changed,
           'files': {}, 'diffs': {}, 'diff_negative_controls': {}}
    checks = {'head_is_46c9d77': head == HEAD_FULL, 'head_parent_is_a87e195': parent == PREV_FULL,
              'clone_tracked_clean': status == '',
              'files_changed_are_main_letter_cover_fig6': sorted(changed) == sorted(['main.tex', LETTER_NAME, 'cover_letter.tex', FIG6])}
    texts = {}
    for f in FILES:
        wc = _rd(os.path.join(CLONE, f))
        hb = _git(['show', f'HEAD:{f}'], cwd=CLONE, binary=True)
        pb = _git(['show', f'{PREV_FULL}:{f}'], cwd=CLONE, binary=True)
        rc, pc = _rd(REVIEW_COPY[f]), _rd(PREV_COPY[f])
        md5 = hashlib.md5(wc).hexdigest()
        out['files'][f] = {'sha256': _shab(wc), 'md5': md5, 'lines': wc.count(b'\n'), 'working_copy_equals_HEAD_blob': wc == hb,
                           'review_copy': REVIEW_COPY[f], 'review_copy_sha256': _shab(rc), 'review_copy_md5': hashlib.md5(rc).hexdigest(),
                           'byte_identical_to_review_copy': wc == rc,
                           'md5_declared': MD5_DECLARED[f][0], 'md5_declared_source': MD5_DECLARED[f][1],
                           'md5_matches_declared': md5.startswith(MD5_DECLARED[f][0]),
                           'a87e195_blob_sha256': _shab(pb), 'a87e195_blob_equals': PREV_COPY[f],
                           'a87e195_blob_byte_identical_to_that_copy': pb == pc, 'unchanged_since_a87e195': pb == hb}
        texts[f] = wc.decode('utf-8')
        clone_diff_text = _git(['diff', PREV_FULL, head, '--', f], cwd=CLONE)
        got = A.parse_udiff(clone_diff_text).get(f, [])
        if f in EXPERT_DIFF:
            exp_all = A.parse_udiff(_rd(EXPERT_DIFF[f]).decode('utf-8'), default_file=f)
            exp = exp_all.get(f, [])
            cmp_ = A.compare_hunks(got, exp)
            old_t, new_t = pb.decode('utf-8'), hb.decode('utf-8')
            applied, err = apply_udiff(old_t, exp)
            diffpm = [p['hunk'] for p in cmp_['per_hunk'] if not p['pm_lines_equal']]
            cmp_.update({'expert_diff': EXPERT_DIFF[f], 'expert_diff_sha256': _sha(EXPERT_DIFF[f]),
                         'expert_diff_files_parsed': sorted(exp_all),
                         'counts_consistent': all(h['counts_consistent'] for h in got + exp),
                         'hunks_pm_differ': diffpm,
                         'hunks_pm_differ_rotation_equivalent': {str(k): rotation_equivalent(got[k - 1], exp[k - 1])
                                                                for k in diffpm if k <= min(len(got), len(exp))},
                         'expert_diff_applied_to_a87e195_equals_head': applied == new_t, 'apply_error': err,
                         'plus_minus_line_counts_expert': {'plus': sum(1 for h in exp for x in h['lines'] if x[0] == '+'),
                                                           'minus': sum(1 for h in exp for x in h['lines'] if x[0] == '-')}})
            cmp_['pm_equal_up_to_equivalent_placement'] = (
                len(got) == len(exp) and applied == new_t
                and all(cmp_['hunks_pm_differ_rotation_equivalent'].get(str(k)) for k in diffpm))
            out['diff_negative_controls'][f] = diff_control(got, exp, old_t, new_t)
        else:
            cmp_ = {'n_hunks_clone': len(got), 'n_hunks_expert': 0, 'every_hunk_pm_equal': len(got) == 0,
                    'every_hunk_range_equal': len(got) == 0, 'every_hunk_context_equal': len(got) == 0,
                    'counts_consistent': all(h['counts_consistent'] for h in got),
                    'note': 'no expert diff this round (Addendum 90: highlights v3 unchanged); the clone diff must be empty'}
        cmp_['clone_diff_sha256'] = _shab(clone_diff_text.encode('utf-8'))
        cmp_['expected_hunks'] = EXPECTED_HUNKS[f]
        cmp_['plus_minus_line_counts_clone'] = {'plus': sum(1 for h in got for x in h['lines'] if x[0] == '+'),
                                                'minus': sum(1 for h in got for x in h['lines'] if x[0] == '-')}
        out['diffs'][f] = cmp_
    fw, fr = _rd(os.path.join(CLONE, FIG6)), _rd(os.path.join(MR, FIG6))
    out['figure6_pdf'] = {'clone_sha256': _shab(fw), 'review_copy_sha256': _shab(fr), 'byte_identical': fw == fr,
                          'working_copy_equals_HEAD_blob': fw == _git(['show', f'HEAD:{FIG6}'], cwd=CLONE, binary=True),
                          'absent_at_a87e195': FIG6 not in _git(['ls-tree', '--name-only', PREV_FULL], cwd=CLONE).split('\n')}
    out['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'1 identity check {k} failed')
    for f in FILES:
        if not out['files'][f]['working_copy_equals_HEAD_blob']:
            _fail(f'1 {f}: working copy differs from the HEAD blob')
        if not out['diffs'][f]['counts_consistent']:
            _fail(f'1 {f}: a diff hunk is not count-consistent')
    for f, c in out['diff_negative_controls'].items():
        if not c['refused']:
            _fail(f'1 {f}: the diff negative control was not refused')
    out['outcome'] = {f: {'byte_identical_to_review_copy': out['files'][f]['byte_identical_to_review_copy'],
                          'md5_matches_declared': out['files'][f]['md5_matches_declared'],
                          'hunks_clone_expert': [out['diffs'][f]['n_hunks_clone'], out['diffs'][f]['n_hunks_expert']],
                          'every_hunk_pm_equal': out['diffs'][f]['every_hunk_pm_equal'],
                          'pm_equal_up_to_equivalent_placement': out['diffs'][f].get('pm_equal_up_to_equivalent_placement',
                                                                                     out['diffs'][f]['every_hunk_pm_equal']),
                          'hunks_pm_differ': out['diffs'][f].get('hunks_pm_differ', []),
                          'expert_diff_applied_equals_head': out['diffs'][f].get('expert_diff_applied_to_a87e195_equals_head'),
                          'ranges_equal': out['diffs'][f]['every_hunk_range_equal'],
                          'context_equal': out['diffs'][f]['every_hunk_context_equal'],
                          'a87e195_blob_is_round8_deliverable': out['files'][f]['a87e195_blob_byte_identical_to_that_copy']}
                      for f in FILES}
    out['held'] = (all(v['byte_identical_to_review_copy'] and v['md5_matches_declared'] and v['pm_equal_up_to_equivalent_placement']
                       and v['hunks_clone_expert'][0] == EXPECTED_HUNKS[f] and v['a87e195_blob_is_round8_deliverable']
                       for f, v in out['outcome'].items()) and out['figure6_pdf']['byte_identical'] and all(checks.values()))
    return out, texts


# ======================================================================================================================
#  2. label map
# ======================================================================================================================
def _num(lines):
    before = len(W.CHECK_FAILURES)
    raw, adj = W._num(lines)
    if len(W.CHECK_FAILURES) != before:
        _fail(f'W182c/W178c numbering reported: {W.CHECK_FAILURES[before:]}')
    return raw, adj


def build_ctx(old_lines, new_lines):
    raw_o, adj_o = _num(old_lines)
    raw_n, adj_n = _num(new_lines)
    io, inn = W.objects_index(adj_o), W.objects_index(adj_n)
    cl_o, cl_n = W.typeset(old_lines), W.typeset(new_lines)
    if len(cl_o) != len(old_lines) or len(cl_n) != len(new_lines):
        _fail('typeset() changed the line count')
    hm = W.heading_map(adj_o, adj_n, cl_o, cl_n)
    return {'adj_o': adj_o, 'adj_n': adj_n, 'raw_o': raw_o, 'raw_n': raw_n, 'io': io, 'inn': inn, 'hm': hm,
            'eo': W.eq_bodies(adj_o, cl_o), 'en': W.eq_bodies(adj_n, cl_n), 'cl_o': cl_o, 'cl_n': cl_n,
            'lo': old_lines, 'ln': new_lines, 'cites_o': R.cite_order(old_lines), 'cites_n': R.cite_order(new_lines)}


def heading_of(adj, line):
    h = W.heading_at(adj, line)
    return h[1] if h else None


def stated_expectation(label, v_old):
    """(stated old number, stated new number, source) for a label of either revision."""
    if label in STATED_NUMBERING:
        so, sn = STATED_NUMBERING[label]
        return so, sn, 'STEP6 table / Addendum 90'
    if v_old is not None:
        top = str(heading_of_adj_o(v_old)).split('.')[0]
        if top in STATED_REMOVED_APPENDICES:
            return v_old['number'], None, 'STEP6: Appendices B, C, D removed -> supplementary S1-S3'
        return v_old['number'], v_old['number'], 'STEP6: everything else identical to a87e195'
    return None, None, 'not stated (added label)'


_ADJ_O = {}


def heading_of_adj_o(v_old):
    if v_old['kind'] in W.HEADING_KINDS:
        return v_old['number']
    return heading_of(_ADJ_O['adj'], v_old['line'])


def task_label_map(ctx):
    adj_o, adj_n, io, inn = ctx['adj_o'], ctx['adj_n'], ctx['io'], ctx['inn']
    _ADJ_O['adj'] = adj_o
    rows = []
    for lab in sorted(set(adj_o['labels']) | set(adj_n['labels']), key=lambda k: (k not in adj_o['labels'], k)):
        vo, vn = adj_o['labels'].get(lab), adj_n['labels'].get(lab)
        st = ('added' if vo is None else 'removed' if vn is None else
              'unchanged' if vo['number'] == vn['number'] else 'renumbered')
        so, sn, src = stated_expectation(lab, vo)
        no, nn = (vo or {}).get('number'), (vn or {}).get('number')
        rows.append({'label': lab, 'class': W.label_class(lab, vo or vn), 'kind': (vo or vn)['kind'],
                     'number_a87e195': no, 'number_46c9d77': nn, 'line_a87e195': (vo or {}).get('line'),
                     'line_46c9d77': (vn or {}).get('line'), 'status': st,
                     'heading_a87e195': heading_of(adj_o, vo['line']) if vo else None,
                     'heading_46c9d77': heading_of(adj_n, vn['line']) if vn else None,
                     'desc_a87e195': W.label_desc(adj_o, io, lab) if vo else None,
                     'desc_46c9d77': W.label_desc(adj_n, inn, lab) if vn else None,
                     'stated_a87e195': so, 'stated_46c9d77': sn, 'stated_source': src,
                     'agrees_with_stated': (so == no and sn == nn)})
    hm = ctx['hm']
    heads = []
    for h in hm.values():
        r = {k: h.get(k) for k in ('kind', 'number_old', 'title_old', 'labels', 'line_old', 'kind_new', 'number_new',
                                   'title_new', 'line_new', 'method', 'status', 'shingle_hint_top3', 'contained_labels')}
        heads.append(r)
    eq_unl_o = [{'number_a87e195': k, 'number_46c9d77': next((n for n, b in ctx['en'].items() if b == v), None),
                 'body_start': v[0][:90]} for k, v in ctx['eo'].items()]
    eq_unl_n = [{'number_46c9d77': k, 'in_a87e195': any(v == w for w in ctx['eo'].values()), 'body_start': v[0][:90]}
                for k, v in ctx['en'].items()]
    disagree = [r for r in rows if not r['agrees_with_stated']]
    titles = {lab: {'stated': t, 'at_46c9d77': W.label_desc(adj_n, inn, lab),
                    'agrees': W._clean_desc(t) == (W.label_desc(adj_n, inn, lab) or '')} for lab, t in STATED_TITLE.items()}
    by_class = {}
    for r in rows:
        by_class.setdefault(r['class'], Counter())[r['status']] += 1
    cites = {'n_keys_a87e195': len(ctx['cites_o']), 'n_keys_46c9d77': len(ctx['cites_n']),
             'order_identical': ctx['cites_o'] == ctx['cites_n'],
             'renumbered': [{'key': k, 'number_a87e195': i + 1,
                             'number_46c9d77': ctx['cites_n'].index(k) + 1 if k in ctx['cites_n'] else None}
                            for i, k in enumerate(ctx['cites_o']) if k not in ctx['cites_n'] or ctx['cites_n'].index(k) != i],
             'new_keys': [k for k in ctx['cites_n'] if k not in ctx['cites_o']]}
    main_eq_max = {rev: max(int(o['number']) for o in adj['objects'] if o['kind'].startswith('equation') and o['number']
                            and str(o['number']).isdigit()) for rev, adj in ((PREV, adj_o), ('46c9d77', adj_n))}
    return {'rows': rows, 'counts': dict(Counter(r['status'] for r in rows)),
            'by_class': {k: dict(v) for k, v in by_class.items()},
            'n_labels_a87e195': len(adj_o['labels']), 'n_labels_46c9d77': len(adj_n['labels']),
            'renumbered': [r for r in rows if r['status'] == 'renumbered'],
            'removed': [r for r in rows if r['status'] == 'removed'], 'added': [r for r in rows if r['status'] == 'added'],
            'heading_map': heads, 'heading_counts': dict(Counter(h['status'] for h in heads)),
            'equations_unlabelled_a87e195': eq_unl_o, 'equations_unlabelled_46c9d77': eq_unl_n,
            'main_text_equation_max_number': main_eq_max, 'stated_main_equations': STATED_EQ_MAIN,
            'n_disagree_with_stated': len(disagree), 'disagree_with_stated': disagree, 'stated_titles': titles,
            'citations': cites, 'object_counts': {PREV: adj_o['counts'], '46c9d77': adj_n['counts']},
            'counter_adjustments': {PREV: adj_o['counter_adjustments'], '46c9d77': adj_n['counter_adjustments']},
            'numbering_warnings': {PREV: adj_o['warnings'], '46c9d77': adj_n['warnings']},
            'elsarticle': B.task_elsarticle(),
            'static_numbering_note': 'static numbering (no TeX on this machine; W182c method); nothing compiled'}


# ======================================================================================================================
#  3. letter pointers
# ======================================================================================================================
def v10_at_prev():
    """W183h (c) re-derived in memory: the v10 pointers rated at a87e195 by W183h's / W182c's own functions."""
    tx = {'main_old': W.show(W.OLD), 'main_head': show(PREV_FULL, 'main.tex'), 'main_r7': W.show(W.ROUND7),
          'main_w180c': W.show(W.W180C_HEAD), 'letter_w180c': W.show(W.W180C_HEAD, W.LETTER_NAME)}
    n0 = len(W.CHECK_FAILURES)
    _lab, ctx10 = W.task_label_map(tx)
    w180c = json.load(open(W180C_JSON, encoding='utf-8'))
    v9_text = _rd(LETTER_V9).decode('utf-8')
    rows9, _, _ = W.pointer_rows(LETTER_V9, ctx10, w180c['pointers']['pointers'], tx['letter_w180c'], v9_text)
    v10_text = _rd(LETTER_V10).decode('utf-8')
    atoms10 = H.all_pointers(LETTER_V10)
    mp = {v: k for k, v in H.line_map(v9_text, v10_text).items()}
    rows10, dropped, used = H.pair_and_rate(atoms10, rows9, ctx10, mp, H.DECLARED_TARGETS)
    if len(W.CHECK_FAILURES) > n0:
        _fail(f'3 W182c functions reported (v10 at a87e195): {W.CHECK_FAILURES[n0:]}')
    committed = json.load(open(W183H_JSON, encoding='utf-8'))['c_pointers']['v10']['pointers']
    key = lambda r: (r['line'], r['text'], r['verdict'], r.get('expected_at_head'))  # noqa: E731
    agree = len(rows10) == len(committed) and all(key(a) == key(b) for a, b in zip(rows10, committed))
    if not agree:
        _fail('3 v10 at a87e195 re-derived differs from the committed W183h rating')
    return rows10, {'counts': dict(Counter(r['verdict'] for r in rows10)), 'n': len(rows10),
                    'equals_committed_w183h': agree, 'n_committed': len(committed)}


def rate_atom_at(atom, ctx):
    """A pointer rated at ctx's new revision; target = the ctx-old object carrying the printed number (lettered pointers;
    the same rules as W182c pointer_rows for floats, equations and headings)."""
    row = {k: atom[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range')}
    if atom['range']:
        members, why = W.range_members(atom, ctx)
        if members is None:
            row['verdict'], row['note'] = 'AMBIGUOUS', why
            return row
    else:
        objs = W.old_objects_of(atom['kind'], ctx)
        members = [objs[atom['numbers'][0]]] if atom['numbers'][0] in objs else None
        if members is None:
            row['verdict'], row['note'] = 'AMBIGUOUS', f'no {atom["kind"]} {atom["numbers"][0]} at {PREV}'
            return row
    res = [W.resolve_identity(m, ctx) for m in members]
    row['target_8f051a2'] = [r_['key'] for r_ in res] if atom['range'] else res[0]['key']
    row['resolution'] = [r_['how'] for r_ in res]
    nn = [r_['number_new'] for r_ in res]
    row['numbers_edd7a63'] = nn
    if atom['range']:
        alive = sorted([n for n in nn if n is not None], key=W.natkey)
        removed = [res[i]['key'] for i, n in enumerate(nn) if n is None]
        row['members_removed'] = removed
        if not alive:
            row['verdict'], row['subclass'] = 'MISMATCH', 'all range members removed'
            return row
        got = [alive[0], alive[-1]]
        row['should_be'] = '--'.join(got)
        row['verdict'] = 'MISMATCH' if (removed or got != atom['numbers']) else 'MATCH'
        row['subclass'] = 'range member removed' if removed else ('renumbered' if got != atom['numbers'] else None)
    else:
        n = nn[0]
        if n is None:
            row['verdict'], row['subclass'] = 'MISMATCH', 'target removed'
        elif n == atom['numbers'][0]:
            row['verdict'] = 'MATCH'
        else:
            row['verdict'], row['subclass'], row['should_be'] = 'MISMATCH', 'renumbered', n
    row['printed_now_denotes'] = [W.now_denotes(atom['kind'], p, ctx) for p in atom['numbers']]
    return row


def label_number(ctx, label):
    v = ctx['adj_n']['labels'].get(label)
    return v['number'] if v else None


def apply_successors(row, ctx):
    """A v10 target removed by a merge STEP6 declares: the successor's number at HEAD."""
    tgt = row.get('target_8f051a2')
    if isinstance(tgt, list) or tgt not in DECLARED_SUCCESSORS or row['verdict'] not in ('MISMATCH', 'AMBIGUOUS'):
        return row
    if row.get('should_be'):
        return row
    succ, why = DECLARED_SUCCESSORS[tgt]
    n = label_number(ctx, succ)
    printed = row['numbers'][0]
    row.update({'declared_successor': succ, 'declared_successor_why': why, 'should_be': n,
                'verdict': 'MATCH' if n == printed else 'MISMATCH',
                'subclass': f'target merged into {succ} (declared successor)'})
    return row


def v10_at_head(ctx, rows10):
    v10_text = _rd(LETTER_V10).decode('utf-8')
    base, _, _ = W.pointer_rows(LETTER_V10, ctx, [], '', v10_text)
    for r in base:
        r['extractor'] = 'W177b-3prime letter_pointers'
    lettered = [dict(rate_atom_at(a, ctx), extractor='W183h lettered') for a in H.lettered_pointers(LETTER_V10)]
    rows = base + lettered
    def occ_keys(rs):
        seen, out_ = Counter(), []
        for r_ in rs:
            k_ = (r_['line'], r_['kind'], r_['text'], tuple(r_['numbers']))
            out_.append(k_ + (seen[k_],))
            seen[k_] += 1
        return out_
    sub = {k for k, r_ in zip(occ_keys(rows10), rows10) if r_['verdict'] == 'SUBMITTED'}
    n_sub = 0
    rows.sort(key=lambda r_: (r_['line'], r_['kind']))      # stable: atom order within a line and kind kept
    for r, k in zip(rows, occ_keys(rows)):
        if k in sub:
            n_sub += 1
            r.update({'verdict': 'SUBMITTED', 'note': 'carried from W183h v10 (W180c SUBMITTED)', 'subclass': None})
            r.pop('should_be', None)
        apply_successors(r, ctx)
    if n_sub != 5:
        _fail(f'3 {n_sub} SUBMITTED v10 pointers carried (expected 5)')
    keys10 = sorted((r['line'], r['kind'], r['text'], tuple(r['numbers'])) for r in rows10)
    keysh = sorted((r['line'], r['kind'], r['text'], tuple(r['numbers'])) for r in rows)
    if keys10 != keysh:
        _fail('3 the v10 pointer set at HEAD differs from the v10 pointer set at a87e195')
    rows.sort(key=lambda r: (r['line'], r['kind'], r['text']))
    return rows


def declared_number(ctx, dec):
    if 'label' in dec:
        return label_number(ctx, dec['label']), dec['label']
    if 'range_labels' in dec:
        a, b = (label_number(ctx, x) for x in dec['range_labels'])
        return (f'{a}--{b}' if a and b else None), '--'.join(dec['range_labels'])
    v = ctx['adj_n']['labels'].get(dec['heading_of_label'])
    if not v:
        return None, None
    h = W.heading_at(ctx['adj_n'], v['line'])
    return (h[1] if h else None), f'heading containing {dec["heading_of_label"]}'


def submitted_denotes(kind, number, sub_ctx):
    if sub_ctx is None:
        return None
    if kind == 'former':
        kind = 'section'
    kinds = W.HEADING_KINDS if kind in ('section', 'new', 'appendix') else (kind,)
    hits = [o for o in sub_ctx['inn'].values() if o['kind'] in kinds and o['number'] == number]
    return W._obj_str(hits[0]) if hits else 'nothing'


def rate_v12(atoms12, rows10h, ctx, mp12to10, declared, declared_sub, sub_ctx):
    # declared targets in the W183h pair_and_rate format; range declarations handled after
    decl_pr = {k: v for k, v in declared.items() if 'range_labels' not in v}
    rows, dropped, used = H.pair_and_rate(atoms12, rows10h, ctx, mp12to10, decl_pr)
    used = set(used)
    for r in rows:
        if 'v9' in r:
            r['v10_at_head'] = r.pop('v9')
        r['line_v10'] = r.pop('line_v9', None)
        key = (r['line'], r['kind'], '--'.join(r['numbers']))
        if r['verdict'] == 'AMBIGUOUS' and key in declared and 'range_labels' in declared[key]:
            num, what = declared_number(ctx, declared[key])
            used.add(key)
            r.update({'verdict': 'MATCH' if num == '--'.join(r['numbers']) else 'MISMATCH', 'expected_at_head': num,
                      'target': what, 'declared_target': True, 'why': declared[key]['why']})
        if r['verdict'] == 'AMBIGUOUS' and key in declared_sub:
            used.add(key)
            r.update({'verdict': 'SUBMITTED', 'declared_submitted': True, 'why': declared_sub[key]})
        if r['verdict'] == 'SUBMITTED':
            r['submitted_denotes'] = [submitted_denotes(r['kind'], p, sub_ctx) for p in r['numbers']]
        if isinstance(r.get('note'), str):
            r['note'] = r['note'].replace('since 8f051a2', f'since {PREV} (W183h wording)')
    stale = sorted(set(declared) | set(declared_sub))
    stale = [k for k in stale if k not in used]
    return rows, dropped, stale


def task_pointers(ctx, texts, sub_ctx, declared=None, declared_sub=None, letter_path=None):
    declared = DECLARED_TARGETS_V12 if declared is None else declared
    declared_sub = DECLARED_SUBMITTED_V12 if declared_sub is None else declared_sub
    rows10, prev_info = v10_at_prev()
    rows10h = v10_at_head(ctx, rows10)
    v10_text = _rd(LETTER_V10).decode('utf-8')
    lp = letter_path or os.path.join(CLONE, LETTER_NAME)
    v12_text = open(os.path.join(REPO, lp), encoding='utf-8').read()
    atoms12 = H.all_pointers(lp)
    mp = {v: k for k, v in H.line_map(v10_text, v12_text).items()}       # v12 line -> v10 line
    rows12, dropped, stale = rate_v12(atoms12, rows10h, ctx, mp, declared, declared_sub, sub_ctx)
    decl_checks = declaration_checks(ctx, rows10h, declared)
    return {'v10_at_a87e195': prev_info, 'v10_at_head': rows10h,
            'v10_at_head_counts': dict(Counter(r['verdict'] for r in rows10h)),
            'v12': rows12, 'v12_counts': dict(Counter(r['verdict'] for r in rows12)),
            'v12_by_extractor': dict(Counter(r['extractor'] for r in rows12)),
            'v12_mismatch': [r for r in rows12 if r['verdict'] == 'MISMATCH'],
            'v12_ambiguous': [r for r in rows12 if r['verdict'] == 'AMBIGUOUS'],
            'v12_submitted': [r for r in rows12 if r['verdict'] == 'SUBMITTED'],
            'v12_declared': [r for r in rows12 if r.get('declared_target') or r.get('declared_submitted')],
            'v10_pointers_dropped_in_v12': [{k: d.get(k) for k in ('line', 'text', 'numbers', 'verdict', 'should_be')}
                                            for d in dropped],
            'stale_declarations': [list(k) for k in stale], 'declaration_checks': decl_checks,
            'declaration_checks_all_hold': all(v['holds'] for v in decl_checks.values()),
            'declared_targets': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in declared.items()},
            'declared_submitted': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in declared_sub.items()},
            'declared_successors': {k: {'successor': v[0], 'why': v[1]} for k, v in DECLARED_SUCCESSORS.items()}}


def target_region(ctx, label):
    """Typeset text of a label's region at HEAD: a heading's own text up to the next heading, or the float block."""
    v = ctx['adj_n']['labels'].get(label)
    if v is None:
        return ''
    cl = ctx['cl_n']
    if v['kind'] in W.HEADING_KINDS:
        return W.heading_own_text(cl, v['line'])
    i = v['line'] - 1
    b = next((k for k in range(i, -1, -1) if re.search(r'\\begin\{(table|figure|longtable|algorithm)\*?\}', cl[k])), i)
    e = next((k for k in range(i, len(cl)) if re.search(r'\\end\{(table|figure|longtable|algorithm)\*?\}', cl[k])), i)
    return ws(W.strip_blue('\n'.join(cl[b:e + 1])))


def declaration_checks(ctx, rows10h, declared):
    out = {}
    for key, dec in declared.items():
        labs = list(dec['range_labels']) if 'range_labels' in dec else [dec['label']]
        rec = {'labels': labs, 'labels_defined_at_head': all(label_number(ctx, x) for x in labs)}
        if dec.get('needle'):
            reg = ' '.join(target_region(ctx, x) for x in labs)
            rec['needle'] = dec['needle']
            rec['needle_in_target_region'] = dec['needle'] in reg or dec['needle'] in reg.replace('~', ' ')
        if dec.get('v10_line'):
            cps = [r for r in rows10h if r['line'] == dec['v10_line'] and r['kind'] == key[1]]
            tg = set()
            for r in cps:
                t = r.get('declared_successor') or r.get('target_8f051a2')
                tg.update(t if isinstance(t, list) else [t])
            rec['v10_counterpart_targets'] = sorted(x for x in tg if x)
            rec['v10_counterpart_agrees'] = bool(set(labs) & tg)
        rec['holds'] = (rec['labels_defined_at_head'] and rec.get('needle_in_target_region', True)
                        and rec.get('v10_counterpart_agrees', True))
        out[f'{key[0]}|{key[1]}|{key[2]}'] = rec
    return out


def table7_check(rows12, texts):
    hits = [r for r in rows12 if r['kind'] == 'table' and r['numbers'] == ['7']]
    L = texts[LETTER_NAME].split('\n')
    out = []
    for r in hits:
        ln = L[r['line'] - 1]
        i = ln.find('Table~7')
        out.append({'line': r['line'], 'verdict': r['verdict'], 'expected_at_head': r.get('expected_at_head'),
                    'context': ws(ln[max(0, i - 200):i + 20]), 'in_R2_6': in_reply(L, r['line'], 'R2.5'),
                    'denotes': r.get('printed_now_denotes')})
    bullet = [i for i, x in enumerate(L, 1) if 'The uncoordinated benchmark was redefined' in x]
    bullet_ptrs = [{'line': r['line'], 'text': r['text'], 'verdict': r['verdict']} for r in rows12 if r['line'] in bullet]
    return {'table7_pointers': out, 'benchmark_bullet_lines': bullet, 'benchmark_bullet_pointers': bullet_ptrs,
            'reading': 'the "benchmark pointer" Addendum 90 names (v11 "three-start bands, Table 6") is the R2.6 sentence '
                       '"three-start bands, Table~7"; the further-changes bullet on the benchmark carries section '
                       'pointers only'}


def in_reply(L, line, rid):
    """The \\subsection* reply heading above a letter line."""
    for k in range(line - 1, -1, -1):
        m = re.match(r'\\subsection\*\{([^}]*)\}', L[k])
        if m:
            return m.group(1)[:60]
        if L[k].startswith('\\section*'):
            return L[k][:60]
    return None


def controls_pointers(ctx, texts, sub_ctx, rows12):
    tmp = tempfile.mkdtemp(prefix='w189p2_ctl_')
    v12 = texts[LETTER_NAME]
    L0 = v12.split('\n')
    cases = []
    t7 = [r for r in rows12 if r['kind'] == 'table' and r['numbers'] == ['7']]
    if t7:
        cases.append(('p1 R2.6 "Table~7" -> "Table~8" (declared target kept)', t7[0]['line'], 'Table~7', 'Table~8',
                      ('table', '8'), '7'))
    m = next((r for r in rows12 if r['verdict'] == 'MATCH' and not r.get('declared_target') and r['kind'] == 'section'
              and r['numbers'] == ['4.7'] and r['text'] in L0[r['line'] - 1]), None)
    if m:
        cases.append(('p2 a paired MATCH "Section~4.7" -> "Section~4.8"', m['line'], m['text'], m['text'].replace('4.7', '4.8'),
                      ('section', '4.8'), '4.7'))
    d = next((r for r in rows12 if r.get('declared_target') and r['verdict'] == 'MATCH' and not r['range']
              and r['kind'] in ('figure', 'appendix', 'section', 'table')), None)
    if d:
        p = d['numbers'][0]
        q = (p[:-1] + str((int(p[-1]) % 8) + 1)) if p[-1].isdigit() else p
        cases.append((f'p3 a declared MATCH "{d["text"]}" -> number {q} (declared target kept)', d['line'], d['text'],
                      d['text'].replace(p, q), (d['kind'], q), p))
    out = []
    for cid, ln, a, b, (kind, num), was in cases:
        L = list(L0)
        ok = a in L[ln - 1]
        L[ln - 1] = L[ln - 1].replace(a, b, 1)
        path = os.path.join(tmp, f'letter_{len(out)}.tex')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(L))
        decl = dict(DECLARED_TARGETS_V12)
        if (ln, kind, was) in decl:
            decl[(ln, kind, num)] = decl[(ln, kind, was)]
        res = task_pointers(ctx, texts, sub_ctx, declared=decl, letter_path=path)
        hit = [r for r in res['v12'] if r['line'] == ln and r['kind'] == kind and '--'.join(r['numbers']) == num]
        out.append({'id': cid, 'source_text_present': ok, 'verdicts': [r['verdict'] for r in hit],
                    'refused': ok and bool(hit) and all(r['verdict'] != 'MATCH' for r in hit)})
    if len(out) < 3:
        _fail(f'3 only {len(out)} pointer controls could be built')
    return out


# ======================================================================================================================
#  4. dangling references and prose mentions
# ======================================================================================================================
PROSE_9B = [
    ('typed Appendix letter B-E (main: should be \\ref or gone; letter: only C-E, removed at HEAD)',
     r'\bAppendi(?:x|ces)\s*~?\s*[B-E]\b'),
    ('typed lettered number of a removed appendix (C.n, D.n, E.n)', r'(?<![\w.])[C-E]\.\d+\b'),
    ('"evaluation record"', r'evaluation record'),
    ('market / network / profile data', r'\b(?:market|network|profile)(?:,\s*network)?(?:\s+and\s+profile)?\s+data\b'),
    ('"scenario mechanics"', r'scenario mechanics'),
    ('"limitations" (Section 4.8 merged)', r'\blimitations?\b'),
    ('"break-even" table / fit (Table 5 removed)', r'break-even\s+(?:table|fit)'),
    ('discount-rate table / sensitivity (Table 11 merged)', r'discount[- ]rate\s+(?:table|sensitivit\w*)'),
    ('supplementary pointer S<n>', r'(?<![\w.])S\d+(?:--S\d+)?\b'),
    ('"supplementary"', r'\bsupplementar\w*'),
]


def prose_hits(lines, rx):
    hits = []
    for i, ln in enumerate(lines, 1):
        for m in re.finditer(rx, ln, re.I if 'S\\d' not in rx else 0):
            hits.append({'line': i, 'match': m.group(0), 'context': ws(ln[max(0, m.start() - 90):m.end() + 60])})
    return hits


def letter_typeset(text):
    s = R._blank_cmd(text, r'\rcomment')
    return [re.sub(r'(?<!\\)%.*', '', ln) for ln in s.split('\n')]


def task_dangling(ctx, lab, texts):
    adj_n = ctx['adj_n']
    refs = C_.task_refs(ctx['ln'], adj_n)
    removed = {r['label'] for r in lab['removed']}
    uses_removed_typeset, uses_removed_commented = [], []
    for i, (raw, cl) in enumerate(zip(ctx['ln'], ctx['cl_n']), 1):
        for m in W.REF_RX.finditer(cl):
            for k in m.group(2).split(','):
                if k.strip() in removed:
                    uses_removed_typeset.append({'line': i, 'label': k.strip(), 'context': ws(cl[max(0, m.start() - 80):m.end() + 30])})
        if raw.lstrip().startswith('%'):
            for m in W.REF_RX.finditer(raw):
                for k in m.group(2).split(','):
                    if k.strip() in removed:
                        uses_removed_commented.append({'line': i, 'label': k.strip(), 'context': ws(raw)[:160]})
    compat = {'rows': [{'label': r['label'], 'status': r['status'], 'desc_8f051a2': r['desc_a87e195'],
                        'desc_edd7a63': r['desc_46c9d77'], 'number_8f051a2': r['number_a87e195'],
                        'number_edd7a63': r['number_46c9d77']} for r in lab['rows']],
              'removed': [{'label': r['label']} for r in lab['removed']]}
    wd = W.task_dangling(ctx, compat, texts[LETTER_NAME])
    wd.pop('prose_mentions', None)              # W182c's round-8 patterns; the round-9b patterns are below
    wd.pop('commented_lines_naming_removed_labels', None)
    wd = rename_keys(wd)
    lt = letter_typeset(texts[LETTER_NAME])
    rx_letter = {PROSE_9B[0][0]: r'\bAppendi(?:x|ces)\s*~?\s*(?:[C-E]\b|B--)'}
    prose = {name: {'main.tex': prose_hits(ctx['cl_n'], rx), 'letter': prose_hits(lt, rx_letter.get(name, rx))}
             for name, rx in PROSE_9B}
    s_ptr = sorted({h['match'] for h in prose['supplementary pointer S<n>']['main.tex']})
    s_nums = sorted({int(x) for p in s_ptr for x in re.findall(r'S(\d+)', p)})
    s_ptr_l = sorted({h['match'] for h in prose['supplementary pointer S<n>']['letter']})
    out = {'refs_static_46c9d77': refs, 'n_undefined': len(refs['undefined_refs']), 'n_duplicate_labels': len(refs['duplicate_labels']),
           'uses_of_removed_labels_typeset': uses_removed_typeset, 'uses_of_removed_labels_in_comments': uses_removed_commented,
           'w182c_task_dangling': wd, 'prose_mentions_round9b': prose,
           'supplementary_pointers': {'main_distinct': s_ptr, 'main_numbers': s_nums, 'letter_distinct': s_ptr_l,
                                      'all_within_S1_S6': all(1 <= n <= 6 for n in s_nums),
                                      'note': 'inventory only; the supplement content is W189a\'s task'}}
    # controls
    L = list(ctx['ln'])
    k = next(i for i, x in enumerate(L) if '\\section{Conclusions}' in x or 'Conclusions}' in x)
    L1 = L[:k + 1] + ['See Table~\\ref{tab:res_discount}.'] + L[k + 1:]
    _, a1 = W._num(L1)
    r1 = C_.task_refs(L1, a1)
    L2 = L[:k + 1] + ['\\label{tab:res_ageing}'] + L[k + 1:]
    _, a2 = W._num(L2)
    r2 = C_.task_refs(L2, a2)
    ctl = [{'id': 'd1 an injected \\ref{tab:res_discount} is undefined', 'refused': r1['undefined_refs'] == ['tab:res_discount']},
           {'id': 'd2 a duplicated \\label{tab:res_ageing} is reported', 'refused': len(r2['duplicate_labels']) == 1}]
    out['held'] = out['n_undefined'] == 0 and out['n_duplicate_labels'] == 0 and not uses_removed_typeset
    return out, ctl


# ======================================================================================================================
#  5. hygiene: [CONFIRM, prefixes, blue
# ======================================================================================================================
TIGHT = re.compile(r'\b(Appendix|Section)~\\ref\{app:')
GENERAL = re.compile(r'([A-Za-z]+)~\\ref\{app:')
NONBLUE_CHANGED = ('new', 'rewritten', 'unwrapped', 'moved_unwrapped')


STRUCTURAL = re.compile(r'\\(?:begin|end|label|includegraphics|ref|eqref|cite[a-z]*|setlength|hspace|vspace|multicolumn|'
                        r'multirow|cmidrule|addlinespace|centering|footnotesize|small|toprule|midrule|bottomrule|hline)\*?'
                        r'(?:\[[^\]]*\]|\{[^{}]*\})*')


def content_token(t):
    """A token holds content when a letter or digit is left after structural commands (with their arguments) and every
    other control-word name are removed."""
    s_ = STRUCTURAL.sub('', t)
    s_ = re.sub(r'\\[A-Za-z@]+\*?', '', s_)
    s_ = re.sub(r'\\.', '', s_)
    return bool(re.search(r'[A-Za-z0-9]', s_))


def token_content_blue(cl):
    """Per token (W182c blue_tokens order: comment-stripped lines, wrappers removed, \\S+ runs), the blue share of its
    alphanumeric characters with control-word names excluded (W183h alnum_blue_share) -- so "\\caption{Storage" with
    "Storage" inside a wrapper counts as blue, which a majority-of-characters rule does not."""
    clean, mask = W.blue_mask_text('\n'.join(cl))
    out, pos = [], 0
    for ln in clean.split('\n'):
        for m in re.finditer(r'\S+', ln):
            out.append(H.alnum_blue_share(clean, mask, pos + m.start(), pos + m.end()))
        pos += len(ln) + 1
    return out


def env_at(cl, ln_no):
    """The innermost table / figure / longtable / algorithm / tabular environment open at a line (scan from the top)."""
    stack = []
    rx = re.compile(r'\\(begin|end)\{(table\*?|longtable|figure\*?|algorithm|tabular\*?|tabularx)\}')
    for k in range(ln_no):
        line = cl[k]
        for m in rx.finditer(line):
            if k == ln_no - 1 and m.group(1) == 'end':
                continue
            if m.group(1) == 'begin':
                stack.append(m.group(2))
            elif stack and stack[-1] == m.group(2):
                stack.pop()
    return stack


def plus_lines(diff_u0):
    out, cur = set(), None
    for ln in diff_u0.split('\n'):
        m = re.match(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', ln)
        if m:
            cur = int(m.group(1))
            continue
        if cur is None or ln.startswith('+++') or ln.startswith('---'):
            continue
        if ln.startswith('+'):
            out.add(cur)
            cur += 1
        elif ln.startswith(' '):
            cur += 1
    return out


def blue_rows(cl_o, cl_n, raw_n, plus, k=6):
    cmp_ = W.blue_compare(cl_o, cl_n)
    fn, st = cmp_['fn'], cmp_['status']
    share = token_content_blue(cl_n)
    if len(share) != len(fn):
        _fail(f'5 token alignment: {len(share)} shares for {len(fn)} tokens')
        share = [1.0 if b else 0.0 for _t, b, _l in fn]
    old_tokens = [t for t, _b, _l in cmp_['fo']]
    old_sh = {tuple(old_tokens[i:i + k]) for i in range(len(old_tokens) - k + 1)}
    moved = [False] * len(fn)
    for j in range(len(fn) - k + 1):
        if tuple(x[0] for x in fn[j:j + k]) in old_sh:
            for d in range(k):
                moved[j + d] = True
    by = collections.defaultdict(list)
    for j, (t, b, ln_no) in enumerate(fn):
        by[ln_no].append(j)
    rows = []
    for ln_no in sorted(by):
        js = by[ln_no]
        bad = [j for j in js if st[j] in NONBLUE_CHANGED and content_token(fn[j][0]) and (share[j] or 0.0) < 0.5]
        if not bad:
            continue
        cont = [j for j in js if content_token(fn[j][0])]
        raw = raw_n[ln_no - 1]
        where = W.caption_or_env(cl_n, ln_no)
        envs = env_at(cl_n, ln_no)
        inner = envs[-1] if envs else None
        rows.append({'line': ln_no, 'on_plus_line': ln_no in plus, 'where': where, 'environments_open': envs,
                     'class': ('caption' if '\\caption' in raw else 'heading' if re.match(r'\s*\\(sub)*section', raw)
                               else 'table cell' if inner and inner.startswith('tabular')
                               else 'table note' if inner and inner.startswith(('table', 'longtable'))
                               else 'figure / algorithm content' if inner else 'prose'),
                     'n_content_tokens': len(cont), 'n_nonblue_changed': len(bad),
                     'n_moved_verbatim_from_a87e195': sum(1 for j in bad if moved[j]),
                     'n_newly_written': sum(1 for j in bad if not moved[j]),
                     'status_counts': dict(Counter(st[j] for j in bad)),
                     'blue_share_content_tokens': round(sum(1 for j in cont if (share[j] or 0) >= 0.5) / len(cont), 2) if cont else None,
                     'first_nonblue_words': ' '.join(fn[j][0] for j in bad)[:160]})
    return rows, cmp_['counts']


def caption_rows(ctx):
    out = []
    for lab, v in ctx['adj_n']['labels'].items():
        if v['kind'] not in W.FLOAT_KINDS and not str(v['kind']).startswith('table'):
            continue
        bn, cn = W.caption_blue(ctx['cl_n'], lab, ctx['adj_n'])
        if bn is None:
            continue
        old = ctx['adj_o']['labels'].get(lab)
        co = W.caption_blue(ctx['cl_o'], lab, ctx['adj_o'])[1] if old else None
        full_n = caption_full(ctx['cl_n'], ctx['adj_n'], lab)
        full_o = caption_full(ctx['cl_o'], ctx['adj_o'], lab) if old else None
        changed = full_o is None or full_o != full_n
        out.append({'label': lab, 'number_46c9d77': v['number'], 'line_46c9d77': v['line'], 'new_in_round9b': old is None,
                    'caption_changed_in_round9b': changed, 'blue_share': bn, 'caption_start': cn,
                    'caption_start_a87e195': co, 'lacks_blue': changed and bn < 0.99})
    return out


def caption_full(cl, adj, label):
    v = adj['labels'][label]
    text = '\n'.join(cl)
    pos = sum(len(x) + 1 for x in cl[:v['line'] - 1])
    m = re.compile(r'\\caption\{').search(text, pos)
    if not m:
        return None
    depth, j = 1, m.end()
    while j < len(text) and depth:
        if text[j] == '\\':
            j += 2
            continue
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        j += 1
    return ws(W.strip_blue(text[m.end():j - 1]))


def task_hygiene(ctx, texts):
    confirm = {f: [{'line': i, 'text': ws(x)[:160]} for i, x in enumerate(texts[f].split('\n'), 1) if '[CONFIRM' in x]
               for f in FILES}
    raw, cl = ctx['ln'], ctx['cl_n']
    tight_raw = [{'line': i, 'text': m.group(0), 'context': ws(x[max(0, m.start() - 60):m.end() + 40])}
                 for i, x in enumerate(raw, 1) for m in TIGHT.finditer(x)]
    tight_ts = [h for h in tight_raw if TIGHT.search(cl[h['line'] - 1])]
    tight_other = {f: sum(1 for x in texts[f].split('\n') for _ in TIGHT.finditer(x)) for f in FILES if f != 'main.tex'}
    gen = [{'line': i, 'word': m.group(1), 'context': ws(x[max(0, m.start() - 50):m.end() + 40])}
           for i, x in enumerate(cl, 1) for m in GENERAL.finditer(x)]
    w180 = [{'line': i, 'word': m.group(1), 'sep': m.group(2)} for i, x in enumerate(cl, 1)
            for m in H.W180C.GENERAL_PREFIX.finditer(x)]
    plus = plus_lines(_git(['diff', '-U0', PREV_FULL, HEAD_FULL, '--', 'main.tex'], cwd=CLONE))
    brows, bcounts = blue_rows(ctx['cl_o'], cl, raw, plus)
    caps = caption_rows(ctx)
    n_conf = sum(len(v) for v in confirm.values())
    nb_new = [r for r in brows if r['n_newly_written'] > 0]
    out = {'confirm': confirm, 'n_confirm': n_conf,
           'tightened_pattern': TIGHT.pattern, 'tightened_matches_raw': tight_raw, 'n_tightened_raw': len(tight_raw),
           'n_tightened_typeset': len(tight_ts), 'tightened_in_other_files': tight_other,
           'general_pattern': GENERAL.pattern, 'general_matches_typeset': gen, 'n_general': len(gen),
           'general_by_word': dict(Counter(g['word'] for g in gen)),
           'w180c_general_prefix_by_word': dict(Counter(f"{g['word']}{g['sep']!r}" for g in w180)),
           'blue_token_status_counts': bcounts, 'n_plus_lines_main': len(plus),
           'blue_changed_lines_lacking_blue': brows, 'n_lines_lacking_blue': len(brows),
           'n_lines_with_newly_written_nonblue': len(nb_new),
           'lines_lacking_blue_by_class': dict(Counter(r['class'] for r in brows)),
           'captions': caps, 'captions_changed_lacking_blue': [c for c in caps if c['lacks_blue']],
           'method_blue': 'W182c blue_compare(a87e195 typeset, HEAD typeset) gives each HEAD token a status; a token is '
                          'listed when its status is new / rewritten / unwrapped / moved_unwrapped, it holds content (a letter '
                          'or digit left after structural commands with their arguments and other control-word names are '
                          'removed), and under half of its alphanumeric characters (control-word names excluded) are inside a '
                          '\\textcolor{blue} wrapper; a token inside a 6-token run that occurs anywhere at a87e195 is "moved '
                          'verbatim", else "newly written"; class from the environments open at the line (scan from the top); '
                          'captions by W182c caption_blue on every float label'}
    # controls
    planted = list(texts['main.tex'].split('\n'))
    planted.insert(10, '% [CONFIRM: control]')
    pl2 = list(cl)
    pl2[-5] = pl2[-5] + ' Appendix~\\ref{app:results}'
    ctl = [{'id': 'h1 a planted "[CONFIRM" is counted', 'refused': sum('[CONFIRM' in x for x in planted) == 1 + len(confirm['main.tex'])},
           {'id': 'h2 a planted "Appendix~\\ref{app:" is matched', 'refused': sum(1 for x in pl2 for _ in TIGHT.finditer(x)) == len(tight_ts) + 1}]
    cap = next((c for c in caps if c['new_in_round9b'] and c['blue_share'] >= 0.99), None) or \
        next((c for c in caps if c['caption_changed_in_round9b'] and c['blue_share'] >= 0.99), None)
    if cap:
        k = cap['line_46c9d77'] - 1
        unw = list(ctx['cl_n'])
        span = next(i for i in range(k, min(k + 15, len(unw))) if '\\caption{' in unw[i])
        unw[span] = unw[span].replace('\\caption{\\textcolor{blue}{', '\\caption{{', 1)
        b2, _ = W.caption_blue(unw, cap['label'], ctx['adj_n'])
        ctl.append({'id': f'h3 the caption of {cap["label"]} unwrapped in a copy is detected', 'share_after': b2,
                    'refused': b2 is not None and b2 < 0.99})
    else:
        _fail('5 no blue changed caption to build the unwrap control')
    out['held'] = n_conf == 0 and len(tight_raw) == 0 and not out['captions_changed_lacking_blue'] and not nb_new
    return out, ctl


# ======================================================================================================================
#  6. counts
# ======================================================================================================================
COUNTING_RULE = ('comment-stripped source (W167 strip_comments); regions: main text = \\section{Introduction} .. '
                 '\\appendix; Appendix A = its \\section .. the next appendix \\section; Appendix B = its \\section .. the '
                 'unnumbered declaration section (or \\bibliography); abstract counted apart; float environments (table, '
                 'figure, longtable, algorithm, starred forms) removed and their \\caption words counted apart; display '
                 'math (equation, align, gather, multline, eqnarray, displaymath, subequations content) removed; inline '
                 'math ($..$, $$..$$, \\(..\\), \\[..\\]) removed; heading titles removed and counted apart; arguments of '
                 '\\ref, \\eqref, \\cite*, \\label, \\url, \\includegraphics, \\begin, \\end, \\hspace, \\vspace and the colour '
                 'argument of \\textcolor removed; every other control word removed with the text of its arguments kept; '
                 'braces, ~ and escaped characters become spaces; a word is a whitespace-separated token holding a letter '
                 'or digit (hyphenated compounds count once)')
FLOAT_ENVS = ('table', 'table*', 'figure', 'figure*', 'longtable', 'algorithm')
MATH_ENVS = ('equation', 'equation*', 'align', 'align*', 'gather', 'gather*', 'multline', 'multline*', 'eqnarray',
             'eqnarray*', 'displaymath', 'subequations')


def brace_arg(text, pos):
    """text[pos] == '{' -> index after the matching '}'."""
    depth, j = 0, pos
    while j < len(text):
        c = text[j]
        if c == '\\':
            j += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(text)


def cut_envs(text, envs):
    blocks = []
    for env in envs:
        b, e = '\\begin{' + env + '}', '\\end{' + env + '}'
        while True:
            i = text.find(b)
            if i < 0:
                break
            depth, j = 0, i
            while True:
                nb, ne = text.find(b, j), text.find(e, j)
                if ne < 0:
                    ne = len(text) - len(e)
                    break
                if 0 <= nb < ne:
                    depth += 1
                    j = nb + len(b)
                else:
                    depth -= 1
                    j = ne + len(e)
                    if depth == 0:
                        break
            end = min(len(text), ne + len(e))
            blocks.append(text[i:end])
            text = text[:i] + ' ' + text[end:]
    return text, blocks


def captions_of(block):
    caps = []
    for m in re.finditer(r'\\caption(?:\[[^\]]*\])?\{', block):
        k = m.end() - 1
        caps.append(block[k + 1:brace_arg(block, k) - 1])
    return caps


def words_of(text):
    t = text
    t = re.sub(r'\$\$.*?\$\$', ' ', t, flags=re.S)
    t = re.sub(r'(?<!\\)\$.*?(?<!\\)\$', ' ', t, flags=re.S)
    t = re.sub(r'\\\(.*?\\\)', ' ', t, flags=re.S)
    t = re.sub(r'\\\[.*?\\\]', ' ', t, flags=re.S)
    t = re.sub(r'\\(?:eqref|ref|autoref|cref|Cref|pageref|nameref|cite[a-z]*|label|url|includegraphics|begin|end|hspace\*?|'
               r'vspace\*?|bibliography|bibliographystyle|addtocounter|setcounter|setlength)(?:\[[^\]]*\])*\{[^}]*\}(?:\{[^}]*\})?',
               ' ', t)
    t = re.sub(r'\\(?:textcolor|color)\{[^}]*\}', ' ', t)
    t = re.sub(r'\\[A-Za-z@]+\*?', ' ', t)
    t = re.sub(r'\\.', ' ', t)
    t = re.sub(r'[{}~\[\]]', ' ', t)
    return [w for w in t.split() if re.search(r'[A-Za-z0-9]', w)]


def split_headings(text):
    heads = []
    out, pos = [], 0
    for m in re.finditer(r'\\(?:sub)*section\*?\{', text):
        if m.start() < pos:
            continue
        k = m.end() - 1
        e = brace_arg(text, k)
        out.append(text[pos:m.start()])
        heads.append(text[k + 1:e - 1])
        pos = e
    out.append(text[pos:])
    return ' '.join(out), heads


def region_counts(lines):
    text = '\n'.join(lines)
    text, floats = cut_envs(text, FLOAT_ENVS)
    text, maths = cut_envs(text, MATH_ENVS)
    text, heads = split_headings(text)
    caps = [c for b in floats for c in captions_of(b)]
    return {'prose_words': len(words_of(text)), 'heading_words': sum(len(words_of(h)) for h in heads),
            'n_headings': len(heads), 'caption_words': sum(len(words_of(c)) for c in caps), 'n_float_blocks': len(floats),
            'n_display_math_blocks': len(maths), 'source_lines': len(lines), 'typeset_chars': len(''.join(lines))}


def regions(lines, adj):
    secs = [o for o in adj['objects'] if o['kind'] == 'section']
    intro = next(o['line'] for o in secs if o['number'] == '1')
    app = next(i for i, x in enumerate(lines, 1) if re.match(r'\s*\\appendix\b', x))
    appx = [o for o in secs if o['number'] and re.fullmatch(r'[A-Z]', str(o['number']))]
    tail = next((o['line'] for o in secs if o['number'] is None and o['line'] > app),
                next(i for i, x in enumerate(lines, 1) if '\\bibliography{' in x))
    abs_b = next(i for i, x in enumerate(lines, 1) if '\\begin{abstract}' in x)
    abs_e = next(i for i, x in enumerate(lines, 1) if '\\end{abstract}' in x)
    reg = {'abstract': (abs_b, abs_e + 1), 'main': (intro, app)}
    for k, o in enumerate(appx):
        end = appx[k + 1]['line'] if k + 1 < len(appx) else tail
        reg[f'appendix {o["number"]}'] = (o['line'], end)
    return reg, [o['number'] for o in appx], app


def floats_in(inn, lo, hi):
    c = Counter()
    for (kind, _n), o in inn.items():
        if lo <= o['line'] < hi and kind in ('table', 'figure', 'algorithm'):
            c[kind] += 1
    return dict(c)


def equations_in(adj, lo, hi):
    eq = [o for o in adj['objects'] if o['kind'].startswith('equation') and o['number'] and lo <= o['line'] < hi]
    nums = [o['number'] for o in eq]
    base = sorted({re.sub(r'[a-z]$', '', str(n)) for n in nums}, key=W.natkey)
    return {'numbered_rows': len(nums), 'distinct_numbers': len(base), 'first': base[0] if base else None,
            'last': base[-1] if base else None}


def counts_for(lines, adj, inn):
    cl = W.typeset(lines)
    reg, appx, app_line = regions(cl, adj)
    out = {'regions_lines': {k: list(v) for k, v in reg.items()}, 'appendices': appx, 'n_appendices': len(appx)}
    for name, (lo, hi) in reg.items():
        rc = region_counts(cl[lo - 1:hi - 1])
        rc['floats'] = floats_in(inn, lo, hi)
        rc['equations'] = equations_in(adj, lo, hi)
        out[name] = rc
    m = out['main']
    out['main_text_floats_tables_plus_figures'] = m['floats'].get('table', 0) + m['floats'].get('figure', 0)
    out['main_text_algorithms'] = m['floats'].get('algorithm', 0)
    out['target_le_10_floats_met'] = out['main_text_floats_tables_plus_figures'] <= 10
    return out


def task_counts(ctx):
    head = counts_for(ctx['ln'], ctx['adj_n'], ctx['inn'])
    prev = counts_for(ctx['lo'], ctx['adj_o'], ctx['io'])
    # controls on a copy of the HEAD main region
    cl = W.typeset(ctx['ln'])
    lo, hi = head['regions_lines']['main']
    base = region_counts(cl[lo - 1:hi - 1])['prose_words']
    seg = list(cl[lo - 1:hi - 1])
    seg.insert(5, 'Seven plain words are added right here $x_{t} + y^{2}$.')
    plus7 = region_counts(seg)['prose_words'] - base
    L = list(ctx['ln'])
    k = next(i for i, x in enumerate(L) if '\\section{Conclusions}' in x)
    L2 = L[:k] + ['\\begin{equation}', 'a = b', '\\end{equation}'] + L[k:]
    _, a2 = W._num(L2)
    e2 = equations_in(a2, 1, k + 4)['numbered_rows'] - equations_in(ctx['adj_n'], 1, k + 1)['numbered_rows']
    ctl = [{'id': 'n1 a 7-word sentence with inline math adds exactly 7 prose words', 'delta': plus7, 'refused': plus7 == 7},
           {'id': 'n2 an added numbered equation adds one numbered row to the main text', 'delta': e2, 'refused': e2 == 1}]
    return {'counting_rule': COUNTING_RULE, 'head_46c9d77': head, 'a87e195': prev, 'stated': STATED_COUNTS,
            'pages': 'NOT estimable statically: no TeX on this machine, no .aux/.log/.pdf of the round-9b compile in the '
                     'repository or the clone; the expert reports 63 pages (STEP6), not confirmed; source lines and '
                     'typeset characters per region are given as proxies only, not as pages'}, ctl


# ======================================================================================================================
#  checklist, guards, output
# ======================================================================================================================
REQUIRED_CAPTURES = {
    '1 identity of the four files, the clone diff vs the expert diffs, Figure 6 PDF': 'task_identity',
    '2 label map a87e195 -> 46c9d77 vs the stated numbering': 'task_label_map',
    '3 letter v12 pointers (v10 chain)': 'task_pointers',
    '3 Table 7 at R2.6 / benchmark bullet': 'table7_check',
    '4 dangling references, prose mentions': 'task_dangling',
    '5 CONFIRM, prefixes, blue': 'task_hygiene',
    '6 floats, equations, words, appendices': 'task_counts',
    'negative controls (pointers)': 'controls_pointers',
    'guards (solve 0, pickle 0)': 'verify_guards',
}
REQUIRED_SYMBOLS = {
    'W183h': (H, ('all_pointers', 'lettered_pointers', 'pair_and_rate', 'line_map', 'DECLARED_TARGETS', 'LETTER_V9',
                  'all_guards', 'W180C', 'IMPORTED_SCRIPTS', 'W182C_JSON', 'W180C_JSON')),
    'W182c': (W, ('task_label_map', 'pointer_rows', 'now_denotes', 'heading_at', '_num', 'objects_index', 'heading_map',
                  'eq_bodies', 'label_desc', 'label_class', 'blue_compare', 'blue_mask_text', 'caption_blue', 'caption_or_env',
                  'strip_blue', 'typeset', 'show', 'task_dangling', 'heading_own_text', 'range_members', 'old_objects_of', 'resolve_identity',
                  'natkey', 'REF_RX', 'HEADING_KINDS', 'FLOAT_KINDS', '_clean_desc', '_obj_str', 'OLD', 'ROUND7',
                  'W180C_HEAD', 'LETTER_NAME', 'CHECK_FAILURES')),
    'W181a': (A, ('parse_udiff', 'compare_hunks')),
    'W177b-3prime': (R, ('letter_pointers', '_blank_cmd', 'cite_order')),
    'W178c': (C_, ('task_refs',)),
    'W179b': (B, ('task_elsarticle', 'FORBIDDEN_MODULE_PREFIXES')),
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
    need = ([os.path.join(CLONE, f) for f in FILES] + [os.path.join(CLONE, FIG6), os.path.join(MR, FIG6),
            os.path.join(CLONE, 'elsarticle.dtx')] + list(REVIEW_COPY.values()) + list(PREV_COPY.values())
            + list(EXPERT_DIFF.values()) + [LETTER_V9, LETTER_V10, SUBMITTED_MAIN, W183H_JSON, W182C_JSON, W180C_JSON,
                                            STEP6_R9, BRIEF])
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def all_guards():
    gs = [('w189p2', GUARD), ('w183h', H.GUARD)] + list(H.all_guards())
    seen, out = set(), []
    for nm, gd in gs:
        if id(gd) in seen:
            continue
        seen.add(id(gd))
        out.append((nm, gd))
    return out


def verify_guards():
    g = {f'{nm}#{i}': {'counts': dict(gd.counts), 'verify_0_failures': gd.verify(0)} for i, (nm, gd) in enumerate(all_guards())}
    pk = {'w189p2': dict(PICKLE_COUNTS)}
    for nm, mod in (('w183h', H), ('w182c', W), ('w181a', A), ('w179b', B), ('w178c', C_), ('w177b3r', R),
                    ('w180c', H.W180C), ('w182b', H.NB), ('w178b', H.W178B), ('w177b1r', H.B1R)):
        if hasattr(mod, 'PICKLE_COUNTS'):
            pk[nm] = dict(mod.PICKLE_COUNTS)
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(B.FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    ok = (all(not v['verify_0_failures'] for v in g.values()) and blocked
          and all(v == {'load': 0, 'loads': 0} for v in pk.values()) and not forbidden)
    if not ok:
        _fail(f'guards: {g} {pk} blocked {blocked} forbidden {forbidden}')
    return {'guards': g, 'n_guards': len(g), 'pickle_counts': pk, 'pickle_blocked_at_end': blocked,
            'production_modules_imported': forbidden, 'ok': ok}


def _md_row(*cells):
    return '| ' + ' | '.join(str(c).replace('|', '/').replace('\n', ' ') for c in cells) + ' |'


def write_md(out, path):
    I, Lm, P, D, Hy, Cn = (out['1_identity'], out['2_label_map'], out['3_pointers'], out['4_dangling'], out['5_hygiene'],
                           out['6_counts'])
    L = [f'# W189 part 2 — round 9b structure audit at Overleaf {I["clone_head"][:7]} (zero solves)', '',
         f'HEAD `{I["clone_head"]}` (parent `{I["clone_head_parent"]}`), {I["clone_head_date"]}. Static numbering (no TeX).',
         'No objective value is computed in this stage.', '', '## 1 Identity', '',
         _md_row('file', 'sha256', 'md5', 'declared md5', '= review copy', 'hunks clone/expert', '+/- equal', 'ranges', 'context',
                 'a87e195 = round-8 copy'), '|---|---|---|---|---|---|---|---|---|---|']
    for f, v in I['outcome'].items():
        fi = I['files'][f]
        L.append(_md_row(f, f'`{fi["sha256"][:12]}`', f'`{fi["md5"][:8]}`', fi['md5_declared'], v['byte_identical_to_review_copy'],
                         v['hunks_clone_expert'], v['every_hunk_pm_equal'], v['ranges_equal'], v['context_equal'],
                         v['a87e195_blob_is_round8_deliverable']))
    L += ['', f'- Figure 6 PDF: {I["figure6_pdf"]}', f'- files changed a87e195 -> HEAD: {I["files_changed_a87e195_head"]}',
          f'- held: **{I["held"]}**', '', '## 2 Label map', '',
          f'- labels {Lm["n_labels_a87e195"]} -> {Lm["n_labels_46c9d77"]}; {Lm["counts"]}; by class {Lm["by_class"]}',
          f'- disagree with the stated numbering: **{Lm["n_disagree_with_stated"]}**; titles {Lm["stated_titles"]}',
          f'- main-text equation max number {Lm["main_text_equation_max_number"]} (stated: {Lm["stated_main_equations"]})',
          f'- citations: order identical {Lm["citations"]["order_identical"]}; renumbered {Lm["citations"]["renumbered"]}', '',
          _md_row('label', 'class', 'a87e195', '46c9d77', 'status', 'stated', 'agrees', 'caption / title at HEAD'),
          '|---|---|---|---|---|---|---|---|']
    for r in Lm['rows']:
        if r['status'] != 'unchanged' or not r['agrees_with_stated']:
            L.append(_md_row(f'`{r["label"]}`', r['class'], r['number_a87e195'], r['number_46c9d77'], r['status'],
                             f'{r["stated_a87e195"]} -> {r["stated_46c9d77"]}', r['agrees_with_stated'],
                             (r['desc_46c9d77'] or r['desc_a87e195'] or '')[:70]))
    L += ['', 'Headings (status not unchanged):', '', _md_row('kind', 'a87e195', 'title', '46c9d77', 'title', 'method', 'status'),
          '|---|---|---|---|---|---|---|']
    for h in Lm['heading_map']:
        if h['status'] != 'unchanged':
            L.append(_md_row(h['kind'], h['number_old'], (h['title_old'] or '')[:45], h['number_new'], (h['title_new'] or '')[:45],
                             h['method'], h['status']))
    L += ['', '## 3 Letter v12 pointers', '',
          f'- v10 at a87e195 re-derived: {P["v10_at_a87e195"]}', f'- v10 at HEAD: {P["v10_at_head_counts"]}',
          f'- **v12: {P["v12_counts"]}** (by extractor {P["v12_by_extractor"]})',
          f'- stale declarations: {P["stale_declarations"]}', f'- Table 7: {json.dumps(out["3_table7"], ensure_ascii=False)[:900]}', '',
          _md_row('line', 'pointer', 'verdict', 'expected at HEAD', 'v10 line', 'note / why'), '|---|---|---|---|---|---|']
    for r in P['v12']:
        if r['verdict'] != 'MATCH' or r.get('declared_target') or r.get('note'):
            L.append(_md_row(r['line'], r['text'][:40], r['verdict'], r.get('expected_at_head'), r.get('line_v10'),
                             str(r.get('why') or r.get('note') or '')[:110]))
    L += ['', f'- v10 pointers dropped in v12: {[(d["line"], d["text"]) for d in P["v10_pointers_dropped_in_v12"]]}', '',
          '## 4 Dangling', '', f'- undefined {D["n_undefined"]} {D["refs_static_46c9d77"]["undefined_refs"]}; duplicate labels '
          f'{D["n_duplicate_labels"]}; uses of removed labels typeset {len(D["uses_of_removed_labels_typeset"])}, in comments '
          f'{len(D["uses_of_removed_labels_in_comments"])}',
          f'- kind-word mismatches {len(D["w182c_task_dangling"]["kind_word_mismatch_46c9d77"])}; literal pointers '
          f'{len(D["w182c_task_dangling"]["literal_pointers_main_tex_46c9d77"])}',
          f'- supplementary pointers {D["supplementary_pointers"]}', '']
    for name, v in D['prose_mentions_round9b'].items():
        L.append(f'- {name}: main {len(v["main.tex"])}, letter {len(v["letter"])}')
        for h in (v['main.tex'][:12]):
            L.append(f'  - main l. {h["line"]}: "{h["match"]}" — {h["context"][:150]}')
        for h in (v['letter'][:6]):
            L.append(f'  - letter l. {h["line"]}: "{h["match"]}" — {h["context"][:150]}')
    L += ['', '## 5 Hygiene', '', f'- [CONFIRM: {Hy["n_confirm"]}; tightened pattern raw {Hy["n_tightened_raw"]} / typeset '
          f'{Hy["n_tightened_typeset"]}; other files {Hy["tightened_in_other_files"]}',
          f'- general `[A-Za-z]+~\\ref{{app:` by word: {Hy["general_by_word"]}',
          f'- blue: token statuses {Hy["blue_token_status_counts"]}; changed lines lacking blue {Hy["n_lines_lacking_blue"]} '
          f'(with newly written text {Hy["n_lines_with_newly_written_nonblue"]}); by class {Hy["lines_lacking_blue_by_class"]}',
          f'- captions changed lacking blue: {[(c["label"], c["blue_share"]) for c in Hy["captions_changed_lacking_blue"]]}', '',
          _md_row('line', '+ line', 'class', 'where', 'non-blue changed', 'moved', 'new', 'first words'),
          '|---|---|---|---|---|---|---|---|']
    for r in Hy['blue_changed_lines_lacking_blue']:
        L.append(_md_row(r['line'], r['on_plus_line'], r['class'], r['where'][:50], r['n_nonblue_changed'],
                         r['n_moved_verbatim_from_a87e195'], r['n_newly_written'], r['first_nonblue_words'][:100]))
    L += ['', '## 6 Counts', '', f'Rule: {Cn["counting_rule"]}', '', f'Pages: {Cn["pages"]}', '',
          _md_row('revision', 'region', 'prose words', 'heading words', 'caption words', 'floats', 'equations', 'lines'),
          '|---|---|---|---|---|---|---|---|']
    for rev in ('head_46c9d77', 'a87e195'):
        c = Cn[rev]
        for name in [k for k in c if k == 'abstract' or k == 'main' or k.startswith('appendix ')]:
            v = c[name]
            L.append(_md_row(rev, name, v['prose_words'], v['heading_words'], v['caption_words'], v['floats'], v['equations'],
                             v['source_lines']))
        L.append(f'- {rev}: main-text tables + figures {c["main_text_floats_tables_plus_figures"]} (algorithms '
                 f'{c["main_text_algorithms"]}); appendices {c["appendices"]}; target <= 10 met: {c["target_le_10_floats_met"]}')
    L += ['', f'Negative controls: {[(c["id"][:60], c["refused"]) for c in out["negative_controls"]]}', '',
          f'Guards ok {out["guards"]["ok"]} ({out["guards"]["n_guards"]} guards); check failures {out["check_failures"]}']
    with open(path, 'x', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def post_run(out_dir):
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    p = os.path.join(od, OUT_NAMES['post'])
    if os.path.exists(p):
        print(f'REFUSED: {p} exists (write-once)', flush=True)
        sys.exit(1)
    names = [OUT_NAMES[k] for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
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
    _log(f'W189p2: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W189 part 2 (Addenda 89-90): round-9b audit of identity, labels, pointers, structure '
                                    'and counts at the Overleaf HEAD 46c9d77',
           'objective_convention': 'no objective value is computed in this stage',
           'static_numbering': 'no TeX installation on this machine; numbering is static (W182c method); nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'script_committed_clean': committed_clean(SCRIPT_NAME),
           'imported': {m: _sha(m) for m in IMPORTED_SCRIPTS}}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    out['1_identity'], texts = task_identity()
    _log(f"1 identity: held {out['1_identity']['held']}; {out['1_identity']['outcome']}")
    ctx = build_ctx(show(PREV_FULL, 'main.tex').split('\n'), texts['main.tex'].split('\n'))
    out['2_label_map'] = lab = task_label_map(ctx)
    _log(f"2 labels {lab['n_labels_a87e195']} -> {lab['n_labels_46c9d77']} {lab['counts']}; disagree with stated "
         f"{lab['n_disagree_with_stated']}: {[(r['label'], r['number_46c9d77'], r['stated_46c9d77']) for r in lab['disagree_with_stated']]}")
    sub_lines = _rd(SUBMITTED_MAIN).decode('utf-8').split('\n')
    _, adj_s = _num(sub_lines)
    sub_ctx = {'inn': W.objects_index(adj_s), 'n_labels': len(adj_s['labels'])}
    out['3_pointers'] = P = task_pointers(ctx, texts, sub_ctx)
    out['3_table7'] = table7_check(P['v12'], texts)
    _log(f"3 v10@a87e195 {P['v10_at_a87e195']['counts']} (= committed W183h {P['v10_at_a87e195']['equals_committed_w183h']}); "
         f"v10@HEAD {P['v10_at_head_counts']}; v12 {P['v12_counts']}; stale {P['stale_declarations']}")
    for r in P['v12_mismatch'] + P['v12_ambiguous']:
        _log(f"   {r['verdict']} l. {r['line']} {r['text']!r} kind {r['kind']} printed {r['numbers']} expected "
             f"{r.get('expected_at_head')} v10 line {r.get('line_v10')} note {r.get('note')}")
    out['4_dangling'], ctl_d = task_dangling(ctx, lab, texts)
    _log(f"4 undefined {out['4_dangling']['n_undefined']}; duplicates {out['4_dangling']['n_duplicate_labels']}; removed-label "
         f"uses {len(out['4_dangling']['uses_of_removed_labels_typeset'])}")
    out['5_hygiene'], ctl_h = task_hygiene(ctx, texts)
    _log(f"5 CONFIRM {out['5_hygiene']['n_confirm']}; tightened {out['5_hygiene']['n_tightened_raw']}; general "
         f"{out['5_hygiene']['general_by_word']}; lines lacking blue {out['5_hygiene']['n_lines_lacking_blue']}; captions "
         f"{[(c['label'], c['blue_share']) for c in out['5_hygiene']['captions_changed_lacking_blue']]}")
    out['6_counts'], ctl_n = task_counts(ctx)
    hc = out['6_counts']['head_46c9d77']
    _log(f"6 main floats {hc['main_text_floats_tables_plus_figures']} (alg {hc['main_text_algorithms']}); words main "
         f"{hc['main']['prose_words']} A {hc.get('appendix A', {}).get('prose_words')} B {hc.get('appendix B', {}).get('prose_words')}")
    ctl = ([{'id': f'1 {f}: expert diff + line altered', 'refused': v['refused']}
            for f, v in out['1_identity']['diff_negative_controls'].items()]
           + controls_label(ctx) + controls_pointers(ctx, texts, sub_ctx, P['v12']) + ctl_d + ctl_h + ctl_n)
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f'negative control not refused: {c["id"]}')
    if P['stale_declarations']:
        _fail(f'3 stale declarations: {P["stale_declarations"]}')
    if not P['declaration_checks_all_hold']:
        _fail(f'3 declaration checks: {[k for k, v in P["declaration_checks"].items() if not v["holds"]]}')
    _log(f"negative controls: {[(c['id'][:44], c['refused']) for c in ctl]}")
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
                        + list(EXPERT_DIFF.values()) + [LETTER_V9, LETTER_V10, SUBMITTED_MAIN, W183H_JSON, W182C_JSON,
                                                        W180C_JSON, STEP6_R9, BRIEF, SCRIPT_NAME] + list(IMPORTED_SCRIPTS)))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md')},
                'out_dir': out_dir, 'clone_head_full': out['1_identity']['clone_head'],
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(show(c, f).encode('utf-8')) for c in (PREV_FULL, HEAD_FULL) for f in FILES},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(CHECK_FAILURES)}; guards ok {out["guards"]["ok"]}; pickle {PICKLE_COUNTS}')
    sys.exit(3 if CHECK_FAILURES else 0)


def controls_label(ctx):
    """An injected captioned table before Table 5 shifts tab:res_energy_ladder_m2 to 6 and the stated comparison flags it."""
    L = list(ctx['ln'])
    k = ctx['adj_n']['labels']['tab:res_energy_ladder_m2']['line'] - 1
    beg = next(i for i in range(k, -1, -1) if '\\begin{table' in L[i])
    L2 = L[:beg] + ['\\begin{table}', '\\caption{Control.}', '\\label{tab:w189p2_control}', '\\end{table}'] + L[beg:]
    c2 = build_ctx(ctx['lo'], L2)
    lab2 = task_label_map(c2)
    row = next(r for r in lab2['rows'] if r['label'] == 'tab:res_energy_ladder_m2')
    _ADJ_O['adj'] = ctx['adj_o']
    return [{'id': 'l1 an injected table shifts tab:res_energy_ladder_m2 5 -> 6 and is flagged against the stated map',
             'number_after': row['number_46c9d77'], 'refused': row['number_46c9d77'] == '6' and not row['agrees_with_stated']}]


if __name__ == '__main__':
    main()
