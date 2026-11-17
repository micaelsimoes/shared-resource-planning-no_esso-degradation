"""P5.15 Addendum 92 (Planner order, W190b line) -- Planner task W190b PART 2: THE INTERIM AUDIT AT THE OVERLEAF HEAD
5f4d3bd -- (iii) LETTER POINTERS AFTER THE AUTHOR'S EDITS, (iv) BLUE WRAPPERS a87e195 -> HEAD, (vi) STRUCTURE (CONFIRM,
PREFIXES, HARD-CODED REFERENCES, UNDEFINED / DUPLICATE LABELS), (vii) COUNTS. ZERO SOLVES, NO MODEL, NO PICKLE, NO
PRODUCTION IMPORT. Read-only on the Overleaf clone (git rev-parse / show / diff / status / log only), on manuscript_review/,
on manuscript_submitted/, on every committed artifact and on every earlier script. Part 1 (W187 numbers, numbers,
formatting, Appendix B gate) is p515_s55_w190b_numbers.py.

NUMBERING IS STATIC (W182c method, as W189 part 2 / W190a part 2). No TeX installation exists on this machine.

REUSE, NOT REIMPLEMENTATION. As W190a: the W189 part-2 harness p515_s55_w189_p2_structure.py (P2) and the W190a part-2
harness p515_s55_w190a_structure.py (S90) are IMPORTED, NOT EDITED and NOT RUN (their main() is never called; nothing is
written into their output directories). Two static contexts are built with P2.build_ctx:
  ctx_A = (a87e195 -> 5f4d3bd): the W189 pointer chain (letter v10 at a87e195, re-rated at HEAD, paired with the letter at
          HEAD) with P2.DECLARED_TARGETS_V12 / DECLARED_SUBMITTED_V12 RE-BOUND to (a) W189's declarations moved to their
          HEAD line numbers (H.line_map of the letter 3c379fa -> HEAD; each declared line must map) and (b) new
          declarations for the pointers whose letter line has no v10 counterpart since the author's edits (each with its
          reason, a content needle in the target region where the sentence names content, and the v10 line of its
          3c379fa counterpart as a cross-check where one exists); dangling references; blue of every change since a87e195
          (P2.task_hygiene), the multicolumn reading of S90.multicolumn_wrapped for the lines the token rule lists;
  ctx_B = (3c379fa -> 5f4d3bd): the label map with the expectation "every label identical to 3c379fa" (P2.STATED_NUMBERING
          re-bound to {}), S90.numbering_compare against the committed W189 numbers; blue of the lines changed since
          3c379fa (S90.line_blue); counts (P2.task_counts) against the committed W190a counts at 3c379fa.
The hard-coded references are the W182c literal-pointer list (P2.task_dangling -> W182c task_dangling), placed by region
(P2.regions); a complementary scan lists lettered (B.n) and supplementary (S<n>) typed pointers, which W182c's pattern
does not match.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0 together with
S90's, P2's and every guard of their chain (P2.verify_guards: pickle.load / loads blocked, counters 0, no production
module); refuses to run concurrently with another copy of itself; refuses unless the output directory exists and holds
nothing but launch.log; capture-path checklist asserted before any task (exit 4); JSON written only through gate_result_io.

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir w190b_audit/structure && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_structure.py \\
      > w190b_audit/structure/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_structure.py --post-run
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W190b part 2 structure audit at Overleaf 5f4d3bd '
                                               '(never solves)').install()

import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s55_w190a_structure as S90  # noqa: E402  (W190a part 2; imports W189 part 2; NOT edited, main() not run)
import gate_result_io as GRIO  # noqa: E402

P2 = S90.P2
W, A, H = P2.W, P2.A, P2.H
os.chdir(REPO)

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s55_w190b_structure'
OUT_DIR = os.path.join('w190b_audit', 'structure')
OUT_NAMES = {'json': 'w190b_structure.json', 'md': 'w190b_structure.md', 'man': 'manifest_sha256.json', 'log': 'launch.log',
             'post': 'manifest_post_run_sha256.json'}
CLONE = P2.CLONE
HEAD_FULL = '5f4d3bd4c96641c40a42829ae837fd954ec04954'
PREV_FULL = '3c379fa44f3755dbdc6fd540df566b55527d921c'
A87_FULL = P2.PREV_FULL
MR = 'manuscript_review'
LETTER_NAME = P2.LETTER_NAME
FILES = P2.FILES
W189P2_JSON = S90.W189P2_JSON
W190A_JSON = os.path.join('w190a_audit', 'structure', 'w190a_structure.json')
W190A_EXPECTED = {'floats': 17, 'equations': 41, 'main_words': 6712, 'labels': 102, 'pointers_submitted': 8}
IMPORTED_SCRIPTS = ('p515_s55_w190a_structure.py',) + S90.IMPORTED_SCRIPTS
CHECK_FAILURES = []

# New declared targets: pointers on letter lines whose v10 pairing the author's edits broke (H.line_map maps only equal
# blocks and equal-size replace blocks; the 1 -> 3 line splits of l. 58 and l. 261 at 3c379fa leave l. 59-60 and the R3.6
# block 262-266 without a v10 line).  Key: (HEAD line, kind, printed).  'prev_line' = the 3c379fa line the text comes
# from (its v10 line, when it has one, is the cross-check).  Each target from the letter's own sentence.
NEW_DECLARED = {
    (59, 'section', '4.3'): {'label': 'subsec:res_degradation', 'needle': 'The planning question was also tested directly',
                             'prev_line': 58,
                             'why': '"What we did not do": "the ageing comparison of Section~4.3 re-evaluates a fixed plan" and '
                                    '"a screen of the fourteen unit neighbours ... (Section~4.3)" -- the W187 paragraph of 4.3'},
    (59, 'section', '4.7'): {'label': 'subsec:res_certification', 'needle': 'without repeating the planning search',
                             'prev_line': 58,
                             'why': '"a full degradation-neutral search over the lattice was not run, and the paper says so '
                                    '(Section~4.7)" -- the 4.7 scope paragraph'},
    (60, 'section', '4.7'): {'label': 'subsec:res_certification', 'prev_line': 58,
                             'why': '"The largest instance of the submitted version ... is stated as future work (Section~4.7)" '
                                    '-- the sentence moved unchanged from 3c379fa l. 58'},
    (60, 'section', '4.5'): {'label': 'subsec:res_multiscenario', 'needle': 'three market scenarios', 'prev_line': 58,
                             'why': '"the multi-scenario results are given on a 3x3 scenario instance (Section~4.5)" -- moved '
                                    'unchanged from 3c379fa l. 58'},
    (262, 'figure', '5'): {'label': 'fig:res_ageing', 'prev_line': 260,
                           'why': 'R3.6 response opening "(Table~6 and Figure~5)": unchanged since 3c379fa (l. 260)'},
    (262, 'table', '6'): {'label': 'tab:res_ageing', 'prev_line': 260,
                          'why': 'R3.6 response opening "(Table~6 and Figure~5)": unchanged since 3c379fa (l. 260)'},
    (263, 'table', '6'): {'label': 'tab:res_ageing', 'prev_line': 261,
                          'why': 'R3.6 "(Table~6, last row)": the end-of-life floor row, from 3c379fa l. 261'},
    (263, 'section', '4.3'): {'label': 'subsec:res_degradation', 'needle': 'The planning question was also tested directly',
                              'prev_line': 261, 'new_pointer': True,
                              'why': 'R3.6 "... and the unit-neighbourhood screen its effect on the selected plan (Section~4.3)" '
                                     '-- the W187 paragraph (was "(Section~4.7)" at 3c379fa l. 261)'},
    (264, 'section', '4.3'): {'label': 'subsec:res_degradation', 'needle': 'The planning question was also tested directly',
                              'prev_line': 261, 'new_pointer': True,
                              'why': 'R3.6 "evaluated under the no-ageing calibration as a planning search would evaluate them '
                                     '(Section~4.3)" -- the W187 paragraph'},
    (265, 'table', '6'): {'label': 'tab:res_ageing', 'prev_line': 261,
                          'why': 'R3.6 "Table~6 gives, per calibration, the year in which the end-of-life floor binds" -- from '
                                 '3c379fa l. 261'},
    (266, 'figure', '5'): {'label': 'fig:res_ageing', 'prev_line': 262,
                           'why': 'R3.6 \\rchanges{Section~4.3; Table~6; Figure~5.}, unchanged since 3c379fa (l. 262)'},
    (266, 'section', '4.3'): {'label': 'subsec:res_degradation', 'prev_line': 262,
                              'why': 'R3.6 \\rchanges, unchanged since 3c379fa (l. 262)'},
    (266, 'table', '6'): {'label': 'tab:res_ageing', 'prev_line': 262,
                          'why': 'R3.6 \\rchanges, unchanged since 3c379fa (l. 262)'},
}


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


# ======================================================================================================================
#  (iii) the declarations at HEAD
# ======================================================================================================================
def build_declarations():
    prev_l = show(PREV_FULL, LETTER_NAME)
    head_l = show(HEAD_FULL, LETTER_NAME)
    mp = H.line_map(prev_l, head_l)                                   # 3c379fa line -> HEAD line
    v10 = _rd(P2.LETTER_V10).decode('utf-8')
    to10 = {v: k for k, v in H.line_map(v10, prev_l).items()}        # 3c379fa line -> v10 line
    moved, problems = {}, []
    for src, name in ((P2.DECLARED_TARGETS_V12, 'targets'), (P2.DECLARED_SUBMITTED_V12, 'submitted')):
        for k, v in src.items():
            if k[0] not in mp:
                problems.append(f'W189 declared {name} line {k[0]} has no HEAD line')
                continue
            moved[(name, (mp[k[0]],) + k[1:])] = (k, v)
    targets = {k: dict(v) for (n, k), (_o, v) in moved.items() if n == 'targets'}
    submitted = {k: v for (n, k), (_o, v) in moved.items() if n == 'submitted'}
    new = {}
    for k, v in NEW_DECLARED.items():
        d = {x: y for x, y in v.items() if x not in ('prev_line', 'new_pointer')}
        l10 = to10.get(v['prev_line'])
        if l10 is not None and not v.get('new_pointer'):
            d['v10_line'] = l10
        elif v.get('new_pointer'):
            d['why'] += (' -- a NEW pointer (the 3c379fa sentence pointed to Section~4.7; the v10 cross-check does not apply, '
                         'the needle checks the target)')
        d['why'] = d['why'] + (f' [3c379fa l. {v["prev_line"]}; v10 l. {l10}]' if l10 else f' [3c379fa l. {v["prev_line"]}; no v10 line]')
        if k in targets:
            problems.append(f'new declaration {k} collides with a moved W189 declaration')
        new[k] = d
    targets.update(new)
    info = {'w189_moved': [{'w189_key': list(o), 'head_key': list(k), 'set': n} for (n, k), (o, _v) in moved.items()],
            'new_declarations': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in new.items()}, 'problems': problems,
            'letter_line_map_3c379fa_to_head_size': len(mp)}
    for p in problems:
        _fail(f'(iii) {p}')
    return targets, submitted, info


# ======================================================================================================================
#  (iv) blue
# ======================================================================================================================
def task_blue_b(ctxB, hyA, hyB):
    byB, aligned = S90.line_blue(ctxB)
    plus = sorted(P2.plus_lines(_git(['diff', '-U0', PREV_FULL, HEAD_FULL, '--', 'main.tex'], cwd=CLONE)))
    changed = []
    for ln in plus:
        t = byB.get(ln, [])
        ct = [(x, s) for x, s, st in t if st in P2.NONBLUE_CHANGED + ('new_blue', 'newly_blue')]
        changed.append({'line': ln, 'n_content_tokens': len(t), 'n_changed_tokens': len(ct),
                        'n_changed_tokens_not_blue': sum(1 for _x, s in ct if (s or 0) < 0.5),
                        'changed_tokens_not_blue': ' '.join(x for x, s in ct if (s or 0) < 0.5)[:200],
                        'inside_or_carrying_a_wrapper': all((s or 0) >= 0.5 for _x, s in ct),
                        'text': ws(ctxB['ln'][ln - 1])[:200]})
    listedA = []
    for r in hyA['blue_changed_lines_lacking_blue']:
        mc = S90.multicolumn_wrapped(ctxB['ln'][r['line'] - 1])
        listedA.append({'line': r['line'], 'class': r['class'], 'first_nonblue_words': r['first_nonblue_words'],
                        'n_newly_written': r['n_newly_written'], 'multicolumn_check': mc,
                        'explained_as_multicolumn_artefact': bool(mc and mc.get('arg3_entirely_one_blue_wrapper')),
                        'text': ws(ctxB['ln'][r['line'] - 1])[:200]})
    out = {'changed_lines_3c379fa_head': changed,
           'changed_lines_3c379fa_head_without_blue': [r['line'] for r in changed if not r['inside_or_carrying_a_wrapper']],
           'w189_rule_3c379fa_head_lines_lacking_blue': hyB['blue_changed_lines_lacking_blue'],
           'w189_rule_a87e195_head_lines_lacking_blue': listedA,
           'w189_rule_a87e195_head_unexplained': [r['line'] for r in listedA if not r['explained_as_multicolumn_artefact']],
           'captions_changed_since_a87e195_lacking_blue': hyA['captions_changed_lacking_blue'],
           'token_alignment_ok': aligned}
    out['held'] = (not out['changed_lines_3c379fa_head_without_blue'] and not hyB['blue_changed_lines_lacking_blue']
                   and not out['w189_rule_a87e195_head_unexplained'] and not out['captions_changed_since_a87e195_lacking_blue'])
    return out


def controls_multicolumn(ctxB, listed):
    out = []
    ln = next((r['line'] for r in listed if r['explained_as_multicolumn_artefact']), None)
    if ln is None:
        return [{'id': 'f-1 multicolumn controls: no listed multicolumn line to build them on', 'refused': False}]
    raw = ctxB['ln'][ln - 1]
    m = re.search(r'\\textcolor\{blue\}\{', raw)
    j = P2.brace_arg(raw, m.end() - 1)
    unwrapped = raw[:m.start()] + raw[m.end():j - 1] + raw[j:]
    a = S90.multicolumn_wrapped(unwrapped)
    out.append({'id': f'f-1 l. {ln} with its wrapper removed fails the multicolumn check', 'changed': unwrapped != raw,
                'refused': unwrapped != raw and bool(a and a['parsed']) and a['arg3_entirely_one_blue_wrapper'] is False})
    inner = raw[m.end():j - 1]
    k = inner.find(',')
    part = (raw[:m.end()] + inner[:k] + '}' + inner[k:] + raw[j:]) if k > 0 else raw
    b = S90.multicolumn_wrapped(part)
    out.append({'id': f'f-2 l. {ln} half-wrapped fails the multicolumn check', 'changed': part != raw,
                'refused': part != raw and bool(b and b['parsed']) and b['arg3_entirely_one_blue_wrapper'] is False})
    return out


# ======================================================================================================================
#  (vi) hard-coded references by region; (iii) 4.7 content for the letter l. 60 claim
# ======================================================================================================================
LETTERED = re.compile(r'(?<![\w.])(Sections?|Subsections?|Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Algorithm)(~|\s)\s*'
                      r'\(?([A-Z]\.?\d+(?:\.\d+)*)\)?')


def task_hardcoded(ctxB, dang):
    cl = ctxB['cl_n']
    reg, _appx, _app = P2.regions(cl, ctxB['adj_n'])

    def region(ln):
        for k, (lo, hi) in reg.items():
            if lo <= ln < hi:
                return k
        return 'other (front matter / declarations / bibliography)'
    lit = dang['w182c_task_dangling'].get('literal_pointers_main_tex_edd7a63') or \
        next((v for k, v in dang['w182c_task_dangling'].items() if k.startswith('literal_pointers_main_tex')), [])
    rows = []
    for r in lit:
        den = next((v for k, v in r.items() if k.startswith('denotes_') and A87_FULL[:7] not in k
                    and '8f051a2' not in k), [])
        rows.append({'line': r['line'], 'region': region(r['line']), 'text': r['text'], 'kind': r['kind'],
                     'numbers': r['numbers'], 'denotes_at_head': den,
                     'a_label_exists': any(isinstance(d, str) and d != 'nothing' and ':' in d for d in (den or [])),
                     'context': r['context'], 'source': 'W182c literal-pointer list (R.PTR / R.PTR_APP)'})
    for i, ln in enumerate(cl, 1):
        s = W.REF_RX.sub(lambda m: ' ' * len(m.group(0)), ln)
        for m in LETTERED.finditer(s):
            num = m.group(3)
            rows.append({'line': i, 'region': region(i), 'text': m.group(0), 'kind': m.group(1), 'numbers': [num],
                         'supplementary': num.startswith('S'),
                         'a_label_exists': (not num.startswith('S')) and any(
                             v['number'] == num for v in ctxB['adj_n']['labels'].values()),
                         'context': ws(ln[max(0, m.start() - 60):m.end() + 30]),
                         'source': 'complementary scan (lettered / supplementary numbers)'})
    main_hits = [r for r in rows if r['region'] == 'main' and not r.get('supplementary')]
    return {'rows': rows, 'regions': {k: list(v) for k, v in reg.items()},
            'main_text_hard_coded_with_label': [r for r in main_hits if r['a_label_exists']],
            'main_text_hard_coded_without_label': [r for r in main_hits if not r['a_label_exists']],
            'supplementary_pointers': sorted({(r['line'], r['text']) for r in rows if r.get('supplementary')}),
            'by_region': dict(Counter(f"{r['region']} | {'supplementary' if r.get('supplementary') else 'typed'}" for r in rows))}


def task_47_content(ctxB):
    v = ctxB['adj_n']['labels']['subsec:res_certification']
    own = W.heading_own_text(ctxB['cl_n'], v['line'])
    pats = {'memory': r'\bmemory\b', 'twenty-five / 25 scenarios': r'twenty-five|\b25\s+(?:operating\s+)?scenarios', 'five representative years':
            r'five representative years', 'future work': r'future work', 'submitted version': r'submitted',
            'largest instance': r'largest', 'without repeating the planning search': r'without repeating the planning search'}
    return {'heading_line': v['line'], 'number': v['number'],
            'hits': {k: bool(re.search(p, own, re.I)) for k, p in pats.items()},
            'note': 'letter l. 60 says the largest submitted instance "is stated as future work (Section~4.7)"; letter l. 59 '
                    'says the paper says a full degradation-neutral search was not run (Section~4.7)'}


# ======================================================================================================================
#  (vii) counts against W190a
# ======================================================================================================================
def counts_vs_w190a(cn, w190a):
    h, p = cn['head_46c9d77'], cn['a87e195']          # here: head = 5f4d3bd, prev = 3c379fa
    w = w190a['f_counts_raw']['head_9c_3c379fa']
    keys = ('prose_words', 'heading_words', 'caption_words', 'n_float_blocks', 'n_display_math_blocks', 'source_lines')
    regions = [r for r in ('abstract', 'main', 'appendix A', 'appendix B') if r in h]
    by = {r: {k: {'w190a_3c379fa': w.get(r, {}).get(k), '3c379fa_here': p.get(r, {}).get(k), 'head_5f4d3bd': h[r].get(k),
                  'change': (h[r].get(k) - p[r].get(k)) if isinstance(h[r].get(k), int) and isinstance(p.get(r, {}).get(k), int)
                  else None} for k in keys} for r in regions}
    for r in regions:
        by[r]['floats'] = {'3c379fa': p[r].get('floats'), 'head': h[r].get('floats')}
        by[r]['equations'] = {'3c379fa': p[r].get('equations'), 'head': h[r].get('equations')}
    rep = all(p.get(r, {}).get(k) == w.get(r, {}).get(k) for r in regions for k in keys)
    out = {'by_region': by, 'method_reproduces_w190a_at_3c379fa': rep,
           'main_text_floats_head': h['main_text_floats_tables_plus_figures'],
           'main_text_equations_head': h['main']['equations'],
           'main_text_prose_words_head': h['main']['prose_words'],
           'appendix_A_prose_words_head': h.get('appendix A', {}).get('prose_words'),
           'appendix_B_prose_words_head': h.get('appendix B', {}).get('prose_words'),
           'abstract_prose_words_head': h.get('abstract', {}).get('prose_words'),
           'w190a': {'floats': w['main_text_floats_tables_plus_figures'], 'equations': w['main']['equations'],
                     'main_words': w['main']['prose_words'], 'appendix_A_words': w['appendix A']['prose_words'],
                     'appendix_B_words': w['appendix B']['prose_words'], 'abstract_words': w['abstract']['prose_words']},
           'n_appendices_head': h['n_appendices']}
    if not rep:
        _fail('(vii) the W189 counting method on 3c379fa does not reproduce the committed W190a counts')
    return out


# ======================================================================================================================
#  checklist, guards, md, post-run, main
# ======================================================================================================================
REQUIRED_CAPTURES = {'(iii) declarations': 'build_declarations', '(iv) blue': 'task_blue_b', '(vi) hard-coded references':
                     'task_hardcoded', '(iii) 4.7 content': 'task_47_content', '(vii) counts': 'counts_vs_w190a',
                     'guards': 'verify_guards'}
REQUIRED_SYMBOLS = {'W190a part 2': (S90, ('numbering_compare', 'line_blue', 'multicolumn_wrapped', 'p2_bound',
                                           'p2_failures_since', 'heading_title', 'verify_guards', 'CHECK_FAILURES',
                                           'EXPECTED_P2_FAILURES', 'rename')),
                    'W189p2': (P2, ('build_ctx', 'task_label_map', 'task_pointers', 'controls_pointers', 'task_dangling',
                                    'task_hygiene', 'task_counts', 'controls_label', 'plus_lines', 'NONBLUE_CHANGED',
                                    'brace_arg', 'regions', 'DECLARED_TARGETS_V12', 'DECLARED_SUBMITTED_V12',
                                    'STATED_NUMBERING', 'STATED_REMOVED_APPENDICES', 'CHECK_FAILURES', '_num',
                                    'LETTER_V10', 'SUBMITTED_MAIN', 'PICKLE_COUNTS')),
                    'W183h': (H, ('line_map',)), 'W182c': (W, ('objects_index', 'heading_own_text', 'REF_RX'))}


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
    need = ([os.path.join(CLONE, f) for f in FILES] + [W189P2_JSON, W190A_JSON, P2.LETTER_V10, P2.LETTER_V9, P2.SUBMITTED_MAIN,
                                                      P2.W183H_JSON, P2.W182C_JSON, P2.W180C_JSON])
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    od = os.path.realpath(out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir))
    if od.startswith(os.path.realpath(os.path.join(REPO, 'data')) + os.sep) or \
            od.startswith(os.path.realpath(os.path.join(REPO, MR)) + os.sep):
        missing.append('output directory under data/ or manuscript_review/ (not permitted for W190b)')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def verify_guards():
    g = S90.verify_guards()
    mine = GUARD.verify(0)
    ok = g['ok'] and not mine
    if not ok:
        _fail(f'guards: W190a / W189p2 chain ok {g["ok"]}; this script {mine}')
    return {'w190a_w189p2_chain': g, 'w190b': {'counts': dict(GUARD.counts), 'verify_0_failures': mine}, 'ok': ok}


def _md_row(*cells):
    return '| ' + ' | '.join(str(c).replace('|', '/').replace('\n', ' ') for c in cells) + ' |'


def write_md(out, path):
    P, D, Hy, B, C, HC, N = (out['iii_pointers'], out['vi_dangling'], out['vi_hygiene_3c379fa_head'], out['iv_blue'],
                             out['vii_counts'], out['vi_hard_coded'], out['vi_numbering_3c379fa_head'])
    L = [f'# W190b part 2 -- structure audit at Overleaf {HEAD_FULL[:7]} (generated by {SCRIPT_NAME}; do not edit)', '',
         'Static numbering (no TeX). No objective value is computed in this stage.', '',
         '## (iii) letter pointers at HEAD', '', f"Counts: {P['v12_counts']}; stale declarations {P['stale_declarations']}; "
                                                f"declaration checks hold {P['declaration_checks_all_hold']}", '',
         _md_row('line', 'printed', 'verdict', 'expected', 'target / note'), '|---:|---|---|---|---|']
    L += [_md_row(r['line'], r['text'], r['verdict'], r.get('expected_at_head'), r.get('target') or r.get('note'))
          for r in P['v12'] if r['verdict'] != 'MATCH' or r.get('declared_target')]
    L += ['', '## (iv) blue', '', _md_row('3c379fa->HEAD changed line', 'changed tokens', 'not blue', 'text'), '|---:|---|---|---|']
    L += [_md_row(r['line'], r['n_changed_tokens'], r['n_changed_tokens_not_blue'], r['text'][:100])
          for r in B['changed_lines_3c379fa_head']]
    L += ['', _md_row('a87e195->HEAD listed line', 'class', 'multicolumn artefact', 'non-blue words'), '|---:|---|---|---|']
    L += [_md_row(r['line'], r['class'], r['explained_as_multicolumn_artefact'], r['first_nonblue_words'][:100])
          for r in B['w189_rule_a87e195_head_lines_lacking_blue']]
    L += ['', '## (vi) structure', '',
          f"[CONFIRM {Hy['n_confirm']}; tightened prefixes {Hy['n_tightened_raw']}; undefined {D['n_undefined']}; duplicate "
          f"labels {D['n_duplicate_labels']}; labels {out['vi_label_count_head']}; numbering identical to 3c379fa {N['identical']}",
          '', _md_row('line', 'region', 'typed pointer', 'label exists', 'context'), '|---:|---|---|---|---|']
    L += [_md_row(r['line'], r['region'], r['text'], r['a_label_exists'], r['context'][:100]) for r in HC['rows']]
    L += ['', '## (vii) counts', '', f"main floats {C['main_text_floats_head']} (W190a {C['w190a']['floats']}); equations "
                                     f"{C['main_text_equations_head']} (W190a {C['w190a']['equations']}); main prose words "
                                     f"{C['main_text_prose_words_head']} (W190a {C['w190a']['main_words']}); Appendix A "
                                     f"{C['appendix_A_prose_words_head']}; Appendix B {C['appendix_B_prose_words_head']}", '']
    for r, v in C['by_region'].items():
        L.append(f"- {r}: " + '; '.join(f"{k} {x['3c379fa_here']} -> {x['head_5f4d3bd']}" for k, x in v.items()
                                         if isinstance(x, dict) and 'head_5f4d3bd' in x))
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


def control_new_declared(ctxA, texts, sub_ctx, targets, submitted):
    """letter l. 264 "Section~4.3" -> "Section~4.4" in a copy (tempfile.mkdtemp, outside the repository), the declaration
    kept: rated MISMATCH."""
    tmp = tempfile.mkdtemp(prefix='w190b_ctl_')
    L = texts[LETTER_NAME].split('\n')
    ok = 'Section~4.3' in L[263]
    L[263] = L[263].replace('Section~4.3', 'Section~4.4', 1)
    p = os.path.join(tmp, 'letter_264.tex')
    with open(p, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))
    decl = dict(targets)
    decl[(264, 'section', '4.4')] = decl[(264, 'section', '4.3')]
    res = P2.task_pointers(ctxA, texts, sub_ctx, declared=decl, declared_sub=submitted, letter_path=p)
    hit = [r for r in res['v12'] if r['line'] == 264 and r['kind'] == 'section']
    return [{'id': 'iii-1 letter l. 264 "Section~4.3" -> "Section~4.4" in a copy (declaration kept) is rated MISMATCH',
             'tempdir': tmp, 'verdicts': [r['verdict'] for r in hit],
             'refused': ok and bool(hit) and all(r['verdict'] == 'MISMATCH' for r in hit)}]


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
    _log(f'W190b part 2: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W190b part 2 (Addendum 92, W190b line (iii), (iv), (vi), (vii)): letter pointers, blue, '
                                    'structure and counts at the Overleaf HEAD 5f4d3bd',
           'objective_convention': 'no objective value is computed in this stage',
           'static_numbering': 'no TeX installation on this machine; numbering is static (W182c method); nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'script_committed_clean': committed_clean(SCRIPT_NAME), 'imported': {m: _sha(m) for m in IMPORTED_SCRIPTS},
           'revisions': {'head_5f4d3bd': HEAD_FULL, 'prev_3c379fa': PREV_FULL, 'round8_a87e195': A87_FULL},
           'key_names': 'outputs of W189 P2 functions keep their key names: "46c9d77" / "head" means HEAD 5f4d3bd and "a87e195" '
                        'the old side of the context (a87e195 for ctx_A, 3c379fa for ctx_B); pointer outputs say "v12" for '
                        'the letter at HEAD; in vi_numbering_3c379fa_head (S90.numbering_compare) "9b" means 3c379fa and '
                        '"9c" means HEAD 5f4d3bd'}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    tracked = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    texts = {f: _rd(os.path.join(CLONE, f)).decode('utf-8') for f in FILES}
    same = {f: texts[f].encode('utf-8') == showb(HEAD_FULL, f) for f in FILES}
    out['state'] = {'clone_head': head, 'tracked_modified': tracked, 'working_equals_head_blob': same,
                    'holds': head == HEAD_FULL and tracked == '' and all(same.values())}
    if not out['state']['holds']:
        _fail(f'(state) {out["state"]}')
    w189 = json.load(open(os.path.join(REPO, W189P2_JSON), encoding='utf-8'))
    w190a = json.load(open(os.path.join(REPO, W190A_JSON), encoding='utf-8'))
    n0 = len(P2.CHECK_FAILURES)
    ctxA = P2.build_ctx(show(A87_FULL, 'main.tex').split('\n'), texts['main.tex'].split('\n'))
    ctxB = P2.build_ctx(show(PREV_FULL, 'main.tex').split('\n'), texts['main.tex'].split('\n'))
    S90.p2_failures_since(n0, 'build_ctx')
    # label maps (ctx_A with W189's stated numbering, for the removed-label set; ctx_B "identical to 3c379fa")
    n0 = len(P2.CHECK_FAILURES)
    with S90.p2_bound(HEAD_FULL=HEAD_FULL):
        labA = P2.task_label_map(ctxA)
    with S90.p2_bound(STATED_NUMBERING={}, STATED_REMOVED_APPENDICES=()):
        labB = P2.task_label_map(ctxB)
    S90.p2_failures_since(n0, 'label maps')
    out['vi_label_map_3c379fa_head'] = {'counts': labB['counts'], 'n_disagree_with_identical': labB['n_disagree_with_stated'],
                                        'disagree': labB['disagree_with_stated']}
    out['vi_numbering_3c379fa_head'] = S90.numbering_compare(ctxB, labB, w189)
    out['vi_label_count_head'] = len(ctxB['adj_n']['labels'])
    _log(f"(vi) numbering = 3c379fa: {out['vi_numbering_3c379fa_head']['identical']}; labels {out['vi_label_count_head']}; "
         f"label map {labB['counts']}")
    # (iii) pointers: the W189 chain on ctx_A with the re-bound declarations
    targets, submitted, dinfo = build_declarations()
    out['iii_declarations'] = dinfo
    sub_lines = _rd(P2.SUBMITTED_MAIN).decode('utf-8').split('\n')
    n0 = len(P2.CHECK_FAILURES)
    _, adj_s = P2._num(sub_lines)
    sub_ctx = {'inn': W.objects_index(adj_s), 'n_labels': len(adj_s['labels'])}
    with S90.p2_bound(DECLARED_TARGETS_V12=targets, DECLARED_SUBMITTED_V12=submitted):
        P = P2.task_pointers(ctxA, texts, sub_ctx)
        ctl_p = P2.controls_pointers(ctxA, texts, sub_ctx, P['v12'])
    S90.p2_failures_since(n0, 'pointers')
    out['iii_pointers'] = P
    out['iii_fig_b9'] = [{'line': r['line'], 'text': r['text'], 'verdict': r['verdict'], 'expected': r.get('expected_at_head')}
                         for r in P['v12'] if r['kind'] == 'figure' and r['numbers'] == ['B.9']]
    out['iii_section_4_7_content'] = task_47_content(ctxB)
    out['iii_section_4_3_heading'] = S90.heading_title(ctxB['adj_n'], '4.3')
    _log(f"(iii) letter {P['v12_counts']}; stale {P['stale_declarations']}; declaration checks {P['declaration_checks_all_hold']}")
    for r in P['v12_mismatch'] + P['v12_ambiguous']:
        _log(f"   {r['verdict']} l. {r['line']} {r['text']!r} kind {r['kind']} printed {r['numbers']} expected "
             f"{r.get('expected_at_head')}")
    if P['stale_declarations']:
        _fail(f'(iii) stale declarations: {P["stale_declarations"]}')
    if not P['declaration_checks_all_hold']:
        _fail(f'(iii) declaration checks: {[k for k, v in P["declaration_checks"].items() if not v["holds"]]}')
    # dangling (ctx_A: labels removed since a87e195) -> undefined, duplicates, W182c literal pointers
    n0 = len(P2.CHECK_FAILURES)
    with S90.p2_bound(HEAD_FULL=HEAD_FULL):
        dang, ctl_d = P2.task_dangling(ctxA, labA, texts)
    S90.p2_failures_since(n0, 'dangling')
    out['vi_dangling'] = dang
    out['vi_hard_coded'] = task_hardcoded(ctxB, dang)
    _log(f"(vi) undefined {dang['n_undefined']}; duplicates {dang['n_duplicate_labels']}; hard-coded with label in the main "
         f"text {len(out['vi_hard_coded']['main_text_hard_coded_with_label'])}; by region {out['vi_hard_coded']['by_region']}")
    # hygiene and blue
    no_cap = '5 no blue changed caption to build the unwrap control'
    n0 = len(P2.CHECK_FAILURES)
    with S90.p2_bound(PREV_FULL=PREV_FULL, HEAD_FULL=HEAD_FULL):
        hyB, ctl_hB = P2.task_hygiene(ctxB, texts)
    S90.p2_failures_since(n0, 'hygiene ctx_B (3c379fa -> HEAD)', declared=(no_cap,))
    n0 = len(P2.CHECK_FAILURES)
    with S90.p2_bound(PREV_FULL=A87_FULL, HEAD_FULL=HEAD_FULL):
        hyA, ctl_hA = P2.task_hygiene(ctxA, texts)
    S90.p2_failures_since(n0, 'hygiene ctx_A (a87e195 -> HEAD)')
    out['vi_hygiene_3c379fa_head'] = hyB
    out['vi_hygiene_a87e195_head'] = hyA
    out['iv_blue'] = task_blue_b(ctxB, hyA, hyB)
    _log(f"(vi) CONFIRM {hyB['n_confirm']}; tightened {hyB['n_tightened_raw']}; (iv) 3c379fa->HEAD lines lacking blue "
         f"{hyB['n_lines_lacking_blue']}; a87e195->HEAD listed {[r['line'] for r in hyA['blue_changed_lines_lacking_blue']]}; "
         f"unexplained {out['iv_blue']['w189_rule_a87e195_head_unexplained']}; changed without blue "
         f"{out['iv_blue']['changed_lines_3c379fa_head_without_blue']}")
    # counts
    n0 = len(P2.CHECK_FAILURES)
    cn, ctl_n = P2.task_counts(ctxB)
    S90.p2_failures_since(n0, 'counts')
    out['vii_counts_raw'] = cn
    out['vii_counts'] = counts_vs_w190a(cn, w190a)
    _log(f"(vii) floats {out['vii_counts']['main_text_floats_head']}; equations {out['vii_counts']['main_text_equations_head']}; "
         f"words main {out['vii_counts']['main_text_prose_words_head']} A {out['vii_counts']['appendix_A_prose_words_head']} "
         f"B {out['vii_counts']['appendix_B_prose_words_head']}; reproduces W190a {out['vii_counts']['method_reproduces_w190a_at_3c379fa']}")
    # negative controls
    n0 = len(P2.CHECK_FAILURES)
    with S90.p2_bound(STATED_NUMBERING={}, STATED_REMOVED_APPENDICES=()):
        ctl_l = P2.controls_label(ctxB)
    S90.p2_failures_since(n0, 'controls')
    ctl = (ctl_l + ctl_p + control_new_declared(ctxA, texts, sub_ctx, targets, submitted)
           + controls_multicolumn(ctxB, out['iv_blue']['w189_rule_a87e195_head_lines_lacking_blue']) + ctl_d
           + [dict(c, id='ctx_B ' + c['id']) for c in ctl_hB] + [dict(c, id='ctx_A ' + c['id']) for c in ctl_hA] + ctl_n)
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f'negative control not refused: {c["id"]}')
    out['p2_failures_declared_not_applicable'] = {
        k: {'messages': v, 'why': 'no float caption changed between 3c379fa and HEAD (W189 P2 builds its caption-unwrap control '
                                  'from a changed caption); the same control is built on ctx_A (a87e195 -> HEAD)'}
        for k, v in S90.EXPECTED_P2_FAILURES.items()}
    out['guards'] = verify_guards()
    out['check_failures'] = list(S90.CHECK_FAILURES) + list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    pj, pm, pn = (os.path.join(od, OUT_NAMES[k]) for k in ('json', 'md', 'man'))
    with open(pj, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, pm)
    inputs = sorted(set([os.path.join(CLONE, f) for f in FILES]
                        + [W189P2_JSON, W190A_JSON, P2.LETTER_V9, P2.LETTER_V10, P2.SUBMITTED_MAIN, P2.W183H_JSON,
                           P2.W182C_JSON, P2.W180C_JSON, SCRIPT_NAME] + list(IMPORTED_SCRIPTS)))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md')},
                'out_dir': out_dir, 'clone_head_full': head,
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(showb(c, f)) for c in (A87_FULL, PREV_FULL, HEAD_FULL) for f in FILES},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(out["check_failures"])}; guards ok {out["guards"]["ok"]}; '
         f'pickle {P2.PICKLE_COUNTS}')
    sys.exit(3 if out['check_failures'] else 0)


if __name__ == '__main__':
    main()
