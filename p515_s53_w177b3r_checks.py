"""
P5.15 Addendum 76 Part 3 RE-RUN at Overleaf e10c104, Planner task W177b-3' (P5_15_W177B_PART3_E10C104.md), with the
change-log checklist items and the letter's numbered pointers (Amendment to Addendum 76, item 2).
ZERO SOLVES, NO MODEL, NO PICKLE; read-only on production code, specs, case files, the frozen JSON, the manuscript
clone, manuscript_review/ and every committed artifact.

What it does (static reading only):
  0. Identity: the Overleaf clone's HEAD (declared e10c104), main.tex sha256 / line count (2,345), the letter
     (response_to_reviewers_draft.tex) sha256 = manuscript_review/response_to_reviewers_v2.tex, clone files unmodified;
     the committed W177b-3 JSON (13182156...) read as the prior result.
  1. Part 3 at e10c104.
     - The line map 51a6fb5 -> e10c104 (difflib on `git show` of both, read-only); every hunk.
     - (a)-(c): the W177b-3 tasks (p515_s53_w177b_part3_checks.task_a/b/c, imported, NOT edited) re-run on BOTH
       51a6fb5 and e10c104 in this run; the two results compared with every manuscript line mapped through the line
       map; every difference listed. The prior rows (A-1..A-9, B-O1..O4, B-P1..P2, C-1..C-2, the stale pointers) get a
       status from their 51a6fb5 lines: line unchanged (moved by an offset) or inside a changed hunk.
     - (d): the W177b-3 nomenclature audit (W167 normalisation, imported) re-run in full on e10c104, and on 51a6fb5 for
       the difference (entries, defined-unused, used-undefined).
     - (e): not re-run unless the figure file or its T8-facing text changed: the clone's fig3_ageing_arms_v2.pdf sha256
       and the text of fig:res_ageing / tab:res_ageing compared between 51a6fb5 and e10c104.
     - (f): the W177b-3 references/labels/citations check re-run in full on e10c104 (and on 51a6fb5 for the
       difference); cited-key order (elsarticle-num is unsorted); literal numbered pointers in main.tex.
  2. Change-log checklist items.
     - Fig. fig:investment_lattice: admissible single-node 2025 plans recomputed with the search's own lattice
       (p515_s47_phase_b_record.Lattice, imported; unit costs = the pinned W2 table it uses) and compared with the
       markers, the budget line and the limit lines read from the PDF content stream (stdlib zlib), and with
       manuscript_review/fig_lattice_case.py's constants (ast; the script is NOT executed). The fourteen-neighbour
       sentence: Lattice.neighbourhood(x = 0) against the s47 Phase B termination certificate.
     - the section 3.6 run-settings sentence against the three campaign specs (8cfa264e, 5ce295e1, 803571c0), by field.
     - every \\ref{subsubsec:certification}: whether the content it cites is in section 2.2.7 or only in Appendix A.5
       (declared needle per reference, located by exact text in each region).
  3. The letter's numbered pointers: the numbering of main.tex derived statically at 6ff8f3c and at e10c104
     (elsarticle: \\section/\\subsection/\\subsubsection counters, \\paragraph unnumbered (secnumdepth 3); \\appendix
     resets section/subsection/equation, letters the sections, prefixes figures/tables/equations with the letter and
     resets equations per appendix section, figure/table counters NOT reset; equation counted per numbered equation
     env, per align row (\\nonumber/\\notag/\\tag honoured), one parent number per subequations with letters inside;
     figure/table counters stepped by \\caption (not \\caption*, not \\subfloat), longtable steps the table counter at
     its start whether or not it has a caption (longtable.dtx \\LT@array: \\refstepcounter{table} in v4.17,
     \\@kernel@refstepcounter{\\LTcaptype} with \\LTcaptype = table in v4.21+); algorithm2e captions step algocf;
     citations numbered in order of first \\cite in the source). Every pointer of the letter outside the quoted
     reviewer comments is extracted and must match exactly one declaration (intended object, optional content
     needle); MATCH / MISMATCH / AMBIGUOUS / SUBMITTED at e10c104, the number at 6ff8f3c, and the changed ones.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at exit; the
guards of the imported W177b-3, W167 and s47 Phase B record modules (each permitted=()) verified at 0 too;
pickle.load / pickle.loads blocked for the whole run and counted (0 expected). No production module is imported
(asserted at exit). Output written through gate_result_io (string flags refused).

Output: the directory must exist and hold nothing but launch.log (it is created by the command below; mkdir refuses
an existing directory); writes w177b3r_checks.json and manifest_sha256.json. `--out-dir DIR`: trial runs only
(absolute path outside the repository; the same refusal rule).

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w177b_part3_e10c104 && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b3r_checks.py \\
      > data/SRP1/Results/P515S53/w177b_part3_e10c104/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label="P5.15 W177b-3' re-run at e10c104 (never solves)").install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError("W177b-3': pickle.load called -- no model loads are permitted")


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError("W177b-3': pickle.loads called -- no model loads are permitted")


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import ast  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import zlib  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import combinations  # noqa: E402

import p515_s53_w177b_part3_checks as P3  # noqa: E402  (W177b-3 tasks; installs its own permitted=() guard)
import p515_s47_phase_b_record as PB  # noqa: E402  (the search's lattice; stdlib at import; its own guard)
import gate_result_io as GRIO  # noqa: E402

W167 = P3.W167
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w177b_part3_e10c104')
OUT_NAME = 'w177b3r_checks.json'
MANIFEST_NAME = 'manifest_sha256.json'
CLONE = P3.CLONE
MAIN_TEX = os.path.join(CLONE, 'main.tex')
BIB = os.path.join(CLONE, 'bibliography.bib')
LETTER_CLONE = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
LETTER_V2 = os.path.join('manuscript_review', 'response_to_reviewers_v2.tex')
DECLARED_CLONE_HEAD = 'e10c104'
DECLARED_MAIN_SHA256 = '1b532035abc65a7f8a08cea6304ac14fcd4ec2e6da553b9a1f250a0b09b5ba1b'
DECLARED_MAIN_LINES = 2345
DECLARED_LETTER_SHA256 = 'bc270fe2cf4b44f7e9547e17ccd85ecd94b56e8f99dd717486d9cf863c20a642'
PRIOR_HEAD = '51a6fb5'          # W177b-3 (be583a4d)
LETTER_HEAD = '6ff8f3c'         # the compile the letter's numbers were taken from
PRIOR_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w177b_part3', 'w177b_part3_checks.json')
PRIOR_JSON_SHA_PREFIX = '13182156'
PRIOR_SCRIPT = 'p515_s53_w177b_part3_checks.py'
PRIOR_SCRIPT_SHA_PREFIX = 'f7a38922'
FIG_V2_CLONE = P3.FIG_V2_CLONE
FIG_V2_SHA_PREFIX = 'c2aec232'
LATTICE_PDF_CLONE = os.path.join(CLONE, 'fig_lattice_case.pdf')
LATTICE_PDF_REVIEW = os.path.join('manuscript_review', 'fig_lattice_case.pdf')
LATTICE_PY = os.path.join('manuscript_review', 'fig_lattice_case.py')
W2_TABLE = PB.INVESTMENT_COST_RESULTS['path']
W167_COSTS = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w167_nomenclature_years', 'question4_cost_file.json')
S47_RESULTS = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b', 'campaign_results.json')
SPECS = {'s47': P3.SPEC_S47, 's51': P3.SPEC_S51, 's53': P3.SPEC_S53}
FORBIDDEN_MODULE_PREFIXES = P3.FORBIDDEN_MODULE_PREFIXES
CHECK_FAILURES = []


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


_sha, _git, _read, _json = P3._sha, P3._git, P3._read, P3._json


def show(commit, path='main.tex'):
    return _git(['show', f'{commit}:{path}'], cwd=CLONE)


def as_lines(text):
    return text.split('\n')


# ======================================================================================================================
#  0. identity
# ======================================================================================================================
def identity():
    head = _git(['rev-parse', '--short', 'HEAD'], cwd=CLONE).strip()
    raw = _read(MAIN_TEX)
    n_lines = len(raw.split('\n')) - (1 if raw.endswith('\n') else 0)
    status = _git(['status', '--porcelain', '--', 'main.tex', 'bibliography.bib', 'response_to_reviewers_draft.tex',
                   'fig_lattice_case.pdf', 'fig3_ageing_arms_v2.pdf'], cwd=CLONE)
    blob_main = hashlib.sha256(show('HEAD').encode('utf-8')).hexdigest()
    out = {'clone_head': head, 'declared_head': DECLARED_CLONE_HEAD, 'head_ok': head.startswith(DECLARED_CLONE_HEAD),
           'main_tex_sha256': _sha(MAIN_TEX), 'main_tex_lines': n_lines,
           'main_tex_equals_head_blob': blob_main == _sha(MAIN_TEX),
           'letter_clone_sha256': _sha(LETTER_CLONE), 'letter_v2_sha256': _sha(LETTER_V2),
           'bibliography_sha256': _sha(BIB), 'clone_files_unmodified': status.strip() == '',
           'prior_json_sha256': _sha(PRIOR_JSON), 'prior_script_sha256': _sha(PRIOR_SCRIPT),
           'commits_51a6fb5_to_head': _git(['log', '--format=%h %ad %s', '--date=iso', f'{PRIOR_HEAD}..HEAD', '--',
                                             'main.tex'], cwd=CLONE).strip().split('\n'),
           'files_changed_51a6fb5_to_head': _git(['diff', '--stat', PRIOR_HEAD, 'HEAD'], cwd=CLONE).strip().split('\n'),
           'files_changed_6ff8f3c_to_head': _git(['diff', '--stat', LETTER_HEAD, 'HEAD'], cwd=CLONE).strip().split('\n')}
    out['sha_ok'] = out['main_tex_sha256'] == DECLARED_MAIN_SHA256
    out['lines_ok'] = n_lines == DECLARED_MAIN_LINES
    out['letter_ok'] = out['letter_clone_sha256'] == DECLARED_LETTER_SHA256 == out['letter_v2_sha256']
    out['prior_json_ok'] = out['prior_json_sha256'].startswith(PRIOR_JSON_SHA_PREFIX)
    out['prior_script_ok'] = out['prior_script_sha256'].startswith(PRIOR_SCRIPT_SHA_PREFIX)
    for k in ('head_ok', 'sha_ok', 'lines_ok', 'letter_ok', 'main_tex_equals_head_blob', 'clone_files_unmodified',
              'prior_json_ok', 'prior_script_ok'):
        if not out[k]:
            _fail(f'identity: {k} is False')
    return out


# ======================================================================================================================
#  1. the line map 51a6fb5 -> e10c104 and the hunks
# ======================================================================================================================
def line_map(old, new):
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    mp, hunks = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                mp[i1 + k + 1] = j1 + k + 1
        else:
            hunks.append({'tag': tag, 'old': [i1 + 1, i2], 'new': [j1 + 1, j2]})
    return mp, hunks


def section_of(lines, ln):
    sm = W167.section_map(lines)
    return sm.get(ln)


# the prior rows of P5_15_W177B_PART3.md, by their 51a6fb5 lines (the report's line citations)
PRIOR_ROWS = [
    ('A-1', 'M', 'hold start for continued evaluations (A.5 holds sentence, A.3, section 3.6 caps, A.4)', [1734, 1710, 1131, 1723]),
    ('A-2', 'L', 'A.4 "the holds start at the first passing cycle" vs A.3/A.5 "from the next cycle"', [1718, 1710, 1734]),
    ('A-3', 'M', 'C4 "at the interface price" vs the code\'s expected wholesale price; section 4.8 defines lambda', [1742, 1531]),
    ('A-4', 'L', 'K_EXCL: "the first three cycles after k0 are not counted"; C3 "all after k0"', [1748, 1741]),
    ('A-5', 'L', 'noise sentence scoped "In (C1)-(C2)"', [1748]),
    ('A-6', 'L', 'monotone-branch wording ("descends without a sign change", "the mean step")', [1748]),
    ('A-7', 'L', '"(C4)-(C5) hold on that window" (C4 is checked at k* only)', [1748]),
    ('A-8', 'L', 'C5 "every local solve" (final accepted exit; ESSO own tolerances)', [1743]),
    ('A-9', 'L', 'accepted clause "An evaluation without one has no defined slack" absent (A.5, 2.2.7)', [1748, 682]),
    ('B-O1', 'L', 'primary acceptable_iter = 5 not printed (A.8)', [1815]),
    ('B-O2', 'L', 'ESSO max_iter 500 not printed (A.8)', [1815]),
    ('B-O3', 'L', 'bound_push family not printed (A.8)', [1815]),
    ('B-O4', 'L', 'primary solves warm-started: only implied by "cold restart" (A.8)', [1815]),
    ('B-P1', 'L', 'A.4 sends the recovery settings to section 3.6 (they are in A.8)', [1727]),
    ('B-P2', 'L', 'section 4.5 cites sec:case_settings for the multi-scenario production exit (cap 500 is in A.8)', [1407]),
    ('C-1', 'L', 'A.2 list names eq:available_capacity (computed after the solve); aggregate rated rows unlisted', [1647]),
    ('C-2', 'L', 'converter circle written with S^Av (code bounds by es_s_rated; equal)', [1647]),
    ('D-1', 'L', 'nomenclature pi_t defined, never used', [248]),
    ('D-2', 'L', 'nomenclature s (settling slack) defined, unused', [308]),
    ('E-1', 'L', '"n/a" (v2 figure, tab:res_ageing) vs "never" (T8 export, W165 figure) for no_ageing', []),
    ('F-1', 'L', 'section 4.7 "counted as clean under the rule of" subsubsec:certification (10x rule only in A.5)', [1523]),
    ('F-2', 'L', 'Algorithm 2 input "certification parameters of" subsubsec:certification (P_max, n_w in A.5 / 3.6)', [1763]),
]


def prior_rows(old, new, mp, hunks):
    out = []
    for rid, sev, desc, lns in PRIOR_ROWS:
        per = []
        for ln in lns:
            nl = mp.get(ln)
            per.append({'old_line': ln, 'old_text_head': re.sub(r'\s+', ' ', old[ln - 1]).strip()[:140],
                        'new_line': nl, 'line_unchanged': nl is not None,
                        'offset': (nl - ln) if nl is not None else None,
                        'in_hunk': [h for h in hunks if h['old'][0] <= ln <= max(h['old'][0], h['old'][1])]})
        out.append({'id': rid, 'severity_prior': sev, 'description': desc, 'lines': per,
                    'all_lines_unchanged': all(p['line_unchanged'] for p in per)})
    return out


# ---- the W177b-3 tasks (a)-(c) on both commits, compared through the line map ----------------------------------------
MS_LINE_KEYS = {'lines', 'line', 'first_line', 'main_tex_lines', 'body_lines', 'appendix_region', 'section_227_region',
                'region_3_6', 'region_A7', 'nomenclature_lines', 'elsewhere', 'in_A7', 'in_3_6', 'printed_anywhere'}


def _map_val(v, mp):
    if isinstance(v, int) and not isinstance(v, bool):
        return mp.get(v, ('changed', v))
    if isinstance(v, list):
        return [_map_val(x, mp) for x in v]
    return v


def compare(old, new, mp, path='', diffs=None, in_code=False):
    diffs = [] if diffs is None else diffs
    if isinstance(old, dict) and isinstance(new, dict):
        code = in_code or ('file' in old)
        for k in sorted(set(old) | set(new)):
            p = f'{path}.{k}'
            if k not in old or k not in new:
                diffs.append({'path': p, 'old': old.get(k, '<absent>'), 'new': new.get(k, '<absent>')})
                continue
            if k in MS_LINE_KEYS and not code:
                if k in ('appendix_region', 'section_227_region', 'region_3_6', 'region_A7', 'nomenclature_lines'):
                    o = [mp.get(old[k][0]), mp.get(old[k][1])]     # region bounds: report both ends mapped
                    if o != new[k]:
                        diffs.append({'path': p, 'old': old[k], 'old_mapped': o, 'new': new[k]})
                    continue
                o = _map_val(old[k], mp)
                if o != new[k]:
                    diffs.append({'path': p, 'old': old[k], 'old_mapped': o, 'new': new[k]})
                continue
            compare(old[k], new[k], mp, p, diffs, code)
    elif isinstance(old, list) and isinstance(new, list):
        if old and new and all(isinstance(x, dict) and {'line', 'context'} <= set(x) for x in old + new):
            # a list of (line, context) records (e.g. the references to a label): compared as sets, lines mapped
            o = {(mp.get(x['line'], ('changed', x['line'])) if isinstance(mp.get(x['line']), int) else ('changed', x['line']),
                  x['context']) for x in old}
            n = {(x['line'], x['context']) for x in new}
            o_ctx = {c for _l, c in o}
            only_old = sorted([list(t) for t in o - n], key=str)
            only_new = sorted([list(t) for t in n - o], key=str)
            if only_old or only_new:
                diffs.append({'path': path, 'only_51a6fb5': only_old, 'only_e10c104': only_new,
                              'contexts_unchanged_but_line_changed': sorted(str(t) for t in n - o if t[1] in o_ctx)})
            return diffs
        if len(old) != len(new):
            diffs.append({'path': path, 'old_len': len(old), 'new_len': len(new)})
        for i, (a, b) in enumerate(zip(old, new)):
            compare(a, b, mp, f'{path}[{i}]', diffs, in_code)
    elif old != new:
        diffs.append({'path': path, 'old': old, 'new': new})
    return diffs


def rerun_abc(old, new, mp):
    out, prior = {}, _json(PRIOR_JSON)
    for name, fn in (('a_certification_rule', P3.task_a), ('b_settings', P3.task_b), ('c_storage_agent', P3.task_c)):
        n0 = len(P3.CHECK_FAILURES)
        r_old = json.loads(json.dumps(fn(old), default=GRIO.json_default_item))
        f_old = P3.CHECK_FAILURES[n0:]
        n1 = len(P3.CHECK_FAILURES)
        r_new = json.loads(json.dumps(fn(new), default=GRIO.json_default_item))
        f_new = P3.CHECK_FAILURES[n1:]
        same = r_old == prior[name]
        out[name] = {'anchor_failures_51a6fb5': f_old, 'anchor_failures_e10c104': f_new,
                     'rerun_on_51a6fb5_equals_committed_W177b3_json': same,
                     'differences_after_line_map': compare(r_old, r_new, mp), 'e10c104_result': r_new}
        if not same:
            out[name]['rerun_vs_committed_differences'] = compare(prior[name], r_old, {i: i for i in range(1, 5000)})
        if f_old:
            _fail(f'{name}: W177b-3 anchors fail on 51a6fb5 itself: {f_old}')
    return out


# ---- (d) nomenclature --------------------------------------------------------------------------------------------------
def task_d(old, new):
    cur = P3.nomenclature_audit(new, DECLARED_CLONE_HEAD)
    prv = P3.nomenclature_audit(old, PRIOR_HEAD)
    k_now = {u['key'] for u in cur['used_undefined']}
    k_prv = {u['key'] for u in prv['used_undefined']}
    extra_raw = {
        r'Q^\star (minimum recourse)': [r'Q^\star', r'Q^{\star}'],
        r'\mathcal{U}_s': [r'\mathcal{U}_s', r'\mathcal{U}_{s}'],
        r'\omega_s': [r'\omega_s'],
        r'\Omega (joint scenarios)': [r'\Omega=', r'\Omega =', r'\in\Omega', r'\in \Omega}', r's\in\Omega'],
        r'\hat{P} (measured period)': [r'\hat{P}'],
        r'n_w (certifying-window length)': [r'n_{\mathrm{w}}'],
        r'u_o (old dispatch symbol)': [r'\boldsymbol{u}_o', r'\mathcal{U}_o'],
        r'C^Op_s (horizon cost of scenario s)': [r'C_s^{\mathrm{Op}}', r'C^{\mathrm{Op}}_s'],
        r'C^Op_{y,d,m,o} (daily cost)': [r'C^{\text{Op}}_{y,d,m,o}'],
        r'D_{e,y^Inv,y} described in the body': [r'fractional cycling loss', r'cycling-degradation exponent'],
    }
    clines = W167.strip_comments(new)
    nom_s, nom_e = W167.nomenclature_bounds(clines)
    raw = {name: {'patterns': pats,
                  'nomenclature_lines': sorted({i for pat in pats for i in range(nom_s, nom_e + 1) if pat in clines[i - 1]}),
                  'body_lines': sorted({i for pat in pats for i in range(nom_e + 1, len(clines) + 1) if pat in clines[i - 1]})}
           for name, pats in extra_raw.items()}
    return {'normalisation': 'W177b-3 (W167 tokeniser, imported unedited, with the W177b-3 label-subscript change)',
            'current': cur,
            'prior_51a6fb5': {k: prv[k] for k in ('nomenclature_lines', 'n_entries', 'n_keys', 'defined_unused',
                                                    'n_used_undefined')},
            'entries_only_now': sorted(set(cur['entry_tex']) - set(prv['entry_tex'])),
            'entries_only_then': sorted(set(prv['entry_tex']) - set(cur['entry_tex'])),
            'defined_unused_now': [u['key'] for u in cur['defined_unused']],
            'defined_unused_then': [u['key'] for u in prv['defined_unused']],
            'used_undefined_now_not_then': sorted(k_now - k_prv),
            'used_undefined_then_not_now': sorted(k_prv - k_now),
            'extra_raw_checks_e10c104': raw}


# ---- (e) the ageing figure: re-run only if the file or its text changed ------------------------------------------------
def float_text(lines, label):
    i = next(k for k, ln in enumerate(lines) if f'\\label{{{label}}}' in ln)
    s = max(k for k in range(i + 1) if re.match(r'\s*\\begin\{(figure|table)\*?\}', lines[k]))
    e = next(k for k in range(i, len(lines)) if re.match(r'\s*\\end\{(figure|table)\*?\}', lines[k]))
    return '\n'.join(lines[s:e + 1]), s + 1, e + 1


def task_e(old, new):
    out = {'clone_pdf_sha256': _sha(FIG_V2_CLONE), 'clone_pdf_log': _git(['log', '--format=%h', '--', 'fig3_ageing_arms_v2.pdf'],
                                                                           cwd=CLONE).split()}
    changed_since = _git(['diff', '--name-only', PRIOR_HEAD, 'HEAD', '--', 'fig3_ageing_arms_v2.pdf'], cwd=CLONE).strip()
    out['pdf_changed_since_51a6fb5'] = changed_since != ''
    for lab in ('fig:res_ageing', 'tab:res_ageing'):
        t_old, s0, e0 = float_text(old, lab)
        t_new, s1, e1 = float_text(new, lab)
        out[lab] = {'old_lines': [s0, e0], 'new_lines': [s1, e1], 'text_identical': t_old == t_new}
    # the section 4.3 prose that reads the figure (between the section label and the figure)
    i0 = next(k for k, ln in enumerate(old) if r'\label{subsec:res_degradation}' in ln)
    i1 = next(k for k, ln in enumerate(new) if r'\label{subsec:res_degradation}' in ln)
    j0 = next(k for k in range(i0 + 1, len(old)) if re.match(r'\s*\\subsection\{', old[k]))
    j1 = next(k for k in range(i1 + 1, len(new)) if re.match(r'\s*\\subsection\{', new[k]))
    out['section_4_3_identical'] = old[i0:j0] == new[i1:j1]
    out['rerun_needed'] = (out['pdf_changed_since_51a6fb5'] or not out['fig:res_ageing']['text_identical']
                           or not out['tab:res_ageing']['text_identical'] or not out['section_4_3_identical'])
    if not out['clone_pdf_sha256'].startswith(FIG_V2_SHA_PREFIX):
        _fail('fig3_ageing_arms_v2.pdf in the clone is not c2aec232')
    return out


# ---- (f) references, labels, citations ---------------------------------------------------------------------------------
def cite_order(lines):
    clines = W167.strip_comments(lines)
    text = '\n'.join(clines)
    start = text.find(r'\begin{document}')
    order = []
    for m in re.finditer(r'\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}', text[start:]):
        for k in m.group(1).split(','):
            k = k.strip()
            if k not in order:
                order.append(k)
    return order


LITERAL_PTR = re.compile(r'\b(Sections?|Subsections?|Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Appendix|Algorithm)'
                         r'~?\s*\(?(\d+(?:\.\d+)*|[A-E]\.\d+)\)?')


def _labels(lines):
    return re.findall(r'\\label\{([^}]*)\}', '\n'.join(W167.strip_comments(lines)))


def task_f(old, new):
    cur = P3.task_f(new)
    prv = P3.task_f(old)
    cl = W167.strip_comments(new)
    literal = []
    for i, ln in enumerate(cl, 1):
        for m in LITERAL_PTR.finditer(ln):
            literal.append({'line': i, 'text': m.group(0), 'context': re.sub(r'\s+', ' ', ln[max(0, m.start() - 120):m.end() + 60]).strip()})
    o_now, o_then = cite_order(new), cite_order(old)
    return {'current': cur,
            'prior_51a6fb5_counts': {k: prv[k] for k in ('n_labels', 'n_ref_keys', 'n_ref_uses', 'n_cite_keys', 'n_cite_uses')},
            'labels_added': sorted(set(_labels(new)) - set(_labels(old))),
            'labels_removed': sorted(set(_labels(old)) - set(_labels(new))),
            'never_referenced_now_not_then': sorted(set(cur['labels_never_referenced']) - set(prv['labels_never_referenced'])),
            'never_referenced_then_not_now': sorted(set(prv['labels_never_referenced']) - set(cur['labels_never_referenced'])),
            'cite_keys_dropped_since_51a6fb5': [k for k in o_then if k not in o_now],
            'cite_keys_new_since_51a6fb5': [k for k in o_now if k not in o_then],
            'cite_order_e10c104': o_now, 'cite_order_51a6fb5': o_then,
            'literal_numbered_pointers_in_main_tex': literal}


# ======================================================================================================================
#  2. change-log checklist items
# ======================================================================================================================
def lattice_unit_costs():
    w2 = _json(W2_TABLE)
    return PB.unit_costs_from_w2(w2), w2


def _pdf_markers(path):
    """Every XObject marker drawn on the page (name, absolute position, fill colour) and the stroked segments, read
    from the content stream (W177b-3 pdf_read for text/segments; the marker pass is added here)."""
    b = open(path, 'rb').read()
    objs = P3._pdf_objects(b)
    page = next(n for n, body in objs.items() if re.search(rb'/Type /Page\b(?!s)', body) and b'/Contents' in body)
    content = P3._pdf_stream(objs[int(re.search(rb'/Contents (\d+) 0 R', objs[page]).group(1))])
    ctm, stack, fill, operands, markers = [1, 0, 0, 1, 0, 0], [], None, [], []
    for kind, val in P3._tokens(content):
        if kind in ('[', ']'):
            operands = []
            continue
        if kind != 'op':
            operands.append((kind, val))
            continue
        nums = [v for k, v in operands if k == 'num']
        if val == 'q':
            stack.append((list(ctm), fill))
        elif val == 'Q':
            if stack:
                ctm, fill = stack.pop()
        elif val == 'cm' and len(nums) >= 6:
            ctm = P3._mat_mul(nums[-6:], ctm)
        elif val == 'rg' and len(nums) >= 3:
            fill = tuple(round(x, 4) for x in nums[-3:])
        elif val == 'g' and nums:
            fill = (round(nums[-1], 4),) * 3
        elif val == 'Do':
            name = next((v for k, v in operands if k == 'name'), None)
            markers.append({'name': name, 'x': ctm[4], 'y': ctm[5], 'fill': fill})
        operands = []
    return markers


def task_lattice():
    unit_costs, w2 = lattice_unit_costs()
    years = (2025, 2030, 2035)
    lat = PB.Lattice(years, unit_costs)
    # (i) single node, 2025: every lattice point S in 0..2.75 MVA, E in 0..5.5 MWh (the figure's window), node 7 slot
    pts = []
    for i in range(0, 12):
        for j in range(0, 12):
            if i == 0 and j == 0:
                continue
            z = (0, 0, i, j, 0, 0, 0)
            why = lat.reasons(z)
            cls = 'adm' if not why else ('budget' if all(r.startswith('budget') for r in why) else 'out')
            pts.append({'S': i * PB.GRANULE_P_MVA, 'E': j * PB.GRANULE_E_MWH, 'class': cls,
                        'I_eur': lat.investment_cost(z) if cls != 'out' else None})
    adm = [(p['S'], p['E']) for p in pts if p['class'] == 'adm']
    over = [(p['S'], p['E']) for p in pts if p['class'] == 'budget']
    budget_margin = min(abs(PB.BUDGET_EUR - p['I_eur']) for p in pts if p['I_eur'] is not None)
    # (ii) the figure script's constants (ast; not executed)
    tree = ast.parse(_read(LATTICE_PY))
    consts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t for t in node.targets]
            for t in names:
                if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for a, v in zip(t.elts, node.value.elts):
                        if isinstance(a, ast.Name):
                            try:
                                consts[a.id] = ast.literal_eval(v)
                            except ValueError:
                                pass
                elif isinstance(t, ast.Name):
                    try:
                        consts[t.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    c25 = unit_costs[2025]
    w167 = _json(W167_COSTS)['as_read_by_production']
    costs = {'power_eur_per_mva_2025_W2': c25['power'], 'energy_eur_per_mwh_2025_W2': c25['energy'],
             'power_W167': w167['expected_power_eur_per_MVA']['2025'],
             'energy_W167': w167['expected_energy_eur_per_MWh_production_function']['2025'],
             'discount_multiplier_2025': w2['expected_unit_costs_per_case_year']['new']['2025']['discount_multiplier'],
             'script_CS_kEUR': consts.get('CS'), 'script_CE_kEUR': consts.get('CE'), 'script_BUDGET_kEUR': consts.get('BUDGET'),
             'script_DS': consts.get('DS'), 'script_DE': consts.get('DE'), 'script_PHI': [consts.get('PHI_MIN'), consts.get('PHI_MAX')],
             'script_EMAX': consts.get('EMAX'),
             'change_log_text': '256.32 kEUR/MVA, 253.88 kEUR/MWh'}
    costs['W2_equals_W167'] = (abs(costs['power_eur_per_mva_2025_W2'] - costs['power_W167']) < 1e-6
                               and abs(costs['energy_eur_per_mwh_2025_W2'] - costs['energy_W167']) < 1e-6)
    costs['change_log_rounding_ok'] = (round(c25['power'] / 1e3, 2) == 256.32 and round(c25['energy'] / 1e3, 2) == 253.88)
    costs['script_constants_ok'] = (abs(consts.get('CS', 0) * 1e3 - c25['power']) < 0.01
                                    and abs(consts.get('CE', 0) * 1e3 - c25['energy']) < 0.01
                                    and consts.get('BUDGET') * 1e3 == PB.BUDGET_EUR and consts.get('DS') == PB.GRANULE_P_MVA
                                    and consts.get('DE') == PB.GRANULE_E_MWH and consts.get('EMAX') == PB.E_MAX_MWH
                                    and [consts.get('PHI_MIN'), consts.get('PHI_MAX')] == [2.0, 4.0])
    # (iii) the PDF: markers, ticks, lines
    pdf = {}
    for tag, path in (('clone', LATTICE_PDF_CLONE), ('review', LATTICE_PDF_REVIEW)):
        pdf[tag + '_sha256'] = _sha(path)
    pdf['clone_equals_review'] = pdf['clone_sha256'] == pdf['review_sha256']
    rd = P3.pdf_read(LATTICE_PDF_CLONE)
    pdf['metadata'] = rd['metadata']
    black = (0.0, 0.0, 0.0)
    norm = lambda t: t.replace('\u2212', '-').replace(' ', '').strip()  # noqa: E731
    vt = sorted({round(s['x0'], 4) for s in rd['segments'] if abs(s['x0'] - s['x1']) < 1e-6
                 and 1.0 < abs(s['y0'] - s['y1']) < 5.0})
    ht = sorted({round(s['y0'], 4) for s in rd['segments'] if abs(s['y0'] - s['y1']) < 1e-6
                 and 1.0 < abs(s['x0'] - s['x1']) < 5.0 and s['x1'] < 40})
    numeric = [t for t in rd['texts'] if t['x'] is not None and re.fullmatch(r'-?\d+(\.\d+)?', norm(t['text']))]
    xlab = sorted([t for t in numeric if t['y'] < 80], key=lambda t: t['x'])
    ylab = sorted([t for t in numeric if t['x'] < 30], key=lambda t: t['y'])
    xv, yv = [float(norm(t['text'])) for t in xlab], [float(norm(t['text'])) for t in ylab]
    pdf['x_ticks_pt'], pdf['x_tick_values'], pdf['y_ticks_pt'], pdf['y_tick_values'] = vt, xv, ht, yv
    if len(vt) != len(xv) or len(ht) != len(yv) or len(vt) < 2 or len(ht) < 2:
        _fail(f'lattice PDF ticks: x {vt} / {xv}; y {ht} / {yv}')
        return {'pdf': pdf}
    sx = (vt[-1] - vt[0]) / (xv[-1] - xv[0])
    sy = (ht[-1] - ht[0]) / (yv[-1] - yv[0])
    to_x = lambda x: xv[0] + (x - vt[0]) / sx  # noqa: E731
    to_y = lambda y: yv[0] + (y - ht[0]) / sy  # noqa: E731
    mk = _pdf_markers(LATTICE_PDF_CLONE)
    groups = {}
    for m in mk:
        key = f"{m['name']} fill={m['fill']}"
        groups.setdefault(key, []).append((round(to_x(m['x']) / PB.GRANULE_P_MVA) * PB.GRANULE_P_MVA,
                                           round(to_y(m['y']) / PB.GRANULE_E_MWH) * PB.GRANULE_E_MWH,
                                           round(to_x(m['x']), 4), round(to_y(m['y']), 4)))
    pdf['marker_groups'] = {k: {'n': len(v), 'snapped_points': sorted((a, b) for a, b, _c, _d in v),
                                'max_snap_error': max(max(abs(a - c), abs(b - d)) for a, b, c, d in v)}
                            for k, v in groups.items()}
    blue = (0.1647, 0.4706, 0.8392)          # "#2a78d6", the admissible-plan colour of fig_lattice_case.py
    adm_pdf = sorted(p[:2] for k, v in groups.items() if k.endswith(f'fill={blue}') for p in v)
    pdf['admissible_markers_equal_lattice'] = adm_pdf == sorted(adm)
    # the remaining two marker groups: the over-budget set (23 expected) and the out-of-limits set
    others = {k: sorted(p[:2] for p in v) for k, v in groups.items() if not k.endswith(f'fill={blue}')}
    pdf['over_budget_group_equal_lattice'] = any(v == sorted(over) for v in others.values())
    # the budget line: the INK-coloured stroke spanning the axes; check both ends against (B - c_S S) / c_E
    ink = (0.0431, 0.0431, 0.0431)
    lines_ = []
    for s in rd['segments']:
        if s['stroke'] == ink and abs(s['x1'] - s['x0']) > 50:
            x0, y0, x1, y1 = to_x(s['x0']), to_y(s['y0']), to_x(s['x1']), to_y(s['y1'])
            e0 = (PB.BUDGET_EUR - c25['power'] * x0) / c25['energy']
            e1 = (PB.BUDGET_EUR - c25['power'] * x1) / c25['energy']
            lines_.append({'S0': round(x0, 5), 'E0_pdf': round(y0, 5), 'E0_model': round(e0, 5),
                           'S1': round(x1, 5), 'E1_pdf': round(y1, 5), 'E1_model': round(e1, 5),
                           'max_abs_err_MWh': round(max(abs(y0 - e0), abs(y1 - e1)), 5)})
    pdf['budget_line'] = lines_
    grey = (0.3216, 0.3176, 0.3059)
    lim = []
    for s in rd['segments']:
        if s['stroke'] == grey and abs(s['x1'] - s['x0']) > 50:
            x0, y0, x1, y1 = to_x(s['x0']), to_y(s['y0']), to_x(s['x1']), to_y(s['y1'])
            lim.append({'S0': round(x0, 4), 'E0': round(y0, 4), 'S1': round(x1, 4), 'E1': round(y1, 4),
                        'slope_MWh_per_MVA': round((y1 - y0) / (x1 - x0), 4) if abs(x1 - x0) > 1e-9 else None})
    pdf['limit_lines'] = lim
    pdf['all_texts'] = [t['text'] for t in rd['texts']]
    # (iv) the fourteen neighbours of x = 0
    nb = lat.neighbourhood(lat.x0())
    labels = [lat.label(z) for z in nb]
    s47 = _json(S47_RESULTS)
    tc = s47.get('termination_certificate', {})
    subsets = [c for r in (1, 2, 3) for c in combinations(PB.ACTIVE_NODES, r)]
    expected = sorted(f'y{y}__' + '__'.join(f'n{n}_p0.25_e0.5' for n in sub) for y in (2025, 2030) for sub in subsets)
    fourteen = {'n_neighbours': len(nb), 'labels': labels, 'equals_7_subsets_x_2025_2030_smallest_unit': sorted(labels) == expected,
                's47_termination_certificate': {k: tc.get(k) for k in ('n_feasible_neighbours', 'all_feasible_neighbours_polled',
                                                                       'n_no_improvement', 'holds')},
                's47_results_sha256': _sha(S47_RESULTS)}
    single_nb = sorted((p['S'], p['E']) for p in pts if p['class'] == 'adm' and p['S'] <= 0.25 and p['E'] <= 0.5)
    return {'unit_costs_2025': costs, 'single_node_2025': {'n_admissible': len(adm), 'admissible': sorted(adm),
                                                          'n_over_budget': len(over), 'over_budget': sorted(over),
                                                          'n_outside_limits': sum(1 for p in pts if p['class'] == 'out'),
                                                          'smallest_admissible': min(adm),
                                                          'admissible_within_one_unit_of_zero': single_nb,
                                                          'closest_point_to_budget_eur': round(budget_margin, 2)},
            'pdf': pdf, 'fourteen_neighbours': fourteen, 'lattice_module': {'path': 'p515_s47_phase_b_record.py',
                                                                           'sha256': _sha('p515_s47_phase_b_record.py'),
                                                                           'W2_table': W2_TABLE, 'W2_sha256': _sha(W2_TABLE)}}


# ---- the section 3.6 run-settings sentence against the three campaign specs ------------------------------------------
def task_run_settings(new):
    sp = {k: _json(p) for k, p in SPECS.items()}
    ex = {k: v['extra'] for k, v in sp.items()}
    pd = {k: v['poll_design'] for k, v in ex.items()}
    s, e = P3.region(new, r'\label{sec:case_settings}', r'\s*% =====')
    reg = '\n'.join(new[s - 1:e])
    rows = []

    def row(rid, needle, source, ok, detail):
        hits = [s + i for i, ln in enumerate(new[s - 1:e]) if needle in ln]
        if not hits:
            _fail(f'3.6 needle not found: {needle!r}')
        rows.append({'id': rid, 'needle': needle, 'lines': hits, 'source': source, 'match': bool(ok), 'detail': detail})
    row('delta0', r'$\Delta_0 = 4$ lattice units', 'poll_design.delta_0 (s47, s51)',
        pd['s47']['delta_0'] == 4 and pd['s51']['delta_0'] == 4, {k: pd[k].get('delta_0') for k in pd})
    row('budgets', 'budgets of 20 new evaluations for the two variant-A searches and 60 for the variant-B certificate',
        'poll_design.max_new_evaluations; design',
        pd['s47']['max_new_evaluations'] == 20 and pd['s51']['max_new_evaluations'] == 20 and pd['s53']['max_new_evaluations'] == 60
        and pd['s47']['design'] == pd['s51']['design'] == 'orthomads_n_plus_1_neg' and pd['s53']['design'] == 'orthomads_2n',
        {k: [pd[k].get('max_new_evaluations'), pd[k].get('design')] for k in pd})
    row('completion_cap', 'a completion cap of 30 points', 'poll_design.completion_cap',
        all(pd[k]['completion_cap'] == 30 for k in pd), {k: pd[k].get('completion_cap') for k in pd})
    row('sigma_Q', r'$\sigma_Q = 18{,}449.66$', 'extra.sigma_Q.sigma_Q_eur',
        all(round(ex[k]['sigma_Q']['sigma_Q_eur'], 2) == 18449.66 for k in ex), {k: ex[k]['sigma_Q']['sigma_Q_eur'] for k in ex})
    row('production_exit', 'ten consecutive passing cycles, cap 500 cycles', 'required_consecutive_cycles, cap',
        all(sp[k]['required_consecutive_cycles'] == 10 and sp[k]['cap'] == 500 for k in sp),
        {k: [sp[k]['required_consecutive_cycles'], sp[k]['cap']] for k in sp})
    row('variant_B_unit', 'variant B polled at unit size throughout', 's53 poll_design.delta / on_success',
        pd['s53'].get('delta') == 1 and 'stays 1' in pd['s53'].get('on_success', ''),
        {'delta': pd['s53'].get('delta'), 'on_success': pd['s53'].get('on_success')})
    row('completion_stop', 'when a completion set exceeded the cap', 'extra.completion_rule / poll_rule',
        all('STOP_FOR_REVIEW' in (ex[k].get('completion_rule') or ex[k].get('poll_rule') or '') for k in ex),
        {k: ('completion_rule' if 'completion_rule' in ex[k] else 'poll_rule') for k in ex})
    row('barrier_stop', 'two barrier points occurred in one poll or three in all', 'extra.stop_rule (+ s53 barrier_stop)',
        all('>= 2 NEW barrier evaluations in one poll or >= 3 overall' in ex[k]['stop_rule'] for k in ex)
        and pd['s53'].get('barrier_stop') == [2, 3], {'s53_barrier_stop': pd['s53'].get('barrier_stop')})
    row('max_polls', 'ran at most 60 polls', 'poll_design.max_polls', all(pd[k]['max_polls'] == 60 for k in pd),
        {k: pd[k].get('max_polls') for k in pd})
    # cache seeding: what each cache held
    cache = {}
    for k in ex:
        ct = ex[k]['cache_table']
        srcs = {}
        for v in ct.values():
            src = v.get('source') or {}
            tag = src.get('campaign_id') or src.get('kind') or '?'
            srcs[tag] = srcs.get(tag, 0) + 1
        labs = sorted(v['label'] for v in ct.values())
        ladders = [lb for lb in labs if re.fullmatch(r'n[579]_[24]h_e\d(_m2)?', lb)]
        multi = []
        for v in ct.values():
            nodes = v['canonical']['nodes']
            if sum(1 for pe in nodes.values() if pe[0] or pe[1]) >= 2:
                multi.append(v['label'])
        on_lattice = {}
        for v in ct.values():
            on_lattice[v['label']] = PB.Lattice((2025, 2030, 2035), lattice_unit_costs()[0]).z_of_canonical(v['canonical']) is not None
        cache[k] = {'n_entries': len(ct), 'by_source': srcs, 'single_node_ladder_labels': ladders,
                    'multi_node_entries': sorted(multi), 'off_lattice_entries': sorted(lb for lb, ok in on_lattice.items() if not ok),
                    'sources_accepted': [s.get('campaign_id') or s.get('path') or s.get('kind') for s in ex[k]['cache_sources_accepted']]}
    # ladder points beyond the budget (s47)
    lat = PB.Lattice((2025, 2030, 2035), lattice_unit_costs()[0])
    over = []
    for v in ex['s47']['cache_table'].values():
        z = lat.z_of_canonical(v['canonical'])
        if z is not None and lat.has_storage(z) and any(r.startswith('budget') for r in lat.reasons(z)):
            over.append(v['label'])
    cache['s47']['ladder_points_over_budget'] = sorted(over)
    row('cache_seeding', 'the cache was seeded with the single-node ladders of the initial design, run with and without the budget, and their combinations',
        'extra.cache_table (by source and content)', False, cache)
    rows[-1]['match'] = None          # decided in the report from `detail`; not a mechanical verdict
    # the 3.6 first paragraph: the three campaigns ran on earlier code states without the tight tail
    tails = {}
    for k in sp:
        txt = json.dumps(sp[k])
        tails[k] = {'mentions_convergence_depth_tail': 'convergence_depth_tail' in txt, 'git_head': sp[k].get('git_head')}
    return {'region': [s, e], 'rows': rows, 'spec_sha256': {k: _sha(p) for k, p in SPECS.items()},
            'tail_and_code_state': tails, 'region_text_sha256': hashlib.sha256(reg.encode()).hexdigest()}


# ---- every \ref{subsubsec:certification}: is the cited content in 2.2.7 or only in A.5? -----------------------------
# (needle located by exact text: in section 2.2.7 (subsubsec:certification .. subsec:subproblem) and in A.5)
CERT_REF_DECL = [
    # (line at e10c104, what it cites, needle for that content)
    (458, 'aggregation of C^Op over years/days with discounting (eq:recourse_value)', r'\label{eq:recourse_value}'),
    (558, 'Q(x), the expected discounted operating cost returned by the recourse', 'The recourse value of a plan is the expected discounted operating cost'),
    (943, 'deviation part of the interface settlement inside Q(x)', 'the deviation part of the interface settlement'),
    (1162, 'the certification rule (parameters delta_R, V, tau)', r'$\tau = \delta_R \mathcal{V}/4$'),
    (1180, 'determinacy: determinate / within resolution', 'is called determinate when it is at least'),
    (1180, 'the uncertified form "of that subsection"', 'reported in an uncertified form'),
    (1180, 'the gross operational cost Q', 'The contracted part of the settlement is a transfer between operators and is excluded'),
    (1245, 'thresholds and uncertified bars', 'A difference involving uncertified evaluations is determinate only if'),
    (1386, 'thresholds', 'is called determinate when it is at least'),
    (1439, 'the certification rule (vs the production exit)', 'the residual pass only opens a certifying regime'),
    (1532, 'the rule (stated in full in A.5)', 'The rule is stated in full in'),
    (1533, 'the rule; its oscillatory and monotone branches', 'certified instead on a monotone branch'),
    (1533, 'the threshold of every reported difference', 'is called determinate when it is at least'),
    (1555, 'the clean rule: acceptable status within ten times the tight-tail tolerances', 'within ten times the tight-tail tolerances'),
    (1608, 'the certification rule', 'the residual pass only opens a certifying regime'),
    (1615, 'the exclusions from C^Op', 'is excluded, as are the regularisation terms'),
    (1698, 'the resolution', r'\paragraph{Resolution}'),
    (1743, 'the certifying regime (coordination settings held fixed)', 'with the coordination settings held fixed'),
    (1751, 'the certification rule or its cycle cap', 'does not certify within its cycle cap'),
    (1767, 'the certifying regime opens at the residual pass k_0', 'the residual pass only opens a certifying regime'),
    (1781, 'the uncertified form', 'reported in an uncertified form'),
    (1796, 'certification parameters (tau, P_max, n_w)', r'P_{\max}'),
]


def task_cert_refs(new):
    s0, e0 = P3.region(new, r'\label{subsubsec:certification}', r'\s*\\subsection\{')
    s1, e1 = P3.region(new, r'\label{app:admm_updated_certification}', r'\s*\\subsection\{')
    cl = W167.strip_comments(new)
    found = []
    for i, ln in enumerate(cl, 1):
        for m in re.finditer(r'\\ref\{subsubsec:certification\}', ln):
            found.append(i)
    declared = [d[0] for d in CERT_REF_DECL]
    # every reference must be declared (multiplicity: >= 1 declaration per line, a line with k refs needs k declarations
    # unless the declarations name distinct contents of the same reference)
    missing = sorted(set(found) - set(declared))
    stale = sorted(set(declared) - set(found))
    if missing or stale:
        _fail(f'cert refs: undeclared lines {missing}; declared lines without a reference {stale}')
    rows = []
    for ln, what, needle in CERT_REF_DECL:
        in227 = [s0 + k for k, t in enumerate(new[s0 - 1:e0]) if needle in t]
        inA5 = [s1 + k for k, t in enumerate(new[s1 - 1:e1]) if needle in t]
        rows.append({'line': ln, 'cites': what, 'needle': needle, 'in_2_2_7': in227, 'in_A_5': inA5,
                     'verdict': ('in 2.2.7' if in227 else ('only in A.5' if inA5 else 'in neither')),
                     'context': re.sub(r'\s+', ' ', cl[ln - 1])[:260]})
    return {'region_2_2_7': [s0, e0], 'region_A_5': [s1, e1], 'n_refs': len(found), 'ref_lines': found, 'rows': rows,
            'refs_to_app_certification': [i for i, ln in enumerate(cl, 1) if r'\ref{app:admm_updated_certification}' in ln]}


# ======================================================================================================================
#  3. numbering (static) and the letter's pointers
# ======================================================================================================================
def _match_env(text, pos, env):
    """Index just after the \\end{env} matching the \\begin{env} that ends at `pos` (nesting of the same env counted)."""
    depth, i = 1, pos
    pat = re.compile(r'\\(begin|end)\{' + re.escape(env) + r'\}')
    while depth:
        m = pat.search(text, i)
        if not m:
            raise ValueError(f'unmatched \\begin{{{env}}}')
        depth += 1 if m.group(1) == 'begin' else -1
        i = m.end()
    return i


def _split_rows(body):
    """Top-level rows of an align body: split on \\\\ outside braces and nested environments."""
    rows, depth, envd, cur, i = [], 0, 0, [], 0
    while i < len(body):
        if body.startswith('\\begin{', i):
            envd += 1
        elif body.startswith('\\end{', i):
            envd -= 1
        c = body[i]
        if c == '\\' and body.startswith('\\\\', i) and depth == 0 and envd == 0:
            rows.append(''.join(cur))
            cur = []
            i += 2
            continue
        if c == '\\':                       # an escaped char or a command: copy the backslash and the next char
            cur.append(body[i:i + 2])
            i += 2
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        cur.append(c)
        i += 1
    rows.append(''.join(cur))
    return rows


def numbering(lines):
    """Static elsarticle numbering of a main.tex (comment-stripped). Returns objects and labels."""
    cl = W167.strip_comments(lines)
    text = '\n'.join(cl)
    begin_doc = text.find(r'\begin{document}')

    def line_of(pos):
        return text.count('\n', 0, pos) + 1
    sec = sub = subsub = eq = fig = tab = alg = 0
    app = False
    sec_letter = lambda: chr(ord('A') + sec - 1)  # noqa: E731
    secnum = lambda: sec_letter() if app else str(sec)  # noqa: E731
    cur_label = (None, None)
    objs, labels, warnings = [], {}, []
    tok = re.compile(r'\\(section|subsection|subsubsection|paragraph)(\*?)\s*(?:\[[^\]]*\])?\s*\{'
                     r'|\\appendix\b'
                     r'|\\begin\{(equation\*?|align\*?|subequations|figure\*?|table\*?|longtable|algorithm)\}'
                     r'|\\label\{([^}]*)\}')
    i = begin_doc
    sub_parent, sub_letter, sub_end = None, 0, -1

    def eqnum():
        return (f'{sec_letter()}.{eq}' if app else str(eq))

    def add_label(name, kind, num, ln):
        if name in labels:
            warnings.append(f'duplicate label {name}')
        labels[name] = {'kind': kind, 'number': num, 'line': ln}

    def brace_arg(pos):
        depth, j = 1, pos
        while depth and j < len(text):
            if text[j] == '\\':
                j += 2
                continue
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            j += 1
        return text[pos:j - 1], j
    while True:
        m = tok.search(text, i)
        if sub_parent is not None and (m is None or m.start() >= sub_end):
            sub_parent = None        # amsmath restores the counter to the parent value; eq was never advanced inside
        if not m:
            break
        ln = line_of(m.start())
        if m.group(1):
            level, star = m.group(1), m.group(2)
            title, j = brace_arg(m.end())
            title = re.sub(r'\\textcolor\{[^}]*\}\{(.*)\}', r'\1', title).strip()
            if level == 'paragraph' or star:
                objs.append({'kind': level, 'number': None, 'line': ln, 'title': title})
                i = j
                continue
            if level == 'section':
                sec += 1
                sub = subsub = 0
                if app:
                    eq = 0
                num = secnum()
            elif level == 'subsection':
                sub += 1
                subsub = 0
                num = f'{secnum()}.{sub}'
            else:
                subsub += 1
                num = f'{secnum()}.{sub}.{subsub}'
            objs.append({'kind': level, 'number': num, 'line': ln, 'title': title, 'appendix': app})
            cur_label = (level, num)
            i = j
            continue
        if m.group(0) == r'\appendix':
            app, sec, sub, eq = True, 0, 0, 0
            i = m.end()
            continue
        if m.group(4) is not None:
            add_label(m.group(4), cur_label[0], cur_label[1], ln)
            i = m.end()
            continue
        env = m.group(3)
        end = _match_env(text, m.end(), env)
        body = text[m.end():end - len(f'\\end{{{env}}}')]
        if env == 'subequations':
            eq += 1
            sub_parent, sub_letter = eqnum(), 0
            objs.append({'kind': 'subequations', 'number': sub_parent, 'line': ln})
            cur_label = ('equation', sub_parent)
            sub_end = end
            i = m.end()
            continue
        if env in ('equation*', 'align*'):
            i = end
            continue
        if env == 'equation':
            if re.search(r'\\(nonumber|notag)\b', body) or re.search(r'\\tag\*?\{', body):
                warnings.append(f'l. {ln}: equation with \\nonumber/\\notag/\\tag')
                num = None
            elif sub_parent is not None:
                sub_letter += 1
                num = f'{sub_parent}{chr(96 + sub_letter)}'
            else:
                eq += 1
                num = eqnum()
            objs.append({'kind': 'equation', 'number': num, 'line': ln})
            for lm in re.finditer(r'\\label\{([^}]*)\}', body):
                add_label(lm.group(1), 'equation', num, ln)
            i = end
            continue
        if env == 'align':
            rows = _split_rows(body)
            if rows and rows[-1].strip() == '':
                warnings.append(f'l. {ln}: align ends with \\\\ (amsmath numbers an empty last row)')
            for r in rows:
                if re.search(r'\\(nonumber|notag)\b', r):
                    num = None
                elif sub_parent is not None:
                    sub_letter += 1
                    num = f'{sub_parent}{chr(96 + sub_letter)}'
                else:
                    eq += 1
                    num = eqnum()
                objs.append({'kind': 'equation(align row)', 'number': num, 'line': ln})
                for lm in re.finditer(r'\\label\{([^}]*)\}', r):
                    add_label(lm.group(1), 'equation', num, ln)
            i = end
            continue
        # floats and longtable
        saved = cur_label
        kind = {'figure': 'figure', 'figure*': 'figure', 'table': 'table', 'table*': 'table', 'longtable': 'table',
                'algorithm': 'algorithm'}[env]
        pos = 0
        local = saved
        if env == 'longtable':
            tab += 1
            local = ('table', f'{sec_letter()}.{tab}' if app else str(tab))
            objs.append({'kind': 'table(longtable)', 'number': local[1], 'line': ln, 'caption': None})
        for em in re.finditer(r'\\caption(\*?)|\\label\{([^}]*)\}', body):
            if em.group(2) is not None:
                if local is saved:
                    warnings.append(f'l. {ln}: \\label{{{em.group(2)}}} before any \\caption in {env}')
                add_label(em.group(2), local[0], local[1], ln)
                continue
            if em.group(1) == '*' or env == 'longtable':
                continue
            if kind == 'figure':
                fig += 1
                n = f'{sec_letter()}.{fig}' if app else str(fig)
            elif kind == 'table':
                tab += 1
                n = f'{sec_letter()}.{tab}' if app else str(tab)
            else:
                alg += 1
                n = str(alg)
            p = m.end() + em.end()
            cap = brace_arg(p + 1)[0] if text[p:p + 1] == '{' else ''
            local = (kind, n)
            objs.append({'kind': kind, 'number': n, 'line': ln, 'caption': re.sub(r'\s+', ' ', cap)[:90]})
        cur_label = saved
        i = end
    return {'objects': objs, 'labels': labels, 'warnings': warnings,
            'counts': {'figures': fig, 'tables': tab, 'algorithms': alg}}


def heading_key(numb, title_regex):
    hits = [o for o in numb['objects'] if o['kind'] in ('section', 'subsection', 'subsubsection')
            and re.search(title_regex, o['title'])]
    return hits[0] if len(hits) == 1 else None


# ---- the letter --------------------------------------------------------------------------------------------------------
def _blank_cmd(text, cmd):
    out, i = list(text), 0
    while True:
        j = text.find(cmd + '{', i)
        if j < 0:
            break
        k, depth = j + len(cmd) + 1, 1
        while depth:
            c = text[k]
            if c == '\\':
                k += 2
                continue
            depth += {'{': 1, '}': -1}.get(c, 0)
            k += 1
        for p in range(j, k):
            if out[p] != '\n':
                out[p] = ' '
        i = k
    return ''.join(out)


KW = {'Section': 'section', 'Sections': 'section', 'Subsection': 'section', 'Subsections': 'section',
      'Table': 'table', 'Tables': 'table', 'Figure': 'figure', 'Figures': 'figure', 'Fig.': 'figure', 'Figs.': 'figure',
      'Eq.': 'equation', 'Eqs.': 'equation', 'Equation': 'equation', 'Equations': 'equation', 'Appendix': 'appendix',
      'Appendices': 'appendix', 'Algorithm': 'algorithm'}
NUM_D = r'\(?\d+(?:\.\d+)*\)?'
NUM_A = r'\(?[A-E](?:\.\d+)?\)?'
SEP = r'\s*(?:--|,\s*and|,|and)\s*~?'
PTR = re.compile(r'(?<![\w.])(Sections?|Subsections?|Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Algorithm)'
                 r'(?:~|\s)\s*(' + NUM_D + r'(?:' + SEP + NUM_D + r')*)')
PTR_APP = re.compile(r'(?<![\w.])(Appendix|Appendices)(?:~|\s)\s*(' + NUM_A + r'(?:' + SEP + NUM_A + r')*)')


def letter_pointers(path):
    t = _read(path)
    s = _blank_cmd(t, r'\rcomment')
    s = '\n'.join(re.sub(r'(?<!\\)%.*', '', ln) for ln in s.split('\n'))
    doc = s.find(r'\begin{document}')
    atoms = []

    def add(pos, kw, raw, nums, rng):
        atoms.append({'line': s.count('\n', 0, pos) + 1, 'keyword': kw, 'kind': KW.get(kw, kw), 'text': raw,
                      'numbers': nums, 'range': rng})
    for m in list(PTR.finditer(s, doc)) + list(PTR_APP.finditer(s, doc)):
        kw, lst = m.group(1), m.group(2)
        kind = KW[kw]
        parts = re.split(r'\s*(?:,\s*and|,|and)\s*~?', lst)
        for p in parts:
            p = p.strip()
            if '--' in p:
                a, b = [x.strip('() ~') for x in p.split('--')]
                add(m.start(), kw, m.group(0), [a, b], True)
            else:
                v = p.strip('() ~')
                if kind != 'appendix' and not re.fullmatch(r'\d+(?:\.\d+)*', v):
                    continue
                if kind == 'appendix' and not re.fullmatch(r'[A-E](?:\.\d+)?', v):
                    continue
                add(m.start(), kw, m.group(0), [v], False)
        if kind == 'appendix':        # "Appendix~A (A.5 and A.8)"
            tail = re.match(r'\s*\(([A-E]\.\d+)\s+and\s+([A-E]\.\d+)\)', s[m.end():])
            if tail:
                for v in tail.groups():
                    add(m.end(), kw, m.group(0) + tail.group(0), [v], False)
    for m in re.finditer(r'\b(former|new) (\d+\.\d+\.\d+)', s[doc:]):
        add(doc + m.start(), m.group(1), m.group(0), [m.group(2)], False)
    for m in re.finditer(r'\[(\d+)\]--\[(\d+)\]|\[(\d+)(?:,~?(\d+))?\]', s[doc:]):
        if m.group(1):
            add(doc + m.start(), 'reference', m.group(0), [m.group(1), m.group(2)], True)
        else:
            add(doc + m.start(), 'reference', m.group(0), [m.group(3)], False)
            if m.group(4):
                add(doc + m.start(), 'reference', m.group(0), [m.group(4)], False)
    # the heading "R1.5 --- Figure 2" quotes the reviewer: it is caught by PTR ("Figure 2"); nothing else to add
    atoms.sort(key=lambda a: (a['line'],))
    return atoms


# targets: label names, 'title:<regex>' for unlabelled headings, 'cite:<key>', 'SUBMITTED:<why>'
SEC = {'1': 'title:^Introduction$', '2.1': 'subsec:architecture', '2.2': 'subsec:master', '2.2.1': 'title:^Objective Function$',
       '2.2.7': 'subsubsec:certification', '2.3': 'subsec:subproblem', '2.3.2': 'title:^Installed and Available Capacity$',
       '2.3.3': 'subsubsec:network_storage', '2.3.4': 'title:^Degradation Chain of the Shared-ESS Agent$',
       '2.3.5': 'subsubsec:commitment', '3': 'sec:case_studies', '3.5': 'sec:case_ess_params', '3.6': 'sec:case_settings',
       '4': 'sec:results', '4.1': 'subsec:res_baseline', '4.2': 'subsec:res_breakeven', '4.3': 'subsec:res_degradation',
       '4.4': 'subsec:res_coordination', '4.5': 'subsec:res_multiscenario', '4.6': 'subsec:res_sensitivities',
       '4.7': 'subsec:res_certification', '4.8': 'subsec:res_limitations', '5': 'sec:conclusions',
       'A': 'app:admm_updated', 'A.1': 'app:admm_updated_agents', 'A.2': 'app:admm_updated_implementation',
       'A.3': 'app:admm_updated_updates', 'A.4': 'app:admm_updated_stopping', 'A.5': 'app:admm_updated_certification',
       'A.8': 'app:admm_updated_settings', 'E': 'app:results'}
TAB = {'1': 'tab:investment_cost', '6': 'tab:ageing_calibrations', '7': 'tab:res_value_ladder', '11': 'tab:res_ageing',
       '12': 'tab:res_benchmark', '14': 'tab:res_year_ladder', '15': 'tab:res_discount'}
TAB_7_15 = ['tab:res_value_ladder', 'tab:res_breakeven', 'tab:res_flex_ladder', 'tab:res_energy_ladder_m2', 'tab:res_ageing',
            'tab:res_benchmark', 'tab:res_multiscenario', 'tab:res_year_ladder', 'tab:res_discount']
FIG = {'1': 'fig:two-stage_tool_framework', '2': 'fig:search_generic', '3': 'fig:tso_dso_coordination_cs1_network_diagram',
       '4': 'fig:investment_lattice', '7': 'fig:res_ageing'}
EQS = {'11': ['eq:recourse_value'], '16': ['eq:soc_limits'],
       '15-19': ['eq:soc_recursion', 'eq:soc_limits', 'eq:soc_closure', 'eq:converter_capability', 'eq:network_complementarity'],
       '20-26': ['eq:esso_net_power', 'eq:cohort_allocation', 'eq:cell_throughput', 'eq:cycling_loss', 'eq:soh_chain',
                 'eq:soh_floor', 'eq:cycle_life_calibration'],
       '29-30': ['eq:salvage_credit', 'eq:salvage_remaining_life']}
# per-pointer exceptions to the default intent (line, kind, number) -> target / needle
SPECIAL = {
    (100, 'figure', '2'): 'SUBMITTED:the R1.5 heading quotes the reviewer\'s figure number (submitted Figure 2, Addendum 75)',
    (102, 'figure', '2'): 'SUBMITTED:"Figure 2 of the submitted version"',
    (141, 'former', '2.2.7'): 'SUBMITTED:"former 2.2.7 Benders\' cuts"',
    (232, 'section', '4.4'): 'SUBMITTED:the reviewer\'s Section 4.4 (submitted version)',
    (232, 'figure', '15'): 'SUBMITTED:the reviewer\'s Figure 15 (Addendum 75: no such figure in the submitted source)',
}
NEEDLES = {   # (line, kind, number) -> exact text expected in the target region at e10c104 (informational)
    (42, 'section', '2.1'): 'mesh adaptive direct search', (45, 'section', '3.5'): r'$\phi^{\text{Cal}} = 0.985$',
    (58, 'section', '4.8'): 'the value is affine in size', (75, 'section', '3'): 'three market scenarios and three operation scenarios',
    (77, 'section', '3.5'): 'lithium-ion', (78, 'section', '4.5'): '0.9331', (79, 'section', '4.8'): 'limits the generality',
    (81, 'section', '4'): 'The objective is the gross operational cost', (82, 'section', '2.3.4'): 'cannot be uniquely attributed',
    (82, 'section', '4.8'): 'not unique under a jointly determined schedule',
    (82, 'section', '5'): 'allocation of the degradation of a shared asset',
    (98, 'section', '1'): 'cost of capital', (102, 'figure', '3'): 'network_diagram_ieee9.pdf',
    (125, 'section', '3.6'): r'$\min\{k_0 + 109, 300\}$', (125, 'section', '4.7'): 'new recourse evaluations in all',
    (136, 'section', '2.3.2'): 'The investment plan is a fixed parameter of the recourse',
    (142, 'new', '2.2.7'): 'Recourse Evaluation and Certification', (157, 'appendix', 'A.3'): 'residual balancing',
    (160, 'section', '4.7'): 'Of the 42 single-scenario evaluations', (160, 'appendix', 'E'): 'settling slack',
    (164, 'appendix', 'A.8'): r'$\rho^{V}_0 = 0.0077$', (168, 'appendix', 'A.5'): 'Clean evidence',
    (182, 'section', '1'): 'aghdam_2025', (205, 'section', '2.3'): 'consensus variable',
    (220, 'section', '4.1'): 'Applying the terminal salvage credit', (220, 'appendix', 'E'): 'salvage',
    (221, 'section', '3.5'): '10--90', (241, 'section', '2.3.5'): r'\bar{\pi}_t',
    (274, 'section', '2.2.1'): r'(1+r)^{y^{\text{Inv}}-y_0}', (283, 'section', '3.6'): 'no-reverse-flow',
    (283, 'section', '4.4'): '1 (passive) and 8 (price-taker)', (284, 'section', '4.8'): 'monotone branch',
    (285, 'section', '4.7'): 'single-threaded', (285, 'appendix', 'A.8'): 'IPOPT~3.14.18',
    (288, 'figure', '4'): 'fig_lattice_case.pdf',
}


def _target(atom):
    key = (atom['line'], atom['kind'], atom['numbers'][0])
    if key in SPECIAL:
        return SPECIAL[key]
    k, nums = atom['kind'], atom['numbers']
    if k == 'reference':
        if atom['range']:
            return ['cite:aghdam_2025', 'cite:aghdam_2023', 'cite:aghdam_2024']
        return {'7': 'cite:aghdam_2025', '8': 'cite:aghdam_2023', '9': 'cite:aghdam_2024'}.get(nums[0])
    if k in ('section', 'appendix', 'new'):
        if atom['range']:
            a, b = nums
            if (a, b) == ('2.3.3', '2.3.5'):
                return [SEC['2.3.3'], SEC['2.3.4'], SEC['2.3.5']]
            if (a, b) == ('A.3', 'A.5'):
                return [SEC['A.3'], SEC['A.4'], SEC['A.5']]
            if (a, b) == ('A.1', 'A.2'):
                return [SEC['A.1'], SEC['A.2']]
            return None
        return SEC.get(nums[0])
    if k == 'table':
        if atom['range']:
            return TAB_7_15 if nums == ['7', '15'] else None
        return TAB.get(nums[0])
    if k == 'figure':
        return FIG.get(nums[0])
    if k == 'equation':
        return EQS.get('-'.join(nums)) if atom['range'] else EQS.get(nums[0])
    if k == 'algorithm':
        return 'alg:shared_ess_planning_mads' if nums == ['1'] else None
    return None


def _resolve(target, numb, cites):
    if target.startswith('cite:'):
        k = target[5:]
        return str(cites.index(k) + 1) if k in cites else None
    if target.startswith('title:'):
        h = heading_key(numb, target[6:])
        return h['number'] if h else None
    lab = numb['labels'].get(target)
    return lab['number'] if lab else None


def _region_text(lines, target):
    """Text of the target's region (sections: to the next heading of the same or higher level; floats: the float)."""
    if target.startswith('cite:'):
        return '\n'.join(lines)
    if target.startswith('title:'):
        idx = next((k for k, ln in enumerate(lines) if re.match(r'\s*\\(section|subsection|subsubsection)\{', ln)
                    and re.search(target[6:], re.sub(r'\\textcolor\{[^}]*\}\{(.*)\}', r'\1',
                                                     re.sub(r'^\s*\\\w+\{(.*)\}\s*$', r'\1', ln)).strip())), None)
    else:
        idx = next((k for k, ln in enumerate(lines) if f'\\label{{{target}}}' in ln), None)
    if idx is None:
        return ''
    if target.startswith(('fig:', 'tab:', 'alg:', 'eq:')):
        s = max((k for k in range(idx + 1) if re.match(r'\s*\\begin\{(figure|table|algorithm|equation)', lines[k])), default=idx)
        e = next((k for k in range(idx, len(lines)) if re.match(r'\s*\\end\{(figure|table|algorithm|equation)', lines[k])), idx)
        return '\n'.join(lines[s:e + 1])
    h = next(k for k in range(idx, -1, -1) if re.match(r'\s*\\(section|subsection|subsubsection)\{', lines[k]))
    lev = re.match(r'\s*\\(section|subsection|subsubsection)\{', lines[h]).group(1)
    stop = {'section': ('section',), 'subsection': ('section', 'subsection'),
            'subsubsection': ('section', 'subsection', 'subsubsection')}[lev]
    e = next((k for k in range(h + 1, len(lines)) if re.match(r'\s*\\(' + '|'.join(stop) + r')\{', lines[k])
              or re.match(r'\s*\\appendix', lines[k])), len(lines))
    return '\n'.join(lines[h:e])


def task_letter(lines_e, lines_6, lines_p):
    num_e, num_6 = numbering(lines_e), numbering(lines_6)
    num_p = numbering(lines_p)
    cites_e, cites_6 = cite_order(lines_e), cite_order(lines_6)
    atoms = letter_pointers(LETTER_CLONE)
    rows, undeclared = [], []
    for a in atoms:
        tgt = _target(a)
        row = {k: a[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range')}
        row['target'] = tgt
        if tgt is None:
            undeclared.append(row)
            row['verdict'] = 'UNDECLARED'
            rows.append(row)
            continue
        if isinstance(tgt, str) and tgt.startswith('SUBMITTED:'):
            row['verdict'] = 'SUBMITTED'
            row['note'] = tgt[10:]
            rows.append(row)
            continue
        tl = tgt if isinstance(tgt, list) else [tgt]
        n_e = [_resolve(t, num_e, cites_e) for t in tl]
        n_6 = [_resolve(t, num_6, cites_6) for t in tl]
        if any(v is None for v in n_e + n_6):
            row['verdict'] = 'AMBIGUOUS'
            row['note'] = f'target not resolvable: e10c104 {n_e}, 6ff8f3c {n_6}'
            rows.append(row)
            continue
        want = a['numbers']
        if a['range']:
            got_e, got_6 = [n_e[0], n_e[-1]], [n_6[0], n_6[-1]]
            # the members must be consecutive at e10c104
            row['members'] = tl
            row['members_e10c104'] = n_e
            row['members_6ff8f3c'] = n_6
        else:
            got_e, got_6 = n_e, n_6
        row['number_e10c104'] = '--'.join(got_e)
        row['number_6ff8f3c'] = '--'.join(got_6)
        row['letter_equals_6ff8f3c'] = got_6 == want
        row['changed_6ff8f3c_to_e10c104'] = got_6 != got_e
        row['verdict'] = 'MATCH' if got_e == want else 'MISMATCH'
        if row['verdict'] == 'MISMATCH':
            row['should_be'] = '--'.join(got_e)
        nd = NEEDLES.get((a['line'], a['kind'], a['numbers'][0]))
        if nd is not None:
            reg = '\n'.join(_region_text(W167.strip_comments(lines_e), t) for t in tl)
            row['content_needle'] = nd
            row['content_present_e10c104'] = nd in reg if not tl[0].startswith('cite:') else (tl[0][5:] in cites_e)
        rows.append(row)
    if undeclared:
        _fail(f'letter: {len(undeclared)} pointer(s) without a declared target: '
              f'{[(r["line"], r["text"]) for r in undeclared]}')
    used = {(r['line'], r['kind'], r['numbers'][0]) for r in rows}
    for d in list(SPECIAL) + list(NEEDLES):
        if d not in used:
            _fail(f'letter: declaration {d} matches no extracted pointer')
    counts = {}
    for r in rows:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    changed = [r for r in rows if r.get('changed_6ff8f3c_to_e10c104')]
    return {'assumptions': [
                'elsarticle.cls as in the clone\'s elsarticle.dtx (\\appendix: section, subsection, equation reset; '
                'figure/table prefixed by the section letter, counters not reset; equations reset per appendix section; '
                '\\thesection = "Appendix A"); \\paragraph unnumbered (article secnumdepth 3); no \\numberwithin, '
                '\\setcounter or \\the... redefinition in main.tex',
                'amsmath: equation numbered unless \\nonumber/\\notag/\\tag; align numbered per top-level row; '
                'subequations: one parent number, inner rows lettered, a \\label inside subequations but outside the '
                'inner environment takes the parent number; aligned inside equation: one number',
                'longtable (all versions in TeX Live 2021-2026): \\LT@array steps the table counter at the start of '
                'every longtable whose \\LTcaptype is table (the default), whether or not it has a \\caption; '
                'table/figure environments step only at \\caption; \\subfloat steps the subfigure counter only',
                'algorithm2e: \\caption steps algocf; not prefixed in the appendix',
                'citations: elsarticle-num.bst (in the clone) has no SORT, so references are numbered in order of first '
                'citation; order taken from the source; floats that move could reorder citations made inside them '
                '(none of the letter\'s cited keys is cited in a float)',
                'the numbering is that of a complete LaTeX+BibTeX compile; no compiler was available to confirm it'],
            'numbering_e10c104': num_e, 'numbering_6ff8f3c': num_6,
            'appendix_A_subsections_51a6fb5': {k: v['number'] for k, v in num_p['labels'].items() if k.startswith('app:admm')},
            'cite_order_e10c104_first_12': cites_e[:12], 'cite_order_6ff8f3c_first_12': cites_6[:12],
            'pointers': rows, 'counts': counts, 'n_pointers': len(rows),
            'changed_6ff8f3c_to_e10c104': [{k: r[k] for k in ('line', 'text', 'number_6ff8f3c', 'number_e10c104')}
                                           for r in changed],
            'letter_not_equal_6ff8f3c': [{k: r.get(k) for k in ('line', 'text', 'numbers', 'number_6ff8f3c')}
                                         for r in rows if r.get('letter_equals_6ff8f3c') is False],
            'content_absent_e10c104': [{k: r.get(k) for k in ('line', 'text', 'target', 'content_needle')}
                                       for r in rows if r.get('content_present_e10c104') is False]}


# ======================================================================================================================
def main():
    global OUT_DIR
    if '--out-dir' in sys.argv:                       # trial runs only; the directory must hold nothing but launch.log
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
    out_json, out_man = os.path.join(OUT_DIR, OUT_NAME), os.path.join(OUT_DIR, MANIFEST_NAME)
    t0 = datetime.now(timezone.utc)
    _log("W177b-3' re-run at e10c104: start")
    out = {'schema': 'p515_s53_w177b3r_checks', 'task': "W177b-3' (Addendum 76 Part 3 at e10c104; change-log items; "
                                                        "letter pointers, Amendment item 2)",
           'started_utc': t0.isoformat(), 'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'script': {'path': os.path.basename(__file__), 'sha256': _sha(__file__)},
           'imported': {m: _sha(m) for m in (PRIOR_SCRIPT, 'p515_s53_w167_nomenclature_years.py',
                                             'p515_s47_phase_b_record.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py')}}
    out['identity'] = identity()
    _log(f"identity: head {out['identity']['clone_head']} sha_ok {out['identity']['sha_ok']} letter_ok {out['identity']['letter_ok']}")
    new = as_lines(_read(MAIN_TEX))
    old = as_lines(show(PRIOR_HEAD))
    six = as_lines(show(LETTER_HEAD))
    mp, hunks = line_map(old, new)
    out['diff'] = {'hunks': hunks, 'n_hunks': len(hunks),
                   'old_lines_changed': sum(h['old'][1] - h['old'][0] + 1 for h in hunks if h['tag'] != 'insert'),
                   'new_lines_changed': sum(h['new'][1] - h['new'][0] + 1 for h in hunks if h['tag'] != 'delete'),
                   'hunk_sections_new': [section_of(new, h['new'][0]) for h in hunks]}
    out['prior_rows'] = prior_rows(old, new, mp, hunks)
    _log(f"diff: {len(hunks)} hunks; prior rows with all lines unchanged: "
         f"{sum(r['all_lines_unchanged'] for r in out['prior_rows'])}/{len(out['prior_rows'])}")
    out['abc_rerun'] = rerun_abc(old, new, mp)
    _log('abc: ' + '; '.join(f"{k}: {len(v['differences_after_line_map'])} differences, anchor failures e10c104 "
                             f"{len(v['anchor_failures_e10c104'])}, rerun==committed {v['rerun_on_51a6fb5_equals_committed_W177b3_json']}"
                             for k, v in out['abc_rerun'].items()))
    out['d_nomenclature'] = task_d(old, new)
    _log(f"d: unused {out['d_nomenclature']['defined_unused_now']}; used-undefined "
         f"{out['d_nomenclature']['current']['n_used_undefined']} (+{len(out['d_nomenclature']['used_undefined_now_not_then'])} "
         f"/ -{len(out['d_nomenclature']['used_undefined_then_not_now'])})")
    out['e_ageing_figure'] = task_e(old, new)
    _log(f"e: rerun needed {out['e_ageing_figure']['rerun_needed']}")
    out['f_refs_labels'] = task_f(old, new)
    fcur = out['f_refs_labels']['current']
    _log(f"f: labels {fcur['n_labels']}, dup {fcur['duplicate_labels']}, undefined refs {fcur['undefined_refs']}, "
         f"undefined cites {fcur['undefined_cites']}")
    out['item2_lattice'] = task_lattice()
    _log(f"lattice: admissible {out['item2_lattice']['single_node_2025']['n_admissible']}, neighbours "
         f"{out['item2_lattice']['fourteen_neighbours']['n_neighbours']}")
    out['item2_run_settings'] = task_run_settings(new)
    _log('3.6 run settings: ' + ', '.join(f"{r['id']}={r['match']}" for r in out['item2_run_settings']['rows']))
    out['item2_cert_refs'] = task_cert_refs(new)
    _log(f"cert refs: {out['item2_cert_refs']['n_refs']} refs; " + ', '.join(
        f"l.{r['line']} {r['verdict']}" for r in out['item2_cert_refs']['rows'] if r['verdict'] != 'in 2.2.7'))
    out['item3_letter'] = task_letter(new, six, old)
    _log(f"letter: {out['item3_letter']['n_pointers']} pointers {out['item3_letter']['counts']}; changed "
         f"{len(out['item3_letter']['changed_6ff8f3c_to_e10c104'])}")
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    g = {'own': GUARD.verify(0), 'w177b3': P3.GUARD.verify(0), 'w167': W167.GUARD.verify(0), 'phase_b_record': PB.PARENT_GUARD.verify(0)}
    out['guards'] = {'verify_0_failures': g, 'counts': {'own': GUARD.counts, 'w177b3': P3.GUARD.counts, 'w167': W167.GUARD.counts,
                                                        'phase_b_record': PB.PARENT_GUARD.counts},
                     'pickle_counts': dict(PICKLE_COUNTS), 'w177b3_pickle_counts': dict(P3.PICKLE_COUNTS),
                     'w167_pickle_counts': dict(W167.PICKLE_COUNTS), 'production_modules_imported': forbidden}
    if any(g.values()) or any(PICKLE_COUNTS.values()) or any(P3.PICKLE_COUNTS.values()) or any(W167.PICKLE_COUNTS.values()) \
            or forbidden:
        _fail(f'guards: {out["guards"]}')
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'w', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    inputs = sorted({MAIN_TEX, BIB, LETTER_CLONE, LETTER_V2, PRIOR_JSON, PRIOR_SCRIPT, FIG_V2_CLONE, LATTICE_PDF_CLONE,
                     LATTICE_PDF_REVIEW, LATTICE_PY, W2_TABLE, W167_COSTS, S47_RESULTS, os.path.join(CLONE, 'elsarticle.dtx'),
                     os.path.join(CLONE, 'elsarticle-num.bst'), 'p515_s53_w167_nomenclature_years.py',
                     'p515_s47_phase_b_record.py', 'p515_s44_campaign_harness.py', 'gate_result_io.py',
                     os.path.basename(__file__)} | set(SPECS.values())
                    | {P3.SPEC_V6, P3.SPEC_EXT, P3.SPEC_A64, P3.SPEC_3X3, P3.SRP1_PARAMS, P3.ESS_PARAMS, P3.CASE9,
                       P3.V6_LOG_SRP1, P3.V6_LOG_3X3, P3.PARAGRAPHS_V5, P3.SC1, P3.SC2, P3.SC5, P3.SC6, P3.H118, P3.H101,
                       P3.H142, P3.C132, P3.SED, P3.SRP, 'network.py', 'definitions.py'} | set(P3.CASE33))
    manifest = {'outputs': {OUT_NAME: _sha(out_json)}, 'out_dir': OUT_DIR,
                'git_show_blobs_sha256': {f'{c}:main.tex': hashlib.sha256(show(c).encode('utf-8')).hexdigest()
                                          for c in (PRIOR_HEAD, LETTER_HEAD, 'HEAD')},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(out_man, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'written {out_json} and {out_man}; check failures: {len(CHECK_FAILURES)}; guards ok: {not any(g.values())}; '
         f'pickle {PICKLE_COUNTS}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
