"""
P5.15 Addendum 77, W178 item (d), Planner task W178c: the letter v4 (response_to_reviewers_draft.tex at Overleaf 9fd19e7
= manuscript_review/response_to_reviewers_v4.tex) against main.tex at 9fd19e7 (= manuscript_review/main_820ee1b_fixed.tex).
ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT; read-only on the clone, manuscript_review/, the frozen JSON
(590088fe), every committed artifact and every earlier script (p515_s53_w177b3r_checks.py is IMPORTED, not edited).

What it does (static reading only):
  0. Identity: clone HEAD (declared 9fd19e7), main.tex / letter sha256 and line counts, both byte-identical to their
     manuscript_review/ copies, clone .tex files equal to their HEAD blobs; the letter diff e10c104 -> HEAD (git, read
     only) and the replies it touches; the committed W177b-3' JSON (c27f4062...) read as the prior pointer result.
  1. Capture-path checklist, asserted BEFORE any task runs: every quantity this task must report has a capture
     function and every input it reads exists (fail fast, exit 4, naming the missing item).
  2. Static numbering of main.tex at 9fd19e7: W177b-3''s numbering() (imported), then the \\addtocounter{table}{n}
     statements applied in document order (LaTeX counters are global; \\addtocounter is not undone by groups); any
     other counter manipulation (\\setcounter, \\addtocounter of a non-table counter, \\numberwithin) refuses.
     Compared, item by item, with the expert's compile numbers of Addendum 77 ruling 5; also 0 undefined references and
     0 duplicate labels (the expert's compile statement), statically.
  3. Every numbered pointer of the letter outside the quoted reviewer comments (W177b-3''s extractor, imported):
     MATCH / MISMATCH (with the correct number) / SUBMITTED / AMBIGUOUS against the adjusted static numbering, beside
     the unadjusted number (what W177b-3''s numbering alone gives) and the number at e10c104; content needles.
  4. Part 1 (numbers) on the letter's CHANGED replies: every numeric token of those replies (pointers and reviewer
     item numbers masked) must carry a declaration; each declared value is checked against the frozen JSON (590088fe),
     the W160 export, a named record or the case files at its written precision; undeclared tokens fail.
  5. Part 2 (claims) on the changed replies, W174b rating (H: a referee re-deriving from the text gets a different
     model/number/verdict; M: imprecise, not a different result; L: wording): each claim's letter fragment must occur
     exactly once in the letter; its evidence (main.tex fragments by region, code anchors by exact text, record values)
     is located mechanically and quoted; a failed expectation is a check failure. L1-L4 checked against Addendum 77's
     rulings 1-4 (their wording is not in manuscript_review/w177b_fixes.tex, which holds A1-A20 and B1-B4 only).
     The W177b-2' letter rows in the changed replies are re-scored.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at exit; the
guards of the imported W177b-3', W177b-3, W167 and s47 Phase B record modules verified at 0 too; pickle.load / loads
blocked and counted (0 expected); no production module imported (asserted). Refuses to run concurrently with another
copy of itself. Output JSON written through gate_result_io (string flags refused).

Output (write-once): the directory must exist and hold nothing but launch.log; writes w178c_letter_checks.json,
w178c_letter_checks.md and manifest_sha256.json. `--post-run` (after the run and the W100 typing test) writes
manifest_post_run_sha256.json over launch.log, the manifest, the MD and the typing-test files, once.
`--out-dir DIR`: trial runs only (absolute path outside the repository; same refusal rule).

Commands (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w178c_letter && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w178c_letter_checks.py \\
      > data/SRP1/Results/P515S53/w178c_letter/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
      --out data/SRP1/Results/P515S53/w178c_letter/w178c_bool_typing_test.json \\
      > data/SRP1/Results/P515S53/w178c_letter/w178c_bool_typing_test.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w178c_letter_checks.py --post-run
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label="P5.15 W178c letter v4 checks (never solves)").install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W178c: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W178c: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s53_w177b3r_checks as R  # noqa: E402  (W177b-3'; installs its own permitted=() guard; NOT edited)
import gate_result_io as GRIO  # noqa: E402

P3, PB, W167 = R.P3, R.PB, R.W167
pickle.load, pickle.loads = _blocked_load, _blocked_loads     # the imported modules re-bound them to their blockers

# ======================================================================================================================
#  constants
# ======================================================================================================================
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w178c_letter')
OUT_NAME = 'w178c_letter_checks.json'
MD_NAME = 'w178c_letter_checks.md'
MANIFEST_NAME = 'manifest_sha256.json'
POST_NAME = 'manifest_post_run_sha256.json'
TYPING_NAMES = ('w178c_bool_typing_test.json', 'w178c_bool_typing_test.log')
CLONE = R.CLONE
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
LETTER_V4 = os.path.join('manuscript_review', 'response_to_reviewers_v4.tex')
MAIN_FIXED = os.path.join('manuscript_review', 'main_820ee1b_fixed.tex')
FIXES_TEX = os.path.join('manuscript_review', 'w177b_fixes.tex')
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'
DECLARED_HEAD = '9fd19e7'
DECLARED_MAIN_SHA = 'd18384d669b7a4fbbcfb95e7ead5181e6ffdc5001f51f60d773053e1473df15d'
DECLARED_MAIN_LINES = 2511
DECLARED_LETTER_SHA = '18992d8f3bcb45f2030a16f981c6cc6f474e6332fe791e23ee7030d130c9857b'
DECLARED_LETTER_LINES = 291
PRIOR_HEAD = 'e10c104'
PRIOR_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w177b_part3_e10c104', 'w177b3r_checks.json')
PRIOR_JSON_SHA_PREFIX = 'c27f4062'
PRIOR_SCRIPT = 'p515_s53_w177b3r_checks.py'
PRIOR_SCRIPT_SHA_PREFIX = 'e8b23002'
FZ = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FZ_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
EXPORT_T6 = os.path.join('manuscript_review', 'export_W160', 'T6.csv')
W153_SALV = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w153_step5_rows', 'w153_salvage_addback.json')
W153_CERT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w153_step5_rows', 'w153_certification_statistics.json')
W153_DISC = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w153_step5_rows', 'w153_discount_row.json')
W163 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w163_paragraphs_v4_check', 'w163_paragraphs_v4_figure_check.json')
SRP1_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
INSTANCE_3X3 = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
CASES = {'case9': os.path.join('data', 'SRP1', 'case9', 'case9_2025.json'),
         'case33_1': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_2025.json'),
         'case33_2': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_2025.json'),
         'case33_3': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_2025.json')}
SUBMITTED = os.path.join('manuscript_submitted', 'main.tex')
F2_SPEC = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w118_resettle', 'campaign_s53_w118_resettle_r2_f2_incumbent',
                       'campaign_spec_s53_w118_resettle_r2_f2_incumbent_b1dd9b82.json')
A1_CAMPAIGN = 'p515_s45_a1_campaign.py'          # read as TEXT (lattice constants); never imported
CODE = {'srp': 'shared_resources_planning.py', 'sesd': 'shared_energy_storage_data.py', 'defs': 'definitions.py',
        'mch': 'model_construction_helpers.py'}
REVISION_CONTEXT = 'REVISION_CONTEXT.md'
FORBIDDEN_MODULE_PREFIXES = R.FORBIDDEN_MODULE_PREFIXES
CHECK_FAILURES = []
SCRIPT_NAME = os.path.basename(__file__)


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


_sha, _git, _read = R._sha, R._git, R._read


def _json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def show(commit, path='main.tex'):
    return _git(['show', f'{commit}:{path}'], cwd=CLONE)


def lines_of(text):
    return text.split('\n')


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


def identity():
    head = _git(['rev-parse', '--short', 'HEAD'], cwd=CLONE).strip()
    head_full = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    main_text, letter_text = _read(MAIN_TEX), _read(LETTER)
    d = {'clone_head': head, 'clone_head_full': head_full, 'clone_head_subject': _git(['log', '-1', '--format=%s'], cwd=CLONE).strip(),
         'clone_tracked_modified': status, 'clone_untracked': _git(['ls-files', '--others', '--exclude-standard'], cwd=CLONE).split(),
         'main_sha256': _sha(MAIN_TEX), 'main_lines': main_text.count('\n'),
         'letter_sha256': _sha(LETTER), 'letter_lines': letter_text.count('\n'),
         'letter_equals_v4': _sha(LETTER) == _sha(LETTER_V4), 'main_equals_820ee1b_fixed': _sha(MAIN_TEX) == _sha(MAIN_FIXED),
         'main_equals_head_blob': hashlib.sha256(show('HEAD').encode('utf-8')).hexdigest() == _sha(MAIN_TEX),
         'letter_equals_head_blob': hashlib.sha256(show('HEAD', 'response_to_reviewers_draft.tex').encode('utf-8')).hexdigest() == _sha(LETTER),
         'frozen_json_sha256_ok': _sha(FZ) == FZ_SHA,
         'prior_json_sha256_prefix_ok': _sha(PRIOR_JSON).startswith(PRIOR_JSON_SHA_PREFIX),
         'prior_script_sha256_prefix_ok': _sha(PRIOR_SCRIPT).startswith(PRIOR_SCRIPT_SHA_PREFIX),
         'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip()}
    checks = {'head': head == DECLARED_HEAD, 'main_sha': d['main_sha256'] == DECLARED_MAIN_SHA,
              'main_lines': d['main_lines'] == DECLARED_MAIN_LINES, 'letter_sha': d['letter_sha256'] == DECLARED_LETTER_SHA,
              'letter_lines': d['letter_lines'] == DECLARED_LETTER_LINES, 'letter_equals_v4': d['letter_equals_v4'],
              'main_equals_820ee1b_fixed': d['main_equals_820ee1b_fixed'], 'main_head_blob': d['main_equals_head_blob'],
              'letter_head_blob': d['letter_equals_head_blob'], 'clone_clean': status == '',
              'frozen_json': d['frozen_json_sha256_ok'], 'prior_json': d['prior_json_sha256_prefix_ok'],
              'prior_script': d['prior_script_sha256_prefix_ok']}
    d['checks'] = checks
    for k, v in checks.items():
        if v is not True:
            _fail(f'identity check {k} failed')
    return d


# ======================================================================================================================
#  1. capture-path checklist (asserted before executing)
# ======================================================================================================================
REQUIRED_CAPTURES = {
    'static numbering at 9fd19e7 with \\addtocounter applied': 'numbering_adjusted',
    'static-vs-compile comparison (Addendum 77 ruling 5)': 'task_compile',
    'undefined references and duplicate labels (static)': 'task_refs',
    'every letter pointer: verdict, correct number, unadjusted number, e10c104 number': 'task_pointers',
    'letter diff e10c104 -> HEAD and the changed replies': 'task_letter_diff',
    'Part 1: every numeric token of the changed replies against its source': 'task_numbers',
    'Part 2: every claim of the changed replies with located evidence and rating': 'task_claims',
    'L1-L4 against Addendum 77 rulings 1-4': 'task_l1_l4',
    'W177b-2\' letter rows re-scored': 'task_prior_rows',
    'guards (solve 0, pickle 0, no production import)': 'verify_guards',
}
REQUIRED_R_SYMBOLS = ('numbering', 'letter_pointers', '_resolve', '_region_text', 'heading_key', 'cite_order', '_blank_cmd',
                      'SEC', 'TAB', 'TAB_7_15', 'FIG', 'PTR', 'PTR_APP', 'NEEDLES', 'SPECIAL', 'KW')


def capture_checklist():
    missing = []
    g = globals()
    for q, fn in REQUIRED_CAPTURES.items():
        if not callable(g.get(fn)):
            missing.append(f'capture function {fn} for "{q}"')
    for s in REQUIRED_R_SYMBOLS:
        if not hasattr(R, s):
            missing.append(f'W177b-3\' symbol R.{s}')
    for p in [MAIN_TEX, LETTER, LETTER_V4, MAIN_FIXED, FIXES_TEX, BRIEF, PRIOR_JSON, PRIOR_SCRIPT, FZ, EXPORT_T6, W153_SALV,
              W153_CERT, W153_DISC, W163, SRP1_JSON, INSTANCE_3X3, ESS_PARAMS, SUBMITTED, F2_SPEC, A1_CAMPAIGN,
              REVISION_CONTEXT] + list(CASES.values()) + list(CODE.values()):
        if not os.path.isfile(p):
            missing.append(f'input {p}')
    if missing:
        for m in missing:
            print(f'REFUSED (capture-path checklist): missing {m}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'r_symbols': list(REQUIRED_R_SYMBOLS), 'all_present': True}


# ======================================================================================================================
#  2. numbering with counter adjustments; compile comparison; refs
# ======================================================================================================================
COUNTER_CMD = re.compile(r'\\(addtocounter|setcounter)\{([^}]*)\}\{([^}]*)\}|\\numberwithin\b')


def numbering_adjusted(lines):
    """R.numbering() then every \\addtocounter{table}{n} applied to the table objects/labels that FOLLOW it."""
    cl = W167.strip_comments(lines)
    begin = next(i for i, ln in enumerate(cl, 1) if r'\begin{document}' in ln)
    adjustments, refused = [], []
    for i, ln in enumerate(cl, 1):
        for m in COUNTER_CMD.finditer(ln):
            if m.group(0).startswith(r'\numberwithin'):
                refused.append((i, m.group(0)))
            elif m.group(1) == 'addtocounter' and m.group(2) == 'table' and i > begin:
                adjustments.append({'line': i, 'counter': 'table', 'delta': int(m.group(3))})
            else:
                refused.append((i, m.group(0)))
    if refused:
        _fail(f'counter manipulation not modelled: {refused}')
    raw = R.numbering(lines)
    adj = json.loads(json.dumps(raw))

    def shift(num, ln):
        delta = sum(a['delta'] for a in adjustments if a['line'] < ln)
        if num is None or delta == 0:
            return num
        m = re.fullmatch(r'([A-Z]\.)?(\d+)', num)
        if not m:
            _fail(f'table number {num!r} at l. {ln} not adjustable')
            return num
        return f'{m.group(1) or ""}{int(m.group(2)) + delta}'
    for o in adj['objects']:
        if o['kind'] in ('table', 'table(longtable)'):
            o['number_unadjusted'] = o['number']
            o['number'] = shift(o['number'], o['line'])
    for k, v in adj['labels'].items():
        if v['kind'] == 'table':
            v['number_unadjusted'] = v['number']
            v['number'] = shift(v['number'], v['line'])
    adj['counter_adjustments'] = adjustments
    adj['assumption_addtocounter'] = ('\\addtocounter is global in LaTeX (counters are not restored at group end); each '
                                      '\\addtocounter{table}{-1} after a caption-less longtable cancels the step that '
                                      'longtable made at its start (W177b-3\' assumption 3), so the next captioned table '
                                      'takes the number it had before the longtables')
    return raw, adj


COMPILE_RULING5 = [   # Addendum 77 ruling 5: the expert's compile (demo graphics) of the fixed main.tex
    {'item': 'Tables 1-15 restored', 'kind': 'table_body_range', 'expect': ['1', '15']},
    {'item': 'Q(x) (13)', 'labels': ['eq:recourse_value'], 'expect': ['13']},
    {'item': 'cohort ratings (14)-(15)', 'labels': ['eq:cohort_rated', 'eq:cohort_rated_energy'], 'expect': ['14', '15']},
    {'item': 'SOC block (20)-(28)', 'labels': ['eq:storage_net_active_power', 'eq:soc_recursion', 'eq:soc_limits',
                                                'eq:soc_initial', 'eq:soc_closure', 'eq:soc_closure_slack',
                                                'eq:storage_active_power_limit', 'eq:converter_capability',
                                                'eq:network_complementarity'], 'expect': [str(i) for i in range(20, 29)]},
    {'item': 'SOC limits (22)', 'labels': ['eq:soc_limits'], 'expect': ['22']},
    {'item': 'agent and degradation (29)-(35)', 'labels': ['eq:esso_net_power', 'eq:cohort_allocation', 'eq:cell_throughput',
                                                           'eq:cycling_loss', 'eq:soh_chain', 'eq:soh_floor',
                                                           'eq:cycle_life_calibration'], 'expect': [str(i) for i in range(29, 36)]},
    {'item': 'salvage (38)-(39)', 'labels': ['eq:salvage_credit', 'eq:salvage_remaining_life'], 'expect': ['38', '39']},
]


def task_compile(raw, adj):
    rows = []
    for c in COMPILE_RULING5:
        if c.get('kind') == 'table_body_range':
            body = [o for o in adj['objects'] if o['kind'] == 'table' and re.fullmatch(r'\d+', o['number'] or '')]
            nums = [int(o['number']) for o in body]
            got = [str(min(nums)), str(max(nums))] if nums else []
            contiguous = nums == list(range(1, len(nums) + 1))
            raw_body = [o['number'] for o in raw['objects'] if o['kind'] == 'table' and re.fullmatch(r'\d+', o['number'] or '')]
            rows.append({'item': c['item'], 'static_adjusted': got, 'static_unadjusted': [raw_body[0], raw_body[-1]] if raw_body else [],
                         'n_body_tables': len(nums), 'contiguous_from_1': contiguous,
                         'agree': got == c['expect'] and contiguous})
            continue
        got = [adj['labels'].get(lb, {}).get('number') for lb in c['labels']]
        rows.append({'item': c['item'], 'labels': c['labels'], 'static': got, 'compile': c['expect'], 'agree': got == c['expect']})
    return {'rows': rows, 'n_agree': sum(r['agree'] for r in rows), 'n_items': len(rows),
            'longtable_inference': ('if longtable did NOT step the table counter, three \\addtocounter{table}{-1} would '
                                    'number the first body table -2, not 1; the compile\'s "Tables 1-15 restored" therefore '
                                    'confirms the W177b-3\' longtable assumption that drove the 16 e10c104 MISMATCHes')}


def task_refs(lines, adj):
    cl = W167.strip_comments(lines)
    text = '\n'.join(cl)
    refs = re.findall(r'\\(?:eq|auto|c|C|page|name)?ref\{([^}]*)\}', text)
    keys = set()
    for r_ in refs:
        keys.update(k.strip() for k in r_.split(','))
    undefined = sorted(k for k in keys if k not in adj['labels'])
    dup = [w for w in adj['warnings'] if w.startswith('duplicate label')]
    return {'n_ref_uses': len(refs), 'n_ref_keys': len(keys), 'n_labels': len(adj['labels']),
            'undefined_refs': undefined, 'duplicate_labels': dup, 'numbering_warnings': adj['warnings']}


# ======================================================================================================================
#  3. letter diff and pointers
# ======================================================================================================================
def letter_line_map(old, new):
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    mp, hunks = {}, []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                mp[i1 + k + 1] = j1 + k + 1
        else:
            hunks.append({'tag': tag, 'old': [i1 + 1, i2], 'new': [j1 + 1, j2]})
    return mp, hunks


# the changed replies: (id, regex of the heading line that opens the block, regex of the line that ends it (exclusive))
REPLY_BLOCKS = [
    ('Opening: What we did not do', r'\\paragraph\{What we did not do', r'\\section\*\{Reviewer 1\}'),
    ('R1.2', r'\\subsection\*\{R1\.2 ', r'\\subsection\*\{R1\.4 '),
    ('R1.4', r'\\subsection\*\{R1\.4 ', r'\\subsection\*\{R1\.5 '),
    ('R2.2', r'\\subsection\*\{R2\.2 ', r'\\subsection\*\{R2\.3'),
    ('R2.5-R2.6', r'\\subsection\*\{R2\.5 ', r'\\subsection\*\{R2\.8 '),
    ('R2.10', r'\\subsection\*\{R2\.10 ', r'\\subsection\*\{R2\.11 '),
    ('R2.11-R2.12', r'\\subsection\*\{R2\.11 ', r'\\section\*\{Reviewer 3\}'),
    ('R3.1', r'\\subsection\*\{R3\.1 ', r'\\subsection\*\{R3\.2 '),
    ('R3.7-R3.8', r'\\subsection\*\{R3\.7 ', r'\\section\*\{Further changes'),
]


def reply_ranges(letter_lines):
    out = []
    for rid, a, b in REPLY_BLOCKS:
        s = next((i for i, ln in enumerate(letter_lines, 1) if re.search(a, ln)), None)
        e = next((i for i, ln in enumerate(letter_lines, 1) if s and i > s and re.search(b, ln)), None)
        if s is None or e is None:
            _fail(f'reply block {rid} not located')
            continue
        out.append({'id': rid, 'first_line': s, 'last_line': e - 1})
    return out


def task_letter_diff(letter_lines):
    old = lines_of(show(PRIOR_HEAD, 'response_to_reviewers_draft.tex'))
    mp, hunks = letter_line_map(old, letter_lines)
    ranges = reply_ranges(letter_lines)
    for h in hunks:
        lo, hi = h['new']
        h['replies'] = [r_['id'] for r_ in ranges if lo <= r_['last_line'] and hi >= r_['first_line']] if hi >= lo else []
        if hi >= lo and not h['replies']:
            _fail(f'letter hunk {h} lies outside the declared changed replies')
    touched = sorted({x for h in hunks for x in h['replies']})
    return {'old_sha256': hashlib.sha256('\n'.join(old).encode('utf-8')).hexdigest(), 'n_old_lines': len(old),
            'n_hunks': len(hunks), 'hunks': hunks, 'reply_ranges': ranges, 'replies_touched': touched,
            'declared_replies_untouched': [r_['id'] for r_ in ranges if r_['id'] not in touched], '_map': mp}


EQS = {'13': ['eq:recourse_value'], '22': ['eq:soc_limits'],
       '20-28': ['eq:storage_net_active_power', 'eq:soc_recursion', 'eq:soc_limits', 'eq:soc_initial', 'eq:soc_closure',
                 'eq:soc_closure_slack', 'eq:storage_active_power_limit', 'eq:converter_capability',
                 'eq:network_complementarity'],
       '29-35': ['eq:esso_net_power', 'eq:cohort_allocation', 'eq:cell_throughput', 'eq:cycling_loss', 'eq:soh_chain',
                 'eq:soh_floor', 'eq:cycle_life_calibration'],
       '38-39': ['eq:salvage_credit', 'eq:salvage_remaining_life']}
SPECIAL = {   # v4 lines (the lines of these five pointers are unchanged from v2; verified by the line map)
    (100, 'figure', '2'): R.SPECIAL[(100, 'figure', '2')],
    (102, 'figure', '2'): R.SPECIAL[(102, 'figure', '2')],
    (141, 'former', '2.2.7'): R.SPECIAL[(141, 'former', '2.2.7')],
    (232, 'section', '4.4'): R.SPECIAL[(232, 'section', '4.4')],
    (232, 'figure', '15'): R.SPECIAL[(232, 'figure', '15')],
}
NEEDLES_CHANGED = {   # v4 lines, for pointers on changed lines or new pointers: (line, kind, first number) -> [needles]
    (58, 'section', '4.8'): ['the value is affine in size', 'could not be evaluated within the memory available'],
    (78, 'section', '4.5'): ['0.9331', 'The planning search was not repeated at this instance'],
    (97, 'section', '4.6'): ['re-weight the per-year costs of the 2\\,\\% evaluations'],
    (97, 'table', '15'): ['rates other than 2\\,\\% re-weight the 2\\,\\% evaluations'],
    (98, 'table', '15'): ['rates other than 2\\,\\% re-weight the 2\\,\\% evaluations'],
    (125, 'equation', '13'): ['\\frac{Y_y}{(1+r)^{y-y_0}}', '\\omega_m \\, \\omega_o'],
    (152, 'section', '2.2.7'): ['No convergence guarantee is available'],
    (171, 'table', '12'): ['best of three solver starts'],
    (136, 'section', '2.3.2'): ['whose ratings are the fixed additions'],
    (194, 'equation', '29'): ['\\eta^{\\text{Ch}}_e P^{\\text{Ch}}_{e,y^\\text{Inv},y,d,t}'],
    (210, 'equation', '20'): ['E^{\\text{SoC}}_{e,y,d,t} = E^{\\text{SoC}}_{e,y,d,t-1}', '\\left(Q^{\\text{E}}_{e,y,d,t}\\right)^2'],
    (220, 'equation', '38'): ['SoH^{\\text{Min}}_e}{1 - SoH^{\\text{Min}}_e}', 'T^\\text{Rem}_{e,y^\\text{Inv}} = \\max'],
    (220, 'table', '14'): ['gross and net of the salvage credit'],
    (221, 'section', '4.6'): ['The investment-year comparison above is not among the 60'],
    (221, 'equation', '22'): ['SoC^{\\text{Min}}_e E^{\\text{Av}}_{e,y}'],
    (222, 'equation', '38'): ['SoH^{\\text{Min}}_e}{1 - SoH^{\\text{Min}}_e}'],
    (272, 'equation', '13'): ['\\frac{Y_y}{(1+r)^{y-y_0}}'],
    (274, 'equation', '13'): ['\\frac{Y_y}{(1+r)^{y-y_0}}'],
    (274, 'section', '3'): ['each standing for a five-year block'],
    (275, 'equation', '13'): ['\\frac{Y_y}{(1+r)^{y-y_0}}'],
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
            pair = tuple(nums)
            return {('2.3.3', '2.3.5'): [R.SEC['2.3.3'], R.SEC['2.3.4'], R.SEC['2.3.5']],
                    ('A.3', 'A.5'): [R.SEC['A.3'], R.SEC['A.4'], R.SEC['A.5']],
                    ('A.1', 'A.2'): [R.SEC['A.1'], R.SEC['A.2']]}.get(pair)
        return R.SEC.get(nums[0])
    if k == 'table':
        if atom['range']:
            return R.TAB_7_15 if nums == ['7', '15'] else None
        return R.TAB.get(nums[0])
    if k == 'figure':
        return R.FIG.get(nums[0])
    if k == 'equation':
        return EQS.get('-'.join(nums))
    if k == 'algorithm':
        return 'alg:shared_ess_planning_mads' if nums == ['1'] else None
    return None


def _consecutive(nums):
    parts = [x.rsplit('.', 1) if '.' in x else ['', x] for x in nums]
    if len({p[0] for p in parts}) != 1 or not all(re.fullmatch(r'\d+', p[1]) for p in parts):
        return False
    last = [int(p[1]) for p in parts]
    return last == list(range(last[0], last[0] + len(last)))


def task_pointers(lines_now, raw_now, adj_now, lines_prior, letter_map, letter_hunks):
    num_prior = R.numbering(lines_prior)
    cites_now, cites_prior = R.cite_order(lines_now), R.cite_order(lines_prior)
    atoms = R.letter_pointers(LETTER)
    inv_map = {v: k for k, v in letter_map.items()}
    # needles: v2 declarations carried through the line map (unchanged lines only), then the changed-line declarations
    needles = {}
    for (ln, kind, num), nd in R.NEEDLES.items():
        if ln in letter_map:
            needles.setdefault((letter_map[ln], kind, num), []).append(nd)
    for k, v in NEEDLES_CHANGED.items():
        needles.setdefault(k, [])
        for nd in v:
            if nd not in needles[k]:
                needles[k].append(nd)
    for (ln, kind, num) in SPECIAL:
        if ln not in inv_map:
            _fail(f'SPECIAL pointer line {ln} is not an unchanged line of the letter')
    prior_rows = {(r_['line'], r_['kind'], r_['numbers'][0], r_['text']): r_ for r_ in _json(PRIOR_JSON)['item3_letter']['pointers']}
    rows = []
    for a in atoms:
        tgt = _target(a)
        row = {k: a[k] for k in ('line', 'keyword', 'kind', 'text', 'numbers', 'range')}
        row['target'] = tgt
        row['line_unchanged_since_e10c104'] = a['line'] in inv_map
        row['e10c104_letter_line'] = inv_map.get(a['line'])
        pr = prior_rows.get((inv_map.get(a['line']), a['kind'], a['numbers'][0], a['text'])) if a['line'] in inv_map else None
        row['e10c104_verdict'] = pr['verdict'] if pr else None
        if tgt is None:
            row['verdict'] = 'UNDECLARED'
            rows.append(row)
            continue
        if isinstance(tgt, str) and tgt.startswith('SUBMITTED:'):
            row['verdict'] = 'SUBMITTED'
            row['note'] = tgt[10:]
            rows.append(row)
            continue
        tl = tgt if isinstance(tgt, list) else [tgt]
        n_adj = [R._resolve(t, adj_now, cites_now) for t in tl]
        n_raw = [R._resolve(t, raw_now, cites_now) for t in tl]
        n_pri = [R._resolve(t, num_prior, cites_prior) for t in tl]
        if any(v is None for v in n_adj):
            row['verdict'] = 'AMBIGUOUS'
            row['note'] = f'target not resolvable at 9fd19e7: {n_adj}'
            rows.append(row)
            continue
        want = a['numbers']
        if a['range']:
            got, got_raw = [n_adj[0], n_adj[-1]], [n_raw[0], n_raw[-1]]
            got_pri = [n_pri[0], n_pri[-1]] if None not in (n_pri[0], n_pri[-1]) else None
            row['members'] = tl
            row['members_9fd19e7'] = n_adj
            consec = _consecutive(n_adj)
            row['members_consecutive'] = consec
            if not consec:
                row['verdict'] = 'AMBIGUOUS'
                row['note'] = f'range members not consecutive at 9fd19e7: {n_adj}'
                rows.append(row)
                continue
        else:
            got, got_raw = n_adj, n_raw
            got_pri = n_pri if None not in n_pri else None
        row['number_9fd19e7'] = '--'.join(got)
        row['number_9fd19e7_unadjusted'] = '--'.join(got_raw)
        row['number_e10c104'] = '--'.join(got_pri) if got_pri else None
        row['verdict'] = 'MATCH' if got == want else 'MISMATCH'
        row['verdict_if_unadjusted'] = 'MATCH' if got_raw == want else 'MISMATCH'
        if row['verdict'] == 'MISMATCH':
            row['should_be'] = '--'.join(got)
        nds = needles.get((a['line'], a['kind'], a['numbers'][0]))
        if nds:
            if tl[0].startswith('cite:'):
                row['content_needles'] = {nd: (tl[0][5:] in cites_now) for nd in nds}
            else:
                reg = '\n'.join(R._region_text(W167.strip_comments(lines_now), t) for t in tl)
                row['content_needles'] = {nd: (nd in reg) for nd in nds}
        rows.append(row)
    used = {(r_['line'], r_['kind'], r_['numbers'][0]) for r_ in rows}
    for d in list(SPECIAL) + list(NEEDLES_CHANGED):
        if d not in used:
            _fail(f'pointer declaration {d} matches no extracted pointer')
    und = [r_ for r_ in rows if r_['verdict'] == 'UNDECLARED']
    if und:
        _fail(f'{len(und)} pointer(s) without a declared target: {[(r_["line"], r_["text"]) for r_ in und]}')
    counts = {}
    for r_ in rows:
        counts[r_['verdict']] = counts.get(r_['verdict'], 0) + 1
    by_kind = {}
    for r_ in rows:
        by_kind.setdefault(r_['kind'], {}).setdefault(r_['verdict'], 0)
        by_kind[r_['kind']][r_['verdict']] += 1
    prior_mism = [r_ for r_ in _json(PRIOR_JSON)['item3_letter']['pointers'] if r_['verdict'] == 'MISMATCH']
    loose = dict(letter_map)
    for h in letter_hunks:          # equal-size replace hunks: map line by line (the line was edited, not moved)
        if h['tag'] == 'replace' and h['old'][1] - h['old'][0] == h['new'][1] - h['new'][0]:
            for k in range(h['old'][1] - h['old'][0] + 1):
                loose[h['old'][0] + k] = h['new'][0] + k
    prior_status = []
    for p in prior_mism:
        new_line = loose.get(p['line'])
        hit = [r_ for r_ in rows if r_['line'] == new_line and r_['kind'] == p['kind'] and r_['numbers'] == p['numbers']]
        verdicts = sorted({h['verdict'] for h in hit})
        prior_status.append({'e10c104_line': p['line'], 'text': p['text'], 'v4_line': new_line,
                             'line_edited_since_e10c104': p['line'] not in letter_map, 'n_hits': len(hit),
                             'verdict_now': verdicts[0] if len(verdicts) == 1 else f'{verdicts}'})
        if len(verdicts) != 1:
            _fail(f'e10c104 MISMATCH {p["line"]} {p["text"]}: no unique counterpart in v4 ({len(hit)} hits)')
    return {'n_pointers': len(rows), 'counts': counts, 'by_kind': by_kind, 'pointers': rows,
            'mismatch': [r_ for r_ in rows if r_['verdict'] == 'MISMATCH'],
            'ambiguous': [r_ for r_ in rows if r_['verdict'] == 'AMBIGUOUS'],
            'would_mismatch_without_adjustment': [{k: r_.get(k) for k in ('line', 'text', 'number_9fd19e7_unadjusted')}
                                                  for r_ in rows if r_.get('verdict_if_unadjusted') == 'MISMATCH'],
            'content_absent': [{k: r_.get(k) for k in ('line', 'text', 'target', 'content_needles')}
                               for r_ in rows if r_.get('content_needles') and not all(r_['content_needles'].values())],
            'n_with_needles': sum(1 for r_ in rows if r_.get('content_needles')),
            'e10c104_mismatches_now': prior_status,
            'new_or_changed_pointer_lines': sorted({r_['line'] for r_ in rows if not r_['line_unchanged_since_e10c104']})}


# ======================================================================================================================
#  4. Part 1 -- numbers of the changed replies
# ======================================================================================================================
NUMWORDS = r'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty-five|sixty|doubled|double|twice'
TOKEN = re.compile(r'(?<![\w.\\])(-?\d+(?:\.\d+)*)|(?<![\w-])(' + NUMWORDS + r')(?![\w])', re.I)


def reply_tokens(letter_text, ranges):
    s = R._blank_cmd(letter_text, r'\rcomment')
    L = s.split('\n')
    out = []
    for rg in ranges:
        for ln in range(rg['first_line'], rg['last_line'] + 1):
            x = re.sub(r'(?<!\\)%.*', '', L[ln - 1])
            for pat in (R.PTR, R.PTR_APP):
                x = pat.sub(lambda m: ' ' * len(m.group(0)), x)
            x = x.replace('--', ' \u2013')
            x = re.sub(r'\b(former|new) \d+\.\d+\.\d+', lambda m: ' ' * len(m.group(0)), x)
            x = re.sub(r'R\d\.\d+', lambda m: ' ' * len(m.group(0)), x)
            x = re.sub(r'Reviewer~\d', lambda m: ' ' * len(m.group(0)), x)
            x = re.sub(r'\$3\\times\s*3\$', lambda m: '3x3' + ' ' * (len(m.group(0)) - 3), x)
            x = re.sub(r'\\eta_\{\\mathrm\{(ch|dch)\}\}', lambda m: ' ' * len(m.group(0)), x)
            seen = {}
            for m in TOKEN.finditer(x):
                tok = m.group(1) or m.group(2)
                if tok == '3' and x[m.start():m.start() + 3] == '3x3':
                    continue
                key = tok.lower()
                seen[key] = seen.get(key, 0) + 1
                out.append({'reply': rg['id'], 'line': ln, 'token': tok, 'occ': seen[key]})
            for m in re.finditer(r'3x3', x):
                seen['3x3'] = seen.get('3x3', 0) + 1
                out.append({'reply': rg['id'], 'line': ln, 'token': '3x3', 'occ': seen['3x3']})
    return out


class Src:
    """Sources read once."""
    def __init__(self):
        self.fz = _json(FZ)['tables']
        self.w153s = _json(W153_SALV)['result']['totals']
        self.w153c = _json(W153_CERT)['result']['totals']
        self.w153d = _json(W153_DISC)['result']
        self.w163 = {c['id']: c for c in _json(W163)['checks_in_v4']}
        self.srp1 = _json(SRP1_JSON)
        self.i33 = _json(INSTANCE_3X3)
        self.ess = _json(ESS_PARAMS)
        self.cases = {k: _json(v) for k, v in CASES.items()}
        self.sub = lines_of(_read(SUBMITTED))
        self.f2 = _json(F2_SPEC)
        self.a1 = _read(A1_CAMPAIGN)
        self.defs = _read(CODE['defs'])
        self.t6 = _read(EXPORT_T6)
        claims = self.fz['claims']
        self.f2claim = next(c for c in claims if c['claim_id'] == 'L:y2030__n5_p0.25_e0.5__n7_p1_e3')


def _k1(v):
    return f'{v / 1000.0:.1f}'


def _sub_line(S, needle):
    hits = [i for i, ln in enumerate(S.sub, 1) if needle in ln]
    return hits


def number_declarations(S):
    """(line, token, occ) -> (status_kind, expected written value or None, source, note). status_kind in
    value | derived | language. A 'value'/'derived' row is MATCH iff expected == token."""
    disc = {round(r_['rate'] * 100): r_ for r_ in S.fz['discount']}
    t11 = S.fz['three_by_three']['rows']
    yl = S.fz['year_ladder']
    n_years_srp1, n_days = len(S.srp1['Years']), len(S.srp1['Days'])
    n_dn = len(S.srp1['DistributionNetworks'])
    nodes_i = {r_['connection_node_id'] for r_ in S.srp1['DistributionNetworks']}
    block_len = sorted(set(S.srp1['Years'].values()))
    i33_years = S.i33['Years']
    n_buses = {k: len(v['nodes']) for k, v in S.cases.items()}
    lat_p = re.search(r'^LATTICE_P_STEP\s*=\s*([\d.]+)', S.a1, re.M).group(1)
    lat_e = re.search(r'^LATTICE_E_STEP\s*=\s*([\d.]+)', S.a1, re.M).group(1)
    socmin = re.search(r'^ENERGY_STORAGE_MIN_ENERGY_STORED\s*=\s*([\d.]+)', S.defs, re.M).group(1)
    socmax = re.search(r'^ENERGY_STORAGE_MAX_ENERGY_STORED\s*=\s*([\d.]+)', S.defs, re.M).group(1)
    f2nodes = sum(1 for v in S.f2claim['instance']['ref:5ca4f86c']['candidate_canonical']['nodes'].values() if v[0] > 0 or v[1] > 0)
    sub5 = _sub_line(S, 'five years (2025, 2028, 2031, 2034, and 2037)')
    sub25 = _sub_line(S, 'resulting in 25 operating scenarios per representative day')
    sub1825 = _sub_line(S, '18.25\\%')
    sub9216 = _sub_line(S, '92.16\\%')
    w = lambda n: ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'][n]  # noqa: E731
    D = {}

    def add(line, tok, occ, kind, exp, src, note=None):
        D[(line, tok.lower(), occ)] = {'kind': kind, 'expected': exp, 'source': src, 'note': note}
    # ---- opening, l. 58
    add(58, 'Two', 1, 'derived', w(len(re.findall(r'We did not ', _read(LETTER).split('\n')[57]))),
        'count of "We did not" sentences in the paragraph itself', 'requests not met in full')
    add(58, 'five', 1, 'value', 'five' if sub5 else None, f'{SUBMITTED} l. {sub5}: "five years (2025, 2028, 2031, 2034, and 2037)"')
    add(58, 'twenty-five', 1, 'value', 'twenty-five' if sub25 else None, f'{SUBMITTED} l. {sub25}: "25 operating scenarios per representative day"')
    add(58, '3x3', 1, 'value', '3x3' if (S.i33['NumMarketScenarios'], S.i33['TransmissionNetwork']['num_operation_scenarios']) == (3, 3) else None,
        f'{INSTANCE_3X3}: NumMarketScenarios 3, num_operation_scenarios 3')
    # ---- R1.2
    add(77, '3x3', 1, 'value', '3x3', f'{INSTANCE_3X3}: 3 market x 3 operation scenarios')
    add(78, 'five', 1, 'value', w(len(i33_years)), f'{INSTANCE_3X3} Years: {sorted(i33_years)} (five blocks)')
    add(78, '0.909', 1, 'value', f'{t11[8]["value"]:.3f}', 'frozen three_by_three.rows[8] (R lower end, (V - D)/V_single)')
    add(78, '0.934', 1, 'value', f'{t11[7]["value"]:.3f}', 'frozen three_by_three.rows[7] (R upper end, V/V_single)')
    add(78, '0.933', 1, 'value', f'{t11[9]["value"]:.3f}', 'frozen three_by_three.rows[9] R predicted 0.9331, at the written 3 dp',
        'the export rounds R predicted to 4 dp (0.9331); 3 dp is the letter\'s precision')
    add(78, '9', 1, 'value', str(n_buses['case9']), f'{CASES["case9"]}: {n_buses["case9"]} nodes')
    add(78, 'three', 1, 'value', w(n_dn), f'{SRP1_JSON}: {n_dn} DistributionNetworks')
    add(78, '33', 1, 'value', str(n_buses['case33_1']) if len({n_buses[k] for k in n_buses if k != 'case9'}) == 1 else None,
        f'case33_1/2/3 2025: {[n_buses[k] for k in ("case33_1", "case33_2", "case33_3")]} nodes')
    add(78, '108', 1, 'derived', str(n_buses['case9'] + sum(n_buses[k] for k in ('case33_1', 'case33_2', 'case33_3'))),
        '9 + 33 + 33 + 33 from the case files')
    add(80, '18.25', 1, 'value', '18.25' if sub1825 else None, f'{SUBMITTED} l. {sub1825} (withdrawn figure)')
    add(80, '92.16', 1, 'value', '92.16' if sub9216 else None, f'{SUBMITTED} l. {sub9216} (withdrawn figure)')
    add(80, '13.9', 1, 'value', f'{S.fz["benchmark"]["benefit_relative"] * 100:.1f}', 'frozen benchmark.benefit_relative')
    add(80, '15', 1, 'derived', str(sum(S.srp1['Years'].values())), f'{SRP1_JSON} Years: sum of block lengths {S.srp1["Years"]}')
    add(81, 'three', 1, 'value', w(n_years_srp1), f'{SRP1_JSON}: {n_years_srp1} representative years')
    add(81, 'five', 1, 'value', w(block_len[0]) if len(block_len) == 1 else None, f'{SRP1_JSON}: block length {block_len} years')
    add(81, 'five', 2, 'value', w(len(i33_years)), f'{INSTANCE_3X3}: {len(i33_years)} representative years')
    add(81, 'three', 2, 'value', w(sorted(set(i33_years.values()))[0]) if len(set(i33_years.values())) == 1 else None,
        f'{INSTANCE_3X3}: block length {sorted(set(i33_years.values()))} years')
    # ---- R1.4
    for occ, rate in ((1, 0), (2, 2), (3, 5), (4, 8)):
        tok = str(rate)
        add(97, tok, 1 if rate != 2 else 1, 'value', str(round(disc[rate]['rate'] * 100)) if rate in disc else None,
            f'frozen discount[rate={rate / 100}]')
    for tok, rate in (('-40.8', 0), ('-64.4', 2), ('-92.7', 5), ('-114.6', 8)):
        add(97, tok, 1, 'value', _k1(disc[rate]['value_minus_I']), f'frozen discount[rate={rate / 100}].value_minus_I (k EUR, 1 dp)')
    for occ in (2, 3, 4):
        add(97, '2', occ, 'value', str(round(S.srp1['DiscountFactor'] * 100)), f'{SRP1_JSON} DiscountFactor {S.srp1["DiscountFactor"]} (production rate)')
    # ---- R2.2
    add(125, '0.25', 1, 'value', lat_p, f'{A1_CAMPAIGN} LATTICE_P_STEP (read as text)')
    add(125, '0.5', 1, 'value', lat_e, f'{A1_CAMPAIGN} LATTICE_E_STEP (read as text)')
    add(125, '2', 1, 'value', f'{S.ess["min_energy_to_power_factor"]:g}', f'{ESS_PARAMS} min_energy_to_power_factor')
    add(125, '4', 1, 'value', f'{S.ess["max_energy_to_power_factor"]:g}', f'{ESS_PARAMS} max_energy_to_power_factor')
    add(125, 'One', 1, 'language', None, 'claim (one evaluation per candidate plan; evaluation cache), see Part 2 R22-5')
    add(125, 'one', 2, 'language', None, 'claim (one local problem per network block), see Part 2 R22-6')
    add(125, 'three', 1, 'value', w(n_years_srp1), f'{SRP1_JSON}: {n_years_srp1} representative years')
    add(125, 'four', 1, 'value', w(n_days), f'{SRP1_JSON}: {n_days} representative days')
    add(125, 'four', 2, 'value', w(1 + n_dn), f'{SRP1_JSON}: 1 TransmissionNetwork + {n_dn} DistributionNetworks')
    add(125, '48', 1, 'derived', str(n_years_srp1 * n_days * (1 + n_dn)), '3 x 4 x 4 from SRP1.json')
    add(125, 'one', 3, 'language', None, 'claim (one local problem per shared-ESS agent), see Part 2 R22-6')
    add(125, 'three', 2, 'value', w(len(nodes_i)), f'{SRP1_JSON}: distinct connection_node_id {sorted(nodes_i)}')
    add(125, 'one', 4, 'language', None, 'one agent per interface node (A.1 l. "one model per interface node")')
    # ---- R2.5-R2.6
    add(153, 'three', 1, 'value', w(S.w163['N7']['value']) if S.w163['N7']['status'] == 'match' else None,
        'W163 N7 (settling_criterion_v2: at_least_3_turning_points)')
    add(155, 'one', 1, 'derived', 'one' if S.w163['N10']['value'] >= 1.0 and S.w163['N10']['status'] == 'match' else None,
        'W163 N9/N10: W = max(20, ceil(1.1 P-hat)) >= one period')
    add(157, '42', 1, 'value', str(S.w153c['n_cells']), 'W153 certification totals n_cells')
    add(157, '32', 1, 'value', str(S.w153c['n_certified']), 'W153 certification totals n_certified')
    add(158, '10', 1, 'value', str(S.w153c['n_uncertified']), 'W153 certification totals n_uncertified')
    add(159, '0.9', 1, 'value', f'{S.w163["N25"]["written_v4"]}' if S.w163['N25']['status'] == 'match' else None,
        'W163 N25 (post-certification movement <= 0.9 tau; recomputed from the records there)')
    add(163, 'two', 1, 'language', None, 'A.3 (main.tex) names the storage channel\'s "two-phase schedule" (exempt, then balanced); '
        'a label of the manuscript, checked as claim R25-5')
    starts = sorted(set(re.findall(r'best of (\d+)', S.t6)))
    n_arms = len(re.findall(r'best of \d+', S.t6))
    add(171, 'three', 1, 'value', w(int(starts[0])) if len(starts) == 1 else None, f'{EXPORT_T6}: "best of {starts}" ({n_arms} arms)')
    # ---- R2.11-R2.12
    add(205, 'two', 1, 'language', None, '"couples the two" (network models and agent)')
    # ---- R3.1
    c57 = next(c_ for c_ in S.fz['claims'] if c_['claim_id'] == 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5')
    n57 = sum(1 for v in c57['instance']['c_6597a79d']['candidate_canonical']['nodes'].values() if v[0] > 0 or v[1] > 0)
    add(220, 'two', 1, 'value', w(n57), f'frozen claims C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5 (d_gross {c57["d_gross"] / 1000:.1f}, '
        f'd_net {c57["d_net"] / 1000:.1f} k EUR = §4.1\'s 93.9 -> 84.0): nodes with storage')
    add(221, '60', 1, 'value', str(S.w153s['n_claims']), 'W153 salvage totals n_claims')
    add(221, 'one', 1, 'value', w(S.w153s['n_verdict_differs']), 'W153 salvage totals n_verdict_differs')
    add(221, 'two', 1, 'value', w(f2nodes), 'frozen claims L:y2030__n5_p0.25_e0.5__n7_p1_e3 ref:5ca4f86c nodes with storage')
    f2txt = json.dumps(S.f2)
    add(221, 'doubled', 1, 'value', 'doubled' if re.search(r'"flex_price_multiplier":\s*2\.0\b', f2txt)
        and S.f2claim['instance']['ref:5ca4f86c']['eval_key'] in f2txt else None,
        f'{F2_SPEC}: flex_price_multiplier 2.0 (the spec that holds eval_key 24c5ccb6..., the F2 incumbent)')
    add(221, 'one', 2, 'value', w(len(S.w153s['claims_verdict_differs'])), 'W153 claims_verdict_differs: one claim '
        '(L:y2030__n5_p0.25_e0.5__n7_p1_e3), the incumbent against its neighbour node 7 1/3')
    add(221, '2035', 1, 'value', '2035' if '2035' in yl['per_year'] else None, 'frozen year_ladder.per_year')
    add(221, '2030', 1, 'value', '2030' if '2030' in yl['per_year'] else None, 'frozen year_ladder.per_year')
    add(221, '42.5', 1, 'value', _k1(yl['D_gross']), 'frozen year_ladder.D_gross (k EUR, 1 dp)')
    add(221, '-2.3', 1, 'value', _k1(yl['D_net']), 'frozen year_ladder.D_net (k EUR, 1 dp)')
    add(221, '10', 1, 'value', str(round(float(socmin) * 100)), f'{CODE["defs"]} ENERGY_STORAGE_MIN_ENERGY_STORED {socmin}')
    add(221, '90', 1, 'value', str(round(float(socmax) * 100)), f'{CODE["defs"]} ENERGY_STORAGE_MAX_ENERGY_STORED {socmax}')
    # ---- R3.7-R3.8
    add(272, 'one', 1, 'language', None, 'claim (one discount factor per representative year), see Part 2 R37-1')
    return D


def task_numbers(S, letter_text, ranges):
    toks = reply_tokens(letter_text, ranges)
    D = number_declarations(S)
    rows, undeclared = [], []
    for t in toks:
        d = D.get((t['line'], t['token'].lower(), t['occ']))
        row = dict(t)
        if d is None:
            row['status'] = 'UNDECLARED'
            undeclared.append(row)
            rows.append(row)
            continue
        row.update({k: d[k] for k in ('kind', 'expected', 'source', 'note')})
        if d['kind'] == 'language':
            row['status'] = 'language (not a quantity)'
        else:
            norm = lambda s: str(s).lower().replace('\u2212', '-')  # noqa: E731
            ok = d['expected'] is not None and norm(d['expected']) == norm(t['token'])
            row['status'] = ('match' if d['kind'] == 'value' else 'derived, holds') if ok else 'MISMATCH'
        rows.append(row)
    used = {(t['line'], t['token'].lower(), t['occ']) for t in toks}
    stale = [k for k in D if k not in used]
    if undeclared:
        _fail(f'Part 1: {len(undeclared)} undeclared token(s): {[(u["line"], u["token"], u["occ"]) for u in undeclared]}')
    if stale:
        _fail(f'Part 1: {len(stale)} declaration(s) match no token: {stale}')
    counts = {}
    for r_ in rows:
        counts[r_['status']] = counts.get(r_['status'], 0) + 1
    return {'n_tokens': len(rows), 'counts': counts, 'rows': rows,
            'mismatch': [r_ for r_ in rows if r_['status'] == 'MISMATCH']}


# ======================================================================================================================
#  5. Part 2 -- claims of the changed replies
# ======================================================================================================================
def heading_spans(adj, n_lines):
    """section-number -> (first line, last line) of its region, from the static numbering's headings."""
    lev = {'section': 1, 'subsection': 2, 'subsubsection': 3}
    hs = [o for o in adj['objects'] if o['kind'] in lev]
    spans = {}
    for i, o in enumerate(hs):
        end = next((h['line'] - 1 for h in hs[i + 1:] if lev[h['kind']] <= lev[o['kind']]), n_lines)
        if o['number'] is not None:
            spans[o['number']] = (o['line'], end)
    return spans


def region_text(cl, adj, spec):
    """spec: None (whole), ('sec', '4.6'), ('label', 'eq:...'), ('env', 'abstract'), ('lines', a, b)."""
    if spec is None:
        return 1, len(cl)
    if spec[0] == 'sec':
        return heading_spans(adj, len(cl))[spec[1]]
    if spec[0] == 'lines':
        return spec[1], spec[2]
    if spec[0] == 'env':
        a = next(i for i, ln in enumerate(cl, 1) if f'\\begin{{{spec[1]}}}' in ln)
        b = next(i for i, ln in enumerate(cl, 1) if i > a and f'\\end{{{spec[1]}}}' in ln)
        return a, b
    if spec[0] == 'label':
        idx = next(i for i, ln in enumerate(cl, 1) if f'\\label{{{spec[1]}}}' in ln)
        a = max(i for i in range(1, idx + 1) if re.match(r'\s*\\begin\{(equation|table|figure|algorithm)', cl[i - 1]))
        b = next(i for i in range(idx, len(cl) + 1) if re.match(r'\s*\\end\{(equation|table|figure|algorithm)', cl[i - 1]))
        return a, b
    raise ValueError(spec)


def find_in(cl, a, b, frag):
    return [i for i in range(a, b + 1) if frag in cl[i - 1]]


def ev_main(frag, spec=None, present=True):
    return {'type': 'main', 'fragment': frag, 'region': spec, 'present': present}


def ev_code(path, frag, min_hits=1):
    return {'type': 'code', 'path': path, 'fragment': frag, 'min_hits': min_hits}


def ev_rec(desc, fn):
    return {'type': 'record', 'desc': desc, 'fn': fn}


def ev_file(path, frag, present=True):
    return {'type': 'file', 'path': path, 'fragment': frag, 'present': present}


def claim_declarations(S):
    """Each claim: id, reply, letter line, fragment (once in the letter), changed (new or edited since e10c104), rating
    (consistent | L | M | H; the Worker's W174b rating), basis, evidence list, prior (W177b-2' row it re-scores)."""
    fz, yl, c = S.fz, S.fz['year_ladder'], S.f2claim
    C = []

    def add(cid, reply, line, frag, changed, rating, basis, evidence, prior=None):
        C.append({'id': cid, 'reply': reply, 'line': line, 'fragment': frag, 'changed_since_e10c104': changed,
                  'rating': rating, 'basis': basis, 'evidence': evidence, 'prior_row': prior})
    # ---------------- opening "What we did not do"
    add('O-1', 'Opening', 58, 'could not be evaluated under the certification rule within the memory available and is stated as future work (Section~4.8)',
        True, 'consistent', '§4.8 now states it (A15): the 5 x 5 set "could not be evaluated within the memory available and is left to future work"; the memory fact is recorded (REVISION_CONTEXT.md, Addendum 44 line). The letter\'s "under the certification rule" is not in §4.8; immaterial.',
        [ev_main('could not be evaluated within the memory available and is left to future work', ('sec', '4.8')),
         ev_file(REVISION_CONTEXT, 'The 5×5 pair does not run on this machine')], 'LM-4')
    add('O-2', 'Opening', 58, 'which we state as an assumption (Section~4.8)', False, 'M',
        'carried LM-3: §4.8 states the inference ("is the search\'s answer under each of them, but this was not run"), not an assumption; the evidence (affinity at the baseline on node-7 ladders) is wider than the claim (unit neighbours not evaluated under other calibrations). Sentence unchanged in letter and manuscript.',
        [ev_main('the plan without storage is the search\'s answer under each of them, but this was not run', ('sec', '4.8')),
         ev_main('assumption', ('sec', '4.8'), present=False)], 'LM-3')
    add('O-3', 'Opening', 58, 'a marginal or Shapley-type allocation is identified as future work', False, 'L',
        'carried LL-1: the Conclusions name "an allocation of the degradation of a shared asset among the operators"; neither "marginal" nor "Shapley" appears in §5.',
        [ev_main('an allocation of the degradation of a shared asset among the operators that use it', ('sec', '5')),
         ev_main('Shapley', None, present=False), ev_main('marginal allocation', None, present=False)], 'LL-1')
    add('O-4', 'Opening', 58, 'under the no-ageing calibration the unit is at break-even within resolution', False, 'consistent',
        'frozen ageing no-ageing row within resolution; Table 11 row "No ageing" 0.32 (w).',
        [ev_rec('frozen ageing: no_ageing verdict within resolution',
                lambda: [(r_['arm'], r_['verdict']) for r_ in fz['ageing']['rows'] if 'no' in r_['arm'].lower() and 'age' in r_['arm'].lower()]),
         ev_main('No ageing$^{w}$ & 313.8 & $-$4.1 & 12.9 & 0.32', ('label', 'tab:res_ageing'))])
    add('O-5', 'Opening', 58, 'the multi-scenario results are given on a $3\\times3$ scenario instance (Section~4.5)', False, 'consistent',
        '§4.5 opens with the 3 x 3 instance.',
        [ev_main('combines three market scenarios with three load and renewable-generation scenarios', ('sec', '4.5'))])
    # ---------------- R1.2
    add('R12-1', 'R1.2', 75, 'not about the cost of capital; the cost of capital is treated separately through the discount-rate sensitivity requested in R1.4', False, 'consistent',
        '§3.1 trajectories of storage cost; §4.6 discount sensitivity.',
        [ev_main('three investment trajectories -- low, medium, and high', ('sec', '3.1')), ev_main('robust to the discount rate', ('sec', '4.6'))])
    add('R12-2', 'R1.2', 75, 'Operational uncertainty is represented through scenarios of load, RES generation and market prices (Section~3), and',
        False, 'consistent', 'the abstract says both: investment-cost uncertainty and load, renewable-generation and market-price uncertainty; §3 states the scenario structure.',
        [ev_main('under investment-cost uncertainty', ('env', 'abstract')),
         ev_main('under load, renewable-generation, and market-price uncertainty', ('env', 'abstract')),
         ev_main('three market scenarios and three operation scenarios', ('sec', '3'))])
    add('R12-3', 'R1.2', 77, 'the reference technology is utility-scale lithium-ion battery storage; the cycling calibrations are datasheet readings (Section~3.5) and the abstract names the chemistry',
        False, 'consistent', 'LM-5 resolved: the abstract now names lithium-ion (A1); §3.5 names the technology and the datasheet calibration.',
        [ev_main('with utility-scale lithium-ion storage', ('env', 'abstract')),
         ev_main('The shared ESS is a utility-scale lithium-ion battery', ('sec', '3.5')),
         ev_main('Ageing is calibrated from datasheet cycle-life statements', ('sec', '3.5'))], 'LM-5')
    add('R12-4', 'R1.2', 78, 'the reference unit again loses determinately to the plan without storage', True, 'consistent',
        'L2 applied (Addendum 77 ruling 2); LL-2 resolved. Record: 3 x 3 value - I = -81.3 k EUR at 4.52 x its recorded resolution (production exit, not the §2.2.7 rule); §4.5 and Table 13 say so ("determinate", "stopped by the production exit").',
        [ev_rec('frozen three_by_three rows[2] value - I and rows[4] multiple',
                lambda: (round(fz['three_by_three']['rows'][2]['value'], 2), round(fz['three_by_three']['rows'][4]['value'], 3),
                         fz['three_by_three']['rows'][2]['label'])),
         ev_main('The planning search was not repeated at this instance; the reference unit was evaluated against the plan without storage.', ('sec', '4.5')),
         ev_main('$-$81.3~k\\euro & determinate', ('label', 'tab:res_multiscenario')),
         ev_main('These evaluations were stopped by the production exit', ('sec', '4.5'))], 'LL-2')
    add('R12-5', 'R1.2', 78, 'its value is 0.909--0.934 of its single-scenario value, against 0.933 predicted from the mean-profile price spread (Section~4.5)',
        False, 'consistent', 'numbers in Part 1; §4.5 states the range and the prediction.',
        [ev_main('from 0.934 to 0.909', ('sec', '4.5')), ev_main('The ratio of 0.9331 estimated before the run', ('sec', '4.5'))])
    add('R12-6', 'R1.2', 79, 'limits the generality of the conclusions and we say so (Section~4.8), and the abstract now states it', False, 'consistent',
        '§4.8 and the abstract (108-bus).',
        [ev_main('limits the generality of the conclusions', ('sec', '4.8')), ev_main('integrated 108-bus transmission--distribution test system', ('env', 'abstract'))])
    add('R12-7', 'R1.2', 80, 'the abstract reports instead the value of coordination, 13.9\\,\\% of the discounted operating cost over the 15-year horizon against the best static no-reverse-flow arrangement',
        False, 'consistent', 'the abstract says "13.9% relative to the best evaluated interface arrangement" (same object as §4.4; wording differs, unchanged since e10c104).',
        [ev_main('reduces discounted operating costs by 13.9\\% relative to the best evaluated interface arrangement', ('env', 'abstract'))])
    add('R12-8', 'R1.2', 81, 'the objective convention (gross operational cost, settlement excluded)', False, 'consistent',
        '§4 opening states the convention and the horizon; captions carry it.',
        [ev_main('The objective is the gross operational cost $Q$', ('sec', '4')),
         ev_main('each standing for a five-year block', ('sec', '3')), ev_main('each standing for a three-year block', ('sec', '3'))])
    add('R12-9', 'R1.2', 82, 'this is now stated explicitly (Section~2.3.4, last paragraph, and Section~4.8), and a marginal allocation is listed as future work (Section~5)',
        False, 'L', 'attribution statements present (§2.3.4 last paragraph, §4.8); carried LL-1 on "marginal allocation" (§5 says "an allocation ... among the operators").',
        [ev_main('cannot be uniquely attributed to either the TSO or the DSOs', ('sec', '2.3.4')),
         ev_main('is not unique under a jointly determined schedule and is not attempted', ('sec', '4.8')),
         ev_main('marginal allocation', ('sec', '5'), present=False)], 'LL-1')
    # ---------------- R1.4
    add('R14-1', 'R1.4', 97, 'and which are outside its scope: policy changes, infrastructure retirements and the cost of capital as a stochastic quantity, left to future work',
        False, 'consistent', 'LM-6 resolved: A3 is in the introduction (l. 381).',
        [ev_main('Demand growth and RES capacity follow predefined trajectories, while policy changes, infrastructure retirements and the cost of capital as a stochastic quantity are outside the present formulation and left to future work.', ('sec', '1'))], 'LM-6')
    add('R14-2', 'R1.4', 97, 'fuel and market prices, load and RES generation, as operational scenarios', False, 'L',
        'the introduction says "operational scenarios of load, RES generation and market prices"; fuel (the reviewer\'s "Fuel Costs") is not named. Wording.',
        [ev_main('operational scenarios of load, RES generation and market prices', ('sec', '1')), ev_main('fuel', ('sec', '1'), present=False)])
    add('R14-3', 'R1.4', 97, 'negative and determinate at every rate (Table~15)', False, 'consistent', 'frozen discount verdicts all determinate.',
        [ev_rec('frozen discount verdicts', lambda: [(r_['rate'], r_['verdict']) for r_ in fz['discount']])])
    add('R14-4', 'R1.4', 97, 'The rates other than 2\\,\\% re-weight the per-year operating costs of the 2\\,\\% evaluations rather than re-solving them',
        True, 'consistent', 'L3 applied (ruling 4). W153 discount row: per-year costs of the two settled 2 % evaluations, re-weighted (sum of weighted blocks = recorded Q to 1.2e-7 EUR); §4.6 says so.',
        [ev_rec('W153 discount row: per-year weighted vs undiscounted, reconstruction',
                lambda: {k: S.w153d['Q_per_year'][k]['abs_sum_minus_recorded'] for k in ('x0', 'unit')}),
         ev_main('The rates other than 2\\,\\% re-weight the per-year costs of the 2\\,\\% evaluations rather than re-solving them', ('sec', '4.6'))], 'LM-7')
    add('R14-5', 'R1.4', 97, 'this is exact at the plan without storage and first-order at the unit plan, where the ageing chain couples the years',
        True, 'consistent', 'ruling 4 wording; at x = 0 the blocks are decoupled (no storage), at the unit plan the SoH chain links years (shared_energy_storage_data.py SoH row); magnitude of the first-order error unmeasured.',
        [ev_main('this is exact at the plan without storage and first-order at the reference unit, whose ageing couples the years', ('sec', '4.6')),
         ev_code(CODE['sesd'], "== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))")], 'LM-7')
    add('R14-6', 'R1.4', 97, 'and the rate enters the coordination weights, and the paper says so (Section~4.6)', True, 'L',
        'true of the code (block weight Y_y D_d (1+r)^-(y-y0)) and printed in A.1, but §4.6 does not say it; the letter attributes it to §4.6.',
        [ev_code(CODE['srp'], 'annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - int(years[0])))'),
         ev_main('w_{y,d} = Y_y D_d (1+r)^{-(y-y_0)}', ('sec', 'A.1')),
         ev_main('coordination weight', ('sec', '4.6'), present=False), ev_main('block weight', ('sec', '4.6'), present=False)])
    # ---------------- R2.2
    add('R22-1', 'R2.2', 125, 'the lattice (0.25~MVA and 0.5~MWh units, duration between 2 and 4~h, investment year)', False, 'consistent',
        '§2.1 states the lattice in Delta^S, Delta^E and one investment year; the values are printed in §3.5 (l. "Capacities are sized on the lattice").',
        [ev_main('in multiples of $\\Delta^S$ and $\\Delta^E$, together with a single investment year', ('sec', '2.1')),
         ev_main('$\\Delta^S = 0.25$~MVA, $\\Delta^E = 0.5$~MWh with durations between 2 and 4~h', ('sec', '3.5'))])
    add('R22-2', 'R2.2', 125, 'the poll and frame-size rules of the mesh adaptive direct search, the completion rule and the evaluation cache', False, 'consistent',
        'Algorithm 1: OrthoMADS polls, Delta doubling/halving, completion cap, cache C.',
        [ev_main('OrthoMADS', ('label', 'alg:shared_ess_planning_mads')), ev_main('$\\Delta \\gets 2\\Delta$', ('label', 'alg:shared_ess_planning_mads')),
         ev_main('exceeds the completion cap', ('label', 'alg:shared_ess_planning_mads')), ev_main('cache $\\mathcal{C}$', ('label', 'alg:shared_ess_planning_mads'))])
    add('R22-3', 'R2.2', 125, 'Eq.~(13) gives how each recourse evaluation aggregates scenarios, representative days and years with discounting', True, 'consistent',
        'eq:recourse_value (static 13): Y_y (1+r)^-(y-y0), D_d, omega_m omega_o; code block weight.',
        [ev_main('\\frac{Y_y}{(1+r)^{y-y_0}}', ('label', 'eq:recourse_value')), ev_main('\\omega_m \\, \\omega_o', ('label', 'eq:recourse_value')),
         ev_code(CODE['srp'], 'return float(network_data.years[year]) * float(network_data.days[day]) * annualization')])
    add('R22-4', 'R2.2', 125, 'with the scenarios inside each block) and one per shared-ESS agent (three, one per interface node, each spanning all years and days)', False, 'consistent',
        'A.1: one model per block holding every scenario; one agent model per interface node spanning all years, days and hours.',
        [ev_main('every scenario of the block lives inside the same model', ('sec', 'A.1')),
         ev_main('The shared-ESS agent holds one model per interface node $e \\in E^S$ spanning all years, days and hours', ('sec', 'A.1'))])
    add('R22-5', 'R2.2', 125, 'One recourse evaluation is made per candidate plan', False, 'consistent', 'Algorithm 1 evaluates each NEW poll point and records it in the cache.',
        [ev_main('Evaluate each new point of $\\mathcal{P}$', ('label', 'alg:shared_ess_planning_mads'))])
    add('R22-6', 'R2.2', 125, 'or, for the evaluations behind the reported comparisons, when the certification rule of Section~2.2.7 (stated in full in Appendix~A.5) holds or its cycle cap is reached (Section~3.6)',
        False, 'L', 'carried LL-4: true of the single-scenario comparisons; the 3 x 3 pair behind §4.5 stopped at the production exit (§4.5 now says so).',
        [ev_main('These evaluations were stopped by the production exit', ('sec', '4.5'))], 'LL-4')
    add('R22-7', 'R2.2', 125, 'The number of candidates evaluated and of cycles per evaluation is reported (Section~4.7)', False, 'L',
        'carried LL-5: §4.7 reports 41 search evaluations and cycle statistics of the 42 rule evaluations; cycles per search evaluation are not reported.',
        [ev_main('made 41 new recourse evaluations in all', ('sec', '4.7')), ev_main('over 8{,}570 ADMM cycles', ('sec', '4.7'))], 'LL-5')
    # ---------------- R2.5-R2.6
    add('R25-1', 'R2.5-R2.6', 152, 'now say so (Section~2.2.7)', True, 'consistent',
        'LM-8 resolved: §2.2.7 opens its stopping paragraph with "No convergence guarantee is available for consensus ADMM on a nonconvex recourse of this kind" (A4).',
        [ev_main('No convergence guarantee is available for consensus ADMM on a nonconvex recourse of this kind', ('sec', '2.2.7'))], 'LM-8')
    add('R25-2', 'R2.5-R2.6', 153, 'three\nmeasured turning points', False, 'L',
        'carried LL-6: the summary is the oscillatory branch only; the monotone branch (8 of 32 certificates) is not mentioned.',
        [ev_main('certified instead on a monotone branch', ('sec', '2.2.7')), ev_main('8 on the monotone branch', ('sec', '4.7'))], 'LL-6')
    add('R25-3', 'R2.5-R2.6', 156, 'the residual-balancing penalty update, frozen once the residuals first pass', False, 'consistent', 'A.3 freeze from the cycle after k0.',
        [ev_main('all $\\rho_g$ are frozen from the cycle after the first residual pass $k_0$', ('sec', 'A.3'))])
    add('R25-4', 'R2.5-R2.6', 159, 'bitwise replays of every gated\nevaluation (Section~4.7 and Appendix~E)', False, 'M',
        'carried LM-9 (until the Appendix E rebuild): §4.7 states the replays; Appendix E still holds the submitted per-block results table (no bands, gaps, slacks or replays).',
        [ev_main('replayed that record bitwise up to its first residual pass', ('sec', '4.7')),
         ev_main('Operational planning results by representative year and representative day', ('sec', 'E')),
         ev_main('settling slack', ('sec', 'E'), present=False), ev_main('bitwise', ('sec', 'E'), present=False)], 'LM-9')
    add('R25-5', 'R2.5-R2.6', 163, 'the two-phase\nschedule of the storage channel and the freeze rule are stated in Appendix~A.3, and their values in Appendix~A.8', False, 'consistent',
        'A.3 (two-phase schedule, freeze) and A.8 (initial penalties, factors).',
        [ev_main('(its two-phase schedule)', ('sec', 'A.3')), ev_main('initial penalties $\\rho^{V}_0 = 0.0077$', ('sec', 'A.8'))])
    add('R25-6', 'R2.5-R2.6', 164, 'the configuration was\nselected on a reference instance before the campaign and then frozen', False, 'L',
        'carried LL-7: true of the penalty configuration; read broadly not (the searches ran on earlier code states without the tight tail, §3.6; the tail tightening is reported in §4.7).',
        [ev_main('tightening the interior-point tail', ('sec', '4.7'))], 'LL-7')
    add('R25-7', 'R2.5-R2.6', 170, 'and we\nstate this limitation explicitly', False, 'L',
        'carried LL-8: §4.8 does not state that a different penalty schedule could reach a different local solution; nearest is §2.1 "no guarantee that the global minimum has been attained".',
        [ev_main('provides no guarantee that the global minimum has been attained', ('sec', '2.1')),
         ev_main('different local solution', None, present=False)], 'LL-8')
    add('R25-8', 'R2.5-R2.6', 168, 'the three-start bands of the benchmark arms, Table~12', False, 'consistent', 'tab:res_benchmark (static 12): best of three starts and their spread.',
        [ev_main('Each arrangement is the best of three solver starts', ('label', 'tab:res_benchmark'))])
    # ---------------- R2.10
    add('R210-1', 'R2.10', 192, '$\\eta_{\\mathrm{ch}} p_{\\mathrm{ch}} + p_{\\mathrm{dch}}/\\eta_{\\mathrm{dch}}$', False, 'consistent',
        'eq:cell_throughput (31) and the production throughput row agree (eta_ch p_ch + p_dch / eta_dch, day-weighted D_d/365).',
        [ev_main('\\eta^{\\text{Ch}}_e P^{\\text{Ch}}_{e,y^\\text{Inv},y,d,t}', ('label', 'eq:cell_throughput')),
         ev_main('\\frac{P^{\\text{Dch}}_{e,y^\\text{Inv},y,d,t}}{\\eta^{\\text{Dch}}_e}', ('label', 'eq:cell_throughput')),
         ev_code(CODE['sesd'], 'avg_ch_dch += (num_days / 365.00) * (eff_ch * pch * dt + pdch * dt / eff_dch)')])
    add('R210-2', 'R2.10', 193, 'with the converter rating\nentering only through the $P$--$Q$--$S$ capability inequality', False, 'L',
        'true of the degradation chain (29)-(35), which has no rating; but the rating also enters the network storage block through the new eq. (26) P^Ch + P^Dch <= S^Av and the normalisation of eq. (28), and the agent through 0 <= P <= S^Rated,Unit (A.2). Wording, not a different model; eq. (26) was printed after e10c104.',
        [ev_main('\\le S^{\\text{Av}}_{e,y}', ('label', 'eq:storage_active_power_limit')),
         ev_main('the bounds $0 \\le P^{\\text{Ch}}, P^{\\text{Dch}} \\le S^{\\text{Rated,Unit}}$', ('sec', 'A.2'))])
    add('R210-3', 'R2.10', 194, 'reactive provision does not age the cells', False, 'consistent', '§2.3.3 states it; eq. (31) has no Q.',
        [ev_main('reactive-power provision affects converter utilisation but does not directly affect stored energy or cell degradation', ('sec', '2.3.3'))])
    add('R210-4', 'R2.10', 194, 'Equations~(29)--(35) were rewritten accordingly and checked against the implementation', True, 'consistent',
        'the numbers name the degradation chain (static 29-35: net power, allocation, throughput, cycling loss, SoH chain, floor, calibration). Anchors for (31)-(35) located below; (29)-(30) and the full code audit are W178a\'s (not re-done here).',
        [ev_code(CODE['sesd'], 'model.es_D_per_unit[y_inv, y] * (2 * shared_energy_storage.cl_eff * model.es_e_investment_fixed[y_inv])'),
         ev_code(CODE['sesd'], '== 365.00 * num_years * model.es_avg_ch_dch_per_unit[y_inv, y])'),
         ev_code(CODE['sesd'], 'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'),
         ev_file(ESS_PARAMS, 'k = cycles_n * reference_dod_d / (-ln eol_retention_r)')])
    # ---------------- R2.11-R2.12
    add('R211-1', 'R2.11-R2.12', 202, 'The state-of-charge recursion with charging and discharging efficiencies, the energy limits, the daily closure condition and the $P$--$Q$--$S$ capability inequality are now printed',
        False, 'consistent', 'eqs. (21), (22), (24)-(25), (27) at static numbers.',
        [ev_main('\\eta^{\\text{Ch}}_e P^{\\text{Ch}}_{e,y,d,t} - \\frac{P^{\\text{Dch}}_{e,y,d,t}}{\\eta^{\\text{Dch}}_e}', ('label', 'eq:soc_recursion')),
         ev_main('\\le SoC^{\\text{Max}}_e E^{\\text{Av}}_{e,y}', ('label', 'eq:soc_limits')),
         ev_main('+ s^+_{e,y,d} - s^-_{e,y,d}', ('label', 'eq:soc_closure')),
         ev_main('\\le \\left(S^{\\text{Av}}_{e,y}\\right)^2', ('label', 'eq:converter_capability'))])
    add('R211-2', 'R2.11-R2.12', 202, 'charging and discharging are separate non-negative variables whose difference is the active-power exchange, reactive power is a separate converter variable', False, 'consistent',
        '§2.3.3 first paragraph and eq. (20).',
        [ev_main('charging and discharging are represented by non-negative active-power variables', ('sec', '2.3.3')),
         ev_main('Reactive power is represented separately by $Q^{\\text{E}}_{e,y,d,t}$', ('sec', '2.3.3'))])
    add('R211-3', 'R2.11-R2.12', 203, 'sit in the TSO and DSO network models, once per representative day and common to the scenarios of that day', False, 'consistent',
        '§2.3.3 ("In each network model"; common to all scenarios within each year-day block; no energy carried between days).',
        [ev_main('In each network model containing shared ESS $e$', ('sec', '2.3.3')),
         ev_main('the shared-ESS schedule is common to all market and operation scenarios within each representative year--day block', ('sec', '2.3.3'))])
    add('R211-4', 'R2.11-R2.12', 203, 'and the capability inequality is enforced in the shared-ESS agent as well', False, 'consistent', '§2.3.4 and A.2 constraint list.',
        [ev_main('it enforces the converter capability inequality on its aggregate schedule', ('sec', '2.3.4')),
         ev_main('the converter capability inequality $(P^{\\text{Net}}_{e,y,d,t})^2 + (Q^{\\text{Net}}_{e,y,d,t})^2 \\le (S^{\\text{Av}}_{e,y})^2$', ('sec', 'A.2'))])
    add('R211-5', 'R2.11-R2.12', 204, 'the shared-ESS agent holds the degradation chain, the available-energy update across representative years and the aggregate cell-side throughput', False, 'consistent',
        '§2.3.2 (agent publishes S^Av, E^Av) and §2.3.4 (throughput, chain).',
        [ev_main('The storage agent publishes $S^{\\text{Av}}_{e,y}$ and $E^{\\text{Av}}_{e,y}$', ('sec', '2.3.2')),
         ev_main('the average daily cell-side throughput of a cohort', ('sec', '2.3.4'))])
    add('R211-6', 'R2.11-R2.12', 205, 'the consensus channel on the storage schedule couples the two (Section~2.3 and Appendix~A)', False, 'consistent', '§2.3.1.',
        [ev_main('is a consensus variable between the operators\' network models and the shared-ESS agent', ('sec', '2.3.1'))])
    add('R211-7', 'R2.11-R2.12', 206, 'reactive power enters only through the capability inequality', False, 'consistent',
        'in the storage model Q^E appears only in eq. (27) (and the agent\'s circle in A.2); the consensus channel also carries Q (A.1), which is coordination, not the storage physics.',
        [ev_main('Net active power and reactive power share the converter\'s apparent-power capability', ('sec', '2.3.3'))])
    add('R211-8', 'R2.11-R2.12', 210, 'Section~2.3.3, Eqs.~(20)--(28)', True, 'consistent', 'eqs. 20-28 are exactly the nine equations of §2.3.3 (static and the expert\'s compile).',
        [ev_rec('static numbers of the §2.3.3 labels', lambda: 'see static_vs_compile row "SOC block (20)-(28)"')])
    # ---------------- R3.1
    add('R31-1', 'R3.1', 220, 'The operational recourse does not internalize terminal value, and we state this', False, 'consistent', '§4 opening.',
        [ev_main('The credit is reported, not optimized: the recourse does not see it', ('sec', '4'))])
    add('R31-2', 'R3.1', 220, 'Salvage is computed as a post-evaluated credit from terminal state of health and remaining calendar life (Eqs.~(38)--(39))', True, 'consistent',
        'LL-9 resolved: eq:salvage_credit (38) now prints the coded form (energy cost only, health normalised above SoH^Min, remaining-life fraction), matching the production settings; the recorded credits 11.76 / 56.50 k EUR are Table 14\'s 11.8 / 56.5.',
        [ev_main('SoH^{\\text{Min}}_e}{1 - SoH^{\\text{Min}}_e}', ('label', 'eq:salvage_credit')),
         ev_main('\\bar{c}^{\\text{Inv,E}}_{y^\\text{Inv}}\\, E^{\\text{Inv}}_{e,y^\\text{Inv}}', ('label', 'eq:salvage_credit')),
         ev_file(ESS_PARAMS, '"cost_basis": "EXPECTED_INSTALLATION_ENERGY_COST"'),
         ev_file(ESS_PARAMS, '"health_basis": "NORMALIZED_ABOVE_MINIMUM_SOH"'),
         ev_rec('frozen year_ladder salvage_last 2030 / 2035 (k EUR)',
                lambda: (_k1(yl['per_year']['2030']['salvage_last']), _k1(yl['per_year']['2035']['salvage_last']))),
         ev_main('2030             & 68.5  & 11.8 & 56.7', ('label', 'tab:res_year_ladder'))], 'LL-9')
    add('R31-3', 'R3.1', 220, 'the investment-year comparison (Table~14), the two-node plan of Section~4.1, and the full claims table of Appendix~E', False, 'M',
        'Table 14 and §4.1 carry the net figures; carried LM-9: Appendix E holds no claims table yet (the submitted per-block table), pending the expert\'s composition (ruling 3).',
        [ev_main('Applying the terminal salvage credit reduces the latter to 84.0~k\\euro{}', ('sec', '4.1')),
         ev_main('salvage', ('sec', 'E'), present=False), ev_main('claims', ('sec', 'E'), present=False)], 'LM-9')
    add('R31-4', 'R3.1', 221, 'Among the 60 comparisons of the full claims table it changes no sign and one verdict', True, 'consistent',
        'L1 applied (ruling 1); LH-1 resolved. W153: 60 claims, 0 sign changes, 1 verdict change; §4.6 says the same (A13).',
        [ev_rec('W153 salvage totals', lambda: (S.w153s['n_claims'], S.w153s['n_sign_differs'], S.w153s['n_verdict_differs'], S.w153s['claims_verdict_differs'])),
         ev_main('applying the salvage credit changes no sign and one verdict', ('sec', '4.6'))], 'LH-1')
    add('R31-5', 'R3.1', 221, 'a two-node plan evaluated under the doubled flexibility price becomes determinate in its favour against one of its unit neighbours', True, 'consistent',
        'the F2 incumbent (2030: node 5 0.25/0.5, node 7 1/3.5; flex x2 spec) against its neighbour node 7 1/3: d = F(neighbour) - F(incumbent) = +26.4 gross (0.95 x, within the uncertified bar) -> +34.1 net (1.22 x, determinate), positive = neighbour worse, so in the incumbent\'s favour.',
        [ev_rec('frozen claim L:y2030__n5_p0.25_e0.5__n7_p1_e3',
                lambda: {'d_gross': round(c['d_gross'], 2), 'd_net': round(c['d_net'], 2), 'gross': [round(c['gross_multiple'], 3), c['gross_verdict']],
                         'net': [round(c['net_multiple'], 3), c['net_verdict_W153']],
                         'incumbent': c['instance']['ref:5ca4f86c']['candidate_canonical'],
                         'neighbour': c['instance']['l_0ee93aca']['candidate_canonical']}),
         ev_main('the comparison of the plan returned at twice the base flexibility price with one of its unit neighbours', ('sec', '4.6'))])
    add('R31-6', 'R3.1', 221, 'Separately, the investment-year comparison depends on the convention', True, 'consistent',
        'T4: +42,458.65 gross (3.46 x, determinate) and -2,286.25 net (0.19 x, within); §4.6: the pair is not among the 60.',
        [ev_rec('frozen year_ladder', lambda: (round(yl['D_gross'], 2), yl['gross_v6']['verdict'], round(yl['D_net'], 2), yl['net_v6']['verdict'])),
         ev_main('The investment-year comparison above is not among the 60', ('sec', '4.6'))])
    add('R31-7', 'R3.1', 221, 'The SOC limits are now stated once, in Section~3.5 (10--90\\,\\% of the available energy), and enter the SOC constraints as such (Eq.~(22))', True, 'consistent',
        '§3.5 is the only place the values are printed; eq:soc_limits (22) uses SoC^Min/Max x E^Av; definitions.py 0.10 / 0.90.',
        [ev_main('a usable state-of-charge window of 10--90\\,\\% of the available energy', ('sec', '3.5')),
         ev_main('10--90', ('lines', 1, 1262), present=False), ev_main('SoC^{\\text{Min}} = 0.10', ('lines', 1, 1262), present=False),
         ev_main('10--90', ('lines', 1306, 2511), present=False), ev_main('SoC^{\\text{Min}} = 0.10', ('lines', 1306, 2511), present=False),
         ev_main('SoC^{\\text{Min}}_e E^{\\text{Av}}_{e,y}', ('label', 'eq:soc_limits'))])
    # ---------------- R3.7-R3.8
    add('R37-1', 'R3.7-R3.8', 272, 'one discount factor per representative year applied to', True, 'consistent',
        'eq:recourse_value (13): Y_y/(1+r)^(y-y0), D_d, omega; code block weight; investment discounting in §2.2.1.',
        [ev_main('\\frac{Y_y}{(1+r)^{y-y_0}}', ('label', 'eq:recourse_value')),
         ev_code(CODE['srp'], 'annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - int(years[0])))')])
    add('R37-2', 'R3.7-R3.8', 274, 'discounted from their installation year (Section~2.2.1)', False, 'consistent', '§2.2.1.',
        [ev_main('(1+r)^{y^{\\text{Inv}}-y_0}', ('sec', '2.2.1'))])
    add('R37-3', 'R3.7-R3.8', 274, 'The revised paper reports no annual figures: every result is a present value over the horizon, aggregated as in Eq.~(13)', True, 'M',
        'L4 as applied: contradicted by Appendix E as it stands - the submitted table "Operational planning results by representative year and representative day" (daily costs, generation and curtailment per year/day for 2025-2037, the submitted instance), introduced for "seasonal and inter-annual performance differences". Pending the expert\'s Appendix E composition (ruling 3), then it holds for costs; non-monetary per-year/per-day figures remain (Table 11 EFC/day; §4.3 "+0.18, 0.20 and 0.27 EFC per day in 2025, 2030 and 2035") - L once Appendix E is rebuilt.',
        [ev_main('Operational planning results by representative year and representative day', ('sec', 'E')),
         ev_main('inter-annual performance differences', ('sec', 'E')),
         ev_main('equivalent full cycles per day in 2025, 2030 and 2035', ('sec', '4.3'))], 'LM-10')
    add('R37-4', 'R3.7-R3.8', 274, 'the representative-day weights and the block length of each representative year are stated (Section~3)', True, 'M',
        'block lengths are stated (§3: five-year / three-year blocks); the representative-day weights are not: §3 says "four seasonal representative days" and §2.2.7 defines D_d only symbolically ("the numbers of ... days represented"); the values (92 / 91 / 91 / 91 days, SRP1.json "Days") appear nowhere in main.tex, so a reader cannot reconstruct the weights the letter says are stated.',
        [ev_main('each standing for a five-year block', ('sec', '3')),
         ev_main('In both instances each year is represented by four seasonal representative days', ('sec', '3')),
         ev_main('The weights $Y_y$ and $D_d$ are the numbers of calendar years and days represented', ('sec', '2.2.7')),
         ev_main('92~days', None, present=False), ev_main('91~days', None, present=False), ev_main('92 days', None, present=False),
         ev_main('91 days', None, present=False),
         ev_rec('SRP1.json Days', lambda: S.srp1['Days']),
         ev_rec('main.tex (comment-stripped): every standalone 91 or 92 with context',
                lambda: [(i, ln[max(0, m.start() - 40):m.end() + 40]) for i, ln in enumerate(W167.strip_comments(lines_of(_read(MAIN_TEX))), 1)
                         for m in re.finditer(r'(?<![\d.])9[12](?![\d.%])', ln)])])
    return C


def task_claims(S, letter_text, main_lines, adj, ranges):
    cl = W167.strip_comments(main_lines)
    letter_flat = letter_text
    C = claim_declarations(S)
    rows = []
    for cd in C:
        row = {k: cd[k] for k in ('id', 'reply', 'line', 'fragment', 'changed_since_e10c104', 'rating', 'basis', 'prior_row')}
        n_occ = letter_flat.count(cd['fragment'])
        frag_line = letter_flat[:letter_flat.find(cd['fragment'])].count('\n') + 1 if n_occ else None
        row['letter_fragment_occurrences'] = n_occ
        row['letter_fragment_line'] = frag_line
        if n_occ != 1:
            _fail(f'claim {cd["id"]}: letter fragment found {n_occ} times')
        rg = next((r_ for r_ in ranges if r_['first_line'] <= (frag_line or -1) <= r_['last_line']), None)
        if rg is None or rg['id'] != cd['reply'].replace('Opening', 'Opening: What we did not do'):
            _fail(f'claim {cd["id"]}: fragment at letter l. {frag_line} not inside reply {cd["reply"]}')
        evs = []
        for e in cd['evidence']:
            out = {k: v for k, v in e.items() if k != 'fn'}
            if e['type'] == 'main':
                a, b = region_text(cl, adj, e['region'])
                hits = find_in(cl, a, b, e['fragment'])
                if not hits and '\n' not in e['fragment']:
                    joined = '\n'.join(cl[a - 1:b])
                    if e['fragment'] in joined:
                        hits = [a + joined[:joined.find(e['fragment'])].count('\n')]
                out.update({'region_lines': [a, b], 'hit_lines': hits,
                            'quote': [cl[h - 1].strip()[:400] for h in hits[:2]]})
                ok = bool(hits) == e['present']
            elif e['type'] == 'code':
                src = _read(e['path']).split('\n')
                hits = [i for i, ln in enumerate(src, 1) if e['fragment'] in ln]
                out.update({'hit_lines': hits, 'sha256': _sha(e['path'])})
                ok = len(hits) >= e['min_hits']
            elif e['type'] == 'file':
                txt = _read(e['path'])
                hits = [i for i, ln in enumerate(txt.split('\n'), 1) if e['fragment'] in ln]
                out.update({'hit_lines': hits})
                ok = bool(hits) == e['present']
            else:
                try:
                    out['value'] = e['fn']()
                    ok = True
                except Exception as exc:  # noqa: BLE001
                    out['value'] = f'ERROR {exc!r}'
                    ok = False
            out['expectation_met'] = ok
            if not ok:
                _fail(f'claim {cd["id"]}: evidence expectation not met: {out}')
            evs.append(out)
        row['evidence'] = evs
        rows.append(row)
    counts = {}
    for r_ in rows:
        counts[r_['rating']] = counts.get(r_['rating'], 0) + 1
    by_reply = {}
    for r_ in rows:
        by_reply.setdefault(r_['reply'], {}).setdefault(r_['rating'], 0)
        by_reply[r_['reply']][r_['rating']] += 1
    return {'n_claims': len(rows), 'counts': counts, 'by_reply': by_reply, 'rows': rows,
            'rating_scale': 'W174b: H - a referee re-deriving from the text gets a different model, number or verdict; '
                            'M - imprecise, not a different result; L - wording'}


def task_l1_l4(letter_text):
    fixes = _read(FIXES_TEX)
    brief = _read(BRIEF)
    a77 = brief[brief.find('# Addendum 77'):]
    rulings = {n: (re.search(r'^' + str(n) + r'\. \*\*[^\n]*', a77, re.M).group(0)[:300] if re.search(r'^' + str(n) + r'\. \*\*', a77, re.M) else None)
               for n in (1, 2, 3, 4)}
    in_fixes = {k: bool(re.search(r'\b' + k + r'\b', fixes)) for k in ('L1', 'L2', 'L3', 'L4')}
    rows = [
        {'id': 'L1', 'ruling': 1, 'reply': 'R3.1',
         'addendum_77': 'R3.1 now say: no sign and one verdict among the 60 (the F2 neighbour, 0.95x -> 1.22x); the 2035 - 2030 pair is outside the 60 and changes verdict (and sign) net of salvage',
         'letter_v4_fragments': ['Among the 60 comparisons of the full claims table it changes no sign and one verdict',
                                 'a two-node plan evaluated under the doubled flexibility price becomes determinate in its favour against one of its unit neighbours',
                                 'Separately, the investment-year comparison depends on the convention',
                                 '($-2.3$~k\\euro{}, within resolution)'],
         'note': 'applied in substance: "Separately" places the year pair outside the 60; its sign change is visible in the numbers (+42.5 / -2.3) but not stated in words; the 0.95x / 1.22x multiples are not quoted (not required by the ruling).'},
        {'id': 'L2', 'ruling': 2, 'reply': 'R1.2',
         'addendum_77': 'R1.2 says the reference unit again loses',
         'letter_v4_fragments': ['the reference unit again loses determinately to the plan without storage'], 'note': 'applied.'},
        {'id': 'L3', 'ruling': 4, 'reply': 'R1.4',
         'addendum_77': 'Letter and §4.6: the 0/5/8 % rows re-weight the 2 % evaluations; exact at x = 0, first-order at the unit plan',
         'letter_v4_fragments': ['The rates other than 2\\,\\% re-weight the per-year operating costs of the 2\\,\\% evaluations rather than re-solving them',
                                 'this is exact at the plan without storage and first-order at the unit plan'],
         'note': 'applied; the letter adds "and the rate enters the coordination weights, and the paper says so (Section~4.6)", a clause §4.6 does not carry (claim R14-6, L).'},
        {'id': 'L4', 'ruling': 3, 'reply': 'R3.7-R3.8',
         'addendum_77': 'R3.7/3.8 is trimmed instead: the paper has no annual figures',
         'letter_v4_fragments': ['The revised paper reports no annual figures'],
         'note': 'applied as a replacement sentence (not a pure trim): it adds "every result is a present value ..." and "the representative-day weights ... are stated (Section~3)" (claims R37-3 M, R37-4 M).'},
    ]
    for r_ in rows:
        r_['fragments_found'] = {f: letter_text.count(f) for f in r_['letter_v4_fragments']}
        r_['applied'] = all(v == 1 for v in r_['fragments_found'].values())
        if not r_['applied']:
            _fail(f'{r_["id"]}: letter fragments not all found once: {r_["fragments_found"]}')
    for r_ in rows:
        r_['addendum_77_ruling_text'] = rulings[r_['ruling']]
        if rulings[r_['ruling']] is None:
            _fail(f'Addendum 77 ruling {r_["ruling"]} not located in the brief')
    return {'labels_present_in_w177b_fixes_tex': in_fixes,
            'note': 'manuscript_review/w177b_fixes.tex holds A1-A20 and B1-B4 only; no L1-L4 text. The rows are checked against '
                    'the letter-edit content of Addendum 77 rulings 1-4 (L-labels as cited there).',
            'rows': rows}


PRIOR_LETTER_ROWS = {   # W177b-2' rows (P5_15_W177B_PART2_E10C104.md, 559ef967) in the changed replies -> claim id
    'LH-1': ('H', 'R31-4'), 'LM-3': ('M', 'O-2'), 'LM-4': ('M', 'O-1'), 'LM-5': ('M', 'R12-3'), 'LM-6': ('M', 'R14-1'),
    'LM-7': ('M (H if read as a model statement)', 'R14-4'), 'LM-8': ('M', 'R25-1'), 'LM-9': ('M', 'R25-4'),
    'LM-10': ('M', 'R37-3'), 'LL-1': ('L', 'O-3'), 'LL-2': ('L', 'R12-4'), 'LL-4': ('L', 'R22-6'), 'LL-5': ('L', 'R22-7'),
    'LL-6': ('L', 'R25-2'), 'LL-7': ('L', 'R25-6'), 'LL-8': ('L', 'R25-7'), 'LL-9': ('L', 'R31-2')}
PRIOR_STATUS_NOTE = {
    'LM-7': 'resolved ("exact at a fixed plan" replaced by L3); the added clause "the rate enters the coordination weights, '
            'and the paper says so (Section 4.6)" is a new L (R14-6)',
    'LM-10': 'replaced: the rename claim is gone (L4); the new sentence is M for a different reason (Appendix E still holds '
             'the submitted per-year/per-day table) - R37-3, plus R37-4 (day weights not stated), M',
    'LM-9': 'unchanged (Appendix E not yet composed from the export; ruling 3)'}
PRIOR_NOT_IN_SCOPE = {'LM-1': 'l. 43 / 137 / 229: opening "Why", R2.3, R3.2 - replies unchanged',
                      'LM-2': 'l. 48: opening "Why" - unchanged', 'LL-3': 'l. 102: R1.5 - unchanged',
                      'LL-10': 'l. 233: R3.2 - unchanged'}


def task_prior_rows(claims):
    by_id = {r_['id']: r_ for r_ in claims['rows']}
    rows = []
    for pid, (prior, cid) in PRIOR_LETTER_ROWS.items():
        now = by_id[cid]['rating']
        status = 'resolved' if now == 'consistent' else ('unchanged' if now == prior.split()[0] else f'{prior.split()[0]} -> {now}')
        if pid in PRIOR_STATUS_NOTE:
            status = PRIOR_STATUS_NOTE[pid]
        rows.append({'prior_row': pid, 'prior_rating': prior, 'claim_now': cid, 'rating_now': now, 'status': status})
    return {'rows': rows, 'not_rescored_unchanged_replies': PRIOR_NOT_IN_SCOPE}


# ======================================================================================================================
#  guards, output
# ======================================================================================================================
def verify_guards():
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    g = {'own': GUARD.verify(0), 'w177b3r': R.GUARD.verify(0), 'w177b3': P3.GUARD.verify(0), 'w167': W167.GUARD.verify(0),
         'phase_b_record': PB.PARENT_GUARD.verify(0)}
    out = {'verify_0_failures': g,
           'counts': {'own': GUARD.counts, 'w177b3r': R.GUARD.counts, 'w177b3': P3.GUARD.counts, 'w167': W167.GUARD.counts,
                      'phase_b_record': PB.PARENT_GUARD.counts},
           'pickle_counts': dict(PICKLE_COUNTS), 'w177b3r_pickle_counts': dict(R.PICKLE_COUNTS),
           'w177b3_pickle_counts': dict(P3.PICKLE_COUNTS), 'w167_pickle_counts': dict(W167.PICKLE_COUNTS),
           'production_modules_imported': forbidden}
    if any(g.values()) or any(PICKLE_COUNTS.values()) or any(R.PICKLE_COUNTS.values()) or any(P3.PICKLE_COUNTS.values()) \
            or any(W167.PICKLE_COUNTS.values()) or forbidden:
        _fail(f'guards: {out}')
    out['ok'] = not (any(g.values()) or forbidden)
    return out


def write_md(out, path):
    P = out['pointers']
    L = [f'# W178c letter v4 checks (generated by {SCRIPT_NAME})', '',
         f'Clone HEAD {out["identity"]["clone_head"]}; main.tex {out["identity"]["main_sha256"][:8]}; letter {out["identity"]["letter_sha256"][:8]}. '
         f'Check failures: {len(out["check_failures"])}.', '',
         '## Pointers', '', f'{P["n_pointers"]} pointers: {P["counts"]}', '',
         '| letter l. | text | kind | target | 9fd19e7 | unadjusted | e10c104 | verdict |', '|---|---|---|---|---|---|---|---|']
    for r_ in P['pointers']:
        tgt = r_['target'] if isinstance(r_['target'], str) else ', '.join(r_['target'] or [])
        L.append(f'| {r_["line"]} | {r_["text"].replace("|", "/")} | {r_["kind"]} | {tgt[:60]} | {r_.get("number_9fd19e7", "")} | '
                 f'{r_.get("number_9fd19e7_unadjusted", "")} | {r_.get("number_e10c104") or ""} | {r_["verdict"]} |')
    L += ['', '## Static vs compile (Addendum 77 ruling 5)', '']
    for r_ in out['static_vs_compile']['rows']:
        L.append(f'- {r_["item"]}: agree {r_["agree"]} ({r_.get("static", r_.get("static_adjusted"))})')
    L += ['', '## Part 1 numbers', '', f'{out["part1_numbers"]["counts"]}', '',
          '| reply | l. | token | status | expected | source |', '|---|---|---|---|---|---|']
    for r_ in out['part1_numbers']['rows']:
        L.append(f'| {r_["reply"]} | {r_["line"]} | {r_["token"]} | {r_["status"]} | {r_.get("expected")} | {str(r_.get("source"))[:110]} |')
    L += ['', '## Part 2 claims', '', f'{out["part2_claims"]["counts"]}', '',
          '| id | l. | changed | rating | prior | basis |', '|---|---|---|---|---|---|']
    for r_ in out['part2_claims']['rows']:
        L.append(f'| {r_["id"]} | {r_["line"]} | {r_["changed_since_e10c104"]} | {r_["rating"]} | {r_["prior_row"] or ""} | {r_["basis"][:200]} |')
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
    _log('W178c letter v4 checks: start')
    out = {'schema': 'p515_s53_w178c_letter_checks', 'task': 'W178c (Addendum 77, W178 item (d)): letter v4 Parts 1-2 on '
           'the changed replies; numbered pointers vs static numbering and the expert\'s compile numbers (ruling 5)',
           'objective_convention': 'Q = certified gross_operational_cost, settlement excluded; net = Q - terminal salvage '
                                   'credit; value = Q(0) - Q(x); F = Q + I',
           'started_utc': t0.isoformat(), 'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'script': {'path': SCRIPT_NAME, 'sha256': _sha(__file__)},
           'imported': {m: _sha(m) for m in (PRIOR_SCRIPT, 'p515_s53_w177b_part3_checks.py', 'p515_s53_w167_nomenclature_years.py',
                                             'p515_s47_phase_b_record.py', 'gate_result_io.py', 'p513_solve_profile_guard.py')}}
    out['capture_checklist'] = capture_checklist()
    _log('capture-path checklist: all present')
    out['identity'] = identity()
    _log(f"identity: head {out['identity']['clone_head']}, checks {sum(out['identity']['checks'].values())}/{len(out['identity']['checks'])}")
    main_lines = lines_of(_read(MAIN_TEX))
    letter_text = _read(LETTER)
    letter_lines = lines_of(letter_text)
    prior_lines = lines_of(show(PRIOR_HEAD))
    raw, adj = numbering_adjusted(main_lines)
    out['numbering_9fd19e7'] = {'adjusted': adj, 'unadjusted_counts': raw['counts']}
    out['static_vs_compile'] = task_compile(raw, adj)
    _log(f"static vs compile: {out['static_vs_compile']['n_agree']}/{out['static_vs_compile']['n_items']} agree")
    out['refs_static'] = task_refs(main_lines, adj)
    _log(f"refs: undefined {out['refs_static']['undefined_refs']}, duplicate labels {out['refs_static']['duplicate_labels']}")
    ld = task_letter_diff(letter_lines)
    letter_map = ld.pop('_map')
    out['letter_diff_e10c104_to_head'] = ld
    _log(f"letter diff: {ld['n_hunks']} hunks; replies touched {ld['replies_touched']}; declared but untouched {ld['declared_replies_untouched']}")
    out['pointers'] = task_pointers(main_lines, raw, adj, prior_lines, letter_map, ld['hunks'])
    _log(f"pointers: {out['pointers']['n_pointers']} {out['pointers']['counts']}")
    S = Src()
    out['part1_numbers'] = task_numbers(S, letter_text, ld['reply_ranges'])
    _log(f"Part 1: {out['part1_numbers']['n_tokens']} tokens {out['part1_numbers']['counts']}")
    out['part2_claims'] = task_claims(S, letter_text, main_lines, adj, ld['reply_ranges'])
    _log(f"Part 2: {out['part2_claims']['n_claims']} claims {out['part2_claims']['counts']}")
    out['l1_l4'] = task_l1_l4(letter_text)
    _log(f"L1-L4 applied: {[r_['applied'] for r_ in out['l1_l4']['rows']]}")
    out['prior_rows_rescored'] = task_prior_rows(out['part2_claims'])
    out['frozen_json_unchanged_at_end'] = _sha(FZ) == FZ_SHA
    if not out['frozen_json_unchanged_at_end']:
        _fail('frozen JSON changed during the run')
    out['guards'] = verify_guards()
    out['prediction'] = {'expert_addendum_77': '(d) 0 pointer MISMATCH',
                         'outcome_n_mismatch': out['pointers']['counts'].get('MISMATCH', 0),
                         'held': out['pointers']['counts'].get('MISMATCH', 0) == 0}
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'w', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, out_md)
    inputs = sorted({MAIN_TEX, LETTER, LETTER_V4, MAIN_FIXED, FIXES_TEX, BRIEF, PRIOR_JSON, PRIOR_SCRIPT, FZ, EXPORT_T6,
                     W153_SALV, W153_CERT, W153_DISC, W163, SRP1_JSON, INSTANCE_3X3, ESS_PARAMS, SUBMITTED, F2_SPEC,
                     A1_CAMPAIGN, REVISION_CONTEXT, os.path.join(CLONE, 'elsarticle.dtx'), 'p515_s53_w177b_part3_checks.py',
                     'p515_s53_w167_nomenclature_years.py', 'p515_s47_phase_b_record.py', 'gate_result_io.py',
                     'p513_solve_profile_guard.py', SCRIPT_NAME} | set(CASES.values()) | set(CODE.values()))
    manifest = {'outputs': {OUT_NAME: _sha(out_json), MD_NAME: _sha(out_md)}, 'out_dir': OUT_DIR,
                'git_show_blobs_sha256': {f'{c}:{p}': hashlib.sha256(show(c, p).encode('utf-8')).hexdigest()
                                          for c in (PRIOR_HEAD, 'HEAD') for p in ('main.tex', 'response_to_reviewers_draft.tex')},
                'inputs': {p: _sha(p) for p in inputs}}
    with open(out_man, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'written {out_json}, {out_md}, {out_man}; check failures: {len(CHECK_FAILURES)}; guards ok: {out["guards"]["ok"]}; '
         f'pickle {PICKLE_COUNTS}; prediction "0 pointer MISMATCH" held: {out["prediction"]["held"]}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
