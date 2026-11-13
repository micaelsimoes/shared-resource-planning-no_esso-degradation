"""P5.15 Addendum 92 (Planner order, W190b line) -- Planner task W190b PART 1: THE INTERIM AUDIT AT THE OVERLEAF HEAD
5f4d3bd -- STATE AND DIFF 3c379fa..HEAD HUNK BY HUNK, (i) THE W187 NUMBERS, (ii) NUMBERS AND THE NEW 4.1 PARAGRAPH'S
FORMATTING, (v) THE APPENDIX B GATE AND verify_appE.py. ZERO SOLVES, NO MODEL, NO PICKLE.
Read-only on the Overleaf clone (git rev-parse / show / diff / log / status only; the fast-forward to 5f4d3bd was done
before this script, by hand), on manuscript_review/, on every committed artifact and on every earlier script. Part 2
(letter pointers, blue, structure, counts) is p515_s55_w190b_structure.py.

REUSE, NOT REIMPLEMENTATION. As W190a: the W189 part-1 harness p515_s55_w189_p1_numbers.py (P1) is IMPORTED, NOT EDITED
and NOT RUN, and so is the W190a part-1 harness p515_s55_w190a_numbers.py (N90; its main() is never called, nothing is
written into its output directory). Their functions are called with their module constants re-bound to this round:
  * P1.task_appendix (full-mode generator run, verify_appE.py with tempfile.mkdtemp() redirected into this stage's output
    directory, the W180a / W181a entry check of the printed tables against the frozen JSON) -- P1.PREV_FULL = a87e195 as
    in W189 / W190a (its integrity check locates the full generator output once in a87e195);
  * N90.task_generator (the --reduced generator run in-process, located bitwise in main.tex) and N90.verifier_summary --
    N90.BASE_FULL re-bound to 3c379fa (so its "9b" occurrence count is the 3c379fa count), N90.OUT_NAMES to W190b names;
  * P1.task_numbers (the W182b method) -- P1.PREV_FULL = 3c379fa, P1.HEAD_FULL = 5f4d3bd, P1.declarations_w189p1 re-bound
    to declarations_w190b below: every '+' line of git diff -U0 3c379fa 5f4d3bd (main.tex, letter, cover letter) carries
    the earlier declaration of its tokens (W182b for the abstract, the contributions and 4.1; W189p1 for the letter; W190a
    for the cover letter), and the W187 tokens get new specs read from the W187 cell records;
  * N90.mark_new (new / carried against the '-' lines of the hunk) -- N90.BASE_FULL / HEAD_FULL re-bound.
NEW HERE (no earlier harness does them): (i) the W187 checks (the 14 Delta F recomputed from the 15 campaign_results.json
files and the investment costs, the printed counts and range, the wording and the not-printed items); (ii) the formatting
scan of the '+' lines (literal euro signs, missing ties, hard-coded references) and the claims of the rewritten 4.1
paragraph; the hunk-by-hunk listing of 3c379fa..HEAD (no review copy exists this round: identity is not a check).

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import; P1's / N90's guards and every guard of
their chain verified at exactly 0 (N90.verify_guards) together with this script's; pickle.load / loads / Unpickler blocked
by P1's blockers (counters verified 0); refuses to run concurrently with another copy of itself; refuses unless the output
directory exists and holds nothing but launch.log; capture-path checklist asserted before any task (exit 4). JSON written
only through gate_result_io. verify_appE.py launches gen_appE.py twice as a subprocess (csv / sys / os / re only).

COMMANDS (repo root, canonical interpreter, attached, both streams captured):
  mkdir -p w190b_audit && mkdir w190b_audit/numbers && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_numbers.py \\
      > w190b_audit/numbers/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u -B p515_s55_w190b_numbers.py --post-run
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W190b part 1 W187 numbers, numbers and Appendix B gate at Overleaf '
                                               '5f4d3bd (never solves)').install()

import collections  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s55_w190a_numbers as N90  # noqa: E402  (W190a part 1; imports W189 part 1; NOT edited, main() not run)
import gate_result_io as GRIO  # noqa: E402

P1 = N90.P1
A, NB, W177A = P1.A, P1.NB, P1.W177A
os.chdir(REPO)
ORIG_DECL_W189 = N90.ORIG_DECL                   # P1.declarations_w189p1 as committed (N90 kept it before re-binding)
DECL_W190A = N90.declarations_w190a

# ======================================================================================================================
#  constants
# ======================================================================================================================
SCRIPT_NAME = os.path.basename(__file__)
STAGE = 'p515_s55_w190b_numbers'
OUT_DIR = os.path.join('w190b_audit', 'numbers')
OUT_NAMES = {'json': 'w190b_numbers.json', 'md': 'w190b_numbers.md', 'man': 'manifest_sha256.json', 'log': 'launch.log',
             'post': 'manifest_post_run_sha256.json', 'gen': 'appE_generated_full_w190b.tex',
             'red': 'appE_generated_reduced_w190b.tex', 'verify': 'verify_appE_output_w190b.txt',
             'verify_tmp': 'verify_appE_tmp'}
CLONE = P1.CLONE
HEAD_FULL = '5f4d3bd4c96641c40a42829ae837fd954ec04954'     # audited HEAD (fetched and fast-forwarded 11:07:34 UTC 11-13)
PREV_FULL = '3c379fa44f3755dbdc6fd540df566b55527d921c'     # round 9c (W190a)
A87_FULL = N90.A87_FULL                                    # round 8
MAIN, LETTER, COVER, HL = P1.MAIN, P1.LETTER, P1.COVER, P1.HL
FILES = (MAIN, LETTER, COVER, HL)
DIFF_FILES = (MAIN, LETTER, COVER)
MR = 'manuscript_review'
GEN_REL, VERIFY_REL, APPX_REL = P1.GEN_REL, P1.VERIFY_REL, P1.APPX_REL
EXPECTED_VERIFY_PASS = 40
W190A_RED = os.path.join('w190a_audit', 'numbers', 'appE_generated_reduced_w190a.tex')
W190A_RED_SHA_PREFIX = '8e61636b'
W190A_RED_LINES = 218
W187_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'e1_screen_1x')
W187_JSON = os.path.join(W187_DIR, 'w187_screen_report.json')
W187_JSON_SHA_PREFIX = '1afc8a78'                          # P5_15_W187_REPORT.md "Changed"
W187_REPORT = 'P5_15_W187_REPORT.md'
W86_RESULTS = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'tight_tail_w86', 'campaign_s53_w86_tail_recert',
                           'campaign_results.json')
X0_CELL = 'x0_A_1x'
IMPORTED_SCRIPTS = ('p515_s55_w190a_numbers.py',) + N90.IMPORTED_SCRIPTS
CHECK_FAILURES = []
W187 = {}          # filled by load_w187() before the declarations are built


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


def texnorm(s):
    return P1.texnorm(s.replace('\u2019', "'"))


# ======================================================================================================================
#  state and the diff 3c379fa..HEAD hunk by hunk (no review copy this round: listed, not compared)
# ======================================================================================================================
def _hunks(d):
    out, cur = [], None
    for ln in d.split('\n'):
        m = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', ln)
        if m:
            cur = {'old_start': int(m.group(1)), 'old_n': int(m.group(2) if m.group(2) is not None else 1),
                   'new_start': int(m.group(3)), 'new_n': int(m.group(4) if m.group(4) is not None else 1),
                   'minus': [], 'plus': []}
            out.append(cur)
            continue
        if cur is None or ln.startswith('+++') or ln.startswith('---'):
            continue
        if ln.startswith('-'):
            cur['minus'].append(ln[1:])
        elif ln.startswith('+'):
            cur['plus'].append(ln[1:])
    for h in out:
        h['old_lines'] = [h['old_start'], h['old_start'] + h['old_n'] - 1] if h['old_n'] else []
        h['new_lines'] = [h['new_start'], h['new_start'] + h['new_n'] - 1] if h['new_n'] else []
        h['counts_consistent'] = len(h['minus']) == h['old_n'] and len(h['plus']) == h['new_n']
    return out


def task_state():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    om = _git(['rev-parse', 'origin/main'], cwd=CLONE).strip()
    tracked = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    allst = _git(['status', '--porcelain'], cwd=CLONE).strip()
    chain = _git(['log', '--format=%H %cI %s', f'{PREV_FULL}..{HEAD_FULL}'], cwd=CLONE).strip().split('\n')
    files = {}
    for f in FILES:
        wb, hb = _rd(os.path.join(CLONE, f)), showb(HEAD_FULL, f)
        files[f] = {'sha256': _shab(wb), 'md5': P1._md5b(wb), 'working_equals_head_blob': wb == hb,
                    'lines': wb.count(b'\n'), 'unchanged_since_3c379fa': hb == showb(PREV_FULL, f)}
    changed = [x for x in _git(['diff', '--name-only', PREV_FULL, HEAD_FULL], cwd=CLONE).split('\n') if x]
    stat = _git(['diff', '--stat', PREV_FULL, HEAD_FULL], cwd=CLONE).strip().split('\n')
    hunks = {}
    for f in DIFF_FILES:
        d = _git(['diff', '-U0', PREV_FULL, HEAD_FULL, '--', f], cwd=CLONE)
        hs = _hunks(d)
        hunks[f] = {'n_hunks_U0': len(hs), 'plus_lines': sum(len(h['plus']) for h in hs),
                    'minus_lines': sum(len(h['minus']) for h in hs), 'diff_U0_sha256': _shab(d.encode('utf-8')),
                    'hunks': hs}
    per_commit = []
    for row in chain:
        if not row:
            continue
        c = row.split(' ', 1)[0]
        cs = {}
        for f in DIFF_FILES:
            d = _git(['diff', '-U0', f'{c}^', c, '--', f], cwd=CLONE)
            hs = _hunks(d)
            if hs:
                cs[f] = [{'old_lines': h['old_lines'], 'new_lines': h['new_lines'], 'n_minus': len(h['minus']),
                          'n_plus': len(h['plus'])} for h in hs]
        other = [x for x in _git(['diff', '--name-only', f'{c}^', c], cwd=CLONE).split('\n') if x and x not in DIFF_FILES]
        per_commit.append({'commit': row[:200], 'hunks_by_file': cs, 'other_files': other})
    out = {'clone_head': head, 'origin_main_at_run': om, 'tracked_modified': tracked, 'porcelain_all': allst,
           'commits_3c379fa_to_head': chain, 'files': files, 'files_changed_3c379fa_head': changed, 'diff_stat': stat,
           'hunks_3c379fa_head': hunks, 'per_commit': per_commit}
    out['holds'] = (head == HEAD_FULL and tracked == '' and all(v['working_equals_head_blob'] for v in files.values())
                    and sorted(changed) == sorted(DIFF_FILES)
                    and all(h['counts_consistent'] for v in hunks.values() for h in v['hunks']))
    if not out['holds']:
        _fail(f'(state) HEAD {head[:8]} tracked {tracked!r} changed {changed} '
              f'{[(f, v["working_equals_head_blob"]) for f, v in files.items()]}')
    return out


# ======================================================================================================================
#  (i) W187: the records, Delta F recomputed, the printed passages
# ======================================================================================================================
def load_w187():
    j = json.loads(_rd(W187_JSON))
    sig, tau = j['sigma_Q_eur'], j['TAU_eur']
    uc = j['lattice']['unit_costs_discounted']
    cells, recs = {}, {}
    for c in j['cell_order']:
        cj = j['cells'][c]
        p = os.path.join(W187_DIR, f"campaign_{cj['campaign_id']}", 'campaign_results.json')
        r = json.loads(_rd(p))
        sc = r['outcome']['screen']
        cand = cj['candidate_canonical']
        y = str(cand['investment_year'])
        i_ind = sum(float(s) * uc[y]['power'] + float(e) * uc[y]['energy'] for s, e in cand['nodes'].values())
        recs[c] = {'path': p, 'sha256': _sha(p), 'committed_clean': committed_clean(p), 'cell_field': r.get('cell'),
                   'status': sc['status'], 'Q_eur': sc['certified_cost'], 'bar_eur': sc['bar_eur'],
                   'cycles_run': sc['cycles_run'], 'stopping_ratio': sc['rule_ten']['terminal_step_over_threshold']}
        cells[c] = {'I_recorded_eur': cj['I_eur'], 'I_from_unit_costs_eur': i_ind, 'candidate': cand,
                    'Q_w187_json_eur': cj['Q_gross_eur'], 'bar_w187_json_eur': cj['bar_eur']}
    x0 = recs[X0_CELL]
    rows = []
    for c in j['cell_order']:
        if c == X0_CELL:
            continue
        r = recs[c]
        dq = r['Q_eur'] - x0['Q_eur']
        df = cells[c]['I_from_unit_costs_eur'] + dq
        t_fr = max(r['bar_eur'] + x0['bar_eur'], sig)
        t_v6 = max(3 * max(r['bar_eur'], x0['bar_eur']), 2 * tau)
        jc = next(x for x in j['comparisons_vs_x0'] if x['cell'] == c)
        rows.append({'cell': c, 'year': cells[c]['candidate']['investment_year'],
                     'nodes': sorted(int(n) for n, v in cells[c]['candidate']['nodes'].items() if v[0] or v[1]),
                     'I_eur': cells[c]['I_from_unit_costs_eur'], 'Q_eur': r['Q_eur'], 'dQ_eur': dq, 'dF_eur': df,
                     'bar_n_eur': r['bar_eur'], 'T_frozen_eur': t_fr, 'ratio_T_frozen': abs(df) / t_fr,
                     'verdict_T_frozen': ('x0 better' if df > t_fr else 'neighbour better' if -df > t_fr else 'within resolution'),
                     'T_v6bar_eur': t_v6, 'verdict_T_v6bar': ('x0 better' if df >= t_v6 else 'neighbour better' if -df >= t_v6
                                                              else 'within resolution'),
                     'dF_w187_json_eur': jc['dF_eur'], 'verdict_T_frozen_w187_json': jc['dF__T_frozen']['verdict'],
                     'dF_agrees_w187_json': abs(df - jc['dF_eur']) <= 1e-6,
                     'verdict_agrees_w187_json': jc['dF__T_frozen']['verdict'] == ('x0 better' if df > t_fr else
                                                                                 'neighbour better' if -df > t_fr else
                                                                                 'within resolution')})
    W187.update({'json': j, 'recs': recs, 'cells': cells, 'rows': rows, 'sigma_Q': sig, 'tau': tau,
                 'n': len(rows), 'n_x0_better': sum(1 for r in rows if r['verdict_T_frozen'] == 'x0 better'),
                 'n_within': sum(1 for r in rows if r['verdict_T_frozen'] == 'within resolution'),
                 'n_neighbour_better': sum(1 for r in rows if r['verdict_T_frozen'] == 'neighbour better'),
                 'dF_min': min(r['dF_eur'] for r in rows), 'dF_max': max(r['dF_eur'] for r in rows)})


def w187_n(X):
    j = W187['json']
    ok = (j['lattice']['x0_lattice_neighbourhood_size'] == W187['n']
          and j['lattice']['screen_neighbours_equal_lattice_neighbourhood_of_x0'] is True)
    return W187['n'] if ok else None


def w187_count(which):
    return lambda X: W187[which]


def w187_df(which):
    return lambda X: W187[which]


def task_w187(texts):
    j, rows, recs = W187['json'], W187['rows'], W187['recs']
    main = texts[MAIN].split('\n')
    letter = texts[LETTER].split('\n')
    cover = texts[COVER].split('\n')
    chk = collections.OrderedDict()
    val = collections.OrderedDict()
    # the records
    chk['W187 JSON sha256 = the report pin 1afc8a78'] = _sha(W187_JSON).startswith(W187_JSON_SHA_PREFIX)
    chk['15 campaign_results.json files read, each cell field = its W187 cell, each status certified'] = (
        len(recs) == 15 and all(v['cell_field'] == c and v['status'] == 'certified' for c, v in recs.items()))
    chk['each campaign_results.json committed and clean'] = all(v['committed_clean'] for v in recs.values())
    chk['Q of every cell (campaign_results.json) = the W187 JSON Q, bitwise'] = all(
        recs[c]['Q_eur'] == W187['cells'][c]['Q_w187_json_eur'] for c in recs)
    chk['bar of every cell (campaign_results.json) = the W187 JSON bar, bitwise'] = all(
        recs[c]['bar_eur'] == W187['cells'][c]['bar_w187_json_eur'] for c in recs)
    chk['I(x) from the recorded unit costs = the W187 JSON I (|diff| <= 1e-6 EUR), 15 cells'] = all(
        abs(v['I_from_unit_costs_eur'] - v['I_recorded_eur']) <= 1e-6 for v in W187['cells'].values())
    chk['14 neighbours = the lattice neighbourhood of x = 0 (W187 JSON)'] = w187_n(None) == 14
    chk['Delta F recomputed = W187 JSON Delta F (|diff| <= 1e-6 EUR), 14 / 14'] = all(r['dF_agrees_w187_json'] for r in rows)
    chk['T_frozen verdict recomputed = W187 JSON verdict, 14 / 14'] = all(r['verdict_agrees_w187_json'] for r in rows)
    chk['Delta F > 0 on all 14'] = all(r['dF_eur'] > 0 for r in rows)
    chk['T_frozen: 0 neighbour better, 8 x0 better, 6 within'] = (
        W187['n_neighbour_better'], W187['n_x0_better'], W187['n_within']) == (0, 8, 6)
    chk['W187 JSON P3 dF__T_frozen counts (8, 6) equal the recomputation'] = (
        j['P3']['dF__T_frozen']['all_14']['n_x0_better'], j['P3']['dF__T_frozen']['all_14']['n_within_resolution']) == (
        W187['n_x0_better'], W187['n_within'])
    chk['range +2,684.02 .. +100,584.21 EUR rounds to 2.7 .. 100.6 kEUR'] = (
        f"{W187['dF_min'] / 1e3:.1f}" == '2.7' and f"{W187['dF_max'] / 1e3:.1f}" == '100.6')
    w86 = json.loads(_rd(W86_RESULTS))
    q86 = w86['per_cell']['x0']['detail']['post_certification']['certified_cost']
    chk['Q(x0) of the screen = W86 tail recert Q, bitwise'] = recs[X0_CELL]['Q_eur'] == q86
    val['records'] = {'sigma_Q_eur': W187['sigma_Q'], 'tau_eur': W187['tau'], 'Q_x0_eur': recs[X0_CELL]['Q_eur'],
                      'bar_x0_eur': recs[X0_CELL]['bar_eur'], 'W86_Q_x0_eur': q86, 'W86_results': W86_RESULTS,
                      'W86_results_sha256': _sha(W86_RESULTS), 'dF_min_eur': W187['dF_min'], 'dF_max_eur': W187['dF_max'],
                      'rows': rows,
                      'unit_costs_discounted': j['lattice']['unit_costs_discounted'],
                      'cells': {c: {k: v for k, v in r.items()} for c, r in recs.items()}}
    # the main-text paragraph (4.3)
    i43 = [i for i, x in enumerate(main) if 'The planning question was also tested directly' in x]
    t43 = texnorm(main[i43[0]]) if len(i43) == 1 else ''
    sec = [i for i, x in enumerate(main) if re.match(r'\\subsection\{', x)]
    own = max((i for i in sec if i43 and i < i43[0]), default=None)
    chk['4.3 paragraph located once'] = len(i43) == 1
    chk['4.3 paragraph lies under \\subsection{Effect of Battery Degradation}'] = (
        own is not None and 'Effect of Battery Degradation' in main[own])
    phrases_43 = collections.OrderedDict([
        ('"the fourteen admissible unit neighbours of the plan without storage"',
         'the fourteen admissible unit neighbours of the plan without storage'),
        ('"under the no-ageing calibration with the production stopping rule"',
         'under the no-ageing calibration with the production stopping rule'),
        ('"none reduces the total discounted cost"', 'none reduces the total discounted cost'),
        ('"eight are more expensive beyond the search\'s acceptance threshold and six lie within it" (the printed form, '
         'Addendum 92 ruling 2)', "eight are more expensive beyond the search's acceptance threshold and six lie within it"),
        ('"with cost increases from 2.7 to 100.6 k€"', 'with cost increases from 2.7 to 100.6 k€'),
        ('"without ageing, therefore, the search would also stop at the plan without storage"',
         'without ageing, therefore, the search would also stop at the plan without storage'),
        ('"at this resolution it cannot discriminate the smallest storage units from no storage"',
         'at this resolution it cannot discriminate the smallest storage units from no storage'),
        ('"with ageing, every unit neighbour is determinately worse"', 'with ageing, every unit neighbour is determinately worse'),
        ('scope: "a bounded screen of the unit neighbourhood at the base flexibility price, not a completed search"',
         'the comparison is a bounded screen of the unit neighbourhood at the base flexibility price, not a completed search')])
    for k, p in phrases_43.items():
        chk[f'4.3 wording {k}'] = p in t43
    val['4.3'] = {'line': i43[0] + 1 if len(i43) == 1 else None, 'text': ws(main[i43[0]]) if len(i43) == 1 else None}
    # the claim "with ageing, every unit neighbour is determinately worse (Subsection~\ref{subsec:res_baseline})"
    lab = next((i for i, x in enumerate(main) if '\\label{subsec:res_baseline}' in x), None)
    nxt = next((i for i in sec if lab is not None and i > lab), None)
    region41 = texnorm(' '.join(main[lab:nxt])) if lab is not None else ''
    tb = next((i for i in range(lab, nxt) if '\\begin{table}' in main[i]), nxt) if lab is not None else None
    prose41 = texnorm(' '.join(main[lab:tb])) if lab is not None else ''
    neigh_sent = [s for s in re.split(r'(?<=\.)\s+', prose41) if 'neighbour' in s]
    b3 = next((i for i, x in enumerate(main) if 'The planning search evaluated all fourteen admissible unit neighbours' in x), None)
    tb3 = texnorm(main[b3]) if b3 is not None else ''
    m = re.search(r'by ([0-9.]+)--([0-9.]+) times the search resolution', tb3)
    clm = collections.OrderedDict()
    clm['claim "every unit neighbour determinately worse with ageing": Appendix B.3 states it (8 certified determinate + '
        '6 at > 1x the search resolution)'] = bool(
        'every one costs determinately more than the plan without storage' in tb3 and m and float(m.group(1)) > 1.0)
    clm['claim "... determinately worse": the pointed target 4.1 (subsec:res_baseline) states it (a sentence of its prose '
        'on the unit neighbours says "determinate")'] = any('determinat' in s for s in neigh_sent)
    val['4.3_claim_pointer'] = {'pointer': 'Subsection~\\ref{subsec:res_baseline}', 'region_4_1_lines': [lab + 1, nxt] if lab else None,
                                'prose_4_1_lines': [lab + 1, tb] if lab else None,
                                '4_1_prose_sentences_on_neighbours': neigh_sent,
                                '4_1_region_mentions_determinate_anywhere': 'determinate' in region41,
                                '4_1_region_note': 'the only "determinate" in the 4.1 region is the tab:res_value_ladder note '
                                                   '("all cost increases are determinate"), on the six listed plans, not the '
                                                   'unit neighbours',
                                '4_1_mentions_resolution': 'resolution' in region41,
                                'appendix_B3_line': b3 + 1 if b3 is not None else None,
                                'appendix_B3_range_times_resolution': [m.group(1), m.group(2)] if m else None}
    # the letter (l. 58-60 "What we did not do"; l. 263-265 R3.6) and the cover letter
    ilt = [i for i, x in enumerate(letter) if 'a screen of the fourteen unit neighbours of the plan without storage' in x]
    tlt = texnorm(letter[ilt[0]]) if len(ilt) == 1 else ''
    chk['letter "What we did not do": the screen sentence located once'] = len(ilt) == 1
    chk['letter "What we did not do": "a screen of the fourteen unit neighbours ... under the no-ageing calibration, which '
        'returns the same plan (section 4.3)"'] = (
        'a screen of the fourteen unit neighbours of the plan without storage under the no-ageing calibration, which returns '
        'the same plan (section 4.3)' in tlt)
    chk['letter "What we did not do": "a full degradation-neutral search over the lattice was not run, and the paper says so '
        '(section 4.7)"'] = 'a full degradation-neutral search over the lattice was not run, and the paper says so (section 4.7)' in tlt
    ir = [i for i, x in enumerate(letter) if 'The planning part of the request was then met at the scale the method allows' in x]
    tr = texnorm(letter[ir[0]]) if len(ir) == 1 else ''
    chk['letter R3.6: the screen sentence located once'] = len(ir) == 1
    chk['letter R3.6: "the fourteen admissible unit neighbours ... under the no-ageing calibration as a planning search would '
        'evaluate them (section 4.3)"'] = (
        'the fourteen admissible unit neighbours of the plan without storage were evaluated under the no-ageing calibration as a '
        'planning search would evaluate them (section 4.3)' in tr)
    chk['letter R3.6: "eight are worse beyond the search\'s acceptance threshold and six within it" (printed form)'] = (
        "eight are worse beyond the search's acceptance threshold and six within it" in tr)
    chk['letter R3.6: "so a degradation-neutral search would return the same plan"'] = (
        'so a degradation-neutral search would return the same plan' in tr)
    chk['letter R3.6: "a full degradation-neutral search over the lattice was not run"'] = (
        'a full degradation-neutral search over the lattice was not run' in tr)
    i263 = [i for i, x in enumerate(letter) if 'and the unit-neighbourhood screen its effect on the selected plan' in x]
    val['letter'] = {'what_we_did_not_do_line': ilt[0] + 1 if ilt else None, 'R3_6_screen_line': ir[0] + 1 if ir else None,
                     'R3_6_selected_plan_line': i263[0] + 1 if i263 else None,
                     'texts': {str(i + 1): ws(letter[i]) for i in ilt + ir + i263}}
    ic = [i for i, x in enumerate(cover) if 'A degradation-neutral evaluation of the unit neighbourhood' in x]
    tc = texnorm(cover[ic[0]]) if len(ic) == 1 else ''
    chk['cover letter: the screen sentence located once'] = len(ic) == 1
    chk['cover letter: "is also reported (section 4.3): without ageing the search would return the same plan"'] = (
        'a degradation-neutral evaluation of the unit neighbourhood of the plan without storage is also reported (section 4.3): '
        'without ageing the search would return the same plan' in tc)
    body = re.sub(r'\\[A-Za-z]+\*?\[[^\]]*\]', ' ', texts[COVER])
    br = [(i + 1, mm.group(0)) for i, ln in enumerate(body.split('\n'))
          for mm in re.finditer(r'\[[^\]]*\]', re.sub(r'(?<!\\)%.*', '', ln))]
    chk['cover letter: the W187 bracket resolved (0 brackets left)'] = len(br) == 0
    val['cover'] = {'line': ic[0] + 1 if ic else None, 'text': ws(cover[ic[0]]) if ic else None, 'brackets': br}
    # not printed (Addendum 92 ruling 2) and other numbers of the record
    alltx = {f: texts[f] for f in (MAIN, LETTER, COVER)}
    new_lines_text = {f: '\n'.join(h for x in W187_STATE['hunks'][f]['hunks'] for h in x['plus']) for f in DIFF_FILES}
    np_ = collections.OrderedDict()
    np_['the 59.9 kEUR bar: "59.9" / "59,9" / "59{,}9" / "59,925" printed nowhere (main, letter, cover)'] = {
        f: [i + 1 for i, x in enumerate(t.split('\n')) if re.search(r'59(\.|,|\{,\})9', x)] for f, t in alltx.items()}
    np_['salvage on the lines changed since 3c379fa (the salvage sensitivity of the screen)'] = {
        f: [x[:120] for x in t.split('\n') if re.search(r'salvage', x, re.I)] for f, t in new_lines_text.items()}
    np_['the 3b/2tau reading (5 / 9) on the changed lines: "five" / "nine" / "5 are" / "9 are"'] = {
        f: [x[:120] for x in t.split('\n') if re.search(r'\b(nine|five are|nine are|5 are|9 are)\b', x, re.I)]
        for f, t in new_lines_text.items()}
    np_['Q(x0) of the screen printed ("653.9", "653,858", "653{,}858", "653.86")'] = {
        f: [i + 1 for i, x in enumerate(t.split('\n')) if re.search(r'653(\.9|,858|\{,\}858|\.86)', x)] for f, t in alltx.items()}
    np_['W86 named'] = {f: [i + 1 for i, x in enumerate(t.split('\n')) if 'W86' in x] for f, t in alltx.items()}
    np_['"not a globally preferred zero investment" (plan 5.3 scope clause) on the changed lines'] = {
        f: 'globally preferred' in t for f, t in new_lines_text.items()}
    chk['not printed: the 59.9 kEUR bar (Addendum 92 ruling 2)'] = not any(np_[next(iter(np_))].values())
    chk['not printed: the salvage sensitivity on the changed lines (Addendum 92 ruling 2)'] = not any(
        np_['salvage on the lines changed since 3c379fa (the salvage sensitivity of the screen)'].values())
    chk['not printed: the 3b/2tau counts 5 / 9 on the changed lines'] = not any(
        np_['the 3b/2tau reading (5 / 9) on the changed lines: "five" / "nine" / "5 are" / "9 are"'].values())
    val['not_printed_scan'] = np_
    out = {'checks': dict(chk), 'claim_pointer_checks': dict(clm), 'values': val, 'holds': all(chk.values()),
           'claim_pointer_holds': all(clm.values()),
           'objective_convention': 'F = I(x) + Q(x), Q = gross_operational_cost with the contracted interface settlement '
                                   'excluded; salvage excluded from F (Addendum 27 item 3); Delta F = F(n) - F(x0)',
           'instance': {c: {'eval_key': j['cells'][c]['eval_key'], 'candidate_key': j['cells'][c]['candidate_key']}
                        for c in j['cell_order']}}
    return out


W187_STATE = {}


# ======================================================================================================================
#  (ii) numbers: declarations W190b
# ======================================================================================================================
def der_we_did_not_par(letter_lines):
    def f(X):
        k = next(i for i, x in enumerate(letter_lines) if x.startswith('\\paragraph{What we did not do'))
        e = next(i for i in range(k + 1, len(letter_lines)) if not letter_lines[i].strip())
        n = sum(len(re.findall(r'We did not ', letter_lines[i])) for i in range(k + 1, e))
        return NB._dval(n, f'count of "We did not" sentences in the "What we did not do" paragraph (letter l. {k + 2}-{e}; '
                           f'the paragraph is split over three lines at 5f4d3bd)')
    return f


def declarations_w190b(X, actx, letter_lines):
    S, SF, U, DER = NB.S, NB.SF, NB.U, NB.DER
    DLF, by_w = P1.DLF, P1.by_w
    b182 = {d['line']: d for d in NB.declarations()}
    w189 = ORIG_DECL_W189(X, actx, letter_lines)
    bl = {d['line']: d for d in w189[LETTER]}
    w190a = DECL_W190A(X, actx, letter_lines)
    cl = {d['line']: d for d in w190a[COVER]}
    ab = b182[146]['specs']
    n14 = SF('fourteen', 'int', w187_n, 'W187 JSON lattice.x0_lattice_neighbourhood_size = number of compared neighbours '
                                        '(screen neighbours = lattice neighbourhood of x = 0)')
    n8 = SF('eight', 'int', w187_count('n_x0_better'), 'W187 cells: Delta F > max(bar_n + bar_x0, sigma_Q) (the search\'s '
                                                       'acceptance criterion), recomputed from the 15 campaign_results.json')
    n6 = SF('six', 'int', w187_count('n_within'), 'W187 cells: |Delta F| <= max(bar_n + bar_x0, sigma_Q), recomputed')
    dmin = SF('2.7', 'k', w187_df('dF_min'), 'W187 cells: min Delta F = I(n) + Q(n) - Q(x0) over the 14 (2025, n5+n7+n9)',
              conv='F = I + Q, Q gross, settlement excluded, salvage excluded')
    dmax = SF('100.6', 'k', w187_df('dF_max'), 'W187 cells: max Delta F over the 14 (2030, n5+n7+n9)',
              conv='F = I + Q, Q gross, settlement excluded, salvage excluded')
    M, L_, C = [], [], []
    M.append(DLF(MAIN, 146, *by_w(ab, '108', '15', '12.2'),
                 ctx='W182b l. 146 (abstract), first part; the abstract line split after "transmission." at 5f4d3bd'))
    M.append(DLF(MAIN, 147, *by_w(ab, '0.25', '1', '64.4', '1.75', '2', '3', '3'),
                 ctx='W182b l. 146 (abstract), second part; text unchanged'))
    M.append(DLF(MAIN, 388, *by_w(b182[145]['specs'], 'two'), ctx='contribution 1 rewritten: "two-stage" (W182b l. 145 spec)'))
    M.append(DLF(MAIN, 391, *by_w(ab, '108'), ctx='contribution 4 rewritten: "108-bus" (W182b abstract spec)'))
    M.append(DLF(MAIN, 1374, *by_w(b182[1373]['specs'], 'fourteen'), ctx='4.1 rewritten (author): W182b l. 1373 spec'))
    M.append(DLF(MAIN, 1375, *by_w(b182[1374]['specs'], '0.25', '1', '7', '2025', '253.5', '318.0', '64.4'),
                 U('two', 'compound adjective', '"two-node plans" (a plan class; the plans are rows of tab:res_value_ladder)'),
                 ctx='4.1 rewritten (author): W182b l. 1374 specs (the node-5 sentence and its 49.7 dropped)'))
    M.append(DLF(MAIN, 1509, dict(n14), n8, n6, dmin, dmax, ctx='new (W187 paragraph, 4.3)'))
    L_.append(DLF(LETTER, 58, DER('Two', der_we_did_not_par(letter_lines), 'requests not met in full'),
                  ctx='W189p1 l. 58 "Two", source re-stated over the paragraph (now three lines)'))
    L_.append(DLF(LETTER, 59, dict(n14), ctx='new (W187 screen in "What we did not do")'))
    L_.append(DLF(LETTER, 60, *by_w(bl[58]['specs'], 'five', 'twenty-five', '3', '3'), ctx='W189p1 l. 58 specs (sentence moved)'))
    L_.append(DLF(LETTER, 263, *by_w(bl[261]['specs'], '0.70', '0.50', '+4.9'), ctx='W189p1 l. 261 specs (first part)'))
    L_.append(DLF(LETTER, 264, dict(n14), dict(n8), dict(n6), ctx='new (W187 screen in R3.6)'))
    L_.append(DLF(LETTER, 265, *by_w(bl[261]['specs'], '2035', '50'), ctx='W189p1 l. 261 specs (last part)'))
    C.append(DLF(COVER, 21, *by_w(cl[21]['specs'], 'two', '20.5', '0.02', '90.9', '12.2', 'two'),
                 ctx='W190a l. 21 specs (the W187 bracket removed from the line)'))
    return {MAIN: M, LETTER: L_, COVER: C}


# ======================================================================================================================
#  (ii) formatting of the '+' lines and the claims of the rewritten 4.1 paragraph
# ======================================================================================================================
UNIT_GAP = re.compile(r'(\d)( +)(k\\euro|M\\euro|\\euro|MVA|MWh|MW\b|kV\b|\\%|h\b)')
NODE_GAP = re.compile(r'\b(nodes?)( +)(\d)')
REF_GAP = re.compile(r'\b(Tables?|Figures?|Fig\.|Sections?|Subsections?|Appendix|Algorithm|Eq\.|Equations?)( +)(\\ref|\\eqref)')
HARD_REF = re.compile(r'\b(Sections?|Subsections?|Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Appendix|Appendices|Algorithms?)'
                      r'(~|\\,|\s+)\(?([A-Z]?\.?\d+(?:\.\d+)*)\)?')


def scan_format(texts):
    rows = []
    for f in DIFF_FILES:
        lines = texts[f].split('\n')
        for h in W187_STATE['hunks'][f]['hunks']:
            for k in range(len(h['plus'])):
                ln = h['new_start'] + k
                raw = lines[ln - 1]
                cs = P1.comment_start(raw)
                x = raw if cs is None else raw[:cs]
                for i, ch in enumerate(x):
                    if ch == '\u20ac':
                        rows.append({'file': f, 'line': ln, 'kind': 'literal euro sign', 'col': i + 1,
                                     'context': x[max(0, i - 40):i + 20], 'fix': 'write \\euro{} (with a tie before it)'})
                for rx, kind, fix in ((UNIT_GAP, 'number before a unit without a tie', 'replace the space by ~'),
                                      (NODE_GAP, '"node(s)" before a number without a tie', 'replace the space by ~'),
                                      (REF_GAP, 'object word before \\ref without a tie', 'replace the space by ~')):
                    for m in rx.finditer(x):
                        rows.append({'file': f, 'line': ln, 'kind': kind, 'col': m.start(2) + 1, 'text': m.group(0),
                                     'context': x[max(0, m.start() - 50):m.end() + 10],
                                     'fix': f"{fix}: {m.group(0).replace(m.group(2), '~')}"})
                if f == MAIN:
                    for m in HARD_REF.finditer(x):
                        rows.append({'file': f, 'line': ln, 'kind': 'hard-coded reference', 'col': m.start() + 1,
                                     'text': m.group(0), 'context': x[max(0, m.start() - 50):m.end() + 10],
                                     'fix': 'use ~\\ref{<label>}'})
                for i, ch in enumerate(x):
                    if ord(ch) > 127 and ch != '\u20ac':
                        rows.append({'file': f, 'line': ln, 'kind': f'non-ASCII character U+{ord(ch):04X}', 'col': i + 1,
                                     'context': x[max(0, i - 30):i + 10],
                                     'fix': "write ' (straight apostrophe), as elsewhere in the file" if ch == '\u2019'
                                     else 'write the LaTeX form'})
                if raw != raw.rstrip():
                    rows.append({'file': f, 'line': ln, 'kind': 'trailing whitespace (harmless)', 'col': len(raw.rstrip()) + 1,
                                 'context': raw[-30:], 'fix': 'none needed'})
    return rows


def task_41(texts, numbers):
    main = texts[MAIN].split('\n')
    chk = collections.OrderedDict()
    val = collections.OrderedDict()
    lab = next(i for i, x in enumerate(main) if '\\label{subsec:res_baseline}' in x)
    sec = [i for i, x in enumerate(main) if re.match(r'\\subsection\{', x)]
    nxt = next(i for i in sec if i > lab)
    par = [i for i in range(lab, nxt) if 'Under the baseline assumptions, no shared storage investment recovers its cost' in main[i]]
    t1 = texnorm(main[par[0]]) if par else ''
    t2 = texnorm(main[par[0] + 1]) if par else ''
    rows = {(r['line'], r['written']): r for r in numbers['tokens'] if r['file'] == MAIN}
    v_val = rows.get((1375, '253.5'), {}).get('counterpart_value')
    v_inv = rows.get((1375, '318.0'), {}).get('counterpart_value')
    v_d = rows.get((1375, '64.4'), {}).get('counterpart_value')
    chk['4.1 paragraph located (two lines)'] = bool(par) and par[0] + 1 == 1374
    if all(isinstance(v, (int, float)) for v in (v_val, v_inv, v_d)):
        chk['"saves 253.5 ... against an investment of 318.0, a shortfall of 64.4": I - value = shortfall (records, |diff| < 1 EUR)'] = (
            abs((v_inv - v_val) - v_d) < 1.0)
        val['records_eur'] = {'value_253.5': v_val, 'I_318.0': v_inv, 'shortfall_64.4': v_d, 'I_minus_value': v_inv - v_val,
                              'printed_318.0_minus_253.5': round(318.0 - 253.5, 1),
                              'note': 'the printed 318.0 - 253.5 = 64.5 while the shortfall prints 64.4: each figure is the '
                                      'record rounded at 1 dp (the same three figures stood at 3c379fa)'}
    # "larger units, longer durations and two-node plans lose more" against tab:res_value_ladder as printed
    i0 = next(i for i, x in enumerate(main) if '\\label{tab:res_value_ladder}' in x)
    i1 = next(i for i in range(i0, len(main)) if '\\bottomrule' in main[i])
    tab = []
    for i in range(i0, i1):
        c = [x.strip() for x in main[i].replace('\\\\', '').split('&')]
        if len(c) == 6 and re.match(r'^\d', c[0]) and re.match(r'^[0-9.]+$', c[4]):
            tab.append({'line': i + 1, 'nodes': c[0], 'S': c[1], 'E': c[2], 'h': c[3], 'cost_increase_k': float(c[4])})
    ref = next((r for r in tab if 'reference' in r['nodes']), None)
    larger = [r for r in tab if r['S'].startswith('0.5')]
    two = [r for r in tab if ' and ' in r['nodes']]
    pairs = [(a, b) for a in tab for b in tab if a['S'] == b['S'] and ' and ' not in a['nodes'] + b['nodes']
             and a['h'] == '4' and b['h'] == '2']
    chk['"larger units ... lose more": every 0.5 MVA row > the reference unit (64.4)'] = bool(ref and larger) and all(
        r['cost_increase_k'] > ref['cost_increase_k'] for r in larger)
    chk['"two-node plans lose more": every two-node row > the reference unit'] = bool(ref and two) and all(
        r['cost_increase_k'] > ref['cost_increase_k'] for r in two)
    chk['"longer durations ... lose more": at equal converter rating the 4 h row > the 2 h row (every pair)'] = bool(pairs) and all(
        a['cost_increase_k'] > b['cost_increase_k'] for a, b in pairs)
    val['tab_res_value_ladder_printed'] = tab
    val['longer_duration_note'] = ('no row is longer than the reference unit\'s 4 h; "longer durations" holds only read as 4 h '
                                   'against 2 h at equal converter rating')
    # scope of "no shared storage investment recovers its cost" against the abstract and 4.7
    abs_t = texnorm(' '.join(main[next(i for i, x in enumerate(main) if '\\begin{abstract}' in x):
                                   next(i for i, x in enumerate(main) if '\\end{abstract}' in x)]))
    s47 = next((i for i, x in enumerate(main) if 'The baseline conclusion applies to the evaluated plans' in x), None)
    chk['4.1 first sentence scoped like the abstract / 4.7 (to the evaluated plans)'] = (
        'no shared storage investment recovers its cost' not in t1)
    val['scope'] = {'4_1_sentence': 'Under the baseline assumptions, no shared storage investment recovers its cost.',
                    'abstract_has': 'none of the evaluated storage plans recovers its investment cost' in abs_t,
                    '4_7_line': s47 + 1 if s47 is not None else None,
                    '4_7_has': 'The baseline conclusion applies to the evaluated plans and does not establish global optimality.'}
    chk['4.1 keeps "beyond the search resolution" (Appendix B.3 l. "at its own resolution (...; Subsection~\\ref{subsec:'
        'res_baseline})" points here for it)'] = 'resolution' in t1 + ' ' + t2
    val['texts'] = {'1374': ws(main[1373]), '1375': ws(main[1374])}
    return {'checks': dict(chk), 'values': val, 'holds': all(chk.values())}


# ======================================================================================================================
#  checklist, guards, md, post-run, main
# ======================================================================================================================
REQUIRED_CAPTURES = {'state and hunks': 'task_state', '(i) W187': 'task_w187', '(i) records': 'load_w187',
                     '(ii) numbers': 'declarations_w190b', '(ii) formatting': 'scan_format', '(ii) 4.1 claims': 'task_41',
                     'guards': 'verify_guards'}
REQUIRED_SYMBOLS = {'W190a part 1': (N90, ('task_generator', 'verifier_summary', 'mark_new', 'declarations_w190a', 'ORIG_DECL',
                                           'BASE_FULL', 'HEAD_FULL', 'OUT_NAMES', '_VERIFY_TEXT', 'verify_guards',
                                           'controls_w190a', 'CHECK_FAILURES')),
                    'W189p1': (P1, ('task_appendix', 'task_numbers', 'controls_appendix', 'load_extras', 'tokens_of', 'DLF',
                                    'by_w', 'texnorm', 'comment_start', 'EX', 'OUT_NAMES', 'PREV_FULL', 'HEAD_FULL',
                                    'CHECK_FAILURES', 'PICKLE_COUNTS')),
                    'W182b': (NB, ('S', 'SF', 'U', 'DER', 'declarations', '_dval'))}


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
    need = [os.path.join(CLONE, f) for f in FILES] + [GEN_REL, VERIFY_REL, APPX_REL, W190A_RED, W187_JSON, W187_REPORT,
                                                     W86_RESULTS]
    for p in need:
        if not os.path.isfile(os.path.join(REPO, p)):
            missing.append(f'input {p}')
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    if os.path.realpath(od).startswith(os.path.realpath(os.path.join(REPO, 'data')) + os.sep) or \
            os.path.realpath(od).startswith(os.path.realpath(os.path.join(REPO, MR)) + os.sep):
        missing.append('output directory under data/ or manuscript_review/ (not permitted for W190b)')
    if missing:
        for x in missing:
            print(f'REFUSED (capture-path checklist): missing {x}', flush=True)
        sys.exit(4)
    return {'captures': REQUIRED_CAPTURES, 'symbols': {k: list(v[1]) for k, v in REQUIRED_SYMBOLS.items()}, 'all_present': True}


def verify_guards():
    g = N90.verify_guards()
    mine = GUARD.verify(0)
    ok = g['ok'] and not mine
    if not ok:
        _fail(f'guards: W190a / W189p1 chain ok {g["ok"]}; this script {mine}')
    return {'w190a_w189p1_chain': g, 'w190b': {'counts': dict(GUARD.counts), 'verify_0_failures': mine}, 'ok': ok}


def _cell(s, n=200):
    return str(s).replace('|', '\\|').replace('\n', ' ')[:n]


def write_md(out, path):
    N, G, V, I7, F4, FM = out['numbers'], out['generator'], out['verifier'], out['w187'], out['para_4_1'], out['format_scan']
    L = [f'# W190b part 1 -- W187 numbers, numbers, formatting and Appendix B gate at Overleaf {HEAD_FULL[:7]} '
         f'(generated by {SCRIPT_NAME}; do not edit)', '',
         'Objective convention: F = I + Q, Q = gross_operational_cost with the contracted interface settlement excluded, '
         'salvage excluded from F; every other rated money token carries its own convention.', '',
         f"Check failures: {len(out['check_failures'])}; guards ok {out['guards']['ok']}.", '', '## Diff 3c379fa..HEAD, hunk by hunk', '']
    for f, v in out['state']['hunks_3c379fa_head'].items():
        L.append(f"### {f}: {v['n_hunks_U0']} hunks, +{v['plus_lines']} / -{v['minus_lines']}")
        for k, h in enumerate(v['hunks'], 1):
            L.append(f"- hunk {k}: old l. {h['old_lines'] or '(none)'} -> new l. {h['new_lines'] or '(none)'}")
            L += [f"  - `-` {_cell(x, 300)}" for x in h['minus']]
            L += [f"  - `+` {_cell(x, 300)}" for x in h['plus']]
    L += ['', '## (i) W187', '', '| neighbour | I (EUR) | dQ (EUR) | dF (EUR) | T_frozen (EUR) | ratio | verdict |',
          '|---|---:|---:|---:|---:|---:|---|']
    L += [f"| {r['cell']} | {r['I_eur']:,.2f} | {r['dQ_eur']:,.2f} | {r['dF_eur']:+,.2f} | {r['T_frozen_eur']:,.2f} | "
          f"{r['ratio_T_frozen']:.2f} | {r['verdict_T_frozen']} |" for r in I7['values']['records']['rows']]
    L += [''] + [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in I7['checks'].items()]
    L += [f"- {'HOLDS' if v else 'DOES NOT HOLD'} (claim / pointer): {k}" for k, v in I7['claim_pointer_checks'].items()]
    L += ['', '## (ii) numbers', '', f"By rating: {N.get('by_rating_total')}", '']
    for f, v in N.get('files', {}).items():
        L.append(f"- {f}: '+' lines {v['plus_lines']}, {v['n_tokens']} tokens, {v['by_rating']}")
    L += ['', f"Negative control: {N.get('negative_control', {}).get('n_refused')} / {N.get('negative_control', {}).get('n')} refused.",
          '', '| file | line | written | new vs 3c379fa | rating | counterpart | at precision | convention |',
          '|---|---:|---|---|---|---|---|---|']
    L += [f"| {r['file']} | {r['line']} | {r['written']} | {r.get('new_vs_9b')} | {r['rating']} | "
          f"{_cell(r['counterpart'] or r.get('category'), 180)} | {_cell(r['value_at_written_precision'], 50)} | "
          f"{_cell(r['convention'], 40)} |" for r in N.get('tokens', [])]
    L += ['', '## (ii) the rewritten 4.1 paragraph', ''] + [f"- {'HOLDS' if v else 'DOES NOT HOLD'}: {k}" for k, v in F4['checks'].items()]
    L += ['', '## (ii) formatting scan of the changed lines', '', '| file | line | kind | text / context | fix |', '|---|---:|---|---|---|']
    L += [f"| {r['file']} | {r['line']} | {r['kind']} | {_cell(r.get('text') or r['context'], 90)} | {_cell(r['fix'], 80)} |" for r in FM]
    L += ['', '## (v) Appendix B gate and verifier', '',
          f"- reduced output {G['reduced']['n_lines']} lines, sha256 `{G['reduced']['sha256']}`; occurrences in HEAD main.tex "
          f"{G['reduced']['n_occurrences_in_9c_main']} (at {[(o['start_line'], o['end_line']) for o in G['reduced']['occurrences_in_9c_main']]}); "
          f"lines equal {G['reduced']['lines_equal_at_located_position']}; bitwise gate {G['gate_bitwise_reduced_in_main']}",
          f"- equals the W190a reduced output bitwise: {out['generator_vs_w190a']['equal_bytes']}",
          f"- verify_appE: PASS {V['n_pass']}, FAIL {V['n_fail']}, exception {V['exception']}", '']
    L += ['## Negative controls', ''] + [f"- {c['id']}: refused {c['refused']}" for c in out['negative_controls']]
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


def controls_w190b(texts):
    out = []
    # (i) a planted "nine" in the 4.3 paragraph is caught by the not-printed scan's pattern
    out.append({'id': 'i-1 a planted "nine are within" on a changed line is matched by the 3b/2tau scan',
                'refused': bool(re.search(r'\b(nine|five are|nine are|5 are|9 are)\b', 'and nine are within it', re.I))})
    # (i) a perturbed printed count is not the recomputed count
    out.append({'id': 'i-2 "seven are more expensive" does not match the recomputed 8', 'refused': W187['n_x0_better'] != 7})
    # (ii) the formatting scan on planted text
    planted = 'saves 12.5~k\\euro{} and 3 MWh at node 7, see Table 4 and 2 \u20ac'
    hits = ([m.group(0) for m in UNIT_GAP.finditer(planted)] + [m.group(0) for m in NODE_GAP.finditer(planted)]
            + [m.group(0) for m in HARD_REF.finditer(planted)])
    out.append({'id': 'ii-1 planted "3 MWh", "node 7", "Table 4" are each flagged; "12.5~k\\euro{}" is not', 'hits': hits,
                'refused': sorted(hits) == sorted(['3 MWh', 'node 7', 'Table 4']) and '\u20ac' in planted})
    return out


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
    _log(f'W190b part 1: start (out {out_dir})')
    out = {'schema': STAGE, 'task': 'W190b part 1 (Addendum 92, W190b line (i), (ii), (v)): W187 numbers, numbers and '
                                    'formatting of the changed lines, Appendix B gate and verifier at Overleaf 5f4d3bd',
           'objective_convention': 'W187: F = I + Q, Q = gross_operational_cost, contracted interface settlement excluded, '
                                   'salvage excluded from F; other rated numbers carry their own convention',
           'static_reading': 'no TeX installation on this machine; nothing compiled',
           'started_utc': t0.isoformat(), 'script': {'path': SCRIPT_NAME, 'sha256': _sha(SCRIPT_NAME),
                                                      'committed_clean': committed_clean(SCRIPT_NAME)},
           'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'repo_tracked_modified': _git(['status', '--porcelain', '--untracked-files=no']).strip(),
           'interpreter': sys.executable, 'imported': {s_: _sha(s_) for s_ in IMPORTED_SCRIPTS},
           'revisions': {'head_5f4d3bd': HEAD_FULL, 'prev_3c379fa': PREV_FULL, 'round8_a87e195': A87_FULL},
           'key_names': 'the W190a functions re-used here keep their key names: "9c" in a generator / mark_new key means '
                        'HEAD 5f4d3bd and "9b" means 3c379fa (their BASE_FULL / HEAD_FULL re-bound)'}
    out['capture_checklist'] = capture_checklist(out_dir)
    _log('capture-path checklist: all present')
    probs = P1.load_extras()
    out['extra_input_pins'] = {'problems': probs}
    for p in probs:
        _fail(f'(inputs) {p}')
    out['state'] = task_state()
    W187_STATE.update({'hunks': out['state']['hunks_3c379fa_head']})
    _log(f"(state) holds {out['state']['holds']}; HEAD {out['state']['clone_head'][:8]}; changed "
         f"{out['state']['files_changed_3c379fa_head']}")
    texts = {f: _rd(os.path.join(CLONE, f)).decode('utf-8') for f in (MAIN, LETTER, COVER)}
    # (v) full mode, verifier, entry check: W189p1 task_appendix (PREV = a87e195 as W189 / W190a)
    P1.OUT_NAMES = dict(P1.OUT_NAMES, gen=OUT_NAMES['gen'], verify=OUT_NAMES['verify'], verify_tmp=OUT_NAMES['verify_tmp'])
    P1.PREV_FULL = A87_FULL
    P1.HEAD_FULL = HEAD_FULL
    apx, actx = P1.task_appendix(out_dir, texts)
    N90._VERIFY_TEXT['text'] = _rd(os.path.join(od, OUT_NAMES['verify'])).decode('utf-8')
    out['appendix_w189p1_method'] = apx
    out['verifier'] = N90.verifier_summary(apx)
    _log(f"(v) verify_appE: PASS {out['verifier']['n_pass']} FAIL {out['verifier']['n_fail']} exception "
         f"{out['verifier']['exception']}; entry check {apx['outcome']}")
    # (v) reduced gate: W190a task_generator with BASE_FULL = 3c379fa
    N90.OUT_NAMES = dict(N90.OUT_NAMES, red=OUT_NAMES['red'], verify_tmp=OUT_NAMES['verify_tmp'])
    N90.BASE_FULL, N90.HEAD_FULL = PREV_FULL, HEAD_FULL
    out['generator'], ctl_b, red = N90.task_generator(out_dir, texts, apx, actx)
    w190a_red = _rd(W190A_RED)
    out['generator_vs_w190a'] = {'w190a_output': W190A_RED, 'w190a_sha256': _shab(w190a_red),
                                 'w190a_sha256_is_8e61636b': _shab(w190a_red).startswith(W190A_RED_SHA_PREFIX),
                                 'equal_bytes': red == w190a_red, 'n_lines': out['generator']['reduced']['n_lines'],
                                 'n_lines_is_218': out['generator']['reduced']['n_lines'] == W190A_RED_LINES}
    _log(f"(v) reduced {out['generator']['reduced']['n_lines']} lines, occurrences in HEAD main "
         f"{out['generator']['reduced']['n_occurrences_in_9c_main']} (3c379fa {out['generator']['reduced']['n_occurrences_in_9b_main']}), "
         f"lines equal {out['generator']['reduced']['lines_equal_at_located_position']}; gate "
         f"{out['generator']['gate_bitwise_reduced_in_main']}; = W190a output {out['generator_vs_w190a']['equal_bytes']}")
    ctl = list(P1.controls_appendix(actx)) + ctl_b
    # (i) W187 records first (the declarations read them)
    load_w187()
    _log(f"(i) W187 records: 14 dF recomputed; x0 better {W187['n_x0_better']}, within {W187['n_within']}, neighbour better "
         f"{W187['n_neighbour_better']}; dF {W187['dF_min']:,.2f} .. {W187['dF_max']:,.2f}")
    # (ii) numbers: W189p1 task_numbers on the 3c379fa -> HEAD '+' lines with the W190b declarations
    P1.PREV_FULL = PREV_FULL
    P1.HEAD_FULL = HEAD_FULL
    P1.declarations_w189p1 = declarations_w190b
    out['numbers'], X = P1.task_numbers(texts, actx)
    P1.declarations_w189p1 = ORIG_DECL_W189
    out['new_tokens_vs_3c379fa'] = N90.mark_new(out['numbers'], texts) if 'tokens' in out['numbers'] else None
    _log(f"(ii) {out['numbers'].get('by_rating_total')}; control {out['numbers'].get('negative_control', {}).get('n_refused')}/"
         f"{out['numbers'].get('negative_control', {}).get('n')}; new tokens {len(out['new_tokens_vs_3c379fa'] or [])}")
    out['w187'] = task_w187(texts)
    _log(f"(i) W187 checks hold {out['w187']['holds']}: {[k for k, v in out['w187']['checks'].items() if not v]}; claim / "
         f"pointer {out['w187']['claim_pointer_holds']}: {[k for k, v in out['w187']['claim_pointer_checks'].items() if not v]}")
    out['para_4_1'] = task_41(texts, out['numbers'])
    _log(f"(ii) 4.1 claims hold {out['para_4_1']['holds']}: {[k for k, v in out['para_4_1']['checks'].items() if not v]}")
    out['format_scan'] = scan_format(texts)
    _log(f"(ii) formatting scan: {dict(Counter((r['file'], r['kind']) for r in out['format_scan']))}")
    ctl += controls_w190b(texts)
    nc = out['numbers'].get('negative_control', {})
    ctl.append({'id': f"ii every rated number perturbed ({nc.get('n')})",
                'refused': bool(nc.get('n')) and nc.get('n_refused') == nc.get('n')})
    out['negative_controls'] = ctl
    for c in ctl:
        if c['refused'] is not True:
            _fail(f"negative control not refused: {c['id']}")
    out['guards'] = verify_guards()
    out['check_failures'] = list(P1.CHECK_FAILURES) + list(N90.CHECK_FAILURES) + list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    pj, pm, pn = (os.path.join(od, OUT_NAMES[k]) for k in ('json', 'md', 'man'))
    with open(pj, 'x', encoding='utf-8') as fh:
        GRIO.dump(out, fh, default=GRIO.json_default_item, indent=1, ensure_ascii=False)
        fh.write('\n')
    write_md(out, pm)
    recs = [v['path'] for v in W187['recs'].values()]
    inputs = sorted(set([os.path.join(CLONE, f) for f in FILES]
                        + [GEN_REL, VERIFY_REL, APPX_REL, W190A_RED, W187_JSON, W187_REPORT, W86_RESULTS, SCRIPT_NAME,
                           P1.A89_JSON, P1.W188A_JSON, P1.W188D_MAN] + recs + list(IMPORTED_SCRIPTS)
                        + [v['path'] for v in (out['numbers'].get('inputs') or {}).values()]))
    manifest = {'outputs': {OUT_NAMES[k]: _sha(os.path.join(od, OUT_NAMES[k])) for k in ('json', 'md', 'gen', 'red', 'verify')},
                'out_dir': out_dir, 'clone_head_full': out['state']['clone_head'],
                'git_show_blobs_sha256': {f'{c}:{f}': _shab(showb(c, f)) for c in (PREV_FULL, HEAD_FULL) for f in FILES},
                'inputs': {p: {'sha256': _sha(p), 'committed_clean': committed_clean(p)} for p in inputs}}
    with open(pn, 'x', encoding='utf-8') as fh:
        fh.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')
    _log(f'written {pj}, {pm}, {pn}; check failures {len(out["check_failures"])}; guards ok {out["guards"]["ok"]}; '
         f'pickle {P1.PICKLE_COUNTS}')
    sys.exit(3 if out['check_failures'] else 0)


if __name__ == '__main__':
    main()
