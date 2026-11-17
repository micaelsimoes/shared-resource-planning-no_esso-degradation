"""P5.15 Addendum 66, Planner task W163 -- FIGURE CHECK OF `paragraphs_v4.md` (v3 corrected for the W162 figure check)
AGAINST THE FROZEN TABLES AND NAMED COMMITTED RECORDS. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the W162 script is
IMPORTED and its functions CALLED (`W162.carry_over`, `W162.new_checks`, `W162.nonmanuscript_checks`,
`W162.definition_checks`, `W162.inventory`, `W162.normalise`, `W162.env_now`); W162, W161 and W160 are not edited.
W162 pins the v3 text: this script passes the v4 text to those functions and overrides the one W162 module constant that
depends on the text layout (`W162.MANUSCRIPT_FROM_LINE`: v4 has one more comment line, so the manuscript starts at 11).
Nothing earlier is modified: every output is a new file opened 'x' in a new directory.

SCOPE. As W162: every number in the v4 prose above "## Unchanged from paragraphs_v2.md"; the title line and the Planner's
HTML comment blocks (lines 1-9) are inventoried as NON-MANUSCRIPT text and checked separately. The text is NOT edited.

WHAT IT DOES
  1. Integrity. paragraphs_v4.md (6b5aaadf) and paragraphs_v3.md (cb0a0bc6) are pinned by sha256; the v3 -> v4 line
     difference (difflib, scope only) must touch exactly the declared v3 lines (V3_CHANGED_LINES). Every record read is
     sha-recorded and must be committed clean; the committed W162 JSON must equal its W162 manifest entry.
  2. Runs the W162 checks on v4. Every W162 token assignment (v3 line, text, occurrence) is carried to v4: through the
     difflib line map on unchanged lines, through the declared CHANGED_TOKEN_MAP on changed lines (None = the token is
     gone from v4); a token that maps nowhere FAILS the run. W162 checks whose sentence v4 rewrote are REBUILT on the v4
     wording (REBUILT); every other W162 check keeps W162's evaluation and must reproduce W162's committed status and
     counterpart value (environment-now values: status only).
  3. New v4 checks: N74 (TSO 5 x 10^-4 before the tail, case9_params.json at the parent of the tail-introduction commit
     a51ad9ba, at the W86 campaign commit and now, and the tail-state baselines of the three W86 references), N75 (the
     residual test's "solve successfully (optimal or locally optimal)" against helper_functions.solver_result_succeeded,
     line cited), the rebuilt N60 (DSO 10^-4 = the IPOPT default, read from the pinned binary's --print-options), N61
     (tail 10^-6 on every network holder), N64 (RES slack 7.8 kEUR), N36 / N25 / N4 / N5 / N57 on the v4 wording;
     X5-X7 for the v4 comment line (identifiers, echoed figures, date).
  4. Definition checks: D1-D3 as W162 (verdicts must reproduce); D4, D5, D6 re-judged on the v4 wording (D6: the
     reference evaluations' old certificates x0 132 / unit 112 are residual-rule certificates).
  5. Token inventory: every numeric token in scope is covered by a check or listed unchecked; an unassigned token or a
     stale assignment FAILS the run.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W162 / W161 / W160 imports arm; `pickle.load` / `pickle.loads` blocked for the whole run, every
counter verified at 0. Environment reads (`sysctl`, `sw_vers`, `ipopt --print-options`, which prints the option
documentation and exits) are plain subprocess calls, not solves; git reads are `git show` / `git log`.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w163_paragraphs_v4_check && set -o noclobber && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w163_paragraphs_v4_check.py \\
        > data/SRP1/Results/P515S53/w163_paragraphs_v4_check/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w163_paragraphs_v4_check/w163_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w163_paragraphs_v4_check.py --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (figure MISMATCHES are FINDINGS and never change
the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault.
"""
import argparse
import difflib
import inspect
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W163 paragraphs v4 figure check (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W163: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W163: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w162_paragraphs_v3_check as W162  # noqa: E402 -- arms its guards, imports W161 / W160 (none edited)
import settling_criterion_v5 as SC5  # noqa: E402
import settling_criterion_v6 as SC6  # noqa: E402

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

W161 = W162.W161
W160 = W162.W160
L132 = W162.L132
GUARDS = W160.W157._dedupe((('w163_paragraphs_v4_check', GUARD),) + tuple(W162.GUARDS))

_log = W160._log
_sha = W160._sha
_sha_bytes = W160._sha_bytes
_git = W160._git
_git_blob = W160._git_blob
_jl = W162._jl
_rows = W162._rows
_text = W162._text
at_prec = W162.at_prec

SCRIPT_REL = os.path.basename(__file__)
S53 = W160.S53
EXPORT = W162.EXPORT
P4_REL = os.path.join(EXPORT, 'paragraphs_v4.md')
P4_SHA = '9ed77949b3f86efe62a2b992485074e6116e24b8dc2e979bbfc001613343dea9'
P4_COMMIT = '6b5aaadf'
P3_REL, P3_SHA, P3_COMMIT = W162.P3_REL, W162.P3_SHA, W162.P3_COMMIT
P2_REL, P2_SHA, P2_COMMIT = W162.P2_REL, W162.P2_SHA, W162.P2_COMMIT
FZ_REL, FZ_SHA = W162.FZ_REL, W162.FZ_SHA
W162_SCRIPT = W162.SCRIPT_REL
W162_SCRIPT_COMMIT = 'a8e4d01c'
W162_RESULTS_COMMIT = 'd1f3e186'
W162_JSON = W162.OUT_JSON
W162_MAN = W162.OUT_MAN
MANUSCRIPT_FROM_LINE_V4 = 11       # "## (ii) Certification paragraph"; lines 1-9 are the title and the two comment blocks

# the v3 scope lines that v4 changed (1-based; difflib on the scope must give exactly these)
V3_CHANGED_LINES = (1, 3, 14, 17, 29, 30, 43, 44, 87, 90, 91, 114, 121, 122, 123)
# W162 token assignments on those lines -> their v4 position (None: the token is not in v4)
CHANGED_TOKEN_MAP = {
    (1, '6', 0): (1, '6', 0),                       # title "Step 6" (unchecked, non-manuscript)
    (1, '66', 0): None,                             # v3 title "(Addendum 66)"; the v4 title has no 66 (X1)
    (3, '2026-10-07', 0): (4, '2026-10-07', 0),     # v3 header date, now line 4 (X2)
    (14, '10⁻⁴', 0): (15, '10⁻⁴', 0),               # eps_rel (N3)
    (17, '15', 0): (18, '15', 0), (17, '21', 0): (18, '21', 0),   # N4 / N5 (rebuilt)
    (29, '5', 0): (30, '5', 0),                     # list item 5 (unchecked enumerator)
    (30, 'four', 0): (31, 'four', 0), (30, '10', 0): (31, '10', 0),  # N13 / N14
    (44, '0.9', 0): (46, '0.9', 0),                 # N25 (rebuilt)
    (87, '62', 0): (89, '62', 0),                   # N35
    (87, '10⁻⁵', 0): None,                          # "(≈ 10⁻⁵ of renewable energy)" dropped in v4 (was NO SOURCE FOUND)
    (90, '3', 0): (92, '3', 0),                     # N36 (rebuilt)
    (91, '44', 0): (93, '44', 0),                   # N37
    (114, '32', 0): (116, '32', 0),                 # N57 (rebuilt)
    (121, '10⁻⁴', 0): (125, '10⁻⁴', 0),             # N60 (rebuilt: the distribution subproblems)
    (121, '10⁻⁶', 0): (125, '10⁻⁶', 0),             # N61 (rebuilt)
    (122, 'three', 0): (124, 'three', 0),           # N62
    (122, '1.1', 0): (124, '1.1', 0), (122, '10⁻⁶', 0): (124, '10⁻⁶', 0),   # N63
    (123, '10⁻⁵', 0): (126, '10⁻⁵', 0),             # N34
    (123, '7.9', 0): (126, '7.8', 0),               # N64 (rebuilt: v4 writes 7.8)
    (123, '1.2', 0): (126, '1.2', 0), (123, '10⁻⁵', 1): (126, '10⁻⁵', 1),   # N65
}
REBUILT = ('N4', 'N5', 'N25', 'N36', 'N57', 'N60', 'N61', 'N64')

W86_DIR = os.path.join(S53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert')
W86_SPEC = os.path.join(W86_DIR, 'campaign_spec_s53_w86_tail_recert_ddd6cd44.json')
W86_EVALS = {'x0': os.path.join(W86_DIR, 'evals', '5cfe69a615ae3708_x0'),
             'n7_4h_e1': os.path.join(W86_DIR, 'evals', 'ca8927e75d628bd1_n7_4h_e1'),
             'c_star': os.path.join(W86_DIR, 'evals', '96c5aa50cc229cc1_c_star')}
TAIL_COMMIT = 'a51ad9ba'           # "P5.15 Addendum 46 ruling 7 W83 step 1: convergence-depth tight tail (DEFAULT OFF)"
CASE9_REL = W162.CASE9_PARAMS
DSO_CASES = {'DSO5': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_params.json'),
             'DSO7': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_params.json'),
             'DSO9': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_params.json')}
IPOPT_BIN = '/usr/local/bin/ipopt'

OUT_DIR = os.path.join(S53, 'w163_paragraphs_v4_check')
OUT_JSON = os.path.join(OUT_DIR, 'w163_paragraphs_v4_figure_check.json')
OUT_MD = os.path.join(OUT_DIR, 'w163_paragraphs_v4_figure_check.md')
OUT_MAN = os.path.join(OUT_DIR, 'manifest_sha256.json')
OUT_LOG = os.path.join(OUT_DIR, 'launch.log')
OUT_TYPING_JSON = os.path.join(OUT_DIR, 'w163_bool_typing_test.json')
OUT_TYPING_LOG = os.path.join(OUT_DIR, 'w163_bool_typing_test.log')
OUT_POST = os.path.join(OUT_DIR, 'manifest_post_run_sha256.json')

TAU = SC6.TAU


def _norm(x):
    return json.loads(GRIO.dumps(x, sort_keys=True))


# ======================================================================================================================
#  1. the v3 -> v4 line map and the token remap
# ======================================================================================================================
def line_map(l3, e3, l4, e4):
    sm = difflib.SequenceMatcher(a=l3[:e3], b=l4[:e4], autojunk=False)
    m, changed, ops = {}, [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                m[i1 + k + 1] = j1 + k + 1
        else:
            changed.extend(range(i1 + 1, i2 + 1))
            ops.append([tag, [i1 + 1, i2], [j1 + 1, j2]])
    same_text = all(l3[a - 1] == l4[b - 1] for a, b in m.items())
    return m, changed, ops, same_text


class Remap:
    def __init__(self, lmap):
        self.lmap = lmap
        self.unmapped, self.dropped, self.via_changed = [], [], []

    def __call__(self, cid, tokens):
        out = []
        for t in tokens:
            k = (t[0], t[1], t[2])
            if k in CHANGED_TOKEN_MAP:
                n = CHANGED_TOKEN_MAP[k]
                if n is None:
                    self.dropped.append([cid, *k])
                    continue
                self.via_changed.append([cid, *k, *n])
                out.append(n)
            elif k[0] in self.lmap:
                out.append((self.lmap[k[0]], k[1], k[2]))
            else:
                self.unmapped.append([cid, *k])
        return [list(x) for x in out]


V4_KEYS = {'written_v3': 'written_v4', 'fragment_v3': 'fragment_v4', 'fragment_found_v3': 'fragment_found_v4'}
PLACEMENT_V4 = {'verbatim in v3': 'verbatim in v4',
                'v3 rewrote the sentence (carried to the v3 fragment)':
                    'rewritten in v3, unchanged in v4 (carried to the v4 fragment)',
                'not in the v3 prose': 'not in the v4 prose'}


def as_v4(c, remap, handling):
    out = {V4_KEYS.get(k, k): v for k, v in c.items()}
    if 'placement' in out:
        out['placement'] = PLACEMENT_V4[out['placement']]
    out['tokens'] = remap(c['id'], c.get('tokens') or [])
    out['w163_handling'] = handling
    return out


def _chk4(cid, tokens, written, counterpart, value, shown, status, kind, fragment, note=None, origin=None):
    return {'id': cid, 'origin': origin or 'W163 new check (v4)', 'tokens': [list(t) for t in tokens],
            'written_v4': written, 'counterpart': counterpart, 'counterpart_kind': kind, 'value': value,
            'value_at_written_precision': shown, 'status': status, 'fragment_v4': fragment, 'note': note,
            'w163_handling': 'new in W163' if origin is None else 'rebuilt by W163 on the v4 wording'}


# ======================================================================================================================
#  2. the v4 records: tail tolerances, the residual-test success clause, the old certificates
# ======================================================================================================================
def ipopt_print_options():
    r = subprocess.run([IPOPT_BIN, '--print-options'], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       cwd=REPO)
    m = re.search(r'^compl_inf_tol\s+\S+\s+<\s+\(\s*([0-9.eE+-]+)\)\s+<', r.stdout, re.M)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith('compl_inf_tol ')), None)
    return {'command': f'{IPOPT_BIN} --print-options (prints the option documentation and exits; not a solve)',
            'returncode': r.returncode, 'stdout_sha256': _sha_bytes(r.stdout.encode('utf-8')),
            'stdout_bytes': len(r.stdout.encode('utf-8')), 'compl_inf_tol_line': line,
            'compl_inf_tol_default': float(m.group(1)) if m else None, 'binary_sha256': _sha_file_abs(IPOPT_BIN)}


def _sha_file_abs(path):
    with open(path, 'rb') as h:
        return _sha_bytes(h.read())


def _blob_json(commit, rel):
    b = _git_blob(commit, rel)
    return (json.loads(b.decode('utf-8')), _sha_bytes(b)) if b is not None else (None, None)


def tail_records(rec):
    """The complementarity tolerance before and in the tail, per subproblem, from the case files at named commits, the
    IPOPT default and the three W86 references' tail states."""
    spec86 = rec['w86_spec']
    head86 = spec86['git_head']
    parent = _git('rev-parse', f'{TAIL_COMMIT}^')
    tail_full = _git('rev-parse', TAIL_COMMIT)
    tail_subject = _git('log', '-1', '--format=%s', TAIL_COMMIT)
    is_anc = subprocess.run(['git', 'merge-base', '--is-ancestor', tail_full, head86], cwd=REPO).returncode == 0
    reads = {}
    for label, commit in (('parent_of_tail_commit', parent), ('w86_campaign_git_head', head86), ('HEAD', 'HEAD')):
        c9, c9sha = _blob_json(commit, CASE9_REL)
        dso = {}
        for lab, rel in DSO_CASES.items():
            d, dsha = _blob_json(commit, rel)
            dso[lab] = {'path': rel, 'blob_sha256': dsha,
                        'has_compl_inf_tol_anywhere': d is not None and 'compl_inf_tol' in json.dumps(d)}
        reads[label] = {'commit': _git('rev-parse', commit), 'case9_blob_sha256': c9sha,
                        'case9_solver_options_compl_inf_tol': (c9 or {}).get('solver', {}).get('options', {}).get(
                            'compl_inf_tol'),
                        'case9_recovery_options_has_compl_inf_tol': 'compl_inf_tol' in json.dumps(
                            (c9 or {}).get('solver', {}).get('recovery_options', {})),
                        'dso_case_files': dso}
    reads['working_tree_case9_sha256'] = _sha(CASE9_REL)
    states = {}
    for nm, ed in W86_EVALS.items():
        st = rec['w86_tail_state'][nm]
        act = [p for p in st['per_cycle'] if p['active']]
        inact = [p for p in st['per_cycle'] if not p['active']]
        states[nm] = {
            'enabled': st['enabled'], 'compl_inf_tol_tail': st['compl_inf_tol_tail'],
            'baseline': {k: {'has_key': v['has_key'], 'value': v['value'], 'network': v['network']}
                         for k, v in st['baseline'].items()},
            'holders': sorted(st['baseline']),
            'n_cycles': len(st['per_cycle']), 'n_active': len(act),
            'every_active_cycle_all_holders_at_tail': bool(act) and all(
                h['after'] == st['compl_inf_tol_tail'] for p in act for h in p['holders'].values()),
            'every_inactive_cycle_holders_at_baseline': all(
                h['after'] == st['baseline'][lab]['value'] for p in inact for lab, h in p['holders'].items())}
    net = rec['code']['network.py']
    adm = rec['code']['admm_parameters.py']
    po = rec['ipopt_print_options']
    return {'tail_commit': {'commit': tail_full, 'subject': tail_subject, 'parent': parent,
                            'ancestor_of_w86_campaign_head': is_anc},
            'case_file_reads': reads, 'w86_tail_states': states,
            'network_py_default': re.search(r'^IPOPT_DEFAULT_COMPL_INF_TOL = ([0-9.eE+-]+)$', net, re.M).group(1)
            if re.search(r'^IPOPT_DEFAULT_COMPL_INF_TOL = ([0-9.eE+-]+)$', net, re.M) else None,
            'network_py_comment': "IPOPT's default 1e-4" in net,
            'admm_parameters_tail_default': re.findall(r"'compl_inf_tol': ([0-9.eE+-]+),", adm),
            'ipopt_print_options': po,
            'ipopt_binary_matches_w159_pin': po['binary_sha256'] == rec['w159']['c']['c3']['ipopt']['sha256'],
            'tail_holders_code': "holders = [('TSO', planning_problem.transmission_network)]" in rec['code'][
                'shared_resources_planning.py'] and "holders.append((f'DSO{node_id}', distribution_network))" in
            rec['code']['shared_resources_planning.py']}


def success_clause_records(rec):
    hf = rec['code']['helper_functions.py'].split('\n')
    srp = rec['code']['shared_resources_planning.py'].split('\n')
    i_def = next((i for i, ln in enumerate(hf) if ln.startswith('def solver_result_succeeded(')), None)
    body = hf[i_def:i_def + 13] if i_def is not None else []
    acc = re.findall(r'po\.TerminationCondition\.(\w+),', '\n'.join(body))
    status_ok = any('result.solver.status == po.SolverStatus.ok' in ln for ln in body)

    def line_of(lines, pat):
        return next((i + 1 for i, ln in enumerate(lines) if pat in ln), None)
    cites = {'helper_functions.py def solver_result_succeeded': (i_def + 1) if i_def is not None else None,
             'helper_functions.py accepted_termination_conditions': line_of(hf, 'accepted_termination_conditions = {'),
             'helper_functions.py status ok': line_of(hf, 'result.solver.status == po.SolverStatus.ok'),
             'shared_resources_planning.py def _admm_local_solves_succeeded':
                 line_of(srp, 'def _admm_local_solves_succeeded('),
             'shared_resources_planning.py local_solves_ok =':
                 line_of(srp, 'local_solves_ok = _admm_local_solves_succeeded(planning_problem, results)'),
             'shared_resources_planning.py cycle_convergence =':
                 line_of(srp, 'cycle_convergence = boyd_all_pass and local_solves_ok')}
    i_ls = cites['shared_resources_planning.py def _admm_local_solves_succeeded']
    ls_body = '\n'.join(srp[i_ls - 1:i_ls + 12]) if i_ls else ''
    covers = {f: (f"results['{f}']" in ls_body) for f in ('tso', 'dso', 'esso')}
    # pyomo's .sol reader (installed environment, not a committed file): the IPOPT solve_result_num classes
    import pyomo
    solp = os.path.join(os.path.dirname(pyomo.__file__), 'opt', 'plugins', 'sol.py')
    sol = open(solp, encoding='utf-8').read().split('\n')
    seg = {}
    for i, ln in enumerate(sol):
        for lab, pat in (('0-99', '(objno[1] >= 0) and (objno[1] <= 99)'),
                         ('100-199', '(objno[1] >= 100) and (objno[1] <= 199)')):
            if pat in ln:
                blk = '\n'.join(sol[i:i + 6])
                seg[lab] = {'line': i + 1,
                            'termination': re.search(r'TerminationCondition\.(\w+)', blk).group(1),
                            'status': re.search(r'SolverStatus\.(\w+)', blk).group(1)}
    # the records: block-rounds whose FINAL attempt ended at IPOPT's acceptable level, and local_solves_ok that cycle
    order = {'primary': 0, 'recovery': 1, 'recovery_tier2': 2}
    emp = {}
    for nm in W86_EVALS:
        last = {}
        for r in rec['w86_solve_records'][nm]:
            key = (r['round'], r['network'], r['year'], r['day'])
            if key not in last or order[r['attempt']] >= order[last[key]['attempt']]:
                last[key] = r
        pc = rec['w86_rows'][nm]
        tally = {}
        for (rd, *_), r in last.items():
            if rd in pc:
                t = f"{r['exit']} | local_solves_ok {pc[rd].get('local_solves_ok')}"
                tally[t] = tally.get(t, 0) + 1
        emp[nm] = tally
    acc_rounds = sum(v for nm in emp for k, v in emp[nm].items() if k.startswith('Solved To Acceptable Level.'))
    acc_ok = sum(v for nm in emp for k, v in emp[nm].items()
                 if k.startswith('Solved To Acceptable Level.') and k.endswith('local_solves_ok True'))
    return {'cites': cites, 'accepted_termination_conditions': acc, 'requires_status_ok': status_ok,
            'local_solves_cover': covers, 'pyomo_sol_reader': {'path': solp, 'sha256': _sha_file_abs(solp),
                                                                'classes': seg},
            'w86_final_attempt_exit_by_local_solves_ok': emp,
            'acceptable_final_attempt_block_rounds': acc_rounds, 'of_which_local_solves_ok_true': acc_ok}


def old_certificates(rec):
    out = {}
    for nm in ('x0', 'n7_4h_e1'):
        decl = rec['w101_g13'][nm]
        er = rec['w86_eval_record'][nm]
        rows = rec['w86_rows'][nm]
        rep = rec['w101_summary']['reports'][nm]
        n = er['certification_cycle']
        run = er['stopped_by_trajectory']['stop_run_cycles']
        req = er['required_consecutive_cycles']
        out[nm] = {
            'w101_replay_reference': decl['replay_reference'], 'w101_hold_after_cycle': decl['hold_after_cycle'],
            'w86_eval_dir': W86_EVALS[nm], 'replay_reference_is_this_record':
                decl['replay_reference']['per_cycle_record'] == os.path.join(W86_EVALS[nm], 'per_cycle_record.jsonl')
                and decl['replay_reference']['sha256'] == _sha(os.path.join(W86_EVALS[nm], 'per_cycle_record.jsonl')),
            'status': er['status'], 'certification_cycle': n, 'cycles_run': er['cycles_run'],
            'required_consecutive_cycles': req, 'stopped_by': er['stopped_by_trajectory']['stopped_by'],
            'converged_at_cycle': er['stopped_by_trajectory']['converged_at_cycle'], 'stop_run_cycles': run,
            'per_cycle_streak': {c: [rows[c].get('boyd_all_pass'), rows[c].get('local_solves_ok'),
                                     rows[c].get('cycle_convergence'), rows[c].get('consecutive_converged_cycles')]
                                 for c in range(run[0] - 1, n + 1)},
            'certified_cost': er['certified_cost'], 'w101_Q_cert_old_Q_N': rep['Q_cert_old_Q_N'], 'w101_N': rep['N'],
            'w101_k0': rep['k0'], 'w101_s_signed': rep['s_signed']}
        o = out[nm]
        o['ok'] = bool(o['replay_reference_is_this_record'] and o['status'] == 'certified' and
                       o['cycles_run'] == n == o['w101_hold_after_cycle'] == o['w101_N'] ==
                       decl['replay_reference']['n_cycles'] and o['stopped_by'] == 'boyd' and
                       run == list(range(n - req + 1, n + 1)) and o['converged_at_cycle'] == run[0] == o['w101_k0'] and
                       all(rows[c]['cycle_convergence'] is True and rows[c]['consecutive_converged_cycles'] == c - run[0] + 1
                           for c in run) and rows[run[0] - 1]['cycle_convergence'] is False and
                       o['certified_cost'] == o['w101_Q_cert_old_Q_N'])
    spec_txt = json.dumps(rec['w86_spec'])
    return out, {'w86_spec_mentions_settling': 'settling' in spec_txt,
                 'w86_spec_required_consecutive_cycles': rec['w86_spec']['required_consecutive_cycles']}


# ======================================================================================================================
#  3. the rebuilt and new checks (v4 wording)
# ======================================================================================================================
def v4_checks(fz, rec, w162_by_id, envnow):
    t = fz['tables']
    cells = t['cells']
    w101 = rec['w101_summary']['reports']
    beside = rec['reference_beside']
    oc, ocm = rec['old_certificates']
    R = {}
    O = 'W162 check rebuilt on the v4 wording'
    # ---- N4 / N5 -------------------------------------------------------------------------------------------------
    frag = 'moved by a further 15–21 k€ after the residual-based stopping rule had certified them'
    sx, su = w101['x0']['s_signed'], w101['n7_4h_e1']['s_signed']
    okanchor = all(v['ok'] for v in oc.values()) and not ocm['w86_spec_mentions_settling']
    note = ('s = Q(k*) − Q(N), N = the old certificate (x0 132, unit 112): W86 records 5cfe69a6 / ca8927e7, status '
            'certified, stopped_by boyd, 10 consecutive residual passes (123-132 / 103-112), no settling rule in the '
            f'W86 spec (D6: anchor {"consistent" if okanchor else "NOT CONFIRMED"}). "The reference evaluations" = '
            'x0 (d110bd1a) and the unit (3f084f2f); C* (uncertified, s = −4,378.88) is outside the range')
    for cid, w, v, nm in (('N4', '15', sx, 'x0'), ('N5', '21', su, 'n7_4h_e1')):
        sh, st = at_prec(w, v / 1000.0)
        R[cid] = _chk4(cid, [(18, w, 0)], w, f'W101 three-reference summary (62bdeafe) reports.{nm}.s_signed (k€), '
                       'measured from the old residual-rule certificate N', v, sh, st if okanchor else 'MISMATCH',
                       'named record', frag, note, origin=O)
    # ---- N25 ------------------------------------------------------------------------------------------------------
    o25 = w162_by_id['N25']
    ov = sorted(abs(v) for v in rec['postcert']['over_tau'].values())
    R['N25'] = _chk4('N25', [(46, '0.9', 0)], '0.9', o25['counterpart'], o25['value'],
                     o25['value_at_written_precision'], o25['status'], o25['counterpart_kind'],
                     'Evaluations continued past a certificate under this rule moved by at most 0.9 τ',
                     'a bound over the runs continued past a SETTLING-RULE certificate (the W162 computation, '
                     f"unchanged): |movement| / τ = {' / '.join(f'{x:.3f}' for x in ov)}. v4 now scopes the sentence to "
                     '"a certificate under this rule", which is the computed scope; runs continued past an old '
                     'residual-rule certificate are outside it (the SRP1 references, s = '
                     f"{abs(sx) / TAU:.2f} τ and {abs(su) / TAU:.2f} τ; the 3 × 3 x = 0, 0.99 τ per W162's note)",
                     origin=O)
    # ---- N36 ------------------------------------------------------------------------------------------------------
    note36 = rec['code'][W162.NOTE_5456]
    txt_ok = '≈ 3 τ left whenever the decay' in note36 and '44-cycle window' in note36 and "C\\*'s is ≈ 102" in note36
    tasks = rec['code']['TASKS.md']
    hl_txt = '0.9932/cycle' in tasks and 'half-life 102 > L_MONO 44' in tasks and '≈ 3 τ gross drift left' in tasks
    hl = math.log(0.5) / math.log(0.9932)
    c = 2.0 ** (-1.0 / 102.0)
    bound = (1.0 / (1.0 - c)) / 44.0
    sh36, st36 = at_prec('3', bound)
    R['N36'] = _chk4('N36', [(92, '3', 0)], '3', f'{W162.NOTE_5456} (ce96d492): "≈ 3 τ left whenever the decay half-life '
                     'exceeds its 44-cycle window (C*\'s is ≈ 102)"; TASKS.md W110 line: |dQ| decay 0.9932/cycle '
                     '(validated out of sample), "half-life 102 > L_MONO 44", "≈ 3 τ gross drift left"; derived: a '
                     'geometric tail with half-life 102 and |last step| × 44 ≤ τ leaves ≤ τ / (44 (1 − 2^(−1/102)))',
                     {'record_text_found': txt_ok, 'tasks_text_found': hl_txt, 'half_life_from_0_9932': hl,
                      'derived_bound_over_tau': bound}, f'{sh36} ({bound:.2f})',
                     st36 if txt_ok and hl_txt else 'MISMATCH', 'record text + derived estimate',
                     '≈ 3 τ is estimated for the reference corner plan\'s measured half-life at the earlier window L = 44',
                     f'v4 calls it an estimate, as the record does; the half-life is a fitted (measured) decay: '
                     f'ln 0.5 / ln 0.9932 = {hl:.1f} cycles ≈ 102; the derived bound {bound:.2f} τ rounds to 3',
                     origin=O)
    # ---- N57 ------------------------------------------------------------------------------------------------------
    mem_now = envnow['hw_memsize']
    mem_v6 = rec['v6']['memory_preflight']['measured_at_freeze_non_gating']['hw_memsize_bytes']
    R['N57'] = _chk4('N57', [(116, '32', 0)], '32 GiB', 'sysctl hw.memsize now; v6 spec memory_preflight '
                     'hw_memsize_bytes', {'now': mem_now, 'v6_spec': mem_v6}, f'{mem_now / 2 ** 30:g} GiB',
                     'match' if mem_now == mem_v6 == 32 * 2 ** 30 else 'MISMATCH', 'spec record + environment now',
                     'with 32 GiB of memory', '34,359,738,368 bytes = 32 GiB exactly', origin=O)
    # ---- the tail: N60 (DSO), N74 (TSO), N61 (tail value) ------------------------------------------------------------
    tr = rec['tail']
    po = tr['ipopt_print_options']
    cr = tr['case_file_reads']
    dso_none = all(not d['has_compl_inf_tol_anywhere'] for r in (cr['parent_of_tail_commit'],
                   cr['w86_campaign_git_head'], cr['HEAD']) for d in r['dso_case_files'].values())
    dso_base = all(not v['has_key'] and v['value'] is None for s in tr['w86_tail_states'].values()
                   for k, v in s['baseline'].items() if k.startswith('DSO'))
    ok60 = po['returncode'] == 0 and po['compl_inf_tol_default'] == 1e-4 and tr['ipopt_binary_matches_w159_pin'] and \
        tr['network_py_default'] == '1e-4' and dso_none and dso_base
    frag_t = 'The complementarity tolerance went to 10⁻⁶ from 10⁻⁴ in the distribution subproblems and from 5 × 10⁻⁴ ' \
             'in the transmission subproblem'
    R['N60'] = _chk4('N60', [(125, '10⁻⁴', 0)], '10⁻⁴ (distribution subproblems)',
                     f'IPOPT default compl_inf_tol: `{IPOPT_BIN} --print-options` (binary sha256 = the W159 / v6 pin); '
                     'network.py IPOPT_DEFAULT_COMPL_INF_TOL; no DSO case file (case33_1/2/3_params.json) sets '
                     f'compl_inf_tol at {TAIL_COMMIT}^, at the W86 campaign commit or now; W86 tail-state baselines '
                     'DSO5/7/9 has_key False (x0, unit, C*)',
                     {'ipopt_print_options_default': po['compl_inf_tol_default'],
                      'ipopt_print_options_line': po['compl_inf_tol_line'],
                      'network_py_default': tr['network_py_default'], 'dso_case_files_set_it': not dso_none,
                      'w86_dso_baselines_unset': dso_base},
                     f"{po['compl_inf_tol_default']:g}" if po['compl_inf_tol_default'] is not None else None,
                     'match' if ok60 else 'MISMATCH', 'solver default + case files + named record', frag_t,
                     'the distribution solves pass no compl_inf_tol, so IPOPT\'s default 1e-4 is in force before the tail',
                     origin=O)
    tso_vals = {k: cr[k]['case9_solver_options_compl_inf_tol'] for k in
                ('parent_of_tail_commit', 'w86_campaign_git_head', 'HEAD')}
    tso_base = {nm: s['baseline']['TSO']['value'] for nm, s in tr['w86_tail_states'].items()}
    ok74 = all(v == 5e-4 for v in tso_vals.values()) and all(v == 5e-4 for v in tso_base.values()) and \
        all(s['baseline']['TSO']['has_key'] for s in tr['w86_tail_states'].values()) and \
        not any(cr[k]['case9_recovery_options_has_compl_inf_tol'] for k in tso_vals) and \
        tr['tail_commit']['ancestor_of_w86_campaign_head']
    R['N74'] = _chk4('N74', [(125, '5', 0), (125, '10⁻⁴', 1)], '5 × 10⁻⁴ (transmission subproblem)',
                     f'{CASE9_REL} solver.options.compl_inf_tol read at {tr["tail_commit"]["parent"][:8]} (the parent of '
                     f'{TAIL_COMMIT}, the commit that introduced the tight tail: "{tr["tail_commit"]["subject"][:90]}"), '
                     'at the W86 campaign commit (spec git_head) and at HEAD; W86 tail-state baselines TSO (x0, unit, C*)',
                     {'case9_by_commit': tso_vals, 'w86_tso_baseline': tso_base,
                      'case9_blob_sha256': {k: cr[k]['case9_blob_sha256'] for k in tso_vals}},
                     ', '.join(f'{k} {v:g}' for k, v in tso_vals.items()), 'match' if ok74 else 'MISMATCH',
                     'case file at named commits + named record', frag_t,
                     'the case file has carried 5e-4 since before the tail was introduced (the same blob at all three '
                     'commits); the tail replaces it with 1e-6 on tail cycles and restores it otherwise')
    v6t = rec['v6']['inputs_in_force_now']['configuration_now']['convergence_depth_tail']['compl_inf_tol']
    st86 = tr['w86_tail_states']
    ok61 = v6t == 1e-6 and all(s['compl_inf_tol_tail'] == 1e-6 and s['enabled'] is True and
                               s['every_active_cycle_all_holders_at_tail'] and
                               s['every_inactive_cycle_holders_at_baseline'] and
                               s['holders'] == ['DSO5', 'DSO7', 'DSO9', 'TSO'] for s in st86.values()) and \
        tr['admm_parameters_tail_default'] == ['1e-6'] and tr['tail_holders_code']
    R['N61'] = _chk4('N61', [(125, '10⁻⁶', 0)], '10⁻⁶ (both)', 'v6 spec convergence_depth_tail.compl_inf_tol; '
                     'admm_parameters.py tail default; W86 tail states: every holder (TSO, DSO5/7/9; never the ESSO) at '
                     'the tail value on every tail-active cycle and at its baseline otherwise',
                     {'v6_spec': v6t, 'admm_parameters_default': tr['admm_parameters_tail_default'],
                      'w86': {nm: {k: s[k] for k in ('compl_inf_tol_tail', 'n_cycles', 'n_active',
                                                     'every_active_cycle_all_holders_at_tail',
                                                     'every_inactive_cycle_holders_at_baseline', 'holders')}
                              for nm, s in st86.items()}},
                     f'{v6t:g}', 'match' if ok61 else 'MISMATCH', 'spec constant + code + named record', frag_t,
                     origin=O)
    # ---- N64 ------------------------------------------------------------------------------------------------------
    w109 = rec['w109']
    pr = w109['tso']['priced_neg_part_eur_block_weighted'] + w109['dso']['priced_neg_part_eur_block_weighted']
    sh78, st78 = at_prec('7.8', abs(pr) / 1000.0)
    R['N64'] = _chk4('N64', [(126, '7.8', 0)], '≈ 7.8', 'W109 (f6e3533f) tso + dso priced_neg_part_eur_block_weighted '
                     '(k€)', pr, sh78, st78, 'named record', 'lowers the objective by ≈ 7.8 k€ first-order',
                     f'{pr:,.2f} € = {abs(pr) / 1000:.4f} k€', origin=O)
    # ---- N75: the residual test's success clause -------------------------------------------------------------------
    sc = rec['success']
    ok75 = sc['accepted_termination_conditions'] == ['optimal', 'locallyOptimal', 'globallyOptimal'] and \
        sc['requires_status_ok'] and all(sc['local_solves_cover'].values()) and \
        all(v is not None for v in sc['cites'].values())
    ci = sc['cites']
    R['N75'] = _chk4('N75', [], 'solve successfully (optimal or locally optimal)',
                     f"helper_functions.solver_result_succeeded (helper_functions.py line "
                     f"{ci['helper_functions.py def solver_result_succeeded']}: status ok "
                     f"(line {ci['helper_functions.py status ok']}) and termination in "
                     f"{{optimal, locallyOptimal, globallyOptimal}} (line "
                     f"{ci['helper_functions.py accepted_termination_conditions']})), called for every TSO, DSO and "
                     f"ESSO result by _admm_local_solves_succeeded (shared_resources_planning.py line "
                     f"{ci['shared_resources_planning.py def _admm_local_solves_succeeded']}) -> local_solves_ok (line "
                     f"{ci['shared_resources_planning.py local_solves_ok =']}) -> cycle_convergence = boyd_all_pass and "
                     f"local_solves_ok (line {ci['shared_resources_planning.py cycle_convergence =']})",
                     sc, 'status ok and ' + '/'.join(sc['accepted_termination_conditions']),
                     'match' if ok75 else 'MISMATCH', 'code (cited lines)',
                     'Every local NLP must also solve successfully (optimal or locally optimal) in the same cycle',
                     'the code also accepts globallyOptimal, which the prose omits. The terms are Pyomo termination '
                     'conditions: Pyomo\'s .sol reader maps IPOPT solve_result_num 0-99 to optimal with status ok '
                     f"(sol.py line {sc['pyomo_sol_reader']['classes'].get('0-99', {}).get('line')}), and in the "
                     f"records every final attempt ending \"Solved To Acceptable Level.\" counted as successful "
                     f"({sc['of_which_local_solves_ok_true']} of {sc['acceptable_final_attempt_block_rounds']} "
                     'block-rounds in the W86 references, local_solves_ok True): "optimal" here includes IPOPT\'s '
                     'acceptable-level exit')
    return R


def x_v4_checks(rec, raw4, l4, inscope):
    X = []
    ids = {'62d26e27': P3_SHA.startswith('62d26e27'),
           'cb0a0bc6': _git('rev-parse', 'cb0a0bc6').startswith(P3_COMMIT) and
           _git('log', '-1', '--format=%h', '--abbrev=8', '--', P3_REL).startswith(P3_COMMIT[:7]),
           'd1f3e186': W162_JSON in _git('show', '--name-only', '--format=', W162_RESULTS_COMMIT).split('\n')}
    X.append(_chk4('X5', [], 'paragraphs_v3.md (sha256 62d26e27..., cb0a0bc6); the W162 figure check (d1f3e186)',
                   'sha256 of paragraphs_v3.md; its last commit; the commit that added the W162 JSON', ids,
                   json.dumps(ids), 'match' if all(ids.values()) else 'MISMATCH', 'repository',
                   'identifiers in the v4 comment line'))
    # the figures the v4 comment echoes: each equals the figure written on the corrected manuscript line
    echo = {'7.8': (126, '7.8'), '15': (18, '15'), '21': (18, '21'), '5': (30, '5'), '3': (92, '3'),
            '0.9': (46, '0.9'), '32': (116, '32')}
    found = {k: (ln, tx, 0) in inscope for k, (ln, tx) in echo.items()}
    ctx = {'7.8': 'RES slack 7.8 k EUR' in l4[2], '15': '15-21 k EUR anchor' in l4[2], '21': '15-21 k EUR anchor' in l4[2],
           '5': 'clause 5 scoped to the window' in l4[2] and l4[29].startswith('> 5. every cycle in the window'),
           '3': 'the 3 tau figure an estimate' in l4[2], '0.9': 'the 0.9 tau bound scoped' in l4[2],
           '32': '32 GiB' in l4[2]}
    X.append(_chk4('X6', [(3, k, 0) for k in echo], ', '.join(echo), 'the corrected manuscript tokens (line, text): '
                   + ', '.join(f'{k} -> line {ln}' for k, (ln, _t) in echo.items()),
                   {'manuscript_token_present': found, 'comment_context': ctx}, 'all present' if all(found.values())
                   and all(ctx.values()) else 'not all present', 'match' if all(found.values()) and all(ctx.values())
                   else 'MISMATCH', 'the v4 text itself', 'the v4 comment line',
                   'each echoed figure is checked where the manuscript writes it (N64, N4, N5, list item 5, N36, N25, N57)'))
    brief = rec['code'][W162.BRIEF]
    hdrs = re.findall(r'^# Addendum (\d+) — .*\((\d{4}-\d{2}-\d{2})\)$', brief, re.M)
    last = max(hdrs, key=lambda h: int(h[0])) if hdrs else None
    p4date = _git('log', '-1', '--format=%ad', '--date=short', '--', P4_REL)
    X.append(_chk4('X7', [(3, '2026-10-07', 0)], '2026-10-07', 'the latest addendum header of the brief; '
                   'paragraphs_v4.md commit date', {'latest_addendum': last, 'commit_date': p4date},
                   last[1] if last else None, 'match' if last and last[0] == '66' and last[1] == '2026-10-07'
                   else 'MISMATCH', 'brief', 'v4, Planner, 2026-10-07',
                   f'equals the Addendum 66 date (the latest addendum); the file was committed {p4date}'))
    return X


# token-only classifications for v4 (W162's, carried; the dropped "10⁻⁵ of renewable energy" is gone)
UNCHECKED_REASON_V4 = {(122, 'Two', 0): ('structural', '"Two effects": the count of the two bullets that follow (each '
                                                       'checked: N60-N65, N74)')}


# ======================================================================================================================
#  4. definition checks on the v4 wording
# ======================================================================================================================
def d_v4(dch162, rec, norm4):
    by = {d['id']: d for d in dch162}
    D = []
    for i in ('D1', 'D2', 'D3'):
        d = dict(by[i])
        d['w163_handling'] = 'W162 definition check, re-run (claim text unchanged in v4)'
        D.append(d)
    sc = rec['success']
    t4 = 'Every local NLP must also solve successfully (optimal or locally optimal) in the same cycle.'
    D.append({'id': 'D4', 'claim': t4, 'counterpart': 'N75 (helper_functions.solver_result_succeeded and its callers)',
              'evidence': {'v4_text_found': t4 in norm4, 'w162_evidence': by['D4']['evidence'], 'w163': sc},
              'verdict': 'consistent: the code requires a successful solve (status ok, termination optimal / '
                         'locallyOptimal / globallyOptimal), as v4 now says' if t4 in norm4 and
                         sc['requires_status_ok'] else 'not confirmed',
              'note': 'globallyOptimal is accepted by the code but not named in the prose; IPOPT acceptable-level exits '
                      'are "optimal" in Pyomo\'s classification and counted successful (N75)',
              'w162_verdict_on_v3': by['D4']['verdict'], 'w163_handling': 're-judged on the v4 wording'})
    src5 = inspect.getsource(SC5.SettlingRuleV5._non_clean_in) + inspect.getsource(SC6.SettlingRuleV6.evaluate)
    oow = rec['w153c']['result']['totals']['certificates_with_a_non_clean_cycle_read_out_of_window']
    oow_br = {c['cell']: c['branch'] for c in rec['w153c']['result']['cells'] if c['cell'] in oow}
    tp_swing = {'at_least_3_turning_points', 'turning_point_floor', 'swings_non_increasing_floored', 'P_hat_and_W'}
    out_reads = {br: [e['sub_test'] for e in SC6.SUB_TEST_READS[br] if not e['inside_the_window_as_implemented']]
                 for br in ('oscillatory', 'monotone')}
    other = {br: [s for s in v if s not in tp_swing] for br, v in out_reads.items()}
    other_desc = {br: {s: next(e['cycles'] for e in SC6.SUB_TEST_READS[br] if e['sub_test'] == s) for s in v}
                  for br, v in other.items()}
    t5a = 'every cycle in the window the test reads was solved cleanly'
    t5b = 'Cycles before the window are read only for the turning points and the swing history, and are not vetoed.'
    win_ok = 'lo <= c <= hi' in src5 and t5a in norm4
    D.append({'id': 'D5', 'claim': f'"{t5a}" ... "{t5b}"',
              'counterpart': 'settling_criterion_v5._non_clean_in (veto over the certifying window only); '
                             'settling_criterion_v6.SUB_TEST_READS (the reads outside the window, per branch); W153 '
                             'certificates_with_a_non_clean_cycle_read_out_of_window',
              'evidence': {'veto_reads_window_only': 'lo <= c <= hi' in src5, 'v4_text_found': [t5a in norm4, t5b in norm4],
                           'out_of_window_sub_tests': out_reads, 'not_turning_point_or_swing_reads': other_desc,
                           'certificates_reading_a_non_clean_cycle_out_of_window': oow,
                           'their_branches': oow_br},
              'verdict': ('window scope CONSISTENT with the veto; "read only for the turning points and the swing '
                          'history" is INCOMPLETE: before the window the test also reads ' +
                          '; '.join(f"{br}: " + ', '.join(f'{s} ({d})' for s, d in v.items()) for br, v in
                                    other_desc.items() if v)) if win_ok and any(other.values()) else
                         ('consistent' if win_ok else 'not confirmed'),
              'note': f'the {len(oow)} certificates that read a non-clean cycle before their window are all '
                      f'{"/".join(sorted(set(oow_br.values())))}; reads before the window are enumerated, not vetoed '
                      '(Addendum 61 ruling 3), as v4 says. Classification used: the sub-tests '
                      f'{sorted(tp_swing)} count as turning-point / swing-history reads; every other sub-test that '
                      'SUB_TEST_READS marks inside_the_window_as_implemented False is listed in the verdict',
              'w162_verdict_on_v3': by['D5']['verdict'], 'w163_handling': 're-judged on the v4 wording'})
    oc, ocm = rec['old_certificates']
    t6 = 'the objective moved by a further 15–21 k€ after the residual-based stopping rule had certified them'
    ok6 = all(v['ok'] for v in oc.values()) and not ocm['w86_spec_mentions_settling'] and t6 in norm4
    D.append({'id': 'D6', 'claim': f'"{t6}"',
              'counterpart': 'W101 G13 declarations (replay_reference, hold_after_cycle); the W86 evaluation records '
                             'of x0 (5cfe69a6) and the unit (ca8927e7): status, certification_cycle, stopped_by, '
                             'required_consecutive_cycles, the per-cycle residual streak; the W86 campaign spec (no '
                             'settling rule); W101 summary N, k0, Q_cert_old_Q_N',
              'evidence': {'certificates': oc, 'w86_spec': ocm, 'v4_text_found': t6 in norm4,
                           'reference_beside': rec['reference_beside']},
              'verdict': 'consistent: x0 certified at 132 and the unit at 112, both by the residual rule (10 '
                         'consecutive cycles passing the residual test with successful local solves, from 123 / 103); '
                         's (N4, N5) is measured from those certificates' if ok6 else 'NOT CONFIRMED',
              'note': 'Q(N) = the W86 certified cost exactly on both; k0 = the first cycle of the certifying streak',
              'w162_verdict_on_v3': by['D6']['verdict'], 'w163_handling': 're-judged on the v4 wording'})
    return D


# ======================================================================================================================
#  inputs, guards, post-run
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(W161.PICKLE_COUNTS), w162=dict(W162.PICKLE_COUNTS),
                  w163=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run():
    for rel in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG):
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W163 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in (OUT_LOG, OUT_MAN, OUT_TYPING_JSON, OUT_TYPING_LOG)}
    with open(os.path.join(REPO, OUT_POST), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f'[W163 post-run] wrote {OUT_POST}: ' + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def md_summary(o):
    s = o['summary']
    L = ['# W163 -- figure check of paragraphs_v4.md (Addendum 66)', '',
         f"Text: `{P4_REL}` sha256 `{P4_SHA}` (commit `{P4_COMMIT}`); v3 `{P3_SHA[:8]}` (`{P3_COMMIT}`) for the line map. "
         f"Frozen tables `{os.path.basename(FZ_REL)}` (sha256 `{FZ_SHA[:8]}…`). Script `{SCRIPT_REL}` (imports "
         f"`{W162_SCRIPT}`, not edited). ZERO SOLVES (guards verified 0), pickle blocked. The text is not edited.", '',
         f"## Counts (manuscript scope: lines {MANUSCRIPT_FROM_LINE_V4}-{o['text']['scope_end']} of paragraphs_v4.md)", '',
         f"- numeric tokens in scope: {s['tokens_scope']} (non-manuscript title/comments: {s['tokens_nonmanuscript']})",
         f"- figure checks touching v4 prose: {s['n_checks_in_v4']} -- match {s['n_match']}, MISMATCH {s['n_mismatch']}, "
         f"approximate {s['n_approximate']}, no table counterpart {s['n_no_table_counterpart']}",
         f"- W161 checks carried: {s['w161_carried_verbatim']} verbatim + {s['w161_carried_rewritten']} on a sentence "
         f"rewritten in v3; {s['w161_not_in_v4']} of 164 not in the v4 prose",
         f"- W162 new checks: {s['n_w162_new_kept']} kept (tokens remapped), {len(REBUILT)} rebuilt on the v4 wording "
         f"({', '.join(REBUILT)}); new in W163: {', '.join(s['new_in_w163'])}; X: {s['x_status']}",
         f"- tokens unchecked: {s['n_unchecked_tokens']} {s['unchecked_by_category']}",
         f"- every token assigned: {s['every_token_assigned']}", '',
         '## W162 non-matches, on v4', '', '| id | W162 on v3 | W163 on v4 | v4 written | counterpart |', '|---|---|---|---|---|']
    for r in o['w162_nonmatches_on_v4']:
        L.append(f"| {r['id']} | {r['w162_status']} | {r['w163_status']} | {r['written_v4']} | {r['counterpart']} |")
    L += ['', '## Mismatches', '']
    for c in o['mismatches']:
        L.append(f"- **{c['id']}** line(s) {sorted({x[0] for x in c['tokens']})}: written `{c.get('written_v4')}`, "
                 f"counterpart {c.get('value_at_written_precision', c.get('table_value_at_written_precision'))} -- "
                 f"{c.get('counterpart', c.get('table_ref'))}. {c.get('note') or ''}")
    if not o['mismatches']:
        L.append('none')
    L += ['', '## Approximate', '']
    for c in o['approximate']:
        L.append(f"- **{c['id']}**: written `{c['written_v4']}`, counterpart {c['value_at_written_precision']}. "
                 f"{c.get('note') or ''}")
    L += ['', '## New and rebuilt checks (v4)', '']
    for c in o['checks_in_v4']:
        if c.get('w163_handling', '').startswith(('new', 'rebuilt')):
            L.append(f"- **{c['id']}** ({c['status']}) written `{c['written_v4']}` -> "
                     f"{c['value_at_written_precision']}; {c['counterpart']}. {c.get('note') or ''}")
    for c in o['nonmanuscript_checks']:
        if c.get('w163_handling', '').startswith('new'):
            L.append(f"- **{c['id']}** ({c['status']}) {c['written_v4']}: {c['counterpart']}. {c.get('note') or ''}")
    L += ['', '## Unchecked numbers', '', '| line | token | category | reason |', '|---|---|---|---|']
    for u in o['unchecked_tokens']:
        L.append(f"| {u['line']} | {u['text']} | {u['category']} | {u['reason']} |")
    L += ['', '## Definition checks', '']
    for d in o['definition_checks']:
        L.append(f"- **{d['id']}** {d['claim']}: **{d['verdict']}**. {d.get('note') or ''}")
    L += ['', '## Every check in v4', '', '| id | lines | written | status | counterpart (at written precision) | '
          'handling |', '|---|---|---|---|---|---|']
    for c in o['checks_in_v4']:
        ln = ','.join(str(x) for x in sorted({x[0] for x in c['tokens']})) or '-'
        val = c.get('value_at_written_precision', c.get('table_value_at_written_precision'))
        L.append(f"| {c['id']} | {ln} | {c.get('written_v4')} | {c['status']} | {str(val).replace('|', '/')[:120]} | "
                 f"{c.get('w163_handling', '')} |")
    L += ['', '## W161 checks not in the v4 prose', '', ', '.join(c['id'] for c in o['w161_not_in_v4'])]
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--out-dir', default=None, help='trial runs only: write every W163 output under this directory')
    args = ap.parse_args()
    if args.out_dir:
        g = globals()
        for k in ('OUT_JSON', 'OUT_MD', 'OUT_MAN', 'OUT_LOG', 'OUT_TYPING_JSON', 'OUT_TYPING_LOG', 'OUT_POST'):
            g[k] = os.path.join(args.out_dir, os.path.basename(g[k]))
    if args.post_run:
        post_run()
    t0 = time.time()
    tag = 'W163'
    failed = []
    # ---- preconditions ------------------------------------------------------------------------------------------
    pre = []
    for rel in (OUT_JSON, OUT_MD, OUT_MAN, OUT_POST, OUT_TYPING_JSON):
        if os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} exists (write-once)')
    if not os.path.isdir(os.path.join(REPO, os.path.dirname(OUT_JSON))):
        pre.append(f'{os.path.dirname(OUT_JSON)} missing (create it before the launch; the launch log goes there)')
    w101_eval = {}
    for nm in ('x0', 'n7_4h_e1', 'c_star'):
        ed_root = os.path.join(W162.W101_DIR, f'campaign_s53_w101_srp1_cont_{nm}', 'evals')
        subs = sorted(os.listdir(os.path.join(REPO, ed_root)))
        if len(subs) != 1:
            pre.append(f'{ed_root}: expected one eval dir, found {subs}')
            continue
        w101_eval[nm] = os.path.join(ed_root, subs[0])
    input_rels = {'PARAGRAPHS_V4': P4_REL, 'PARAGRAPHS_V3': P3_REL, 'PARAGRAPHS_V2': P2_REL, 'FROZEN_JSON': FZ_REL,
                  'W161_JSON': W162.W161_JSON, 'W162_JSON': W162_JSON, 'W162_MANIFEST': W162_MAN,
                  'W162_SCRIPT': W162_SCRIPT, 'W153C': W162.W153C_REL, 'W153D': W162.W153D_REL,
                  'W153_MANIFEST': W162.W153_MAN, 'V6_SPEC': W162.V6_SPEC, 'V41_SPEC': W162.V41_SPEC,
                  'W118_SPEC': W162.W118_SPEC, 'W118_SUMMARY': W162.W118_SUMMARY, 'W101_SUMMARY': W162.W101_SUMMARY,
                  'W86_RESULTS': W162.W86_RESULTS, 'W86_SPEC': W86_SPEC, 'W98_RESULTS': W162.W98_RESULTS,
                  'W98_EVAL': W162.W98_EVAL, 'W109_JSON': W162.W109_JSON, 'W141_JSON': W162.W141_JSON,
                  'W137_CELL3_RECORD': W162.W137_CELL3, 'W132_CELL1_RECORD': W162.W132_CELL1,
                  'W139_C52_RECORD': W162.W139_C52_REC, 'W139_C52_RESULTS': W162.W139_C52_RES,
                  'W159_JSON': W162.W159_JSON, 'W155_SOH050_SPEC': W162.W155_SOH050, 'W155_M175_SPEC': W162.W155_M175,
                  'SRP1_PARAMS': W162.SRP1_PARAMS, 'CASE9_PARAMS': CASE9_REL, 'NOTE_5456': W162.NOTE_5456,
                  'BRIEF': W162.BRIEF, 'TASKS': 'TASKS.md', 'HELPER_FUNCTIONS': 'helper_functions.py',
                  'ADMM_PARAMETERS': 'admm_parameters.py'}
    for lab, rel in DSO_CASES.items():
        input_rels[f'CASE_{lab}'] = rel
    for nm, ed in W86_EVALS.items():
        for f in ('evaluation_record.json', 'per_cycle_record.jsonl', 'convergence_depth_tail_state.json',
                  'network_ipopt_solve_records.jsonl'):
            input_rels[f'W86_{nm}_{f}'] = os.path.join(ed, f)
    for nm, ed in w101_eval.items():
        for f in ('per_cycle_record.jsonl', 'settling_decision.json'):
            input_rels[f'W101_{nm}_{f}'] = os.path.join(ed, f)
        input_rels[f'W101_{nm}_campaign_results'] = os.path.join(os.path.dirname(os.path.dirname(ed)),
                                                                 'campaign_results.json')
    for f in W162.CODE_FILES:
        input_rels[f'CODE_{f}'] = f
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel)}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    for key, want in (('PARAGRAPHS_V4', P4_SHA), ('PARAGRAPHS_V3', P3_SHA), ('PARAGRAPHS_V2', P2_SHA),
                      ('FROZEN_JSON', FZ_SHA), ('V6_SPEC', W162.V6_SHA)):
        if inputs[key]['sha256'] != want:
            pre.append(f"{inputs[key]['path']} sha {inputs[key]['sha256']} != {want}")
    if not (inputs['PARAGRAPHS_V4']['last_commit'] or '').startswith(P4_COMMIT):
        pre.append(f"{P4_REL} last commit {inputs['PARAGRAPHS_V4']['last_commit']} != {P4_COMMIT}")
    w153man = _jl(W162.W153_MAN)
    for key in ('W153C', 'W153D'):
        if w153man.get(inputs[key]['path']) != inputs[key]['sha256']:
            pre.append(f"{inputs[key]['path']} != its W153 manifest entry")
    w162man = _jl(W162_MAN)
    if w162man.get(W162_JSON) != inputs['W162_JSON']['sha256']:
        pre.append(f'{W162_JSON} != its W162 manifest entry')
    if not (inputs['W162_SCRIPT']['last_commit'] or '').startswith(W162_SCRIPT_COMMIT):
        pre.append(f"{W162_SCRIPT} last commit {inputs['W162_SCRIPT']['last_commit']} != {W162_SCRIPT_COMMIT}")
    fz = _jl(FZ_REL)
    for k in ('tables', 'constants', 'paragraph_figure_checks'):
        if k not in fz:
            pre.append(f'frozen JSON lacks {k}')
    if not os.path.exists(IPOPT_BIN):
        pre.append(f'{IPOPT_BIN} missing')
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; '
         f'{len(inputs)} inputs committed clean; paragraphs_v4 {P4_SHA[:8]}, paragraphs_v3 {P3_SHA[:8]}, frozen JSON '
         f'{FZ_SHA[:8]}, v6 spec {W162.V6_SHA[:8]}, W162 JSON {inputs["W162_JSON"]["sha256"][:8]} verified')

    # ---- the texts, the line map, the inventory ------------------------------------------------------------------
    raw3, raw4 = _text(P3_REL), _text(P4_REL)
    l3, e3 = W162.scope_lines(raw3)
    l4, e4 = W162.scope_lines(raw4)
    lmap, changed, ops, same_text = line_map(l3, e3, l4, e4)
    W162.MANUSCRIPT_FROM_LINE = MANUSCRIPT_FROM_LINE_V4     # override: v4's manuscript scope starts one line later
    norm4 = W162.normalise(l4, e4)
    toks = W162.inventory(l4, e4)
    inscope_keys = {(x['line'], x['text'], x['occ']) for x in toks}
    remap = Remap(lmap)

    # ---- the records ---------------------------------------------------------------------------------------------
    rec = {'v6': _jl(W162.V6_SPEC), 'v41': _jl(W162.V41_SPEC), 'w118_spec': _jl(W162.W118_SPEC),
           'w118_summary': _jl(W162.W118_SUMMARY), 'w101_summary': _jl(W162.W101_SUMMARY), 'w86': _jl(W162.W86_RESULTS),
           'w98': _jl(W162.W98_RESULTS), 'w109': _jl(W162.W109_JSON), 'w141': _jl(W162.W141_JSON),
           'w159': _jl(W162.W159_JSON), 'srp1_params': _jl(W162.SRP1_PARAMS), 'case9': _jl(CASE9_REL),
           'w153c': _jl(W162.W153C_REL), 'w153d': _jl(W162.W153D_REL), 'w139_c52': _jl(W162.W139_C52_RES),
           'inputs_extra': {}, 'p2_sha': _sha(P2_REL), 'p2_commit': W160._last_commit(P2_REL),
           'p3_commit_date': _git('log', '-1', '--format=%ad', '--date=short', '--', P3_REL),
           'code': {f: _text(f) for f in W162.CODE_FILES + (W162.NOTE_5456, W162.BRIEF, 'helper_functions.py',
                                                            'admm_parameters.py', 'TASKS.md')},
           'w86_spec': _jl(W86_SPEC)}
    rec['w101_rows'] = {nm: _rows(os.path.join(ed, 'per_cycle_record.jsonl')) for nm, ed in w101_eval.items()}
    rec['w101_replays'], rec['w101_g13'] = {}, {}
    for nm, ed in w101_eval.items():
        cr = _jl(os.path.join(os.path.dirname(os.path.dirname(ed)), 'campaign_results.json'))
        rec['w101_replays'][nm] = {'G19': cr['gates']['G19_replay_bitwise_1_N_every_field'],
                                   'through': cr['gate_detail']['G13']['summary']['replay_bitwise_through_cycle'],
                                   'N': rec['w101_summary']['reports'][nm]['N']}
        rec['w101_g13'][nm] = cr['gate_detail']['G13']['summary']['declaration']
    w98e = _jl(W162.W98_EVAL)
    rec['w98_eval_key_x0'] = w98e['candidate_key']
    rec['w98_x0_canonical_all_zero'] = all(v == [0.0, 0.0] for v in w98e['candidate_canonical']['nodes'].values())
    q3 = {k: v['gross_operational_cost'] for k, v in _rows(W162.W137_CELL3).items()}
    q5 = {k: v['gross_operational_cost'] for k, v in _rows(W162.W139_C52_REC).items()}
    q1 = {k: v['gross_operational_cost'] for k, v in _rows(W162.W132_CELL1).items()}
    mv = {'b_2a0ba8b2 (v3 run, Q(213) - Q(173); v4 k* 173)': q1[max(q1)] - q1[173],
          'b_4649234b (v4 run, Q(193) - Q(148); v5 k* 148)': q3[max(q3)] - q3[148],
          'd_c52e1670 (v5 run, Q(198) - Q(150); v6 k* 150)': q5[max(q5)] - q5[150],
          'd_c52e1670 (v5 run, Q(198) - Q(148); last-pair reading k* 148)': q5[max(q5)] - q5[148]}
    w141d = rec['w141']['d_c52e1670_detail']['per_variant']
    rec['postcert'] = {'movement_eur': mv, 'over_tau': {k: v / TAU for k, v in mv.items()},
                       'w141_V3_10_over_tau': w141d['V3_10']['Q_last_recorded_minus_Q_k_star_over_tau'],
                       'w141_V1_over_tau': w141d['V1']['Q_last_recorded_minus_Q_k_star_over_tau'],
                       'last_cycles': {'b_2a0ba8b2_v3': max(q1), 'b_4649234b_v4': max(q3), 'd_c52e1670_v5': max(q5)}}
    envnow = W162.env_now()
    w101 = rec['w101_summary']['reports']
    bes = {}
    for nm in ('x0', 'n7_4h_e1', 'c_star'):
        r = w101[nm]
        rows = rec['w101_rows'][nm]
        k0, n, endc = r['k0'], r['N'], (r.get('k_star') or r.get('k_cap'))
        q = {k: rows[k]['gross_operational_cost'] for k in rows}
        bes[nm] = {'status': r['status'], 'k0': k0, 'N': n, 'end': endc, 's_signed': r['s_signed'],
                   'Q_end_minus_Q_N_recomputed': q[endc] - q[n], 'Q_end_minus_Q_k0': q[endc] - q[k0],
                   'max_abs_Q_k_minus_Q_N': max(abs(q[k] - q[n]) for k in range(n, endc + 1)),
                   'max_abs_Q_k_minus_Q_k0': max(abs(q[k] - q[k0]) for k in range(k0, endc + 1)),
                   'boyd_all_pass_k0_minus_1': rows[k0 - 1].get('boyd_all_pass'),
                   'boyd_all_pass_k0': rows[k0].get('boyd_all_pass')}
    rec['reference_beside'] = bes
    # W163 additions
    rec['w86_eval_record'] = {nm: _jl(os.path.join(ed, 'evaluation_record.json')) for nm, ed in W86_EVALS.items()}
    rec['w86_rows'] = {nm: _rows(os.path.join(ed, 'per_cycle_record.jsonl')) for nm, ed in W86_EVALS.items()}
    rec['w86_tail_state'] = {nm: _jl(os.path.join(ed, 'convergence_depth_tail_state.json'))
                             for nm, ed in W86_EVALS.items()}
    rec['w86_solve_records'] = {}
    for nm, ed in W86_EVALS.items():
        with open(os.path.join(REPO, ed, 'network_ipopt_solve_records.jsonl'), encoding='utf-8') as h:
            rec['w86_solve_records'][nm] = [json.loads(ln) for ln in h if ln.strip()]
    rec['ipopt_print_options'] = ipopt_print_options()
    rec['tail'] = tail_records(rec)
    rec['success'] = success_clause_records(rec)
    rec['old_certificates'] = old_certificates(rec)

    # ---- 2. W162's checks on v4 ----------------------------------------------------------------------------------
    w162doc = _jl(W162_JSON)
    carried0, carry_meta = W162.carry_over(fz, raw4, norm4, rec['w153c'], _jl(W162.W161_JSON))
    carried = [as_v4(c, remap, 'W161 check carried by W162, re-run on v4 (tokens remapped)') for c in carried0]
    new162 = W162.new_checks(fz, norm4, rec, envnow)
    w162_by_id = {c['id']: c for c in new162}
    kept = [as_v4(c, remap, 'W162 new check, re-run on v4 (tokens remapped)') for c in new162 if c['id'] not in REBUILT]
    v4c = v4_checks(fz, rec, w162_by_id, envnow)
    newc = []
    for c in new162:
        newc.append(v4c[c['id']] if c['id'] in REBUILT else next(k for k in kept if k['id'] == c['id']))
    newc += [v4c[i] for i in sorted(set(v4c) - set(REBUILT), key=lambda s: int(s[1:]))]
    x162 = [as_v4(c, remap, 'W162 identifier check, re-run (tokens remapped)') for c in W162.nonmanuscript_checks(rec)]
    xch = x162 + x_v4_checks(rec, raw4, l4, inscope_keys)
    dch = d_v4(W162.definition_checks(fz, rec), rec, norm4)
    unchecked_v4 = {}
    for k, v in W162.UNCHECKED.items():
        n = remap('UNCHECKED', [k])
        if n:
            nk = tuple(n[0])
            unchecked_v4[nk] = UNCHECKED_REASON_V4.get(nk, v)

    # ---- 5. token coverage -------------------------------------------------------------------------------------
    in_v4 = [c for c in carried if c['placement'] != 'not in the v4 prose'] + newc + xch
    key = {(x['line'], x['text'], x['occ']): x for x in toks}
    covered, stale = {}, []
    for c in in_v4:
        for ln, tx, oc in c['tokens']:
            if (ln, tx, oc) not in key:
                stale.append((c['id'], ln, tx, oc))
            covered.setdefault((ln, tx, oc), []).append(c['id'])
    unchecked = []
    for k, (cat, why) in unchecked_v4.items():
        if k not in key:
            stale.append(('UNCHECKED', *k))
        elif k in covered:
            stale.append(('UNCHECKED-and-checked', *k))
        else:
            unchecked.append({'line': k[0], 'text': k[1], 'occ': k[2], 'category': cat, 'reason': why})
    unassigned = [x for x in toks if (x['line'], x['text'], x['occ']) not in covered and
                  (x['line'], x['text'], x['occ']) not in unchecked_v4]
    for x in toks:
        x['covered_by'] = covered.get((x['line'], x['text'], x['occ']), [])
    frag_missing = []
    for c in carried:
        if c['placement'].startswith('rewritten in v3') and not c['fragment_found_v4']:
            frag_missing.append(c['id'])
    for c in newc:
        if c['fragment_v4'] and W162.norm_frag(c['fragment_v4']) not in norm4:
            frag_missing.append(c['id'])
    no_tok = [c['id'] for c in carried if c['placement'] != 'not in the v4 prose' and not c['tokens']
              and c['id'] not in ('S2c', 'S2e', 'V1b')]

    # ---- comparison with the committed W162 results ---------------------------------------------------------------
    w162_carry = {c['id']: c for c in w162doc['w161_carry_over']}
    w162_chk = {c['id']: c for c in w162doc['checks_in_v3'] + w162doc['nonmanuscript_checks']}
    w162_def = {d['id']: d for d in w162doc['definition_checks']}
    diff = []
    for c in carried:
        s = w162_carry.get(c['id'])
        if s is None or PLACEMENT_V4[s['placement']] != c['placement'] or s.get('status') != c.get('status') or \
                str(s.get('table_value_at_written_precision')) != str(c.get('table_value_at_written_precision')):
            diff.append(c['id'])
    for c in kept + x162:
        s = w162_chk.get(c['id'])
        if s is None or s['status'] != c['status']:
            diff.append(c['id'])
        elif 'environment now' not in str(c.get('counterpart_kind')) and \
                json.dumps(_norm(s['value']), sort_keys=True) != json.dumps(_norm(c['value']), sort_keys=True):
            diff.append(c['id'])
    for d in dch[:3]:
        if w162_def[d['id']]['verdict'] != d['verdict']:
            diff.append(d['id'])
    nonmatch162 = []
    for i, c in w162_chk.items():
        if c['status'] != 'match':
            now = next((x for x in in_v4 if x['id'] == i), None)
            nonmatch162.append({'id': i, 'w162_status': c['status'], 'w163_status': now['status'] if now else None,
                                'written_v3': c.get('written_v3'), 'written_v4': now.get('written_v4') if now else None,
                                'counterpart': now.get('value_at_written_precision', now.get(
                                    'table_value_at_written_precision')) if now else None})
    for i in ('D4', 'D5', 'D6'):
        nonmatch162.append({'id': i, 'w162_status': w162_def[i]['verdict'],
                            'w163_status': next(d['verdict'] for d in dch if d['id'] == i), 'written_v3': None,
                            'written_v4': None, 'counterpart': None})

    checks_v4 = [c for c in in_v4 if c['id'][0] != 'X']
    mism = [c for c in in_v4 if c['status'] == 'MISMATCH']
    appr = [c for c in in_v4 if c['status'] == 'approximate']
    ucat = {}
    for u in unchecked:
        ucat[u['category']] = ucat.get(u['category'], 0) + 1
    summary = {
        'tokens_total': len(toks), 'tokens_scope': sum(1 for x in toks if x['region'] == 'scope'),
        'tokens_nonmanuscript': sum(1 for x in toks if x['region'] != 'scope'),
        'n_checks_in_v4': len(checks_v4), 'n_match': sum(c['status'] == 'match' for c in checks_v4),
        'n_mismatch': sum(c['status'] == 'MISMATCH' for c in checks_v4),
        'n_approximate': sum(c['status'] == 'approximate' for c in checks_v4),
        'n_no_table_counterpart': sum(c['status'] == 'no table counterpart' for c in checks_v4),
        'mismatch_ids': [c['id'] for c in mism], 'approximate_ids': [c['id'] for c in appr],
        'no_table_counterpart_ids': [c['id'] for c in checks_v4 if c['status'] == 'no table counterpart'],
        'w161_carried_verbatim': sum(c['placement'] == 'verbatim in v4' for c in carried),
        'w161_carried_rewritten': sum(c['placement'].startswith('rewritten in v3') for c in carried),
        'w161_not_in_v4': sum(c['placement'] == 'not in the v4 prose' for c in carried),
        'n_w162_new_kept': len(kept), 'rebuilt': list(REBUILT), 'new_in_w163': sorted(set(v4c) - set(REBUILT)) +
        ['X5', 'X6', 'X7'], 'n_new_checks_total': len(newc), 'n_x': len(xch),
        'x_status': {c['id']: c['status'] for c in xch},
        'n_unchecked_tokens': len(unchecked), 'unchecked_by_category': ucat,
        'every_token_assigned': not unassigned and not stale and not remap.unmapped,
        'definition_verdicts': {d['id']: d['verdict'] for d in dch},
    }
    checks = {
        'paragraphs_v4_sha_pinned': inputs['PARAGRAPHS_V4']['sha256'] == P4_SHA,
        'paragraphs_v3_sha_pinned': inputs['PARAGRAPHS_V3']['sha256'] == P3_SHA,
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'v3_to_v4_changed_lines_as_declared': tuple(changed) == V3_CHANGED_LINES and same_text,
        'changed_token_map_on_changed_lines_only': all(k[0] in V3_CHANGED_LINES for k in CHANGED_TOKEN_MAP),
        'every_w162_token_remapped': not remap.unmapped,
        'w161_counterparts_rederived_equal_to_committed': carry_meta['all_same_counterpart'],
        'w161_count_164': carry_meta['n_w161_checks'] == 164 == carry_meta['n_w161_stored'],
        'w161_evaluator_self_test_on_v1': carry_meta['w161_evaluator_self_test_on_v1_reproduces'],
        'w162_evaluator_self_test_on_v2_values': carry_meta['w162_evaluator_self_test_all_reproduce'],
        'unchanged_checks_reproduce_w162': not diff,
        'every_token_assigned_no_stale_assignment': not unassigned and not stale,
        'every_v4_fragment_found': not frag_missing,
        'every_carried_check_names_its_tokens': not no_tok,
    }
    failed += [k for k, v in checks.items() if v is not True]
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    out = {'schema': 'p515_s53_w163_paragraphs_v4_figure_check', 'version': 1,
           'stage': 'P5.15 W163 -- figure check of paragraphs_v4.md (Addendum 66)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {W162_SCRIPT: _sha(W162_SCRIPT)}},
           'text': {'path': P4_REL, 'sha256': P4_SHA, 'commit': P4_COMMIT, 'scope_end': e4,
                    'scope': f'lines 1-{e4} (above "{W162.SCOPE_END}"); manuscript scope lines '
                             f'{MANUSCRIPT_FROM_LINE_V4}-{e4}', 'edited': False},
           'v3_to_v4': {'v3': {'path': P3_REL, 'sha256': P3_SHA, 'commit': P3_COMMIT}, 'difflib_opcodes': ops,
                        'v3_changed_lines': changed, 'declared': list(V3_CHANGED_LINES),
                        'equal_lines_identical_text': same_text,
                        'changed_token_map': [[*k, *(v or [None, None, None])] for k, v in CHANGED_TOKEN_MAP.items()],
                        'tokens_via_changed_map': remap.via_changed, 'tokens_dropped': remap.dropped,
                        'tokens_unmapped': remap.unmapped},
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA},
           'inputs': inputs, 'inputs_read_in_checks': rec['inputs_extra'], 'environment_now': envnow,
           'summary': summary, 'w162_nonmatches_on_v4': nonmatch162, 'checks_in_v4': checks_v4,
           'nonmanuscript_checks': xch, 'mismatches': mism, 'approximate': appr, 'unchecked_tokens': unchecked,
           'w161_carry_over': carried, 'w161_carry_meta': carry_meta,
           'w161_not_in_v4': [{'id': c['id'], 'fragment_w161': c['fragment_w161'], 'written_v2': c['written_v2'],
                               'w161_status': c['w161_status']} for c in carried
                              if c['placement'] == 'not in the v4 prose'],
           'definition_checks': dch, 'w162_comparison': {'differences': diff, 'w162_json': W162_JSON,
                                                         'w162_json_sha256': inputs['W162_JSON']['sha256']},
           'tail_tolerances': rec['tail'], 'success_clause': rec['success'],
           'old_certificates': {'certificates': rec['old_certificates'][0], 'w86_spec': rec['old_certificates'][1]},
           'reference_evaluations_beside': rec['reference_beside'],
           'post_certification_movement': rec['postcert'],
           'token_inventory': toks, 'unassigned_tokens': unassigned, 'stale_assignments': [list(s) for s in stale],
           'fragments_not_found': frag_missing, 'carried_without_tokens': no_tok,
           'checks': checks, 'failed': sorted(set(failed)), 'guards': guards, 'pickle_guard': pk,
           'exit_code': code, 'wall_s': time.time() - t0}

    written = {}

    def wr(rel, data):
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel)
    wr(OUT_JSON, GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr(OUT_MD, md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    man.update(rec['inputs_extra'])
    wr(OUT_MAN, GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log -----------------------------------------------------------------------------------------------------
    s = summary
    _log(f"[{tag}] v3 -> v4: changed v3 lines {changed} (declared {list(V3_CHANGED_LINES)}); tokens via changed map "
         f"{len(remap.via_changed)}, dropped {remap.dropped}, unmapped {remap.unmapped}")
    _log(f"[{tag}] tokens: {s['tokens_total']} ({s['tokens_scope']} in scope, {s['tokens_nonmanuscript']} non-manuscript); "
         f"every token assigned {s['every_token_assigned']} (unassigned {len(unassigned)}, stale {len(stale)})")
    _log(f"[{tag}] W161 carry-over: {carry_meta['n_w161_checks']} re-derived, same counterpart as committed "
         f"{carry_meta['all_same_counterpart']}; verbatim {s['w161_carried_verbatim']}, rewritten in v3 "
         f"{s['w161_carried_rewritten']}, not in v4 {s['w161_not_in_v4']}")
    _log(f"[{tag}] checks in v4: {s['n_checks_in_v4']} -- match {s['n_match']}, MISMATCH {s['n_mismatch']} "
         f"{s['mismatch_ids']}, approximate {s['n_approximate']} {s['approximate_ids']}, no table counterpart "
         f"{s['n_no_table_counterpart']} {s['no_table_counterpart_ids']}; W162 kept {s['n_w162_new_kept']}, rebuilt "
         f"{list(REBUILT)}, new {s['new_in_w163']}; X {s['x_status']}")
    _log(f"[{tag}] unchanged checks reproduce W162: {not diff} {diff}")
    for r in nonmatch162:
        _log(f"[{tag}]   W162 non-match {r['id']}: v3 {r['w162_status']!r} -> v4 {r['w163_status']!r} "
             f"(written {r['written_v4']!r}, counterpart {r['counterpart']!r})")
    for c in mism + appr:
        _log(f"[{tag}]   {c['id']} {c['status']}: written {c.get('written_v4')!r}, counterpart "
             f"{c.get('value_at_written_precision', c.get('table_value_at_written_precision'))!r} -- "
             f"{c.get('note') or ''}")
    for c in newc + xch:
        if c.get('w163_handling', '').startswith(('new', 'rebuilt')):
            _log(f"[{tag}]   {c['id']} {c['status']}: written {c.get('written_v4')!r} -> "
                 f"{c.get('value_at_written_precision')!r}")
    for u in unchecked:
        _log(f"[{tag}]   UNCHECKED line {u['line']} {u['text']!r} [{u['category']}] {u['reason']}")
    for d in dch:
        _log(f"[{tag}]   {d['id']}: {d['verdict']}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if unassigned:
        _log(f'[{tag}] UNASSIGNED tokens: ' + ', '.join(f"{x['line']}:{x['text']}#{x['occ']}" for x in unassigned))
    if stale or frag_missing or no_tok:
        _log(f'[{tag}] stale {stale}; fragments not found {frag_missing}; carried without tokens {no_tok}')
    if failed:
        _log(f'[{tag}] FAILED: {sorted(set(failed))}')
    _log(f"[{tag}] wrote {', '.join(f'{k} {v[:8]}' for k, v in written.items())}")
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle ok {pk['ok']}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
