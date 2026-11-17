"""
P5.15 Addendum 76 Part 3, Planner task W177b-3 -- STRUCTURE CHECKS of the revised manuscript at Overleaf 51a6fb5
(P5_15_W177B_PART3.md). ZERO SOLVES, NO MODEL, NO PICKLE; read-only on production code, specs, case files, the frozen
JSON, the manuscript clone, manuscript_review/ and every committed artifact.

What it does (static reading only):
  0. Identity: the Overleaf clone's HEAD (declared 51a6fb5), main.tex sha256 (declared, full hash), line count (2,312),
     main.tex / bibliography.bib unmodified in the clone.
  a. Appendix A "Certification Rule" (app:admm_updated_certification) and the Section 2.2.7 summary: every clause of
     rule v6 located in main.tex by exact text, and its code anchor located by exact text in settling_criterion*.py,
     the v6 hook chain and the scorer (file:line at repo HEAD). A missing anchor FAILS the run.
  b. Appendix A "Solver and Coordination Settings" (app:admm_updated_settings) and Section 3.6 (sec:case_settings):
     every setting located in its region, its printed value compared with the source (case files, SRP1_params.json,
     SRP1_ESS_Params.json, definitions.py, the frozen campaign specs by field and sha256, committed run logs);
     occurrences counted in both regions (duplicates) and elsewhere in the body (informational); settings of the
     frozen configuration found in no region listed (omissions; the Worker's list, each source anchored).
  c. Appendix A.2: f_E (eq:esso_objective) and the agent's constraint list against shared_energy_storage_data.py
     _build_subproblem and shared_resources_planning.update_shared_energy_storage_model_to_admm (static: which rows
     the code builds; no model is built).
  d. Nomenclature: symbols defined but unused, used but undefined, after the moves. Tokeniser and normalisation of
     W167 (p515_s53_w167_nomenclature_years, imported, not edited) with ONE stated change: a subscript that is a single
     font-wrapped word (\\text{sum}, \\mathrm{w}) or \\max / \\min is kept as a label (t_sum, n_w, P_max). Every symbol
     of a multi-symbol nomenclature cell is audited (W167 took the first). Rows whose normalised key collides with
     another symbol are checked on their raw form. The same audit on e6e58e8 and dce05d0 (git show, read-only) says
     which unused symbols were still used before the moves.
  e. manuscript_review/fig3_ageing_arms_v2.pdf and .py against T8 of frozen_step6_tables_v1_590088fe.json (and the
     T1 E-claim thresholds the frozen figure used): the script's rows (ast.literal_eval of its `rows` literal; the
     script is NOT executed); the PDF's text and errorbar geometry read from its content stream (stdlib zlib; fonts'
     ToUnicode CMaps); the frozen W165 figure read the same way for the code-label row mapping.
  f. \\ref / \\eqref / \\cite against \\label / bibliography.bib keys; duplicate labels; the references to
     subsubsec:certification and sec:case_settings with their context.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at exit (the
W167 module, imported for its tokeniser, installs its own permitted=() guard, also verified at 0); pickle.load /
pickle.loads blocked for the whole run and counted (0 expected). No production module is imported (asserted at exit).

Output: the directory must exist and hold nothing but launch.log (it is created by the command below; mkdir refuses
an existing directory); writes w177b_part3_checks.json and manifest_sha256.json. `--out-dir DIR`: trial runs only
(absolute path; the same refusal rule).

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w177b_part3 && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b_part3_checks.py \\
      > data/SRP1/Results/P515S53/w177b_part3/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W177b-3 structure checks (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W177b-3: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W177b-3: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import ast  # noqa: E402
import hashlib  # noqa: E402
import importlib.metadata  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import zlib  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import p515_s53_w167_nomenclature_years as W167  # noqa: E402  (tokeniser; installs its own permitted=() guard)

pickle.load, pickle.loads = _blocked_load, _blocked_loads     # W167 re-binds them to its own blockers; ours again

# ======================================================================================================================
#  constants
# ======================================================================================================================
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w177b_part3')
OUT_JSON = os.path.join(OUT_DIR, 'w177b_part3_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
BIB = os.path.join(CLONE, 'bibliography.bib')
DECLARED_CLONE_HEAD = '51a6fb5'
DECLARED_MAIN_SHA256 = '61fe630cc427d622a855e127f33f96d4be1a7362fe23552b87d2d54318a39346'
DECLARED_MAIN_LINES = 2312
PRE_MOVE_COMMITS = ('e6e58e8', 'dce05d0')     # Addendum 75 section-4 paste; the source of the section-2.2.7 restore

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
SPEC_V6 = os.path.join(P53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json')
SPEC_EXT = os.path.join(P53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json')
SPEC_A64 = os.path.join(P53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json')
SPEC_3X3 = os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'campaign_spec_s53_w91_3x3_pair_231558f0.json')
SPEC_S47 = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b', 'campaign_spec_s47_phase_b_8cfa264e.json')
SPEC_S51 = os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_phase_b',
                        'campaign_spec_s51_f2_phase_b_5ce295e1.json')
SPEC_S53 = os.path.join(P53, 'campaign_s53_f2_certificate_r1', 'campaign_spec_s53_f2_certificate_r1_803571c0.json')
SRP1_PARAMS = os.path.join('data', 'SRP1', 'SRP1_params.json')
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
CASE9 = os.path.join('data', 'SRP1', 'case9', 'case9_params.json')
CASE33 = tuple(os.path.join('data', 'SRP1', f'case33_{i}', f'case33_{i}_params.json') for i in (1, 2, 3))
FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
T8_CSV = os.path.join(P53, 'w160_step6_frozen', 'export', 'T8.csv')
FIG_V2_PDF = os.path.join('manuscript_review', 'fig3_ageing_arms_v2.pdf')
FIG_V2_PY = os.path.join('manuscript_review', 'fig3_ageing_arms_v2.py')
FIG_V2_CLONE = os.path.join(CLONE, 'fig3_ageing_arms_v2.pdf')
FIG_W165 = os.path.join(P53, 'w160_step6_frozen', 'export', 'figures', 'fig3_ageing_arms.pdf')
FIG_W165_DATA = os.path.join(P53, 'w160_step6_frozen', 'export', 'figures', 'fig3_ageing_arms_data.json')
V6_LOG_SRP1 = os.path.join(P53, 'w142_resettle_v6', 'campaign_s53_w142_resettle_v6_c_156ce2d1', 'evals',
                           '3a4c10c859114e20_c_156ce2d1', 'stdout_s39_D.log')
V6_LOG_3X3 = os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals', 'f6e9cd53fdbb8ee8_x0', 'child_stdout.log')
PARAGRAPHS_V5 = os.path.join(P53, 'w160_step6_frozen', 'export', 'paragraphs_v5.md')
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')
CHECK_FAILURES = []


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _fail(msg):
    CHECK_FAILURES.append(msg)
    _log('CHECK FAILURE: ' + msg)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, capture_output=True, text=True, check=True).stdout


def _read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def _json(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


_FILE_CACHE = {}


def locate(rel, needle, occurrence=0, required=True):
    """1-based line of the `occurrence`-th line containing `needle` (exact text) in repo file `rel`."""
    if rel not in _FILE_CACHE:
        _FILE_CACHE[rel] = _read(rel).split('\n')
    hits = [i for i, ln in enumerate(_FILE_CACHE[rel], 1) if needle in ln]
    if len(hits) <= occurrence:
        if required:
            _fail(f'anchor not found: {rel!r} :: {needle!r}')
        return None
    return hits[occurrence]


def anchor(rel, needle, occurrence=0):
    return {'file': rel, 'line': locate(rel, needle, occurrence), 'text': needle}


def anchor_after(rel, after, needle):
    """The first line containing `needle` after the first line containing `after` (exact text)."""
    start = locate(rel, after)
    hits = [i for i, ln in enumerate(_FILE_CACHE[rel], 1) if needle in ln and start is not None and i > start]
    if not hits:
        _fail(f'anchor not found after {after!r}: {rel!r} :: {needle!r}')
    return {'file': rel, 'line': hits[0] if hits else None, 'text': needle, 'after': after, 'after_line': start}


# ======================================================================================================================
#  the manuscript
# ======================================================================================================================
def manuscript_identity():
    head = _git(['rev-parse', '--short', 'HEAD'], cwd=CLONE).strip()
    sha = _sha(MAIN_TEX)
    raw = _read(MAIN_TEX)
    n_lines = len(raw.split('\n')) - (1 if raw.endswith('\n') else 0)
    status = _git(['status', '--porcelain', '--', 'main.tex', 'bibliography.bib'], cwd=CLONE)
    out = {'clone_head': head, 'declared_head': DECLARED_CLONE_HEAD, 'head_ok': head.startswith(DECLARED_CLONE_HEAD),
           'main_tex_sha256': sha, 'sha_ok': sha == DECLARED_MAIN_SHA256, 'main_tex_lines': n_lines,
           'lines_ok': n_lines == DECLARED_MAIN_LINES, 'main_and_bib_unmodified_in_clone': status.strip() == '',
           'bibliography_sha256': _sha(BIB)}
    for k in ('head_ok', 'sha_ok', 'lines_ok', 'main_and_bib_unmodified_in_clone'):
        if not out[k]:
            _fail(f'manuscript identity: {k} is False ({out})')
    return out


def main_lines():
    return _read(MAIN_TEX).split('\n')


def find_lines(lines, needle, lo=1, hi=None):
    hi = hi or len(lines)
    return [i for i in range(lo, hi + 1) if needle in lines[i - 1]]


def ms_anchor(lines, needle, lo=1, hi=None, required=True):
    hits = find_lines(lines, needle, lo, hi)
    if not hits and required:
        _fail(f'manuscript anchor not found in l. {lo}-{hi}: {needle!r}')
    return {'needle': needle, 'lines': hits}


def region(lines, start_needle, end_regex):
    s = next(i for i, ln in enumerate(lines, 1) if start_needle in ln)
    e = next(i for i in range(s + 1, len(lines) + 1) if re.match(end_regex, lines[i - 1]))
    return s, e - 1


# ======================================================================================================================
#  (a) the certification rule
# ======================================================================================================================
SC1, SC2, SC5, SC6 = ('settling_criterion.py', 'settling_criterion_v2.py', 'settling_criterion_v5.py',
                      'settling_criterion_v6.py')
H118, H101, H142 = ('p515_s53_w118_resettle_hooks.py', 'p515_s53_w101_settling_continuation_hooks.py',
                    'p515_s53_w142_resettle_v6_hooks.py')
C132 = 'p515_s53_w132_resettle_v3_campaign.py'


def task_a(lines):
    app_s, app_e = region(lines, r'\label{app:admm_updated_certification}', r'\s*\\subsection\{')
    s227_s, s227_e = region(lines, r'\label{subsubsec:certification}', r'\s*\\subsection\{')
    clauses = [
        ('holds', r'Throughout the regime acceleration is off, the tight interior-point tail is on and the penalty',
         [anchor(H118, 'def held(self, c):'), anchor(H118, 'return self.first_pass is not None and c > self.first_pass'),
          anchor(H101, 'for every cycle > N the certifying regime HELD exactly as W98 did'),
          anchor(H142, "'pass (AA off, tight tail on, rho frozen); settling rule v6")]),
        ('holds_start', 'from the next cycle in a fresh run or from the stopping cycle of the earlier run',
         [anchor(H118, 'return self.first_pass is not None and c > self.first_pass'),
          anchor(H101, 'for every cycle > N the certifying regime HELD exactly as W98 did'),
          anchor(H118, 'GATED cells (the six Phase B neighbours and the F2 pair): cycles 1..k0 (k0 = the original')]),
        ('k0_reset_holds_stay', r'a later lapse of the residual test, or a cycle with a failed solve, resets $k_0$',
         [anchor(SC2, 'if q is None or not boyd:'), anchor(SC2, 'self.sign = SC1._SignState()'),
          anchor(H118, 'return self.first_pass is not None and c > self.first_pass')]),
        ('first_cycle_after_k0', r'at the first cycle $k^{\star} > k_0$ at which',
         [anchor(SC2, 'if k < self.k0 + K_EXCL:')]),
        ('C1', r'\item[(C1)] \emph{Period}',
         [anchor(SC1, "a_parts = {'n_turning_points': len(T), 'at_least_3_turning_points': len(T) >= 3,"),
          anchor(SC1, 'p_hat = T[-1][0] - T[-3][0]')]),
        ('C2', r'\item[(C2)] \emph{Damping}',
         [anchor(SC6, 'ok = all(A[j2] <= A[j1] for j1, j2 in pairs)'), anchor(SC6, 'kept = [i for i, a in enumerate(A) if a >= floor]')]),
        ('C3', r'\item[(C3)] \emph{Settling}',
         [anchor(SC1, 'return max(W_MIN, int(math.ceil(W_FACTOR * p_hat)))'),
          anchor(SC2, "a_parts.update({'P_hat': p_hat, 'W': w, 'window': [lo, k], 'window_inside_run': lo >= self.k0})"),
          anchor(SC2, "a_parts.update({'range': rng, 'range_le_tau': rng <= TAU, 'band': [mn, mx]})")]),
        ('C4', r'\item[(C4)] \emph{Consensus}',
         [anchor(SC2, 'GAP_BOUND = TAU / 2.0'), anchor(SC2, 'gap_ok = t_sum is not None and abs(t_sum) <= GAP_BOUND'),
          anchor(H118, 'term = w * pi * (xd[p] - zt[p])'),
          anchor(H118, "out[(n, y, d, p)] = (srp_module._expected_market_price(tso_model[y][d], network, p), w)"),
          anchor('shared_resources_planning.py', 'def _expected_market_price(model, network, p):')]),
        ('C5', r'\item[(C5)] \emph{Clean evidence}',
         [anchor(SC5, 'if exit_class == OPTIMAL_CLASS:'), anchor(SC5, 'elif attempt != PRIMARY_ATTEMPT:'),
          anchor(SC5, "elif all(float(metrics[m]) <= CLEAN_FACTOR * tols[METRIC_TABLE[m]['option']] for m in METRICS):"),
          anchor(SC5, "'network': {'tol': 1e-5, 'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-6},"),
          anchor(SC5, "'esso': {'tol': 1e-10, 'dual_inf_tol': 1.0, 'constr_viol_tol': 1e-4, 'compl_inf_tol': 1e-4},"),
          anchor(SC6, "hit = self._non_clean_in(a_parts['window'])")]),
        ('noise_step', r'a step smaller than $\tau/100$ carries no sign',
         [anchor(SC1, 'EPS0 = TAU / 100.0'), anchor(SC1, 'if abs(dq) < eps0:')]),
        ('noise_swing', r'a swing smaller than $\tau/10$ is treated as noise',
         [anchor(SC6, 'SWING_FLOOR = TAU / 10.0'), anchor(SC6, 'if swing < self.floor:'),
          anchor(SC6, 'self.j_change = self.jc_stack.pop()')]),
        ('k_excl', r'the first three cycles after $k_0$ are not counted',
         [anchor(SC1, 'K_EXCL = 3'), anchor(SC2, 'if k < self.k0 + K_EXCL:')]),
        ('monotone_window', r'over a window of $2P_{\max}$ cycles after $k_0$',
         [anchor(SC2, 'lo = k - self.l_mono + 1'), anchor(SC2, "'lo_ge_k0_plus_K_EXCL': lo >= self.k0 + K_EXCL,"),
          anchor(SC2, "'j_change': s.j_change, 'no_sign_change_in_window': s.j_change is None or s.j_change < lo,"),
          anchor(H118, 'P_MAX = 30')]),
        ('M1', r'(M1) its steps are decreasing',
         [anchor(SC2, 'steps = [abs(self._dq(c)) for c in range(lo, k + 1)]'),
          anchor(SC2, "'steps_decreasing': bool(max(steps) < EPS0 or m2 < m1)})")]),
        ('M2', r'(M2) its range is at most $\tau$',
         [anchor(SC2, "b_parts.update({'range': rng, 'range_le_tau': rng <= TAU, 'band': [mn, mx], 'max_abs_dq': max(steps),")]),
        ('M3', r'(M3) $|\text{last step}| \cdot 2P_{\max} \le \tau$',
         [anchor(SC2, "'last_step_times_L_le_tau': last * self.l_mono <= TAU, 'band': None}")]),
        ('M_C4_C5', r'and (C4)--(C5) hold on that window',
         [anchor(SC6, "hit = self._non_clean_in(b_parts['window'])"), anchor(SC2, 'if cert_a or cert_b:')]),
        ('branch_priority', 'is certified instead on the monotone branch',
         [anchor(SC6, 'cert_b = (not cert_a) and b_ok')]),
        ('uncertified_gap', r'its consensus gap is $|t_{\text{sum}}|$ at the cap',
         [anchor(C132, "'gap': None if certified or t is None else abs(t),")]),
        ('uncertified_slack', 'its settling slack is the movement of the objective between the certification cycle',
         [anchor(C132, "rep.update({'Q_N_old': q_n_old, 's_signed': (q_end - q_n_old) if q_end is not None else None,"),
          anchor(C132, "'slack': None if certified else (abs(s) if (rep.get('gated') and s is not None) else None),")]),
    ]
    rows = []
    for cid, needle, code in clauses:
        ms = ms_anchor(lines, needle, app_s, app_e)
        rows.append({'clause': cid, 'manuscript': ms, 'code': code})
    # the accepted-text clause that is absent (paragraphs_v5: an evaluation without an earlier certificate has no slack)
    p5 = _read(PARAGRAPHS_V5)
    acc = 'An evaluation without one has no defined slack'
    absent = {'paragraphs_v5_has_it': acc in p5, 'main_tex_lines': find_lines(lines, 'no defined slack')
              + find_lines(lines, 'without one has no')}
    # section 2.2.7 summary sentences
    summary = []
    for sid, needle in [
            ('residual_test', 'pass the absolute-plus-relative test of Boyd et al.'),
            ('successful_solves', 'in a cycle in which every local solve is successful'),
            ('regime_opened', 'so the residual pass only opens a certifying regime, with the coordination settings held fixed'),
            ('period', 'the oscillation must have a measured period'),
            ('growth', 'must not be growing (the growth test)'),
            ('tau_over_period', r'must stay within a tolerance $\tau$ over that period'),
            ('gap', r'must be within $\tau/2$ (the gap clause)'),
            ('clean', 'the evidence must come from cleanly solved cycles'),
            ('monotone', 'is certified instead on a monotone branch that bounds its remaining descent'),
            ('pointer', r'The rule is stated in full in \ref{app:admm_updated_certification}'),
            ('uncertified', 'is reported in an uncertified form, with its consensus gap at the cap and its settling slack'),
            ('determinacy_certified', r'when it is at least $\max\{3\,b, 2\tau\}$'),
            ('determinacy_uncertified', 'exceeds three times the largest of their consensus gaps and settling slacks, in both gross and gap-corrected terms')]:
        summary.append({'id': sid, 'manuscript': ms_anchor(lines, needle, s227_s, s227_e)})
    det_code = [anchor(SC6, 'return abs(margin) >= thr, thr,'), anchor(C132, 'bar = 3.0 * max(comps)'),
                anchor(C132, 'det = abs(d_q) > bar and m_cc > bar')]
    # every statement of when the holds start, anywhere in Appendix A
    hold_statements = []
    for needle in [r'are frozen from the cycle after the first residual pass, or from the stopping cycle of the earlier run',
                   'the holds start at the first passing cycle',
                   r"from the cycle after $k_0$ on (or after the earlier run's stopping cycle in a continued evaluation)",
                   r'the certifying regime holds the tail on from the cycle after $k_0$',
                   'from the next cycle in a fresh run or from the stopping cycle of the earlier run']:
        hold_statements.append(ms_anchor(lines, needle))
    # the two gated cells whose N_old + 100 cap exceeds 300 (fresh-run formula would give 300), from the v6 spec
    spec = _json(SPEC_V6)
    above = spec['stop_rule']['caps'].get('above_300')
    gated = {c: {'N_old': spec['cells'][c]['N_old'], 'cap_rule': spec['cells'][c]['cap_rule']}
             for c in spec['cell_order'] if spec['cells'][c]['gated']}
    ungated = [c for c in spec['cell_order'] if not spec['cells'][c]['gated']]
    dec = {}
    for c in ('l_2ab0ce2d', 'l_b2251bc5'):
        root = os.path.join(P53, 'w142_resettle_v6', f'campaign_s53_w142_resettle_v6_{c}', 'evals')
        sub = [d for d in sorted(os.listdir(root)) if d.endswith(c)]
        d = _json(os.path.join(root, sub[0], 'resettle_decision.json'))
        dec[c] = {k: d.get(k) for k in ('status', 'k_star', 'k_cap', 'k0', 'N', 'cap', 'cap_mode', 'branch', 'reasons')}
        dec[c]['path'] = os.path.join(root, sub[0], 'resettle_decision.json')
        dec[c]['sha256'] = _sha(dec[c]['path'])
    # every uncertified decision of the three v6 campaigns: is it gated (an earlier run exists, so the slack is defined)?
    uncert = []
    for root, sp in ((os.path.join(P53, 'w142_resettle_v6'), SPEC_V6), (os.path.join(P53, 'w142_resettle_ext_v6'), SPEC_EXT),
                     (os.path.join(P53, 'w155_a64_cells'), SPEC_A64)):
        cells = _json(sp)['cells']
        for c in cells:
            croot = cells[c].get('campaign_root')
            if not croot or not os.path.isdir(os.path.join(croot, 'evals')):
                continue
            for sub in sorted(os.listdir(os.path.join(croot, 'evals'))):
                f = os.path.join(croot, 'evals', sub, 'resettle_decision.json')
                if os.path.isfile(f):
                    dd = _json(f)
                    if dd.get('status') != 'certified':
                        uncert.append({'cell': c, 'cap_mode': dd.get('cap_mode'), 'k_cap': dd.get('k_cap'),
                                       'reasons': dd.get('reasons'), 'path': f, 'sha256': _sha(f)})
    return {'appendix_region': [app_s, app_e], 'section_227_region': [s227_s, s227_e], 'clauses': rows,
            'uncertified_decisions_v6_ext_a64': uncert,
            'all_uncertified_are_gated': all(u['cap_mode'] == 'fixed' for u in uncert),
            'accepted_clause_absent': absent, 'section_227_summary': summary, 'determinacy_code': det_code,
            'hold_statements': hold_statements,
            'v6_spec': {'path': SPEC_V6, 'sha256': _sha(SPEC_V6), 'caps_above_300': above,
                        'holds_after_first_residual_pass': spec['holds_after_first_residual_pass'],
                        'N_and_holds_and_dynamic_cap': spec['stop_rule']['N_and_holds_and_dynamic_cap'],
                        'n_gated': len(gated), 'n_ungated': len(ungated), 'ungated': ungated,
                        'gated_k0_is_N_old_minus_9_on_the_two_cells': {c: dec[c]['k0'] == spec['cells'][c]['N_old'] - 9
                                                                        for c in dec}},
            'cells_with_cap_above_300': dec}


# ======================================================================================================================
#  (b) settings
# ======================================================================================================================
def _num(x):
    return float(x)


def task_b(lines):
    s36 = region(lines, r'\label{sec:case_settings}', r'\s*% =====')
    a7 = region(lines, r'\label{app:admm_updated_settings}', r'\s*\\section\{')
    p = _json(SRP1_PARAMS)
    ess = _json(ESS_PARAMS)
    c9 = _json(CASE9)
    c33 = [_json(f) for f in CASE33]
    s3 = _json(SPEC_3X3)
    s47, s51, s53 = _json(SPEC_S47), _json(SPEC_S51), _json(SPEC_S53)
    v6 = _json(SPEC_V6)
    log1 = _read(V6_LOG_SRP1)
    log3 = _read(V6_LOG_3X3)
    k1 = float(re.search(r'al_scale_esso=([0-9.e+]+)', log1).group(1))
    k3 = float(re.search(r'al_scale_esso=([0-9.e+]+)', log3).group(1))
    ipopt_banner = sorted(set(re.findall(r'Ipopt (\d+\.\d+\.\d+)', log1)))
    S = []

    def add(sid, where, needle, printed, source, source_value, ok, note=None):
        reg = s36 if where == '3.6' else a7
        hits = find_lines(lines, needle, reg[0], reg[1])
        if not hits:
            _fail(f'setting {sid}: needle not found in {where}: {needle!r}')
        S.append({'id': sid, 'region': where, 'needle': needle, 'lines': hits, 'printed': printed,
                  'source': source, 'source_value': source_value, 'match': None if ok is None else bool(ok), 'note': note})
        if ok is False or (ok is not None and not ok):
            _log(f'setting {sid}: printed {printed!r} vs source {source_value!r} -- MISMATCH')

    # ---- Appendix A, Solver and Coordination Settings --------------------------------------------------------------
    add('ipopt_version', 'A.7', r'IPOPT~3.14.18', '3.14.18', {'file': V6_LOG_SRP1, 'what': 'IPOPT banner'},
        ipopt_banner, ipopt_banner == ['3.14.18'])
    add('pyomo_version', 'A.7', r'Pyomo~6.9.5', '6.9.5', {'what': 'importlib.metadata.version("pyomo") in the canonical env'},
        importlib.metadata.version('pyomo'), importlib.metadata.version('pyomo') == '6.9.5')
    pyv = '.'.join(str(x) for x in sys.version_info[:3])
    add('python_version', 'A.7', r'Python~3.11.11', '3.11.11', {'what': 'sys.version_info of the canonical interpreter'},
        pyv, pyv == '3.11.11')
    add('network_linear_solver', 'A.7', 'the network models with the MA97 linear solver', 'ma97',
        {'files': [CASE9] + list(CASE33), 'field': 'solver.options.linear_solver'},
        [c9['solver']['options']['linear_solver']] + [c['solver']['options']['linear_solver'] for c in c33],
        all(x['solver']['options']['linear_solver'] == 'ma97' for x in [c9] + c33))
    add('esso_linear_solver', 'A.7', 'the storage agent with MA57', 'ma57', {'file': ESS_PARAMS, 'field': 'solver.options.linear_solver'},
        ess['solver']['options']['linear_solver'], ess['solver']['options']['linear_solver'] == 'ma57')
    th = s3.get('thread_caps')
    add('single_threaded', 'A.7', 'single-threaded throughout', 'single-threaded',
        {'file': SPEC_3X3, 'field': 'thread_caps', 'also': 'the v6 cell campaign specs carry the same field'}, th,
        th and all(v == '1' for v in th.values()))
    add('network_tol', 'A.7', r'a convergence tolerance of $10^{-5}$', 1e-5, {'files': [CASE9] + list(CASE33), 'field': 'solver.options.tol'},
        [x['solver']['options']['tol'] for x in [c9] + c33], all(x['solver']['options']['tol'] == 1e-5 for x in [c9] + c33))
    add('network_acceptable_tol', 'A.7', r'an acceptable tolerance of $10^{-4}$', 1e-4,
        {'files': [CASE9] + list(CASE33), 'field': 'solver.options.acceptable_tol'},
        [x['solver']['options']['acceptable_tol'] for x in [c9] + c33],
        all(x['solver']['options']['acceptable_tol'] == 1e-4 for x in [c9] + c33))
    dso_cit = [x['solver']['options'].get('compl_inf_tol') for x in c33]
    add('network_compl_inf_tol', 'A.7', r'a complementarity tolerance of $5 \times 10^{-4}$ (TSO) and $10^{-4}$ (DSOs)',
        {'TSO': 5e-4, 'DSO': 1e-4}, {'file': CASE9, 'field': 'solver.options.compl_inf_tol',
                                     'DSOs': 'not passed in case33_* -> IPOPT 3.14 default 1e-4 (settling_criterion_v5.TOLERANCE_SOURCES)',
                                     'anchor': anchor(SC5, "'compl_inf_tol': ('passed by the convergence-depth tail: 1e-6")},
        {'TSO': c9['solver']['options']['compl_inf_tol'], 'DSO_passed': dso_cit},
        c9['solver']['options']['compl_inf_tol'] == 5e-4 and all(v is None for v in dso_cit))
    add('network_max_iter', 'A.7', 'and at most 500 iterations', 500, anchor('network.py', "options['max_iter'] = 500"), 500, True)
    add('tail_compl', 'A.7', r'the tight tail sets the complementarity tolerance to $10^{-6}$', 1e-6,
        {'file': SPEC_V6, 'field': 'cells.*.configuration.convergence_depth_tail.compl_inf_tol'},
        v6['cells'][v6['cell_order'][0]]['configuration']['convergence_depth_tail'],
        v6['cells'][v6['cell_order'][0]]['configuration']['convergence_depth_tail'] == {'compl_inf_tol': 1e-6, 'enabled': True})
    add('esso_tol', 'A.7', r'The storage agent solves to a tolerance of $10^{-10}$ with an acceptable tolerance of $10^{-9}$',
        {'tol': 1e-10, 'acceptable_tol': 1e-9}, anchor('shared_energy_storage_data.py', "ESSO_TOL_OVERRIDES = {'tol': 1e-10, 'acceptable_tol': 1e-9}"),
        {'tol': 1e-10, 'acceptable_tol': 1e-9}, True,
        note='case file tol 1e-6 / acceptable_tol 1e-5 are overridden for every ESSO solve (option_overrides)')
    add('retry_triggers', 'A.7', 'A solve that ends at the iteration limit, infeasible or in a solver error is retried',
        ['maxIterations', 'infeasible', 'internalSolverError'],
        [anchor('network.py', 'po.TerminationCondition.maxIterations,'),
         anchor('shared_energy_storage_data.py', 'po.TerminationCondition.maxIterations,')],
        ['maxIterations', 'infeasible', 'internalSolverError'], True)
    add('recovery_tso', 'A.7', r'TSO: acceptable tolerance $10^{-4}$ after one acceptable iteration',
        {'acceptable_tol': 1e-4, 'acceptable_iter': 1}, {'file': CASE9, 'field': 'solver.recovery_options'},
        c9['solver']['recovery_options'], c9['solver']['recovery_options'] == {'acceptable_tol': 1e-4, 'acceptable_iter': 1})
    rec33 = [x['solver'].get('recovery_options') for x in c33]
    add('recovery_dso', 'A.7', 'DSOs: their primary settings', 'primary settings (cold)',
        {'files': list(CASE33), 'field': 'solver.recovery_options', 'code': anchor('network.py', "recovery_options['warm_start_init_point'] = 'no'")},
        rec33, all(not r for r in rec33), note='case33_1 has no recovery_options key, case33_2/3 an empty dict')
    add('recovery_esso', 'A.7', r'storage agent: acceptable tolerance $10^{-9}$ after one iteration',
        {'acceptable_tol': 1e-9, 'acceptable_iter': 1},
        {'file': ESS_PARAMS, 'field': 'solver.recovery_options', 'override': anchor('shared_energy_storage_data.py', 'recovery_options.update(esso_option_overrides)')},
        {'case_file': ess['solver']['recovery_options'], 'after_override_acceptable_tol': 1e-9},
        ess['solver']['recovery_options'] == {'acceptable_tol': 1e-4, 'acceptable_iter': 1})
    add('recovery_tier2', 'A.7', 'and then once more with the adaptive barrier strategy', "mu_strategy='adaptive'",
        [anchor('network.py', "tier2_options['mu_strategy'] = 'adaptive'"),
         anchor('shared_energy_storage_data.py', "tier2_options['mu_strategy'] = 'adaptive'")], 'adaptive', True)
    add('sigma', 'A.7', r'common objective scale $\sigma = 93{,}635{,}360$', 93635360.0,
        {'file': SRP1_PARAMS, 'field': 'admm.objective_scale', 'line': locate(SRP1_PARAMS, '"objective_scale"')},
        p['admm']['objective_scale'], p['admm']['objective_scale'] == 93635360.0)
    add('s_ref', 'A.7', r'reference converter rating $S^{\text{ref}} = 2.5$~MVA', 2.5,
        {'file': SRP1_PARAMS, 'field': 'admm.shared_ess_reference_rating_mva',
         'line': locate(SRP1_PARAMS, '"shared_ess_reference_rating_mva"')},
        p['admm']['shared_ess_reference_rating_mva'], p['admm']['shared_ess_reference_rating_mva'] == 2.5)
    add('interface_ratings', 'A.7', r'$R^{\text{I}} = 2.0$, $1.0$ and $1.5$~p.u. (200, 100 and 150~MVA', [2.0, 1.0, 1.5],
        {'audit': 'P5_15_W172_APPENDIX_A_AUDIT.md section "R^I for section 3.5" (case33_<i>_<year>.json transformers[1].rating; network.get_interface_branch_rating)',
         'line': locate('P5_15_W172_APPENDIX_A_AUDIT.md', '## R^I for')}, [2.0, 1.0, 1.5], True,
        note='not re-derived here; W172 read the case JSONs and the per-cycle worst_pf_primal_rating')
    add('kappa', 'A.7', r'$227{,}211$ at the single-scenario instance, $386{,}259$ at the multi-scenario instance',
        {'single': 227211, 'multi': 386259}, {'logs': [V6_LOG_SRP1, V6_LOG_3X3], 'line': '[ADMM ESSO AL SCALE] ... al_scale_esso=',
                                             'mode': p['admm']['esso_al_scale']},
        {'single': k1, 'multi': k3}, round(k1) == 227211 and round(k3) == 386259)
    rho = p['admm']['rho']
    add('rho0', 'A.7', r'initial penalties $\rho^{V}_0 = 0.0077$, $\rho^{\text{PF}}_0 = 0.198$, $\rho^{\text{E}}_0 = 0.01$',
        {'v': 0.0077, 'pf': 0.198, 'ess': 0.01}, {'file': SRP1_PARAMS, 'field': 'admm.rho'},
        {'v': sorted(set(rho['v'].values())), 'pf': sorted(set(rho['pf'].values())), 'ess': sorted(set(rho['ess'].values()))},
        set(rho['v'].values()) == {0.0077} and set(rho['pf'].values()) == {0.198} and set(rho['ess'].values()) == {0.01})
    pu = p['admm']['penalty_update']
    add('residual_balancing', 'A.7', 'residual balancing with ratio 5 (3 for the decrease of',
        {'ratio': 5, 'ratio_pf_decrease': 3, 'factor': 1.5, 'bounds': [1e-4, 1e4], 'freeze_after': 10, 'backstop': 200,
         'ess_exempt': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}},
        {'file': SRP1_PARAMS, 'field': 'admm.penalty_update', 'line': locate(SRP1_PARAMS, '"penalty_update"')}, pu,
        pu['residual_balance_ratio'] == 5.0 and pu['residual_balance_ratio_pf_decrease'] == 3.0
        and pu['increase_factor'] == 1.5 and pu['decrease_factor'] == 1.5 and pu['min'] == 1e-4 and pu['max'] == 1e4
        and pu['freeze_after_unchanged_cycles'] == 10 and pu['freeze_backstop_cycle'] == 200
        and pu['balancing_exempt_until'] == {'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}})
    add('boyd_tolerances', 'A.7', r'residual tolerances $\varepsilon_{\text{abs}} = 10^{-5}$, $\varepsilon_{\text{rel}} = 10^{-4}$',
        {'eps_abs': 1e-5, 'eps_rel': 1e-4}, {'file': SRP1_PARAMS, 'field': 'admm.tol.boyd'}, p['admm']['tol']['boyd'],
        p['admm']['tol']['boyd'] == {'eps_abs': 1e-5, 'eps_rel': 1e-4})
    aa = p['admm']['anderson_acceleration']
    add('anderson', 'A.7', r'Anderson acceleration with memory 5 and regularisation $10^{-10}$', {'memory': 5, 'regularization': 1e-10},
        {'file': SRP1_PARAMS, 'field': 'admm.anderson_acceleration'}, aa, aa['memory'] == 5 and aa['regularization'] == 1e-10)
    add('production_exit_A7', 'A.7', 'The production exit is ten consecutive passing cycles, with a cap of 500 cycles at the multi-scenario instance',
        {'consecutive': 10, 'cap_multi': 500}, {'files': [SRP1_PARAMS, SPEC_3X3],
                                                'fields': ['admm.minimum_consecutive_converged_cycles', 'cap', 'required_consecutive_cycles']},
        {'case_file': p['admm']['minimum_consecutive_converged_cycles'], 'spec_cap': s3['cap'],
         'spec_required': s3['required_consecutive_cycles']},
        p['admm']['minimum_consecutive_converged_cycles'] == 10 and s3['cap'] == 500 and s3['required_consecutive_cycles'] == 10)
    add('voltage_regularisation', 'A.7', r'the solver-side voltage regularisation weight is $9 \times 10^{4}$', 9e4,
        anchor('definitions.py', 'PENALTY_SCENARIO_DEVIATION = 9e4'), 9e4, True)
    # ---- Section 3.6 -------------------------------------------------------------------------------------------------
    alpha = None
    for c in s3['candidates']:
        alpha = (c.get('interface_deviation_premium') or {}).get('alpha', alpha)
    add('alpha', '3.6', r'the commitment premium of Subsection~\ref{subsubsec:commitment} is $\alpha = 0.5$', 0.5,
        {'file': SPEC_3X3, 'field': 'candidates[].interface_deviation_premium.alpha'}, alpha, alpha == 0.5)
    add('delta_R_V_tau', '3.6', r'uses $\delta_R = 0.07$ on the reference value $\mathcal{V} = 259{,}375.33$',
        {'delta_R': 0.07, 'V': 259375.33, 'tau': 4539.07},
        [anchor(SC1, 'DELTA_R = 0.07'), anchor(SC1, 'R_REF = 259375.33'), anchor(SC1, 'TAU = DELTA_R * R_REF / 4.0')],
        {'tau': 0.07 * 259375.33 / 4}, round(0.07 * 259375.33 / 4, 2) == 4539.07)
    add('p_max', '3.6', r'and $P_{\max} = 30$ cycles', 30, anchor(H118, 'P_MAX = 30'), 30, True)
    add('caps', '3.6', r'an evaluation continued from an earlier run is capped 100 cycles after that run',
        {'continued': 'N_old + 100', 'fresh': 'min(k0 + 109, 300)'},
        {'file': SPEC_V6, 'field': 'stop_rule.caps', 'code': [anchor(SC2, 'CAP_AFTER_K0 = 109'), anchor(SC2, 'CAP_CEILING = 300')]},
        v6['stop_rule']['caps'], v6['stop_rule']['caps']['gated'].startswith('N_old + 100')
        and v6['stop_rule']['caps']['ungated'].startswith('min(k0_run + 109, 300)'))
    pd47, pd51, pd53 = s47['extra']['poll_design'], s51['extra']['poll_design'], s53['extra']['poll_design']
    add('delta0', '3.6', r'used $\Delta_0 = 4$ lattice units', 4, {'files': [SPEC_S47, SPEC_S51], 'field': 'extra.poll_design.delta_0'},
        [pd47['delta_0'], pd51['delta_0']], pd47['delta_0'] == 4 and pd51['delta_0'] == 4)
    add('budgets', '3.6', 'budgets of 20 new evaluations for the two variant-A searches and 60 for the variant-B certificate',
        {'s47': 20, 's51': 20, 's53': 60}, {'files': [SPEC_S47, SPEC_S51, SPEC_S53], 'field': 'extra.poll_design.max_new_evaluations'},
        [pd47['max_new_evaluations'], pd51['max_new_evaluations'], pd53['max_new_evaluations']],
        [pd47['max_new_evaluations'], pd51['max_new_evaluations'], pd53['max_new_evaluations']] == [20, 20, 60])
    add('completion_cap', '3.6', 'a completion cap of 30 points', 30, {'files': [SPEC_S47, SPEC_S51, SPEC_S53], 'field': 'extra.poll_design.completion_cap'},
        [pd47['completion_cap'], pd51['completion_cap'], pd53['completion_cap']],
        [pd47['completion_cap'], pd51['completion_cap'], pd53['completion_cap']] == [30, 30, 30])
    sq = [s47['extra']['sigma_Q']['sigma_Q_eur'], s51['extra']['sigma_Q']['sigma_Q_eur'], s53['extra']['sigma_Q']['sigma_Q_eur']]
    add('sigma_Q', '3.6', r'the measured resolution $\sigma_Q = 18{,}449.66$', 18449.66,
        {'files': [SPEC_S47, SPEC_S51, SPEC_S53], 'field': 'extra.sigma_Q.sigma_Q_eur'}, sq, all(round(x, 2) == 18449.66 for x in sq))
    add('production_exit_3_6', '3.6', 'The search evaluations ran under the production exit (ten consecutive passing cycles, cap 500 cycles)',
        {'consecutive': 10, 'cap': 500}, {'files': [SPEC_S47, SPEC_S51, SPEC_S53], 'fields': ['cap', 'required_consecutive_cycles']},
        [[s['cap'], s['required_consecutive_cycles']] for s in (s47, s51, s53)],
        all(s['cap'] == 500 and s['required_consecutive_cycles'] == 10 for s in (s47, s51, s53)))
    add('variant_B_unit', '3.6', 'variant B polled at unit size throughout', {'delta': 1}, {'file': SPEC_S53, 'field': 'extra.poll_design.delta / on_success'},
        {'delta': pd53['delta'], 'on_success': pd53['on_success']}, pd53['delta'] == 1)
    add('review_stops', '3.6', 'when two barrier points occurred in one poll or three in all, and ran at most 60 polls',
        {'barrier': [2, 3], 'max_polls': 60}, {'files': [SPEC_S47, SPEC_S51, SPEC_S53], 'fields': ['extra.stop_rule', 'extra.poll_design.max_polls']},
        {'max_polls': [pd47['max_polls'], pd51['max_polls'], pd53['max_polls']], 'barrier_stop_s53': pd53.get('barrier_stop'),
         'stop_rule_text_s47': s47['extra']['stop_rule'][:60]},
        [pd47['max_polls'], pd51['max_polls'], pd53['max_polls']] == [60, 60, 60] and pd53.get('barrier_stop') == [2, 3]
        and '>= 2 NEW barrier evaluations in one poll or >= 3 overall' in s47['extra']['stop_rule'])
    add('cache_seed', '3.6', 'the cache was seeded with the single-node ladders of the initial design',
        'single-node ladders, with and without the budget, and their combinations',
        {'file': SPEC_S47, 'field': 'extra.cache_sources_accepted[].campaign_id'},
        [c.get('campaign_id') for c in s47['extra']['cache_sources_accepted']], None,
        note='partially read: s47 cache = s47_a1a_baseline + s47_recert; "with and without the budget, and their '
             'combinations" not verified (see Not confirmed)')
    add('benchmark_curtailment_term', '3.6', r'by a minimum-curtailment term of 1~\euro/MWh', 1.0,
        anchor('p515_s53_w116_benchmark_nrf.py', 'tie-breaker 1 EUR/MWh'), 1.0, True)
    add('benchmark_starts', '3.6', 'and reported as the best of three solver starts', 3,
        anchor('p515_s53_w116_benchmark_spec_v3.py', 'passive and price-taker x three starts'), 3, True)
    # ---- duplicates: the same setting stated in both regions, and elsewhere --------------------------------------------
    dup_needles = {
        'production_exit_10': ['ten consecutive passing cycles', 'ten consecutive cycles passing'],
        'cap_500': ['cap 500', 'cap of 500'],
        'tail_1e-6': [r'$10^{-6}$'],
        'eps_abs_rel': [r'\varepsilon_{\text{abs}} = 10^{-5}'],
        'rb_ratio_1.5': [r'1.5\,\rho_g', 'factor 1.5'],
        'rb_bounds': [r'[10^{-4}, 10^{4}]'],
        'aa_memory_5': ['memory 5'],
        'retry_triggers': ['iteration limit, infeasible or in a solver error'],
        'adaptive_barrier': ['adaptive barrier strategy'],
        'single_threaded': ['single-threaded'],
        'ipopt_version': ['3.14.18'],
        'tau_value': ['4{,}539.07'],
        'sigma_Q_value': ['18{,}449.66'],
        'delta0': [r'\Delta_0 = 4'],
        'alpha_value': [r'\alpha = 0.5'],
    }
    occ = {}
    for k, needles in dup_needles.items():
        allh = sorted({i for nd in needles for i in find_lines(lines, nd)})
        occ[k] = {'needles': needles, 'in_3_6': [i for i in allh if s36[0] <= i <= s36[1]],
                  'in_A7': [i for i in allh if a7[0] <= i <= a7[1]],
                  'elsewhere': [i for i in allh if not (s36[0] <= i <= s36[1] or a7[0] <= i <= a7[1])]}
    # ---- omissions: settings of the frozen configuration printed in neither region (the Worker's list, anchored) -------
    omissions = [
        {'id': 'acceptable_iter_primary', 'value': 5, 'source': [{'file': CASE9, 'line': locate(CASE9, '"acceptable_iter": 5')},
                                                                   {'file': ESS_PARAMS, 'line': locate(ESS_PARAMS, '"acceptable_iter": 5')}],
         'printed_anywhere': find_lines(lines, 'acceptable iteration') + find_lines(lines, 'acceptable_iter')},
        {'id': 'esso_max_iter', 'value': 500, 'source': [anchor('shared_energy_storage_data.py', "options['max_iter'] = 500")],
         'printed_anywhere': find_lines(lines, '500 iterations')},
        {'id': 'network_bound_push_frac', 'value': {'TSO': 1e-6, 'DSO': 1e-5},
         'source': [{'file': CASE9, 'line': locate(CASE9, '"bound_push": 1e-6')}, {'file': CASE33[0], 'line': locate(CASE33[0], '"bound_push": 1e-5')}],
         'printed_anywhere': find_lines(lines, 'bound_push') + find_lines(lines, 'bound push')},
        {'id': 'warm_start_primary', 'value': "warm_start_init_point 'yes' on warm-started primary solves",
         'source': [anchor('network.py', "solver.options['warm_start_init_point'] = 'yes'")],
         'printed_anywhere': find_lines(lines, 'warm') },
        {'id': 'single_scenario_production_cap_300', 'value': 300, 'source': [{'file': SRP1_PARAMS, 'line': locate(SRP1_PARAMS, '"num_max_iters": 300')}],
         'printed_anywhere': find_lines(lines, '300\\}') + find_lines(lines, ' 300 ')},
        {'id': 'proximal_tso_weight_zero', 'value': 0.0, 'source': [{'file': SRP1_PARAMS, 'line': locate(SRP1_PARAMS, '"tau": 0.0')}],
         'printed_anywhere': find_lines(lines, 'run with weight zero')},
        {'id': 'aa_reject_keep_memory', 'value': 'keep_memory', 'source': [{'file': SRP1_PARAMS, 'line': locate(SRP1_PARAMS, '"reject_policy": "keep_memory"')}],
         'printed_anywhere': find_lines(lines, 'on rejection the memory is kept')},
        {'id': 'esso_initialisation_standalone', 'value': 'standalone', 'source': [{'file': SRP1_PARAMS, 'line': locate(SRP1_PARAMS, '"shared_ess_initialization": "standalone"')}],
         'printed_anywhere': find_lines(lines, "solve it standalone with its powers fixed")},
        {'id': 'objective_change_tests_diagnostic', 'value': p['admm']['tol']['objective'],
         'source': [{'file': SRP1_PARAMS, 'line': locate(SRP1_PARAMS, '"objective": {"abs": 1e3')}],
         'printed_anywhere': find_lines(lines, 'Objective-change tests are computed for diagnosis only')},
    ]
    # ---- references that send the reader to section 3.6 for settings now in Appendix A.7 ------------------------------
    refs = []
    for i, ln in enumerate(lines, 1):
        if r'\ref{sec:case_settings}' in ln:
            refs.append({'line': i, 'context': re.sub(r'\s+', ' ', ln.strip())[:600]})
    return {'region_3_6': list(s36), 'region_A7': list(a7), 'settings': S, 'occurrences': occ, 'omissions': omissions,
            'refs_to_sec_case_settings': refs,
            'spec_hashes': {f: _sha(f) for f in (SPEC_V6, SPEC_EXT, SPEC_A64, SPEC_3X3, SPEC_S47, SPEC_S51, SPEC_S53,
                                                 SRP1_PARAMS, ESS_PARAMS, CASE9) + CASE33}}


# ======================================================================================================================
#  (c) the storage agent
# ======================================================================================================================
SED = 'shared_energy_storage_data.py'
SRP = 'shared_resources_planning.py'


def task_c(lines):
    eq = ms_anchor(lines, r'\label{eq:esso_objective}')
    cons = ms_anchor(lines, r'subject to \eqref{eq:cohort_rated}--\eqref{eq:available_capacity}, \eqref{eq:esso_net_power}--\eqref{eq:soh_floor}')
    code = {
        'objective_slack': anchor(SED, 'slack_penalty += PENALTY_ESSO_SLACK * (model.slack_es_pnet_up[y_inv, d, p] + model.slack_es_pnet_down[y_inv, d, p])'),
        'objective_throughput': anchor(SED, 'throughput += model.es_pch_per_unit[y_inv, y, d, p] + model.es_pdch_per_unit[y_inv, y, d, p]'),
        'objective_eps': anchor(SED, 'slack_penalty += EPS_ESSO_THROUGHPUT * throughput'),
        'objective_is_penalty_only': anchor(SED, 'expr=model.feasibility_penalty,'),
        'salvage_not_in_objective': anchor(SED, 'model.salvage_credit = pe.Expression(expr=-model.salvage_value)'),
        'c_sigma': anchor('definitions.py', 'PENALTY_ESSO_SLACK = 1e3'),
        'eps_E': anchor('definitions.py', 'EPS_ESSO_THROUGHPUT = 1e-5'),
        'slacks_enabled_case_file': {'file': ESS_PARAMS, 'line': locate(ESS_PARAMS, '"slacks": true')},
        'AL_terms_scaled_by_kappa': [anchor(SRP, 'obj += models[node_id].admm_esso_al_scale * (models[node_id].dual_p_req[y, d, p] * constraint_p_req)'),
                                     anchor(SRP, 'obj += models[node_id].admm_esso_al_scale * ((models[node_id].rho / 2) * constraint_p_req ** 2)')],
        'AL_normalisation_2Sref': [anchor(SRP, 'constraint_p_req = (models[node_id].es_pnet[y, d, p] - models[node_id].p_req[y, d, p]) / (2 * shared_ess_rating)'),
                                   anchor(SRP, 'def _shared_ess_admm_normalization_mva(rating_mva, floor_mva, reference_mva=None):')],
        'base_objective_not_scaled': anchor_after(SRP, 'def update_shared_energy_storage_model_to_admm(',
                                                  'obj = copy(models[node_id].objective.expr)'),
    }
    rows = {
        'cohort_rated (eq:cohort_rated)': anchor(SED, 'model.rated_s_capacity_unit.add(model.es_s_rated_per_unit[y_inv, y] == model.es_s_investment_fixed[y_inv])'),
        'aggregate rated S^Rated, E^Rated (eq:master_rated_s form; not in the A.2 list)': anchor(SED, 'model.rated_s_capacity.add(model.es_s_rated[y] == total_s_capacity)'),
        'cohort_available (eq:cohort_available), end-of-block SoH': anchor(SED, 'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y])'),
        'cohort_available, mid-block variant (arm C3_midblock only)': anchor(SED, 'soh_mid = prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y] / 2.00) * (phi_cal ** (num_years / 2.00))'),
        'available_capacity aggregate (eq:available_capacity): computed after the solve, not a row': anchor(SED, 's_available, e_available = shared_ess_data.get_available_capacities(models[node_id], y)'),
        'esso_net_power with slack pair (eq:esso_net_power)': anchor(SED, 'model.energy_storage_operation_agg.add(model.es_pnet[y, d, p] == agg_pnet + model.slack_es_pnet_up[y, d, p] - model.slack_es_pnet_down[y, d, p])'),
        'cohort_allocation H3 rows (eq:cohort_allocation), N_active-1 active': anchor(SED, '== model.es_pnet_cohort_share_h3[y_inv, y] * agg_pnet)'),
        'cell_throughput (eq:cell_throughput)': anchor(SED, 'avg_ch_dch += (num_days / 365.00) * (eff_ch * pch * dt + pdch * dt / eff_dch)'),
        'cycling_loss (eq:cycling_loss)': anchor(SED, 'model.es_D_per_unit[y_inv, y] * (2 * shared_energy_storage.cl_eff * model.es_e_investment_fixed[y_inv])'),
        'cycling_loss, ageing off (arm no_ageing)': anchor(SED, 'model.es_D_per_unit[y_inv, y] == 0.00)'),
        'soh_chain (eq:soh_chain)': anchor(SED, '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))'),
        'soh_floor (eq:soh_floor)': anchor(SED, 'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'),
        'power bounds 0 <= Pch, Pdch <= S^Rated,Unit': anchor(SED, "_add_esso_cohort_constraint(model, 'energy_storage_limits', y_inv, y, pch <= s_max)"),
        'converter circle on the aggregate (bound: es_s_rated)': anchor(SED, 'model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2)'),
        'available power = rated power (so S^Av = S^Rated)': anchor(SED, 'model.available_s_capacity_unit.add(model.es_s_available_per_unit[y_inv, y] == model.es_s_rated_per_unit[y_inv, y])'),
        'retired complementarity rows (none built)': anchor(SED, '# P5.15-1 (Step 1 item 3): `energy_storage_complementarity`,'),
    }
    regularisation = {'min_pch_pdch_bound_text': ms_anchor(lines, r'at the certified points it was below $4 \times 10^{-5}$ of the rating'),
                      'record': 'REVISION_CONTEXT.md "Slacks at the certified point (W169)": <= 3.6e-5 S',
                      'record_line': locate('REVISION_CONTEXT.md', 'the min(pch, pdch) ESSO check is recorded, not enforced')}
    return {'eq_esso_objective': eq, 'constraint_list_sentence': cons, 'code_objective': code, 'code_rows': rows,
            'regularisation': regularisation}


# ======================================================================================================================
#  (d) nomenclature
# ======================================================================================================================
def my_sub(raw):
    s = W167.normalise_sub(raw)
    if s:
        return s
    x = re.sub(r'\s+', '', raw or '')
    m = re.fullmatch(r'\{?\\(?:text|mathrm|operatorname|textrm)\{([A-Za-z]+)\}\}?', x)
    if m:
        return m.group(1)
    m = re.fullmatch(r'\{?\\(max|min)\}?', x)
    if m:
        return m.group(1)
    return ''


def my_key(at):
    sup, _it = W167.normalise_sup(at.sup_raw)
    sup = sup.replace(r'\star', '*')                    # k^{\star} == k^{*}
    sub = my_sub(at.sub_raw)
    return at.base + (f'^{{{sup}}}' if sup else '') + (f'_{{{sub}}}' if sub else '')


def is_index(at):
    return at.base in W167.INDEX_LETTERS and not at.bold and not W167.normalise_sup(at.sup_raw)[0] and not my_sub(at.sub_raw)


# raw-form checks for nomenclature rows whose normalised key collides with another symbol
RAW_CHECKS = {
    r'\pi_t (wholesale price)': [r'\pi_t', r'\pi_{t}', r'\pi_{t,'],
    r'\bar{\pi}_t (expected price)': [r'\bar{\pi}_t', r'\bar{\pi}'],
    r'\lambda_t (TN marginal value at the interface)': [r'\lambda_t', r'\lambda_{t}', r'\lambda_{t,'],
    r'\lambda_a (agent dual)': [r'\lambda_a', r'\lambda_{a', r'\lambda^{\xi', r'\lambda^{V'],
    's (settling slack, uncertified form)': [r'|s|', r'slack $s$', r'slack~$s$', r'slack $s '],
    's (ADMM dual residual, A.3-A.4; not in the nomenclature)': [r'$s = \rho', r's_{\text{E}}', r's_g'],
    's = (m,o) (recourse scenario)': [r'$s = (m, o)$', r'\omega_s', r'_{i,s,t}', r's \in \Omega_M \times \Omega_O'],
    r'x_a (agent copy)': [r'x_a', r'x_{a', r'x_{\text{E}}'],
    r'\boldsymbol{x} (plan)': [r'\boldsymbol{x}'],
    r'z (global value)': [r'z_{e,y,d,t}', r'z^{\xi', r'- z^{', r'$z$'],
    r'\boldsymbol{z} (lattice coordinates)': [r'\boldsymbol{z}'],
    r'\Delta (poll size)': [r'$\Delta$', r'\Delta \gets', r'\Delta = ', r'\Delta \boldsymbol{v}'],
    r'\Delta t (time instant)': [r'\Delta t'],
    r'D (set of days)': [r'd \in D', r'\in D}'],
    r'D_d (days represented)': [r'D_d'],
    r'D_{e,y^Inv,y} (cycling loss)': [r'D_{e,y'],
    r'k_e (cycle life)': [r'k_e'],
    r'\sigma (common scale)': [r'\sigma$', r'{\sigma}', r'\sigma /', r'\sigma = 93'],
    r'w_{y,d} (block weight)': [r'w_{y,d}', r'w_b'],
    r'\alpha (commitment premium)': [r'\alpha \, \bar{\pi}_t', r'$\alpha = 0.5$', r'$\alpha$ an imbalance premium', r'fraction $\alpha$'],
    r'\alpha = 1/(2 S^ref) (A.3-A.4 storage-channel factor; not in the nomenclature)': [r'\alpha = 1/(2 S^{\text{ref}})', r'\rho^{\text{E}} \alpha', r'\lVert \alpha (x - z)'],
    'D (post-hoc settled descent, tab:res_multiscenario; not in the nomenclature)': [r'$D$ is the post-hoc', r'(V - D)', r'descent $D$'],
    'b (band, section 2.2.7) / b (channel normalisation, A.4) / b (fit slope, section 4.2)': [r'where $b$ is the larger', r'with $b$ the channel', r'a+bE+cP', r'e^{*}=b+c/4'],
    'R (ratio, tab:res_multiscenario; not in the nomenclature)': [r'the ratio $R$', r'$R$ at operational', r'$R$ after', r'$R$ predicted'],
    'V (value, tab:res_multiscenario; not in the nomenclature)': [r'$V / V^{\text{single}}$', r'(V - D)'],
}


def nomenclature_audit(lines_raw, label):
    clines = W167.strip_comments(lines_raw)
    ctext = '\n'.join(clines)
    nom_s, nom_e = W167.nomenclature_bounds(clines)
    entries = []
    for ln_no in range(nom_s, nom_e + 1):
        ln = clines[ln_no - 1].strip()
        if '&' not in ln or r'\textbf{Symbol}' in ln:
            continue
        c0, c1 = ln.split('&', 1)
        desc = re.sub(r'\\\\\s*$', '', c1).strip()
        desc = re.sub(r'\\textcolor\{[^}]*\}\{(.*?)\}', r'\1', desc).strip()
        maths = re.findall(r'(?<!\\)\$(.+?)(?<!\\)\$', c0)
        for mth in maths:
            ats = [t for t in W167.tokenize(mth) if not t.in_script]
            if not ats:
                continue
            head = ats[0]
            entries.append({'line': ln_no, 'tex': mth, 'key': my_key(head), 'index': is_index(head), 'description': desc})
    body = W167.body_atoms(ctext, clines, nom_s, nom_e)[0]
    body = [(ln, at) for ln, at in body if ln > nom_e]                 # the body after the nomenclature
    secmap = W167.section_map(lines_raw)
    used = {}
    for ln, at in body:
        if at.kind != 'symbol' or is_index(at):
            continue
        if at.in_script and at.base.isalpha() and len(at.base) > 1 and at.base not in W167.GREEK:
            continue          # a word inside a label subscript/superscript (t_{\text{sum}}, \varepsilon_{\text{abs}}): not a symbol
        used.setdefault(my_key(at), []).append(ln)
    defined = {}
    for e in entries:
        defined.setdefault(e['key'], []).append(e)
    unused = []
    for k, es in defined.items():
        if all(e['index'] for e in es):
            continue
        if k not in used:
            unused.append({'key': k, 'entries': [{'line': e['line'], 'tex': e['tex'], 'description': e['description']} for e in es]})
    undefined = []
    for k, lns in sorted(used.items(), key=lambda kv: kv[1][0]):
        if k in defined:
            continue
        if re.search(r'\^\{[+-]\}', k) and re.sub(r'\^\{[+-]\}', r'^{\\pm}', k) in defined:
            continue          # s^{+}, s^{-} are defined by the nomenclature's s^{\pm}
        undefined.append({'key': k, 'first_line': lns[0], 'count': len(lns), 'lines': sorted(set(lns)),
                          'section_first': secmap.get(lns[0])})
    dup = {k: [{'line': e['line'], 'tex': e['tex'], 'description': e['description']} for e in es]
           for k, es in defined.items() if len(es) > 1}
    body_text = '\n'.join(clines[nom_e:])
    raw = {}
    for name, pats in RAW_CHECKS.items():
        hits = sorted({nom_e + i for pat in pats for i, ln in enumerate(clines[nom_e:], 1) if pat in ln})
        raw[name] = {'patterns': pats, 'body_lines': hits}
    return {'label': label, 'nomenclature_lines': [nom_s, nom_e], 'n_entries': len(entries),
            'entry_tex': [e['tex'] for e in entries],
            'n_keys': len(defined), 'entries': entries, 'defined_unused': unused, 'used_undefined': undefined,
            'n_used_undefined': len(undefined), 'duplicate_keys': dup, 'raw_checks': raw,
            'n_body_atoms': len(body), 'body_chars': len(body_text), 'used_keys': sorted(used)}


def task_d(lines):
    cur = nomenclature_audit(lines, DECLARED_CLONE_HEAD)
    prev = {}
    for c in PRE_MOVE_COMMITS:
        txt = _git(['show', f'{c}:main.tex'], cwd=CLONE)
        prev[c] = nomenclature_audit(txt.split('\n'), c)
    # for every symbol unused now: was it used at the pre-move commits (normalised key and raw checks)?
    then = {}
    for u in cur['defined_unused']:
        then[u['key']] = {c: {'key_used_in_body_then': u['key'] in prev[c]['used_keys'],
                              'key_in_nomenclature_then': u['key'] in [e['key'] for e in prev[c]['entries']]}
                          for c in PRE_MOVE_COMMITS}
    pickaxe = {nd: _git(['log', '--format=%h', '-S' + nd, '--', 'main.tex'], cwd=CLONE).split()
               for nd in (r'|s|\}', r'\pi_t', r'\lambda_t', r'\mathcal{L}^{E,P}')}
    undef_now = {u['key'] for u in cur['used_undefined']}
    undef_diff = {c: {'undefined_now_not_then': sorted(undef_now - {u['key'] for u in prev[c]['used_undefined']}),
                      'undefined_then_not_now': sorted({u['key'] for u in prev[c]['used_undefined']} - undef_now)}
                  for c in PRE_MOVE_COMMITS}
    entries_diff = {c: {'only_now': sorted(set(cur['entry_tex']) - set(prev[c]['entry_tex'])),
                        'only_then': sorted(set(prev[c]['entry_tex']) - set(cur['entry_tex']))} for c in PRE_MOVE_COMMITS}
    raw_then = {name: {c: prev[c]['raw_checks'][name]['body_lines'] for c in PRE_MOVE_COMMITS}
                for name in RAW_CHECKS}
    return {'normalisation': ('W167 (p515_s53_w167_nomenclature_years: font wrappers dropped, bold dropped, '
                              'decorations kept, iteration markers removed from superscripts, index subscripts dropped '
                              'unless an upper-case label or digits, letter runs split except SoH/SoC/..., index letters '
                              'cdegijklmnosty set aside) with ONE change: a subscript that is a single font-wrapped word or '
                              '\\max/\\min is kept as a label (t_sum, n_w, eps_abs, P_max); every symbol of a multi-symbol '
                              'nomenclature cell is audited; body = comment-stripped math after the nomenclature'),
            'current': cur, 'pre_move': {c: {k: prev[c][k] for k in ('nomenclature_lines', 'n_entries', 'n_keys',
                                                                     'defined_unused', 'n_used_undefined', 'raw_checks')}
                                         for c in PRE_MOVE_COMMITS},
            'unused_now_status_then': then, 'raw_checks_then': raw_then, 'nomenclature_entries_diff': entries_diff,
            'git_log_S_in_clone': pickaxe, 'used_undefined_diff': undef_diff}


# ======================================================================================================================
#  (e) the ageing figure
# ======================================================================================================================
def _pdf_objects(b):
    return {int(n): body for n, body in re.findall(rb'(\d+) 0 obj(.*?)endobj', b, re.S)}


def _pdf_stream(body):
    m = re.search(rb'stream\r?\n(.*?)\r?\nendstream', body, re.S)
    if not m:
        return None
    data = m.group(1)
    return zlib.decompress(data) if b'/FlateDecode' in body else data


def _cmap(data):
    out = {}
    txt = data.decode('latin-1')
    for blk in re.findall(r'beginbfchar(.*?)endbfchar', txt, re.S):
        for a, u in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', blk):
            out[int(a, 16)] = bytes.fromhex(u).decode('utf-16-be')
    for blk in re.findall(r'beginbfrange(.*?)endbfrange', txt, re.S):
        for a, bnd, rest in re.findall(r'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(\[[^\]]*\]|<[0-9A-Fa-f]+>)', blk):
            lo, hi = int(a, 16), int(bnd, 16)
            if rest.startswith('['):
                us = re.findall(r'<([0-9A-Fa-f]+)>', rest)
                for i, u in enumerate(us):
                    out[lo + i] = bytes.fromhex(u).decode('utf-16-be')
            else:
                base = int(rest[1:-1], 16)
                for i in range(hi - lo + 1):
                    out[lo + i] = chr(base + i)
    return out


def _tokens(s):
    """PDF content-stream tokens: ('num', v) ('name', s) ('str', bytes) ('op', s) ('[', None) (']', None)."""
    i, n = 0, len(s)
    while i < n:
        c = s[i:i + 1]
        if c in b' \t\r\n\f\x00':
            i += 1
        elif c == b'%':
            while i < n and s[i:i + 1] not in b'\r\n':
                i += 1
        elif c == b'(':
            depth, j, buf = 1, i + 1, bytearray()
            while j < n and depth:
                ch = s[j:j + 1]
                if ch == b'\\':
                    nx = s[j + 1:j + 2]
                    esc = {b'n': b'\n', b'r': b'\r', b't': b'\t', b'b': b'\b', b'f': b'\f', b'(': b'(', b')': b')', b'\\': b'\\'}
                    if nx in esc:
                        buf += esc[nx]
                        j += 2
                    elif nx.isdigit():
                        k = j + 1
                        while k < n and k < j + 4 and s[k:k + 1].isdigit():
                            k += 1
                        buf.append(int(s[j + 1:k], 8) & 0xFF)
                        j = k
                    elif nx in b'\r\n':
                        j += 2
                    else:
                        buf += nx
                        j += 2
                    continue
                if ch == b'(':
                    depth += 1
                elif ch == b')':
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                buf += ch
                j += 1
            yield ('str', bytes(buf))
            i = j
        elif c == b'<' and s[i + 1:i + 2] != b'<':
            j = s.index(b'>', i)
            yield ('str', bytes.fromhex(s[i + 1:j].decode().replace(' ', '')))
            i = j + 1
        elif c == b'[':
            yield ('[', None)
            i += 1
        elif c == b']':
            yield (']', None)
            i += 1
        elif c == b'/':
            j = i + 1
            while j < n and s[j:j + 1] not in b' \t\r\n/[]()<>':
                j += 1
            yield ('name', s[i + 1:j].decode())
            i = j
        else:
            j = i
            while j < n and s[j:j + 1] not in b' \t\r\n/[]()<>%':
                j += 1
            tok = s[i:j].decode('latin-1')
            try:
                yield ('num', float(tok))
            except ValueError:
                yield ('op', tok)
            i = j if j > i else i + 1


def _mat_mul(a, b):
    return [a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3], a[4] * b[0] + a[5] * b[2] + b[4], a[4] * b[1] + a[5] * b[3] + b[5]]


def pdf_read(path):
    b = open(path, 'rb').read()
    objs = _pdf_objects(b)
    page = next(n for n, body in objs.items() if re.search(rb'/Type /Page\b(?!s)', body) and b'/Contents' in body)
    pbody = objs[page]
    content = _pdf_stream(objs[int(re.search(rb'/Contents (\d+) 0 R', pbody).group(1))])
    res = objs[int(re.search(rb'/Resources (\d+) 0 R', pbody).group(1))]
    fdict = objs[int(re.search(rb'/Font (\d+) 0 R', res).group(1))]
    fonts = {}
    for name, num in re.findall(rb'/(\w+) (\d+) 0 R', fdict):
        fb = objs[int(num)]
        tu = re.search(rb'/ToUnicode (\d+) 0 R', fb)
        fonts[name.decode()] = _cmap(_pdf_stream(objs[int(tu.group(1))])) if tu else {}
    meta = {}
    for n, body in objs.items():
        if b'/Producer' in body:
            meta = {k.decode(): v.decode('latin-1') for k, v in re.findall(rb'/(\w+) \(([^)]*)\)', body)}
    texts, segs = [], []
    ctm, stack = [1, 0, 0, 1, 0, 0], []
    stroke = None
    operands, arr, in_arr = [], [], False
    font, cur_text, text_pos, td = None, None, None, [0.0, 0.0]
    pth, cur = [], None
    for kind, val in _tokens(content):
        if kind == '[':
            in_arr, arr = True, []
            continue
        if kind == ']':
            in_arr = False
            operands.append(('arr', arr))
            continue
        if kind != 'op':
            (arr if in_arr else operands).append((kind, val))
            continue
        op = val
        nums = [v for k, v in operands if k == 'num']
        if op == 'q':
            stack.append((list(ctm), stroke))
        elif op == 'Q':
            if stack:
                ctm, stroke = stack.pop()
        elif op == 'cm' and len(nums) >= 6:
            ctm = _mat_mul(nums[-6:], ctm)
        elif op == 'RG' and len(nums) >= 3:
            stroke = tuple(round(x, 4) for x in nums[-3:])
        elif op == 'G' and nums:
            stroke = (round(nums[-1], 4),) * 3
        elif op == 'BT':
            cur_text, td = '', [0.0, 0.0]
            text_pos = None
        elif op == 'Tf':
            font = next(v for k, v in operands if k == 'name')
        elif op == 'Td' and len(nums) >= 2:
            if text_pos is None:
                text_pos = (ctm[4] + nums[-2], ctm[5] + nums[-1])
            td = [td[0] + nums[-2], td[1] + nums[-1]]
        elif op in ('Tj', 'TJ'):
            strs = []
            for k, v in operands:
                if k == 'str':
                    strs.append(v)
                elif k == 'arr':
                    strs.extend(x for kk, x in v if kk == 'str')
            cmap = fonts.get(font, {})
            for sb in strs:
                for i in range(0, len(sb) - 1, 2):
                    code = (sb[i] << 8) | sb[i + 1]
                    cur_text += cmap.get(code, '?')
            if text_pos is None:
                text_pos = (ctm[4], ctm[5])
        elif op == 'ET':
            texts.append({'text': cur_text, 'x': round(text_pos[0], 3) if text_pos else None,
                          'y': round(text_pos[1], 3) if text_pos else None})
            cur_text = None
        elif op == 'm' and len(nums) >= 2:
            cur = (nums[-2], nums[-1])
            pth.append([cur])
        elif op == 'l' and len(nums) >= 2 and pth:
            pth[-1].append((nums[-2], nums[-1]))
        elif op in ('S', 'B', 'b', 's'):
            for sub in pth:
                for p0, p1 in zip(sub, sub[1:]):
                    x0, y0 = p0[0] * ctm[0] + p0[1] * ctm[2] + ctm[4], p0[0] * ctm[1] + p0[1] * ctm[3] + ctm[5]
                    x1, y1 = p1[0] * ctm[0] + p1[1] * ctm[2] + ctm[4], p1[0] * ctm[1] + p1[1] * ctm[3] + ctm[5]
                    segs.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'stroke': stroke})
            pth = []
        elif op in ('n', 'f', 'f*', 'W', 'W*', 'h', 're'):
            if op in ('n', 'f', 'f*'):
                pth = []
        operands = []
    return {'texts': texts, 'segments': segs, 'metadata': meta, 'sha256': _sha(path)}


def figure_rows(pdf):
    texts, segs = pdf['texts'], pdf['segments']
    norm = lambda t: t.replace('\u2212', '-').strip()  # noqa: E731
    # x ticks: short vertical segments below the axis; their labels: numeric texts at the lowest y
    black = (0.0, 0.0, 0.0)
    vt = [s for s in segs if abs(s['x0'] - s['x1']) < 1e-6 and 1.0 < abs(s['y0'] - s['y1']) < 5.0 and s['stroke'] == black]
    xt = sorted({round(s['x0'], 4) for s in vt})
    numeric = [t for t in texts if re.fullmatch(r'-?\d+(\.\d+)?', norm(t['text']))]
    ylow = min(t['y'] for t in numeric) if numeric else None
    xlabels = sorted([t for t in numeric if abs(t['y'] - ylow) < 0.5], key=lambda t: t['x'])
    xvals = [float(norm(t['text'])) for t in xlabels]
    if len(xt) != len(xvals) or len(xt) < 2:
        _fail(f'figure {pdf["sha256"][:8]}: x ticks {xt} vs labels {xvals}')
        return None
    scale = (xt[-1] - xt[0]) / (xvals[-1] - xvals[0])
    to_data = lambda x: xvals[0] + (x - xt[0]) / scale  # noqa: E731
    # y ticks: short horizontal segments left of the axis; bars: coloured horizontal segments
    ht = [s for s in segs if abs(s['y0'] - s['y1']) < 1e-6 and 1.0 < abs(s['x0'] - s['x1']) < 5.0 and s['stroke'] == black]
    yticks = sorted({round(s['y0'], 4) for s in ht}, reverse=True)
    bars = [s for s in segs if abs(s['y0'] - s['y1']) < 1e-6 and abs(s['x0'] - s['x1']) >= 5.0
            and s['stroke'] not in (None, (0.0, 0.0, 0.0), (0.6, 0.6, 0.6)) and len(set(s['stroke'])) > 1]
    rows = []
    for yt in yticks:
        bs = [s for s in bars if abs(s['y0'] - yt) < 0.5]
        if not bs:
            continue
        bar = bs[0]
        lo, hi = sorted((bar['x0'], bar['x1']))
        centre, half = to_data((lo + hi) / 2), (hi - lo) / 2 / scale
        label = [t for t in texts if t['x'] is not None and t['x'] < min(xt) - 5 and abs(t['y'] - yt) < 5.0
                 and not re.fullmatch(r'-?\d+(\.\d+)?', norm(t['text']))]
        right = [t for t in texts if t['x'] is not None and t['x'] > max(xt) + 10 and abs(t['y'] - yt) < 4.0]
        above = [t for t in texts if t['x'] is not None and min(xt) - 5 < t['x'] < max(xt) + 10 and 0 < t['y'] - yt < 12.0
                 and re.fullmatch(r'-?\d+(\.\d+)?', norm(t['text']))]
        rows.append({'y': yt, 'label': label[0]['text'] if label else None, 'bar_centre_kEUR': round(centre, 3),
                     'bar_half_width_kEUR': round(half, 3), 'stroke': bar['stroke'],
                     'value_text': above[0]['text'] if above else None, 'right_text': right[0]['text'] if right else None})
    return {'x_ticks_pt': xt, 'x_tick_values': xvals, 'pt_per_kEUR': scale, 'rows': rows,
            'all_texts': [t['text'] for t in texts]}


def task_e():
    src = _read(FIG_V2_PY)
    tree = ast.parse(src)
    rows_lit = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'rows' for t in node.targets):
            rows_lit = ast.literal_eval(node.value)
    fz = _json(FROZEN)
    if _sha(FROZEN) != FROZEN_SHA:
        _fail('frozen JSON hash differs')
    t8 = {r['arm']: r for r in fz['tables']['ageing']['rows']}
    claims = {c['claim_id']: c for c in fz['tables']['claims']}
    thr = {'C3_unit': claims['E:C3_unit_value_minus_I'], 'C2': claims['E:n7_4h_e1_C2:value_minus_I'],
           'C2_calfade': claims['E:n7_4h_e1_C2_calfade:value_minus_I'], 'C3_midblock': claims['E:n7_4h_e1_C3_midblock:value_minus_I'],
           'C4': claims['E:n7_4h_e1_C4:value_minus_I'], 'no_ageing': claims['E:n7_4h_e1_no_ageing:value_minus_I']}
    mapping = {'Baseline (cycling + calendar)': 'C2_calfade', 'Cycling only': 'C2', 'Retention 50 %': 'C3_unit',
               'Retention 50 %, mid-block': 'C3_midblock', 'Datasheet 8,000 cycles': 'C4', 'No ageing': 'no_ageing'}
    variants = _json(SPEC_EXT)['model_variant']['arms']
    comp = []
    for lab, v, th, det, yr in rows_lit:
        arm = mapping.get(lab)
        r = t8[arm]
        c = thr[arm]
        fy = r['floor_year_070']
        comp.append({'label': lab, 'arm': arm, 'script_value_minus_I': v, 'T8_value_minus_I_kEUR': r['value_minus_I'] / 1e3,
                     'value_ok': round(r['value_minus_I'] / 1e3, 1) == v, 'script_threshold': th,
                     'T1_E_threshold_kEUR': c['gross_threshold_or_bar'] / 1e3,
                     'threshold_ok': round(c['gross_threshold_or_bar'] / 1e3, 1) == th,
                     'script_determinate': det, 'T8_verdict': r['verdict'], 'verdict_ok': det == (r['verdict'] == 'determinate'),
                     'script_year': yr, 'T8_floor_year_070': fy,
                     'year_ok': (str(fy) == yr) if fy is not None else (yr in ('never', 'n/a')),
                     'variant': variants.get(arm), 'k': r['k']})
    v2 = pdf_read(FIG_V2_PDF)
    v2_rows = figure_rows(v2)
    w165 = pdf_read(FIG_W165)
    w165_rows = figure_rows(w165)
    # the PDF rows against the script rows (by label) and the W165 rows against T8 (by code label)
    pdf_vs_script = []
    for r in (v2_rows or {}).get('rows', []):
        nl = re.sub('[\u202f\xa0\u2009]', ' ', r['label'] or '').strip()
        s = next((x for x in comp if x['label'] == nl), None)
        vt = (r['value_text'] or '').replace('\u2212', '-')
        pdf_vs_script.append({'pdf_label': r['label'], 'script_label': s['label'] if s else None,
                              'pdf_value_text': r['value_text'], 'pdf_bar_centre': r['bar_centre_kEUR'],
                              'pdf_bar_half_width': r['bar_half_width_kEUR'], 'pdf_right_text': r['right_text'],
                              'pdf_colour': r['stroke'],
                              'value_text_ok': bool(s) and vt == f'{s["script_value_minus_I"]:.1f}',
                              'centre_ok': bool(s) and abs(r['bar_centre_kEUR'] - s['script_value_minus_I']) < 0.05,
                              'half_width_ok': bool(s) and abs(r['bar_half_width_kEUR'] - s['script_threshold']) < 0.05,
                              'right_text_ok': bool(s) and r['right_text'] == s['script_year'],
                              'colour_ok': bool(s) and (r['stroke'] == (0.0, 0.4471, 0.698)) == s['script_determinate']})
    w165_vs_t8 = []
    for r in (w165_rows or {}).get('rows', []):
        arm = (r['label'] or '').strip().split(' ')[0]           # 'C2_calfade (baseline)' -> 'C2_calfade'
        t = t8.get(arm)
        vt = (r['value_text'] or '').replace('\u2212', '-')
        w165_vs_t8.append({'pdf_label': r['label'], 'arm': arm if t else None, 'pdf_value_text': r['value_text'],
                           'pdf_bar_centre': r['bar_centre_kEUR'], 'pdf_bar_half_width': r['bar_half_width_kEUR'],
                           'pdf_right_text': r['right_text'],
                           'T8_value_minus_I_kEUR': t['value_minus_I'] / 1e3 if t else None,
                           'value_ok': bool(t) and vt == f'{t["value_minus_I"] / 1e3:.1f}',
                           'T8_floor_year_070': t['floor_year_070'] if t else None})
    t8_csv = _read(T8_CSV)
    return {'script': {'path': FIG_V2_PY, 'sha256': _sha(FIG_V2_PY), 'rows': rows_lit, 'executed': False},
            'pdf_v2': {'path': FIG_V2_PDF, 'sha256': v2['sha256'], 'clone_copy_sha256': _sha(FIG_V2_CLONE),
                       'clone_copy_equal': _sha(FIG_V2_CLONE) == v2['sha256'], 'metadata': v2['metadata'],
                       'read': v2_rows},
            'pdf_w165': {'path': FIG_W165, 'sha256': w165['sha256'], 'metadata': w165['metadata'], 'read': w165_rows,
                         'data_json_sha256': _sha(FIG_W165_DATA)},
            'script_vs_T8': comp, 'pdf_v2_vs_script': pdf_vs_script, 'pdf_w165_vs_T8': w165_vs_t8,
            'frozen': {'path': FROZEN, 'sha256': _sha(FROZEN), 'T8_json_path': 'tables.ageing.rows[]',
                       'threshold_json_path': 'tables.claims[] E:*:value_minus_I .gross_threshold_or_bar'},
            'T8_csv': {'path': T8_CSV, 'sha256': _sha(T8_CSV), 'no_ageing_floor_cell': [ln for ln in t8_csv.split('\n') if ln.startswith('no_ageing')]}}


# ======================================================================================================================
#  (f) references and labels
# ======================================================================================================================
def task_f(lines):
    clines = W167.strip_comments(lines)
    text = '\n'.join(clines)

    def line_of(pos):
        return text.count('\n', 0, pos) + 1
    labels = {}
    for m in re.finditer(r'\\label\{([^}]*)\}', text):
        labels.setdefault(m.group(1), []).append(line_of(m.start()))
    refs = {}
    for m in re.finditer(r'\\(ref|eqref|pageref|autoref|cref|Cref|nameref)\{([^}]*)\}', text):
        for k in m.group(2).split(','):
            refs.setdefault(k.strip(), []).append(line_of(m.start()))
    cites = {}
    for m in re.finditer(r'\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}', text):
        for k in m.group(1).split(','):
            cites.setdefault(k.strip(), []).append(line_of(m.start()))
    bib = _read(BIB)
    bibkeys = re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', bib)
    bibfile = re.findall(r'\\bibliography\{([^}]*)\}', text)
    dups = {k: v for k, v in labels.items() if len(v) > 1}
    undefined_refs = {k: v for k, v in refs.items() if k not in labels}
    undefined_cites = {k: v for k, v in cites.items() if k not in set(bibkeys)}
    bib_dups = sorted({k for k in bibkeys if bibkeys.count(k) > 1})
    unreferenced = sorted(k for k in labels if k not in refs)
    cert_refs = []
    for i, ln in enumerate(clines, 1):
        if r'\ref{subsubsec:certification}' in ln:
            for m in re.finditer(r'\\ref\{subsubsec:certification\}', ln):
                cert_refs.append({'line': i, 'context': re.sub(r'\s+', ' ', ln[max(0, m.start() - 220):m.end() + 120]).strip()})
    # labels referenced only from commented-out text (they would be missing after comment stripping)
    raw_text = '\n'.join(lines)
    commented_only = sorted(k for k in set(re.findall(r'\\(?:ref|eqref)\{([^}]*)\}', raw_text)) if k not in refs)
    return {'n_labels': len(labels), 'n_ref_keys': len(refs), 'n_ref_uses': sum(len(v) for v in refs.values()),
            'n_cite_keys': len(cites), 'n_cite_uses': sum(len(v) for v in cites.values()), 'n_bib_keys': len(bibkeys),
            'bibliography_command': bibfile, 'duplicate_labels': dups, 'undefined_refs': undefined_refs,
            'undefined_cites': undefined_cites, 'duplicate_bib_keys': bib_dups,
            'labels_never_referenced': unreferenced, 'refs_only_in_comments': commented_only,
            'refs_to_subsubsec_certification': cert_refs}


# ======================================================================================================================
def main():
    global OUT_DIR, OUT_JSON, OUT_MANIFEST
    if '--out-dir' in sys.argv:                       # trial runs only; the directory must hold nothing but launch.log
        OUT_DIR = sys.argv[sys.argv.index('--out-dir') + 1]
        OUT_JSON = os.path.join(OUT_DIR, 'w177b_part3_checks.json')
        OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if not os.path.isdir(OUT_DIR):
        print(f'REFUSED: {OUT_DIR} does not exist (create it with mkdir; it must hold only launch.log)', flush=True)
        sys.exit(1)
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        print(f'REFUSED: {OUT_DIR} already holds {extra}', flush=True)
        sys.exit(1)
    t0 = datetime.now(timezone.utc)
    _log('W177b-3 structure checks: start')
    lines = main_lines()
    out = {'schema': 'p515_s53_w177b_part3_checks', 'task': 'W177b-3 (Addendum 76 Part 3)',
           'started_utc': t0.isoformat(), 'repo_head': _git(['rev-parse', 'HEAD']).strip(),
           'script': {'path': os.path.basename(__file__), 'sha256': _sha(__file__)},
           'w167_module': {'path': 'p515_s53_w167_nomenclature_years.py', 'sha256': _sha('p515_s53_w167_nomenclature_years.py')}}
    out['identity'] = manuscript_identity()
    _log(f"identity: {out['identity']}")
    out['a_certification_rule'] = task_a(lines)
    _log('a: done')
    out['b_settings'] = task_b(lines)
    _log('b: done')
    out['c_storage_agent'] = task_c(lines)
    _log('c: done')
    out['d_nomenclature'] = task_d(lines)
    _log(f"d: unused now {[u['key'] for u in out['d_nomenclature']['current']['defined_unused']]}; "
         f"used-undefined {out['d_nomenclature']['current']['n_used_undefined']}")
    out['e_ageing_figure'] = task_e()
    _log('e: done')
    out['f_refs_labels'] = task_f(lines)
    _log(f"f: duplicate labels {out['f_refs_labels']['duplicate_labels']}; undefined refs "
         f"{out['f_refs_labels']['undefined_refs']}; undefined cites {out['f_refs_labels']['undefined_cites']}")
    forbidden = sorted(m for m in sys.modules if m.split('.')[0].startswith(FORBIDDEN_MODULE_PREFIXES)
                       and m.split('.')[0] not in ('networkx',))
    g1 = GUARD.verify(0)
    g2 = W167.GUARD.verify(0)
    out['guards'] = {'solve_profile_guard': {'permitted': [], 'counts': GUARD.counts, 'verify_0_failures': g1},
                     'w167_guard': {'counts': W167.GUARD.counts, 'verify_0_failures': g2},
                     'pickle_counts': dict(PICKLE_COUNTS), 'w167_pickle_counts': dict(W167.PICKLE_COUNTS),
                     'production_modules_imported': forbidden}
    if g1 or g2 or any(PICKLE_COUNTS.values()) or any(W167.PICKLE_COUNTS.values()) or forbidden:
        _fail(f'guards: {out["guards"]}')
    out['check_failures'] = list(CHECK_FAILURES)
    out['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
        fh.write('\n')
    inputs = sorted({MAIN_TEX, BIB, FROZEN, T8_CSV, FIG_V2_PDF, FIG_V2_PY, FIG_V2_CLONE, FIG_W165, FIG_W165_DATA,
                     SPEC_V6, SPEC_EXT, SPEC_A64, SPEC_3X3, SPEC_S47, SPEC_S51, SPEC_S53, SRP1_PARAMS, ESS_PARAMS, CASE9,
                     V6_LOG_SRP1, V6_LOG_3X3, PARAGRAPHS_V5, SC1, SC2, SC5, SC6, H118, H101, H142, C132, SED, SRP,
                     'network.py', 'definitions.py', 'p515_s53_w167_nomenclature_years.py', os.path.basename(__file__)}
                    | set(CASE33))
    manifest = {'outputs': {os.path.basename(OUT_JSON): _sha(OUT_JSON)}, 'out_dir': OUT_DIR,
                'inputs': {p: _sha(p) for p in inputs}}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'written {OUT_JSON} and {OUT_MANIFEST}; check failures: {len(CHECK_FAILURES)}; guards ok: '
         f'{not (g1 or g2)}; pickle {PICKLE_COUNTS}')
    sys.exit(3 if CHECK_FAILURES else 0)


if __name__ == '__main__':
    main()
