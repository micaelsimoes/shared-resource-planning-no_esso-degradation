"""P5.15 Addendum 73, Planner task W177a -- CHECK OF THE EXPERT'S SECTION 4 FRAME `section4_expert_draft.tex` (repository
root, committed in 4f941b29) AGAINST CODE, RECORDS AND THE FROZEN TABLES. ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the
W176b script is IMPORTED (it imports W174a -> W171a -> W164 -> ... -> W160, each arming its guards) and its tokenizer
(W174.tokenize_v3 over W164.Doc), its automatic index over the frozen JSON (W164.AutoIndex / W164.json_leaves) and the
lineage helpers are USED on this file as a separate scope; nothing earlier is edited. Every output is a new file opened
'x' in a new directory.

WHAT IT CHECKS (Addendum 73, W177(a))
  1. Every [CONFIRM -- W177] point of the frame (l. 43-45, 104-105, 131, 208-209, 269, 342, 391-392, 410-411, 452):
       C1  eq:salvage_credit / eq:salvage_remaining_life and the l. 43 items (a)-(d) against the production salvage code
           (every salvage function of the tracked *.py files is located; the one behind the reported net is named) and
           reproduced numerically on two recorded cells (yl_y2030, yl_y2035: recorded terminal SoH, cost workbook);
       C2  S / E / duration / node / year of every row of tab:res_value_ladder against the frozen candidate of its
           claim's cell AND that cell's own evaluation_record.json (candidate_canonical, candidate_key);
       C3  the I/J candidate keys (cc4eb5df, fb21d822, 11725e45, cab98853) against the B cells n7_4h_e2 ... e5;
       C4  the F2 plan behind ref:5ca4f86c (year, nodes, P, E) from its frozen candidate, its evaluation record and W173;
       C5  the 0.50-floor cell e_soh050 (value, AE, EFC/day, floor year) from the frozen A64 row and the W155 summary;
       C6  the 3 x 3 "16 cycles" of the post-hoc continuation (W98 r2 continuation record, W99 fit window);
       C7  W153's "0 sign changes in 60 claims" and the one verdict that changes category;
       C8  the 2025 cohort's zero salvage (code + every 2025-plan cell's recorded salvage + T1 d_net = d_gross);
       C9  "41 new recourse evaluations (14, 20, 7) against budgets of 20, 20 and 60" (W173).
  2. Every numeric token of the frame (tables, captions, the expert's prose, the [AUTHOR -- write] slot facts, the
     source comments): declared against the frozen JSON at the export's printed rounding (export/README.md) or a named
     record (path + sha256); status match / MISMATCH / approximate (printed coarser than the export rule, consistent at
     its printed precision) / no counterpart / unchecked (category: notation, language, instruction, cross-reference,
     compound adjective) / excluded (structure, identifiers). Every token must be assigned (else exit 3). Text checks
     beside the tokens: table verdict words and the u / f superscripts against the frozen claims; the four accepted
     sentences against paragraphs_v5.md (verbatim); the T4 caption sentence.
  3. Labels: every label the frame introduces is unique, matches the header list, does not collide with a main.tex label
     outside the replaced range; every \\ref of the frame resolves; no old section-4 label is referenced outside it; the
     figure files match export/figures/*.pdf (and the sha256 prefixes in captions.md); the literal section references
     the author must replace (Addendum 73's list) are located in main.tex at the clone HEAD (read only).

RECORDED PREDICTIONS (expert, Addendum 73), scored in the outputs: zero MISMATCH in the composed tables and captions; the
salvage equation needs one correction among its four points; the four other structural CONFIRMs (S/E, I/J keys, F2
plan, 16 cycles) hold.

OBJECTIVE CONVENTION. Q = gross_operational_cost, settlement excluded (the frozen primary); net = Q - terminal salvage
credit; value = Q(0) - Q(x); F = Q + I (the frozen JSON's objective_convention, recorded in the output).

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W176 chain arms; `pickle.load` / `pickle.loads` blocked for the whole run, every counter
verified at 0. git reads are read-only (`git show` / `git log` / `git rev-parse` / `git ls-files` / `git status`, also in
the manuscript clone, which is never pulled or modified). The cost workbook is read with openpyxl in read-only mode.

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w177a_section4_frame && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177a_section4_frame_check.py \\
        > data/SRP1/Results/P515S53/w177a_section4_frame/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w177a_section4_frame/w177a_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w177a_section4_frame/w177a_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177a_section4_frame_check.py --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES and CONFIRM differences are FINDINGS
and never change the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault.
"""
import argparse
import collections
import glob
import json
import math
import os
import pickle
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W177a section-4 frame check (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W177a: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W177a: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w176_manuscript_number_check as W176  # noqa: E402 -- arms its guards, imports W174a -> W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

import openpyxl  # noqa: E402  (third-party, read-only workbook read as in W175 / W176b; not project code)

W174, W171, W164, W160 = W176.W174, W176.W171, W176.W164, W176.W160
GUARDS = W160.W157._dedupe((('w177a_section4_frame_check', GUARD),) + tuple(W176.GUARDS))
_log, _sha, _sha_bytes, _git = W160._log, W160._sha, W160._sha_bytes, W160._git

SCRIPT_REL = os.path.basename(__file__)
S53 = W160.S53
OUT_DIR = os.path.join(S53, 'w177a_section4_frame')
OUT_NAMES = {'json': 'w177a_section4_frame_check.json', 'md': 'w177a_section4_frame_check.md',
             'man': 'manifest_sha256.json', 'log': 'launch.log', 'typing_json': 'w177a_bool_typing_test.json',
             'typing_log': 'w177a_bool_typing_test.log', 'post': 'manifest_post_run_sha256.json'}

FRAME = 'section4_expert_draft.tex'
FRAME_SHA = 'e73b6d918ff635164a54207e56f9ac9c9e1aef6307f87d018eac2684d4528a5d'
FRAME_COMMIT = '4f941b29'
CLONE = W164.DEFAULT_MANUSCRIPT_DIR                       # manuscript/6a67305f25e8348fb71380c3 (read only)
CLONE_HEAD = '8b764234533f19ffc8a8e72a276f2f96c20a456d'
MAIN_SHA = '7aa9e105deccdf61cced665f03d95b02ea736fe5c34f3402f6281db89d19624b'
ADD73_RANGE = (1126, 1445)                                # Addendum 73: main.tex l. 1126-1445 replaced by the frame
ADD73_LITERALS = {415: 'Section~4', 445: 'Section~4.7', 631: 'Section~4.5', 1066: 'Section~4.3', 1116: '(Section~4.4)'}
FZ_REL, FZ_SHA = W164.FZ_REL, W164.FZ_SHA
EXPORT = os.path.join(S53, 'w160_step6_frozen', 'export')
FIGS = os.path.join(EXPORT, 'figures')
FIG_FILES = ('fig1_breakeven.pdf', 'fig2_flexibility_ladder.pdf', 'fig3_ageing_arms.pdf', 'fig4_q_versus_cycle.pdf',
             'fig5_benchmark.pdf')
YL_EVALS = {'2030': os.path.join(S53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_yl_y2030', 'evals',
                                 '6e4f11a95dc3d585_yl_y2030'),
            '2035': os.path.join(S53, 'w118_resettle', 'campaign_s53_w118_resettle_r2_yl_y2035', 'evals',
                                 '3a2387f46448f9db_yl_y2035')}
W98R2 = os.path.join(S53, 'w98_continuation', 'campaign_s53_w98_x0_continuation_r2')

INPUTS = {
    'FRAME': FRAME,
    'FROZEN_JSON': FZ_REL,
    'EXPORT_README': os.path.join(EXPORT, 'README.md'),
    'PARAGRAPHS_V5': os.path.join(EXPORT, 'paragraphs_v5.md'),
    'T4_TEX': os.path.join(EXPORT, 'T4.tex'),
    'T7_TEX': os.path.join(EXPORT, 'T7.tex'),
    'T6_CSV': os.path.join(EXPORT, 'T6.csv'),
    'CAPTIONS_MD': os.path.join(FIGS, 'captions.md'),
    'FIG1_DATA': os.path.join(FIGS, 'fig1_breakeven_data.json'),
    'FIG2_DATA': os.path.join(FIGS, 'fig2_flexibility_ladder_data.json'),
    'FIG4_DATA': os.path.join(FIGS, 'fig4_q_versus_cycle_data.json'),
    'FIG5_DATA': os.path.join(FIGS, 'fig5_benchmark_data.json'),
    'W165_MANIFEST': os.path.join(FIGS, 'manifest_sha256.json'),
    'W153_CERT': os.path.join(S53, 'w153_step5_rows', 'w153_certification_statistics.json'),
    'W153_SALV': os.path.join(S53, 'w153_step5_rows', 'w153_salvage_addback.json'),
    'W155_SOH050': os.path.join(S53, 'w155_a64_cells', 'w155_summary_after_04_e_soh050.json'),
    'W99': os.path.join(S53, 'w99_stage1_posthoc', 'w99_stage1_posthoc_analysis.json'),
    'W98R2_RESULTS': os.path.join(W98R2, 'campaign_results.json'),
    'W98R2_CONT': os.path.join(W98R2, 'evals', '25b92ae0f1f2c02e_x0_cont', 'continuation_cycle_record.jsonl'),
    'W173': os.path.join(S53, 'w173_search_counts', 'w173_search_counts.json'),
    'W163': W164.W163_JSON,
    'W170': os.path.join(S53, 'w170_r_prediction', 'w170_r_prediction.json'),
    'SRP1_JSON': W164.SRP1_JSON,
    'CASE_3X3': os.path.join(S53, 'w89_3x3', 'instance', 'SRP1__s53_3x3.json'),
    'ESS_PARAMS': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
    'ESS_XLSX': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
    'YL2030_BOYD': os.path.join(YL_EVALS['2030'], 'boyd_terminal.json'),
    'YL2035_BOYD': os.path.join(YL_EVALS['2035'], 'boyd_terminal.json'),
    'CODE_SESD': 'shared_energy_storage_data.py',
    'CODE_SESP': 'shared_energy_storage_parameters.py',
    'CODE_SES': 'shared_energy_storage.py',
    'CODE_SRP': 'shared_resources_planning.py',
    'CASE9': os.path.join('data', 'SRP1', 'case9', 'case9_2025.json'),
    'CASE33_1': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_2025.json'),
    'CASE33_2': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_2025.json'),
    'CASE33_3': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_2025.json'),
    'SUBMITTED_MAIN': W171.SUBMITTED_MAIN,
    'BRIEF': 'PLANNER_BRIEF_2026-09-13.md',
}
for _f in FIG_FILES:
    INPUTS['PDF_' + _f.split('_')[0].upper()] = os.path.join(FIGS, _f)
JSON_INPUTS = ('FROZEN_JSON', 'FIG1_DATA', 'FIG2_DATA', 'FIG4_DATA', 'FIG5_DATA', 'W165_MANIFEST', 'W153_CERT', 'W153_SALV',
               'W155_SOH050', 'W99', 'W98R2_RESULTS', 'W173', 'W163', 'W170', 'SRP1_JSON', 'CASE_3X3', 'ESS_PARAMS',
               'YL2030_BOYD', 'YL2035_BOYD', 'CASE9', 'CASE33_1', 'CASE33_2', 'CASE33_3')

STATUSES = ('match', 'MISMATCH', 'approximate', 'no counterpart', 'unchecked', 'excluded')
IDENT_TYPES = ('label', 'hash', 'date', 'dotted', 'alnum')


# ======================================================================================================================
#  0. small helpers
# ======================================================================================================================
def _text(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return h.read()


def _json(rel):
    with open(os.path.join(REPO, rel), encoding='utf-8') as h:
        return json.load(h)


def _norm(s):
    s = unicodedata.normalize('NFKC', s)
    for a, b in (('\u2212', '-'), ('\u2013', '-'), ('\u2014', '-'), ('\u00a0', ' '), ('\u2009', ' '), ('k\u20ac', 'kEUR'),
                 ('\u20ac', 'EUR'), ('\u2248', '~'), ('\u2019', "'"), ('\u201c', '"'), ('\u201d', '"')):
        s = s.replace(a, b)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _dp(w):
    s = str(w).replace(',', '').lstrip('+-\u2212')
    return len(s.split('.')[1]) if '.' in s else 0


def _fmt(x, dp):
    s = f'{x:.{dp}f}'
    return s[1:] if s.startswith('-') and float(s) == 0.0 else s


_SEG = re.compile(r'([^.\[\]]+)|\[([^\]]+)\]')
_IDK = ('claim_id', 'cell', 'arm', 'quantity', 'id', 'name')


def jget(obj, path):
    """`a.b[sel].c`: dict keys by name or [key]; list elements by [index] or [id] (claim_id / cell / arm / quantity / id /
    name equal to sel, or rate equal to float(sel))."""
    cur = obj
    for m in _SEG.finditer(path):
        name, sel = m.group(1), m.group(2)
        key = name if name is not None else sel
        if isinstance(cur, dict):
            if key not in cur:
                raise KeyError(f'{path}: no key {key!r}')
            cur = cur[key]
        elif isinstance(cur, list):
            if re.fullmatch(r'-?\d+', key):
                cur = cur[int(key)]
                continue
            hits = [e for e in cur if isinstance(e, dict) and (any(e.get(k) == key for k in _IDK) or
                    ('rate' in e and _isnum(key) and float(e['rate']) == float(key)))]
            if len(hits) != 1:
                raise KeyError(f'{path}: selector {key!r} -> {len(hits)} hits')
            cur = hits[0]
        else:
            raise KeyError(f'{path}: cannot descend into {type(cur).__name__} at {key!r}')
    return cur


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


# ======================================================================================================================
#  1. context: inputs and accessors
# ======================================================================================================================
class Ctx:
    def __init__(self, inputs):
        self.inputs = inputs
        self.j = {k: _json(inputs[k]['path']) for k in JSON_INPUTS}
        self.fz = self.j['FROZEN_JSON']
        self.t = self.fz['tables']
        self.cells = dict(self.t['cells'])
        self.cells.update(self.t['cells_appended_w160'])
        self.CL = {c['claim_id']: c for c in self.t['claims']}
        self.A64 = {r['claim_id']: r for r in self.t['a64']['rows']}
        self.W163 = {c['id']: c for c in self.j['W163']['checks_in_v4']}
        self.fig4 = {v['name']: v['value'] for v in self.j['FIG4_DATA']['values']}
        self.fig4_checks = self.j['FIG4_DATA']['checks']
        self.fig5 = {v['name']: v for v in self.j['FIG5_DATA']['caption_values']}
        self.fig5_computed = {v['name']: v['value'] for v in self.j['FIG5_DATA']['computed']}
        self.fig2 = {v['name']: v['value'] for v in self.j['FIG2_DATA']['values']}
        self.w153c = self.j['W153_CERT']['result']['totals']
        self.w153s = self.j['W153_SALV']['result']
        self.w155 = self.j['W155_SOH050']['reports']['e_soh050']
        self.w173 = self.j['W173']
        self.frame = _text(FRAME)
        self.texts = {k: _text(inputs[k]['path']) for k in ('EXPORT_README', 'PARAGRAPHS_V5', 'T4_TEX', 'T7_TEX', 'CAPTIONS_MD',
                                                             'CODE_SESD', 'CODE_SESP', 'CODE_SES', 'CODE_SRP', 'SUBMITTED_MAIN')}
        self.main = None
        self.main_lines = None
        self.cont_cycles = [json.loads(ln)['cycle'] for ln in _text(inputs['W98R2_CONT']['path']).splitlines() if ln.strip()]
        self.xlsx = self._xlsx()
        self.evalrec_index = self._evalrec_index()

    # -- frozen JSON / records -----------------------------------------------------------------------------------------
    def fzp(self, path):
        return jget(self.fz, path[2:]) if path.startswith('$.') else jget(self.t, path)

    def rec(self, key, path):
        return jget(self.j[key], path)

    def cand(self, cell):
        return self.cells[cell]['candidate_canonical']

    def other(self, claim):
        return self.CL[claim]['other_cell'] if claim in self.CL else self.A64[claim]['other_cell']

    def _xlsx(self):
        wb = openpyxl.load_workbook(os.path.join(REPO, self.inputs['ESS_XLSX']['path']), read_only=True, data_only=True)
        out = {}
        for sheet in ('Scenarios', 'Investment Cost, Power', 'Investment Cost, Energy'):
            out[sheet] = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
        wb.close()
        return out

    def expected_cost(self, kind, year):
        """sum_c omega_c c_{year,c}: the cost workbook's scenario probabilities and the cost row of each scenario."""
        probs = [float(x) for x in self.xlsx['Scenarios'][0][2:] if x is not None]
        rows = self.xlsx['Investment Cost, Power' if kind == 'power' else 'Investment Cost, Energy']
        col = [i for i, v in enumerate(rows[0]) if v is not None and str(v) == str(year)][0]
        vals = [float(r[col]) for r in rows[1:1 + len(probs)]]
        return sum(p * v for p, v in zip(probs, vals)), probs, vals

    def _evalrec_index(self):
        out = collections.defaultdict(list)
        for p in _git('ls-files', '--', 'data/SRP1/Results/*evaluation_record.json').splitlines():
            d = os.path.basename(os.path.dirname(p))
            if re.fullmatch(r'[0-9a-f]{16}_.+', d):
                out[d[:16]].append(p)
        return out

    def evalrecs(self, eval_key):
        return sorted(self.evalrec_index.get(eval_key[:16], []))


# ======================================================================================================================
#  2. the CONFIRM points
# ======================================================================================================================
def _find_lines(text, needle):
    return [i + 1 for i, ln in enumerate(text.split('\n')) if needle in ln]


def _code_loc(X, key, needle):
    rel = X.inputs[key]['path']
    return {'file': rel, 'text': needle, 'lines': _find_lines(X.texts[key], needle)}


def confirm_salvage(X):
    """C1 / C8 evidence: the production salvage code, its parameters, the numerical reproduction on two recorded cells."""
    srp1, ess = X.j['SRP1_JSON'], X.j['ESS_PARAMS']
    years = list(srp1['Years'])
    y0, blocks = int(years[0]), [float(v) for v in srp1['Years'].values()]
    y_end = y0 + sum(blocks)
    r = float(srp1['DiscountFactor'])
    t_cal = float(ess['ageing']['calendar_life_years'])
    soh_min = float(ess['ageing']['minimum_soh'])
    sv = ess['salvage_value']
    named = _git('grep', '-n', '_get_terminal_salvage_value', '--', '*.py')
    defs = _git('grep', '-n', '-E', r'def [A-Za-z_]*salvage[A-Za-z_]*\(', '--', '*.py')
    prod_defs = [ln for ln in defs.splitlines() if not re.match(r'p\d', ln)]
    code = {
        'search_for_get_terminal_salvage_value': {'command': "git grep -n _get_terminal_salvage_value -- '*.py'",
                                                  'hits': named.splitlines()},
        'salvage_function_definitions_tracked_py': defs.splitlines(),
        'production_module_definitions': prod_defs,
        'locations': [
            _code_loc(X, 'CODE_SESD', 'def _build_terminal_salvage_value_expression(shared_ess_data, model, shared_ess_idx):'),
            _code_loc(X, 'CODE_SESD', 'model.salvage_value = pe.Expression(expr=salvage_value)'),
            _code_loc(X, 'CODE_SESD', 'def get_salvage_value(self, models):'),
            _code_loc(X, 'CODE_SESD', 'salvage_value += pe.value(models[node_id].salvage_value)'),
            _code_loc(X, 'CODE_SRP', "terminal_salvage_value = planning_problem.shared_ess_data.get_salvage_value(models['esso'])"),
            _code_loc(X, 'CODE_SRP', 'net_operational_recourse = gross_operational_cost - terminal_salvage_value'),
            _code_loc(X, 'CODE_SESD', 'def _get_terminal_discount_factor(shared_ess_data):'),
            _code_loc(X, 'CODE_SESD', 'terminal_years = sum(float(num_years) for num_years in shared_ess_data.years.values())'),
            _code_loc(X, 'CODE_SESD', 'return 1.00 / ((1.00 + shared_ess_data.discount_factor) ** terminal_years)'),
            _code_loc(X, 'CODE_SESD', 'def _get_remaining_calendar_life(shared_ess_data, investment_year_idx, shared_energy_storage):'),
            _code_loc(X, 'CODE_SESD', 'age_at_terminal = sum(float(value) for value in represented_years[investment_year_idx:])'),
            _code_loc(X, 'CODE_SESD', 'remaining_life = max(float(shared_energy_storage.t_cal) - age_at_terminal, 0.00)'),
            _code_loc(X, 'CODE_SESD', 'remaining_life_fraction = min(remaining_life / float(shared_energy_storage.t_cal), 1.00)'),
            _code_loc(X, 'CODE_SESD', 'def _get_expected_energy_investment_cost(shared_ess_data, year):'),
            _code_loc(X, 'CODE_SESD', "expected_cost += probability * shared_ess_data.cost_investment['energy'][scenario][year]"),
            _code_loc(X, 'CODE_SESD', 'e_rated = model.es_e_rated_per_unit[y_inv, terminal_year_idx]'),
            _code_loc(X, 'CODE_SESD', 'e_available = model.es_e_available_per_unit[y_inv, terminal_year_idx]'),
            _code_loc(X, 'CODE_SESD', 'e_available = e_rated * model.es_soh_per_unit_cumul[y_inv, terminal_year_idx]'),
            _code_loc(X, 'CODE_SESD', 'usable_energy_above_eol = (e_available - min_soh * e_rated) / (1.00 - min_soh)'),
            _code_loc(X, 'CODE_SESD', 'params.recycling_floor_fraction * e_rated'),
            _code_loc(X, 'CODE_SESD', '+ (1.00 - params.recycling_floor_fraction) * usable_energy_above_eol'),
            _code_loc(X, 'CODE_SESD', 'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y])'),
            _code_loc(X, 'CODE_SESD', '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))'),
            _code_loc(X, 'CODE_SESD', 'def _get_salvage_value_results(shared_ess_data, models):'),
            _code_loc(X, 'CODE_SESD', 'terminal_date = int(years[0]) + terminal_horizon_years'),
            _code_loc(X, 'CODE_SESD', 'def _get_salvage_value_sensitivities(shared_ess_data, models):'),
            _code_loc(X, 'CODE_SESD', "model.es_e_investment_fixed[y_inv].set_value(e_candidate)"),
            _code_loc(X, 'CODE_SESP', "self.t_cal = _read_optional_number(params_data, 'calendar_life_years', self.t_cal)"),
            _code_loc(X, 'CODE_SESP', "self.energy_recovery_fraction = 1.00"),
            _code_loc(X, 'CODE_SESP', "self.recycling_floor_fraction = 0.00"),
        ],
        'variants': [
            {'function': '_build_terminal_salvage_value_expression', 'role': 'the Pyomo expression model.salvage_value of '
             'each ESSO node model; summed by get_salvage_value; THE REPORTED NET USES THIS ONE '
             '(shared_resources_planning.py: terminal_salvage_value = get_salvage_value(models["esso"]); '
             'net_operational_recourse = gross_operational_cost - terminal_salvage_value)'},
            {'function': '_get_salvage_value_results', 'role': 'report / Excel variant (cohort table): the same product, '
             'gated by active_in_terminal_block and remaining_life > SMALL_TOLERANCE; not used for the net'},
            {'function': '_get_salvage_value_sensitivities', 'role': 'derivative of the credit with respect to E '
             '(Benders sensitivity, sign negative); not used for the net'},
            {'function': '_get_terminal_discount_factor, _get_remaining_calendar_life, _get_expected_energy_investment_cost',
             'role': 'shared helpers of the three'}],
        'production_formula': ('C = sum_nodes sum_{y_inv} DF_term * f_rec * cbar^E_{y_inv} * RLF_{y_inv} * '
                               '[phi * E_rated + (1 - phi) * (E_available - SoH_min * E_rated) / (1 - SoH_min)], '
                               'DF_term = 1 / (1 + r)^(sum of block lengths), cbar^E = sum_c omega_c c^{Inv,E}_{y_inv,c} '
                               '(ENERGY cost only), RLF = min(max(T_cal - age_at_terminal, 0) / T_cal, 1), '
                               'age_at_terminal = sum of the block lengths from y_inv on, E_available = E_rated * SoH '
                               '(terminal index; end-of-block SoH, also under the mid-block variant)'),
        'parameters': {'energy_recovery_fraction f_rec': sv['energy_recovery_fraction'],
                       'recycling_floor_fraction phi': sv['recycling_floor_fraction'], 'cost_basis': sv['cost_basis'],
                       'health_basis': sv['health_basis'], 'calendar_life_basis': sv['calendar_life_basis'],
                       'enabled': sv['enabled'], 'calendar_life_years': t_cal, 'minimum_soh': soh_min,
                       'SRP1 Years': srp1['Years'], 'DiscountFactor': r, 'y0': y0, 'y_end (years[0] + sum of blocks)': y_end,
                       'last representative year': int(years[-1])},
    }
    repro = []
    for yr, cell in (('2030', 'yl_y2030'), ('2035', 'yl_y2035')):
        boyd = X.j[f'YL{yr}_BOYD']['soh_floor_multiplier_and_efc_per_cohort_year_terminal']['per_node_per_cohort_year']
        yidx = years.index(yr)
        row = [b for b in boyd if int(b['node_id']) == 7 and int(b['y_inv']) == yidx and int(b['y']) == len(years) - 1]
        soh = float(row[0]['es_soh_per_unit_cumul'])
        cand = X.cand(cell)
        S, E = cand['nodes']['7']
        ce, probs, ce_rows = X.expected_cost('energy', yr)
        cs, _, cs_rows = X.expected_cost('power', yr)
        df = 1.0 / (1.0 + r) ** (y_end - y0)
        rem = max(0.0, int(yr) + t_cal - y_end)
        rlf = min(rem / t_cal, 1.0)
        health = (soh - soh_min) / (1.0 - soh_min)
        prod = df * sv['energy_recovery_fraction'] * ce * rlf * (sv['recycling_floor_fraction'] * E +
                                                                    (1 - sv['recycling_floor_fraction']) * E * health)
        frame_eq = df * (cs * S + ce * E) * rlf * soh
        recorded = X.t['year_ladder']['per_year'][yr]['salvage_last']
        repro.append({'cell': cell, 'investment_year': int(yr), 'node': 7, 'S_MVA': S, 'E_MWh': E,
                      'terminal_SoH_recorded': soh, 'terminal_SoH_source': f"{X.inputs[f'YL{yr}_BOYD']['path']} "
                      f"soh_floor_multiplier_and_efc_per_cohort_year_terminal (node 7, y_inv {yidx}, y {len(years) - 1})",
                      'probabilities': probs, 'c_E_by_scenario': ce_rows, 'c_S_by_scenario': cs_rows,
                      'expected_c_E': ce, 'expected_c_S': cs, 'DF_term': df, 'T_rem': rem, 'RLF': rlf,
                      'normalized_health': health, 'production_form_eur': prod, 'frame_equation_eur': frame_eq,
                      'recorded_salvage_eur': recorded,
                      'recorded_source': f'{FZ_REL} tables.year_ladder.per_year.{yr}.salvage_last',
                      'production_rel_diff': prod / recorded - 1.0, 'frame_over_recorded': frame_eq / recorded})
    code['reproduction'] = repro
    cohort2025 = {'T_rem': max(0.0, 2025 + t_cal - y_end), 'age_at_terminal_2025': sum(blocks)}
    return code, (y0, y_end, r, t_cal, soh_min), cohort2025


def confirm_checks(X, sal, consts, coh2025):
    y0, y_end, r, t_cal, soh_min = consts
    out = []

    # ---- C1 salvage ---------------------------------------------------------------------------------------------------
    rep = sal['reproduction']
    prod_ok = all(abs(x['production_rel_diff']) < 1e-9 for x in rep)
    frame_ok = all(abs(x['frame_over_recorded'] - 1) < 1e-6 for x in rep)
    points = {
        '(a) y^End is the end of the last block (2035 + Y_y)': {
            'outcome': 'confirmed' if y_end == int(list(X.j['SRP1_JSON']['Years'])[-1]) + X.j['SRP1_JSON']['Years'][list(X.j['SRP1_JSON']['Years'])[-1]] else 'differs',
            'evidence': f'_get_salvage_value_results: terminal_date = years[0] + sum(block lengths) = {y_end}; '
                        f'_get_remaining_calendar_life: age_at_terminal = sum of the block lengths from y_inv on (2025: '
                        f'{coh2025["age_at_terminal_2025"]}); SRP1.json Years {X.j["SRP1_JSON"]["Years"]}'},
        '(b) the SoH is the terminal SoH of the last block': {
            'outcome': 'confirmed (with the correction below: it enters normalized above the floor, not as SoH)',
            'evidence': 'e_available = es_e_available_per_unit[y_inv, terminal_year_idx] = E_rated x es_soh_per_unit_cumul'
                        '[y_inv, terminal] (end of the last block, after its num_years of fade); the mid-block variant '
                        'explicitly re-reads the end-of-block SoH; recorded terminal SoH reproduces the recorded salvage'},
        '(c) the cohort cost is the expected (probability-weighted) one': {
            'outcome': 'confirmed as worded (probability-weighted, undiscounted at y^Inv) -- but the cost weighted is the '
                       'ENERGY cost only (cost_basis EXPECTED_INSTALLATION_ENERGY_COST); the frame defines I with the '
                       'power term c^{Inv,S} S as well',
            'evidence': "_get_expected_energy_investment_cost: sum_s prob_s x cost_investment['energy'][s][year]; the "
                        'investment-cost discount 1/(1+r)^(y - y0) is applied only in I(x), not here'},
        '(d) the discount exponent is y^End - y_0': {
            'outcome': 'confirmed' if (y_end - y0) == 15 else 'differs',
            'evidence': f'_get_terminal_discount_factor: 1/(1+r)^(sum of block lengths) = 1/(1+{r})^{y_end - y0}'},
    }
    corrections = [
        {'term': 'I_{e,y^Inv} (the where-clause)', 'frame': 'sum_c omega_c (c^{Inv,S} S + c^{Inv,E} E)',
         'code': 'sum_c omega_c c^{Inv,E} E (energy only; the converter/power cost earns no salvage)'},
        {'term': 'SoH_{e,y^Inv,y^End} (last factor)', 'frame': 'SoH (the state of health itself)',
         'code': '(SoH - SoH^Min) / (1 - SoH^Min): the health normalized above the end-of-life floor (health_basis '
                 'NORMALIZED_ABOVE_MINIMUM_SOH; recycling floor fraction 0 and energy recovery fraction 1 as configured)'},
    ]
    corrected_tex = (
        r'C^\text{Salvage}(\boldsymbol{x}) = \sum_{e \in E^S} \sum_{y^\text{Inv} \in Y} '
        r'\frac{\bar{c}^{\text{Inv,E}}_{y^\text{Inv}} E^{\text{Inv}}_{e,y^\text{Inv}}}{(1+r)^{\,y^\text{End}-y_0}}'
        r'\;\frac{T^\text{Rem}_{e,y^\text{Inv}}}{T^\text{Cal}_{e,y^\text{Inv}}}'
        r'\;\frac{SoH_{e,y^\text{Inv},y^\text{End}} - SoH^\text{Min}}{1 - SoH^\text{Min}}, \qquad '
        r'\bar{c}^{\text{Inv,E}}_{y^\text{Inv}} = \sum_{c \in \Omega_C} \omega_c c^{\text{Inv,E}}_{y^\text{Inv},c}'
        ' -- and the where-clause: "the expected undiscounted energy cost of the cohort" (eq:salvage_remaining_life '
        'unchanged)')
    out.append({'id': 'C1', 'lines': '28-45 (comment 43-45)', 'point': 'eq:salvage_credit, eq:salvage_remaining_life and '
                'items (a)-(d) against _get_terminal_salvage_value_*',
                'outcome': 'differs' if not frame_ok else 'confirmed',
                'summary': ('no function named _get_terminal_salvage_value_* exists in the tracked *.py files; the reported '
                            'net uses _build_terminal_salvage_value_expression (via get_salvage_value). Items (a), (b), (c), '
                            '(d) hold as worded; eq:salvage_remaining_life holds; eq:salvage_credit needs TWO corrections: '
                            'the cohort cost is the expected ENERGY cost only (no c^{Inv,S} S term), and the health factor '
                            'is (SoH - SoH^Min)/(1 - SoH^Min), not SoH. The production form reproduces the recorded salvage '
                            f"of yl_y2030 / yl_y2035 to {max(abs(x['production_rel_diff']) for x in rep):.1e} relative; the "
                            f"frame's equation gives {rep[0]['frame_equation_eur']:,.2f} / {rep[1]['frame_equation_eur']:,.2f} "
                            f"EUR against the recorded {rep[0]['recorded_salvage_eur']:,.2f} / "
                            f"{rep[1]['recorded_salvage_eur']:,.2f} EUR"),
                'items': points, 'corrections': corrections, 'corrected_text': corrected_tex,
                'production_reproduces_records': prod_ok, 'frame_equation_reproduces_records': frame_ok,
                'evidence': sal})

    # ---- C2 S/E/duration of the value-ladder cells ----------------------------------------------------------------------
    rows = []
    for ln, claim, nodes, S, E, dur, year in VALUE_LADDER_ROWS:
        cell = X.other(claim)
        c = X.cand(cell)
        nz = sorted(int(n) for n, v in c['nodes'].items() if v[0] or v[1])
        ok_f = (nz == sorted(nodes) and all(c['nodes'][str(n)] == [S, E] for n in nodes) and
                all(abs(c['nodes'][str(n)][1] / c['nodes'][str(n)][0] - dur) < 1e-12 for n in nodes) and
                c['investment_year'] == year)
        recs = []
        for p in X.evalrecs(X.cells[cell]['eval_key']):
            d = _json(p)
            recs.append({'path': p, 'sha256': _sha(p), 'candidate_canonical': d.get('candidate_canonical'),
                         'candidate_key': d.get('candidate_key'), 'candidate_label': d.get('candidate_label'),
                         'eval_key': d.get('eval_key'),
                         'equal_to_frozen': d.get('candidate_canonical') == c and d.get('candidate_key') ==
                         X.cells[cell]['candidate_key'] and d.get('eval_key') == X.cells[cell]['eval_key']})
        rows.append({'line': ln, 'claim': claim, 'cell': cell, 'written': {'nodes': nodes, 'S': S, 'E': E, 'duration_h': dur,
                     'year': year}, 'frozen_candidate': c, 'frozen_candidate_key': X.cells[cell]['candidate_key'],
                     'frozen_eval_key': X.cells[cell]['eval_key'], 'written_equals_frozen': ok_f,
                     'evaluation_records': recs,
                     'records_found': bool(recs), 'records_equal_frozen': bool(recs) and all(x['equal_to_frozen'] for x in recs)})
    allok = all(x['written_equals_frozen'] and x['records_equal_frozen'] for x in rows)
    out.append({'id': 'C2', 'lines': '83-98 (comment 103-105)', 'point': 'S / E / duration of each B and C cell against its '
                'candidate record (S = E / duration)', 'outcome': 'confirmed' if allok else 'differs',
                'summary': f"{sum(x['written_equals_frozen'] for x in rows)} of {len(rows)} rows: node, S, E, E/S and year "
                           f"equal the frozen candidate; evaluation records found for {sum(x['records_found'] for x in rows)}, "
                           f"equal to the frozen candidate, candidate key and eval key for "
                           f"{sum(x['records_equal_frozen'] for x in rows)}", 'rows': rows})

    # ---- C3 I/J keys -----------------------------------------------------------------------------------------------------
    ij = []
    for (claim, cell_b), pref in zip((('I:m2:e2_value_minus_I', 'd_c52e1670'), ('J:e3_value_minus_I', 'd_4a82a64a'),
                                      ('J:e4_value_minus_I', 'd_3632b0ae'), ('J:e5_value_minus_I', 'd_36686489')),
                                     ('cc4eb5df', 'fb21d822', '11725e45', 'cab98853')):
        cell = X.other(claim)
        k_ij, k_b = X.cells[cell]['candidate_key'], X.cells[cell_b]['candidate_key']
        recs = [{'path': p, 'sha256': _sha(p), 'candidate_canonical': _json(p).get('candidate_canonical'),
                 'candidate_key': _json(p).get('candidate_key')} for p in X.evalrecs(X.cells[cell]['eval_key'])]
        ref = X.CL[claim]['ref_cell']
        ij.append({'claim': claim, 'cell': cell, 'frame_prefix': pref, 'candidate_key': k_ij, 'B_cell': cell_b,
                   'B_candidate_key': k_b, 'prefix_matches': k_ij.startswith(pref), 'same_as_B': k_ij == k_b,
                   'candidate': X.cand(cell), 'records': recs,
                   'records_equal': bool(recs) and all(x['candidate_key'] == k_ij and x['candidate_canonical'] == X.cand(cell)
                                                       for x in recs),
                   'ref_cell (x = 0 at m = 2)': ref, 'ref_candidate': X.cand(ref)})
    ok3 = all(x['prefix_matches'] and x['same_as_B'] and x['records_equal'] and x['candidate']['nodes']['7'][1] /
              x['candidate']['nodes']['7'][0] == 4.0 for x in ij)
    out.append({'id': 'C3', 'lines': '208-209', 'point': 'the I/J cells are the node-7 4 h plans of 2-5 MWh (keys cc4eb5df, '
                'fb21d822, 11725e45, cab98853 = the B cells n7_4h_e2 ... e5)', 'outcome': 'confirmed' if ok3 else 'differs',
                'summary': '; '.join(f"{x['cell']} {x['candidate_key'][:8]} = {x['B_cell']} ({x['candidate']['nodes']['7']} at "
                                     f"node 7, {x['candidate']['investment_year']})" for x in ij), 'rows': ij})

    # ---- C4 the F2 plan ----------------------------------------------------------------------------------------------------
    f2 = X.cells['ref:5ca4f86c']
    f2c = f2['candidate_canonical']
    recs = [{'path': p, 'sha256': _sha(p), 'candidate_canonical': _json(p).get('candidate_canonical'),
             'candidate_key': _json(p).get('candidate_key'), 'candidate_label': _json(p).get('candidate_label')}
            for p in X.evalrecs(f2['eval_key'])]
    w173 = X.w173['instances']['F2']
    written = {'year': 2030, 'nodes': {'5': [0.25, 0.5], '7': [1.0, 3.0]}}
    actual = {'year': f2c['investment_year'], 'nodes': {n: v for n, v in f2c['nodes'].items() if v[0] or v[1]}}
    out.append({'id': 'C4', 'lines': '130-131', 'point': 'the F2 plan behind ref:5ca4f86c (year, nodes, P, E)',
                'outcome': 'confirmed' if written == actual else 'differs',
                'summary': f"recorded: {actual['year']}, " + ', '.join(f"node {n} {v[0]:g} MVA / {v[1]:g} MWh" for n, v in
                                                                       actual['nodes'].items()) +
                           f"; W173 label {w173['label']} (z {w173['z']}, eval key {w173['eval_key'][:8]}); the frame writes "
                           f"node 7 1 MVA / 3 MWh",
                'corrected_text': 'node 5 0.25 MVA / 0.5 MWh and node 7 1 MVA / 3.5 MWh (2030)',
                'written': written, 'frozen': {'candidate': f2c, 'candidate_key': f2['candidate_key'], 'eval_key': f2['eval_key']},
                'evaluation_records': recs, 'w173_instance': w173,
                'consistency': 'the T1 L neighbour y2030__n5_p0.25_e0.5__n7_p1_e3 (the "F2 neighbour" of l. 391-392) is the '
                               'incumbent with node-7 E one lattice unit (0.5 MWh) lower -- consistent with 3.5 MWh, and it '
                               'would be the incumbent itself if the plan were 1 MVA / 3 MWh'})

    # ---- C5 the 0.50-floor cell ------------------------------------------------------------------------------------------
    a = X.A64['E:soh050:value_minus_I']
    value = a['d_gross'] + a['I_other']
    w155 = X.w155
    q_ref = X.j['W155_SOH050']['scored']['claims']['E:soh050:value_minus_I']
    v_rec = q_ref['ref']['Q'] - q_ref['other']['Q']
    out.append({'id': 'C5', 'lines': '262, 269', 'point': 'the value and AE / EFC of the 0.50-floor cell (e_soh050)',
                'outcome': 'confirmed (values available)',
                'summary': f"value {value:,.2f} EUR ({value / 1e3:.1f} kEUR) = value - I {a['d_gross']:,.2f} + I "
                           f"{a['I_other']:,.2f} (frozen A64 row) = Q(0) - Q(e_soh050) {v_rec:,.2f} (W155 summary); AE "
                           f"{w155['AE']:.4f} ({w155['AE']:.3f}), EFC/day {w155['EFC']:.4f} ({w155['EFC']:.2f}) PV-weighted "
                           f"(W155 summary reports.e_soh050.AE / EFC; the T8 definition M46A.pv of the ageing series); "
                           f"floor binds: never (floor_year {w155['floor_year']}, minimum_soh {w155['minimum_soh']}, |dual| <= "
                           f"{X.t['a64']['scored']['B']['floor_duals_abs_max']:.1e})",
                'fill_for_row_262': {'Value (kEUR)': f'{value / 1e3:.1f}', 'AE': f"{w155['AE']:.3f}",
                                     'EFC/day': f"{w155['EFC']:.2f}", 'Floor binds': 'never'},
                'sources': {'value': f'{FZ_REL} tables.a64.rows[E:soh050:value_minus_I] d_gross + I_other',
                            'value_check': f"{X.inputs['W155_SOH050']['path']} scored.claims.E:soh050:value_minus_I ref.Q - other.Q",
                            'AE_EFC': f"{X.inputs['W155_SOH050']['path']} reports.e_soh050.AE / EFC",
                            'definition': 'p515_s53_w155_a64_campaign.py and p515_s53_w142_resettle_ext_v6_campaign.py: '
                                          "'AE': M46A.pv(series['soh_used']), 'EFC': M46A.pv(series['efc_per_day'])"},
                'value_eur': value, 'value_recomputed_from_W155_Q': v_rec, 'AE': w155['AE'], 'EFC': w155['EFC'],
                'note': 'these values are in a committed record but not in the frozen T1-T11 (T10 carries value - I and the '
                        'per-year EFC only); filling the row makes them named-record figures'})

    # ---- C6 16 cycles -----------------------------------------------------------------------------------------------------
    rg = X.j['W98R2_RESULTS']['replay_gate']
    w99 = X.j['W99']['item3_POSTHOC_damped_oscillation']
    win = w99['damped_cosine_primary']['window']
    n_past = max(X.cont_cycles) - rg['N']
    out.append({'id': 'C6', 'lines': '342', 'point': 'the 3 x 3 "16 cycles" of the post-hoc continuation',
                'outcome': 'confirmed' if n_past == 16 else 'differs',
                'summary': f"W98 r2 continuation: replay bitwise through N = {rg['N']} (the production exit, cycle 72), "
                           f"continuation record cycles {min(X.cont_cycles)}-{max(X.cont_cycles)} -> {n_past} cycles past the "
                           f"exit ({rg['N'] + 1}-{max(X.cont_cycles)}), stopped by the frozen |dQ| < 500 EUR x 3 rule; W99 "
                           f"damped-cosine fit window {win} ({w99['damped_cosine_primary']['n_points']} points), L = "
                           f"{w99['damped_cosine_primary']['params']['L']:,.2f} EUR (= D, T11 6.2 kEUR)",
                'wording': 'the 16 cycles are the continuation of the evaluation past the exit; the damped-cosine fit is '
                           'over cycles 72-88 (17 points). "a damped-cosine fit continued 16 cycles past the exit" reads as '
                           'if the fit were continued; suggested: "from a damped-cosine fit to the evaluation continued 16 '
                           'cycles past the exit"',
                'sources': {'continuation': X.inputs['W98R2_CONT']['path'], 'replay_gate': X.inputs['W98R2_RESULTS']['path'],
                            'fit': X.inputs['W99']['path']}})

    # ---- C7 W153 sign changes ----------------------------------------------------------------------------------------------
    tot = X.w153s['totals']
    vd = [c for c in X.w153s['claims'] if c['verdict_differs_between_conventions']]
    cl = X.CL['L:y2030__n5_p0.25_e0.5__n7_p1_e3']
    fz_sign = sum(1 for c in X.t['claims'] if (c['d_gross'] > 0) != (c['d_net'] > 0))
    ok7 = (tot['n_claims'] == 60 and tot['n_sign_differs'] == 0 and tot['n_verdict_differs'] == 1 and
           tot['claims_verdict_differs'] == ['L:y2030__n5_p0.25_e0.5__n7_p1_e3'] and fz_sign == 0 and
           f"{cl['gross_multiple']:.2f}" == '0.95' and f"{cl['net_multiple']:.2f}" == '1.22')
    out.append({'id': 'C7', 'lines': '390-392', 'point': 'W153 totals: "0 sign changes in 60 claims"; one verdict changes '
                'category (the F2 neighbour y2030__n5_p0.25_e0.5__n7_p1_e3, 0.95 x gross, 1.22 x net)',
                'outcome': 'confirmed' if ok7 else 'differs',
                'summary': f"W153 totals: n_claims {tot['n_claims']}, n_sign_differs {tot['n_sign_differs']}, n_verdict_differs "
                           f"{tot['n_verdict_differs']} ({tot['claims_verdict_differs']}: {vd[0]['gross_verdict']} gross -> "
                           f"{vd[0]['net_verdict']} net); frozen T1: {len(X.t['claims'])} claims, {fz_sign} sign changes; the "
                           f"claim at {cl['gross_multiple']:.4f} x gross ({cl['gross_verdict']}) and {cl['net_multiple']:.4f} "
                           f"x net ({cl['net_verdict_recomputed_here']})",
                'sources': {'W153': X.inputs['W153_SALV']['path'], 'T1': f'{FZ_REL} tables.claims'}})

    # ---- C8 the 2025 cohort's salvage -------------------------------------------------------------------------------------
    c2025 = {k: v.get('salvage') for k, v in X.t['cells'].items() if v['candidate_canonical']['investment_year'] == 2025}
    max_abs = max(abs(v) for v in c2025.values() if v is not None)
    d_eq = [c['claim_id'] for c in X.t['claims'] if all(X.cand(cc)['investment_year'] == 2025 for cc in
                                                         (c['ref_cell'], c['other_cell']) if cc in X.cells)
            and c['d_gross'] != c['d_net']]
    ok8 = coh2025['T_rem'] == 0.0 and max_abs < 1e-20 and not d_eq and X.t['cells']['ref:bd504ecf']['salvage'] == 0.0
    out.append({'id': 'C8', 'lines': '403, 410-411', 'point': "the 2025 cohort's salvage is zero (T^Rem = 0 at y^End for "
                'T^Cal = 15 and y^Inv = 2025)', 'outcome': 'confirmed' if ok8 else 'differs',
                'summary': f"code: age_at_terminal(2025) = {coh2025['age_at_terminal_2025']:g}, T_rem = max({t_cal:g} - "
                           f"{coh2025['age_at_terminal_2025']:g}, 0) = {coh2025['T_rem']:g} -> remaining-life fraction 0, the "
                           f"2025 term is exactly 0; records: {len(c2025)} cells with a 2025 plan, max |salvage| "
                           f"{max_abs:.2e} EUR (numerical residue of the inactive cohorts' terms, not the 2025 cohort), "
                           f"ref:bd504ecf salvage {X.t['cells']['ref:bd504ecf']['salvage']!r}; T1: d_net = d_gross on every "
                           f"claim whose cells are all 2025 plans ({'none differ' if not d_eq else d_eq}); the row may print "
                           f"0.0", 'cells_2025_salvage': c2025})

    # ---- C9 41 = 14 + 20 + 7 -------------------------------------------------------------------------------------------
    it = X.w173['item1']
    n = [it[k]['n_new_evaluations_total'] for k in ('s47', 's51', 's53')]
    b = [it[k]['max_new_evaluations'] for k in ('s47', 's51', 's53')]
    out.append({'id': 'C9', 'lines': '452', 'point': '"41 new recourse evaluations (14, 20 and 7 against budgets of 20, 20 '
                'and 60)" = W173 realized use, summed', 'outcome': 'confirmed' if (n == [14, 20, 7] and b == [20, 20, 60]
                                                                                  and sum(n) == 41) else 'differs',
                'summary': f"W173 item1: s47 {n[0]}/{b[0]} ({it['s47']['termination']['reason']}), s51 {n[1]}/{b[1]} "
                           f"({it['s51']['termination']['reason']}), s53 {n[2]}/{b[2]} ({it['s53']['termination']['reason']}); "
                           f"sum {sum(n)}", 'source': X.inputs['W173']['path']})
    return out


# ======================================================================================================================
#  3. the number check: declarations
# ======================================================================================================================
# (line, claim, nodes, S, E, duration h, year) of tab:res_value_ladder
VALUE_LADDER_ROWS = (
    (83, 'C:y2025__n5_p0.25_e0.5', [5], 0.25, 0.5, 2, 2025),
    (84, 'B:n5_4h_e1', [5], 0.25, 1.0, 4, 2025),
    (85, 'B:n7_4h_e1', [7], 0.25, 1.0, 4, 2025),
    (86, 'B:n7_4h_e2', [7], 0.5, 2.0, 4, 2025),
    (87, 'B:n7_4h_e3', [7], 0.75, 3.0, 4, 2025),
    (88, 'B:n7_4h_e4', [7], 1.0, 4.0, 4, 2025),
    (89, 'B:n7_4h_e5', [7], 1.25, 5.0, 4, 2025),
    (90, 'B:n7_2h_e1', [7], 0.5, 1.0, 2, 2025),
    (91, 'B:n7_2h_e2', [7], 1.0, 2.0, 2, 2025),
    (92, 'B:n7_2h_e3', [7], 1.5, 3.0, 2, 2025),
    (93, 'B:n7_2h_e4', [7], 2.0, 4.0, 2, 2025),
    (94, 'B:n7_2h_e5', [7], 2.5, 5.0, 2, 2025),
    (95, 'B:n9_4h_e1', [9], 0.25, 1.0, 4, 2025),
    (96, 'B:n9_4h_e3', [9], 0.75, 3.0, 4, 2025),
    (97, 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', [5, 9], 0.25, 0.5, 2, 2025),
    (98, 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', [5, 7], 0.25, 0.5, 2, 2030),
)


def S(w, kind, src, desc=None, **o):
    """one token spec: `src` a frozen-table path (str), ('rec', KEY, path), or a callable X -> value (desc names it)."""
    return dict(w=w, kind=kind, src=src, desc=desc, **o)


def U(w, cat, why):
    return dict(w=w, kind='unchecked', cat=cat, why=why)


def P(w, field, cell, nodes=None):
    return dict(w=w, kind='plan', field=field, cell=cell, nodes=nodes)


def W3(w, cid, value, why=None):
    return dict(w=w, kind='w163', id=cid, value=value, why=why)


def DEF(w, key, phrase, why=None):
    return dict(w=w, kind='def', key=key, phrase=phrase, why=why)


def D(line, frag, *specs, note=None):
    return {'line': line, 'frag': frag, 'specs': list(specs), 'note': note}


def NOT(w):
    return U(w, 'notation', 'the plan without storage x = 0 / Q(0) / F(0) (a symbol, not a figure)')


def SINGLE(w):
    return U(w, 'compound adjective', '"single-scenario" (the instance name, not a count)')


def INSTR(w):
    return U(w, 'instruction', 'the expert\'s instruction to the author (paragraph count), not a manuscript figure')


def LANG(w, why='a counting word of the prose, not a figure'):
    return U(w, 'language', why)


def XREF(w, why='a cross-reference number (section, sentence, algorithm or table numbering), not a figure'):
    return U(w, 'cross-reference', why)


def _cl(cid, f):
    return f'claims[{cid}].{f}'


def _t10(cid, f):
    return f'a64.rows[{cid}].{f}'


def ladder_row(line, claim, nodes, s_, e_, dur, year, written):
    """specs of a tab:res_value_ladder row: node(s), S, E, [year], duration, d gross, threshold or bar, x."""
    cell = claim
    out = [P(str(n), 'node', cell, [n]) for n in nodes]
    out += [P(written[0], 'S', cell, nodes), P(written[1], 'E', cell, nodes)]
    if year != 2025:
        out.append(P(str(year), 'year', cell))
    out.append(P(written[2], 'dur', cell, nodes))
    out += [S(written[3], 'k', _cl(claim, 'd_gross'), signed=True), S(written[4], 'k', _cl(claim, 'gross_threshold_or_bar')),
            S(written[5], 'x', _cl(claim, 'gross_multiple'))]
    return D(line, None, *out)


def _min_max_aged(X, f, which):
    vals = [r[f] for r in X.t['ageing']['rows'] if r['arm'] != 'no_ageing']
    return (min if which == 'min' else max)(vals)


def _range(X, path_list, which):
    vals = [X.fzp(p) for p in path_list]
    return (min if which == 'min' else max)(vals)


def declarations():
    VL = {r[0]: r for r in VALUE_LADDER_ROWS}
    L = []
    # ---- header comment ------------------------------------------------------------------------------------------------
    L.append(D(2, None, XREF('4', 'the section number of the frame ("SECTION 4")')))
    L.append(D(3, None, S('1126', 'int', lambda X: X.main_results_line, 'main.tex at the clone HEAD: the line of '
                          '\\section{Results}'),
               S('1448', 'int', lambda X: X.main_before_banner, 'main.tex at the clone HEAD: the line before the CONCLUSIONS '
                 'banner (the header\'s own rule; Addendum 73: 1126-1445)')))
    # ---- preamble (body) -----------------------------------------------------------------------------------------------
    L.append(D(25, None, SINGLE('single'),
               S('3', 'int', ('rec', 'CASE_3X3', 'NumMarketScenarios'), 'market scenarios of the 3 x 3 instance'),
               S('3', 'int', ('rec', 'CASE_3X3', 'TransmissionNetwork.num_operation_scenarios'),
                 'operation scenarios of the 3 x 3 instance'),
               LANG('two', 'a difference between two evaluations (definition)'), NOT('0'),
               S('15', 'int', lambda X: sum(X.j['SRP1_JSON']['Years'].values()), 'SRP1.json Years: sum of the block lengths'),
               S('2', 'pct0', ('rec', 'SRP1_JSON', 'DiscountFactor'), 'SRP1.json DiscountFactor (x 100)')))
    L.append(D(31, None, U('1', 'notation', 'the 1 of (1 + r) in eq:salvage_credit (formula; checked by CONFIRM C1)')))
    L.append(D(37, None, U('0', 'notation', 'max{0, ...} in eq:salvage_remaining_life (formula; CONFIRM C1: '
                                            '_get_remaining_calendar_life max(..., 0.00))')))
    L.append(D(42, None, NOT('0')))
    L.append(D(44, None, S('2035', 'int', lambda X: int(list(X.j['SRP1_JSON']['Years'])[-1]),
                           'SRP1.json: the last representative year')))
    L.append(D(45, None, LANG('one', 'pronoun ("the expected ... one")')))
    # ---- 4.1 author slot -----------------------------------------------------------------------------------------------
    L.append(D(52, None, INSTR('One'), INSTR('two')))
    L.append(D(53, None, XREF('1', 'Algorithm 1'), NOT('0'), SINGLE('single')))
    L.append(D(54, None, S('fourteen', 'int', ('rec', 'W173', 'item2.x0.b_box.n_feasible'),
                           'W173: admissible unit neighbours of x = 0 (all evaluated)'), XREF('2.1', 'section 2.1')))
    L.append(D(56, None, P('7', 'node', 'CHECK:headline_V_minus_I_settled', [7]),
               P('0.25', 'S', 'CHECK:headline_V_minus_I_settled', [7]), P('1', 'E', 'CHECK:headline_V_minus_I_settled', [7]),
               P('2025', 'year', 'CHECK:headline_V_minus_I_settled'), S('253.5', 'k', 'ageing.rows[C2_calfade].value')))
    L.append(D(57, None, S('318.0', 'k', _cl('CHECK:headline_V_minus_I_settled', 'I_other')),
               S('-64.4', 'k', _cl('CHECK:headline_V_minus_I_settled', 'd_gross')),
               S('4.93', 'x', _cl('CHECK:headline_V_minus_I_settled', 'gross_multiple')),
               S('13.1', 'k', _cl('CHECK:headline_V_minus_I_settled', 'gross_threshold_or_bar')),
               S('2', 'pct0', 'discount[0.02].rate', 'T7 row 2 % (rate x 100)')))
    L.append(D(58, None, S('4.39', 'x', lambda X: min(X.CL[c]['gross_multiple'] for c in X.CL if c.startswith('B:')),
                           'T1 B rows: smallest multiple (n5_4h_e1); note: the C row n5 0.5 MWh is 3.94 x'),
               S('45.94', 'x', lambda X: max(X.CL[c]['gross_multiple'] for c in X.CL if c.startswith('B:')),
                 'T1 B rows: largest multiple (n7_2h_e5)'),
               note='"every evaluated plan ... 4.39 x to 45.94 x" spans the B rows only: the C rows of the same table are '
                    '3.94 x, 8.09 x and 7.00 x, so over the table the range is 3.94-45.94 x'))
    L.append(D(59, None, S('64.4', 'k', _cl('B:n7_4h_e1', 'd_gross')), S('347.2', 'k', _cl('B:n7_4h_e5', 'd_gross')),
               P('1', 'E', 'B:n7_4h_e1', [7]), P('5', 'E', 'B:n7_4h_e5', [7]), P('4', 'dur', 'B:n7_4h_e5', [7]),
               S('114.1', 'k', _cl('B:n7_2h_e1', 'd_gross')), S('616.6', 'k', _cl('B:n7_2h_e5', 'd_gross')),
               P('2', 'dur', 'B:n7_2h_e5', [7])))
    L.append(D(60, None, P('2', 'dur', 'B:n7_2h_e1', [7]), P('0.5', 'E', 'C:y2025__n5_p0.25_e0.5', [5]),
               P('5', 'node', 'C:y2025__n5_p0.25_e0.5', [5])))
    L.append(D(61, None, S('49.7', 'k', _cl('C:y2025__n5_p0.25_e0.5', 'd_gross')),
               S('3.94', 'x', _cl('C:y2025__n5_p0.25_e0.5', 'gross_multiple'))))
    L.append(D(63, None, P('Two', 'nnodes', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5'),
               S('102.1', 'k', _cl('C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', 'd_gross')),
               P('5', 'node', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', [5]),
               P('9', 'node', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', [9]),
               P('2025', 'year', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5'),
               S('93.9', 'k', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'd_gross')),
               P('5', 'node', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', [5]),
               P('7', 'node', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', [7]),
               note='"lose more than their parts would" is not resolved by the records: the parts (T5 / T1 C, single-node '
                    '0.25 MVA / 0.5 MWh) sum to 49.7 + 52.5 = 102.2 kEUR against the 2025 pair\'s 102.1 (pair lower by 0.1) '
                    'and to 43.3 + 49.0 = 92.3 kEUR against the 2030 pair\'s 93.9 (+1.6); every term carries a threshold '
                    'of 12.6-13.5 kEUR, so neither difference is determinate'))
    L.append(D(64, None, P('2030', 'year', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5'),
               S('84.0', 'k', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'd_net'))))
    L.append(D(65, None, P('2025', 'year', 'C:y2025__n5_p0.25_e0.5'),
               note='"identical d net, since all cohorts reach y^End with the same remaining life": d net = d gross on every '
                    '2025 plan holds; the reason is that the 2025 cohort\'s remaining life at y^End is ZERO (CONFIRM C8)'))
    L.append(D(66, None, P('2030', 'year', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5'),
               S('93.9', 'k', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'd_gross')),
               S('84.0', 'k', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'd_net'))))
    L.append(D(67, None, NOT('0'), S('-81.3', 'k', 'three_by_three.rows[2].value'),
               S('18.0', 'k', 'three_by_three.rows[3].value'), S('4.52', 'x', 'three_by_three.rows[4].value')))
    L.append(D(69, None, P('0.25', 'S', 'CHECK:headline_V_minus_I_settled', [7]),
               P('1', 'E', 'CHECK:headline_V_minus_I_settled', [7])))
    L.append(D(70, None, S('0.5', 'raw', lambda X: float(re.search(r'granules [0-9.]+ MVA, ([0-9.]+) MWh',
                                                                   X.w173['box_definition']).group(1)),
                           'W173 box_definition: energy granule 0.5 MWh (smallest admissible E at E/P >= 2 with 0.25 MVA)')))
    L.append(D(71, None, P('0.25', 'S', 'CHECK:headline_V_minus_I_settled', [7]),
               P('1', 'E', 'CHECK:headline_V_minus_I_settled', [7])))
    L.append(D(72, None, P('0.25', 'S', 'C:y2025__n5_p0.25_e0.5', [5]), P('0.5', 'E', 'C:y2025__n5_p0.25_e0.5', [5])))
    # ---- tab:res_value_ladder ---------------------------------------------------------------------------------------
    L.append(D(76, None, SINGLE('single'), P('2025', 'year', 'B:n7_4h_e1'), NOT('0'),
               W3('3', 'N28', 3.0, 'DETERMINACY_BAR_FACTOR 3 (max{3 x larger band, 2 tau})'),
               W3('2', 'N29', 2.0, 'DETERMINACY_TAU_MULTIPLE 2'),
               W3('3', 'N26', 3.0, 'the uncertified bar 3 max{|t_sum|, |s|}'),
               S('70.7', 'k', _cl('B:n7_4h_e3', 'uncertified_form_beside_A64_ruling4.bar')),
               S('2.94', 'x', _cl('B:n7_4h_e3', 'uncertified_form_beside_A64_ruling4.multiple_Q')),
               P('2025', 'year', 'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5'),
               P('2030', 'year', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5'),
               S('84.0', 'k', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'd_net')),
               S('7.00', 'x', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'gross_multiple')),
               S('6.27', 'x', _cl('C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5', 'net_multiple'))))
    L.append(D(81, None, NOT('0')))
    LADDER_W = {83: ('0.25', '0.5', '2', '49.7', '12.6', '3.94'), 84: ('0.25', '1', '4', '58.0', '13.2', '4.39'),
                85: ('0.25', '1', '4', '64.4', '13.1', '4.93'), 86: ('0.5', '2', '4', '135.5', '13.4', '10.13'),
                87: ('0.75', '3', '4', '208.3', '12.6', '16.49'), 88: ('1', '4', '4', '274.0', '13.6', '20.19'),
                89: ('1.25', '5', '4', '347.2', '23.9', '14.52'), 90: ('0.5', '1', '2', '114.1', '12.6', '9.03'),
                91: ('1', '2', '2', '231.7', '24.2', '9.56'), 92: ('1.5', '3', '2', '361.7', '41.4', '8.74'),
                93: ('2', '4', '2', '494.4', '13.4', '36.91'), 94: ('2.5', '5', '2', '616.6', '13.4', '45.94'),
                95: ('0.25', '1', '4', '61.1', '12.6', '4.84'), 96: ('0.75', '3', '4', '199.1', '13.5', '14.78'),
                97: ('0.25', '0.5', '2', '102.1', '12.6', '8.09'), 98: ('0.25', '0.5', '2', '93.9', '13.4', '7.00')}
    for ln, w in LADDER_W.items():
        _, claim, nodes, s_, e_, dur, year = VL[ln]
        L.append(ladder_row(ln, claim, nodes, s_, e_, dur, year, w))
    L.append(D(103, None, S('0.25', 'raw', lambda X: 1 / 4, 'S / E at 4 h = 1/4 (arithmetic; CONFIRM C2)'),
               P('4', 'dur', 'B:n7_4h_e1', [7]), S('0.5', 'raw', lambda X: 1 / 2, 'S / E at 2 h = 1/2'),
               P('2', 'dur', 'B:n7_2h_e1', [7])))
    # ---- 4.2 author slot ---------------------------------------------------------------------------------------------
    L.append(D(112, None, INSTR('Three')))
    L.append(D(114, None, S('7', 'int', lambda X: _single_node(X, 'conservative_A64'), 'T3: every fitted point is a node-7 '
                                                                                         'plan (certified list)')))
    L.append(D(115, None, P('4', 'dur', 'B:n7_4h_e1', [7]), P('2', 'dur', 'B:n7_2h_e1', [7]),
               DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'), DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4')))
    L.append(D(116, None, S('256,320', 'e10', 'break_even.p_cost_eur_per_MVA'),
               S('6', 'int', 'break_even.conservative_A64.certified_only.n'),
               S('183,080', 'e10', 'break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost'),
               S('10', 'int', 'break_even.conservative_A64.banded_mid.n')))
    L.append(D(117, None, S('four', 'int', lambda X: len(X.t['break_even']['conservative_A64']['uncertified_as_intervals']),
                            'T3 conservative A64: uncertified_as_intervals (3 uncertified + the flagged n7_4h_e3)'),
               S('171,310', 'e10', 'break_even.conservative_A64.breakeven_range[0]'),
               S('192,560', 'e10', 'break_even.conservative_A64.breakeven_range[1]')))
    L.append(D(118, None, S('253,880', 'e10', 'break_even.e_cost_eur_per_MWh'),
               S('61,320', 'e10', 'break_even.conservative_A64.margin_to_cost_range[0]'),
               S('82,570', 'e10', 'break_even.conservative_A64.margin_to_cost_range[1]')))
    L.append(D(119, None, DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'), LANG('two', 'the two fits (certified-only and banded)'),
               S('0.47', 'pct2', 'break_even.conservative_A64.slope_b_plus_c_over_4_rel_mid'),
               S('4.76', 'pct2', 'break_even.conservative_A64.slope_b_plus_c_over_4_rel_corners[0]'),
               S('3.83', 'pct2', 'break_even.conservative_A64.slope_b_plus_c_over_4_rel_corners[1]')))
    L.append(D(120, None, S('173,490', 'e10', 'break_even.committed_W145.breakeven_range[0]'),
               S('190,380', 'e10', 'break_even.committed_W145.breakeven_range[1]')))
    L.append(D(122, None, XREF('1', 'accepted sentence 1 (paragraphs_v5.md); checked verbatim in the text checks')))
    L.append(D(123, None, S('1.75', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value'),
                            'fig2 data (W165): the multiplier of the m = 1.75 cells'),
               S('2', 'raw', ('rec', 'FIG2_DATA', 'values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value'), 'fig2 data: m = 2'),
               S('+14.2', 'k', _t10('H:m1.75:value_minus_I', 'd_gross')), S('+46.3', 'k', _cl('H:m2:value_minus_I', 'd_gross'))))
    L.append(D(124, None, S('1.5', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value')),
               S('1.75', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value')),
               W3('1.63', 'S1c', 1.6258759476587705, 'linear interpolation of value - I between m = 1.5 and 1.75'),
               S('1.5', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value'))))
    L.append(D(125, None, S('-14.5', 'k', _cl('H:m1.5:value_minus_I', 'd_gross')),
               S('18.8', 'k', _cl('H:m1.5:value_minus_I', 'gross_threshold_or_bar')),
               S('0.77', 'x', _cl('H:m1.5:value_minus_I', 'gross_multiple')),
               S('1.5', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value'))))
    L.append(D(127, None, S('2', 'raw', ('rec', 'FIG2_DATA', 'values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value')),
               P('7', 'node', 'J:e5_value_minus_I', [7]), P('4', 'dur', 'J:e5_value_minus_I', [7])))
    L.append(D(128, None, S('46.3', 'k', _cl('H:m2:value_minus_I', 'd_gross')), P('1', 'E', 'H:m2:value_minus_I', [7]),
               S('228.4', 'k', _cl('J:e5_value_minus_I', 'd_gross')), P('5', 'E', 'J:e5_value_minus_I', [7]),
               S('46.5', 'k', _cl('I:m2:second_MWh', 'd_gross')), S('41.0', 'k', _cl('J:e2_to_e3', 'd_gross')),
               S('42.8', 'k', _cl('J:e3_to_e4', 'd_gross'))))
    L.append(D(129, None, S('51.8', 'k', _cl('J:e4_to_e5', 'd_gross')),
               S('12.8', 'k', _cl('I:m2:second_MWh', 'gross_threshold_or_bar'))))
    L.append(D(130, None, S('26.5', 'k', lambda X: _range(X, [_cl(c, 'gross_threshold_or_bar') for c in
                                                             ('J:e2_to_e3', 'J:e3_to_e4', 'J:e4_to_e5')], 'min'),
                            'T1 J increments: smallest uncertified bar (J:e4_to_e5)'),
               S('27.7', 'k', lambda X: _range(X, [_cl(c, 'gross_threshold_or_bar') for c in
                                                   ('J:e2_to_e3', 'J:e3_to_e4', 'J:e4_to_e5')], 'max'),
                 'T1 J increments: largest uncertified bar'),
               P('two', 'nnodes', 'ref:5ca4f86c'), P('2030', 'year', 'ref:5ca4f86c'), P('5', 'node', 'ref:5ca4f86c', [5])))
    L.append(D(131, None, P('0.25', 'S', 'ref:5ca4f86c', [5]), P('0.5', 'E', 'ref:5ca4f86c', [5]),
               P('7', 'node', 'ref:5ca4f86c', [7]), P('1', 'S', 'ref:5ca4f86c', [7]), P('3', 'E', 'ref:5ca4f86c', [7])))
    L.append(D(132, None, S('9.2', 'k', 'cells[ref:5ca4f86c].gap'), S('1.5', 'k', 'cells[ref:5ca4f86c].slack'),
               S('61', 'int', ('rec', 'W173', 'item2.F2.b_box.n_feasible')),
               S('seventeen', 'int', ('rec', 'W173', 'item2.F2.b_box.n_evaluated_in_search'))))
    L.append(D(133, None, S('thirteen', 'int', ('rec', 'W173', 'item2.F2.c_L_claims_box_neighbours.n'))))
    L.append(D(134, None, S('twelve', 'int', ('rec', 'W173', 'item2.F2.c_L_claims_box_neighbours.n_positive_plan_better')),
               S('seven', 'int', ('rec', 'W173', 'item2.F2.c_L_claims_box_neighbours.n_positive_determinate')),
               S('five', 'int', ('rec', 'W173', 'item2.F2.c_L_claims_box_neighbours.n_positive_within_bar'))))
    L.append(D(135, None, XREF('2.1', 'section 2.1'),
               S('+5.6', 'k', lambda X: min(c['d_gross'] for k, c in X.CL.items() if k.startswith('L:') and
                                            'not_a_neighbour' not in k and c['d_gross'] > 0),
                 'T1 L rows (box neighbours): smallest positive d gross'),
               S('+94.3', 'k', lambda X: max(c['d_gross'] for k, c in X.CL.items() if k.startswith('L:') and
                                             'not_a_neighbour' not in k), 'T1 L rows: largest d gross'),
               S('-6.1', 'k', _cl('L:y2030__n5_p0.25_e1__n7_p1_e3', 'd_gross'))))
    L.append(D(136, None, S('27.9', 'k', _cl('L:y2030__n5_p0.25_e1__n7_p1_e3', 'gross_threshold_or_bar')),
               LANG('one', 'pronoun ("when one pays")'), NOT('0')))
    L.append(D(138, None, S('2', 'raw', ('rec', 'FIG2_DATA', 'values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value')),
               P('two', 'nnodes', 'ref:5ca4f86c'), S('44', 'int', ('rec', 'W173', 'item2.F2.b_box.n_not_evaluated_in_search')),
               S('61', 'int', ('rec', 'W173', 'item2.F2.b_box.n_feasible'))))
    # ---- fig:res_breakeven ----------------------------------------------------------------------------------------------
    L.append(D(144, None, P('7', 'node', 'B:n7_4h_e1', [7]), P('2025', 'year', 'B:n7_4h_e1'),
               P('4', 'dur', 'B:n7_4h_e1', [7]), P('2', 'dur', 'B:n7_2h_e1', [7]),
               W3('3', 'N26', 3.0, 'the uncertified bar 3 max{|t_sum|, |s|}'),
               DEF('2', 'CAPTIONS', 'bar + 2 tau', 'the interval of the banded fit: bar + 2 tau (captions.md, W165)'),
               S('6', 'int', 'break_even.conservative_A64.certified_only.n'),
               S('10', 'int', 'break_even.conservative_A64.banded_mid.n'),
               DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'), DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'),
               S('253.88', 'kmwh', 'break_even.e_cost_eur_per_MWh'),
               S('183.08', 'kmwh', 'break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost'),
               S('171.31', 'kmwh', 'break_even.conservative_A64.breakeven_range[0]'),
               S('192.56', 'kmwh', 'break_even.conservative_A64.breakeven_range[1]'),
               S('61.32', 'kmwh', 'break_even.conservative_A64.margin_to_cost_range[0]')))
    # ---- tab:res_breakeven -----------------------------------------------------------------------------------------------
    L.append(D(150, None, P('7', 'node', 'B:n7_4h_e1', [7]), DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'),
               DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'), S('256,320', 'e10', 'break_even.p_cost_eur_per_MVA'),
               S('253,880', 'e10', 'break_even.e_cost_eur_per_MWh'), DEF('4', 'BE_CONV', 'e* = b + c/4 - p_cost/4'),
               LANG('two', 'the two fits')))
    L.append(D(158, None, S('Four', 'int', lambda X: len(X.t['break_even']['conservative_A64']['uncertified_as_intervals']),
                            'T3 conservative A64 intervals'),
               S('6', 'int', 'break_even.conservative_A64.certified_only.n'),
               S('183,080', 'e10', 'break_even.conservative_A64.certified_only.breakeven_marginal_4h_energy_cost'),
               S('171,310', 'e10', 'break_even.conservative_A64.breakeven_range[0]'),
               S('192,560', 'e10', 'break_even.conservative_A64.breakeven_range[1]'),
               S('61,320', 'e10', 'break_even.conservative_A64.margin_to_cost_range[0]'),
               S('82,570', 'e10', 'break_even.conservative_A64.margin_to_cost_range[1]'),
               S('0.47', 'pct2', 'break_even.conservative_A64.slope_b_plus_c_over_4_rel_mid')))
    L.append(D(159, None, S('Three', 'int', lambda X: len(X.t['break_even']['committed_W145']['uncertified_as_intervals']),
                            'T3 committed W145 intervals'),
               S('7', 'int', 'break_even.committed_W145.certified_only.n'),
               S('182,700', 'e10', 'break_even.committed_W145.certified_only.breakeven_marginal_4h_energy_cost'),
               S('173,490', 'e10', 'break_even.committed_W145.breakeven_range[0]'),
               S('190,380', 'e10', 'break_even.committed_W145.breakeven_range[1]'),
               S('63,500', 'e10', 'break_even.committed_W145.margin_to_cost_range[0]'),
               S('80,390', 'e10', 'break_even.committed_W145.margin_to_cost_range[1]'),
               S('0.31', 'pct2', 'break_even.committed_W145.slope_b_plus_c_over_4_rel_mid')))
    L.append(D(162, None, S('4', 'int', lambda X: len(X.t['break_even']['conservative_A64']['uncertified_as_intervals']),
                            'T3 row name "(4 intervals)"'),
               S('3', 'int', lambda X: len(X.t['break_even']['committed_W145']['uncertified_as_intervals']),
                 'T3 row name "(3 intervals)"')))
    # ---- fig:res_flex_ladder / tab:res_flex_ladder -------------------------------------------------------------------
    UNIT = 'CHECK:headline_V_minus_I_settled'
    L.append(D(168, None, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]),
               P('2025', 'year', UNIT), NOT('0'), W3('3', 'N28', 3.0), W3('2', 'N29', 2.0),
               LANG('one', '"where one evaluation is uncertified"'),
               S('1.5', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier].value')),
               S('1.75', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value')),
               S('2', 'raw', ('rec', 'FIG2_DATA', 'values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value')),
               S('+14.2', 'k', _t10('H:m1.75:value_minus_I', 'd_gross')),
               S('1.32', 'x', _t10('H:m1.75:value_minus_I', 'gross_multiple')),
               S('+46.3', 'k', _cl('H:m2:value_minus_I', 'd_gross')), S('3.77', 'x', _cl('H:m2:value_minus_I', 'gross_multiple')),
               S('64.4', 'k', _cl(UNIT, 'd_gross')), S('4.93', 'x', _cl(UNIT, 'gross_multiple'))))
    L.append(D(174, None, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]),
               P('2025', 'year', UNIT)))
    for ln, m, key, src in ((181, '1.00', 'CHECK:headline_V_minus_I_settled.other_cell.ref:bd504ecf.flex_price_multiplier', 'cl'),
                            (182, '1.50', 'H:m1.5:value_minus_I.other_cell.h_f9eae48f.flex_price_multiplier', 'cl'),
                            (183, '1.75', 'H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier', 't10'),
                            (184, '2.00', 'H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier', 'cl')):
        cid = {181: UNIT, 182: 'H:m1.5:value_minus_I', 183: 'H:m1.75:value_minus_I', 184: 'H:m2:value_minus_I'}[ln]
        f = _t10 if src == 't10' else _cl
        w = {181: ('64.4', '13.1', '4.93'), 182: ('14.5', '18.8', '0.77'), 183: ('14.2', '10.8', '1.32'),
             184: ('46.3', '12.3', '3.77')}[ln]
        L.append(D(ln, None, S(m, 'raw', ('rec', 'FIG2_DATA', f'values[{key}].value'), 'fig2 data: the multiplier'),
                   S(w[0], 'k', f(cid, 'd_gross'), signed=True), S(w[1], 'k', f(cid, 'gross_threshold_or_bar')),
                   S(w[2], 'x', f(cid, 'gross_multiple'))))
    # ---- tab:res_energy_ladder_m2 -------------------------------------------------------------------------------------
    L.append(D(192, None, P('7', 'node', 'J:e5_value_minus_I', [7]), P('4', 'dur', 'J:e5_value_minus_I', [7]),
               P('2025', 'year', 'J:e5_value_minus_I'),
               S('15.9', 'k', _cl('J:e4_value_minus_I', 'uncertified_form_beside_A64_ruling4.bar')),
               S('11.12', 'x', _cl('J:e4_value_minus_I', 'uncertified_form_beside_A64_ruling4.multiple_Q')),
               S('27.7', 'k', _cl('J:e3_to_e4', 'uncertified_form_beside_A64_ruling4.bar')),
               S('1.55', 'x', _cl('J:e3_to_e4', 'uncertified_form_beside_A64_ruling4.multiple_Q')),
               S('26.5', 'k', _cl('J:e4_to_e5', 'uncertified_form_beside_A64_ruling4.bar')),
               S('1.95', 'x', _cl('J:e4_to_e5', 'uncertified_form_beside_A64_ruling4.multiple_Q'))))
    EL = {199: ('H:m2:value_minus_I', None, ('1', '46.3', '12.3', '3.77')),
          200: ('I:m2:e2_value_minus_I', 'I:m2:second_MWh', ('2', '92.8', '12.8', '7.26', '46.5', '12.8', '3.64')),
          201: ('J:e3_value_minus_I', 'J:e2_to_e3', ('3', '133.8', '27.7', '4.83', '41.0', '27.7', '1.48')),
          202: ('J:e4_value_minus_I', 'J:e3_to_e4', ('4', '176.6', '9.7', '18.14', '42.8', '27.7', '1.55')),
          203: ('J:e5_value_minus_I', 'J:e4_to_e5', ('5', '228.4', '26.5', '8.60', '51.8', '26.5', '1.95'))}
    for ln, (c1, c2, w) in EL.items():
        sp = [P(w[0], 'E', c1, [7]), S(w[1], 'k', _cl(c1, 'd_gross'), signed=True), S(w[2], 'k', _cl(c1, 'gross_threshold_or_bar')),
              S(w[3], 'x', _cl(c1, 'gross_multiple'))]
        if c2:
            sp += [S(w[4], 'k', _cl(c2, 'd_gross'), signed=True), S(w[5], 'k', _cl(c2, 'gross_threshold_or_bar')),
                   S(w[6], 'x', _cl(c2, 'gross_multiple'))]
        L.append(D(ln, None, *sp))
    L.append(D(207, None, P('1', 'E', 'H:m2:value_minus_I', [7]),
               S('2', 'raw', ('rec', 'FIG2_DATA', 'values[H:m2:value_minus_I.other_cell.h_74eda68d.flex_price_multiplier].value'))))
    L.append(D(208, None, P('7', 'node', 'J:e5_value_minus_I', [7]), P('4', 'dur', 'J:e5_value_minus_I', [7]),
               P('2', 'E', 'I:m2:e2_value_minus_I', [7]), P('5', 'E', 'J:e5_value_minus_I', [7])))
    # ---- 4.3 author slot -------------------------------------------------------------------------------------------------
    L.append(D(216, None, INSTR('Two')))
    L.append(D(218, None, XREF('3', 'accepted sentence 3'), S('-4.1', 'k', _cl('E:n7_4h_e1_no_ageing:value_minus_I', 'd_gross'))))
    L.append(D(219, None, S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh'), 'SRP1_ESS_Params.json minimum_soh'),
               S('31.6', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'max'), 'T8 aged arms: smallest loss (C2)'),
               S('73.7', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'min'), 'T8 aged arms: largest loss (C3_unit)')))
    L.append(D(220, None, S('-64.4', 'k', 'ageing.rows[C2_calfade].value_minus_I'),
               S('4.93', 'x', _cl('E:n7_4h_e1_C2_calfade:value_minus_I', 'gross_multiple')),
               S('-31.6', 'k', 'ageing.rows[C2].value_minus_I'), S('2.50', 'x', _cl('E:n7_4h_e1_C2:value_minus_I', 'gross_multiple')),
               S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit: R = exp(-N D / k), N D = 8,000'),
               S('-73.7', 'k', 'ageing.rows[C3_unit].value_minus_I'), S('5.74', 'x', _cl('E:C3_unit_value_minus_I', 'gross_multiple'))))
    L.append(D(221, None, S('50', 'pct0', lambda X: _retention(X, 'C3_midblock'), 'T8 k of C3_midblock: R = exp(-N D / k)'),
               S('-65.9', 'k', 'ageing.rows[C3_midblock].value_minus_I'),
               S('5.12', 'x', _cl('E:n7_4h_e1_C3_midblock:value_minus_I', 'gross_multiple')),
               S('8,000', 'int', lambda X: round(_arm(X, 'C4')['k'] * -math.log(0.70)),
                 'T8 k of C4 x (-ln 0.70) = N D (main.tex tab:ageing_calibrations: (8,000, 1.00, 0.70))'),
               S('-45.2', 'k', 'ageing.rows[C4].value_minus_I'), S('3.58', 'x', _cl('E:n7_4h_e1_C4:value_minus_I', 'gross_multiple')),
               S('-4.1', 'k', 'ageing.rows[no_ageing].value_minus_I'),
               S('0.32', 'x', _cl('E:n7_4h_e1_no_ageing:value_minus_I', 'gross_multiple'))))
    L.append(D(222, None, S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050: minimum_soh 0.50 (W163 N41)'),
               S('-59.5', 'k', _t10('E:soh050:value_minus_I', 'd_gross')),
               S('4.71', 'x', _t10('E:soh050:value_minus_I', 'gross_multiple'))))
    L.append(D(223, None, S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit'),
               S('+42.1', 'k', _cl('E:n7_4h_e1_C2:vs_C3', 'd_gross')), S('3.28', 'x', _cl('E:n7_4h_e1_C2:vs_C3', 'gross_multiple')),
               S('+28.5', 'k', _cl('E:n7_4h_e1_C4:vs_C3', 'd_gross'))))
    L.append(D(224, None, S('2.22', 'x', _cl('E:n7_4h_e1_C4:vs_C3', 'gross_multiple')),
               S('+69.6', 'k', _cl('E:n7_4h_e1_no_ageing:vs_C3', 'd_gross')),
               S('5.41', 'x', _cl('E:n7_4h_e1_no_ageing:vs_C3', 'gross_multiple')),
               S('+9.3', 'k', _cl('E:n7_4h_e1_C2_calfade:vs_C3', 'd_gross')),
               S('0.71', 'x', _cl('E:n7_4h_e1_C2_calfade:vs_C3', 'gross_multiple'))))
    L.append(D(225, None, S('+7.9', 'k', _cl('E:n7_4h_e1_C3_midblock:vs_C3', 'd_gross')),
               S('0.61', 'x', _cl('E:n7_4h_e1_C3_midblock:vs_C3', 'gross_multiple'))))
    L.append(D(226, None, S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit'),
               LANG('two', '"the two are not resolved"')))
    L.append(D(227, None, S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')),
               S('2035', 'int', lambda X: _same(X, ('C2_calfade', 'C3_unit', 'C3_midblock'), 'floor_year_070'),
                 'T8 floor binds (0.70) of the baseline, C3_unit and C3_midblock (all 2035; the other three never)'),
               S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit')))
    L.append(D(228, None, S('0.786', 'r3', lambda X: min(r['AE'] for r in X.t['ageing']['rows']), 'T8 AE min'),
               S('1.000', 'r3', lambda X: max(r['AE'] for r in X.t['ageing']['rows']), 'T8 AE max')))
    L.append(D(229, None, S('0.76', 'efc', lambda X: min(r['EFC'] for r in X.t['ageing']['rows']), 'T8 EFC/day min'),
               S('1.29', 'efc', lambda X: max(r['EFC'] for r in X.t['ageing']['rows']), 'T8 EFC/day max')))
    L.append(D(230, None, XREF('2', 'accepted sentence 2'), S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')),
               S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh')))
    L.append(D(231, None, W3('+0.18', 'N70', 0.17541226610870808, 'EFC/day at 0.50 - at 0.70, 2025'),
               W3('+0.20', 'N71', 0.19994545338686776, '2030'), W3('+0.27', 'N72', 0.27253194078611853, '2035'),
               S('+4.9', 'k', _t10('E:soh050:delta_value_vs_070', 'd_gross')),
               S('59.5', 'k', _t10('E:soh050:value_minus_I', 'd_gross'))))
    L.append(D(233, None, SINGLE('single'), S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh'),
               S('1.19', 'efc', 'a64.scored.B.efc_per_day.2025'), S('1.10', 'efc', 'a64.scored.B.efc_per_day.2030'),
               S('0.91', 'efc', 'a64.scored.B.efc_per_day.2035'),
               *[S(y, 'int', (lambda yy: (lambda X: int(yy) if yy in X.t['a64']['scored']['B']['efc_per_day'] else None))(y),
                   'a64.scored.B.efc_per_day: the block-year keys') for y in ('2025', '2030', '2035')]))
    L.append(D(234, None, S('2035', 'int', lambda X: 2035 if 'soh_end_2035' in X.t['a64']['scored']['B'] else None,
                            'a64.scored.B.soh_end_2035 (the key names the year)'), S('0.677', 'r3', 'a64.scored.B.soh_end_2035')))
    L.append(D(235, None, S('1.04', 'efc', 'ageing.eps_AE_band_over_resolvable_070[0]'),
               S('1.85', 'efc', 'ageing.eps_AE_band_over_resolvable_070[1]'),
               S('three', 'int', lambda X: sum(1 for r in X.t['ageing']['rows'] if r['eps_AE_resolvable_070']),
                 'T8 resolvable arms at 0.70'), S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh'))))
    # ---- fig:res_ageing / tab:res_ageing ---------------------------------------------------------------------------------
    L.append(D(243, None, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('2025', 'year', UNIT),
               W3('3', 'N28', 3.0), W3('2', 'N29', 2.0), S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh')),
               S('-4.1', 'k', 'ageing.rows[no_ageing].value_minus_I'),
               S('31.6', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'max'), 'T8 aged arms: smallest loss'),
               S('73.7', 'k', lambda X: _min_max_aged(X, 'value_minus_I', 'min'), 'T8 aged arms: largest loss')))
    L.append(D(249, None, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('2025', 'year', UNIT),
               NOT('0'), S('318.0', 'k', 'ageing.rows[C2_calfade].I'), W3('3', 'N28', 3.0), W3('2', 'N29', 2.0)))
    AG = {256: ('C2_calfade', 'E:n7_4h_e1_C2_calfade:value_minus_I', None, ('253.5', '64.4', '13.1', '4.93', '2035', '0.793', '0.86')),
          257: ('C2', 'E:n7_4h_e1_C2:value_minus_I', None, ('286.3', '31.6', '12.6', '2.50', None, '0.888', '1.19')),
          258: ('C3_unit', 'E:C3_unit_value_minus_I', '50', ('244.2', '73.7', '12.8', '5.74', '2035', '0.786', '0.76')),
          259: ('C3_midblock', 'E:n7_4h_e1_C3_midblock:value_minus_I', '50', ('252.1', '65.9', '12.9', '5.12', '2035', '0.836', '0.76')),
          260: ('C4', 'E:n7_4h_e1_C4:value_minus_I', '8,000', ('272.8', '45.2', '12.6', '3.58', None, '0.834', '1.13')),
          261: ('no_ageing', 'E:n7_4h_e1_no_ageing:value_minus_I', None, ('313.8', '4.1', '12.9', '0.32', None, '1.000', '1.29'))}
    for ln, (arm, claim, lead, w) in AG.items():
        sp = []
        if lead == '50':
            sp.append(S('50', 'pct0', (lambda a: (lambda X: _retention(X, a)))(arm), f'T8 k of {arm}: R = exp(-N D / k)'))
        elif lead == '8,000':
            sp.append(S('8,000', 'int', lambda X: round(_arm(X, 'C4')['k'] * -math.log(0.70)), 'T8 k of C4 x (-ln 0.70)'))
        sp += [S(w[0], 'k', f'ageing.rows[{arm}].value'), S(w[1], 'k', f'ageing.rows[{arm}].value_minus_I', signed=True),
               S(w[2], 'k', _cl(claim, 'gross_threshold_or_bar')), S(w[3], 'x', _cl(claim, 'gross_multiple'))]
        if w[4]:
            sp.append(S(w[4], 'int', f'ageing.rows[{arm}].floor_year_070'))
        sp += [S(w[5], 'r3', f'ageing.rows[{arm}].AE'), S(w[6], 'efc', f'ageing.rows[{arm}].EFC')]
        L.append(D(ln, None, *sp))
    L.append(D(262, None, S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh'),
               S('59.5', 'k', _t10('E:soh050:value_minus_I', 'd_gross'), signed=True),
               S('12.6', 'k', _t10('E:soh050:value_minus_I', 'gross_threshold_or_bar')),
               S('4.71', 'x', _t10('E:soh050:value_minus_I', 'gross_multiple'))))
    L.append(D(266, None, S('-59.5', 'k', _t10('E:soh050:value_minus_I', 'd_gross')),
               S('12.6', 'k', _t10('E:soh050:value_minus_I', 'gross_threshold_or_bar')),
               S('4.71', 'x', _t10('E:soh050:value_minus_I', 'gross_multiple'))))
    L.append(D(267, None, S('1.19', 'efc', 'a64.scored.B.efc_per_day.2025'), S('1.10', 'efc', 'a64.scored.B.efc_per_day.2030'),
               S('0.91', 'efc', 'a64.scored.B.efc_per_day.2035')))
    L.append(D(268, None, S('50', 'pct0', lambda X: _retention(X, 'C3_unit'), 'T8 k of C3_unit'),
               S('50', 'pct0', lambda X: _retention(X, 'C3_midblock'), 'T8 k of C3_midblock'),
               S('8,000', 'int', lambda X: round(_arm(X, 'C4')['k'] * -math.log(0.70)), 'T8 k of C4 x (-ln 0.70)')))
    L.append(D(269, None, S('0.50', 'raw', ('rec', 'W155_SOH050', 'reports.e_soh050.minimum_soh'), 'A64 e_soh050 minimum_soh')))
    # ---- 4.4 author slot -----------------------------------------------------------------------------------------------
    BM = 'benchmark'
    AF = 'benchmark.w160_additions.arms_in_full'
    L.append(D(276, None, INSTR('Two'), INSTR('three')))
    L.append(D(278, None, S('653,873.7', 'k', f'{BM}.coordinated_Q'), S('71.9', 'k', f'{BM}.coordinated_reproducibility_band')))
    L.append(D(279, None, S('744,770.3', 'k', f'{BM}.arms.price_taker.Q_best'),
               S('0.0', 'k', f'{BM}.arms.price_taker.multimodality_band'),
               S('three', 'int', f'{AF}.price_taker.n_starts')))
    L.append(D(280, None, S('895,049.9', 'k', f'{BM}.arms.passive.Q_best'), S('25.8', 'k', f'{BM}.arms.passive.multimodality_band'),
               S('90,896.6', 'k', f'{BM}.benefit'), S('13.9', 'pct', f'{BM}.benefit_relative')))
    L.append(D(281, None, S('1,264', 'x', f'{BM}.multiple'),
               S('241,176.2', 'k', f'{BM}.decomposition.passive_NRF_minus_coordinated_eur')))
    L.append(D(282, None, S('36.9', 'pct', f'{BM}.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative')))
    L.append(D(283, None, S('90,896.6', 'k', f'{BM}.benefit'),
               S('77.2', 'pct', ('rec', 'FIG5_DATA', 'computed[share_TSO].value'), 'W165 fig5 data (W130 totals)')))
    L.append(D(284, None, S('0.81', 'twh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted'),
               S('807.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted')))
    L.append(D(285, None, S('781.7', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.TSO.mwh_day_weighted'),
               S('22.8', 'pct', ('rec', 'FIG5_DATA', 'computed[share_DSO].value'), 'W165 fig5 data (W130 totals)')))
    L.append(D(286, None, S('0.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted'),
               S('201.2', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.DSO.mwh_day_weighted')))
    L.append(D(287, None, XREF('4', 'accepted sentence 4')))
    L.append(D(292, None, S('1', 'int', f'{BM}.sweep.sweep_passive_cold.n_blocks'),
               S('12', 'int', lambda X: len(X.j['SRP1_JSON']['Years']) * len(X.j['SRP1_JSON']['Days']),
                 'SRP1.json: 3 years x 4 days = 12 TN blocks'),
               S('6', 'int', f'{BM}.sweep.sweep_passive_cold.n_hours'),
               S('2035', 'int', lambda X: int(X.fzp(f'{AF}.passive.sweep_unconstrained_cold.failing_blocks[0]').split('|')[2]),
                 'T6 passive sweep failing block TSO|-|2035|Summer')))
    L.append(D(293, None, S('8', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_blocks'),
               S('12', 'int', lambda X: len(X.j['SRP1_JSON']['Years']) * len(X.j['SRP1_JSON']['Days']), 'SRP1.json: 12 blocks'),
               S('25', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_hours')))
    L.append(D(294, None, S('4', 'int', f'{BM}.reverse_flow_interface_hours'),
               S('3,308.3', 'mwh1', f'{BM}.reverse_flow_energy_mwh_block_weighted')))
    L.append(D(295, None, S('-44.0', 'k', f'{AF}.passive.consistency_nrf_by_start.cold.pass_effect_eur'),
               S('-69.8', 'k', f'{AF}.passive.consistency_nrf_by_start.perturbed.pass_effect_eur'),
               S('-69.8', 'k', f'{AF}.passive.consistency_nrf_by_start.warm_from_certified.pass_effect_eur')))
    L.append(D(296, None, S('+0.2', 'k', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.pass_effect_eur',
                            note='+0.2 on all three price-taker starts (234.10 / 234.48 / 233.94 EUR)'),
               S('29', 'int', f'{AF}.passive.consistency_nrf_by_start.perturbed.nrf_violations_n'),
               S('0.00618', 'pu', f'{AF}.passive.consistency_nrf_by_start.perturbed.nrf_violations_max_excess_pu'),
               S('70', 'int', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.nrf_violations_n')))
    L.append(D(297, None, S('0.000321', 'pu', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.nrf_violations_max_excess_pu')))
    L.append(D(298, None, S('18.25', 'submitted', None, 'manuscript_submitted/main.tex (submitted-version figure)'),
               S('92.16', 'submitted', None, 'manuscript_submitted/main.tex (submitted-version figure)')))
    L.append(D(299, None, S('13.9', 'pct', f'{BM}.benefit_relative')))
    L.append(D(300, None, S('18.25', 'submitted', None, 'manuscript_submitted/main.tex (submitted-version figure)')))
    # ---- fig:res_benchmark / tab:res_benchmark ----------------------------------------------------------------------------
    L.append(D(305, None, S('two', 'int', lambda X: len(X.t['benchmark']['arms']), 'T6: two NRF arms'),
               S('three', 'int', f'{AF}.passive.n_starts'), S('90,896.6', 'k', f'{BM}.benefit'),
               S('13.9', 'pct', f'{BM}.benefit_relative'), S('1,264', 'x', f'{BM}.multiple'),
               S('71.9', 'k', f'{BM}.larger_band'), S('90,896.6', 'k', f'{BM}.benefit'),
               S('77.2', 'pct', ('rec', 'FIG5_DATA', 'computed[share_TSO].value')),
               S('0.81', 'twh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted'),
               S('22.8', 'pct', ('rec', 'FIG5_DATA', 'computed[share_DSO].value')),
               S('1', 'int', f'{BM}.sweep.sweep_passive_cold.n_blocks'), S('8', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_blocks'),
               S('12', 'int', lambda X: len(X.j['SRP1_JSON']['Years']) * len(X.j['SRP1_JSON']['Days']), 'SRP1.json: 12 blocks'),
               S('4', 'int', f'{BM}.reverse_flow_interface_hours')))
    L.append(D(311, None, SINGLE('single'), S('three', 'int', f'{AF}.passive.n_starts'),
               S('0.011', 'pct3', lambda X: X.t['benchmark']['coordinated_reproducibility_band'] / X.t['benchmark']['coordinated_Q'],
                 'T6 coordinated band / Q181 (README: 0.011 % of Q181)'),
               S('three', 'int', f'{AF}.price_taker.n_starts'),
               S('12', 'int', lambda X: len(X.j['SRP1_JSON']['Years']) * len(X.j['SRP1_JSON']['Days']), 'SRP1.json: 12 blocks')))
    L.append(D(319, None, S('653,873.7', 'k', f'{BM}.coordinated_Q'), S('71.9', 'k', f'{BM}.coordinated_reproducibility_band')))
    L.append(D(320, None, S('744,770.3', 'k', f'{BM}.arms.price_taker.Q_best'), S('0.0', 'k', f'{BM}.arms.price_taker.multimodality_band'),
               S('90,896.6', 'k', f'{BM}.benefit'), S('13.9', 'pct', f'{BM}.benefit_relative'), S('1,263.75', 'x', f'{BM}.multiple'),
               S('70', 'int', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.nrf_violations_n'),
               S('0.000321', 'pu', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.nrf_violations_max_excess_pu'),
               S('807.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted'),
               S('0.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted'),
               S('8', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_blocks'), S('25', 'int', f'{BM}.sweep.sweep_price_taker_cold.n_hours')))
    L.append(D(321, None, S('895,049.9', 'k', f'{BM}.arms.passive.Q_best'), S('25.8', 'k', f'{BM}.arms.passive.multimodality_band'),
               S('241,176.2', 'k', f'{BM}.decomposition.passive_NRF_minus_coordinated_eur'),
               S('36.9', 'pct', f'{BM}.w160_additions.passive_arm_vs_coordinated_derived.benefit_passive_relative'),
               S('3,353.11', 'x', f'{BM}.w160_additions.passive_arm_vs_coordinated_derived.multiple_of_larger_band'),
               S('29', 'int', f'{AF}.passive.consistency_nrf_by_start.perturbed.nrf_violations_n'),
               S('0.00618', 'pu', f'{AF}.passive.consistency_nrf_by_start.perturbed.nrf_violations_max_excess_pu'),
               S('781.7', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.TSO.mwh_day_weighted'),
               S('201.2', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.DSO.mwh_day_weighted'),
               S('1', 'int', f'{BM}.sweep.sweep_passive_cold.n_blocks'), S('6', 'int', f'{BM}.sweep.sweep_passive_cold.n_hours')))
    L.append(D(324, None, S('3', 'int', f'{AF}.price_taker.n_starts'), S('3', 'int', f'{AF}.passive.n_starts')))
    L.append(D(326, None, S('70', 'int', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.nrf_violations_n'),
               S('0.000321', 'pu', f'{AF}.price_taker.consistency_nrf_by_start.warm_from_certified.nrf_violations_max_excess_pu'),
               S('807.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.TSO.mwh_day_weighted'),
               S('0.6', 'gwh', f'{AF}.price_taker.curtailment_phase_A_by_start.warm_from_certified.by_agent.DSO.mwh_day_weighted'),
               S('29', 'int', f'{AF}.passive.consistency_nrf_by_start.perturbed.nrf_violations_n')))
    L.append(D(327, None, S('0.00618', 'pu', f'{AF}.passive.consistency_nrf_by_start.perturbed.nrf_violations_max_excess_pu'),
               S('781.7', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.TSO.mwh_day_weighted'),
               S('201.2', 'gwh', f'{AF}.passive.curtailment_phase_A_by_start.perturbed.by_agent.DSO.mwh_day_weighted')))
    # ---- 4.5 author slot and tab:res_multiscenario -----------------------------------------------------------------
    L.append(D(335, None, INSTR('One'), S('3', 'int', ('rec', 'CASE_3X3', 'NumMarketScenarios')),
               S('3', 'int', ('rec', 'CASE_3X3', 'TransmissionNetwork.num_operation_scenarios'))))
    L.append(D(336, None, S('five', 'int', lambda X: len(X.j['CASE_3X3']['Years']), '3 x 3 case: Years'),
               S('three', 'int', lambda X: _same_vals(X.j['CASE_3X3']['Years'].values()), '3 x 3 case: block length'),
               *[S(y, 'int', (lambda yy: (lambda X: int(list(X.j['CASE_3X3']['Years'])[yy])))(i), '3 x 3 case: Years')
                 for i, y in enumerate(('2025', '2028', '2031', '2034', '2037'))]))
    L.append(D(338, None, NOT('0'), S('236.7', 'k', 'three_by_three.rows[0].value'),
               S('72', 'int', ('rec', 'W98R2_RESULTS', 'replay_gate.N'), 'W98 r2: the certified 3 x 3 x = 0 cycle (N)')))
    L.append(D(339, None, S('318.0', 'k', 'three_by_three.rows[1].value'), S('-81.3', 'k', 'three_by_three.rows[2].value'),
               S('18.0', 'k', 'three_by_three.rows[3].value'), S('4.52', 'x', 'three_by_three.rows[4].value')))
    L.append(D(340, None, SINGLE('single'), S('253.5', 'k', 'three_by_three.rows[6].value')))
    L.append(D(341, None, S('0.934', 'r3', 'three_by_three.rows[7].value'), S('0.909', 'r3', 'three_by_three.rows[8].value'),
               S('6.2', 'k', 'three_by_three.rows[5].value'), NOT('0')))
    L.append(D(342, None, S('16', 'int', lambda X: max(X.cont_cycles) - X.j['W98R2_RESULTS']['replay_gate']['N'],
                            'W98 r2 continuation: last cycle 88 - exit 72 (CONFIRM C6)'),
               S('16', 'int', lambda X: max(X.cont_cycles) - X.j['W98R2_RESULTS']['replay_gate']['N'], 'W98 r2 (CONFIRM C6)'),
               S('0.9331', 'r4', 'three_by_three.rows[9].value')))
    L.append(D(343, None, DEF('4', 'W170_FORMULA', 'mean of the 4 highest minus mean of the 4 lowest',
                              'the 4 h spread of the W170 formula'),
               LANG('two', 'the two horizons (numerator 3 x 3, denominator SRP1; W170)')))
    L.append(D(344, None, NOT('0'), S('72', 'int', lambda X: sum(1 for c in X.j['W98R2_RESULTS']['replay_gate']['per_cycle']
                                                               if c['equal']), 'W98 r2 replay gate: cycles equal')))
    L.append(D(348, None, S('0.909', 'r3', 'three_by_three.rows[8].value'), S('0.934', 'r3', 'three_by_three.rows[7].value'),
               SINGLE('single')))
    L.append(D(349, None, S('7', 'pct', lambda X: 1 - X.fzp('three_by_three.rows[7].value'), '1 - R(upper) = 6.6 %'),
               S('9', 'pct', lambda X: 1 - X.fzp('three_by_three.rows[8].value'), '1 - R(lower) = 9.1 %')))
    L.append(D(353, None, S('3', 'int', ('rec', 'CASE_3X3', 'NumMarketScenarios')),
               S('3', 'int', ('rec', 'CASE_3X3', 'TransmissionNetwork.num_operation_scenarios')),
               P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('2025', 'year', UNIT),
               SINGLE('single'), NOT('0')))
    L.append(D(360, None, NOT('0'), S('72', 'int', ('rec', 'W98R2_RESULTS', 'replay_gate.N')),
               S('236.7', 'k', 'three_by_three.rows[0].value')))
    L.append(D(361, None, S('318.0', 'k', 'three_by_three.rows[1].value')))
    L.append(D(362, None, S('81.3', 'k', 'three_by_three.rows[2].value', signed=True)))
    L.append(D(363, None, S('18.0', 'k', 'three_by_three.rows[3].value'), S('4.52', 'x', 'three_by_three.rows[4].value')))
    L.append(D(364, None, NOT('0'), S('6.2', 'k', 'three_by_three.rows[5].value')))
    L.append(D(365, None, SINGLE('single'), S('253.5', 'k', 'three_by_three.rows[6].value')))
    L.append(D(366, None, S('0.934', 'r3', 'three_by_three.rows[7].value'), SINGLE('single')))
    L.append(D(367, None, S('0.909', 'r3', 'three_by_three.rows[8].value'), SINGLE('single')))
    L.append(D(368, None, S('0.9331', 'r4', 'three_by_three.rows[9].value')))
    L.append(D(369, None, NOT('0'), S('72', 'int', ('rec', 'W98R2_RESULTS', 'replay_gate.N')),
               S('72', 'int', lambda X: sum(1 for c in X.j['W98R2_RESULTS']['replay_gate']['per_cycle'] if c['equal']),
                 'W98 r2 replay gate: cycles equal (bitwise)')))
    # ---- 4.6 author slot, tab:res_year_ladder, tab:res_discount ----------------------------------------------------------
    YL = 'year_ladder'
    L.append(D(379, None, INSTR('Two')))
    L.append(D(381, None, P('2030', 'year', 'yl_y2030'), S('68.5', 'k', f'{YL}.per_year.2030.M_gross_w153_form'),
               S('56.7', 'k', f'{YL}.per_year.2030.M_net_w153_form'), S('11.8', 'k', f'{YL}.per_year.2030.salvage_last'),
               note='"Deferring the reference unit changes little gross and nothing net" does not fit the figures that '
                    'follow: 2035 costs 111.0 kEUR gross against 64.4 in 2025 (+46.6) and the 2035 - 2030 gross difference '
                    'is determinate (42.5 kEUR, 3.46 x); net, 56.7 / 54.5 against 64.4 is -7.7 / -9.9 kEUR (no frozen claim '
                    'resolves 2025 against 2030 / 2035)'))
    L.append(D(382, None, P('2035', 'year', 'yl_y2035'), S('111.0', 'k', f'{YL}.per_year.2035.M_gross_w153_form'),
               S('54.5', 'k', f'{YL}.per_year.2035.M_net_w153_form'), S('56.5', 'k', f'{YL}.per_year.2035.salvage_last'),
               P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030'), S('42.5', 'k', f'{YL}.D_gross')))
    L.append(D(383, None, S('3.46', 'x', f'{YL}.gross_v6.multiple'), S('12.3', 'k', f'{YL}.gross_v6.threshold'),
               S('-2.3', 'k', f'{YL}.D_net'), S('0.19', 'x', f'{YL}.net_v6.multiple')))
    L.append(D(385, None, P('7', 'node', 'G:n7_2h_e1_y2030:gross', [7]), P('0.5', 'S', 'G:n7_2h_e1_y2030:gross', [7]),
               P('1', 'E', 'G:n7_2h_e1_y2030:gross', [7]), P('2030', 'year', 'G:n7_2h_e1_y2030:gross'),
               S('99.0', 'k', _cl('G:n7_2h_e1_y2030:gross', 'd_gross')), S('88.3', 'k', _cl('G:n7_2h_e1_y2030:gross', 'd_net'))))
    L.append(D(386, None, P('2035', 'year', 'G:n7_2h_e1_y2035:gross'), S('139.7', 'k', _cl('G:n7_2h_e1_y2035:gross', 'd_gross')),
               S('85.1', 'k', _cl('G:n7_2h_e1_y2035:gross', 'd_net')), P('0.5', 'S', 'G:n7_4h_e2_y2030:gross', [7]),
               P('2', 'E', 'G:n7_4h_e2_y2030:gross', [7]), P('2030', 'year', 'G:n7_4h_e2_y2030:gross'),
               S('137.4', 'k', _cl('G:n7_4h_e2_y2030:gross', 'd_gross')), S('113.1', 'k', _cl('G:n7_4h_e2_y2030:gross', 'd_net')),
               P('2035', 'year', 'G:n7_4h_e2_y2035:gross'), S('223.6', 'k', _cl('G:n7_4h_e2_y2035:gross', 'd_gross')),
               S('109.1', 'k', _cl('G:n7_4h_e2_y2035:gross', 'd_net'))))
    L.append(D(387, None, S('0', 'pct0', 'discount[0.0].rate'), S('8', 'pct0', 'discount[0.08].rate'),
               S('-40.8', 'k', 'discount[0.0].value_minus_I')))
    L.append(D(388, None, S('0', 'pct0', 'discount[0.0].rate'), S('2.57', 'x', 'discount[0.0].multiple'),
               S('-114.6', 'k', 'discount[0.08].value_minus_I'), S('8', 'pct0', 'discount[0.08].rate'),
               S('8.78', 'x', 'discount[0.08].multiple'),
               DEF('one', 'T7_TEX', 'one discount factor per representative year', 'T7 caption')))
    L.append(D(389, None, DEF('five', 'T7_TEX', 'applied to the five years of its block', 'T7 caption'),
               DEF('2025', 'T7_TEX', 'I paid in 2025', 'T7 caption')))
    L.append(D(390, None, S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims'))))
    L.append(D(391, None, S('0', 'int', ('rec', 'W153_SALV', 'result.totals.n_sign_differs')),
               S('60', 'int', ('rec', 'W153_SALV', 'result.totals.n_claims')),
               S('one', 'int', ('rec', 'W153_SALV', 'result.totals.n_verdict_differs'))))
    L.append(D(392, None, S('0.95', 'x', _cl('L:y2030__n5_p0.25_e0.5__n7_p1_e3', 'gross_multiple')),
               S('1.22', 'x', _cl('L:y2030__n5_p0.25_e0.5__n7_p1_e3', 'net_multiple'))))
    L.append(D(396, None, P('7', 'node', 'yl_y2030', [7]), P('0.25', 'S', 'yl_y2030', [7]), P('1', 'E', 'yl_y2030', [7]),
               NOT('0'), P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030')))
    L.append(D(403, None, P('2025', 'year', UNIT), S('64.4', 'k', _cl(UNIT, 'd_gross'), note='M = I + Q(x) - Q(0) = -(value - I)'),
               S('0.0', 'k', 'cells[ref:bd504ecf].salvage', note='the 2025 cohort: T^Rem = 0 (CONFIRM C8)'),
               S('64.4', 'k', _cl(UNIT, 'd_net'), note='M net = -(value - I) net')))
    L.append(D(404, None, P('2030', 'year', 'yl_y2030'), S('68.5', 'k', f'{YL}.per_year.2030.M_gross_w153_form'),
               S('11.8', 'k', f'{YL}.per_year.2030.salvage_last'), S('56.7', 'k', f'{YL}.per_year.2030.M_net_w153_form')))
    L.append(D(405, None, P('2035', 'year', 'yl_y2035'), S('111.0', 'k', f'{YL}.per_year.2035.M_gross_w153_form'),
               S('56.5', 'k', f'{YL}.per_year.2035.salvage_last'), S('54.5', 'k', f'{YL}.per_year.2035.M_net_w153_form')))
    L.append(D(406, None, P('2035', 'year', 'yl_y2035'), P('2030', 'year', 'yl_y2030'), S('42.5', 'k', f'{YL}.D_gross', signed=True),
               S('2.3', 'k', f'{YL}.D_net', signed=True), S('12.3', 'k', f'{YL}.gross_v6.threshold'),
               S('3.46', 'x', f'{YL}.gross_v6.multiple'), S('0.19', 'x', f'{YL}.net_v6.multiple')))
    L.append(D(409, None, P('2030', 'year', 'yl_y2030'), P('2035', 'year', 'yl_y2035'), P('2035', 'year', 'yl_y2035'),
               P('2030', 'year', 'yl_y2030'), P('2025', 'year', UNIT), S('64.4', 'k', _cl(UNIT, 'd_gross'))))
    L.append(D(410, None, P('2025', 'year', UNIT), S('0', 'raw', lambda X: X.coh2025['T_rem'], 'code: T_rem(2025) (CONFIRM C8)'),
               S('15', 'int', ('rec', 'ESS_PARAMS', 'ageing.calendar_life_years')), P('2025', 'year', UNIT)))
    L.append(D(411, None, P('2025', 'year', UNIT), S('0.0', 'k', 'cells[ref:bd504ecf].salvage')))
    L.append(D(416, None, P('7', 'node', UNIT, [7]), P('0.25', 'S', UNIT, [7]), P('1', 'E', UNIT, [7]), P('2025', 'year', UNIT),
               DEF('one', 'T7_TEX', 'one discount factor per representative year', 'T7 caption'),
               DEF('five', 'T7_TEX', 'applied to the five years of its block', 'T7 caption'),
               DEF('2025', 'T7_TEX', 'I paid in 2025', 'T7 caption')))
    for ln, rate, w in ((423, '0.0', ('0', '277.1', '318.0', '40.8', '15.9', '2.57')),
                        (424, '0.02', ('2', '253.5', '318.0', '64.4', '13.1', '4.93')),
                        (425, '0.05', ('5', '225.2', '318.0', '92.7', '13.1', '7.10')),
                        (426, '0.08', ('8', '203.4', '318.0', '114.6', '13.1', '8.78'))):
        L.append(D(ln, None, S(w[0], 'pct0', f'discount[{rate}].rate'), S(w[1], 'k', f'discount[{rate}].V'),
                   S(w[2], 'k', f'discount[{rate}].I'), S(w[3], 'k', f'discount[{rate}].value_minus_I', signed=True),
                   S(w[4], 'k', f'discount[{rate}].threshold_a61_conservative'), S(w[5], 'x', f'discount[{rate}].multiple')))
    # ---- 4.7 (expert's prose) -------------------------------------------------------------------------------------------
    WC = ('rec', 'W153_CERT')
    L.append(D(437, None, S('42', 'int', WC + ('result.totals.n_cells',)), SINGLE('single'),
               S('32', 'int', WC + ('result.totals.n_certified',)),
               S('24', 'int', WC + ('result.totals.branch_counts_certified.oscillatory',)),
               S('8', 'int', WC + ('result.totals.branch_counts_certified.monotone',)),
               S('10', 'int', WC + ('result.totals.n_uncertified',)),
               S('5', 'int', WC + ('result.totals.uncertified_by_cause.gap clause.n',)),
               S('2', 'int', WC + ('result.totals.uncertified_by_cause.lapse reset.n',)),
               S('3', 'int', WC + ('result.totals.uncertified_by_cause.growth test.n',)),
               S('174', 'raw', WC + ('result.totals.k_star.median',)),
               S('65', 'raw', WC + ('result.totals.cycles_after_k0_k_star_minus_N.median',)),
               S('21', 'int', WC + ('result.totals.cycles_after_k0_k_star_minus_N.min',)),
               S('89', 'int', WC + ('result.totals.cycles_after_k0_k_star_minus_N.max',)),
               S('0.86', 'efc', WC + ('result.totals.range_over_tau_at_k_star.median',)),
               S('four', 'int', lambda X: sum(1 for c in X.t['cells_appended_w160'].values() if c.get('row_origin') == 'T10'
                                              and c['status'] == 'certified'), 'frozen cells_appended_w160 (T10 rows), certified'),
               S('Ten', 'int', 'at_or_above_0_95_tau.n_counted'),
               W3('46', 'N23', 46, 'certificates the tables use (49 certified cells, 1 superseded, 2 bitwise twins)'),
               W3('5', 'N24', 5, '1 - 0.95 (the >= 0.95 tau flag)'),
               W3('0.9', 'N25', 0.9, 'largest movement past a settling-rule certificate: 0.882 tau'),
               W3('15', 'N4', 15, 'x0: 14,970.62 EUR after the residual-rule certificate N = 132'),
               W3('21', 'N5', 21, 'unit: 20,806.32 EUR after N = 112')))
    F4 = ('rec', 'FIG4_DATA')
    L.append(D(441, None, LANG('two', 'the two outcomes (certified / refused)'),
               S('181', 'int', lambda X: X.fig4['a.k_star'], 'W165 fig4 data a.k_star'),
               S('58', 'int', lambda X: X.fig4['a.k_star'] - X.fig4['a.k0'], 'fig4 data: k* - k0'),
               S('123', 'int', lambda X: X.fig4['a.k0'], 'fig4 data a.k0'),
               S('0.93', 'efc', 'cells[ref:7aa017f0].range_over_tau'),
               S('132', 'int', lambda X: X.fig4['a.N'], 'fig4 data a.N'),
               S('15.0', 'k', lambda X: X.fig4_checks['a.N_and_slack_equal_summary']['detail'][1],
                 'fig4 data check a.N_and_slack_equal_summary: Q(N) - Q(k*)'),
               S('1.5', 'raw', ('rec', 'FIG4_DATA', 'values[b.h_f9eae48f.flex_price_multiplier].value')),
               W3('2', 'N12', 2.0, 'GAP_BOUND = tau / 2'),
               S('2.5', 'k', 'cells[h_f9eae48f].gap'), S('6.3', 'k', 'cells[h_f9eae48f].slack')))
    L.append(D(447, None, LANG('two', 'two recorded evaluations (panels a and b-c)'), SINGLE('single'),
               S('123', 'int', lambda X: X.fig4['a.k0'], 'fig4 data a.k0'),
               S('181', 'int', lambda X: X.fig4['a.k_star'], 'fig4 data a.k_star'),
               S('32', 'int', lambda X: X.fig4['a.window'][1] - X.fig4['a.window'][0] + 1, 'fig4 data a.window [150, 181]'),
               S('0.93', 'efc', 'cells[ref:7aa017f0].range_over_tau'),
               S('132', 'int', lambda X: X.fig4['a.N'], 'fig4 data a.N'),
               S('15.0', 'k', lambda X: X.fig4_checks['a.N_and_slack_equal_summary']['detail'][1], 'fig4 data: Q(N) - Q(k*)'),
               P('7', 'node', 'H:m1.5:value_minus_I', [7]), P('0.25', 'S', 'H:m1.5:value_minus_I', [7]),
               P('1', 'E', 'H:m1.5:value_minus_I', [7]), P('2025', 'year', 'H:m1.5:value_minus_I'),
               S('1.5', 'raw', ('rec', 'FIG4_DATA', 'values[b.h_f9eae48f.flex_price_multiplier].value')),
               S('85', 'int', lambda X: X.fig4['b.k0'], 'fig4 data b.k0'),
               S('194', 'int', lambda X: X.fig4['b.cap'], 'fig4 data b.cap'),
               W3('2', 'N12', 2.0, 'GAP_BOUND = tau / 2'),
               S('59', 'int', lambda X: X.fig4_checks['b.refusal_count_equals_T2_contributing']['detail'][1],
                 'fig4 data: gap refusals (T2 contributing "59 refusals")')))
    L.append(D(452, None, S('three', 'int', lambda X: len(X.w173['item1']), 'W173 item1: s47, s51, s53'),
               S('41', 'int', lambda X: sum(X.w173['item1'][k]['n_new_evaluations_total'] for k in X.w173['item1']), 'W173 sum'),
               S('14', 'int', ('rec', 'W173', 'item1.s47.n_new_evaluations_total')),
               S('20', 'int', ('rec', 'W173', 'item1.s51.n_new_evaluations_total')),
               S('7', 'int', ('rec', 'W173', 'item1.s53.n_new_evaluations_total')),
               S('20', 'int', ('rec', 'W173', 'item1.s47.max_new_evaluations')),
               S('20', 'int', ('rec', 'W173', 'item1.s51.max_new_evaluations')),
               S('60', 'int', ('rec', 'W173', 'item1.s53.max_new_evaluations')),
               S('14', 'int', ('rec', 'W173', 'item1.s47.n_new_evaluations_total')),
               S('20', 'int', ('rec', 'W173', 'item1.s47.max_new_evaluations')),
               S('20', 'int', ('rec', 'W173', 'item1.s51.n_new_evaluations_total')),
               S('20', 'int', ('rec', 'W173', 'item1.s51.max_new_evaluations')),
               S('7', 'int', ('rec', 'W173', 'item1.s53.n_new_evaluations_total')),
               S('60', 'int', ('rec', 'W173', 'item1.s53.max_new_evaluations')),
               S('41', 'int', lambda X: sum(X.w173['item1'][k]['n_new_evaluations_total'] for k in X.w173['item1']), 'W173 sum')))
    L.append(D(453, None, S('42', 'int', WC + ('result.totals.n_cells',)),
               S('220,876', 'raw', WC + ('result.totals.wall_s_total',)),
               S('8,570', 'int', WC + ('result.totals.cycles_total',)),
               S('25.6', 'raw', WC + ('result.totals.wall_s_per_cycle.median',)),
               W3('32', 'N57', 32, '34,359,738,368 bytes = 32 GiB'), W3('27.0', 'N55', 27.0, 'macOS ProductVersion'),
               W3('one', 'N44', 1, 'one evaluation at a time'), W3('single', 'N45', 1, 'single-threaded (thread caps)'),
               W3('one', 'N45', 1, 'one thread (OMP / MKL / OpenBLAS / vecLib / NumExpr)')))
    L.append(D(457, None, W3('72', 'N47', 72, '3 x 3 x = 0 replay 72/72'), W3('three', 'N49', 3, 'three SRP1 reference continuations'),
               SINGLE('single'), W3('Two', 'N58', 2, 'two distribution blocks'), W3('7', 'N58', 7, 'node 7 (DSO|7|2025|Winter)'),
               W3('2025', 'N58', 2025, 'DSO|7|2025|Winter'), W3('5', 'N58', 5, 'node 5 (DSO|5|2035|Winter)'),
               W3('2035', 'N58', 2035, 'DSO|5|2035|Winter'), W3('ten', 'N59', 10, 'CLEAN_FACTOR 10'),
               LANG('Two', 'two offsets (the tight tail and the RES slack), enumerated in the sentence'),
               W3('three', 'N62', 3, 'the three SRP1 references'), SINGLE('single'),
               S('1.0', 'raw', lambda X: min(abs(v) for v in X.W163['N63']['value'].values()) * 1e6,
                 'W163 N63 (W86 per_cell dQ_relative x 1e6): smallest 1.036'),
               S('1.2', 'raw', lambda X: max(abs(v) for v in X.W163['N63']['value'].values()) * 1e6, 'W163 N63: largest 1.200'),
               S('10^{-6}', 'sci', lambda X: 1e-6, 'the unit of the 1.0-1.2 range (W163 N63)'),
               W3('10^{-6}', 'N61', 1e-6, 'tail compl_inf_tol'), W3('10^{-4}', 'N60', 1e-4, 'DSO compl_inf_tol before the tail'),
               W3('5', 'N74', 5, 'TSO 5 x 10^-4'), W3('10^{-4}', 'N74', 1e-4, 'TSO 5 x 10^-4'),
               W3('10^{-5}', 'N33', 1e-5, 'RES availability slack'), W3('7.8', 'N64', 7.8, '7,848.08 EUR'),
               W3('1.2', 'N65', 1.2, '1.2002e-5 of Q'), W3('10^{-5}', 'N65', 1e-5, '1.2 x 10^-5')))
    # ---- 4.8 (expert's prose) -------------------------------------------------------------------------------------------
    DZ = ('j_5f3cccb4', 'l_45aa25a6', 'l_7c455554', 'l_b2251bc5', 'l_0ee93aca')
    L.append(D(465, None, U('0', 'notation', 'lambda in [0, c^flex] (interval endpoint)'), LANG('one', 'one fingerprint'),
               S('-8.8', 'k', lambda X: max(_dz(X, c)['t_sum_at_cap'] for c in DZ), 'T9 dead-zone cells: t_sum at cap, largest'),
               S('-9.4', 'k', lambda X: min(_dz(X, c)['t_sum_at_cap'] for c in DZ), 'T9: smallest'),
               S('55', 'pctr', lambda X: [_dz(X, c)['share_by_node']['5'] for c in DZ], 'T9 node-5 share (five cells)'),
               S('5', 'int', lambda X: 5 if all('5' in _dz(X, c)['share_by_node'] for c in DZ) else None, 'T9 share_by_node key 5'),
               S('31', 'pctr', lambda X: [_dz(X, c)['share_by_node']['9'] for c in DZ], 'T9 node-9 share'),
               S('9', 'int', lambda X: 9 if all('9' in _dz(X, c)['share_by_node'] for c in DZ) else None, 'T9 share_by_node key 9'),
               S('14', 'pctr', lambda X: [_dz(X, c)['share_by_node']['7'] for c in DZ], 'T9 node-7 share'),
               S('7', 'int', lambda X: 7 if all('7' in _dz(X, c)['share_by_node'] for c in DZ) else None, 'T9 share_by_node key 7'),
               S('0.69', 'near', lambda X: [_dz(X, c)['pf_primal_last'] for c in DZ], 'T9 pf_primal last (0.682-0.700)'),
               P('5', 'E', 'B:n7_4h_e5', [7]), S('+1.1', 'k', 'cells[d_36686489].t_sum_end',
                                                 desc='T9 d_36686489 t_sum (+1,082.23 EUR)')))
    L.append(D(469, None, S('21', 'int', WC + ('result.totals.non_clean_cycles_after_N_total',)),
               S('10', 'int', lambda X: len(X.w153c['cells_with_non_clean_after_N']), 'W153: cells with non-clean cycles after N'),
               W3('2035', 'N32', 2035, 'recurring block TSO|2035|Spring (6 of 21)'), W3('10^{-5}', 'N33', 1e-5, 'RES slack'),
               W3('62', 'N35', 62, '61.91 MWh-equivalent'), W3('3', 'N36', 3, '3.36 tau estimate'),
               W3('44', 'N37', 44, 'L_MONO = 44 (spec v41)')))
    L.append(D(473, None, S('one', 'int', lambda X: _w173_year_dims(X), 'W173 box_definition: one year coordinate zY'),
               S('three', 'int', lambda X: len(X.j['SRP1_JSON']['Years']), 'SRP1.json Years'),
               S('five', 'int', lambda X: _same_vals(X.j['SRP1_JSON']['Years'].values()), 'SRP1.json block length'),
               S('five', 'int', lambda X: len(X.j['CASE_3X3']['Years']), '3 x 3 case Years'),
               S('three', 'int', lambda X: _same_vals(X.j['CASE_3X3']['Years'].values()), '3 x 3 block length'),
               S('four', 'int', lambda X: len(X.j['SRP1_JSON']['Days']), 'SRP1.json Days'),
               S('one', 'int', lambda X: 1 if isinstance(X.j['SRP1_JSON']['TransmissionNetwork'], dict) else 0,
                 'SRP1.json TransmissionNetwork (one)'),
               S('9', 'int', lambda X: len(X.j['CASE9']['nodes']), 'case9_2025.json nodes'),
               S('three', 'int', lambda X: len(X.j['SRP1_JSON']['DistributionNetworks']), 'SRP1.json DistributionNetworks'),
               S('33', 'int', lambda X: _same_vals([len(X.j[k]['nodes']) for k in ('CASE33_1', 'CASE33_2', 'CASE33_3')]),
                 'case33_*_2025.json nodes'),
               W3('0.70', 'N38', 0.7, 'baseline floor'), W3('1.04', 'L9', 1.04), W3('1.85', 'L10', 1.85),
               W3('0.41', 'L11a', 0.41), W3('0.62', 'L11b', 0.62), W3('three', 'L11c', 3), W3('0.50', 'N39', 0.5)))
    # ---- 4.9 ------------------------------------------------------------------------------------------------------------
    L.append(D(480, None, S('four', 'int', lambda X: X.frame_items, 'the frame\'s enumerate (4.9): \\item count')))
    L.append(D(482, None, S('four', 'int', lambda X: X.frame_items, 'the frame\'s enumerate (4.9): \\item count')))
    L.append(D(484, None, S('0.70', 'raw', ('rec', 'ESS_PARAMS', 'ageing.minimum_soh'))))
    L.append(D(485, None, S('171', 'kmwh', 'break_even.conservative_A64.breakeven_range[0]'),
               S('193', 'kmwh', 'break_even.conservative_A64.breakeven_range[1]'),
               S('254', 'kmwh', 'break_even.e_cost_eur_per_MWh'),
               S('1.75', 'raw', ('rec', 'FIG2_DATA', 'values[H:m1.75:value_minus_I.other_cell.h_unit_m175.flex_price_multiplier].value')),
               P('two', 'nnodes', 'ref:5ca4f86c')))
    L.append(D(486, None, S('13.9', 'pct', f'{BM}.benefit_relative')))
    L.append(D(487, None, S('654', 'meur', f'{BM}.coordinated_Q')))
    L.append(D(494, None, XREF('3', 'round 3 (the revision round)'), S('60', 'int', lambda X: len(X.t['claims']), 'frozen T1 claims')))
    return L


def _single_node(X, fit):
    nodes = {re.match(r'n(\d+)_', p).group(1) for p in X.t['break_even'][fit]['certified'] +
             X.t['break_even'][fit]['uncertified_as_intervals']}
    return int(nodes.pop()) if len(nodes) == 1 else None


def _arm(X, arm):
    return [r for r in X.t['ageing']['rows'] if r['arm'] == arm][0]


def _retention(X, arm):
    return math.exp(-8000.0 / _arm(X, arm)['k'])


def _same(X, arms, f):
    vals = {_arm(X, a)[f] for a in arms}
    others = {r[f] for r in X.t['ageing']['rows'] if r['arm'] not in arms}
    return vals.pop() if len(vals) == 1 and others == {None} else None


def _same_vals(vals):
    s = set(vals)
    return s.pop() if len(s) == 1 else None


def _dz(X, cell):
    return [c for c in X.t['dead_zone']['cells'] if c['cell'] == cell][0]


def _w173_year_dims(X):
    m = re.search(r'\(([^)]*zY[^)]*)\)', X.w173['box_definition'])
    return sum(1 for v in m.group(1).split(',') if v.strip().startswith('zY'))


# ======================================================================================================================
#  4. evaluation of a declared token
# ======================================================================================================================
RULE = {'k': (1e-3, 1), 'x': (1.0, 2), 'pct': (100.0, 1), 'pct2': (100.0, 2), 'pct3': (100.0, 3), 'pct0': (100.0, 0),
        'kmwh': (1e-3, 2), 'e10': (1.0, -1), 'efc': (1.0, 2), 'r3': (1.0, 3), 'r4': (1.0, 4), 'gwh': (1e-3, 1),
        'twh': (1e-6, 4), 'meur': (1e-6, 4), 'mwh1': (1.0, 1), 'raw': (1.0, None)}
RULE_NOTE = {'k': 'kEUR 1 dp', 'x': 'multiple 2 dp', 'pct': '% 1 dp', 'pct2': '% 2 dp (T3 agreement)', 'pct3': '% (README)',
             'pct0': '% (rate) 0 dp', 'kmwh': 'EUR/MWh to the nearest 10 = kEUR/MWh 2 dp', 'e10': 'EUR/MWh nearest 10',
             'efc': '2 dp', 'r3': '3 dp', 'r4': '4 dp (R predicted, as recorded)', 'gwh': 'GWh 1 dp',
             'twh': 'curtailment GWh 1 dp = TWh 4 dp', 'meur': 'kEUR 1 dp = MEUR 4 dp', 'mwh1': 'MWh 1 dp',
             'raw': 'as recorded (the written precision)'}


def _written_value(t, line):
    if t['type'] == 'word':
        return float(W174.wv3(t['text']))
    v = t['value']
    pre = line[:t['col0']].rstrip()
    latex_minus = pre.endswith('$-$')
    if latex_minus and v is not None:
        v = -abs(v)
    return v


def _explicit_sign(t, line):
    return t.get('sign') in ('+', '-') or line[:t['col0']].rstrip().endswith('$-$')


def _source_value(X, spec):
    src = spec['src']
    if callable(src):
        return src(X), spec.get('desc') or 'computed'
    if isinstance(src, tuple) and src[0] == 'rec':
        return X.rec(src[1], src[2]), f"{X.inputs[src[1]]['path']} {src[2]} (sha256 {X.inputs[src[1]]['sha256'][:8]})"
    return X.fzp(src), f'{FZ_REL} tables.{src}'


def evaluate(X, spec, t, line):
    kind = spec['kind']
    w_txt = t['written']
    if kind == 'unchecked':
        return {'status': 'unchecked', 'category': spec['cat'], 'note': spec['why']}
    wv = _written_value(t, line)
    explicit = _explicit_sign(t, line)
    if kind == 'plan':
        cell = spec['cell']
        cell = X.other(cell) if (cell in X.CL or cell in X.A64) else cell
        c = X.cand(cell)
        nz = {int(n): v for n, v in c['nodes'].items() if v[0] or v[1]}
        f, nodes = spec['field'], spec['nodes'] or []
        if f == 'node':
            val = nodes[0] if nodes[0] in nz else None
        elif f == 'nnodes':
            val = len(nz)
        elif f == 'year':
            val = c['investment_year']
        else:
            vals = set()
            for n in nodes:
                s_, e_ = c['nodes'][str(n)]
                vals.add({'S': s_, 'E': e_, 'dur': (e_ / s_ if s_ else None)}[f])
            val = vals.pop() if len(vals) == 1 else sorted(vals)
        ok = isinstance(val, (int, float)) and abs(float(val) - abs(wv)) < 1e-12
        return {'status': 'match' if ok else 'MISMATCH', 'counterpart': f"{FZ_REL} cells[{cell}].candidate_canonical "
                f"({f}{' node ' + ','.join(map(str, nodes)) if nodes else ''})", 'counterpart_value': val,
                'value_at_written_precision': str(val), 'rule': 'plan (candidate record)'}
    if kind == 'w163':
        c = X.W163.get(spec['id'])
        if c is None:
            return {'status': 'no counterpart', 'note': f"W163 check {spec['id']} not found"}
        if t['type'] in ('num',) and '.' in t['text']:
            ok_val = float(_fmt(abs(float(spec['value'])), _dp(w_txt))) == abs(wv)      # the record at the written precision
        else:
            ok_val = abs(abs(wv) - abs(float(spec['value']))) <= 1e-12 * max(1.0, abs(spec['value']))
        st = c['status'] if ok_val and c['status'] in ('match', 'approximate') else 'MISMATCH'
        return {'status': st, 'counterpart': f"{X.inputs['W163']['path']} checks_in_v4[{spec['id']}] (sha256 "
                f"{X.inputs['W163']['sha256'][:8]}; committed check, status {c['status']}): {str(c.get('counterpart') or c.get('table_ref'))[:200]}",
                'counterpart_value': c.get('value', c.get('table_value_full_precision')),
                'value_at_written_precision': c.get('value_at_written_precision', c.get('table_value_at_written_precision')),
                'rule': 'named record (W163 committed check)', 'note': spec.get('why')}
    if kind == 'def':
        txt = _norm(X.texts[spec['key']] if spec['key'] in X.texts else X.defs[spec['key']])
        ok = _norm(spec['phrase']) in txt
        return {'status': 'match' if ok else 'no counterpart',
                'counterpart': f"{X.def_src[spec['key']]}: \"{spec['phrase']}\"", 'counterpart_value': spec['phrase'],
                'rule': 'definition (phrase in a named source)', 'note': spec.get('why')}
    if kind == 'submitted':
        sub = X.texts['SUBMITTED_MAIN']
        hits = _find_lines(sub, w_txt)
        return {'status': 'match' if hits else 'no counterpart',
                'counterpart': f"{X.inputs['SUBMITTED_MAIN']['path']} lines {hits} (sha256 {X.inputs['SUBMITTED_MAIN']['sha256'][:8]})",
                'counterpart_value': w_txt, 'rule': 'submitted-version figure (named, to be replaced)', 'note': spec.get('desc')}
    val, src_desc = _source_value(X, spec)
    note = spec.get('note')
    if kind == 'int':
        ok = val is not None and float(val) == (wv if explicit else abs(wv))
        return {'status': 'match' if ok else 'MISMATCH', 'counterpart': src_desc, 'counterpart_value': val,
                'value_at_written_precision': str(val), 'rule': 'count / integer (exact)', 'note': note}
    if kind == 'sci':
        ok = abs(val - wv) <= 1e-15
        return {'status': 'match' if ok else 'MISMATCH', 'counterpart': src_desc, 'counterpart_value': val,
                'rule': 'exponent', 'note': note}
    if kind == 'pu':
        shown = f'{val:.3g}'
        ok = float(shown) == abs(wv)
        return {'status': 'match' if ok else 'MISMATCH', 'counterpart': src_desc, 'counterpart_value': val,
                'value_at_written_precision': shown, 'rule': 'p.u. 3 significant figures', 'note': note}
    if kind in ('pctr', 'near'):
        vals = val
        dp = _dp(w_txt)
        if kind == 'pctr':
            shown = sorted({_fmt(100 * v, dp) for v in vals})
            ok = shown == [_fmt(abs(wv), dp)]
            st = 'approximate' if ok and dp < 1 else ('match' if ok else 'MISMATCH')
            return {'status': st, 'counterpart': src_desc, 'counterpart_value': vals, 'value_at_written_precision': shown,
                    'rule': '% 1 dp (export); written as a whole-percent summary of the five cells', 'note': note}
        ok = all(abs(v - abs(wv)) <= 0.01 + 1e-12 for v in vals)
        return {'status': 'approximate' if ok else 'MISMATCH', 'counterpart': src_desc, 'counterpart_value': vals,
                'value_at_written_precision': f'{min(vals):.3f}-{max(vals):.3f}',
                'rule': 'near (within 0.01 of every value of the range; the written figure summarizes a range)', 'note': note}
    scale, rdp = RULE[kind]
    x = float(val) * scale
    cmp_x = x if explicit else abs(x)
    cmp_w = wv if explicit else abs(wv)
    sign_note = None
    if not explicit and x < 0:
        sign_note = 'sign not written; magnitude compared (the prose / column states the direction)'
    if spec.get('signed') and not explicit and x < 0:
        sign_note = 'MISMATCH of sign: an unsigned table cell for a negative value'
    if kind == 'e10':
        shown = f'{int(round(x / 10.0)) * 10:,}'
        ok = int(round(cmp_x / 10.0)) * 10 == int(round(cmp_w))
        st = 'match' if ok else 'MISMATCH'
    else:
        dp = _dp(w_txt)
        shown = _fmt(cmp_x, dp)
        ok = float(shown) == cmp_w
        st = 'MISMATCH' if not ok else ('approximate' if rdp is not None and dp < rdp else 'match')
    if sign_note and sign_note.startswith('MISMATCH'):
        st = 'MISMATCH'
    return {'status': st, 'counterpart': src_desc, 'counterpart_value': val, 'value_at_written_precision': shown,
            'rule': f"{RULE_NOTE[kind]}", 'note': '; '.join(x_ for x_ in (note, sign_note) if x_) or None}


# ======================================================================================================================
#  5. text checks: verdicts, superscripts, accepted sentences, the T4 caption
# ======================================================================================================================
def text_checks(X, lines):
    out = []

    def row_check(ln, expect_verdict, flags_expected, what):
        row = lines[ln - 1]
        flags = {'u': '^{u}' in row, 'f': '^{f}' in row}
        verdict_ok = all(v in row for v in expect_verdict)
        out.append({'line': ln, 'what': what, 'verdict_expected': expect_verdict, 'verdict_ok': verdict_ok,
                    'flags_written': flags, 'flags_expected': flags_expected, 'flags_ok': flags == flags_expected,
                    'ok': verdict_ok and flags == flags_expected})

    def flags_of(claim):
        c = X.CL[claim]
        return {'u': X.cells[c['other_cell']]['status'] == 'uncertified', 'f': bool(c.get('uncertified_form_beside_A64_ruling4'))}
    for ln, claim, *_ in VALUE_LADDER_ROWS:
        row_check(ln, [X.CL[claim]['gross_verdict']], flags_of(claim), f'tab:res_value_ladder {claim}')
    for ln, cid, src in ((181, 'CHECK:headline_V_minus_I_settled', 'cl'), (182, 'H:m1.5:value_minus_I', 'cl'),
                         (183, 'H:m1.75:value_minus_I', 't10'), (184, 'H:m2:value_minus_I', 'cl')):
        c = X.CL[cid] if src == 'cl' else X.A64[cid]
        unc = (X.cells[c['other_cell']]['status'] == 'uncertified' if src == 'cl' else
               any(v['status'] != 'certified' for v in c['cells'].values()))
        row_check(ln, [c['gross_verdict'].replace('within the uncertified bar', 'within the uncertified bar')],
                  {'u': unc, 'f': False}, f'tab:res_flex_ladder {cid}')
    for ln, c1, c2 in ((199, 'H:m2:value_minus_I', None), (200, 'I:m2:e2_value_minus_I', 'I:m2:second_MWh'),
                       (201, 'J:e3_value_minus_I', 'J:e2_to_e3'), (202, 'J:e4_value_minus_I', 'J:e3_to_e4'),
                       (203, 'J:e5_value_minus_I', 'J:e4_to_e5')):
        verd = [X.CL[c1]['gross_verdict']] + ([X.CL[c2]['gross_verdict']] if c2 else [])
        row_check(ln, verd, flags_of(c1), f'tab:res_energy_ladder_m2 {c1} / {c2}')
    for ln, arm in ((256, 'C2_calfade'), (257, 'C2'), (258, 'C3_unit'), (259, 'C3_midblock'), (260, 'C4'), (261, 'no_ageing')):
        r = _arm(X, arm)
        row_check(ln, [r['verdict'], 'never' if r['floor_year_070'] is None else str(r['floor_year_070'])],
                  {'u': False, 'f': False}, f'tab:res_ageing {arm}')
    row_check(262, [X.A64['E:soh050:value_minus_I']['gross_verdict'], 'never'], {'u': False, 'f': False},
              'tab:res_ageing 0.50 floor (floor never binds: a64 scored B floor_never_binds_as_predicted)')
    yl = X.t['year_ladder']
    row_check(406, [yl['gross_v6']['verdict'], yl['net_v6']['verdict']], {'u': False, 'f': False}, 'tab:res_year_ladder 2035 - 2030')
    for ln, rate in ((423, 0.0), (424, 0.02), (425, 0.05), (426, 0.08)):
        r = [d for d in X.t['discount'] if float(d['rate']) == rate][0]
        row_check(ln, [r['verdict']], {'u': False, 'f': False}, f'tab:res_discount {rate}')
    row_check(362, [X.t['three_by_three']['rows'][2]['label'].split(';')[1].strip().split(' ')[0]], {'u': False, 'f': False},
              'tab:res_multiscenario value - I (T11 label "determinate True")')
    # accepted sentences, verbatim
    p5 = X.texts['PARAGRAPHS_V5']
    acc = re.findall(r'^\d\. [^:]+: \*(.+?)\*\.?$', re.sub(r'\n   ', ' ', p5.split('## Accepted sentences')[1].split('##')[0]),
                     re.M)
    frame_flat = re.sub(r'\n%\s*', ' ', X.frame)
    quotes = re.findall(r'Accepted sentence (\d)[^"]*"([^"]+)"', frame_flat)
    sent = []
    for n, q in quotes:
        ref = acc[int(n) - 1] if int(n) - 1 < len(acc) else None
        a, b = _norm(q), _norm(ref or '')
        a2 = a.replace('$\\lambda_t$', 'lambda_t')
        sent.append({'sentence': int(n), 'frame': q, 'paragraphs_v5': ref, 'verbatim': a == b,
                     'difference': None if a == b else _first_diff(a, b)})
    out.append({'line': 'accepted sentences', 'what': 'paragraphs_v5.md accepted sentences 1-4, verbatim', 'sentences': sent,
                'ok': len(sent) == 4 and all(s['verbatim'] for s in sent)})
    # source rows named in the comments exist in the frozen tables
    ids = set(X.CL) | set(X.A64)
    named = []
    for i, ln in enumerate(lines):
        com = W164.split_comment(ln)[1]
        if not com:
            continue
        for m in re.finditer(r'\b((?:CHECK|[BCEGHIJL]):[A-Za-z0-9_.:]+)', com):
            nm = m.group(1).rstrip('.,;:)')
            exact = nm in ids
            pref = sorted(c for c in ids if c.startswith(nm)) if not exact else []
            named.append({'line': i + 1, 'name': nm, 'exists': exact or len(pref) == 1,
                          'resolved': nm if exact else (pref[0] if len(pref) == 1 else pref)})
    arms = {r['arm'] for r in X.t['ageing']['rows']}
    t8_named = re.search(r'T8 rows ([^(]+)\(', X.frame).group(1)
    t8_list = [a.strip() for a in t8_named.split(',') if a.strip()]
    t6 = _text(os.path.join(EXPORT, 'T6.csv'))
    t3 = _text(os.path.join(EXPORT, 'T3.csv'))
    rows_named = {'T6 "coordinated"': 'coordinated (settled x = 0' in t6,
                  'T6 "price-taker NRF, best of 3: warm_from_certified"': 'price-taker NRF,best of 3: warm_from_certified' in t6,
                  'T6 "passive NRF, best of 3: perturbed"': 'passive NRF,best of 3: perturbed' in t6,
                  'T3 "conservative A64 (4 intervals)"': 'conservative A64 (4 intervals' in t3,
                  'T3 "committed W145 (3 intervals)"': 'committed W145 (3 intervals)' in t3}
    out.append({'line': 'source comments', 'what': 'every claim / row named in the frame\'s comments exists in the frozen tables '
                '(claim ids exact or a unique prefix; T8 arm names; T6 / T3 row names in the export CSV)',
                'claims_named': named, 't8_arms_named': t8_list, 't8_arms_exist': all(a in arms for a in t8_list),
                'rows_named': rows_named,
                'ok': all(x['exists'] for x in named) and all(a in arms for a in t8_list) and all(rows_named.values())})
    j4 = lines[202 - 1]
    out.append({'line': 202, 'what': 'Addendum 64 ruling 4: "the manuscript uses the conservative figures (61.3 kEUR/MWh; J at its '
                'uncertified bar)" -- the 4 MWh row of tab:res_energy_ladder_m2 in its main columns', 'ok': '15.9' in j4,
                'row': j4.strip(), 'uncertified_bar_kEUR': round(X.CL['J:e4_value_minus_I']['uncertified_form_beside_A64_ruling4']['bar'] / 1e3, 1),
                'note': 'the row prints the flagged certified threshold 9.7 / 18.14 x and puts the uncertified bar 15.9 / 11.12 x in the '
                        'caption; the frozen claim notes "the MANUSCRIPT uses J 4 MWh at its uncertified bar" (no verdict changes)'})
    t4 = _norm(re.search(r'\\caption\{(.*?)\}\n', X.texts['T4_TEX'], re.S).group(1))
    phrase = 'the investment-year comparison in this instance is decided by the salvage convention, not by operation'
    out.append({'line': 383, 'what': 'T4 caption sentence quoted "verbatim" (l. 383-384)', 'ok': phrase.lower() in t4.lower(),
                'source': X.inputs['T4_TEX']['path']})
    return out


def _first_diff(a, b):
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return {'at': i, 'frame': a[max(0, i - 40):i + 60], 'paragraphs_v5': b[max(0, i - 40):i + 60]}


# ======================================================================================================================
#  6. labels, figures, references
# ======================================================================================================================
LABEL_RX = re.compile(r'\\label\{([^}]+)\}')
REF_RX = re.compile(r'\\(?:ref|eqref|autoref|cref|Cref|pageref)\{([^}]+)\}')


def label_checks(X, main_lines):
    fl = X.frame.split('\n')
    labels = [(i + 1, m.group(1)) for i, ln in enumerate(fl) for m in LABEL_RX.finditer(W164.split_comment(ln)[0])]
    names = [n for _, n in labels]
    dup = sorted(n for n, c in collections.Counter(names).items() if c > 1)
    hdr = re.search(r'% Labels introduced:(.*?)% ====', X.frame, re.S).group(1)
    hdr = re.sub(r'\n%\s*', ' ', hdr)
    hdr_list = []
    for part in re.split(r';\s*', hdr):
        part = part.strip().rstrip('.')
        if not part:
            continue
        items = [p.strip() for p in part.split(',')]
        prefix = items[0].split(':')[0]
        for it in items:
            it = it.replace('(kept)', '').strip().rstrip('.')
            if not it:
                continue
            hdr_list.append(it if ':' in it else f'{prefix}:{it}')
    a, b = ADD73_RANGE
    main_labels = [(i + 1, m.group(1)) for i, ln in enumerate(main_lines) for m in LABEL_RX.finditer(W164.split_comment(ln)[0])]
    outside = {n for ln, n in main_labels if not a <= ln <= b}
    inside = {n for ln, n in main_labels if a <= ln <= b}
    collide = sorted(n for n in names if n in outside)
    refs = []
    for i, ln in enumerate(fl):
        body, com, _ = W164.split_comment(ln)
        for scope, seg in (('body', body), ('comment', com or '')):
            for m in REF_RX.finditer(seg):
                tgt = m.group(1)
                refs.append({'line': i + 1, 'scope': scope, 'target': tgt,
                             'resolves_to': 'frame' if tgt in names else ('main.tex (outside the replaced range)'
                                                                           if tgt in outside else None)})
    unresolved = [r for r in refs if r['resolves_to'] is None]
    main_refs_to_old = []
    for i, ln in enumerate(main_lines):
        if a <= i + 1 <= b:
            continue
        for m in REF_RX.finditer(W164.split_comment(ln)[0]):
            if m.group(1) in inside:
                main_refs_to_old.append({'line': i + 1, 'target': m.group(1),
                                         'kept_by_frame': m.group(1) in names})
    figs = [m.group(1) for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', X.frame)]
    on_disk = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REPO, FIGS, '*.pdf')))
    cap = X.texts['CAPTIONS_MD']
    fig_rows = []
    for f in figs:
        rel = os.path.join(FIGS, f)
        sha = _sha(rel) if os.path.exists(os.path.join(REPO, rel)) else None
        m = re.search(re.escape(f'`{f}` (sha256 ') + r'([0-9a-f]{8})', cap)
        fig_rows.append({'file': f, 'exists': sha is not None, 'sha256': sha, 'captions_md_prefix': m.group(1) if m else None,
                         'sha_matches_captions_md': bool(m and sha and sha.startswith(m.group(1))),
                         'w165_manifest_sha256': X.j['W165_MANIFEST'].get(rel),
                         'in_clone_already': os.path.exists(os.path.join(REPO, CLONE, f))})
    return {'labels': [{'line': ln, 'label': n} for ln, n in labels], 'n_labels': len(labels), 'duplicates': dup,
            'header_list': hdr_list, 'header_equals_labels': sorted(hdr_list) == sorted(names),
            'header_minus_labels': sorted(set(hdr_list) - set(names)), 'labels_minus_header': sorted(set(names) - set(hdr_list)),
            'collide_with_main_outside_range': collide, 'old_section4_labels_in_main': sorted(inside),
            'main_refs_outside_range_to_old_section4_labels': main_refs_to_old, 'refs': refs, 'unresolved_refs': unresolved,
            'figures': fig_rows, 'export_figures_pdf': on_disk, 'figure_set_equals_export': sorted(figs) == on_disk}


def literal_refs(main_lines):
    rx = re.compile(r'(\(?)(Sections?|Subsections?)~(4(?:\.\d+)?)\b(\)?)')
    a, b = ADD73_RANGE
    found = []
    for i, ln in enumerate(main_lines):
        if a <= i + 1 <= b:
            continue
        body = W164.split_comment(ln)[0]
        for m in rx.finditer(body):
            found.append({'line': i + 1, 'text': m.group(0), 'context': body[max(0, m.start() - 70):m.end() + 30].strip()})
    listed = []
    for ln, txt in ADD73_LITERALS.items():
        hits = [f for f in found if f['line'] == ln and txt.strip('()') in f['text']]
        same_text = [f for f in found if txt.strip('()') == f['text'].strip('()')]
        listed.append({'addendum73_line': ln, 'addendum73_text': txt, 'found_at_listed_line': bool(hits),
                       'found_at_lines': [f['line'] for f in same_text]})
    listed_lines = {ln for ln in ADD73_LITERALS}
    matched = set()
    for x in listed:
        matched.update(x['found_at_lines'] if not x['found_at_listed_line'] else [x['addendum73_line']])
    extra = [f for f in found if f['line'] not in matched]
    return {'found': found, 'addendum73_list': listed, 'not_on_addendum73_list': extra, 'listed_lines': sorted(listed_lines)}


# ======================================================================================================================
#  7. guards, post-run, summary
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    _g, pk176, _ok = W176.guards_state()
    counts = dict(pk176['counts'], w177a=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W177a post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W177a post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def md_summary(o):
    L = [f"# W177a -- section 4 frame check ({o['frame']['path']} sha256 {o['frame']['sha256'][:8]}, commit {FRAME_COMMIT})", '',
         f"Zero solves; script `{SCRIPT_REL}`; frozen tables `{FZ_REL}` (sha256 {FZ_SHA[:8]}); clone {CLONE} HEAD "
         f"{o['clone']['head'][:8]} (main.tex sha256 {o['clone']['main_sha256'][:8]}). Objective convention: "
         f"{o['objective_convention']}", '', '## Predictions (expert, Addendum 73)', '']
    for p in o['prediction']:
        L.append(f"- **{p['item']}** -- {p['outcome']}")
    L += ['', '## CONFIRM points', '', '| id | lines | point | outcome | summary |', '|---|---|---|---|---|']
    for c in o['confirm']:
        L.append(f"| {c['id']} | {c['lines']} | {c['point']} | {c['outcome']} | {c['summary']} |")
    s = o['summary']
    L += ['', '## Number check', '', f"Tokens {s['n_tokens']}: " + ', '.join(f'{k} {v}' for k, v in s['by_status'].items()),
          '', f"Composed tables and captions (body tokens in table / figure environments): {s['tables_captions']}", '',
          '| line | scope | written | status | value at written precision | source | note |', '|---|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['status'] in ('MISMATCH', 'approximate', 'no counterpart'):
            L.append(f"| {t['line']} | {t['scope']} | {t['written']} | {t['status']} | {t.get('value_at_written_precision')} | "
                     f"{str(t.get('counterpart'))[:160]} | {(t.get('note') or '')[:200]} |")
    L += ['', '## Text checks', '']
    for c in o['text_checks']:
        L.append(f"- l. {c['line']} {c['what']}: {'ok' if c['ok'] else 'NOT OK'}")
    lab = o['labels']
    L += ['', '## Labels, figures, references', '',
          f"- labels {lab['n_labels']}, duplicates {lab['duplicates']}, header list equal {lab['header_equals_labels']} "
          f"(header - labels {lab['header_minus_labels']}, labels - header {lab['labels_minus_header']}), collide with main.tex "
          f"outside the range {lab['collide_with_main_outside_range']}",
          f"- unresolved refs {[(r['line'], r['target']) for r in lab['unresolved_refs']]}",
          f"- main.tex refs outside the range to old section-4 labels: {lab['main_refs_outside_range_to_old_section4_labels']}",
          f"- figures {[(f['file'], f['exists'], f['sha_matches_captions_md']) for f in lab['figures']]}; set equals export "
          f"{lab['figure_set_equals_export']}"]
    lr = o['literal_refs']
    L += ['', '### Literal references in main.tex at the clone HEAD (outside the replaced range)', '']
    for f in lr['found']:
        L.append(f"- l. {f['line']}: `{f['text']}` -- {f['context']}")
    L += ['', 'Addendum 73 list: ' + '; '.join(f"l. {x['addendum73_line']} {x['addendum73_text']} -> "
                                                f"{'found there' if x['found_at_listed_line'] else 'NOT at that line; found at ' + str(x['found_at_lines'])}"
                                                for x in lr['addendum73_list']),
          'Not on the Addendum 73 list: ' + '; '.join(f"l. {f['line']} {f['text']}" for f in lr['not_on_addendum73_list'])]
    L += ['', '## Integrity', '', f"checks: {sum(v is True for v in o['checks'].values())}/{len(o['checks'])} True; failed "
          f"{o['failed']}; guards ok {o['guards_ok']}; exit {o['exit_code']}", '']
    return '\n'.join(L) + '\n'


# ======================================================================================================================
#  8. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or OUT_DIR
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W177a'
    pre = []
    od = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        extra = sorted(set(os.listdir(od)) - {OUT_NAMES['log']})
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
    inputs = {}
    for key, rel in INPUTS.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'capture path missing: {rel}')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel), 'blob': _git('rev-parse', f'HEAD:{rel}')}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if not pre:
        if inputs['FRAME']['sha256'] != FRAME_SHA:
            pre.append(f"{FRAME} sha256 {inputs['FRAME']['sha256']} != declared {FRAME_SHA}")
        if not (inputs['FRAME']['last_commit'] or '').startswith(FRAME_COMMIT):
            pre.append(f"{FRAME} last commit {inputs['FRAME']['last_commit']} != {FRAME_COMMIT}")
        if inputs['FROZEN_JSON']['sha256'] != FZ_SHA:
            pre.append(f'frozen JSON sha256 != {FZ_SHA}')
    mabs = os.path.join(REPO, CLONE)
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex'], capture_output=True,
                               text=True).stdout.strip()
    main_rel = os.path.join(CLONE, 'main.tex')
    main_sha = _sha(main_rel) if os.path.exists(os.path.join(REPO, main_rel)) else None
    if head != CLONE_HEAD:
        pre.append(f'clone HEAD {head} != declared {CLONE_HEAD}')
    if main_sha != MAIN_SHA:
        pre.append(f'clone main.tex sha256 {main_sha} != declared {MAIN_SHA}')
    if porcelain:
        pre.append(f'clone *.tex not clean: {porcelain!r}')
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] preconditions hold: {len(inputs)} inputs committed clean; frame {FRAME_SHA[:8]}; clone {head[:8]} main.tex '
         f'{main_sha[:8]}; script committed clean {script_clean}')

    X = Ctx(inputs)
    X.texts['BE_CONV'] = X.t['break_even']['objective_convention']
    X.texts['CAPTIONS'] = X.texts['CAPTIONS_MD']
    X.texts['W170_FORMULA'] = X.j['W170']['formula']
    X.def_src = {'BE_CONV': f'{FZ_REL} tables.break_even.objective_convention', 'CAPTIONS': inputs['CAPTIONS_MD']['path'],
                 'W170_FORMULA': f"{inputs['W170']['path']} formula", 'T7_TEX': inputs['T7_TEX']['path']}
    main_text = _text(main_rel)
    main_lines = main_text.split('\n')
    X.main_results_line = [i + 1 for i, ln in enumerate(main_lines) if ln.startswith('\\section{Results}')][0]
    concl = [i + 1 for i, ln in enumerate(main_lines) if 'CONCLUSIONS' in ln and ln.lstrip().startswith('%')][0]
    banner_start = concl
    while banner_start > 1 and re.fullmatch(r'%\s*=+\s*', main_lines[banner_start - 2]):
        banner_start -= 1
    X.main_before_banner = banner_start - 1
    X.frame_items = len(re.findall(r'\\item\b', X.frame.split('\\label{subsec:res_insights}')[1]))
    sal, consts, coh2025 = confirm_salvage(X)
    X.coh2025 = coh2025
    confirm = confirm_checks(X, sal, consts, coh2025)

    # ---- tokens and declarations ------------------------------------------------------------------------------------
    doc = W164.Doc(FRAME, X.frame)
    toks = W174.tokenize_v3(doc)
    lines = X.frame.split('\n')
    leaves = W164.json_leaves(X.fz)
    AX = W164.AutoIndex(leaves)
    for t in toks:
        t['status'] = None
        if t['excluded']:
            t.update(status='excluded', category=f"structure: {t['excluded']}")
        elif t['type'] in IDENT_TYPES:
            t.update(status='excluded', category=f"identifier ({t['type']})")
    decls = declarations()
    stale, claimed_twice, decl_out = [], [], []
    by_line = collections.defaultdict(list)
    for t in toks:
        by_line[t['line']].append(t)
    for d in decls:
        ln = d['line']
        text = lines[ln - 1] if ln - 1 < len(lines) else ''
        if d['frag']:
            k = text.find(d['frag'])
            if k < 0 or text.count(d['frag']) != 1:
                stale.append({'line': ln, 'frag': d['frag'], 'why': 'fragment not found once on the line'})
                continue
            span = (k, k + len(d['frag']))
        else:
            span = (0, len(text))
        cand = [t for t in by_line[ln] if t['status'] != 'excluded' and span[0] <= t['col0'] < span[1]]
        if len(cand) != len(d['specs']):
            stale.append({'line': ln, 'why': f"{len(d['specs'])} specs for {len(cand)} tokens",
                          'tokens': [t['written'] for t in cand], 'specs': [s['w'] for s in d['specs']]})
            continue
        bad = [(t['written'], s['w']) for t, s in zip(cand, d['specs']) if t['written'].lower() != s['w'].lower()]
        if bad:
            stale.append({'line': ln, 'why': 'written differs from the declaration', 'pairs': bad})
            continue
        res = []
        for t, s in zip(cand, d['specs']):
            if t['status'] is not None:
                claimed_twice.append(t['key'])
                continue
            try:
                r = evaluate(X, s, t, text)
            except Exception as e:  # noqa: BLE001 -- a declaration that cannot be evaluated is a run failure, listed
                stale.append({'line': ln, 'why': f'evaluation error {type(e).__name__}: {e}', 'written': t['written']})
                r = {'status': None}
            if r.get('status') is None:
                continue
            t.update(r)
            t['declared'] = True
            res.append({'written': t['written'], 'col': t['col'], 'status': r['status']})
        decl_out.append({'line': ln, 'tokens': res, 'note': d['note']})
        if d['note']:
            for t in cand:
                t.setdefault('line_note', d['note'])
    unassigned = [t for t in toks if t['status'] is None]
    for t in toks:
        if t['status'] in ('match', 'MISMATCH', 'approximate', 'no counterpart') and t['type'] == 'num':
            c = AX.candidates(t)
            t['auto_candidates_n'] = None if c is None else len(c)
    # ---- scopes ------------------------------------------------------------------------------------------------------
    for t in toks:
        envs = t.get('envs') or []
        if t['scope'] == 'comment':
            t['part'] = 'comment (author slot / source note / header)'
        elif any(e in ('table', 'tabular') for e in envs):
            t['part'] = 'table (tabular)' if 'tabular' in envs else 'table caption'
        elif 'figure' in envs:
            t['part'] = 'figure caption'
        elif any(e.startswith('equation') for e in envs):
            t['part'] = 'equation'
        else:
            t['part'] = 'prose (body)'
    by_status = collections.Counter(str(t['status']) for t in toks)
    tc = [t for t in toks if t['part'] in ('table (tabular)', 'table caption', 'figure caption')]
    tables_captions = dict(collections.Counter(str(t['status']) for t in tc))
    by_part = {p: dict(collections.Counter(str(t['status']) for t in toks if t['part'] == p)) for p in sorted({t['part'] for t in toks})}
    unc_cat = dict(collections.Counter(str(t.get('category')) for t in toks if t['status'] in ('unchecked', 'excluded')))

    txt = text_checks(X, lines)
    lab = label_checks(X, main_lines)
    lit = literal_refs(main_lines)
    sub_cert = [l_ for l_ in main_lines if '\\label{subsubsec:certification}' in l_]
    mdoc = W164.Doc('main.tex', main_text)
    cert_line = [i + 1 for i, l_ in enumerate(main_lines) if '\\label{subsubsec:certification}' in l_]
    cert_num = None
    if cert_line:
        hs = [h for h in mdoc.heads if h['line'] <= cert_line[0]]
        cert_num = hs[-1]['number'] if hs else None
    xref_checks = {'subsubsec:certification number (frame header "Subsection 2.2.7")': cert_num,
                   'equals 2.2.7': cert_num == '2.2.7',
                   'replaced range (Results line, line before the CONCLUSIONS banner)': [X.main_results_line, X.main_before_banner],
                   'equals Addendum 73 range': [X.main_results_line, X.main_before_banner] == list(ADD73_RANGE)}

    # ---- predictions ----------------------------------------------------------------------------------------------------
    n_mis_tc = tables_captions.get('MISMATCH', 0)
    cmap = {c['id']: c for c in confirm}
    c1 = cmap['C1']
    struct = {k: cmap[k]['outcome'] for k in ('C2', 'C3', 'C4', 'C6')}
    prediction = [
        {'item': 'zero MISMATCH in the composed tables and captions',
         'outcome': ('HELD' if n_mis_tc == 0 else 'FAILED') + f': {n_mis_tc} MISMATCH among {len(tc)} table / caption tokens '
                    f"({tables_captions})"},
        {'item': 'the salvage equation needs one correction among its four points',
         'outcome': 'FAILED as worded: items (a)-(d) all hold as worded, and the equation needs TWO corrections -- the cohort '
                    'cost is the expected ENERGY cost only (the definition of I, adjacent to (c)) and the health factor is '
                    '(SoH - SoH^Min)/(1 - SoH^Min) (adjacent to (b)); the frame\'s equation gives '
                    f"{c1['evidence']['reproduction'][0]['frame_over_recorded']:.2f} x and "
                    f"{c1['evidence']['reproduction'][1]['frame_over_recorded']:.2f} x the recorded salvage of yl_y2030 / yl_y2035"},
        {'item': 'the four other structural CONFIRMs (S/E, I/J keys, F2 plan, 16 cycles) hold',
         'outcome': ('HELD' if all(v.startswith('confirmed') for v in struct.values()) else 'FAILED') + ': ' +
                    ', '.join(f'{k} {v}' for k, v in struct.items()) + ' (C4: node 7 is 1 MVA / 3.5 MWh, not 3 MWh)'},
    ]
    checks = {
        'frame_sha_equals_declared': inputs['FRAME']['sha256'] == FRAME_SHA,
        'frame_equals_its_blob_at_HEAD': _sha_bytes(subprocess.run(['git', 'show', f'HEAD:{FRAME}'], cwd=REPO,
                                                                   capture_output=True).stdout) == FRAME_SHA,
        'frozen_json_unchanged': _sha(FZ_REL) == FZ_SHA,
        'clone_head_is_declared': head == CLONE_HEAD,
        'clone_main_sha_is_declared': main_sha == MAIN_SHA,
        'clone_tex_clean': porcelain == '',
        'every_declaration_found_and_evaluated': not stale,
        'no_token_claimed_twice': not claimed_twice,
        'every_token_assigned': not unassigned,
        'statuses_in_vocabulary': all(t['status'] in STATUSES for t in toks),
        'every_confirm_point_has_an_outcome': len(confirm) == 9 and all(c['outcome'] for c in confirm),
        'every_input_committed_clean': all(v['committed_clean'] for v in inputs.values()),
        'production_salvage_form_reproduces_records': c1['production_reproduces_records'],
    }
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
    out = {'schema': 'p515_s53_w177a_section4_frame_check', 'version': 1,
           'stage': 'P5.15 Addendum 73 W177a -- check of the expert section 4 frame (section4_expert_draft.tex)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {'p515_s53_w176_manuscript_number_check.py': _sha('p515_s53_w176_manuscript_number_check.py')}},
           'command_line': sys.argv,
           'frame': {'path': FRAME, 'sha256': inputs['FRAME']['sha256'], 'commit': FRAME_COMMIT,
                     'last_commit': inputs['FRAME']['last_commit'], 'lines': len(lines)},
           'clone': {'dir': CLONE, 'head': head, 'main_sha256': main_sha, 'porcelain_tex': porcelain, 'edited': False},
           'objective_convention': X.fz.get('objective_convention'),
           'frozen_tables': {'path': FZ_REL, 'sha256': FZ_SHA, 'numeric_leaves': len(leaves)},
           'inputs': inputs, 'confirm': confirm, 'prediction': prediction,
           'summary': {'n_tokens': len(toks), 'by_status': dict(by_status), 'tables_captions': tables_captions,
                       'n_tables_captions_tokens': len(tc), 'by_part': by_part, 'unchecked_excluded_by_category': unc_cat,
                       'n_declarations': len(decls), 'n_declarations_applied': len(decl_out)},
           'conventions': {'column': '1-based', 'line': '1-based',
                           'statuses': 'match = equal to the frozen JSON (or the named record) at the export\'s printed '
                                       'rounding (export/README.md); approximate = printed coarser than that rule and equal at '
                                       'its own precision, or a summary of a recorded range; MISMATCH; no counterpart; '
                                       'unchecked (category); excluded (structure / identifier)',
                           'sign': 'a written sign (- / + / $-$) is compared signed; an unsigned figure by magnitude (the '
                                   'prose states the direction); an unsigned table cell for a negative value is a MISMATCH'},
           'declarations_applied': decl_out, 'stale_declarations': stale, 'claimed_twice': claimed_twice,
           'unassigned': [{'line': t['line'], 'written': t['written']} for t in unassigned],
           'text_checks': txt, 'labels': lab, 'literal_refs': lit, 'xref_checks': xref_checks,
           'checks': checks, 'failed': failed, 'tokens': toks, 'guards': guards, 'pickle_guard': pk, 'guards_ok': guards_ok,
           'exit_code': code, 'wall_s': time.time() - t0}
    written = {}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        full = rel if os.path.isabs(rel) else os.path.join(REPO, rel)
        with open(full, 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        with open(full, 'rb') as h:
            written[rel] = _sha_bytes(h.read())
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    man[main_rel] = main_sha
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    # ---- log ---------------------------------------------------------------------------------------------------------
    for c in confirm:
        _log(f"[{tag}] CONFIRM {c['id']} (l. {c['lines']}): {c['outcome']} -- {c['summary'][:600]}")
    for p in prediction:
        _log(f"[{tag}] PREDICTION {p['item']}: {p['outcome']}")
    _log(f"[{tag}] tokens {len(toks)}: {dict(by_status)}; tables + captions {tables_captions}; by part {by_part}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no counterpart'):
            _log(f"[{tag}]   l. {t['line']}:{t['col']} ({t['part']}) {t['status']}: written {t['written']!r} -> "
                 f"{t.get('value_at_written_precision')!r} [{str(t.get('counterpart'))[:180]}] {t.get('note') or ''}")
    for c in txt:
        if not c['ok']:
            _log(f"[{tag}]   text check NOT OK: {json.dumps(c, default=str)[:600]}")
    _log(f"[{tag}] text checks ok {sum(c['ok'] for c in txt)}/{len(txt)}")
    _log(f"[{tag}] labels {lab['n_labels']} duplicates {lab['duplicates']} header equal {lab['header_equals_labels']} collide "
         f"{lab['collide_with_main_outside_range']} unresolved {[(r['line'], r['target']) for r in lab['unresolved_refs']]} "
         f"main refs to old section-4 labels {lab['main_refs_outside_range_to_old_section4_labels']}; figures "
         f"{[(f['file'], f['exists'], f['sha_matches_captions_md']) for f in lab['figures']]} set equal "
         f"{lab['figure_set_equals_export']}")
    _log(f"[{tag}] literal references: {[(f['line'], f['text']) for f in lit['found']]}; not on the Addendum 73 list "
         f"{[(f['line'], f['text']) for f in lit['not_on_addendum73_list']]}; list {lit['addendum73_list']}")
    _log(f"[{tag}] xref checks {xref_checks}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if stale or claimed_twice or unassigned:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; unassigned '
             f"{[(t['line'], t['written']) for t in unassigned]}")
    if failed:
        _log(f'[{tag}] FAILED: {failed}')
    _log(f"[{tag}] wrote {', '.join(f'{k} {v[:8]}' for k, v in written.items())}")
    _log(f"[{tag}] guards {[(k, v['counts']['permitted_solve'], v['counts']['blocked_solve'], v['verify_0_failures']) for k, v in guards.items()]}; "
         f"pickle ok {pk['ok']}; exit {code}; wall {time.time() - t0:.1f} s")
    for _n, g in reversed(GUARDS):
        g.uninstall()
    pickle.load, pickle.loads = _PICKLE_ORIG
    sys.exit(code)


if __name__ == '__main__':
    main()
