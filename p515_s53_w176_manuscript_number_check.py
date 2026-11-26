"""P5.15 Step 6, Planner task W176b -- NUMBER CHECK OF THE MANUSCRIPT .tex FILES AT OVERLEAF 8b76423 (the W164 / W171a /
W174a checker, DECLARATIONS VERSION 4). ZERO SOLVES, NO MODEL LOADS. A NEW FILE: the W174a script is IMPORTED (it imports
W171a -> W164 -> W163 -> W162 -> W161 -> W160, each arming its guards) and its records, value-table machinery, quotation
check and version-3 declarations are USED; W160-W174 are not edited. Every output is a new file opened 'x' in a new
directory.

WHAT CHANGES AGAINST W174a (version 3 -> version 4)
  1. The manuscript is the Overleaf clone at 8b76423: round 4 (STEP6_ROUND4_CORRECTIONS.md, 14 edits) and the l. 858
     reference fix are in (PLANNER_BRIEF_2026-09-13.md Addenda 71-72): section 2 (l. 445 stop reasons and "every
     evaluated poll", l. 639 determinacy paragraph, l. 671 end-of-block SoH, l. 748 / 858 -> sec:case_ess_params, l. 833
     comma), section 3.5 (chemistry: "lithium-ion ... the NREL ATB utility-scale battery category"; the C4 footnote
     "entered as 10,000 cycles at 0.80 depth of discharge"), section 3.6 (the "one frozen configuration" sentence scoped
     to the certified evaluations, the search campaigns without the tight tail; the benchmark paragraph with the
     consistency pass, the 1 EUR/MWh minimum-curtailment term, the common Q and the instance), Appendix A (reflows, the
     AA-off clause, the settlement weight set to one at the ADMM conversion, k <- k + 1 restored), every [CONFIRM] /
     [AUTHOR] comment deleted; the letter's R1.2(iv) (chemistry wording, "Section~3.5").
  2. Every version-3 declaration (the set W174a applied -- the 128 carried version-2 declarations and the 92 version-3
     ones -- reconstructed and checked against its committed results) is RE-LOCATED in the 8b76423 texts and reported
     as carried (fragment found once, evaluated unchanged), superseded, or removed (fragment gone; the version-4
     declaration that replaced it is named, with the reason).
  3. New declarations for every number of round 4: the C4 footnote (10,000 / 0.80, V42), the search campaigns of
     section 3.6 ("three", V43), the benchmark paragraph (1 EUR/MWh, V44; "three" starts, V41v4; the consistency pass,
     the common Q and the instance as named-setting rows of V44 / V45), the stop reasons of section 2 ("three runs ...
     two ... one", S401) and the settlement weight of Algorithm 2 ("one", A401); the chemistry sentence as named-setting
     rows (V01v4). Revised passages are never handed to the automatic value index (W174a rule, kept).
  4. The main.tex line rules are RE-PINNED to main.tex sha256 7aa9e105...: W174a's submitted-version regions are
     recomputed from the W171a pin exactly as W174a did (and checked against W174a's constant and committed results),
     then mapped line by line (difflib equal blocks) from the W174a pin (260bd83, main.tex cb237b2d, read with
     `git show` in the clone) and kept only where every line is unchanged; the revised passages, the value-table range,
     the network tables and Appendices B-D are mapped the same way and checked against the constants below.
  5. Reconciliations against W174a's committed results: the submitted-version figures (559 at 260bd83) token by token
     (mapped line, column, text), and the approximate tokens (12 at 260bd83), each with what changed and why.
  6. Reviewer quotations re-checked against manuscript_review/Reviewers Comments.docx (W171a's check, unchanged).
  The refusal behaviours stay: declared Overleaf commit and the sha256 of every *.tex must match; the output directory
  must be new (only the launch log); every token assigned; a stale declaration or a token claimed twice exits 3.

RECORDED PREDICTION (expert, Addendum 72): zero MISMATCH in main.tex and the letter, the same twelve approximates as
W174a (printed roundings), and the submitted-version count reduced only where section 3.5-3.6 text replaced submitted
text. Scored item by item in the outputs.

GUARDS. `SolveProfileGuard(permitted=())` installed BEFORE any other project import and verified at exactly 0 at the
end, with every guard the W174a chain arms; `pickle.load` / `pickle.loads` blocked for the whole run, every counter
verified at 0. git reads are read-only (`git show` / `git log` / `git status` / `git rev-parse` / `git ls-files`, also in
the clone). The cost workbook data/SRP1/SharedESS/SRP1_ESS.xlsx is read with openpyxl in read-only mode (as W175).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w176_manuscript_check && \\
    mkdir data/SRP1/Results/P515S53/w176_manuscript_check/overleaf_8b76423 && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w176_manuscript_number_check.py \\
        --overleaf-commit 8b76423 --expect main.tex=<sha256> ... (every *.tex of the clone) \\
        > data/SRP1/Results/P515S53/w176_manuscript_check/overleaf_8b76423/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_gate_result_bool_typing_test.py \\
        --out data/SRP1/Results/P515S53/w176_manuscript_check/overleaf_8b76423/w176_bool_typing_test.json \\
        > data/SRP1/Results/P515S53/w176_manuscript_check/overleaf_8b76423/w176_bool_typing_test.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w176_manuscript_number_check.py \\
        --overleaf-commit 8b76423 --post-run
Exit: 0 = written, every integrity check holds and the guards are at 0 (MISMATCHES are FINDINGS and never change the
exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault (nothing written).
"""
import argparse
import collections
import difflib
import glob
import json
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W176b manuscript number check v4 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}
_PICKLE_ORIG = (pickle.load, pickle.loads)


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W176: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W176: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s53_w174_manuscript_number_check as W174  # noqa: E402 -- arms its guards, imports W171a -> W160 (none edited)

pickle.load, pickle.loads = _blocked_load, _blocked_loads  # this script's block, re-installed after the imports

import openpyxl  # noqa: E402  (third-party, read-only workbook scan as in W175; not project code)

W171, W164, W163, W162, W161, W160 = W174.W171, W174.W164, W174.W163, W174.W162, W174.W161, W174.W160
GUARDS = W160.W157._dedupe((('w176_manuscript_number_check', GUARD),) + tuple(W174.GUARDS))

_log, _sha, _sha_bytes, _git, _jl, _text = W160._log, W160._sha, W160._sha_bytes, W160._git, W162._jl, W162._text
R, num, unch, ntc, D = W164.R, W164.num, W164.unch, W164.ntc, W164.D
row, V, jf, wv3, count_eq3 = W174.row, W174.V, W174.jf, W174.wv3, W174.count_eq3
cmp_exact, cmp_round = W174.cmp_exact, W174.cmp_round
_cell = W171._cell

SCRIPT_REL = os.path.basename(__file__)
W174_SCRIPT, W174_SCRIPT_COMMIT = 'p515_s53_w174_manuscript_number_check.py', 'bb3830fb'
W174_RESULTS = os.path.join(W160.S53, 'w174_manuscript_check', 'overleaf_260bd83', 'w174_manuscript_number_check.json')
W174_RESULTS_COMMIT = 'b4f93d7b'
W174_COMMIT_IN_CLONE = '260bd83'
W171_COMMIT_IN_CLONE = W174.W171_COMMIT_IN_CLONE
OUT_ROOT = os.path.join(W160.S53, 'w176_manuscript_check')
DEFAULT_MANUSCRIPT_DIR = W164.DEFAULT_MANUSCRIPT_DIR
DECLARATIONS_VERSION = 4
S53 = W160.S53
LETTER, HIGHLIGHTS, COVER, MAIN, DRAFT = W174.LETTER, W174.HIGHLIGHTS, W174.COVER, W174.MAIN, W174.DRAFT
BIB = 'bibliography.bib'
ESS_XLSX = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')

# inputs added to W174a's (each committed clean; sha256 and git blob recorded)
EXTRA_INPUTS_V4 = {
    'W174_RESULTS': W174_RESULTS,
    'W174_SCRIPT': W174_SCRIPT,
    'W175_JSON': os.path.join(S53, 'w175_confirm', 'w175_confirm.json'),
    'W174B_JSON': os.path.join(S53, 'w174b_reaudit', 'w174b_checks.json'),
    'BENCH_SPEC_V5': os.path.join(S53, 'w116_benchmark_nrf', 'frozen_s53_benchmark_spec_v5_bca69f97.json'),
    'BENCH_REPORT_V3': os.path.join(S53, 'w116_benchmark_nrf', 'report_v3', 'report_v3.json'),
    'ESS_XLSX': ESS_XLSX,
    'ROUND4_MD': 'STEP6_ROUND4_CORRECTIONS.md',
}
JSON_KEYS_V4 = ('W174_RESULTS', 'W175_JSON', 'W174B_JSON', 'BENCH_SPEC_V5', 'BENCH_REPORT_V3')
CODE_V4 = {'uncoordinated_benchmark': 'uncoordinated_benchmark.py'}

OUT_NAMES = {'json': 'w176_manuscript_number_check.json', 'md': 'w176_manuscript_number_check.md',
             'man': 'manifest_sha256.json', 'log': 'launch.log', 'typing_json': 'w176_bool_typing_test.json',
             'typing_log': 'w176_bool_typing_test.log', 'post': 'manifest_post_run_sha256.json'}

STATUSES = W174.STATUSES
ROW_STATUSES = W174.ROW_STATUSES

# ---- main.tex line rules, re-pinned to this main.tex (Overleaf 8b76423) -------------------------------------------------
MAIN_LINE_RULES_SHA = '7aa9e105deccdf61cced665f03d95b02ea736fe5c34f3402f6281db89d19624b'
# revised passages (W174a's, mapped from cb237b2d; the run recomputes the mapping and refuses on a difference)
REVISED = (
    ('sec2', (315, 901), 'section 2 (revised, rounds 1-4)', r'\section{Shared ESS Planning Framework}'),
    ('sec3_1', (939, 967), 'section 3.1 Investment Costs (revised table and paragraph, round 3 C.1)', r'\subsection{Investment Costs}'),
    ('sec3_3', (975, 977), 'section 3.3 sentence on TN generation (round 3 C.2)', r'\textcolor{blue}{'),
    ('sec3_5_6', (1058, 1125), 'sections 3.5-3.6 (round 3 C.4-C.5, round 4)', r'\subsection{\textcolor{blue}{Shared Energy Storage Parameters}}'),
    ('app_a', (1467, 1666), 'Appendix A (rewritten, rounds 2-4)', r'\section{\textcolor{blue}{TSO--DSO Coordinated Operational Planning}}'),
)
VALUE_TABLE_LINES = (1058, 1124)          # W174a's l. 1066-1150 mapped (section 3.5 heading to the line before RESULTS)
NETWORK_TABLES = (972, 1057)              # W174a's 980-1065 mapped
APP_BCD = (1667, 2027)                    # W174a's 1720-2080 mapped
# the expected re-pinned submitted-version regions (W174a order: r47, r_delete x2, keyins, results, appE)
EXPECTED_MAIN_REGIONS = ((1189, 1201), (1202, 1339), (1340, 1428), (1429, 1448), (1126, 1188), (2028, 2141))
REGION_DROPPED = {}                       # no W174a region is expected to change
V3_RANGES = {'REVISED': {k: r for k, r, _d, _h in W174.REVISED}, 'VALUE_TABLE_LINES': W174.VALUE_TABLE_LINES,
             'NETWORK_TABLES': W174.NETWORK_TABLES, 'APP_BCD': W174.APP_BCD}

CHEM_TERMS = re.compile(r'\bLFP\b|iron phosphate|LiFePO|\bNMC\b|lithium', re.I)


# ======================================================================================================================
#  1. records
# ======================================================================================================================
class RecordsV4(W174.RecordsV3):
    def __init__(self, inputs, man_doc, sub_doc):
        super().__init__(inputs, man_doc, sub_doc)
        self.code.update({k: W171.Src(v) for k, v in CODE_V4.items()})
        self.j.update({k: _jl(inputs[k]['path']) for k in JSON_KEYS_V4})
        self.xlsx_scan = None
        self.bib_entry = None
        self.chem_in_manuscript = None


def scan_cost_workbook(rel):
    """every string cell of the committed cost workbook (openpyxl, read-only), searched for the terms the section 3.5
    chemistry sentence names; the sheet names are recorded."""
    pats = {'li_ion': re.compile(r'li-?ion|lithium', re.I), 'atb': re.compile(r'\bATB\b|annual technology baseline', re.I),
            'utility': re.compile(r'utility', re.I), 'nrel': re.compile(r'\bNREL\b', re.I),
            'other_chemistry': re.compile(r'\bLFP\b|phosphate|LiFePO|\bNMC\b', re.I)}
    out = {'path': rel, 'sheets': [], 'cells_scanned': 0, 'hits': {k: [] for k in pats}, 'n_hits': {k: 0 for k in pats}}
    wb = openpyxl.load_workbook(os.path.join(REPO, rel), read_only=True, data_only=False)
    for ws in wb.worksheets:
        out['sheets'].append(ws.title)
        for r_ in ws.iter_rows():
            for c in r_:
                out['cells_scanned'] += 1
                v = c.value
                if not isinstance(v, str):
                    continue
                for k, rx in pats.items():
                    if rx.search(v):
                        out['n_hits'][k] += 1
                        if len(out['hits'][k]) < 6:
                            out['hits'][k].append({'sheet': ws.title, 'cell': c.coordinate, 'value': v[:120]})
    wb.close()
    out['sheet_name_hits'] = {k: [s for s in out['sheets'] if rx.search(s)] for k, rx in pats.items()}
    return out


def bib_entry(text, key):
    m = re.search(r'@\w+\{' + re.escape(key) + r',(.*?)\n\}', text, re.S)
    if not m:
        return None
    return {f: v.strip() for f, v in re.findall(r'(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$', m.group(1), re.M)}


def ref_some(src, rx):
    """src.ref for a pattern that must match at least once (every matching line is cited)."""
    hits = src.cite(rx, expect=None)
    if not hits:
        raise KeyError(f'{src.rel}: /{rx}/ matched no line')
    return src.ref(rx, expect=None)


def enclosing_def(src, line_no):
    for i in range(line_no - 1, -1, -1):
        m = re.match(r'\s*def (\w+)\(', src.lines[i])
        if m:
            return m.group(1)
    return None


# ======================================================================================================================
#  2. declarations version 4
# ======================================================================================================================
def retarget(decl, new_id, new_fragment):
    """a version-3 value declaration whose fragment changed but whose evaluation is unchanged: same tokens, same
    function, new fragment; its value-table rows are filed under the new id."""
    old_id, fn = decl['id'], decl['fn']

    def wrapped(X, w):
        out = fn(X, w)
        X.value_rows[new_id] = X.value_rows.pop(old_id)
        return out
    d = dict(decl, id=new_id, fragment=new_fragment, fn=wrapped)
    d['retargeted_from'] = old_id
    return d


def _w174_value_decl(vid):
    hits = [d for d in W174.value_declarations() if d['id'] == vid]
    if len(hits) != 1:
        raise KeyError(f'W174a value declaration {vid}: {len(hits)} hits')
    return hits[0]


def _search_campaigns(X):
    """the planning-search campaigns of record: every committed campaign spec carrying a poll design, and whether a
    committed campaign_results.json sits beside it (a spec superseded before any run has none)."""
    specs = [p for p in subprocess.run(['git', 'ls-files', 'data/SRP1/Results/*campaign_spec*.json'], cwd=REPO,
                                       capture_output=True, text=True).stdout.splitlines() if p]
    out = []
    for p in specs:
        if '"poll_design"' not in _text(p):
            continue
        res = os.path.join(os.path.dirname(p), 'campaign_results.json')
        tracked = bool(subprocess.run(['git', 'ls-files', '--error-unmatch', res], cwd=REPO, capture_output=True).returncode == 0)
        out.append({'spec': p, 'results': res if tracked else None})
    return out


def value_declarations_v4():
    out = []

    def v01(X, w):
        xs, bib, chem = X.xlsx_scan, X.bib_entry or {}, X.chem_in_manuscript
        xi = X.inputs['ESS_XLSX']
        xsrc = f"{xi['path']} (sha256 {xi['sha256'][:8]}, blob {xi['blob'][:8]})"
        li = xs['hits']['li_ion']
        rows = [row([], 'chemistry of the shared ESS', 'a utility-scale lithium-ion battery',
                    [f"'{h['sheet']}'!{h['cell']} = {h['value']!r}" for h in li],
                    [f'{xsrc}: every string cell scanned ({xs["cells_scanned"]} cells, sheets {xs["sheets"]})',
                     jf(X, 'W175_JSON', 'chemistry.xlsx.hits')],
                    'match' if li else 'no source found', None,
                    'the cost workbook names the battery line "Li-Ion Battery cabinet"; no other chemistry term in the '
                    f'workbook ({xs["n_hits"]["other_chemistry"]} hits for LFP / phosphate / LiFePO / NMC). Chemistry terms '
                    f'in the compiled manuscript at this commit (body text, l. : term): {chem}')]
        title = bib.get('title', '')
        ok_nrel = bool(xs['sheet_name_hits']['nrel']) and 'NREL' in bib.get('author', '') and \
            'Utility-Scale Battery Storage' in title
        rows.append(row([], 'cost source: NREL utility-scale battery storage (cited nrel_ess_costs)',
                        'the NREL ... utility-scale battery category~\\cite{nrel_ess_costs}',
                        {'workbook_sheets_named_NREL': xs['sheet_name_hits']['nrel'], 'bib_nrel_ess_costs': bib},
                        [f'{xsrc} sheet names', f'{X.bib_src} @misc{{nrel_ess_costs}}'],
                        'match' if ok_nrel else 'no source found', None,
                        'the investment costs are read from the workbook\'s NREL sheets ("Investment Cost NREL, USD": '
                        '"Total Cost (4-h System)", Low / Mid / High Scenario; W167); the cited entry is NREL, '
                        f'"{title}"'))
        atb = xs['hits']['atb']
        rows.append(row([], "'ATB' (Annual Technology Baseline) as the name of the category", 'NREL ATB',
                        [f"'{h['sheet']}'!{h['cell']} = {h['value']!r}" for h in atb] or None,
                        [f'{xsrc}: string cells and sheet names searched for ATB / "annual technology baseline"',
                         f'{X.bib_src} @misc{{nrel_ess_costs}} fields {sorted(bib)}',
                         f"{W171.jsrc(X, 'ESS_PARAMS')} (no cost-source field)"],
                        'match' if atb or 'ATB' in json.dumps(bib) else 'no source found', None,
                        'scope: the workbook (every string cell, every sheet name), the bibliography entry and the ESS '
                        'parameter file were searched; "ATB" is in none of them. The ATB identity of the cited NREL page '
                        '("Utility-Scale Battery Storage, 2024") is the author\'s citation; the page was not opened '
                        '(W175 likewise). A named setting, not a value.'))
        return rows
    out.append(V('V01v4', 'The shared ESS is a utility-scale lithium-ion battery (the NREL ATB utility-scale battery '
                          'category~\\cite{nrel_ess_costs})', [], v01))

    out.append(retarget(_w174_value_decl('V18'), 'V18v4',
                        'Datasheet 8\\,000 cycles$^{a}$ & (8\\,000, 1.00, 0.70) & 22\\,429 & 1.000 & end & 0.70 \\\\'))
    out.append(retarget(_w174_value_decl('V41'), 'V41v4', 'and reported as the best of three solver starts'))

    def v42(X, w):
        cal = X.ess['ageing']['calibration']
        esrc = W171.jsrc(X, 'ESS_PARAMS')
        ext = W171.jsrc(X, 'EXT_SPEC_V3')
        c4 = W174._ageing_arms(X)['C4']
        cit = [c for c in X.j['SPEC_V18']['citations']['cycle_life'] if '8,000' in c['statement']]
        rows = []
        st, _ = cmp_exact(w[0], cal['cycles_n'])
        no_override = 'cycles_n' not in c4 and 'reference_dod_d' not in c4
        rows.append(row([0], 'C4 as entered: cycles N', 'entered as 10,000 cycles', cal['cycles_n'],
                        [f'{esrc} ageing.calibration.cycles_n', f'{ext} model_variant.arms.C4 = {c4} (no N / D override)'],
                        st if no_override else 'MISMATCH', None, 'the C4 arm varies eol_retention_r only; N and D are the '
                                                                  "file's"))
        st, _ = cmp_exact(w[1], cal['reference_dod_d'])
        rows.append(row([1], 'C4 as entered: depth of discharge D', 'at 0.80 depth of discharge', cal['reference_dod_d'],
                        [f'{esrc} ageing.calibration.reference_dod_d', f'{ext} model_variant.arms.C4 (no override)'],
                        st if no_override else 'MISMATCH'))
        nd_file = cal['cycles_n'] * cal['reference_dod_d']
        cal_txt = cit[0]['calibration'] if len(cit) == 1 else None
        m = re.search(r'cycles (\d+), DoD ([0-9.]+), EOL ([0-9.]+)', cal_txt or '')
        nd_ds = float(m.group(1)) * float(m.group(2)) if m else None
        k_spec = X.ext3['model_variant']['k_closed_form_by_arm']['C4']
        k_file = W174._k_closed(cal['cycles_n'], cal['reference_dod_d'], c4['eol_retention_r'])
        ok = nd_ds is not None and W174._close(nd_file, nd_ds) and W174._close(k_spec, k_file)
        rows.append(row([], 'the same product N delta as the datasheet row, all the calibration uses',
                        'the same product $N^{DS} \\delta^{DS}$, which is all (eq:cycle_life_calibration) uses',
                        {'N_x_D_as_entered': nd_file, 'N_x_D_datasheet_row': nd_ds, 'k_C4_spec': k_spec,
                         'k_from_entered_values': k_file},
                        [f'{esrc} ageing.calibration', jf(X, 'SPEC_V18', 'citations.cycle_life[EVE MB31].calibration') +
                         f': {cal_txt!r}', f'{ext} model_variant.k_closed_form_by_arm.C4',
                         'k = N D / (-ln R) (W174a _k_closed)'],
                        'match' if ok else 'MISMATCH', None,
                        '10,000 x 0.80 = 8,000 x 1.00; k(C4) from the entered values equals the spec\'s k'))
        return rows
    out.append(V('V42', '$^{a}$~entered as 10\\,000 cycles at 0.80 depth of discharge, the same product $N^{\\text{DS}} '
                        '\\delta^{\\text{DS}}$, which is all \\eqref{eq:cycle_life_calibration} uses.',
                 [('10\\,000', 0), ('0.80', 0)], v42))

    def v43(X, w):
        camps = _search_campaigns(X)
        ran = [c for c in camps if c['results']]
        st, _ = cmp_exact(w[0], len(ran))
        rows = [row([0], 'number of planning-search campaigns that ran', 'the three planning-search campaigns',
                    [c['spec'] for c in ran],
                    ['git ls-files data/SRP1/Results/*campaign_spec*.json carrying "poll_design" (HEAD): ' +
                     f'{[c["spec"] for c in camps]}; with a committed campaign_results.json beside: {len(ran)}'],
                    st, None, 'specs carrying a poll design without a committed campaign_results.json beside them: ' +
                    str([c['spec'] for c in camps if not c['results']]))]
        e2 = X.j['W174B_JSON']['E2_search_tail']
        ok = set(e2) == {'s47', 's51', 's53'} and all(
            v['spec_configuration_declares_convergence_depth_tail'] is False and v['spec_text_mentions_tail'] is False
            and all(c == 0 for c in v['tail_code_occurrences_at_head'].values()) for v in e2.values())
        t6 = X.v6['inputs_in_force_now']['configuration_now']['convergence_depth_tail']
        rows.append(row([], 'the search campaigns ran on earlier code states, without the tight tail',
                        'ran on earlier states of the code, without the tight tail',
                        {k: {'git_head_at_run': v['git_head_at_run'][:8], 'tail_code_occurrences': v['tail_code_occurrences_at_head']}
                         for k, v in e2.items()},
                        [jf(X, 'W174B_JSON', 'E2_search_tail'),
                         f"{W171.jsrc(X, 'V6_SPEC')} inputs_in_force_now.configuration_now.convergence_depth_tail = {t6}"],
                        'match' if ok and t6.get('enabled') is True else 'MISMATCH', None,
                        'at each campaign\'s run head the tail code occurs 0 times in shared_resources_planning.py, '
                        'admm_parameters.py and the campaign harness, and no spec declares it; the frozen configuration of '
                        'the certified evaluations (v6) enables it'))
        return rows
    out.append(V('V43', 'the three planning-search campaigns that proposed the incumbents ran on earlier states of the code, '
                        'without the tight tail, and their incumbents were re-evaluated under this configuration before '
                        'being reported', [('three', 0)], v43))

    def v44(X, w):
        spec, rep, w175 = X.j['BENCH_SPEC_V5'], X.j['BENCH_REPORT_V3'], X.j['W175_JSON']['benchmark']
        defs, mch, ub = X.c('definitions'), X.c('model_construction_helpers'), X.c('uncoordinated_benchmark')
        cc, cc3 = spec['consistency_convention'], spec['consistency_convention_v3']
        ok_c = ("each DN re-evaluated at the TN's actual interface voltage" in cc and 'one sequential pass' in cc and
                'if a DN limit is violated' in cc and 'a violation triggers the one sequential pass' in cc3 and
                "the DSO re-solves with the rows at the TN's actual voltage" in cc3)
        trig = {r: v['phase_B_trigger_sequential_pass'] for r, v in w175['nrf_runs'].items()}
        rows = [row([], 'consistency pass', "each DN is then re-evaluated at the TN's interface voltage and, where a DN "
                    'limit is violated, re-solved once at that voltage before the TSO dispatches again',
                    {'consistency_convention': cc, 'consistency_convention_v3': cc3},
                    [jf(X, 'BENCH_SPEC_V5', 'consistency_convention'), jf(X, 'BENCH_SPEC_V5', 'consistency_convention_v3'),
                     jf(X, 'W175_JSON', 'benchmark.nrf_runs[*].phase_B_trigger_sequential_pass') + f' = {trig}'],
                    'match' if ok_c else 'MISMATCH', None,
                    'one sequential pass (48 solves: the DSOs re-solved, then the TSO) defines the arm cost if a DN limit '
                    'is violated; no magnitude of the pass is printed in the paragraph')]
        tb, tbr = spec['tie_breaker'], rep['tie_breaker']
        pen = defs.value(r'^PENALTY_GENERATION_CURTAILMENT = ([0-9.e+-]+)')
        ok_t = tb['decision']['passive_dso'] == tbr['decision']['passive_dso'] == pen
        st, _ = cmp_exact(w[0], tb['decision']['passive_dso'])
        rows.append(row([0], 'passive DSO minimum-curtailment term (EUR/MWh)', 'a minimum-curtailment term of 1 EUR/MWh',
                        tb['decision']['passive_dso'],
                        [jf(X, 'BENCH_SPEC_V5', 'tie_breaker.decision.passive_dso'),
                         jf(X, 'BENCH_REPORT_V3', 'tie_breaker.decision.passive_dso'),
                         defs.ref(r'^PENALTY_GENERATION_CURTAILMENT = '),
                         mch.ref(r'gen_curt_penalty \+= penalty \* network\.baseMVA \* \(model\.pg_avail\[g, s_o, p\] - '
                                 r'model\.pg\[g, s_m, s_o, p\]\)'),
                         ref_some(ub, r'block\.penalty_gen_curtailment\.set_value\(float\(curtailment_penalty\)\)')],
                        st if ok_t else 'MISMATCH', None,
                        'unit: the penalty multiplies baseMVA x (available - dispatched) p.u. per hourly period, i.e. EUR '
                        'per MWh curtailed; decision tie-breaker of the price-taker DSOs and the TSO 0 '
                        f"({tb['decision']})"))
        oc = spec['objective_convention']
        g = w175['common_q_gate']
        ok_q = (oc.startswith('Q = gross_operational_cost, settlement EXCLUDED') and tb['evaluation'] == 0.0 and
                tbr['evaluation'] == 0.0 and g['status'] == 'PASS_BITWISE' and g['evaluation_curtailment_penalty'] == 0.0)
        rows.append(row([], 'arm cost = the common Q, settlement excluded, minimum-curtailment term at zero',
                        'every arrangement is costed with the same function $Q$ as the coordinated value (settlement '
                        'excluded, that term at zero)',
                        {'objective_convention': oc, 'evaluation_tie_breaker': [tb['evaluation'], tbr['evaluation']],
                         'common_q_gate': g['status']},
                        [jf(X, 'BENCH_SPEC_V5', 'objective_convention'), jf(X, 'BENCH_SPEC_V5', 'tie_breaker.evaluation'),
                         jf(X, 'BENCH_REPORT_V3', 'tie_breaker.evaluation'), jf(X, 'W175_JSON', 'benchmark.common_q_gate')],
                        'match' if ok_q else 'MISMATCH'))
        return rows
    out.append(V('V44', "each DN is then re-evaluated at the TN's interface voltage and, where a DN limit is violated, "
                        're-solved once at that voltage before the TSO dispatches again. The passive DNs select their '
                        'curtailment by a minimum-curtailment term of 1~\\euro/MWh; every arrangement is costed with the '
                        'same function $Q$ as the coordinated value (settlement excluded, that term at zero)',
                 [('1', 0)], v44))

    def v45(X, w):
        spec, rep = X.j['BENCH_SPEC_V5'], X.j['BENCH_REPORT_V3']
        inst, rinst, t6 = spec['instance'], rep['instance'], X.t['benchmark']['instance']
        zero = all(list(v) == [0.0, 0.0] for v in inst['candidate'].values())
        nets = [X.srp1['TransmissionNetwork']] + list(X.srp1['DistributionNetworks'])
        n_op = sorted({n['num_operation_scenarios'] for n in nets})
        single = X.srp1['NumMarketScenarios'] == 1 and n_op == [1]
        ok = (zero and inst['problem'] == 'SRP1' and inst['label'] == 'x0' and rinst['candidate_key_matches'] is True
              and rinst['candidate_key'] == inst['candidate_key'] == t6['candidate_key'] and single)
        return [row([], 'benchmark instance', 'evaluated at the plan without shared storage, at the single-scenario instance',
                     {'candidate': inst['candidate'], 'label': inst['label'], 'problem': inst['problem'],
                      'NumMarketScenarios': X.srp1['NumMarketScenarios'], 'num_operation_scenarios': n_op},
                     [jf(X, 'BENCH_SPEC_V5', 'instance'), jf(X, 'BENCH_REPORT_V3', 'instance'),
                      'T6 tables.benchmark.instance', f"{X.src('SRP1_JSON')} NumMarketScenarios, *.num_operation_scenarios"],
                     'match' if ok else 'MISMATCH', None, 'x = 0 (every node (P, E) = (0, 0)) on SRP1, one market and one '
                                                          'operation scenario per network')]
    out.append(V('V45', 'The benchmark is evaluated at the plan without shared storage, at the single-scenario instance.',
                 [], v45))
    return out


def sec2_declarations_v4():
    def s401(X, w):
        term = {k: X.j[f'{k}_RESULTS']['termination'] for k in ('S47', 'S51', 'S53')}
        unit = sorted(k for k, t in term.items() if t['reason'] in ('mesh_local_optimum_unit_poll_failed',
                                                                    'poll_failure_at_unit_mesh') and t.get('poll_size_reached') == 1)
        cap = sorted(k for k, t in term.items() if t['reason'] == 'STOP_FOR_REVIEW_completion_cap'
                     and t.get('completion_n_feasible', 0) > t.get('cap', float('inf')))
        src = '; '.join(jf(X, f'{k}_RESULTS', 'termination') + f" = {{reason {t['reason']}, poll_size_reached "
                        f"{t.get('poll_size_reached')}" + (f", completion_n_feasible {t.get('completion_n_feasible')}, cap "
                                                           f"{t.get('cap')}" if 'cap' in t else '') + '}'
                        for k, t in term.items())
        return [count_eq3(w[0], len(term) if len(unit) + len(cap) == len(term) else -1, src, 'committed record',
                          'the three search runs: s47 (variant A, baseline), s51 (variant A, doubled flexibility price), s53 '
                          '(variant B certificate)'),
                count_eq3(w[1], len(unit), src, 'committed record', f'stopped when a unit poll failed: {unit}'),
                count_eq3(w[2], len(cap), src, 'committed record', f'stopped for review at the completion cap: {cap}')]
    return [D('S401', W174.MAIN, 'of the three runs reported here, two stopped when a unit poll failed and one was stopped '
                                 'for review when its completion set exceeded the cap',
              [('three', 0), ('two', 0), ('one', 0)], s401)]


def appendix_a_declarations_v4():
    def a401(X, w):
        srp, mch = X.c('shared_resources_planning'), X.c('model_construction_helpers')
        sites = srp.cite(r'interface_settlement_weight\.set_value\(', expect=None)
        ones = [s for s in sites if re.search(r'set_value\(1\.00\)', s[1])]
        funcs = sorted({enclosing_def(srp, ln) for ln, _ in ones})
        init = mch.value(r'model\.interface_settlement_weight = pe\.Param\(initialize=([0-9.]+), mutable=True\)')
        ok = (len(ones) == len(sites) == 2 and funcs == ['_prepare_distribution_objectives_for_admm',
                                                         '_prepare_transmission_objectives_for_admm'] and init == 0.0)
        src = '; '.join([srp.ref(r'interface_settlement_weight\.set_value\(1\.00\)', expect=None) + f' (in {funcs})',
                         mch.ref(r'model\.interface_settlement_weight = pe\.Param\(initialize=0\.00, mutable=True\)')])
        return R('match' if ok and wv3(w[0]) == 1 else 'MISMATCH', src, 'code (named record)', 1, 'one',
                 'the settlement weight is built at 0 and set to 1 for every TSO and DSO block by the two ADMM-conversion '
                 f'functions; every set_value site of the weight: {[s[0] for s in sites]}')
    return [D('A401', W174.MAIN, "the interface settlement's weight is set to one here and, in the multi-scenario instance, "
                                 'the commitment charge is activated', [('one', 0)], a401)]


def all_declarations_v4():
    return value_declarations_v4() + sec2_declarations_v4() + appendix_a_declarations_v4()


# v3 declarations whose fragment is still found but which are re-pointed in version 4 (reason)
SUPERSEDED_V4 = {}
# v3 declarations whose fragment is gone: the version-4 declaration that replaced it, or why no number is left
REPLACED_BY_V4 = {
    'V01': ('V01v4', 'section 3.5 chemistry (Addendum 71 item 7, Addendum 72; round 4): "a utility-scale lithium iron '
                     'phosphate battery" -> "a utility-scale lithium-ion battery (the NREL ATB utility-scale battery '
                     'category~\\cite{nrel_ess_costs})"; named-setting rows, no number'),
    'V18': ('V18v4', 'the C4 row of the ageing table gains the footnote mark: "Datasheet 8\\,000 cycles &" -> "Datasheet '
                     '8\\,000 cycles$^{a}$ &" (Addendum 71 item 8); same tokens, same evaluation (W174a V18 retargeted); the '
                     'footnote itself is V42'),
    'V41': ('V41v4', 'benchmark paragraph rewritten (Addendum 71 item 5; round 4): "Each arrangement is reported as the best '
                     'of three solver starts." -> "... every arrangement is costed with the same function $Q$ ... and '
                     'reported as the best of three solver starts"; same evaluation (W174a V41 retargeted); the new '
                     'sentences are V44 / V45'),
}


# ======================================================================================================================
#  3. main.tex line rules v4 (re-pinned regions) and rules
# ======================================================================================================================
def line_map(old_lines, new_lines):
    sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    lm = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                lm[i1 + k + 1] = j1 + k + 1
    return lm


def map_range(lm, a, b):
    s, e1 = lm.get(a), lm.get(b + 1)
    return (s, e1 - 1) if s is not None and e1 is not None else None


def map_regions_v4(lm, kept_v3):
    """W174a's submitted-version regions mapped into this main.tex: kept iff every line lies in an unchanged block and the
    mapped lines are consecutive."""
    kept, dropped = [], []
    for (a, b), key, kind, why in kept_v3:
        ms = [lm.get(x) for x in range(a, b + 1)]
        ok = all(m is not None for m in ms) and all(ms[i + 1] == ms[i] + 1 for i in range(len(ms) - 1))
        if ok:
            kept.append(((ms[0], ms[-1]), key, kind, why))
        else:
            dropped.append({'old': [a, b], 'key': key, 'n_lines_changed': sum(m is None for m in ms),
                            'reason': REGION_DROPPED.get(key, 'TEXT CHANGED (no reason declared)')})
    return tuple(kept), dropped


def region_of(ln):
    for key, (a, b), *_ in REVISED:
        if a <= ln <= b:
            return key
    return None


def rule_assign_main_v4(X, doc, t, map_xref_numbers):
    """W174a's rule_assign_main_v3 with the version-4 line rules (REVISED, the re-pinned regions, the network tables and
    Appendices B-D of this main.tex); the rules themselves are unchanged."""
    typ, ln, txt = t['type'], t['line'], t['text']
    line = doc.lines[ln - 1]
    reg = region_of(ln)
    if t['scope'] == 'comment':
        sub_ = 'template boilerplate' if ln <= 60 or line.lstrip().startswith('%%') else \
            ('instruction comment in a revised passage (' + reg + ')') if reg else 'commented-out text'
        return unch('comment (not typeset)', f'{sub_}; comments are not manuscript text')
    lines_ok = X.main_sha == MAIN_LINE_RULES_SHA
    if lines_ok and reg is None:
        for (a, b), key, kind, why in X.main_regions_v4:
            if a <= ln <= b:
                return W164._map_sub(X, key, kind, why)
    if typ == 'doi':
        return unch('reference identifier', 'DOI (reference-list identifier)')
    if typ == 'label':
        if re.fullmatch(r'R\d\.\d+', txt):
            return unch('enumerator', 'reviewer item label')
        if re.fullmatch(r'T\d+', txt):
            return unch('enumerator', 'table identifier of the frozen tables')
        if re.fullmatch(r'C\d', txt):
            return unch('enumerator', 'ageing calibration name')
        return unch('identifier', 'label containing digits')
    if typ in ('hash', 'date', 'alnum'):
        return unch('notation' if reg else 'identifier', f'{typ} in a revised passage (notation)' if reg else f'{typ} in the submitted text')
    if typ in ('num', 'dotted') and W171._in_xref(doc, t):
        return unch('cross-reference', 'section / table / figure / equation / algorithm number written in prose' +
                    W171._xref_note(X, doc, t))
    if typ == 'word' and W164._compound(doc, t):
        return unch('not a figure', f'compound adjective ({txt}{line[t["end0"]:t["end0"] + 12].split()[0]})')
    if typ == 'word' and txt.lower() == 'single':
        return unch('not a figure', 'article sense ("a single ...")')
    if reg:
        where = dict((k, d) for k, _r, d, _h in REVISED)[reg]
        if typ == 'word':
            return unch('method statement', f'number word in {where} describing structure (no value to check; the statements '
                        'are audited against the code by W171b / W172 / W174b / W176a)')
        if 'algorithm' in t['envs'] or t['in_math']:
            return unch('notation', f'formula constant in {where} (an index offset, exponent, bound or coefficient of the printed '
                        'formula; the equations and algorithms are audited against the code by W171b / W172 / W174b)')
        return None
    if typ == 'num' and re.fullmatch(r'(19|20)\d\d', txt):
        if txt in X.years():
            return R('match', f'{X.src("SRP1_JSON")} Years {sorted(X.years())}', 'named record', sorted(X.years()), txt,
                     'a representative year of the revised instance')
        return W164._map_sub(X, 'years', 'input replaced', 'non-representative year of the submitted horizon')
    if typ == 'num' and re.search(r'IEEE\s*$', line[:t['col0']]) and line[t['end0']:t['end0'] + 4] == '-bus':
        return W164._bus(X, 'case9', t['written']) if txt == '9' else W164._bus(X, 'case33', t['written']) if txt == '33' else None
    if typ == 'num' and re.search(r'[Nn]odes?(?:~|\s)*(?:\d+\s*,\s*(?:and\s*)?|\d+\s*,?\s*and\s*)*$', line[:t['col0']]) \
            and 'tabular' not in ' '.join(t['envs']):
        ids = sorted(d['connection_node_id'] for d in X.srp1['DistributionNetworks'])
        return R('match' if int(float(txt)) in ids else 'MISMATCH', f'{X.src("SRP1_JSON")} DistributionNetworks[*].connection_node_id '
                 f'{ids}', 'named record', ids, txt, 'TN interface node of a distribution network (an identifier)')
    if 'postcode=' in line:
        return unch('address', 'postcode of an affiliation')
    if not lines_ok:
        return None
    if 'algorithm' in t['envs']:
        return unch('notation', f'constant in an algorithm (audited: {W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appA"])})')
    if t['in_math']:
        return unch('notation', 'formula constant (equations are audited against the code, W166 / W171b / W172)')
    if any(e.startswith('tabular') for e in t['envs']):
        if NETWORK_TABLES[0] <= ln <= NETWORK_TABLES[1]:
            return unch('network/data parameter', 'case-data table (RES capacity / generator and node IDs), outside the frozen '
                        f'tables ({W164.MAP_REL} line {X.map_line(W164.MAP_CITES["years"])})')
        if APP_BCD[0] <= ln <= APP_BCD[1]:
            return unch('network/data parameter', 'IEEE 33-bus / market data table, outside the frozen tables '
                        f'({W164.MAP_REL} line {X.map_line(W164.MAP_CITES["appBCD"])}: keep)')
    if typ == 'word':
        if txt.lower() == 'third':
            return unch('enumerator', 'ordinal word')
        return unch('descriptive count', 'number word in unrevised submitted prose (counts of lists, layers, agents, reference '
                    'cases); wording checks are not required this round')
    return None


def rule_assign_v4(X, doc, role, t, map_xref_numbers, qstat):
    if role == 'main':
        return rule_assign_main_v4(X, doc, t, map_xref_numbers)
    if t['type'] == 'word' and t.get('tokenizer') == 'v3 extra number word' and t['scope'] == 'comment':
        return unch('comment (not typeset)', 'number word in a comment')
    return W171.rule_assign_v2(X, doc, role, t, map_xref_numbers, qstat)


# ======================================================================================================================
#  4. findings and reconciliations
# ======================================================================================================================
def chem_terms(doc):
    out = []
    for i, ln in enumerate(doc.lines):
        body, com, _j = W164.split_comment(ln)
        for scope, seg in (('body', body), ('comment', com)):
            for m in CHEM_TERMS.finditer(seg or ''):
                out.append({'line': i + 1, 'scope': scope, 'term': m.group(), 'context': seg[max(0, m.start() - 40):m.end() + 40]})
    return out


def letter_findings_v4(X, docs, qrows):
    lf = W174.letter_findings_v3(X, docs, qrows)
    letter = docs[LETTER]
    want = ('the reference technology is utility-scale lithium-ion battery storage; the cycling calibrations are datasheet '
            'readings (Section~3.5)')
    hits, _n = letter.find_fragment(want)
    ch = chem_terms(letter)
    lf.append({'id': 'G9', 'lines': str(letter.flat_idx[hits[0]][0]) if len(hits) == 1 else '-', 'kind': 'Addendum 72 wording',
               'finding': f'R1.2(iv) reads "{want}" ({len(hits)} occurrence); chemistry terms in the letter: '
                          f'{[(c["line"], c["scope"], c["term"]) for c in ch]}. '
                          + ('No "LFP" / "iron phosphate" left. ' if not any(re.search(r'LFP|phosphate', c['term'], re.I) for c in ch)
                             else 'LFP / phosphate STILL PRESENT. ') +
                          'Section~3.5 is the shared-ESS parameters subsection of this main.tex (cross-reference rule).'})
    return lf


def main_findings_v4(X):
    doc = X.main
    heads = {h['number']: (h['title'], h['line']) for h in doc.heads if h['number']}
    labels = {}
    for i, ln in enumerate(doc.lines):
        for m in re.finditer(r'\\label\{(sec:case_[a-z_]+)\}', ln):
            labels[m.group(1)] = i + 1
    sec_of_label = {}
    for lab, ln in labels.items():
        prev = [h for h in doc.heads if h['line'] <= ln]
        sec_of_label[lab] = (prev[-1]['number'], prev[-1]['title']) if prev else None
    sec2 = REVISED[0][1]
    refs = []
    for i, ln in enumerate(doc.lines):
        if sec2[0] <= i + 1 <= sec2[1]:
            for m in re.finditer(r'given in Section~\\ref\{(sec:case_[a-z_]+)\}', ln):
                refs.append({'line': i + 1, 'label': m.group(1), 'section': sec_of_label.get(m.group(1)),
                             'context': ln[max(0, m.start() - 260):m.start()][-200:]})
    syms = ('\\eta', 'SoC^{\\text{Min}}', '\\varepsilon^{\\text{Cl}}', 'c^{\\sigma}', '\\varepsilon^{\\text{E}}', '\\alpha')
    printed_in = {}
    for s in syms:
        rx = re.escape(s) + r'(?:\^\{\\text\{[A-Za-z]+\}\})?\s*='
        hit = [i + 1 for i, ln in enumerate(doc.lines) if VALUE_TABLE_LINES[0] <= i + 1 <= VALUE_TABLE_LINES[1] and re.search(rx, ln)]
        printed_in[s] = hit
    wrong = []
    for r in refs:
        names = [s for s in syms if s in r['context']]
        where = sorted({(sec_of_label['sec:case_ess_params'][0] if any(h < labels.get('sec:case_settings', 10 ** 9)
                                                                       for h in printed_in[n])
                         else sec_of_label['sec:case_settings'][0]) for n in names if printed_in[n]})
        r['symbols'], r['values_printed_in'] = names, where
        if names and r['section'] and where and [r['section'][0]] != where:
            wrong.append(r)
    lit = [(i + 1, m.group()) for i, ln in enumerate(doc.lines) if region_of(i + 1) in ('sec2', 'sec3_5_6', 'app_a')
           for m in re.finditer(r'Section~(?:[34]\.\d+|4)\b', ln) if W174.split_ok(ln, m.start())]
    sec34 = {k: v[0] for k, v in heads.items() if re.fullmatch(r'[34]\.\d', k)}
    ch = chem_terms(doc)
    confirm = [i + 1 for i, ln in enumerate(doc.lines) if '[CONFIRM' in ln or '[AUTHOR' in ln]
    return [
        {'id': 'M1', 'lines': ', '.join(str(r['line']) for r in refs) or '-', 'kind': 'cross-references to sections 3.5 / 3.6',
         'finding': ('every "given in Section~\\ref{sec:case_...}" in section 2 points at the subsection that prints the '
                     'values: ' + '; '.join(f"l. {r['line']} -> {r['label']} ({r['section'][0] if r['section'] else None}) for "
                                           + (f"{', '.join(r['symbols'])} (printed in {r['values_printed_in']})" if r['symbols']
                                              else 'no parameter symbol of the list in the sentence (not scored)')
                                           for r in refs) +
                     '. W174a M1 (l. 756 / 866 -> sec:case_settings) RESOLVED (Addenda 71-72).')
         if not wrong else 'WRONG SUBSECTION: ' + '; '.join(f"l. {r['line']} {r['label']} for {r['symbols']} printed in "
                                                            f"{r['values_printed_in']}" for r in wrong)},
        {'id': 'M2', 'lines': ', '.join(str(x[0]) for x in lit) or '-', 'kind': 'literal section numbers',
         'finding': f'literal "Section~3.x / 4 / 4.x" in sections 2, 3.5-3.6 and Appendix A: {lit}; sections 3.x and 4.x at this '
                    f'commit: {sec34} (section 4 is still the submitted text; the literals are on the round-4 final-pass list)'},
        {'id': 'M3', 'lines': ', '.join(sorted({str(c['line']) for c in ch})) or '-', 'kind': 'chemistry terms (Addendum 72)',
         'finding': f'chemistry terms in main.tex: {[(c["line"], c["scope"], c["term"]) for c in ch]}; '
                    + ('no "LFP" / "iron phosphate" left' if not any(re.search(r'LFP|phosphate', c['term'], re.I) for c in ch)
                       else 'LFP / phosphate STILL PRESENT')},
        {'id': 'M4', 'lines': ', '.join(map(str, confirm)) or '-', 'kind': '[CONFIRM / [AUTHOR comments',
         'finding': f'{len(confirm)} line(s) carry "[CONFIRM" or "[AUTHOR" in main.tex at this commit'},
    ]


def reconcile(old_tokens, new_tokens, lm, status, file_=MAIN):
    """tokens of `status` in W174a's committed results vs this run, matched by (mapped line, column, text); a token whose
    line was edited (no mapped line) is matched in a second pass only by declaration lineage: same column, same text,
    and the W176 check is the W174a check itself or its declared replacement (REPLACED_BY_V4)."""
    new = {(t['line'], t['col'], t['written']): t for t in new_tokens if t['file'] == file_ and t.get('status') == status}
    used, same, removed = set(), [], []
    for t in old_tokens:
        if t['file'] != file_ or t.get('status') != status:
            continue
        nl = lm.get(t['line'])
        k = (nl, t['col'], t['written'])
        lineage = None
        if nl is None and t.get('check_id'):
            heirs = {t['check_id'], REPLACED_BY_V4.get(t['check_id'], (None,))[0]}
            cand = [kk for kk, nt in new.items() if kk not in used and kk[1] == t['col'] and kk[2] == t['written']
                    and nt.get('check_id') in heirs]
            if len(cand) == 1:
                k, lineage = cand[0], f"line edited; same column and text; check {t['check_id']} -> {new[cand[0]].get('check_id')}"
        if (nl is not None or lineage) and k in new and k not in used:
            used.add(k)
            same.append({'old_line': t['line'], 'new_line': k[0], 'col': t['col'], 'written': t['written'],
                         'check_old': t.get('check_id'), 'check_new': new[k].get('check_id'),
                         'section': (t.get('section') or '').split(' > ')[0],
                         'match_basis': lineage or 'mapped line (unchanged), same column and text'})
        else:
            o = {'old_line': t['line'], 'new_line': nl, 'col': t['col'], 'written': t['written'], 'check_old': t.get('check_id'),
                 'section': (t.get('section') or '').split(' > ')[0], 'old_region': W174.region_of(t['line'])}
            if nl is None:
                o['why'] = 'the line changed between 260bd83 and 8b76423'
            else:
                nt = [x for x in new_tokens if x['file'] == file_ and x['line'] == nl and x['col'] == t['col']]
                o['why'] = f"line unchanged; token now {nt[0]['status_display'] if nt else 'absent'}"
            removed.append(o)
    added = [{'new_line': t['line'], 'col': t['col'], 'written': t['written'], 'check_new': t.get('check_id'),
              'section': (t.get('section') or '').split(' > ')[0], 'region': region_of(t['line']) if file_ == MAIN else None}
             for k, t in new.items() if k not in used]
    by_sec_old = collections.Counter(x['section'] for x in same + removed)
    by_sec_new = collections.Counter(x['section'] for x in same + added)
    return {'status': status, 'n_w174a': len(same) + len(removed), 'n_w176': len(same) + len(added), 'n_same': len(same),
            'removed': removed, 'added': added, 'same': same, 'by_section_w174a': dict(by_sec_old),
            'by_section_w176': dict(by_sec_new)}


# ======================================================================================================================
#  5. guards, post-run, MD
# ======================================================================================================================
def guards_state():
    guards = {nm: {'counts': dict(g.counts), 'verify_0_failures': g.verify(0)} for nm, g in GUARDS}
    base = W160.pickle_state()
    counts = dict(base['counts'], w161=dict(W161.PICKLE_COUNTS), w162=dict(W162.PICKLE_COUNTS), w163=dict(W163.PICKLE_COUNTS),
                  w164=dict(W164.PICKLE_COUNTS), w171=dict(W171.PICKLE_COUNTS), w174=dict(W174.PICKLE_COUNTS),
                  w176=dict(PICKLE_COUNTS))
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    pk = {'counts': counts, 'pickle_load_and_loads_blocked': blocked,
          'ok': blocked and all(v == {'load': 0, 'loads': 0} for v in counts.values())}
    return guards, pk, all(not v['verify_0_failures'] for v in guards.values()) and pk['ok']


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('log', 'man', 'json', 'md', 'typing_json', 'typing_log')]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W176 post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W176 post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}: " + ', '.join(f'{k} {v[:8]}' for k, v in man.items()))
    sys.exit(0)


def md_summary(o):
    s = o['summary']
    p = o['prediction']
    L = ['# W176b -- number check of the manuscript .tex files at Overleaf 8b76423 (declarations v4)', '',
         f"Overleaf clone `{o['manuscript']['dir']}` at commit `{o['manuscript']['commit']}` (declared "
         f"`{o['manuscript']['declared_commit']}`); files: " +
         ', '.join(f"`{f['name']}` sha256 `{f['sha256']}`" for f in o['manuscript']['files']) + '.',
         f"Frozen tables `{os.path.basename(W164.FZ_REL)}` (sha256 `{W164.FZ_SHA[:8]}`). Script `{SCRIPT_REL}` (imports "
         f"`{W174_SCRIPT}` @ {W174_SCRIPT_COMMIT} and through it W171a, W164-W160, none edited). Compared with W174a's "
         f"committed results `{W174_RESULTS}` ({W174_RESULTS_COMMIT}). Reviewers' document `{W171.DOCX}` (sha256 "
         f"`{W171.DOCX_SHA[:8]}`). ZERO SOLVES (guards verified 0), pickle blocked. Nothing in the clone is edited.", '',
         'Statuses as W174a: match (declared / rule / auto), MISMATCH, approximate (section 3.5-3.6 value rows only: the '
         'printed figure is the source rounded at the printed precision), no table counterpart, submitted-version figure, '
         'reviewer quotation, verified, unchecked (with the reason), no source found (section 3.5-3.6 rows only). Excluded '
         'LaTeX structure is counted separately.', '',
         '## Recorded prediction (expert, Addendum 72), item by item', '', f"Statement: {p['statement']}", '',
         '| item | W174a (260bd83) | W176 (8b76423) | outcome |', '|---|---|---|---|']
    for it in p['items']:
        L.append(f"| {it['item']} | {_cell(it['w174a'], 120)} | {_cell(it['w176'], 120)} | **{it['outcome']}** |")
    L += ['', '## Counts per file -- manuscript', '']
    hdr = ('| file | tokens | excluded | in scope (body) | comments | match declared | match rule | match auto-unique | '
           'match auto-ambiguous | MISMATCH | approximate | no table counterpart | submitted-version | reviewer quotation, '
           'verified | unchecked | no source found | unassigned |')
    sep = '|---|' + '---:|' * 16

    def frow(f, c):
        return (f"| {f} | {c['tokens']} | {c['excluded']} | {c['in_scope_body']} | {c['in_scope_comment']} | "
                f"{c['match_declared']} | {c['match_rule']} | {c['match_auto_unique']} | {c['match_auto_ambiguous']} | "
                f"{c['MISMATCH']} | {c['approximate']} | {c['no table counterpart']} | {c['submitted-version figure']} | "
                f"{c['reviewer quotation, verified']} | {c['unchecked']} | {c['no source found']} | {c['unassigned']} |")
    L += [hdr, sep] + [frow(f, c) for f, c in s['per_file'].items() if f != DRAFT]
    L += ['', '## Counts -- draft, not compiled (`section2_expert_draft.tex`; apart from the manuscript counts)', '', hdr, sep] + \
        [frow(f, c) for f, c in s['per_file'].items() if f == DRAFT]
    L += ['', f"Every token assigned: **{s['every_token_assigned']}**; stale declarations: {s['stale_declarations']}; tokens "
              f"claimed twice: {s['claimed_twice']}.", '']
    L += ['## MISMATCH (all files)', '', '| file | line | col | written | counterpart at written precision | source | note |',
          '|---|---:|---:|---|---|---|---|']
    mm = [t for t in o['tokens'] if t.get('status') == 'MISMATCH']
    for t in mm:
        L.append(f"| {t['file']}{' (draft)' if t['file'] == DRAFT else ''} | {t['line']} | {t['col']} | {t['written']} | "
                 f"{_cell(t.get('value_at_written_precision'))} | {_cell(t.get('counterpart'), 300)} | {_cell(t.get('note'), 400)} |")
    if not mm:
        L.append('| none | | | | | | |')
    ap = o['reconciliation']['approximate']
    L += ['', '## Approximate tokens against W174a', '',
          f"W174a {ap['n_w174a']}, W176 {ap['n_w176']}, same token (mapped line, or declaration lineage on an edited line; same column and text) {ap['n_same']}, removed {len(ap['removed'])}, "
          f"added {len(ap['added'])}.", '', '| 260bd83 l. | 8b76423 l. | col | written | W174a check | W176 check | matched by | note |',
          '|---:|---:|---:|---|---|---|---|---|']
    notes = {(t['line'], t['col']): t.get('note') for t in o['tokens'] if t['file'] == MAIN and t.get('status') == 'approximate'}
    for x in ap['same']:
        L.append(f"| {x['old_line']} | {x['new_line']} | {x['col']} | {x['written']} | {x['check_old']} | {x['check_new']} | "
                 f"{x['match_basis']} | {_cell(notes.get((x['new_line'], x['col'])), 200)} |")
    for x in ap['removed']:
        L.append(f"| {x['old_line']} | {x['new_line']} | {x['col']} | {x['written']} | {x['check_old']} | REMOVED | - | {x['why']} |")
    for x in ap['added']:
        L.append(f"| - | {x['new_line']} | {x['col']} | {x['written']} | - | ADDED {x['check_new']} | - | "
                 f"{_cell(notes.get((x['new_line'], x['col'])), 200)} |")
    sv = o['reconciliation']['submitted_version']
    L += ['', '## Submitted-version figures against W174a (main.tex)', '',
          f"W174a {sv['n_w174a']}, W176 {sv['n_w176']}, same (mapped line, column, text) {sv['n_same']}, removed "
          f"{len(sv['removed'])}, added {len(sv['added'])}.", '', '| section | W174a | W176 |', '|---|---:|---:|']
    for sec in sorted(set(sv['by_section_w174a']) | set(sv['by_section_w176'])):
        L.append(f"| {sec} | {sv['by_section_w174a'].get(sec, 0)} | {sv['by_section_w176'].get(sec, 0)} |")
    L += ['', f"Regions changed: {o['reconciliation']['submitted_regions_note']}", '']
    if sv['removed'] or sv['added']:
        L += ['| change | 260bd83 l. | 8b76423 l. | col | written | section | why |', '|---|---:|---:|---:|---|---|---|']
        for x in sv['removed']:
            L.append(f"| removed | {x['old_line']} | {x['new_line']} | {x['col']} | {x['written']} | {x['section']} | {x['why']} |")
        for x in sv['added']:
            L.append(f"| added | - | {x['new_line']} | {x['col']} | {x['written']} | {x['section']} | region {x['region']} |")
    L += ['', '## Version-3 declarations re-located (version 3 -> version 4)', '',
          f"Outcomes: {s['relocation']}. Carried declarations are evaluated unchanged; the others:", '',
          '| v3 id | file | fragment found (times) | outcome | v4 declaration | reason / what replaced it |', '|---|---|---:|---|---|---|']
    for r in o['relocation']:
        if r['outcome'] != 'carried':
            L.append(f"| {r['id']} | {r['file']} | {r['hits']} | {r['outcome']} | {r.get('v4') or ''} | {_cell(r.get('reason'), 400)} |")
    L += ['', '## Version-4 declarations (new numbers of round 4)', '', '| id | file | line | tokens | statuses |', '|---|---|---:|---|---|']
    for d in o['declarations_applied']:
        if d['version'] == 4:
            sts = [next((t['status'] for t in o['tokens'] if t['file'] == d['file'] and t['line'] == x[0] and t['col'] == x[1]), None)
                   for x in d['tokens']]
            L.append(f"| {d['id']} | {d['file']} | {d['line']} | {', '.join(f'{x[2]} (l. {x[0]})' for x in d['tokens']) or '-'} | "
                     f"{', '.join(map(str, sts)) or 'named settings only'} |")
    vt = o['value_table']
    L += ['', f'## Sections 3.5-3.6 (main.tex l. {VALUE_TABLE_LINES[0]}-{VALUE_TABLE_LINES[1]}): every value and named setting', '',
          f"Rows: {len(vt)} -- " + ', '.join(f'{k} {v}' for k, v in s['value_table_status'].items()) +
          f" (values: {s['value_table_status_values']}; named settings: {s['value_table_status_settings']}).", '',
          f"Not rows: {o['value_table_not_rows']}", '',
          '| id | line | printed | quantity | value in force | source | status | precision / note |', '|---|---:|---|---|---|---|---|---|']
    for r in vt:
        L.append(f"| {r['id']} | {r['line']} | {_cell(r['printed'], 120)} | {_cell(r['quantity'], 90)} | "
                 f"{_cell(r['value_in_force'], 160)} | {_cell('; '.join(r['sources']), 520)} | **{r['status']}** | "
                 f"{_cell(' '.join(x for x in (r['precision'], r['note']) if x), 400)} |")
    L += ['', '### Tokens in the section 3.5-3.6 range that are not values (assigned by rule)', '',
          '| line | written | status | reason |', '|---:|---|---|---|']
    for t in o['value_range_other_tokens']:
        L.append(f"| {t['line']} | {t['written']} | {t['status']} | {_cell(t['reason'], 200)} |")
    L += ['', '## Round-4 lines: every token on a main.tex or letter line changed since W174a (260bd83)', '',
          '| file | line | written | status | check | source / reason |', '|---|---:|---|---|---|---|']
    for t in o['tokens']:
        if not t.get('excluded') and t['line'] in o['changed_lines'].get(t['file'], []):
            L.append(f"| {t['file']} | {t['line']} | {t['written']} | {t['status_display']} | "
                     f"{t.get('check_id') or t.get('category') or ''} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
    L += ['', "## Reviewer quotations against the reviewers' document", '',
          f"Document: `{W171.DOCX}` sha256 `{W171.DOCX_SHA[:8]}`, word/document.xml sha256 `{o['docx']['document_xml_sha256'][:8]}`, "
          f"{o['docx']['n_paragraphs']} paragraphs (W171a normalisation).", '',
          '| quote | segment | letter lines | chars | result | match ratio | docx paragraph | differences (letter -> document) |',
          '|---:|---:|---|---:|---|---:|---:|---|']
    for r in o['quotations']:
        d = '; '.join(f"{x['op']} l.{x['letter_line']}: ...{x['context_before'][-20:]}[{x['letter']!r} -> {x['document']!r}]"
                      f"{x['context_after'][:20]}..." for x in r['diffs'][:6])
        L.append(f"| {r['quote']} | {r['segment']} | {r['lines']} | {r['chars']} | {r['status']} | {r['ratio']:.3f} | "
                 f"{r['docx_paragraph']} | {_cell(d, 600)} |")
    L += ['', f"Quotation segments: {s['quotation_segments']}; quotation tokens: {s['quotation_tokens']}", '']
    for key, (a, b), desc, _h in REVISED:
        if key == 'sec3_5_6':
            continue
        L += [f'## {desc} (main.tex l. {a}-{b}): every number with its source', '',
              '| line | written | status | check | counterpart (at written precision) | source / reason |', '|---:|---|---|---|---|---|']
        for t in o['tokens']:
            if t['file'] == MAIN and a <= t['line'] <= b and not t.get('excluded'):
                L.append(f"| {t['line']} | {t['written']} | {t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                         f"{_cell(t.get('value_at_written_precision'), 50)} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
        L.append('')
    L += ['## Response letter: every number with its source', '',
          '| line | written | scope | status | check | counterpart (at written precision) | source / reason |', '|---:|---|---|---|---|---|---|']
    for t in o['tokens']:
        if t['file'] == LETTER and not t.get('excluded'):
            L.append(f"| {t['line']} | {t['written']} | {t['scope']}{'/' + t['macro'] if t.get('macro') else ''} | "
                     f"{t['status_display']} | {t.get('check_id') or t.get('category') or ''} | "
                     f"{_cell(t.get('value_at_written_precision'), 60)} | {_cell(t.get('counterpart') or t.get('note'), 260)} |")
    L += ['', '## Findings -- letter', '']
    for f in o['letter_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    L += ['', '## Findings -- main.tex', '']
    for f in o['main_findings']:
        L.append(f"- **{f['id']}** (line {f['lines']}, {f['kind']}): {f['finding']}")
    for fname in (HIGHLIGHTS, COVER):
        L += ['', f'## {fname}: every number with its source', '']
        rows_ = [t for t in o['tokens'] if t['file'] == fname and not t.get('excluded')]
        for t in rows_:
            L.append(f"- line {t['line']} `{t['written']}`: {t['status_display']} -- {t.get('counterpart') or t.get('note') or ''}")
        if not rows_:
            L.append('no in-scope numeric token')
    L += ['', '## main.tex line rules re-pinned to 7aa9e105', '',
          'Revised passages (declared or ruled; never the auto index): ' +
          '; '.join(f'{k} l. {a}-{b}' for k, (a, b), _d, _h in REVISED) +
          f". Value table l. {VALUE_TABLE_LINES[0]}-{VALUE_TABLE_LINES[1]}; network tables l. {NETWORK_TABLES[0]}-"
          f"{NETWORK_TABLES[1]}; Appendices B-D l. {APP_BCD[0]}-{APP_BCD[1]}. Each mapped from W174a's range (difflib, "
          f"cb237b2d -> 7aa9e105): {o['main_regions']['line_rules_mapped_from_v3']}", '',
          '| W174a region (260bd83 l.) | key | kind | 8b76423 (l.) |', '|---|---|---|---|']
    for r in o['main_regions']['kept']:
        L.append(f"| {r['old'][0]}-{r['old'][1]} | {r['key']} | {r['kind']} | {r['new'][0]}-{r['new'][1]} |")
    for r in o['main_regions']['dropped']:
        L.append(f"| {r['old'][0]}-{r['old'][1]} | {r['key']} | dropped ({r['n_lines_changed']} lines changed) | {_cell(r['reason'], 200)} |")
    L += ['', '## main.tex: submitted-version figures by section', '', '| section | n | map lines cited |', '|---|---:|---|']
    for sec, v in s['main_submitted_by_section'].items():
        L.append(f"| {sec} | {v['n']} | {', '.join(v['cites'])} |")
    L += ['', '## W171a parameter checks re-evaluated on this run (source of several value-table rows)', '',
          '| id | quantity | found | status | sources |', '|---|---|---|---|---|']
    for p_ in o['parameter_table']:
        L.append(f"| {p_['id']} | {_cell(p_['quantity'])} | {_cell(p_['found'], 250)} | {p_['status']} | {_cell('; '.join(p_['sources']), 400)} |")
    L += ['', '## Unchecked: by category (all files)', '', '| file | category | n |', '|---|---|---:|']
    for f, cats in s['unchecked_by_category'].items():
        for c, n in cats.items():
            L.append(f'| {f} | {c} | {n} |')
    L += ['', '## Excluded LaTeX structure', '', '| file | category | n | tokens |', '|---|---|---:|---|']
    for f, cats in s['excluded_by_category'].items():
        for c, v in cats.items():
            L.append(f"| {f} | {c} | {v['n']} | {', '.join(v['examples'])} |")
    L += ['', '## Integrity checks', '']
    for k, v in o['checks'].items():
        L.append(f'- {k}: {v}')
    return '\n'.join(L) + '\n'


VALUE_EXCLUDED = ('the named settings of the benchmark paragraph that carry no number and were confirmed by W175 (the '
                  'no-reverse-flow rule, the passive and price-taker arm definitions, the TSO dispatch at fixed interface '
                  'exchanges) are not rows; the round-4 settings of that paragraph (consistency pass, common Q, the '
                  'instance) and of the first section 3.6 paragraph (the search campaigns without the tight tail) are')


# ======================================================================================================================
#  6. main
# ======================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--manuscript-dir', default=DEFAULT_MANUSCRIPT_DIR)
    ap.add_argument('--overleaf-commit', required=True)
    ap.add_argument('--expect', action='append', default=[], help='NAME=SHA256 for every *.tex file (declared)')
    ap.add_argument('--post-run', action='store_true')
    ap.add_argument('--trial-out-dir', default=None, help='trial runs only: an output directory outside the results')
    args = ap.parse_args()
    out_dir = args.trial_out_dir or os.path.join(OUT_ROOT, f'overleaf_{args.overleaf_commit[:7]}')
    if args.post_run:
        post_run(out_dir)
    t0 = time.time()
    tag = 'W176'
    pre = []
    od = os.path.join(REPO, out_dir) if not os.path.isabs(out_dir) else out_dir
    if not os.path.isdir(od):
        pre.append(f'{out_dir} missing (the launch command creates it with mkdir; the launch log goes there)')
    else:
        extra = sorted(set(os.listdir(od)) - {OUT_NAMES['log']})
        if extra:
            pre.append(f'{out_dir} is not new: it holds {extra} (write-once; never overwrite a previous output)')
    if not re.fullmatch(r'[0-9a-f]{7,40}', args.overleaf_commit):
        pre.append(f'--overleaf-commit {args.overleaf_commit!r} is not a hex commit')
    mdir = args.manuscript_dir
    mabs = os.path.join(REPO, mdir)
    expect = {}
    for e in args.expect:
        if '=' not in e:
            pre.append(f'--expect {e!r}: NAME=SHA256')
            continue
        k, v = e.split('=', 1)
        expect[k] = v
    files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(mabs, '*.tex')))
    head = subprocess.run(['git', '-C', mabs, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    porcelain = subprocess.run(['git', '-C', mabs, 'status', '--porcelain', '--', '*.tex', BIB], capture_output=True,
                               text=True).stdout.strip()
    mfiles = []
    for f in files:
        with open(os.path.join(mabs, f), 'rb') as h:
            b = h.read()
        blob = subprocess.run(['git', '-C', mabs, 'show', f'{args.overleaf_commit}:{f}'], capture_output=True)
        mfiles.append({'name': f, 'path': os.path.join(mdir, f), 'sha256': _sha_bytes(b), 'bytes': len(b),
                       'lines': b.decode('utf-8').count('\n') + 1,
                       'blob_at_commit_sha256': _sha_bytes(blob.stdout) if blob.returncode == 0 else None,
                       'declared_sha256': expect.get(f), 'scope': 'draft, not compiled' if f == DRAFT else 'manuscript'})
    with open(os.path.join(mabs, BIB), 'rb') as h:
        bib_bytes = h.read()
    bib_blob = subprocess.run(['git', '-C', mabs, 'show', f'{args.overleaf_commit}:{BIB}'], capture_output=True)
    bib_rec = {'path': os.path.join(mdir, BIB), 'sha256': _sha_bytes(bib_bytes),
               'blob_at_commit_sha256': _sha_bytes(bib_blob.stdout) if bib_blob.returncode == 0 else None}

    def show(commit, f):
        r = subprocess.run(['git', '-C', mabs, 'show', f'{commit}:{f}'], capture_output=True)
        return r.stdout if r.returncode == 0 else None
    main_v3_b, main_v2_b, letter_v3_b = show(W174_COMMIT_IN_CLONE, MAIN), show(W171_COMMIT_IN_CLONE, MAIN), show(W174_COMMIT_IN_CLONE, LETTER)
    main_v3_sha = _sha_bytes(main_v3_b) if main_v3_b is not None else None
    main_v2_sha = _sha_bytes(main_v2_b) if main_v2_b is not None else None
    mchecks = {
        'clone_head_is_declared_commit': bool(head) and head.startswith(args.overleaf_commit),
        'file_set_equals_declared': set(files) == set(expect),
        'every_sha_equals_declared': all(m['sha256'] == m['declared_sha256'] for m in mfiles),
        'every_file_equals_its_blob_at_commit': all(m['sha256'] == m['blob_at_commit_sha256'] for m in mfiles),
        'no_tex_or_bib_modified_or_untracked_in_clone': porcelain == '',
        'bibliography_equals_its_blob_at_commit': bib_rec['sha256'] == bib_rec['blob_at_commit_sha256'],
        'w174a_pin_readable_in_clone': main_v3_sha == W174.MAIN_LINE_RULES_SHA,
        'w171a_pin_readable_in_clone': main_v2_sha == W171.MAIN_LINE_RULES_SHA,
    }
    for k, v in mchecks.items():
        if not v:
            pre.append(f'manuscript: {k} FAILED (head {head}, files {files}, declared {sorted(expect)}, porcelain {porcelain!r}, '
                       f'shas {[(m["name"], m["sha256"][:8], (m["declared_sha256"] or "")[:8]) for m in mfiles]}, w174a pin '
                       f'{main_v3_sha}, w171a pin {main_v2_sha}, bib {bib_rec})')
    input_rels = {'FROZEN_JSON': W164.FZ_REL, 'FROZEN_MD': W164.FZ_MD_REL, 'W163_JSON': W164.W163_JSON,
                  'W163_MANIFEST': W164.W163_MAN, 'W163_SCRIPT': W164.W163_SCRIPT, 'W164_SCRIPT': W171.W164_SCRIPT,
                  'W164_RESULTS': W171.W164_RESULTS, 'P5': W164.P5_REL, 'P2': W164.P2_REL, 'MAP': W164.MAP_REL,
                  'SRP1_JSON': W164.SRP1_JSON, 'SRP1_PARAMS': W171.SRP1_PARAMS, 'ESS_PARAMS': W171.ESS_PARAMS,
                  'STEP4': W164.STEP4, 'A1_CAMPAIGN': W164.A1_CAMPAIGN, 'V41_SPEC': W164.V41_SPEC, 'V6_SPEC': W171.V6_SPEC,
                  'SPEC_3X3': W171.SPEC_3X3, 'INST_3X3': W171.INST_3X3, 'EXT_SPEC_V3': W171.EXT_SPEC_V3,
                  'SUBMITTED_MAIN': W171.SUBMITTED_MAIN, 'DOCX': W171.DOCX, 'CORR_MD': W171.CORR_MD}
    for k, v in W164.CASE_FILES.items():
        input_rels[f'CASE_{k}'] = v
    for k, v in W171.CASE_PARAMS.items():
        input_rels[f'CP_{k}'] = v
    for k, v in {**W171.CODE, **W174.CODE_V3, **CODE_V4}.items():
        input_rels[f'CODE_{k}'] = v
    input_rels.update(W174.EXTRA_INPUTS)
    input_rels.update(W174.DN_YEAR_FILES)
    input_rels.update(EXTRA_INPUTS_V4)
    inputs = {}
    for key, rel in input_rels.items():
        if not os.path.exists(os.path.join(REPO, rel)):
            pre.append(f'{rel} missing')
            continue
        inputs[key] = {'path': rel, 'sha256': _sha(rel), 'committed_clean': W160.W157.L132._committed_clean(rel),
                       'last_commit': W160._last_commit(rel), 'blob': _git('rev-parse', f'HEAD:{rel}')}
        if not inputs[key]['committed_clean']:
            pre.append(f'{rel} not committed clean')
    if not pre:
        for key, want in (('FROZEN_JSON', W164.FZ_SHA), ('P5', W164.P5_SHA), ('P2', W164.P2_SHA),
                          ('SUBMITTED_MAIN', W171.SUBMITTED_MAIN_SHA), ('DOCX', W171.DOCX_SHA)):
            if inputs[key]['sha256'] != want:
                pre.append(f"{inputs[key]['path']} sha {inputs[key]['sha256']} != {want}")
        if _jl(W164.W163_MAN).get(W164.W163_JSON) != inputs['W163_JSON']['sha256']:
            pre.append(f'{W164.W163_JSON} != its W163 manifest entry')
        for key, want in (('W164_SCRIPT', W171.W164_SCRIPT_COMMIT), ('W163_SCRIPT', W164.W163_SCRIPT_COMMIT),
                          ('W171_SCRIPT', W174.W171_SCRIPT_COMMIT), ('W171_RESULTS', W174.W171_RESULTS_COMMIT),
                          ('W174_SCRIPT', W174_SCRIPT_COMMIT), ('W174_RESULTS', W174_RESULTS_COMMIT)):
            if not (inputs[key]['last_commit'] or '').startswith(want):
                pre.append(f"{inputs[key]['path']} last commit {inputs[key]['last_commit']} != {want}")
    script_clean = W160.W157.L132._committed_clean(SCRIPT_REL)
    if pre:
        _log(f'[{tag} PRECONDITION FAILED] {pre}')
        sys.exit(1)
    _log(f'[{tag}] script {SCRIPT_REL} sha256 {_sha(SCRIPT_REL)} committed clean {script_clean}; {len(inputs)} inputs committed '
         f'clean; frozen JSON {W164.FZ_SHA[:8]}, submitted main.tex {W171.SUBMITTED_MAIN_SHA[:8]}, reviewers\' docx '
         f'{W171.DOCX_SHA[:8]} verified; manuscript {mdir} HEAD {head} (declared {args.overleaf_commit}); files ' +
         ', '.join(f"{m['name']} {m['sha256'][:8]}" for m in mfiles) + f'; bibliography {bib_rec["sha256"][:8]}; W174a pin '
         f'{W174_COMMIT_IN_CLONE}:main.tex {main_v3_sha[:8]}; W171a pin {W171_COMMIT_IN_CLONE}:main.tex {main_v2_sha[:8]}')
    # ---- documents, tokens, records ----------------------------------------------------------------------------------
    docs = {m['name']: W164.Doc(m['name'], _text(m['path'])) for m in mfiles}
    roles = {n: ('main' if n == MAIN else 'letter' if n.startswith('response_to_reviewers') else 'highlights' if n == HIGHLIGHTS
                 else 'cover' if n == COVER else 'draft' if n == DRAFT else 'other') for n in docs}
    toks = []
    for n in files:
        toks += W174.tokenize_v3(docs[n])
    sub_doc = W164.Doc('main.tex', _text(W171.SUBMITTED_MAIN))
    X = RecordsV4(inputs, docs.get(MAIN), sub_doc)
    X.letter_doc = docs.get(LETTER)
    X.letter_raw = docs[LETTER].raw if LETTER in docs else ''
    X.cover_raw = docs[COVER].raw if COVER in docs else ''
    X.main_tokens = [t for t in toks if t['file'] == MAIN and t['scope'] == 'body' and not t['excluded']]
    X.sub_tokens = [t for t in W164.tokenize(sub_doc) if t['scope'] == 'body' and not t['excluded']]
    X.main_sha = next((m['sha256'] for m in mfiles if m['name'] == MAIN), None)
    X.corr_text = _text(W171.CORR_MD)
    X.xlsx_scan = scan_cost_workbook(ESS_XLSX)
    X.bib_entry = bib_entry(bib_bytes.decode('utf-8'), 'nrel_ess_costs')
    X.bib_src = f"{bib_rec['path']} (sha256 {bib_rec['sha256'][:8]}, = blob at {args.overleaf_commit})"
    X.chem_in_manuscript = [(c['line'], c['term']) for c in chem_terms(docs[MAIN]) if c['scope'] == 'body']
    # ---- line rules: W174a's regions recomputed, mapped cb237b2d -> this main.tex ---------------------------------------
    v2_lines = main_v2_b.decode('utf-8').split('\n')
    v3_lines = main_v3_b.decode('utf-8').split('\n')
    kept3, dropped3, _lm23 = W174.map_regions(v2_lines, v3_lines)
    w174 = X.j['W174_RESULTS']
    w174_kept_new = [r['new'] for r in w174['main_regions']['kept']]
    regions_v3_ok = ([list(r[0]) for r in kept3] == [list(r) for r in W174.EXPECTED_MAIN_REGIONS] == w174_kept_new and
                     [d_['key'] for d_ in dropped3] == [d_['key'] for d_ in w174['main_regions']['dropped']])
    lm = line_map(v3_lines, docs[MAIN].lines)
    kept, dropped = map_regions_v4(lm, kept3)
    X.main_regions_v4 = kept
    mapped = {'REVISED': {k: map_range(lm, *r) for k, r in V3_RANGES['REVISED'].items()},
              'VALUE_TABLE_LINES': map_range(lm, *V3_RANGES['VALUE_TABLE_LINES']),
              'NETWORK_TABLES': map_range(lm, *V3_RANGES['NETWORK_TABLES']),
              'APP_BCD': map_range(lm, *V3_RANGES['APP_BCD'])}
    rules_ok = (mapped['REVISED'] == {k: r for k, r, _d, _h in REVISED} and mapped['VALUE_TABLE_LINES'] == VALUE_TABLE_LINES
                and mapped['NETWORK_TABLES'] == NETWORK_TABLES and mapped['APP_BCD'] == APP_BCD)
    changed_main = sorted(set(range(1, len(docs[MAIN].lines) + 1)) - set(lm.values()))
    letter_lm = line_map(letter_v3_b.decode('utf-8').split('\n'), docs[LETTER].lines) if letter_v3_b is not None else {}
    changed_letter = sorted(set(range(1, len(docs[LETTER].lines) + 1)) - set(letter_lm.values()))
    region_heads_ok = {k: docs[MAIN].lines[a - 1].strip().startswith(h) for k, (a, b), _d, h in REVISED}
    try:
        ptable = W171.parameter_checks(X)
        ptable_error = None
    except Exception as e:  # a parameter whose source moved fails the run, loudly
        ptable, ptable_error = [], f'{type(e).__name__}: {e}'
    X.ptab = {p['id']: p for p in ptable}
    leaves = W164.json_leaves(X.fz)
    AX = W164.AutoIndex(leaves)
    map_cite_lines, map_cite_fail = {}, []
    for k, frag in W164.MAP_CITES.items():
        try:
            map_cite_lines[k] = X.map_line(frag)
        except KeyError as e:
            map_cite_fail.append(str(e))
    map_xref_numbers = set(re.findall(r'(?<![\d.])(\d\.\d(?:\.\d)?)(?![\d.])', '\n'.join(X.map)))
    # ---- reviewer quotations -------------------------------------------------------------------------------------------
    dx = W171.docx_text(W171.DOCX)
    qrows = W171.quotation_check(docs[LETTER], dx)
    qstat = W171.quotation_token_status(qrows, docs[LETTER], toks)
    # ---- the version-3 declaration set (W174a applied), reconstructed and re-located ------------------------------------
    w171 = X.j['W171_RESULTS']
    v1 = W164.letter_declarations() + W164.cover_declarations() + W164.main_declarations()
    v1_carried = {r['id'] for r in w171['relocation'] if r['outcome'] == 'carried'}
    v2 = W171.letter_declarations_v2() + [d for d in v1 if d['id'] in v1_carried] + \
        [d for d in W171.all_declarations_v2() if d['file'] != LETTER and not re.fullmatch(r'(L|CL|M)\d+', d['id'])]
    v2_carried_in_w174 = {r['id'] for r in w174['relocation'] if r['outcome'] == 'carried'}
    v3_new = W174.all_declarations_v3()
    v3 = [dict(d, _v=2) for d in v2 if d['id'] in v2_carried_in_w174] + [dict(d, _v=3) for d in v3_new]
    w174_applied = {d['id']: d['version'] for d in w174['declarations_applied']}
    v3_reconstructed = (sorted(d['id'] for d in v3) == sorted(w174_applied) and
                        all(w174_applied[d['id']] == d['_v'] for d in v3))
    relocation = []
    for d in v3:
        doc = docs.get(d['file'])
        hits = len(doc.find_fragment(d['fragment'])[0]) if doc else 0
        if d['id'] in SUPERSEDED_V4:
            outcome, v4, why = 'superseded', SUPERSEDED_V4[d['id']][0], SUPERSEDED_V4[d['id']][1]
        elif hits == 1:
            outcome, v4, why = 'carried', d['id'], 'fragment found once; evaluated unchanged'
        elif d['id'] in REPLACED_BY_V4:
            outcome, v4, why = 'removed', REPLACED_BY_V4[d['id']][0], REPLACED_BY_V4[d['id']][1]
        else:
            outcome, v4, why = 'removed (NO REPLACEMENT DECLARED)', None, f'fragment found {hits} times and no v4 entry names it'
        relocation.append({'id': d['id'], 'file': d['file'], 'fragment': d['fragment'], 'hits': hits, 'outcome': outcome,
                           'v3_version': d['_v'], 'v4': v4, 'reason': why})
    carried_ids = {r['id'] for r in relocation if r['outcome'] == 'carried'}
    unreplaced = [r['id'] for r in relocation if r['outcome'].startswith('removed (NO')]
    replaced_stale = [k for k in REPLACED_BY_V4 if k in carried_ids]
    new_decls = all_declarations_v4()
    retarget_clash = [d['id'] for d in new_decls if d.get('retargeted_from') in carried_ids]
    decls = [d for d in v3 if d['id'] in carried_ids] + new_decls
    new_ids = {d['id'] for d in new_decls}
    value_ids = {d['id'] for d in decls if d.get('value_table')}
    seen_ids = collections.Counter(d['id'] for d in decls)
    dup_ids = sorted(k for k, n in seen_ids.items() if n > 1)
    # ---- declared checks -----------------------------------------------------------------------------------------------
    by_key = {t['key']: t for t in toks}
    assigned, claimed_twice, stale, decl_out = {}, [], [], []
    for d in decls:
        if d['file'] not in docs:
            stale.append({'id': d['id'], 'why': f"{d['file']} not in the manuscript directory"})
            continue
        doc = docs[d['file']]
        hits, flen = doc.find_fragment(d['fragment'])
        if len(hits) != 1:
            stale.append({'id': d['id'], 'why': f'fragment found {len(hits)} times', 'fragment': d['fragment']})
            continue
        a = hits[0]
        inside = [t for t in toks if t['file'] == d['file'] and not t['excluded'] and doc.flat_pos(t['line'], t['col0']) is not None
                  and a <= doc.flat_pos(t['line'], t['col0']) < a + flen]
        picked, ok = [], True
        for txt, occ in d['tokens']:
            c = [t for t in inside if t['text'] == txt]
            if len(c) <= occ:
                ok = False
                stale.append({'id': d['id'], 'why': f'token {txt!r}#{occ} not in the fragment'})
                break
            picked.append(c[occ])
        if not ok:
            continue
        try:
            res = d['fn'](X, [t['written'] for t in picked])
        except Exception as e:  # a declaration that cannot evaluate is stale, loudly
            stale.append({'id': d['id'], 'why': f'evaluation error {type(e).__name__}: {e}'})
            continue
        if isinstance(res, dict):
            res = [res] * len(picked)
        if len(res) != len(picked):
            stale.append({'id': d['id'], 'why': f'{len(res)} results for {len(picked)} tokens'})
            continue
        fl = doc.flat_idx[a]
        decl_out.append({'id': d['id'], 'file': d['file'], 'fragment': d['fragment'], 'line': fl[0],
                         'version': 4 if d['id'] in new_ids else d['_v'], 'retargeted_from': d.get('retargeted_from'),
                         'tokens': [[t['line'], t['col'], t['written']] for t in picked]})
        for t, r in zip(picked, res):
            if t['key'] in assigned:
                claimed_twice.append([t['key'], assigned[t['key']]['check_id'], d['id']])
                continue
            r = dict(r)
            r['check_id'] = d['id']
            r['rule'] = 'declared'
            assigned[t['key']] = r
    # ---- rules, auto index -----------------------------------------------------------------------------------------------
    for t in toks:
        t['auto_candidates'] = AX.candidates(t) if not t['excluded'] else None
        if t['excluded']:
            t['status'], t['status_display'] = 'excluded', f"excluded ({t['excluded']})"
            continue
        reg = region_of(t['line']) if t['file'] == MAIN else None
        r = assigned.get(t['key'])
        if r is None:
            r = rule_assign_v4(X, docs[t['file']], roles[t['file']], t, map_xref_numbers, qstat)
            if r is not None:
                r = dict(r)
                r['rule'] = 'rule'
        if r is not None and r['status'] == 'match' and r['rule'] == 'rule':
            r['match_kind'] = 'rule (named record)'
        if r is None and roles[t['file']] == 'main' and reg is None and t['auto_candidates']:
            c = t['auto_candidates']
            r = R('match', c[0]['path'] if len(c) == 1 else f'{len(c)} paths', 'frozen JSON (auto index)', None, None, None)
            r['rule'], r['match_kind'] = 'auto', 'auto-unique' if len(c) == 1 else 'auto-ambiguous'
        if r is None and roles[t['file']] == 'main' and reg is None and X.main_sha == MAIN_LINE_RULES_SHA:
            r = ntc('auto index: no candidate; no declaration or rule this round (submitted text outside the revised passages)')
            r['rule'] = 'leftover'
        if r is None:
            t['status'], t['status_display'] = None, 'UNASSIGNED'
            continue
        t.update({k: v for k, v in r.items()})
        if t['status'] == 'match' and 'match_kind' not in r:
            t['match_kind'] = 'declared'
        mk = t.get('match_kind')
        t['status_display'] = (f"match (auto, {t['counterpart']})" if mk == 'auto-unique' else
                               f"match (auto, ambiguous: {len(t['auto_candidates'])} paths)" if mk == 'auto-ambiguous'
                               else "MISMATCH vs reviewers' document" if t.get('mismatch_vs') else t['status'])
    unassigned = [t['key'] for t in toks if not t['excluded'] and t['status'] is None]
    # ---- the value table -------------------------------------------------------------------------------------------------
    vtable = []
    claimed_value_keys = set()
    for d in decl_out:
        if d['id'] not in value_ids:
            continue
        for k, r in enumerate(X.value_rows.get(d['id'], [])):
            tl = [d['tokens'][i] for i in r['tok']]
            for i in r['tok']:
                claimed_value_keys.add(f"{MAIN}:{d['tokens'][i][0]}:{d['tokens'][i][1]}")
            vtable.append({'id': f"{d['id']}.{k + 1}", 'declaration': d['id'], 'line': tl[0][0] if tl else d['line'],
                           'tokens': tl, 'printed': r['printed'], 'quantity': r['quantity'],
                           'value_in_force': r['value_in_force'], 'sources': r['sources'], 'status': r['status'],
                           'precision': r['precision'], 'note': r['note'], 'kind': 'value' if tl else 'named setting',
                           'declaration_version': d['version']})
    vtable.sort(key=lambda r: (r['line'], r['id']))
    value_range_other = [{'line': t['line'], 'written': t['written'], 'status': t['status_display'],
                          'reason': t.get('note') or t.get('counterpart') or t.get('category')}
                         for t in toks if t['file'] == MAIN and VALUE_TABLE_LINES[0] <= t['line'] <= VALUE_TABLE_LINES[1]
                         and not t['excluded'] and t['key'] not in claimed_value_keys]
    vt_inputs = sorted({(v['path'], v['sha256'], v['blob']) for v in inputs.values()
                        if any(v['path'] in src for r in vtable for src in r['sources'])})
    vstat = collections.Counter(r['status'] for r in vtable)
    vstat_values = collections.Counter(r['status'] for r in vtable if r['kind'] == 'value')
    vstat_settings = collections.Counter(r['status'] for r in vtable if r['kind'] == 'named setting')
    # ---- summary -----------------------------------------------------------------------------------------------------------
    per_file = {}
    for n in files:
        ft = [t for t in toks if t['file'] == n]
        sc = [t for t in ft if not t['excluded']]
        c = {'scope': 'draft, not compiled' if n == DRAFT else 'manuscript', 'tokens': len(ft), 'excluded': len(ft) - len(sc),
             'in_scope_body': sum(t['scope'] == 'body' for t in sc), 'in_scope_comment': sum(t['scope'] == 'comment' for t in sc),
             'match_declared': sum(t['status'] == 'match' and t.get('match_kind') == 'declared' for t in sc),
             'match_rule': sum(t['status'] == 'match' and t.get('match_kind') == 'rule (named record)' for t in sc),
             'match_auto_unique': sum(t.get('match_kind') == 'auto-unique' for t in sc),
             'match_auto_ambiguous': sum(t.get('match_kind') == 'auto-ambiguous' for t in sc),
             'unassigned': sum(t['status'] is None for t in sc)}
        for st in STATUSES[1:]:
            c[st] = sum(t['status'] == st for t in sc)
        c['MISMATCH_vs_reviewers_document'] = sum(bool(t.get('mismatch_vs')) for t in sc)
        c['match_total'] = c['match_declared'] + c['match_rule'] + c['match_auto_unique'] + c['match_auto_ambiguous']
        c['assigned_total'] = c['match_total'] + sum(c[st] for st in STATUSES[1:])
        per_file[n] = c
    sub_by_sec, stat_by_sec = collections.OrderedDict(), collections.OrderedDict()
    for t in toks:
        if t['file'] != MAIN or t['excluded']:
            continue
        sec = t['section'].split(' > ')[0] if t['scope'] == 'body' else '(comments)'
        stat_by_sec.setdefault(sec, collections.Counter())[f"match ({t['match_kind']})" if t['status'] == 'match' else t['status_display']] += 1
        if t['status'] == 'submitted-version figure':
            v = sub_by_sec.setdefault(sec, {'n': 0, 'cites': []})
            v['n'] += 1
            if t.get('map_cite') and t['map_cite'].split(':')[1] not in v['cites']:
                v['cites'].append(t['map_cite'].split(':')[1])
    unc_cat, exc_cat = {}, {}
    for t in toks:
        if t['excluded']:
            e = exc_cat.setdefault(t['file'], {}).setdefault(t['excluded'], {'n': 0, 'examples': []})
            e['n'] += 1
            if len(e['examples']) < 8 and t['text'] not in e['examples']:
                e['examples'].append(t['text'])
        elif t['status'] == 'unchecked':
            unc_cat.setdefault(t['file'], collections.Counter())[t.get('category') or 'other'] += 1
    overridden = collections.Counter(t['status'] or 'UNASSIGNED' for t in toks if not t['excluded'] and t['auto_candidates']
                                     and not t.get('match_kind', '').startswith('auto'))
    qtok = collections.Counter(t['status_display'] for t in toks if t['file'] == LETTER and t.get('macro') == 'rcomment' and not t['excluded'])
    # ---- reconciliations against W174a ------------------------------------------------------------------------------------
    rec_sub = reconcile(w174['tokens'], toks, lm, 'submitted-version figure')
    rec_apx = reconcile(w174['tokens'], toks, lm, 'approximate')
    w174_pf = w174['summary']['per_file']
    sv_regions_note = ('every W174a submitted-version region maps unchanged (' + ', '.join(
        f'{r[1]} {r[0][0]}-{r[0][1]}' for r in kept) + ')' if not dropped else f'dropped: {dropped}') + \
        '; no round-4 hunk lies outside section 2, sections 3.5-3.6 and Appendix A (lines changed outside the revised ' + \
        f'passages: {[x for x in changed_main if region_of(x) is None]})'
    # ---- the prediction, item by item -------------------------------------------------------------------------------------
    mm_main, mm_letter = per_file[MAIN]['MISMATCH'], per_file[LETTER]['MISMATCH']
    apx_same = not rec_apx['removed'] and not rec_apx['added'] and rec_apx['n_w174a'] == 12
    sub_removed_outside = [x for x in rec_sub['removed'] if x.get('old_region') != 'sec3_5_6']
    pred_items = [
        {'item': 'zero MISMATCH in main.tex', 'w174a': w174_pf[MAIN]['MISMATCH'], 'w176': mm_main,
         'outcome': 'HELD' if mm_main == 0 else f'FAILED ({mm_main} MISMATCH)'},
        {'item': 'zero MISMATCH in the letter', 'w174a': w174_pf[LETTER]['MISMATCH'], 'w176': mm_letter,
         'outcome': 'HELD' if mm_letter == 0 else f'FAILED ({mm_letter} MISMATCH)'},
        {'item': 'the same twelve approximates as W174a (printed roundings)', 'w174a': rec_apx['n_w174a'], 'w176': rec_apx['n_w176'],
         'outcome': (f"HELD: the same {rec_apx['n_same']} tokens -- " + '; '.join(
                         f"l. {x['old_line']}->{x['new_line']} {x['written']} ({x['check_old']}"
                         + (f"->{x['check_new']}" if x['check_new'] != x['check_old'] else '') + ')' for x in rec_apx['same'])
                     + '; matched by mapped line except ' + (', '.join(f"l. {x['new_line']} ({x['match_basis']})" for x in rec_apx['same']
                                                                  if not x['match_basis'].startswith('mapped')) or 'none')
                     if apx_same else
                     f"FAILED: removed {[(x['old_line'], x['written'], x['check_old']) for x in rec_apx['removed']]}, added "
                     f"{[(x['new_line'], x['written'], x['check_new']) for x in rec_apx['added']]}")},
        {'item': 'submitted-version count reduced only where section 3.5-3.6 text replaced submitted text',
         'w174a': rec_sub['n_w174a'], 'w176': rec_sub['n_w176'],
         'outcome': (('HELD on "only where" (no reduction anywhere else); NO REDUCTION OBSERVED: the count is unchanged, '
                      f"{rec_sub['n_w174a']} -> {rec_sub['n_w176']}, every token at its mapped line -- sections 3.5-3.6 held "
                      'no submitted-version figure at 260bd83 (W174a already treated l. 1066-1151 as a revised passage), so '
                      'none could be replaced')
                     if not rec_sub['removed'] and not rec_sub['added'] else
                     ('HELD' if not sub_removed_outside and not rec_sub['added'] else 'FAILED') +
                     f": {rec_sub['n_w174a']} -> {rec_sub['n_w176']}; removed {len(rec_sub['removed'])} (outside sections "
                     f"3.5-3.6: {len(sub_removed_outside)}), added {len(rec_sub['added'])}")},
    ]
    prediction = {'statement': 'zero MISMATCH in main.tex and the letter, with the same twelve approximates as W174a (printed '
                               'roundings) and the submitted-version count reduced only where section 3.5-3.6 text replaced '
                               'submitted text (expert, Addendum 72)',
                  'items': pred_items}
    summary = {'per_file': per_file, 'every_token_assigned': not unassigned, 'unassigned': unassigned,
               'stale_declarations': stale, 'claimed_twice': claimed_twice, 'duplicate_declaration_ids': dup_ids,
               'main_submitted_by_section': sub_by_sec, 'main_status_by_section': {k: dict(v) for k, v in stat_by_sec.items()},
               'unchecked_by_category': {k: dict(v) for k, v in unc_cat.items()}, 'excluded_by_category': exc_cat,
               'auto_candidates_overridden': dict(overridden), 'n_frozen_json_numeric_leaves': len(leaves),
               'n_declarations': len(decls), 'n_declarations_applied': len(decl_out),
               'n_declarations_v4': sum(d['version'] == 4 for d in decl_out), 'quotation_tokens': dict(qtok),
               'quotation_segments': dict(collections.Counter(r['status'] for r in qrows)),
               'relocation': dict(collections.Counter(r['outcome'] for r in relocation)),
               'parameter_table': dict(collections.Counter(p['status'] for p in ptable)),
               'value_table_status': dict(vstat), 'value_table_status_values': dict(vstat_values),
               'value_table_status_settings': dict(vstat_settings), 'n_value_rows': len(vtable),
               'extra_word_tokens': sum(1 for t in toks if t.get('tokenizer') == 'v3 extra number word')}
    lf = letter_findings_v4(X, docs, qrows)
    mf = main_findings_v4(X)
    checks = dict(mchecks)
    checks.update({
        'frozen_json_unchanged': _sha(W164.FZ_REL) == W164.FZ_SHA,
        'v3_declaration_set_reconstructed_equals_w174a_applied': v3_reconstructed,
        'every_declaration_found_and_evaluated': not stale,
        'no_token_claimed_twice': not claimed_twice,
        'no_duplicate_declaration_id': not dup_ids,
        'every_token_assigned': not unassigned,
        'every_map_citation_found_once': not map_cite_fail,
        'statuses_in_vocabulary': all(t['excluded'] or t['status'] in STATUSES for t in toks if t['status']),
        'main_line_rules_pinned_to_this_main': X.main_sha == MAIN_LINE_RULES_SHA,
        'revised_passages_start_at_their_headings': all(region_heads_ok.values()),
        'w174a_regions_recomputed_equal_w174a_constant_and_results': regions_v3_ok,
        'w174a_regions_kept_or_dropped_with_reason': all(not d_['reason'].startswith('TEXT CHANGED') for d_ in dropped),
        'w174a_regions_equal_pinned_expectation_v4': [list(r[0]) for r in kept] == [list(r) for r in EXPECTED_MAIN_REGIONS],
        'line_rules_v4_equal_mapped_from_w174a': rules_ok,
        'every_v3_declaration_carried_superseded_or_replaced': not unreplaced,
        'no_replacement_entry_for_a_carried_declaration': not replaced_stale and not retarget_clash,
        'parameter_table_evaluated': ptable_error is None and bool(ptable),
        'every_value_declaration_applied': value_ids <= {d['id'] for d in decl_out},
        'every_value_row_status_in_vocabulary': all(r['status'] in ROW_STATUSES for r in vtable),
        'every_quotation_segment_located': all(r['status'] != 'not found' for r in qrows),
        'docx_sha_equals_declared': _sha(W171.DOCX) == W171.DOCX_SHA,
    })
    failed = sorted(k for k, v in checks.items() if v is not True)
    guards, pk, guards_ok = guards_state()
    code = 0 if not failed else 3
    if not guards_ok:
        code = 1
    for t in toks:
        for k in ('col0', 'end0'):
            t.pop(k, None)
        if t.get('auto_candidates') is not None:
            t['auto_candidates_n'] = len(t['auto_candidates'])
            t['auto_candidates'] = t['auto_candidates'][:10]
    for r in qrows:
        r.pop('offsets', None)
    out = {'schema': 'p515_s53_w176_manuscript_number_check', 'version': 1, 'declarations_version': DECLARATIONS_VERSION,
           'stage': 'P5.15 Step 6 W176b -- number check of the manuscript .tex files at Overleaf 8b76423 (declarations v4)',
           'utc': datetime.now(timezone.utc).isoformat(), 'git_head': _git('rev-parse', 'HEAD'),
           'script': {'path': SCRIPT_REL, 'sha256': _sha(SCRIPT_REL), 'committed_clean': script_clean,
                      'imports': {W174_SCRIPT: _sha(W174_SCRIPT), W174.W171_SCRIPT: _sha(W174.W171_SCRIPT)}},
           'command_line': sys.argv,
           'manuscript': {'dir': mdir, 'commit': head, 'declared_commit': args.overleaf_commit, 'files': mfiles, 'roles': roles,
                          'bibliography': bib_rec, 'clone_porcelain_tex_bib': porcelain, 'edited': False,
                          'w174a_pin': {'commit': W174_COMMIT_IN_CLONE, 'main_sha256': main_v3_sha},
                          'w171a_pin': {'commit': W171_COMMIT_IN_CLONE, 'main_sha256': main_v2_sha}},
           'docx': {k: v for k, v in dx.items() if k not in ('text', 'para_of')},
           'frozen_tables': {'path': W164.FZ_REL, 'sha256': W164.FZ_SHA, 'numeric_leaves': len(leaves)},
           'inputs': inputs, 'cost_workbook_scan': X.xlsx_scan, 'bib_nrel_ess_costs': X.bib_entry,
           'map_citations': map_cite_lines, 'map_citation_failures': map_cite_fail,
           'main_regions': {'sha256': MAIN_LINE_RULES_SHA,
                            'revised': [{'key': k, 'lines': list(r), 'what': d_, 'heading_ok': region_heads_ok[k]} for k, r, d_, _h in REVISED],
                            'line_rules_mapped_from_v3': {k: (v if not isinstance(v, dict) else v) for k, v in mapped.items()},
                            'kept': [{'old': list(o_[0]), 'new': list(n_[0]), 'key': n_[1], 'kind': n_[2], 'why': n_[3]}
                                     for o_, n_ in zip([r for r in kept3 if r[1:] in {k[1:] for k in kept}], kept)],
                            'dropped': dropped, 'network_tables': NETWORK_TABLES, 'appendix_BCD': APP_BCD,
                            'value_table_lines': VALUE_TABLE_LINES},
           'changed_lines': {MAIN: changed_main, LETTER: changed_letter},
           'conventions': {
               'column': '1-based', 'line': '1-based',
               'written': 'the token as written, sign included, LaTeX thousands separators as ","',
               'precedence': 'as W174a: excluded structure -> declared checks (v3 carried, v4) -> rules -> auto index '
                             '(main.tex outside the revised passages) -> leftover. A revised-passage token left unassigned '
                             'fails the run.',
               'value_table': 'rows per value (numeric token) and per named setting; match = equal to the source; approximate = '
                              'the source rounded at the printed precision (stated); MISMATCH; no source found (where looked).',
               'reconciliation': 'W174a tokens matched to this run by (line mapped cb237b2d -> 7aa9e105 through difflib equal '
                                 'blocks, column, text as written)',
               'objective_convention': X.fz.get('objective_convention')},
           'summary': summary, 'checks': checks, 'failed': failed, 'relocation': relocation, 'declarations_applied': decl_out,
           'value_table': vtable, 'value_range_other_tokens': value_range_other, 'value_table_not_rows': VALUE_EXCLUDED,
           'value_table_inputs_cited': [{'path': a, 'sha256': b, 'git_blob_at_HEAD': c} for a, b, c in vt_inputs],
           'prediction': prediction,
           'reconciliation': {'submitted_version': rec_sub, 'approximate': rec_apx, 'submitted_regions_note': sv_regions_note,
                              'w174a_per_file': w174_pf},
           'quotations': qrows, 'parameter_table': ptable, 'parameter_table_error': ptable_error,
           'letter_findings': lf, 'main_findings': mf, 'tokens': toks, 'guards': guards, 'pickle_guard': pk,
           'exit_code': code, 'wall_s': time.time() - t0}
    written = {}

    def wr(name, data):
        rel = os.path.join(out_dir, OUT_NAMES[name])
        with open(os.path.join(REPO, rel), 'xb') as h:
            h.write(data if isinstance(data, bytes) else data.encode('utf-8'))
        written[rel] = _sha(rel) if not os.path.isabs(rel) else _sha_bytes(open(rel, 'rb').read())
    wr('json', GRIO.dumps(out, indent=1, sort_keys=True) + '\n')
    wr('md', md_summary(out))
    man = dict(written)
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    for v in inputs.values():
        man[v['path']] = v['sha256']
    for m in mfiles:
        man[m['path']] = m['sha256']
    man[bib_rec['path']] = bib_rec['sha256']
    wr('man', GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    for n, c in per_file.items():
        _log(f"[{tag}] {n} ({c['scope']}): tokens {c['tokens']} (excluded {c['excluded']}, in scope {c['in_scope_body']} body + "
             f"{c['in_scope_comment']} comment) -- match {c['match_total']} (declared {c['match_declared']}, rule {c['match_rule']}, "
             f"auto unique {c['match_auto_unique']}, auto ambiguous {c['match_auto_ambiguous']}), MISMATCH {c['MISMATCH']} (vs "
             f"reviewers' document {c['MISMATCH_vs_reviewers_document']}), approximate {c['approximate']}, no table counterpart "
             f"{c['no table counterpart']}, submitted-version {c['submitted-version figure']}, reviewer quotation verified "
             f"{c['reviewer quotation, verified']}, unchecked {c['unchecked']}, no source found {c['no source found']}, "
             f"unassigned {c['unassigned']}")
    for t in toks:
        if t['status'] in ('MISMATCH', 'approximate', 'no table counterpart', 'no source found') and \
                (t['file'] != MAIN or t.get('rule') == 'declared'):
            _log(f"[{tag}]   {t['file']}:{t['line']}:{t['col']} {t['status_display']}: written {t['written']!r} -> "
                 f"{t.get('value_at_written_precision')!r} [{t.get('check_id')}] {str(t.get('counterpart'))[:200]}")
    _log(f"[{tag}] relocation of v3 declarations: {summary['relocation']}; " +
         '; '.join(f"{r['id']} {r['outcome']} -> {r['v4']}" for r in relocation if r['outcome'] != 'carried'))
    _log(f"[{tag}] v4 declarations applied: {[(d['id'], d['line'], [x[2] for x in d['tokens']]) for d in decl_out if d['version'] == 4]}")
    _log(f"[{tag}] quotation segments: {summary['quotation_segments']}; quotation tokens: {summary['quotation_tokens']}")
    for r in qrows:
        if r['status'] != 'verbatim':
            _log(f"[{tag}]   quotation {r['quote']}.{r['segment']} (l. {r['lines']}) {r['status']} ratio {r['ratio']:.3f}: " +
                 '; '.join(f"{x['op']} {x['letter']!r} -> {x['document']!r}" for x in r['diffs'][:6]))
    _log(f"[{tag}] value table (l. {VALUE_TABLE_LINES[0]}-{VALUE_TABLE_LINES[1]}): {len(vtable)} rows {dict(vstat)}; values "
         f"{dict(vstat_values)}; named settings {dict(vstat_settings)}")
    for r in vtable:
        if r['status'] != 'match' or r['declaration_version'] == 4:
            _log(f"[{tag}]   {r['id']} l. {r['line']} {r['quantity']}: {r['status']} -- printed {r['printed']!r}, in force "
                 f"{str(r['value_in_force'])[:120]}; {r['precision'] or ''} {(r['note'] or '')[:200]}")
    _log(f"[{tag}] submitted-version: W174a {rec_sub['n_w174a']} -> W176 {rec_sub['n_w176']} (same {rec_sub['n_same']}, removed "
         f"{len(rec_sub['removed'])}, added {len(rec_sub['added'])}); by section {rec_sub['by_section_w176']}")
    _log(f"[{tag}] approximate: W174a {rec_apx['n_w174a']} -> W176 {rec_apx['n_w176']} (same {rec_apx['n_same']}, removed "
         f"{[(x['old_line'], x['written']) for x in rec_apx['removed']]}, added {[(x['new_line'], x['written']) for x in rec_apx['added']]})")
    for it in pred_items:
        _log(f"[{tag}] PREDICTION {it['item']}: W174a {it['w174a']} / W176 {it['w176']} -- {it['outcome']}")
    _log(f"[{tag}] parameter checks re-evaluated: {summary['parameter_table']}" + (f' ERROR {ptable_error}' if ptable_error else ''))
    _log(f"[{tag}] regions kept {[(r[1], r[0]) for r in kept]}; dropped {[(d_['key'], d_['old']) for d_ in dropped]}; line rules "
         f"mapped {mapped}")
    _log(f"[{tag}] main.tex lines changed since 260bd83: {len(changed_main)} ({changed_main}); letter: {changed_letter}")
    for f in lf + mf:
        _log(f"[{tag}]   finding {f['id']} (l. {f['lines']}, {f['kind']}): {f['finding'][:300]}")
    for k, v in checks.items():
        _log(f'[{tag}] check {k}: {v}')
    if unassigned:
        _log(f'[{tag}] UNASSIGNED tokens ({len(unassigned)}): ' + ', '.join(f"{by_key[k]['key']} {by_key[k]['written']!r}"
                                                                          for k in unassigned[:400]))
    if stale or claimed_twice or map_cite_fail or dup_ids or unreplaced or replaced_stale or retarget_clash:
        _log(f'[{tag}] stale {stale}; claimed twice {claimed_twice}; map citation failures {map_cite_fail}; duplicate ids '
             f'{dup_ids}; unreplaced v3 {unreplaced}; replacement entries for carried {replaced_stale}; retarget clash '
             f'{retarget_clash}')
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
