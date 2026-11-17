"""
P5.15 Addendum 80 order, Planner task W181b (W181 items (c) and (d)) -- the five round-7 sentences re-rated, and the
ageing citations of Section 3.5, at the Overleaf HEAD that carries main_e5322f8_round7.tex and letter v8.
ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT. A short check, not a re-audit (Addendum 80 ruling 5).

(c) The five touched sentences, each MATCH / MISMATCH / L, against its exact source:
  S1 Section 2.3.3 "Without storage the closure row is deactivated and its slacks are fixed at zero."
     -- network.py:511 (row built, unconditionally in _build_model); model_construction_helpers.py:1143 (fix at 0),
        :1166 (slack bound [0, 0]), :1176 (deactivate), each inside configure_shared_ess_operational_state under the
        inactive branch; the inactive test, the candidate path (network_data._update_model_with_candidate_solution),
        and the same lines at the git heads of the five x = 0 runs (W179c code_identity_at_run_heads, hash-verified).
  S2 Table 13 "(cycle 72 for the plan without storage, 69 for the unit)", 236.7 k EUR
     -- the T11 pair's campaign_results.json (per_cell.x0 / n7_4h_e1 cycles_run and certification_cycle;
        value_and_R.value_eur) and the frozen JSON label (three_by_three.rows[0], "cycle 72", not edited).
  S3 Appendix E.4 second phase "a re-evaluation ... with its decisions held at the first phase and its voltage,
     thermal and no-reverse-flow rows switched off" -- p515_s53_w116_benchmark_nrf.stage_nrf_arm phase B and
     uncoordinated_benchmark.reevaluate_dso_at_actual_voltage / build_consistency_reevaluation_block (source read as
     text), plus the six committed NRF arm records (relaxed_row_families, n_decisions_fixed, module sha256 at run).
  S4 Table E.21 note "Evaluations~1 and~2 continue earlier runs, and evaluations~40 and~41 are re-decided from the
     records of earlier runs" -- W180b's committed JSON (hash-verified) and W163 N49 / W101 summary.
  S5 Letter R2.2 "the cycles of every single-scenario evaluation behind a reported comparison in Appendix~E" --
     R22-b re-scored with W180c's definition; the letter pointer multiset of v8 against v7.
(d) Ageing citations in Section 3.5 / around tab:ageing_calibrations: the three statements, every \\cite key in
    Section 3.5 and whether it resolves in the clone's bibliography.bib; the restored % [AUTHOR] comment; the counts
    of "[CONFIRM" and "[AUTHOR" in main.tex and the letter.

Negative controls (in memory): an altered sentence, an altered code line number, swapped Table 13 cycles, an altered
campaign_results cycle count, a dropped row family in E.4, the old E.21 note, the v7 letter wording, and an
unresolvable \\cite key -- each must be refused.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0; pickle.load /
loads blocked and counted; no production module imported (asserted; production source is read as text and parsed
with ast, never executed); JSON written only through gate_result_io. Refuses to run unless the output directory exists
and holds nothing but launch.log, and when another copy of this script is running. Read-only on the clone.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w181b_sentences_citations && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w181b_sentences_citations.py \\
      > data/SRP1/Results/P515S53/w181b_sentences_citations/launch.log 2>&1
Optional: --out-dir <dir> (scratch trials only).
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W181b sentences and citations (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W181b: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W181b: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

SCRIPT = os.path.basename(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w181b_sentences_citations')

# ------------------------------------------------------------------------------------------------------- identity
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
BIB = os.path.join(CLONE, 'bibliography.bib')
DECLARED_HEAD = 'cd57efcb54399cb652a0ac2dcd64e5d3a78d89fc'
PREVIOUS_HEAD = 'e5322f85dced8034312c9afb97c19001bce6bfa0'
ROUND7_TEX = os.path.join('manuscript_review', 'main_e5322f8_round7.tex')
LETTER_V8 = os.path.join('manuscript_review', 'response_to_reviewers_v8.tex')
LETTER_V7 = os.path.join('manuscript_review', 'response_to_reviewers_v7.tex')

# ------------------------------------------------------------------------------------------------------- sources
P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
T11_ROOT = os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair')
T11_RESULTS = os.path.join(T11_ROOT, 'campaign_results.json')
W169_CELLS = os.path.join(P53, 'w169_slack_inventory', 'cells')
W179C_JSON = os.path.join(P53, 'w179c_closure_scope', 'w179c_closure_scope.json')
W179C_MANIFEST = os.path.join(P53, 'w179c_closure_scope', 'manifest_sha256.json')
W179C_SHA = '417c270f8fae66c884271aa403ea15e92931af531ff70397300ef653c3f4e19c'   # P5_15_W179C / W180D reports
W180B_JSON = os.path.join(P53, 'w180b_row16_class', 'w180b_row16_class.json')
W180B_MANIFEST = os.path.join(P53, 'w180b_row16_class', 'manifest_sha256.json')
W180B_SHA = 'bd6343ed7970b343311afe025309c9729d298dfb770db148895574469252b639'   # P5_15_W180B_ROW16_CLASS.md
W180C_JSON = os.path.join(P53, 'w180c_text_letter', 'w180c_text_letter.json')
W180C_MANIFEST = os.path.join(P53, 'w180c_text_letter', 'manifest_sha256.json')
W180C_SHA_PREFIX = '88e1d1d9'                                                     # P5_15_W180C_TEXT_LETTER.md
W163_JSON = os.path.join(P53, 'w163_paragraphs_v4_check', 'w163_paragraphs_v4_figure_check.json')
W101_SUMMARY = os.path.join(P53, 'w101_srp1_continuation', 'w101_three_reference_summary.json')
W116_ROOT = os.path.join(P53, 'w116_benchmark_nrf')
W116_RUNS = tuple(f'nrf_arm_{arm}_{start}_r2' for arm in ('passive', 'price_taker')
                  for start in ('cold', 'perturbed', 'warm_from_certified'))
W116_SCRIPT = 'p515_s53_w116_benchmark_nrf.py'
UB_SRC = 'uncoordinated_benchmark.py'

FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions',
                             'uncoordinated_benchmark', 'p515_s53_w116', 'p515_s53_w93')

# ------------------------------------------------------------------------------------------------------- sentences
S1_SENTENCE = 'Without storage the closure row is deactivated and its slacks are fixed at zero.'
S1_OLD = 'Without storage the closure row is not built.'
S2_PHRASE = ('Value $Q(\\boldsymbol{0}) - Q(\\text{unit})$ at operational stopping (cycle 72 for the plan without '
             'storage, 69 for the unit) & 236.7~k\\euro & recorded \\\\')
S2_OLD = 'at operational stopping (cycle 72) &'
S2_PARSE = re.compile(r'at operational stopping \(cycle (\d+) for the plan without storage, (\d+) for the unit\) & '
                      r'([\d.]+)~k\\euro')
S3_PHRASE = ("a re-evaluation of each distribution network at the transmission network's interface voltage, with its "
             "decisions held at the first phase and its voltage, thermal and no-reverse-flow rows switched off, in "
             "which the exceedance columns count")
S3_OLD = "a re-solve of each distribution network at the transmission network's interface voltage with its no-reverse-flow rows switched off"
S3_FAMILIES = re.compile(r'with its decisions held at the first phase and its (.*?) rows switched off')
S3_WORD_TO_FAMILIES = {'voltage': ('voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons'),
                       'thermal': ('branch_flow_limit', 'branch_flow_limit_ji'),
                       'no-reverse-flow': ('uncoord_no_reverse_flow',)}
S4_PHRASE = ('Evaluations~1 and~2 continue earlier runs, and evaluations~40 and~41 are re-decided from the records of '
             'earlier runs; their first residual passes are not tabulated')
S4_OLD = 'Evaluations~1, 2, 40 and~41 continue earlier runs'
S4_PARSE = re.compile(r'Evaluations~(\d+) and~(\d+) continue earlier runs, and evaluations~(\d+) and~(\d+) are '
                      r're-decided from the records of earlier runs')
S5_PHRASE = 'and the cycles of every single-scenario evaluation behind a reported comparison in Appendix~E.'
S5_OLD = 'and the cycles of every evaluation behind a reported comparison in Appendix~E.'
S5_CLASSES = ('the evaluation ends at the production exit during the search, or, for the evaluations behind the '
              'reported comparisons, when the certification rule')
APPE_INTRO_MS = ('The multi-scenario evaluations, stopped by the production exit rather than certified, are reported '
                 'in Table~\\ref{tab:res_multiscenario}.')
E3_SEARCH_ONLY = 'were evaluated by the search only, on an earlier state of the code and stopped by the production exit'

# Code expectations (file, expected line at repo HEAD, exact stripped text, enclosing def, extra structural test key)
CODE = {
    'row_built': ('network.py', 511,
                  'model.sess_soc_final = pe.Constraint(model.shared_energy_storages, model.scenarios_market, '
                  'model.scenarios_operation, rule=partial(sess_soc_final_rule, network=network, params=params))',
                  '_build_model'),
    'fix_at_0': ('model_construction_helpers.py', 1143, 'entry.fix(0.0)', 'configure_shared_ess_operational_state'),
    'bounds_00': ('model_construction_helpers.py', 1166,
                  'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE',
                  'configure_shared_ess_operational_state'),
    'setlb_0': ('model_construction_helpers.py', 1168, 'entry.setlb(0.0)', 'configure_shared_ess_operational_state'),
    'setub_slack_ub': ('model_construction_helpers.py', 1169, 'entry.setub(slack_ub)',
                       'configure_shared_ess_operational_state'),
    'deactivate': ('model_construction_helpers.py', 1176, 'entry.deactivate()', 'configure_shared_ess_operational_state'),
    'inactive_test': ('model_construction_helpers.py', 1069,
                      'return abs(s_capacity) <= tolerance or abs(e_capacity) <= tolerance',
                      'shared_ess_capacity_is_inactive'),
    'inactive_flag': ('model_construction_helpers.py', 1121,
                      'inactive = shared_ess_capacity_is_inactive(s_capacity, e_capacity)',
                      'configure_shared_ess_operational_state'),
}
W116_CODE = {   # p515_s53_w116_benchmark_nrf.py, stage_nrf_arm
    'phase_A_open': (565, "account.open_phase('A_nrf_arm', dso_labels + tso_labels, (f'{arm}:{start}:dso', f'{arm}:{start}:tso'))"),
    'phase_A_solve': (566, 'arm_out = UB.run_operational_planning_uncoordinated('),
    'consistency_branch': (614, 'if consistency:'),
    'tn_voltage': (615, "mismatch = UB.interface_voltage_mismatch(planning, arm_out['models']['tso'], arm_out['models']['dso'])"),
    'phase_B_open': (616, "account.open_phase('B_consistency_reevaluation', dso_labels, ('consistency:reevaluation',))"),
    'phase_B_call': (617, "reeval = UB.reevaluate_dso_at_actual_voltage(planning, arm_out['models']['dso'], mismatch['v_actual_dn_pu'],"),
}
UB_CODE = {     # uncoordinated_benchmark.py
    'voltage_rows': (235, "VOLTAGE_LIMIT_ROWS = ('voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons')"),
    'thermal_rows': (236, "THERMAL_LIMIT_ROWS = ('branch_flow_limit', 'branch_flow_limit_ji')"),
    'nrf_row': (212, "NO_REVERSE_FLOW_ROW = 'uncoord_no_reverse_flow'"),
    'builder_def': (1411, 'def build_consistency_reevaluation_block(dso_block, network, *, v_actual_dn_pu):'),
    'clone': (1427, 'm = dso_block.clone()'),
    'decision_loop': (1434, 'for family in DSO_DECISION_VAR_FAMILIES:'),
    'decision_fix': (1442, 'var_data.fix(var_data.value, skip_validation=True)'),
    'nonref_gen_fix': (1452, 'var_data.fix(var_data.value, skip_validation=True)'),
    'limit_loop': (1487, 'for family in VOLTAGE_LIMIT_ROWS + THERMAL_LIMIT_ROWS:'),
    'limit_deactivate': (1492, 'con_data.deactivate()'),
    'nrf_get': (1495, 'nrf_component = getattr(m, NO_REVERSE_FLOW_ROW, None)'),
    'nrf_deactivate': (1498, 'con_data.deactivate()'),
    'reeval_def': (1580, 'def reevaluate_dso_at_actual_voltage(planning_problem, dso_models, v_actual_dn_pu, *, tolerances,'),
    'reeval_arm_block': (1591, 'arm_block = dso_models[node_id][year][day]'),
    'reeval_build': (1592, 'clone, build = build_consistency_reevaluation_block('),
}

# ------------------------------------------------------------------------------------------------------- registry
INPUTS = {}
_TRACKED = None


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(path):
    with open(path, 'rb') as f:
        return _sha_bytes(f.read())


def _git(args, cwd=REPO, binary=False):
    r = subprocess.run(['git'] + list(args), cwd=cwd, check=True, capture_output=True, text=not binary)
    return r.stdout


def _tracked(path):
    global _TRACKED
    if _TRACKED is None:
        _TRACKED = set(_git(['ls-files', '-z']).split('\0'))
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel.startswith(CLONE + os.sep):
        return 'overleaf clone (tracked there; read only)'
    return 'committed' if rel in _TRACKED else 'UNCOMMITTED'


def _reg(path):
    rel = os.path.relpath(os.path.abspath(path), REPO)
    if rel not in INPUTS:
        INPUTS[rel] = {'sha256': _sha(path), 'status': _tracked(path)}
    return INPUTS[rel]


def _reg_blob(label, data, status):
    INPUTS[label] = {'sha256': _sha_bytes(data), 'status': status}


def _read(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _ck(name, ok, detail=None):
    return {'check': name, 'ok': bool(ok), 'detail': detail}


def _rating(checks):
    return 'MATCH' if checks and all(c['ok'] for c in checks) else 'MISMATCH'


def _lines_with(lines, needle):
    return [i + 1 for i, ln in enumerate(lines) if needle in ln]


# ------------------------------------------------------------------------------------------------------- identity
def identity():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    parent = _git(['rev-parse', 'HEAD^'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', 'main.tex', 'response_to_reviewers_draft.tex', 'bibliography.bib'],
                  cwd=CLONE).strip()
    out = {'clone': CLONE, 'head': head, 'declared_head': DECLARED_HEAD, 'parent': parent}
    blobs = {}
    for f in ('main.tex', 'response_to_reviewers_draft.tex', 'bibliography.bib'):
        for rev in (DECLARED_HEAD, PREVIOUS_HEAD):
            b = _git(['show', f'{rev}:{f}'], cwd=CLONE, binary=True)
            _reg_blob(f'{CLONE}@{rev[:7]}:{f}', b, 'overleaf clone git object (read only)')
            blobs[(f, rev)] = b
    with open(MAIN_TEX, 'rb') as fh:
        main_b = fh.read()
    with open(LETTER, 'rb') as fh:
        letter_b = fh.read()
    with open(BIB, 'rb') as fh:
        bib_b = fh.read()
    _reg(MAIN_TEX), _reg(LETTER), _reg(BIB), _reg(ROUND7_TEX), _reg(LETTER_V8), _reg(LETTER_V7)
    checks = [
        _ck('clone HEAD = declared cd57efc', head == DECLARED_HEAD, head),
        _ck('HEAD^ = e5322f8 (one commit after)', parent == PREVIOUS_HEAD, parent),
        _ck('main.tex, letter, bib unmodified in the clone', status == '', status),
        _ck('main.tex working copy = HEAD blob', main_b == blobs[('main.tex', DECLARED_HEAD)]),
        _ck('letter working copy = HEAD blob', letter_b == blobs[('response_to_reviewers_draft.tex', DECLARED_HEAD)]),
        _ck('bibliography.bib working copy = HEAD blob', bib_b == blobs[('bibliography.bib', DECLARED_HEAD)]),
        _ck('main.tex = manuscript_review/main_e5322f8_round7.tex', _sha_bytes(main_b) == INPUTS[ROUND7_TEX]['sha256']),
        _ck('letter = manuscript_review/response_to_reviewers_v8.tex', _sha_bytes(letter_b) == INPUTS[LETTER_V8]['sha256']),
        _ck('letter at e5322f8 = response_to_reviewers_v7.tex',
            _sha_bytes(blobs[('response_to_reviewers_draft.tex', PREVIOUS_HEAD)]) == INPUTS[LETTER_V7]['sha256']),
        _ck('bibliography.bib unchanged e5322f8 -> HEAD',
            blobs[('bibliography.bib', DECLARED_HEAD)] == blobs[('bibliography.bib', PREVIOUS_HEAD)]),
    ]
    out.update({'main_sha256': _sha_bytes(main_b), 'letter_sha256': _sha_bytes(letter_b), 'bib_sha256': _sha_bytes(bib_b),
                'main_n_lines': main_b.decode('utf-8').count('\n'), 'checks': checks, 'ok': all(c['ok'] for c in checks)})
    prev = {'main': blobs[('main.tex', PREVIOUS_HEAD)].decode('utf-8').split('\n'),
            'letter': blobs[('response_to_reviewers_draft.tex', PREVIOUS_HEAD)].decode('utf-8').split('\n')}
    return out, main_b.decode('utf-8').split('\n'), letter_b.decode('utf-8').split('\n'), bib_b.decode('utf-8'), prev


# ------------------------------------------------------------------------------------------------------- AST helpers
def _ancestors(tree, lineno):
    """Chain of nodes (outermost first) whose span contains `lineno`, ending at the statement that starts there."""
    chain = []

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, ast.stmt):
                continue
            if child.lineno <= lineno <= getattr(child, 'end_lineno', child.lineno):
                chain.append(child)
                visit(child)
                return
    visit(tree)
    return chain


def _in_body_of_if(chain, lineno, test_name):
    """True if the innermost If in `chain` tests the bare Name `test_name` and `lineno` lies in its body."""
    ifs = [n for n in chain if isinstance(n, ast.If)]
    if not ifs:
        return False
    node = ifs[-1]
    in_body = any(s.lineno <= lineno <= s.end_lineno for s in node.body)
    return isinstance(node.test, ast.Name) and node.test.id == test_name and in_body


def _module_literal(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def _for_iter_src(src, chain):
    return [ast.get_source_segment(src, n.iter) for n in chain if isinstance(n, ast.For)]


def code_checks_at(srcs, code=CODE):
    """srcs: {'network.py': text, 'model_construction_helpers.py': text, 'definitions.py': text, 'network_data.py': text}."""
    trees = {k: ast.parse(v) for k, v in srcs.items()}
    lines = {k: v.split('\n') for k, v in srcs.items()}
    res = {}
    checks = []
    for key, (fname, line, text, func) in code.items():
        got = lines[fname][line - 1].strip() if 0 < line <= len(lines[fname]) else None
        chain = _ancestors(trees[fname], line)
        funcs = [n.name for n in chain if isinstance(n, ast.FunctionDef)]
        res[key] = {'file': fname, 'line': line, 'text_at_line': got, 'enclosing_def': funcs[-1] if funcs else None,
                    'for_iters': _for_iter_src(srcs[fname], chain),
                    'all_lines_with_text': _lines_with([ln.strip() for ln in lines[fname]], text)
                    if len(text) > 30 else None}
        checks.append(_ck(f'{key}: {fname}:{line} reads "{text[:60]}"', got == text, got))
        checks.append(_ck(f'{key}: inside def {func}', bool(funcs) and funcs[-1] == func, funcs))
    # structure
    rb = _ancestors(trees['network.py'], code['row_built'][1])
    checks.append(_ck('row_built: a direct statement of _build_model (no if / loop around it: built for every unit)',
                      len(rb) == 2 and isinstance(rb[0], ast.FunctionDef) and rb[0].name == '_build_model',
                      [type(n).__name__ for n in rb]))
    mch = 'model_construction_helpers.py'
    ops_vars = _module_literal(trees[mch], '_SHARED_ESS_OPERATIONAL_VARIABLES') or ()
    ops_cons = _module_literal(trees[mch], '_SHARED_ESS_OPERATIONAL_CONSTRAINTS') or ()
    fx = _ancestors(trees[mch], code['fix_at_0'][1])
    checks.append(_ck('fix_at_0: under "if inactive:" in the loop over _SHARED_ESS_OPERATIONAL_VARIABLES',
                      _in_body_of_if(fx, code['fix_at_0'][1], 'inactive')
                      and '_SHARED_ESS_OPERATIONAL_VARIABLES' in _for_iter_src(srcs[mch], fx), _for_iter_src(srcs[mch], fx)))
    checks.append(_ck('both closure slacks are in _SHARED_ESS_OPERATIONAL_VARIABLES',
                      {'slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down'} <= set(ops_vars)))
    bd = _ancestors(trees[mch], code['bounds_00'][1])
    checks.append(_ck('bounds_00: in the loop over the two closure slacks; setlb(0.0) and setub(slack_ub) follow in it',
                      "('slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down')" in _for_iter_src(srcs[mch], bd)
                      and all(_for_iter_src(srcs[mch], _ancestors(trees[mch], code[k][1]))[:1] == _for_iter_src(srcs[mch], bd)[:1]
                              for k in ('setlb_0', 'setub_slack_ub')), _for_iter_src(srcs[mch], bd)))
    dc = _ancestors(trees[mch], code['deactivate'][1])
    checks.append(_ck('deactivate: under "if inactive:" in the loop over _SHARED_ESS_OPERATIONAL_CONSTRAINTS',
                      _in_body_of_if(dc, code['deactivate'][1], 'inactive')
                      and '_SHARED_ESS_OPERATIONAL_CONSTRAINTS' in _for_iter_src(srcs[mch], dc), _for_iter_src(srcs[mch], dc)))
    checks.append(_ck("'sess_soc_final' (the closure row) is in _SHARED_ESS_OPERATIONAL_CONSTRAINTS", 'sess_soc_final' in ops_cons))
    tol = _module_literal(trees['definitions.py'], 'SHARED_ESS_ZERO_CAPACITY_TOLERANCE')
    checks.append(_ck('inactive test tolerance SHARED_ESS_ZERO_CAPACITY_TOLERANCE = 1e-10 (a zero capacity is inactive)',
                      tol == 1e-10, tol))
    # candidate path
    nd = srcs['network_data.py']
    tnd = trees['network_data.py']
    fn = next((n for n in ast.walk(tnd) if isinstance(n, ast.FunctionDef) and n.name == '_update_model_with_candidate_solution'), None)
    seg = ast.get_source_segment(nd, fn) if fn else ''
    checks.append(_ck('candidate path: _update_model_with_candidate_solution calls configure_shared_ess_operational_state '
                      'with |candidate s, e| / s_base on both the TN and the DN branch',
                      seg.count('configure_shared_ess_operational_state(') == 2
                      and seg.count("abs(candidate_solution[node_id][year]['s']) / s_base") == 1
                      and seg.count("abs(candidate_solution[tn_node_id][year]['e']) / s_base") == 1,
                      {'n_calls': seg.count('configure_shared_ess_operational_state('), 'def_line': fn.lineno if fn else None}))
    res['ops_variables'] = list(ops_vars)
    res['ops_constraints'] = list(ops_cons)
    res['zero_capacity_tolerance'] = tol
    res['candidate_path_def_line'] = fn.lineno if fn else None
    return checks, res


def _repo_srcs(rev=None):
    out = {}
    for f in ('network.py', 'model_construction_helpers.py', 'definitions.py', 'network_data.py'):
        if rev is None:
            out[f] = _read(f)
        else:
            b = _git(['show', f'{rev}:{f}'], binary=True)
            _reg_blob(f'repo@{rev[:8]}:{f}', b, 'committed (git object)')
            out[f] = b.decode('utf-8')
    return out


# ------------------------------------------------------------------------------------------------------- S1
def s1(main_lines, srcs_head, x0_heads, code=CODE):
    hits = _lines_with(main_lines, S1_SENTENCE)
    checks = [_ck('sentence present exactly once', len(hits) == 1, hits),
              _ck('old wording ("not built") absent', not _lines_with(main_lines, S1_OLD))]
    c_head, res = code_checks_at(srcs_head, code)
    checks += [dict(c, check='repo HEAD: ' + c['check']) for c in c_head]
    per_head = {}
    for cell, head in x0_heads.items():
        diff = _git(['diff', '--stat', head, 'HEAD', '--', 'network.py', 'model_construction_helpers.py',
                     'definitions.py', 'network_data.py']).strip()
        c_run, _ = code_checks_at(_repo_srcs(head), code)
        per_head[cell] = {'head': head, 'diff_stat_vs_HEAD': diff or 'NO CHANGE', 'all_code_checks_hold': all(c['ok'] for c in c_run)}
        checks.append(_ck(f'x = 0 run {cell} at {head[:8]}: same lines hold, and the four files are unchanged to HEAD',
                          diff == '' and all(c['ok'] for c in c_run), per_head[cell]))
    return {'sentence_lines': hits, 'code': res, 'x0_run_heads': per_head, 'checks': checks, 'rating': _rating(checks)}


# ------------------------------------------------------------------------------------------------------- S2
def s2(main_lines, cr, frozen, w169):
    hits = _lines_with(main_lines, S2_PHRASE)
    line = main_lines[hits[0] - 1] if len(hits) == 1 else ''
    m = S2_PARSE.search(line)
    x0, unit = cr['per_cell'].get('x0', {}), cr['per_cell'].get('n7_4h_e1', {})
    v = cr['value_and_R']['value_eur']
    fz_row = frozen['tables']['three_by_three']['rows'][0]
    evals = sorted(os.listdir(os.path.join(T11_ROOT, 'evals')))
    checks = [
        _ck('Table 13 row present exactly once (tab:res_multiscenario)', len(hits) == 1, hits),
        _ck('old "(cycle 72) &" absent', not _lines_with(main_lines, S2_OLD)),
        _ck('printed x = 0 cycle = per_cell.x0.cycles_run = certification_cycle',
            bool(m) and int(m.group(1)) == x0.get('cycles_run') == x0.get('certification_cycle'),
            {'printed': m.group(1) if m else None, 'cycles_run': x0.get('cycles_run'),
             'certification_cycle': x0.get('certification_cycle')}),
        _ck('printed unit cycle = per_cell.n7_4h_e1.cycles_run = certification_cycle',
            bool(m) and int(m.group(2)) == unit.get('cycles_run') == unit.get('certification_cycle'),
            {'printed': m.group(2) if m else None, 'cycles_run': unit.get('cycles_run'),
             'certification_cycle': unit.get('certification_cycle')}),
        _ck('both cells certified (production exit of the 3 x 3 pair)', x0.get('status') == unit.get('status') == 'certified'),
        _ck('value_eur = Q(x0) - Q(unit)', abs((x0['Q'] - unit['Q']) - v) <= 1e-6, {'Q_x0': x0['Q'], 'Q_unit': unit['Q'], 'value_eur': v}),
        _ck('236.7 k EUR = value_eur rounded to 0.1 k EUR', bool(m) and m.group(3) == f'{v / 1000:.1f}', m.group(3) if m else None),
        _ck('frozen JSON three_by_three.rows[0]: same value, label still "(cycle 72)" (frozen, not edited)',
            fz_row['value'] == v and '(cycle 72)' in fz_row['quantity'], fz_row['quantity']),
        _ck('instance: W169 eval keys of t11_3x3_x0 / t11_3x3_unit name the campaign eval dirs; x0 all-zero, unit node 7 [0.25, 1]',
            any(e.startswith(w169['t11_3x3_x0']['eval_key'][:16]) and e.endswith('_x0') for e in evals)
            and any(e.startswith(w169['t11_3x3_unit']['eval_key'][:16]) and e.endswith('_n7_4h_e1') for e in evals)
            and all(vv == [0.0, 0.0] for vv in w169['t11_3x3_x0']['candidate_canonical']['nodes'].values())
            and w169['t11_3x3_unit']['candidate_canonical']['nodes']['7'] == [0.25, 1.0], evals),
    ]
    return {'line': hits, 'printed': list(m.groups()) if m else None, 'campaign_results': T11_RESULTS,
            'git_head_at_run': cr.get('git_head_at_run'), 'objective_convention': cr.get('objective_convention'),
            'instance': {c: {k: w169[c][k] for k in ('eval_key', 'candidate_key', 'candidate_canonical')} for c in w169},
            'checks': checks, 'rating': _rating(checks)}


# ------------------------------------------------------------------------------------------------------- S3
def s3(main_lines, w116_src, ub_src, runs, w116_code=W116_CODE, ub_code=UB_CODE):
    hits = _lines_with(main_lines, S3_PHRASE)
    line = main_lines[hits[0] - 1] if len(hits) == 1 else ''
    fm = S3_FAMILIES.search(line)
    words = [w.strip() for w in re.split(r',| and ', fm.group(1))] if fm else []
    words = [w for w in words if w]
    named = sorted(f for w in words for f in S3_WORD_TO_FAMILIES.get(w, ('UNKNOWN:' + w,)))
    wl, ul = w116_src.split('\n'), ub_src.split('\n')
    checks = [_ck('E.4 sentence present exactly once', len(hits) == 1, hits),
              _ck('old wording ("a re-solve ... with its no-reverse-flow rows switched off") absent', not _lines_with(main_lines, S3_OLD))]
    for k, (ln, text) in w116_code.items():
        got = wl[ln - 1].strip() if 0 < ln <= len(wl) else None
        checks.append(_ck(f'W116 {k}: {W116_SCRIPT}:{ln}', got == text, got))
    for k, (ln, text) in ub_code.items():
        got = ul[ln - 1].strip() if 0 < ln <= len(ul) else None
        checks.append(_ck(f'UB {k}: {UB_SRC}:{ln}', got == text, got))
    ut = ast.parse(ub_src)
    voltage = _module_literal(ut, 'VOLTAGE_LIMIT_ROWS') or ()
    thermal = _module_literal(ut, 'THERMAL_LIMIT_ROWS') or ()
    nrf = _module_literal(ut, 'NO_REVERSE_FLOW_ROW')
    code_families = sorted(list(voltage) + list(thermal) + [nrf])
    checks.append(_ck('row families named in the sentence = families the builder deactivates', named == code_families,
                      {'sentence_words': words, 'named': named, 'code': code_families}))
    # "held at the first phase": phase B's input is phase A's DSO models; the builder fixes DSO decisions on a clone
    fn = next(n for n in ast.walk(ut) if isinstance(n, ast.FunctionDef) and n.name == 'build_consistency_reevaluation_block')
    seg = ast.get_source_segment(ub_src, fn)
    checks.append(_ck('phase B is fed arm_out["models"]["dso"] (the phase-A solution) and no TSO solve',
                      "arm_out['models']['dso']" in wl[w116_code['phase_B_call'][0] - 1]
                      and 'solve_tso_model' not in '\n'.join(wl[w116_code['phase_B_open'][0] - 1: w116_code['phase_B_open'][0] + 3])))
    checks.append(_ck('builder fixes every DSO decision family and the non-reference generators at the arm value',
                      seg.count('var_data.fix(var_data.value, skip_validation=True)') == 2 and 'if g == ref_gen:' in seg))
    # run records (committed)
    rr = {}
    for run, d in runs.items():
        blocks = d['phase_B_consistency']['reevaluation']['blocks']
        fams = {tuple(sorted(b['build']['relaxed_row_families'])) for b in blocks.values()}
        rr[run] = {'git_head': d['git_head'], 'n_blocks': len(blocks), 'relaxed_row_families': [list(f) for f in fams],
                   'min_n_decisions_fixed': min(b['build']['n_decisions_fixed'] for b in blocks.values()),
                   'ub_sha_at_run_equals_HEAD': d['module_sha256'].get(UB_SRC) == _sha_bytes(ub_src.encode('utf-8')),
                   'w116_sha_at_run_equals_HEAD': d['module_sha256'].get(W116_SCRIPT) == _sha_bytes(w116_src.encode('utf-8')),
                   'arm_cost_source': d['arm_cost']['source']}
        checks.append(_ck(f'run {run}: 36 DSO blocks, relaxed families = the five, decisions fixed > 0, '
                          f'modules at run = HEAD bytes',
                          rr[run]['n_blocks'] == 36 and fams == {tuple(code_families)} and rr[run]['min_n_decisions_fixed'] > 0
                          and rr[run]['ub_sha_at_run_equals_HEAD'] and rr[run]['w116_sha_at_run_equals_HEAD'], rr[run]))
    not_in_sentence = {
        'slacks_fixed_at_zero': list(_module_literal(ut, 'DSO_REEVALUATION_ZERO_SLACKS') or ()),
        'bounds_relaxed': 'node-voltage (vmag_sqr, vmag, free e/f) and reference-generator P/Q bounds relaxed, originals recorded',
        'reference_generator_free': 'the reference generator (the TN supply) is not fixed: it carries the re-evaluated interface flow',
        'trivial_rows_deactivated': 'rows left without a free variable deactivated, residual recorded'}
    return {'line': hits, 'families': {'sentence': named, 'code': code_families}, 'runs': rr,
            'builder_also_does_not_in_sentence': not_in_sentence, 'checks': checks, 'rating': _rating(checks)}


# ------------------------------------------------------------------------------------------------------- S4
def _e21_block(lines):
    i_lab = next(i for i, ln in enumerate(lines) if '\\label{tab:appE_evaluations}' in ln)
    i0 = max(i for i in range(i_lab) if lines[i].startswith('\\begin{longtable}'))
    i1 = next(i for i in range(i_lab, len(lines)) if lines[i].startswith('\\end{longtable}'))
    return i0, i1


def _e21_rows(lines):
    i0, i1 = _e21_block(lines)
    rows = {}
    for ln in lines[i0:i1 + 1]:
        mm = re.match(r'^(\d+) & ', ln)
        if mm:
            rows[int(mm.group(1))] = [c.strip() for c in ln.rstrip().rstrip('\\').split(' & ')]
    return rows


def s4(main_lines, prev_main, w180b, w163_n49, w101):
    hits = _lines_with(main_lines, S4_PHRASE)
    line = main_lines[hits[0] - 1] if len(hits) == 1 else ''
    m = S4_PARSE.search(line)
    cont = [int(m.group(1)), int(m.group(2))] if m else []
    redec = [int(m.group(3)), int(m.group(4))] if m else []
    rows = {r['no']: r for r in w180b['rows']}
    n49_cells = sorted(w163_n49.keys())
    n49_keys = {c: w101['reports'][c]['eval_key'] for c in n49_cells}
    e21_now, e21_prev = _e21_rows(main_lines), _e21_rows(prev_main)
    i0, i1 = _e21_block(main_lines)
    j0, j1 = _e21_block(prev_main)
    changed = [k for k in range(max(i1 - i0, j1 - j0) + 1)
               if (main_lines[i0 + k] if i0 + k <= i1 else None) != (prev_main[j0 + k] if j0 + k <= j1 else None)]
    checks = [
        _ck('E.21 note sentence present exactly once', len(hits) == 1, hits),
        _ck('old note ("Evaluations~1, 2, 40 and~41 continue earlier runs") absent', not _lines_with(main_lines, S4_OLD)),
        _ck('E.21 block: 61 rows, identical to e5322f8; the only changed line is the note',
            len(e21_now) == 61 and e21_now == e21_prev and len(changed) == 1
            and S4_PHRASE in main_lines[i0 + changed[0]], {'changed_offsets': changed, 'n_rows': len(e21_now)}),
        _ck('W163 N49: the reference continuations are {x0, n7_4h_e1, c_star}, all replayed bitwise (G19)',
            n49_cells == ['c_star', 'n7_4h_e1', 'x0'] and all(v['G19'] for v in w163_n49.values()), w163_n49),
    ]
    for no in cont:
        r = rows.get(no, {})
        rc = r.get('reference_continuation', {})
        checks.append(_ck(f'evaluation {no} "continues an earlier run": W163 N49 reference continuation, eval key = W101 summary',
                          rc.get('is_reference_continuation_W163_N49') is True and rc.get('w101_cell') in n49_keys
                          and r.get('frozen', {}).get('eval_key') == n49_keys.get(rc.get('w101_cell')),
                          {'cell': r.get('cell'), 'w101_cell': rc.get('w101_cell'), 'class': r.get('class')}))
    for no in redec:
        r = rows.get(no, {})
        rc = r.get('reference_continuation', {})
        v6 = r.get('v6_decision', {})
        checks.append(_ck(f'evaluation {no} "re-decided from the records of earlier runs": not an N49 continuation; '
                          f'v6 decision on record (W142 item 1); issued under an earlier rule; printed Rule "record"',
                          rc.get('is_reference_continuation_W163_N49') is False
                          and r.get('frozen', {}).get('eval_key') not in n49_keys.values()
                          and v6.get('kind') == 'v6 on record (W142 item 1)' and r.get('issuing', {}).get('issued_rule_version', 6) < 6
                          and r.get('printed', {}).get('rule') == 'record' and v6.get('v6_equals_replay_arm_AB') is True,
                          {'cell': r.get('cell'), 'class': r.get('class'), 'v6_kind': v6.get('kind'),
                           'w142_record': v6.get('w142_record'), 'issued_rule_version': r.get('issuing', {}).get('issued_rule_version'),
                           'issuing_campaign_results': r.get('issuing', {}).get('campaign_results')}))
    checks.append(_ck('"their first residual passes are not tabulated": k_0 printed "---" for all four',
                      bool(cont + redec) and all(e21_now.get(no, [None] * 6)[5] == '---' for no in cont + redec),
                      {no: e21_now.get(no, [None] * 6)[5] for no in cont + redec}))
    checks.append(_ck('the four rows are 1, 2 (continuations) and 40, 41 (re-decided)', cont == [1, 2] and redec == [40, 41],
                      {'continue': cont, 're_decided': redec}))
    return {'line': hits, 'continue': cont, 're_decided': redec, 'n49_eval_keys': n49_keys, 'checks': checks,
            'rating': _rating(checks)}


# ------------------------------------------------------------------------------------------------------- S5
POINTER_RE = re.compile(r'(Sections?|Subsections?|Appendix|Appendices|Tables?|Figures?|Fig\.|Eqs?\.|Equations?|'
                        r'Algorithms?)(~|\s)\(?([A-Z](?:\.\d+)*|\d+(?:\.\d+)*[a-z]?)')
REF_RE = re.compile(r'\\(?:ref|eqref|cite)\{[^}]*\}')


def _pointers(lines):
    c = Counter()
    for i, ln in enumerate(lines):
        for mm in POINTER_RE.finditer(ln):
            c[(mm.group(1), mm.group(3))] += 1
        for mm in REF_RE.finditer(ln):
            c[('ref', mm.group(0))] += 1
    return c


def s5(letter_lines, prev_letter, main_lines, w180c):
    r22 = [ln for ln in letter_lines if ln.startswith('\\rresponse{Algorithm~1 and Section~2.1 now state the outer loop')]
    r22_line = _lines_with(letter_lines, '\\rresponse{Algorithm~1 and Section~2.1 now state the outer loop')
    text = r22[0] if len(r22) == 1 else ''
    changed = [i + 1 for i in range(max(len(letter_lines), len(prev_letter)))
               if (letter_lines[i] if i < len(letter_lines) else None) != (prev_letter[i] if i < len(prev_letter) else None)]
    p_now, p_prev = _pointers(letter_lines), _pointers(prev_letter)
    l125_now = _pointers([text]) if text else Counter()
    l125_prev = _pointers([prev_letter[r22_line[0] - 1]]) if r22_line else Counter()
    w180c_row = next((r for r in w180c['rows']['rows'] if r['row'] == 'R22-b'), {})
    e21 = _e21_rows(main_lines)
    checks = [
        _ck('R2.2 reply present exactly once (letter l. 125)', len(r22) == 1 and r22_line == [125], r22_line),
        _ck('R2.2: "the cycles of every single-scenario evaluation behind a reported comparison in Appendix~E."', S5_PHRASE in text),
        _ck('R2.2: v7 wording ("every evaluation behind") absent', S5_OLD not in text),
        _ck("R2.2's own two classes still stated (search: production exit; behind the comparisons: certification rule or cap)",
            S5_CLASSES in text),
        _ck('W180c R22-b residue was the 3 x 3 pair (multi-scenario); W180c verdict on record "still (narrowed)" / L',
            w180c_row.get('status') == 'still (narrowed)' and w180c_row.get('rating_now') == 'L', w180c_row.get('status')),
        _ck('the 3 x 3 pair is multi-scenario: Appendix E intro sends the multi-scenario evaluations to Table 13 (outside E.21)',
            bool(_lines_with(main_lines, APPE_INTRO_MS))),
        _ck('every E.21 row prints a k*/end cycle (61 rows, column 7 numeric)',
            len(e21) == 61 and all(re.fullmatch(r'\d+', r[6] or '') for r in e21.values()),
            sorted(no for no, r in e21.items() if not re.fullmatch(r'\d+', r[6] or ''))),
        _ck('the six search-only neighbours: stopped at the production exit (E.3) -- under R2.2\'s classes a search '
            'evaluation; their bands accepted as L by Addendum 80 ruling 4', bool(_lines_with(main_lines, E3_SEARCH_ONLY))),
        _ck('letter v7 -> v8: changed lines exactly {1, 125}', changed == [1, 125], changed),
        _ck('letter pointer multiset v8 = v7 (whole letter)', p_now == p_prev,
            {'n_v8': sum(p_now.values()), 'n_v7': sum(p_prev.values()),
             'only_v8': [list(k) for k in (p_now - p_prev)], 'only_v7': [list(k) for k in (p_prev - p_now)]}),
        _ck('R2.2 line pointer multiset v8 = v7', l125_now == l125_prev and sum(l125_now.values()) > 0,
            sorted(['|'.join(k) for k in l125_now.elements()])),
    ]
    rating = _rating(checks)
    return {'line': r22_line, 'changed_lines_v7_v8': changed, 'n_pointers': sum(p_now.values()),
            'r22_line_pointers': sorted(['|'.join(k) for k in l125_now.elements()]),
            'R22b': 'closed' if rating == 'MATCH' else 'still', 'checks': checks, 'rating': rating,
            'pointer_method': 'regex multiset (label words + number, \\ref/\\eqref/\\cite) over the whole letter; '
                              'not W180c\'s 165-pointer mapping, which is not re-run here'}


# ------------------------------------------------------------------------------------------------------- (d)
def bib_keys(bib_text):
    return sorted(set(re.findall(r'^\s*@\w+\s*\{\s*([^,\s]+)\s*,', bib_text, flags=re.M)))


def citations(main_lines, letter_lines, bib_text):
    i0 = next(i for i, ln in enumerate(main_lines) if ln.startswith('\\subsection') and 'Shared Energy Storage Parameters' in ln)
    i1 = next(i for i in range(i0 + 1, len(main_lines)) if re.match(r'\\(sub)?section\{', main_lines[i]))
    sec = main_lines[i0:i1]
    keys_bib = set(bib_keys(bib_text))
    cites = []
    for k, ln in enumerate(sec):
        if ln.lstrip().startswith('%'):
            continue
        for mm in re.finditer(r'\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])?\{([^}]*)\}', ln):
            for key in mm.group(1).split(','):
                cites.append({'key': key.strip(), 'line': i0 + k + 1, 'resolves_in_bib': key.strip() in keys_bib})
    typeset = '\n'.join(ln for ln in sec if not ln.lstrip().startswith('%'))
    sentences = re.split(r'(?<=[.;])\s+', re.sub(r'\s+', ' ', typeset))

    def stmt(needles):
        ss = [s for s in sentences if all(n in s for n in needles)]
        return {'sentences': len(ss), 'with_cite': sum(1 for s in ss if '\\cite' in s),
                'cite_keys': sorted({k.strip() for s in ss for mm in re.finditer(r'\\cite[a-zA-Z]*\{([^}]*)\}', s)
                                     for k in mm.group(1).split(',')})}
    statements = {
        '10,000 cycles at 0.80 DoD': stmt(['10{,}000 cycles at 80\\,\\% depth of discharge']),
        '8,000 full cycles (EVE MB31 datasheet)': stmt(['8{,}000 full cycles']),
        'calendar retention 0.985/yr': stmt(['0.985']),
    }
    auth = [(i + 1, ln) for i, ln in enumerate(main_lines) if '[AUTHOR' in ln]
    auth_block = []
    if auth:
        j = auth[0][0] - 1
        while j < len(main_lines) and main_lines[j].lstrip().startswith('%'):
            auth_block.append((j + 1, main_lines[j]))
            j += 1
    label_line = _lines_with(main_lines, '\\label{tab:ageing_calibrations}')
    counts = {f: {'[CONFIRM': sum(ln.count('[CONFIRM') for ln in L), '[AUTHOR': sum(ln.count('[AUTHOR') for ln in L)}
              for f, L in (('main.tex', main_lines), ('letter', letter_lines))}
    other_hits = {n: _lines_with(main_lines, n) for n in ('MB31', 'EVE ', 'datasheet')}
    bib_ageing = [k for k in keys_bib if re.search(r'eve|mb31|calendar|lfp|ageing|aging|degrad', k, flags=re.I)]
    present = {k: v['with_cite'] > 0 for k, v in statements.items()}
    return {'section_lines': [i0 + 1, i1], 'section_heading': main_lines[i0].strip(), 'cites': cites,
            'n_bib_entries': len(keys_bib), 'thebibliography_in_main': bool(_lines_with(main_lines, '\\begin{thebibliography}')),
            'bibliography_command': [ln.strip() for ln in main_lines if ln.startswith('\\bibliography')],
            'statements': statements, 'citation_present': present,
            'ageing_table_label_line': label_line, 'author_comment': auth_block, 'author_comment_lines_main': [a[0] for a in auth],
            'author_comment_directly_after_ageing_table': bool(auth and label_line and auth[0][0] - label_line[0] == 2
                                                               and main_lines[label_line[0]].strip() == '\\end{table}'),
            'counts': counts, 'main_tex_hits_elsewhere': other_hits, 'bib_keys_matching_ageing_terms': sorted(bib_ageing)}


# ------------------------------------------------------------------------------------------------------- controls
def negative_controls(main_lines, letter_lines, prev_letter, prev_main, srcs_head, cr, frozen, w169, w116_src, ub_src,
                      runs, w180b, n49, w101, w180c, bib_text):
    out = []

    def mutate(lines, old, new):
        return [ln.replace(old, new) for ln in lines]

    def rec(name, r, refused):
        out.append({'control': name, 'rating': r, 'refused': bool(refused)})

    r = s1(mutate(main_lines, 'its slacks are fixed at zero.', 'its slacks are fixed at one.'), srcs_head, {})
    rec('NC1 S1 sentence altered ("fixed at one")', r['rating'], r['rating'] != 'MATCH')
    code2 = dict(CODE)
    code2['fix_at_0'] = (CODE['fix_at_0'][0], 1144) + CODE['fix_at_0'][2:]
    r = s1(main_lines, srcs_head, {}, code=code2)
    rec('NC2 S1 code line number altered (1143 -> 1144)', r['rating'], r['rating'] != 'MATCH')
    r = s2(mutate(main_lines, '(cycle 72 for the plan without storage, 69 for the unit)',
                  '(cycle 69 for the plan without storage, 72 for the unit)'), cr, frozen, w169)
    rec('NC3 S2 Table 13 cycles swapped', r['rating'], r['rating'] != 'MATCH')
    cr2 = copy.deepcopy(cr)
    cr2['per_cell']['n7_4h_e1']['cycles_run'] = 72
    r = s2(main_lines, cr2, frozen, w169)
    rec('NC4 S2 campaign_results n7_4h_e1 cycles_run altered to 72', r['rating'], r['rating'] != 'MATCH')
    r = s3(mutate(main_lines, 'its voltage, thermal and no-reverse-flow rows switched off',
                  'its voltage and no-reverse-flow rows switched off'), w116_src, ub_src, runs)
    rec('NC5 S3 "thermal" dropped from the row families', r['rating'], r['rating'] != 'MATCH')
    w2 = dict(W116_CODE)
    w2['phase_B_call'] = (618, W116_CODE['phase_B_call'][1])
    r = s3(main_lines, w116_src, ub_src, runs, w116_code=w2)
    rec('NC6 S3 W116 phase-B line number altered (617 -> 618)', r['rating'], r['rating'] != 'MATCH')
    r = s4(mutate(main_lines, S4_PHRASE, 'Evaluations~1 and~40 continue earlier runs, and evaluations~2 and~41 are '
                  're-decided from the records of earlier runs; their first residual passes are not tabulated'),
           prev_main, w180b, n49, w101)
    rec('NC7 S4 note with evaluations 2 and 40 swapped', r['rating'], r['rating'] != 'MATCH')
    r = s4(mutate(main_lines, S4_PHRASE, 'Evaluations~1, 2, 40 and~41 continue earlier runs; their first residual passes are not tabulated'),
           prev_main, w180b, n49, w101)
    rec('NC8 S4 the old (round-6) note', r['rating'], r['rating'] != 'MATCH')
    r = s5(prev_letter, prev_letter, main_lines, w180c)
    rec('NC9 S5 the v7 letter (no "single-scenario")', r['rating'] + '/' + r['R22b'], r['R22b'] != 'closed')
    r = s5(mutate(letter_lines, 'Section~4.7, and the cycles', 'Section~4.8, and the cycles'), prev_letter, main_lines, w180c)
    rec('NC10 S5 a pointer changed in R2.2 (Section 4.7 -> 4.8)', r['rating'], r['rating'] != 'MATCH')
    c = citations(mutate(main_lines, '\\cite{nrel_ess_costs}', '\\cite{eve_mb31_datasheet}'), letter_lines, bib_text)
    rec('NC11 (d) a \\cite key not in bibliography.bib', str([x['resolves_in_bib'] for x in c['cites']]),
        any(not x['resolves_in_bib'] for x in c['cites']))
    c = citations(mutate(main_lines, 'as cycles to 80\\,\\% retention', 'as cycles to 80\\,\\% retention~\\cite{nrel_ess_costs}'),
                  letter_lines, bib_text)
    rec('NC12 (d) a \\cite planted in the 10,000-cycle sentence is detected', str(c['citation_present']),
        c['citation_present']['10,000 cycles at 0.80 DoD'] is True)
    return out


# ------------------------------------------------------------------------------------------------------- main
def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        # an interpreter process whose script argument is this file (shell wrappers that merely mention it are not)
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    out_dir = args.out_dir
    started = datetime.now(timezone.utc).isoformat()
    entries = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else None
    if entries is None or any(e != 'launch.log' for e in entries):
        raise SystemExit(f'refusing [fresh-dir check]: {out_dir} must exist and hold nothing but launch.log (found {entries})')
    others = _other_copy_running()
    if others:
        raise SystemExit(f'refusing [single-copy check]: another copy is running: {others}')
    repo_head = _git(['rev-parse', 'HEAD']).strip()

    # capture-path assertion: every source the order names must exist before anything is read
    required = [MAIN_TEX, LETTER, BIB, ROUND7_TEX, LETTER_V8, LETTER_V7, FROZEN, T11_RESULTS, W179C_JSON, W179C_MANIFEST,
                W180B_JSON, W180B_MANIFEST, W180C_JSON, W180C_MANIFEST, W163_JSON, W101_SUMMARY, W116_SCRIPT, UB_SRC,
                'network.py', 'model_construction_helpers.py', 'definitions.py', 'network_data.py',
                os.path.join(W169_CELLS, 't11_3x3_x0.json'), os.path.join(W169_CELLS, 't11_3x3_unit.json')] + \
               [os.path.join(W116_ROOT, r, f'{r}.json') for r in W116_RUNS]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'refusing [capture-path check]: missing sources {missing}')

    # prior evidence, hash-verified before use
    pv = {}
    w179c_man = _read_json(W179C_MANIFEST)
    pv['W179c'] = {'sha256': _reg(W179C_JSON)['sha256'], 'equals_report': INPUTS[W179C_JSON]['sha256'] == W179C_SHA,
                   'equals_manifest': w179c_man['outputs'].get(W179C_JSON) == INPUTS[W179C_JSON]['sha256']}
    w180b_man = _read_json(W180B_MANIFEST)
    pv['W180b'] = {'sha256': _reg(W180B_JSON)['sha256'], 'equals_report': INPUTS[W180B_JSON]['sha256'] == W180B_SHA,
                   'equals_manifest': w180b_man['outputs'].get(W180B_JSON) == INPUTS[W180B_JSON]['sha256']}
    w180c_man = _read_json(W180C_MANIFEST)
    pv['W180c'] = {'sha256': _reg(W180C_JSON)['sha256'],
                   'equals_report': INPUTS[W180C_JSON]['sha256'].startswith(W180C_SHA_PREFIX),
                   'equals_manifest': w180c_man['outputs'].get('w180c_text_letter.json') == INPUTS[W180C_JSON]['sha256']}
    pv['frozen'] = {'sha256': _reg(FROZEN)['sha256'], 'equals_declared': INPUTS[FROZEN]['sha256'] == FROZEN_SHA}
    for k in ('W179c', 'W180b', 'W180c', 'frozen'):
        pv[k]['status'] = INPUTS[{'W179c': W179C_JSON, 'W180b': W180B_JSON, 'W180c': W180C_JSON, 'frozen': FROZEN}[k]]['status']
    bad = [k for k, v in pv.items() if not all(x for x in v.values() if isinstance(x, bool)) or v['status'] != 'committed']
    if bad:
        raise SystemExit(f'refusing [prior-evidence hash check]: {bad}: {pv}')

    ident, main_lines, letter_lines, bib_text, prev = identity()
    if not ident['ok']:
        raise SystemExit(f'refusing [identity check]: {[c for c in ident["checks"] if not c["ok"]]}')

    w179c = _read_json(W179C_JSON)
    x0_heads = {c: v['git_head_at_run'] for c, v in w179c['D_x0'].items()}
    w180b = _read_json(W180B_JSON)
    w180c = _read_json(W180C_JSON)
    frozen = _read_json(FROZEN)
    cr = _read_json(T11_RESULTS)
    w169 = {c: _read_json(os.path.join(W169_CELLS, f'{c}.json'))['cell_meta'] for c in ('t11_3x3_x0', 't11_3x3_unit')}
    w163 = _read_json(W163_JSON)
    n49 = next(x for x in w163['checks_in_v4'] if x.get('id') == 'N49')['value']
    w101 = _read_json(W101_SUMMARY)
    runs = {r: _read_json(os.path.join(W116_ROOT, r, f'{r}.json')) for r in W116_RUNS}
    w116_src, ub_src = _read(W116_SCRIPT), _read(UB_SRC)
    srcs_head = _repo_srcs()

    sentences = {
        'S1_section_2_3_3_closure': s1(main_lines, srcs_head, x0_heads),
        'S2_table13_cycles': s2(main_lines, cr, frozen, w169),
        'S3_E4_phase_B': s3(main_lines, w116_src, ub_src, runs),
        'S4_E21_note': s4(main_lines, prev['main'], w180b, n49, w101),
        'S5_letter_R2_2': s5(letter_lines, prev['letter'], main_lines, w180c),
    }
    cit = citations(main_lines, letter_lines, bib_text)
    controls = negative_controls(main_lines, letter_lines, prev['letter'], prev['main'], srcs_head, cr, frozen, w169,
                                 w116_src, ub_src, runs, w180b, n49, w101, w180c, bib_text)
    controls_ok = all(c['refused'] for c in controls)

    pred_c = {'expected': 'all five MATCH and R22-b closes',
              'ratings': {k: v['rating'] for k, v in sentences.items()}, 'R22b': sentences['S5_letter_R2_2']['R22b']}
    pred_c['held'] = all(v == 'MATCH' for v in pred_c['ratings'].values()) and pred_c['R22b'] == 'closed'
    n_present = sum(1 for v in cit['citation_present'].values() if v)
    pred_d = {'expected': 'citations absent, or present with three keys', 'n_statements_with_cite': n_present,
              'cite_keys_section_3_5': [c['key'] for c in cit['cites']],
              'all_keys_resolve': all(c['resolves_in_bib'] for c in cit['cites'])}
    pred_d['outcome'] = 'absent' if n_present == 0 else ('present' if n_present == 3 else 'partial')
    pred_d['held'] = n_present == 0 or (n_present == 3 and len({k for v in cit['statements'].values()
                                                                   for k in v['cite_keys']}) >= 3)
    counts_expected = {'main.tex': {'[CONFIRM': 0}, 'letter': {'[CONFIRM': 0}}
    counts = {'counts': cit['counts'], 'CONFIRM_zero': all(cit['counts'][f]['[CONFIRM'] == 0 for f in counts_expected),
              'AUTHOR_total': sum(v['[AUTHOR'] for v in cit['counts'].values()),
              'AUTHOR_total_is_1': sum(v['[AUTHOR'] for v in cit['counts'].values()) == 1}

    bad_mods = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    frozen_unchanged = _sha(FROZEN) == FROZEN_SHA
    if bad_mods or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads'] or not controls_ok or not frozen_unchanged:
        raise SystemExit(f'harness fault: production modules {bad_mods}; guard {gfail}; pickle {PICKLE_COUNTS}; '
                         f'controls {[c for c in controls if not c["refused"]]}; frozen unchanged {frozen_unchanged}')
    result = {
        'schema': 'w181b_sentences_citations/1', 'stage': 'P5.15 W181b (Addendum 80, W181 items (c) and (d))',
        'started_utc': started, 'repo_head': repo_head, 'interpreter': sys.executable,
        'objective_convention': 'Q = gross_operational_cost (settlement excluded); value = Q(0) - Q(unit); no Q computed '
                                'here beyond the recorded difference of Table 13',
        'rating_scale': 'MATCH = every check of the sentence holds against its source; MISMATCH otherwise; L is not '
                        'assigned by the harness (wording residues are listed in the report)',
        'prior_evidence': pv, 'identity': ident, 'sentences': sentences, 'citations': cit, 'counts': counts,
        'negative_controls': controls, 'negative_controls_all_refused': controls_ok,
        'predictions': {'c': pred_c, 'd': pred_d},
        'guards': {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                   'pickle_counts': PICKLE_COUNTS, 'production_modules_imported': bad_mods,
                   'frozen_json_unchanged_at_end': frozen_unchanged},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    out_json = os.path.join(out_dir, 'w181b_sentences_citations.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(result, indent=1, sort_keys=False) + '\n')
    manifest = {'outputs': {out_json: _sha(out_json)}, 'script': {SCRIPT: _sha(SCRIPT)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')

    # ------------------------------------------------------------------------------------------------ console
    print(f'repo HEAD {repo_head}; clone HEAD {ident["head"]} (parent {ident["parent"][:7]}); identity ok {ident["ok"]}')
    print(f'main.tex sha256 {ident["main_sha256"]}; letter sha256 {ident["letter_sha256"]}; bib {ident["bib_sha256"][:12]}')
    print(f'prior evidence: {pv}')
    for k, v in sentences.items():
        print(f'{k}: {v["rating"]}  (line {v.get("line") or v.get("sentence_lines")})')
        for c in v['checks']:
            print(f'    [{"ok" if c["ok"] else "FAIL"}] {c["check"]}' + ('' if c['ok'] else f' -- {c["detail"]}'))
    s3r = sentences['S3_E4_phase_B']
    print(f'  S3 families sentence {s3r["families"]["sentence"]} code {s3r["families"]["code"]}')
    print(f'  S1 x0 run heads: {sentences["S1_section_2_3_3_closure"]["x0_run_heads"]}')
    print(f'  S5 R22-b: {sentences["S5_letter_R2_2"]["R22b"]}; letter changed lines {sentences["S5_letter_R2_2"]["changed_lines_v7_v8"]}; '
          f'pointers {sentences["S5_letter_R2_2"]["n_pointers"]}')
    print(f'(d) section 3.5 lines {cit["section_lines"]}: cites {cit["cites"]}; bib entries {cit["n_bib_entries"]}; '
          f'thebibliography {cit["thebibliography_in_main"]}; {cit["bibliography_command"]}')
    for k, v in cit['statements'].items():
        print(f'    {k}: {v}')
    print(f'    [AUTHOR] comment: {cit["author_comment"]} (directly after the ageing table: '
          f'{cit["author_comment_directly_after_ageing_table"]}; table label l. {cit["ageing_table_label_line"]})')
    print(f'    hits elsewhere in main.tex {cit["main_tex_hits_elsewhere"]}; bib keys matching ageing terms '
          f'{cit["bib_keys_matching_ageing_terms"]}')
    print(f'counts {counts}')
    for c in controls:
        print(f'  control {c["control"]}: rating {c["rating"]} refused {c["refused"]}')
    print(f'prediction (c): {pred_c}')
    print(f'prediction (d): {pred_d}')
    print(f'inputs {len(INPUTS)} ({manifest["n_uncommitted_inputs"]} uncommitted)')
    print(f'guards: {result["guards"]}')
    print(f'output {out_json} sha256 {_sha(out_json)}')


if __name__ == '__main__':
    main()
