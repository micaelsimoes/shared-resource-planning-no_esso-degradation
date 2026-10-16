"""
P5.15 Addendum 79 order, Planner task W180d -- Section 2.3.3 closure sentence as reworded in round 6, re-rated at the
Overleaf HEAD e5322f8. ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT.

The sentence (main.tex at e5322f8; STEP6_ROUND6_CORRECTIONS.md item A.3):
  "At every certificate used in a reported comparison, and at both multi-scenario evaluations, both daily closure
   slacks sat at their lower bound to within the solver's bound relaxation ($10^{-6}$~MWh per slack); in the
   superseded evaluation~16 of Table~\\ref{tab:appE_evaluations} their sum reached $1.8 \\times 10^{-6}$~MWh,
   $5 \\times 10^{-6}$ of the available energy. Without storage the closure row is not built."

Clauses re-rated (MATCH / MISMATCH / L):
  (i)   the population "every certificate used in a reported comparison": enumerated from Appendix E as printed at
        e5322f8 (Table E.21 rows whose Status is a certificate, and which of them Tables E.20 / E.22 / E.24, the E.20
        note and the E.21 group headings use), reconciled with W179c's scored points; the x = 0 cells counted.
  (ii)  "to within the solver's bound relaxation (10^-6 MWh per slack)": IPOPT's bound_relax_factor (default, not
        user-set: checked in the case files and in the 'List of options' of every IPOPT log W179c hash-recorded),
        IPOPT 3.14.18's relax_bounds formula (quoted below, source hash-recorded), the p.u. energy base of the slack
        Vars in production source; the observed distances, from W179c's per-block closure sums.
  (iii) row 16's "1.8 x 10^-6 MWh, 5 x 10^-6 of the available energy" against W179c's per-block values, and that
        evaluation 16 of tab:appE_evaluations is that cell (generator order, parsed with ast, not executed).
  (iv)  "Without storage the closure row is not built" against production source at HEAD (where the row is
        constructed; what configure_shared_ess_operational_state does to it and to the slack Vars when the unit is
        inactive).
  (v)   "both multi-scenario evaluations": the 3 x 3 pair of Table res_multiscenario.

Sources: W179c's committed JSON (sha256 verified against its committed manifest BEFORE use; its script hash verified
too); the Overleaf clone at e5322f8 (read only; main.tex bytes verified against `git show e5322f8:main.tex` and
against manuscript_review/main_48099ab_round6.tex); main.tex at 48099ab via `git show` (row-by-row E.21 comparison);
manuscript_review/gen_appE.py via `git show HEAD:` (parsed with ast); production source text; case files; IPOPT logs
(read only, uncommitted, hash-recorded).

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked and counted (0 expected); no production module imported (asserted). Output written
only through gate_result_io (the shared writer that refuses string-typed flags). Refuses to run unless the output
directory exists and holds nothing but launch.log.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w180d_closure_sentence && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w180d_closure_sentence.py \\
      > data/SRP1/Results/P515S53/w180d_closure_sentence/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W180d closure sentence (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W180d: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W180d: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import ast  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w180d_closure_sentence')
OUT_JSON = os.path.join(OUT_DIR, 'w180d_closure_sentence.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
DECLARED_CLONE_HEAD = 'e5322f85dced8034312c9afb97c19001bce6bfa0'
DECLARED_MAIN_SHA256 = 'b3282295ece4cc3b0f9bc4ff5cd00ad2d86a7e60650221bfd2d0bd00fa61c492'
PREVIOUS_CLONE_HEAD = '48099ab'
ROUND6_TEX = os.path.join('manuscript_review', 'main_48099ab_round6.tex')
STEP6 = 'STEP6_ROUND6_CORRECTIONS.md'
GEN_APPE = 'manuscript_review/gen_appE.py'          # read via `git show HEAD:` (other Workers may run it)

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W179C_DIR = os.path.join(P53, 'w179c_closure_scope')
W179C_JSON = os.path.join(W179C_DIR, 'w179c_closure_scope.json')
W179C_MANIFEST = os.path.join(W179C_DIR, 'manifest_sha256.json')
W179C_SCRIPT = 'p515_s53_w179c_closure_scope.py'
W179C_DECLARED = {   # P5_15_W179C_CLOSURE_SCOPE.md / P5_15_W179_REPORT.md (commit 43c72e06)
    'json': '417c270f8fae66c884271aa403ea15e92931af531ff70397300ef653c3f4e19c',
    'manifest_prefix': 'd4197604',
    'script_prefix': '2b27da2d',
}
W169_CONT = os.path.join(P53, 'w169_slack_inventory', 'cells', 't11_3x3_x0_continuation.json')

T11_POINTS = ('t11_3x3_x0', 't11_3x3_unit')
BOUND_RELAX_FACTOR_DEFAULT = 1e-8      # IPOPT 3.14.18 registered default (quoted source below)
CONSTR_VIOL_TOL_DEFAULT = 1e-4         # IPOPT 3.14.18 registered default
IPOPT_BOUND_OPTIONS = ('bound_relax_factor', 'honor_original_bounds', 'constr_viol_tol')

# IPOPT 3.14.18 src/Algorithm/IpOrigIpoptNLP.cpp, fetched 2026-10-16 from
# https://raw.githubusercontent.com/coin-or/Ipopt/releases/3.14.18/src/Algorithm/IpOrigIpoptNLP.cpp
# (sha256 of the fetched file below). Not read at run time (no network); quoted verbatim for the record.
IPOPT_SOURCE = {
    'url': 'https://raw.githubusercontent.com/coin-or/Ipopt/releases/3.14.18/src/Algorithm/IpOrigIpoptNLP.cpp',
    'sha256_as_fetched': 'c640f9dfd65d6dc70d8c1cfae1fbe115fff2566acf2de2cfa1744ea037910a4a',
    'registration': ['roptions->AddLowerBoundedNumberOption(', '"bound_relax_factor",', '0., false,', '1e-8,',
                     '"Before start of the optimization, the bounds given by the user are relaxed. "',
                     '"Additional, the constraint violation tolerance constr_viol_tol is used to bound the relaxation '
                     'by an absolute value. "',
                     'roptions->AddBoolOption(', '"honor_original_bounds",', 'false,'],
    'relax_bounds': ['tmp->ElementWiseAbs();',
                     'tmp->Scal(std::abs(bound_relax_factor)); // |relaxfactor|*|bounds|',
                     'ones->Set(std::abs(bound_relax_factor));',
                     'tmp->ElementWiseMax(*ones);    // |relaxfactor|*max(|bounds|,1)',
                     'ones->Set(constr_viol_tol_);',
                     'tmp->ElementWiseMin(*ones);    // min(constr_viol_tol_,|relaxfactor|*max(|bounds|,1))',
                     'bounds.Axpy(bound_relax_factor < 0.0 ? -1.0 : 1.0, *tmp);'],
    'applied_to': 'original (unscaled) x_L, x_U, d_L, d_U, once before the optimisation (relax_bounds(-factor, *x_L))',
    'projection': 'only if honor_original_bounds (registered default false)',
}

CLAUSES = {
    'population': 'At every certificate used in a reported comparison',
    'multiscenario': 'and at both multi-scenario evaluations',
    'bound': "both daily closure slacks sat at their lower bound to within the solver's bound relaxation "
             "($10^{-6}$~MWh per slack)",
    'row16': 'in the superseded evaluation~16 of Table~\\ref{tab:appE_evaluations} their sum reached '
             '$1.8 \\times 10^{-6}$~MWh, $5 \\times 10^{-6}$ of the available energy.',
    'x0': 'Without storage the closure row is not built.',
}
OLD_SENTENCE_NEEDLE = 'within solver tolerance, at every certified evaluation reported in this study'

CODE_CHECKS = (   # (name, file, needle, expected line at HEAD or None, expected enclosing def or None)
    ('slack_up_var', 'network.py', 'model.slack_shared_es_soc_final_up = pe.Var(model.shared_energy_storages, '
     'model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)', 407, None),
    ('slack_down_var', 'network.py', 'model.slack_shared_es_soc_final_down = pe.Var(model.shared_energy_storages, '
     'model.scenarios_market, model.scenarios_operation, domain=pe.NonNegativeReals, initialize=0.0)', 408, None),
    ('closure_row_constructed', 'network.py', 'model.sess_soc_final = pe.Constraint(model.shared_energy_storages, '
     'model.scenarios_market, model.scenarios_operation, rule=partial(sess_soc_final_rule, network=network, '
     'params=params))', 511, None),
    ('closure_rule_def', 'model_construction_helpers.py', 'def sess_soc_final_rule(m, e, s_m, s_o, network, params):',
     None, None),
    ('closure_row_expression', 'model_construction_helpers.py',
     'return m.shared_es_soc[e, s_m0, s_o0, final_p] == final_soc + m.slack_shared_es_soc_final_up[e, s_m0, s_o0] '
     '- m.slack_shared_es_soc_final_down[e, s_m0, s_o0]', 990, 'sess_soc_final_rule'),
    ('ops_vars_slack_up', 'model_construction_helpers.py', "    'slack_shared_es_soc_final_up',", 1012, None),
    ('ops_vars_slack_down', 'model_construction_helpers.py', "    'slack_shared_es_soc_final_down',", 1013, None),
    ('ops_constraints_closure', 'model_construction_helpers.py', "    'sess_soc_final',", 1062, None),
    ('inactive_test', 'model_construction_helpers.py',
     'return abs(s_capacity) <= tolerance or abs(e_capacity) <= tolerance', 1069, 'shared_ess_capacity_is_inactive'),
    ('inactive_fix_slacks', 'model_construction_helpers.py', '                entry.fix(0.0)', 1143,
     'configure_shared_ess_operational_state'),
    ('inactive_slack_bounds_00', 'model_construction_helpers.py',
     'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE', 1166,
     'configure_shared_ess_operational_state'),
    ('slack_setlb_0', 'model_construction_helpers.py', '            entry.setlb(0.0)', None, None),
    ('inactive_deactivate_closure', 'model_construction_helpers.py', '                entry.deactivate()', 1176,
     'configure_shared_ess_operational_state'),
    ('penalty_base_is_baseMVA', 'model_construction_helpers.py', '    base = network.baseMVA', None, None),
    ('closure_penalty', 'model_construction_helpers.py',
     'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m0, s_o0] + '
     'model.slack_shared_es_soc_final_down[e, s_m0, s_o0])', 2341, 'shared_ess_day_balance_slack_penalty'),
    ('capacity_to_pu_e_tn', 'network_data.py',
     "e_capacity = abs(candidate_solution[node_id][year]['e']) / s_base", None, '_update_model_with_candidate_solution'),
    ('capacity_to_pu_e_dn', 'network_data.py',
     "e_capacity = abs(candidate_solution[tn_node_id][year]['e']) / s_base", None,
     '_update_model_with_candidate_solution'),
    ('slack_reported_times_base', 'network.py',
     'slack_soc_final = pe.value(model.slack_shared_es_soc_final_up[e, s_m0, s_o0] - '
     'model.slack_shared_es_soc_final_down[e, s_m0, s_o0]) * s_base', None, None),
    ('dt_formula', 'model_construction_helpers.py', 'return HOURS_PER_REPRESENTATIVE_DAY / n_periods', None,
     'period_duration_hours'),
)
PRODUCTION_PY = ('admm_anderson_acceleration.py', 'admm_parameters.py', 'admm_persistent_workers.py', 'branch.py',
                 'definitions.py', 'energy_storage.py', 'generator.py', 'helper_functions.py', 'load.py',
                 'model_construction_helpers.py', 'network.py', 'network_data.py', 'network_parameters.py',
                 'node.py', 'planning_parameters.py', 'shared_energy_storage.py', 'shared_energy_storage_data.py',
                 'shared_energy_storage_parameters.py', 'shared_resources_planning.py', 'solver_parameters.py')
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

# ------------------------------------------------------------------------------------------------------ input registry
INPUTS = {}
_TRACKED = None


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


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


def _reg_blob(label, data_bytes, status):
    INPUTS[label] = {'sha256': _sha_bytes(data_bytes), 'status': status}
    return INPUTS[label]


def _read(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _norm(s):
    return re.sub(r'\s+', ' ', s).strip()


# ------------------------------------------------------------------------------------------------------ W179c source
def verify_w179c():
    man = _read_json(W179C_MANIFEST)
    man_sha = INPUTS[W179C_MANIFEST]['sha256']
    json_sha = _reg(W179C_JSON)['sha256']
    script_sha = _reg(W179C_SCRIPT)['sha256']
    out = {
        'json_sha256': json_sha,
        'json_sha256_equals_manifest_entry': man['outputs'].get(W179C_JSON) == json_sha,
        'json_sha256_equals_report': json_sha == W179C_DECLARED['json'],
        'manifest_sha256': man_sha,
        'manifest_sha256_prefix_equals_report': man_sha.startswith(W179C_DECLARED['manifest_prefix']),
        'script_sha256': script_sha,
        'script_sha256_equals_manifest_entry': man['script'].get(W179C_SCRIPT) == script_sha,
        'script_sha256_prefix_equals_report': script_sha.startswith(W179C_DECLARED['script_prefix']),
        'json_status': INPUTS[W179C_JSON]['status'], 'manifest_status': INPUTS[W179C_MANIFEST]['status'],
        'script_status': INPUTS[W179C_SCRIPT]['status'],
    }
    out['ok'] = all(v for k, v in out.items() if isinstance(v, bool)) and all(
        out[k] == 'committed' for k in ('json_status', 'manifest_status', 'script_status'))
    return out, man


# ------------------------------------------------------------------------------------------------------- manuscript
def check_manuscript():
    head = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    dirty = _git(['status', '--porcelain', '--', 'main.tex'], cwd=CLONE).strip()
    tex = _read(MAIN_TEX)
    sha = INPUTS[os.path.relpath(os.path.abspath(MAIN_TEX), REPO)]['sha256']
    blob_head = _git(['show', f'{DECLARED_CLONE_HEAD}:main.tex'], cwd=CLONE, binary=True)
    with open(MAIN_TEX, 'rb') as f:
        main_bytes = f.read()
    round6_sha = _reg(ROUND6_TEX)['sha256']
    blob_prev = _git(['show', f'{PREVIOUS_CLONE_HEAD}:main.tex'], cwd=CLONE, binary=True)
    _reg_blob(f'{CLONE}@{PREVIOUS_CLONE_HEAD}:main.tex', blob_prev, 'overleaf clone git object (read only)')
    lines = tex.split('\n')
    # the sentence
    hits = [i + 1 for i, l in enumerate(lines) if CLAUSES['population'] in l]
    sentence_line = lines[hits[0] - 1].strip() if len(hits) == 1 else None
    clause_found = {k: (sentence_line is not None and v in sentence_line) for k, v in CLAUSES.items()}
    step6 = _read(STEP6)
    m = re.search(r'3\. \*\*§2\.3\.3 closure sentence.*?```latex\n(.*?)```', step6, flags=re.S)
    step6_block = _norm(m.group(1)) if m else None
    ms = {
        'clone': CLONE, 'clone_head': head, 'declared_head': DECLARED_CLONE_HEAD,
        'head_ok': head == DECLARED_CLONE_HEAD, 'main_tex_sha256': sha, 'sha_ok': sha == DECLARED_MAIN_SHA256,
        'main_tex_equals_git_object_at_head': main_bytes == blob_head,
        'main_tex_unmodified_in_clone': dirty == '',
        'main_tex_equals_round6_file': sha == round6_sha, 'round6_file_sha256': round6_sha,
        'n_lines': len(lines), 'sentence_hits': hits, 'sentence_line_no': hits[0] if len(hits) == 1 else None,
        'sentence': sentence_line, 'clauses_found': clause_found,
        'step6_A3_block_found': step6_block is not None,
        'sentence_equals_step6_A3_block': step6_block is not None and sentence_line is not None
                                          and _norm(sentence_line) == step6_block,
        'old_sentence_occurrences': sum(1 for l in lines if OLD_SENTENCE_NEEDLE in l),
        'previous_head_main_sha256': _sha_bytes(blob_prev),
    }
    ms['ok'] = (ms['head_ok'] and ms['sha_ok'] and ms['main_tex_equals_git_object_at_head']
                and ms['main_tex_unmodified_in_clone'] and ms['main_tex_equals_round6_file']
                and all(clause_found.values()) and ms['sentence_equals_step6_A3_block']
                and ms['old_sentence_occurrences'] == 0)
    return tex, lines, blob_prev.decode('utf-8').split('\n'), ms


def e21_cell_order():
    """Cells of Table E.21 in printed order, from the committed gen_appE.py's G.append((title, [(cell, ...), ...]))
    calls, parsed with ast (the generator is NOT executed). Read with `git show HEAD:`."""
    src = _git(['show', f'HEAD:{GEN_APPE}'])
    _reg_blob(f'HEAD:{GEN_APPE}', src.encode('utf-8'), 'committed (git object at repository HEAD)')
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'append'
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'G']
    calls.sort(key=lambda n: n.lineno)
    order = []
    for c in calls:
        for row in c.args[0].elts[1].elts:
            order.append(row.elts[0].value)
    return order


def _table_span(lines, label, end_token):
    i0 = next(i for i, l in enumerate(lines) if '\\label{' + label + '}' in l)
    i1 = next(i for i in range(i0, len(lines)) if end_token in lines[i])
    return i0, i1


def parse_e21(lines):
    i0, i1 = _table_span(lines, 'tab:appE_evaluations', '\\end{longtable}')
    rows, groups, note = [], [], None
    current_group = None
    for i in range(i0, i1):
        l = lines[i]
        g = re.match(r'^\\multicolumn\{14\}\{@\{\}l\}\{\\textit\{\((\w)\) (.*)\}\}\\\\\[1pt\]$', l.strip())
        if g:
            current_group = {'key': g.group(1), 'title': g.group(2), 'line': i + 1,
                             'refs': re.findall(r'\\ref\{([^}]+)\}', g.group(2))}
            groups.append(current_group)
            continue
        if 'textit{Note:}' in l:
            note = {'line': i + 1, 'text': l.strip()}
        m = re.match(r'^(\d+) & ', l)
        if not m:
            continue
        cols = [c.strip() for c in l.rstrip().rstrip('\\').split(' & ')]
        rows.append({'no': int(cols[0]), 'line': i + 1, 'plan': cols[1], 'year': cols[2], 'condition': cols[3],
                     'status': cols[4], 'k0': cols[5], 'kstar_end': cols[6], 'range_tau': cols[7], 'band': cols[8],
                     'gap': cols[9], 'slack': cols[10], 'nc': cols[11], 'replay': cols[12], 'rule': cols[13],
                     'group': current_group['key'] if current_group else None})
    return rows, groups, note


def _ints(s):
    return [int(x) for x in re.findall(r'\d+', re.sub(r'\$\^\{[a-z]\}\$', '', s))]


def parse_e20(lines):
    i0, i1 = _table_span(lines, 'tab:appE_claims', '\\end{longtable}')
    rows, note = [], None
    for i in range(i0, i1):
        l = lines[i]
        if 'textit{Note:}' in l:
            note = {'line': i + 1, 'text': l.strip()}
        m = re.match(r'^(\d+) & ', l)
        if not m:
            continue
        cols = [c.strip() for c in l.rstrip().rstrip('\\').split(' & ')]
        rows.append({'no': int(cols[0]), 'line': i + 1, 'comparison': cols[1], 'eval': cols[2],
                     'evals': _ints(cols[2]), 'verdict_gross': cols[6]})
    return rows, note


def parse_e22(lines):
    i0, i1 = _table_span(lines, 'tab:appE_neighbours', '\\end{table}')
    rows, note = [], None
    for i in range(i0, i1):
        l = lines[i].strip()
        if 'textit{Note:}' in l:
            note = {'line': i + 1, 'text': l}
        m = re.match(r'^([0-9][0-9 and]*) & (\d{4}) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\$', l)
        if m:
            rows.append({'line': i + 1, 'nodes': m.group(1), 'year': m.group(2), 'eval': int(m.group(3))})
    return rows, note


def parse_e24(lines):
    i0, i1 = _table_span(lines, 'tab:appE_gap', '\\end{table}')
    rows = []
    for i in range(i0, i1):
        l = lines[i].strip()
        m = re.match(r'^(\d+) & ', l)
        if m and '\\\\' in l:
            rows.append({'line': i + 1, 'eval': int(m.group(1))})
    return rows


def note_eval_refs(text):
    """Evaluation numbers named in a note: 'evaluation~N', 'evaluations~A, B and~C' (an 'and~N' counts only inside
    such a list, so 'rows~30 and~31' -- E.20 row numbers -- is not read as evaluations)."""
    refs = set()
    for m in re.finditer(r'[Ee]valuations?~(\d+(?:(?:,\s*|,?\s*and~)\d+)*)', text):
        refs |= set(int(x) for x in re.findall(r'\d+', m.group(1)))
    return sorted(refs)


# -------------------------------------------------------------------------------------------------- code and config
def _enclosing_def(lines, idx):
    for j in range(idx, -1, -1):
        m = re.match(r'^def (\w+)\(', lines[j])
        if m:
            return m.group(1)
    return None


def check_code():
    out = {}
    cache = {}
    for name, fn, needle, exp_line, exp_def in CODE_CHECKS:
        if fn not in cache:
            cache[fn] = _read(fn).split('\n')
        lines = cache[fn]
        hits = [i + 1 for i, l in enumerate(lines) if needle in l]
        rec = {'file': fn, 'needle': needle, 'lines': hits,
               'enclosing_defs': [_enclosing_def(lines, h - 1) for h in hits]}
        if exp_line is not None:
            rec['expected_line'] = exp_line
            rec['expected_line_holds'] = exp_line in hits
        if exp_def is not None:
            rec['expected_def'] = exp_def
            rec['expected_def_holds'] = exp_def in rec['enclosing_defs']
        out[name] = rec
    # the closure row: indentation level and enclosing def (unconditional inside its builder?)
    nl = cache['network.py']
    i = out['closure_row_constructed']['lines'][0] - 1
    indent = len(nl[i]) - len(nl[i].lstrip(' '))
    governing = None
    for j in range(i - 1, -1, -1):
        s = nl[j]
        if s.strip() and not s.lstrip().startswith('#'):
            ind = len(s) - len(s.lstrip(' '))
            if ind < indent:
                governing = {'line': j + 1, 'text': s.strip()[:120]}
                break
    out['closure_row_constructed']['indent'] = indent
    out['closure_row_constructed']['governing_statement'] = governing
    out['closure_row_constructed']['unconditional_in_builder'] = bool(governing and governing['text'].startswith('def '))
    # the slack Vars: governed by `if params.slacks.shared_ess.day_balance:`
    i = out['slack_up_var']['lines'][0] - 1
    out['slack_up_var']['preceding_line'] = nl[i - 1].strip()
    # sess_soc_final_rule body: the only Skip is the duplicate-scenario guard
    mh = cache['model_construction_helpers.py']
    d0 = out['closure_rule_def']['lines'][0] - 1
    d1 = next(j for j in range(d0 + 1, len(mh)) if mh[j].startswith('def '))
    body = mh[d0:d1]
    out['closure_rule_def']['body_lines'] = [d0 + 1, d1]
    out['closure_rule_def']['n_skip_returns'] = sum(1 for b in body if 'pe.Constraint.Skip' in b)
    out['closure_rule_def']['skip_guard'] = next((b.strip() for b in body if 'sess_row_is_duplicate' in b), None)
    # the deactivate loop at 1176 iterates _SHARED_ESS_OPERATIONAL_CONSTRAINTS
    j = 1176 - 1
    loop = next((mh[k].strip() for k in range(j, j - 8, -1) if 'for constraint_name in' in mh[k]), None)
    out['inactive_deactivate_closure']['loop_header'] = loop
    # the fix loop at 1143 iterates _SHARED_ESS_OPERATIONAL_VARIABLES
    j = 1143 - 1
    loop = next((mh[k].strip() for k in range(j, j - 8, -1) if 'for variable_name in' in mh[k]), None)
    out['inactive_fix_slacks']['loop_header'] = loop
    # bound options anywhere in production source
    out['production_sets_ipopt_bound_options'] = {
        fn: [o for o in IPOPT_BOUND_OPTIONS if o in _read(fn)] for fn in PRODUCTION_PY}
    out['no_production_file_sets_bound_options'] = not any(out['production_sets_ipopt_bound_options'].values())
    return out


def read_constants():
    txt = _read('definitions.py')
    out = {}
    for name in ('PENALTY_SHARED_ESS_BALANCE', 'EQUALITY_TOLERANCE', 'SHARED_ESS_ZERO_CAPACITY_TOLERANCE',
                 'HOURS_PER_REPRESENTATIVE_DAY'):
        m = re.search(r'^' + name + r'\s*=\s*([^#\n]+)', txt, flags=re.M)
        out[name] = float(m.group(1).strip()) if m else None
    return out


def read_case_config():
    base, solver_opts = {}, {}
    for net in ('case9', 'case33_1', 'case33_2', 'case33_3'):
        d = os.path.join('data', 'SRP1', net)
        for fn in sorted(os.listdir(d)):
            if re.match(re.escape(net) + r'_\d{4}\.json$', fn):
                base[f'{net}/{fn}'] = _read_json(os.path.join(d, fn)).get('baseMVA')
        p = _read_json(os.path.join(d, f'{net}_params.json'))
        s = p.get('solver', {})
        solver_opts[net] = {'options': s.get('options'), 'recovery_options': s.get('recovery_options'),
                            'shared_ess_day_balance_slack': p['slacks']['shared_ess']['day_balance'],
                            'bound_options_set': [o for o in IPOPT_BOUND_OPTIONS
                                                  if o in (s.get('options') or {}) or o in (s.get('recovery_options') or {})]}
    return {'base_mva': base, 'all_base_100': all(v == 100.0 for v in base.values()), 'n_case_files': len(base),
            'solver': solver_opts,
            'no_case_sets_bound_options': not any(v['bound_options_set'] for v in solver_opts.values()),
            'day_balance_slack_on_everywhere': all(v['shared_ess_day_balance_slack'] for v in solver_opts.values())}


def scan_ipopt_logs(w179c_manifest):
    """Every IPOPT log W179c hash-recorded: verify its sha256 against W179c's manifest, then collect the IPOPT version
    lines and every option name printed under 'List of options:'."""
    logs = sorted(p for p in w179c_manifest['inputs'] if p.endswith('.log') and '/logs/' in p)
    out = {'n_logs': len(logs), 'logs': {}}
    versions, options_seen = set(), set()
    opt_re = re.compile(r'^\s+(\S+) = (\S+)\s+\d+\s*$')
    for p in logs:
        rec = {'sha256_equals_w179c_manifest': None, 'n_option_lists': 0}
        if not os.path.exists(p):
            rec['missing'] = True
            out['logs'][p] = rec
            continue
        reg = _reg(p)
        rec['sha256_equals_w179c_manifest'] = reg['sha256'] == w179c_manifest['inputs'][p]['sha256']
        in_list = False
        with open(p, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('This is Ipopt version'):
                    versions.add(line.strip())
                if line.strip() == 'List of options:':
                    in_list = True
                    rec['n_option_lists'] += 1
                    continue
                if in_list:
                    m = opt_re.match(line)
                    if m:
                        options_seen.add(m.group(1))
                    elif line.startswith('*') or line.startswith('This is Ipopt'):
                        in_list = False
        out['logs'][p] = rec
    out['ipopt_versions'] = sorted(versions)
    out['options_printed_union'] = sorted(options_seen)
    out['bound_options_printed'] = [o for o in IPOPT_BOUND_OPTIONS if o in options_seen]
    out['all_logs_hash_verified'] = all(r.get('sha256_equals_w179c_manifest') for r in out['logs'].values())
    out['n_option_lists_total'] = sum(r['n_option_lists'] for r in out['logs'].values())
    return out


# ------------------------------------------------------------------------------------------------------------ main
def main():
    started = datetime.now(timezone.utc).isoformat()
    entries = sorted(os.listdir(OUT_DIR)) if os.path.isdir(OUT_DIR) else None
    if entries is None or any(e != 'launch.log' for e in entries):
        raise SystemExit(f'refusing: {OUT_DIR} must exist and hold nothing but launch.log (found {entries})')

    w179c_check, w179c_manifest = verify_w179c()
    if not w179c_check['ok']:
        raise SystemExit(f'refusing: W179c source does not verify: {w179c_check}')
    w = _read_json(W179C_JSON)   # already hash-verified (same bytes: _reg hashed the file on disk)
    per_cell = w['D_per_cell']

    tex, lines, prev_lines, ms = check_manuscript()
    if not ms['ok']:
        raise SystemExit(f'refusing: manuscript checks failed: {ms}')

    order = e21_cell_order()
    e21, groups, e21_note = parse_e21(lines)
    e21_prev, _, _ = parse_e21(prev_lines)
    for r in e21:
        r['cell'] = order[r['no'] - 1]
    # E.21 at e5322f8 against 48099ab, column by column
    cols = ('plan', 'year', 'condition', 'status', 'k0', 'kstar_end', 'range_tau', 'band', 'gap', 'slack', 'nc',
            'replay', 'rule')
    prev_by_no = {r['no']: r for r in e21_prev}
    e21_diffs = [{'no': r['no'], 'column': c, '48099ab': prev_by_no[r['no']][c], 'e5322f8': r[c]}
                 for r in e21 for c in cols if prev_by_no[r['no']][c] != r[c]]
    # W179c's own row -> cell mapping (from gen_appE at 67c081be) must agree with the current generator order
    w_rows = {r['no']: r for r in w['E21_rows']}
    mapping_agrees = all(w_rows[r['no']]['cell'] == r['cell'] for r in e21)

    e20, e20_note = parse_e20(lines)
    e22, e22_note = parse_e22(lines)
    e24 = parse_e24(lines)
    used_e20 = sorted({x for r in e20 for x in r['evals']})
    used_e22 = sorted({r['eval'] for r in e22})
    used_e24 = sorted({r['eval'] for r in e24})
    used_tables = sorted(set(used_e20) | set(used_e22) | set(used_e24))
    e20_note_refs = note_eval_refs(e20_note['text']) if e20_note else []
    e22_note_refs = note_eval_refs(e22_note['text']) if e22_note else []
    e21_note_refs = note_eval_refs(e21_note['text']) if e21_note else []

    cert_rows = [r for r in e21 if r['status'].startswith('cert.')]
    non_cert = [r for r in e21 if not r['status'].startswith('cert.')]
    superseded = [r for r in e21 if r['status'].startswith('superseded')]
    twins = [r['no'] for r in cert_rows if '$^{t}$' in r['status']]
    cert_nos = [r['no'] for r in cert_rows]
    cert_not_in_tables = [n for n in cert_nos if n not in used_tables]
    group_of = {r['no']: r['group'] for r in e21}
    group_refs = {g['key']: g['refs'] for g in groups}
    not_in_tables_where = {n: {'e20_note': n in e20_note_refs, 'e21_group': group_of[n],
                               'e21_group_tables': group_refs.get(group_of[n])} for n in cert_not_in_tables}
    # the year-ladder comparison in the main text (rows 24, 25 of E.21 serve Table res_year_ladder)
    yl_anchor = 'The gross difference between the two deferral years, 42.5~k\\euro{}, is determinate'
    yl_hits = [i + 1 for i, l in enumerate(lines) if yl_anchor in l]

    population = {
        'e21_rows': len(e21), 'e21_certified_rows': len(cert_rows), 'e21_certified_nos': cert_nos,
        'e21_non_certified_rows': [(r['no'], r['status']) for r in non_cert],
        'superseded_rows': [(r['no'], r['status'], r['rule'], r['cell']) for r in superseded],
        'twins_of_eval_2': twins,
        'used_by_E20': used_e20, 'used_by_E22': used_e22, 'used_by_E24': used_e24, 'used_by_tables': used_tables,
        'E20_note_refs': e20_note_refs, 'E21_note_refs': e21_note_refs, 'E22_note_refs': e22_note_refs,
        'certified_used_by_tables': [n for n in cert_nos if n in used_tables],
        'certified_not_used_by_E20_E22_E24': not_in_tables_where,
        'year_ladder_comparison_anchor_lines': yl_hits,
        'superseded_16_used_by_tables': 16 in used_tables, 'superseded_16_in_notes': {
            'E20': 16 in e20_note_refs, 'E21': 16 in e21_note_refs, 'E22': 16 in e22_note_refs},
        'E24_rows_all_uncertified': all(not next(r for r in e21 if r['no'] == n)['status'].startswith('cert.')
                                        for n in used_e24),
        'n_distinct_certificates_excluding_twins': len(cert_rows) - len(twins),
    }
    # the points of the sentence
    points = [(r['cell'], int(r['kstar_end']), f"E.21 row {r['no']}", r['no']) for r in cert_rows]
    points += [(c, int(per_cell[c]['point']), 'Table res_multiscenario (production exit)', None) for c in T11_POINTS]

    const = read_constants()
    cfg = read_case_config()
    base = 100.0 if cfg['all_base_100'] else None
    dt_h = w['eta']['dt_h']
    per_slack_floor_mwh = -min(CONSTR_VIOL_TOL_DEFAULT, BOUND_RELAX_FACTOR_DEFAULT * max(1.0, abs(0.0))) * base * dt_h

    rated = []
    for cell, point, where, no in points:
        r = per_cell[cell]
        blocks = r['blocks']
        active = [b for b in blocks if b['active_units']]
        inactive = [b for b in blocks if not b['active_units']]
        dist_relaxed = []      # per block: upper end of each Var minus the per-Var floor = S + 2n x 1e-6
        dist_nominal_max = []  # per block: largest |s| any Var can have = max(|floor|, |upper|)
        dist_nominal_min = []  # per block: smallest |s| any Var can have, when upper < 0
        bad = []
        for b in active:
            n = len(b['active_units'])
            floor = 2 * n * per_slack_floor_mwh
            s = b['S_mwh']
            upper = s - (2 * n - 1) * per_slack_floor_mwh
            ok = floor <= s <= 0.0
            if not ok:
                bad.append({'block': b['block'], 'S_mwh': s, 'n_active': n, 'S_over_min_e_av': b['S_over_min_e_av'],
                            'e_av_terminal_min_mwh': b['e_av_terminal_min_mwh'],
                            'each_var_interval_mwh': [per_slack_floor_mwh, upper]})
                continue
            if abs(upper - b['each_var_interval_mwh'][1]) > 1e-15:
                raise SystemExit(f'interval recomputation disagrees with W179c at {cell} {b["block"]}')
            dist_relaxed.append(upper - per_slack_floor_mwh)
            dist_nominal_max.append(max(abs(per_slack_floor_mwh), abs(upper)))
            dist_nominal_min.append(abs(upper) if upper < 0 else 0.0)
        inactive_exact = all(b['S_mwh'] == 0.0 for b in inactive)
        candidate = r['candidate_canonical']
        x0 = all(v == [0.0, 0.0] for v in candidate['nodes'].values())
        if x0:
            verdict = 'exact_zero_no_storage' if (not active and inactive_exact) else 'FAIL'
        else:
            verdict = 'at_relaxed_lower_bound' if (active and not bad and inactive_exact) else (
                'POSITIVE_BLOCK' if bad else 'FAIL')
        rated.append({'cell': cell, 'where': where, 'e21_no': no, 'point': point,
                      'w179c_point': r['point'], 'point_agrees_with_w179c': r['point'] == point,
                      'eval_key': r['eval_key'], 'candidate_key': r['candidate_key'], 'candidate_canonical': candidate,
                      'x_equals_0': x0, 'n_blocks': len(blocks), 'n_active_blocks': len(active),
                      'inactive_blocks_exactly_zero': inactive_exact,
                      'max_dist_each_var_to_relaxed_floor_mwh': max(dist_relaxed) if dist_relaxed else None,
                      'max_abs_each_var_mwh': max(dist_nominal_max) if dist_nominal_max else (0.0 if x0 else None),
                      'min_abs_each_var_mwh': min(dist_nominal_min) if dist_nominal_min else (0.0 if x0 else None),
                      'positive_blocks': bad, 'verdict': verdict, 'source': (r.get('source') or {}).get('field')})
    by_cell = {x['cell']: x for x in rated}

    # twins: bitwise equal per block to evaluation 2
    twin_check = {}
    c2 = next(r['cell'] for r in e21 if r['no'] == 2)
    for n in twins:
        ct = next(r['cell'] for r in e21 if r['no'] == n)
        a = {b['block']: b['S_mwh'] for b in per_cell[c2]['blocks']}
        t = {b['block']: b['S_mwh'] for b in per_cell[ct]['blocks']}
        twin_check[n] = {'cell': ct, 'blocks_bitwise_equal_eval_2': a == t}

    # row 16
    r16 = next(r for r in e21 if r['no'] == 16)
    pc16 = per_cell[r16['cell']]
    pos16 = [b for b in pc16['blocks'] if b['S_mwh'] > 0.0]
    act16 = [b for b in pc16['blocks'] if b['active_units']]
    row16 = {
        'e21_row': {k: r16[k] for k in ('no', 'line', 'plan', 'year', 'condition', 'status', 'kstar_end', 'rule')},
        'cell': r16['cell'], 'w179c_point': pc16['point'], 'kstar_agrees': pc16['point'] == int(r16['kstar_end']),
        'w179c_where': pc16.get('where'), 'eval_key': pc16['eval_key'], 'candidate_key': pc16['candidate_key'],
        'candidate_canonical': pc16['candidate_canonical'],
        'plan_agrees': r16['plan'] == '5: 0.25/0.5' and pc16['candidate_canonical']['nodes']['5'] == [0.25, 0.5]
                       and pc16['candidate_canonical']['investment_year'] == int(r16['year']),
        'n_positive_blocks': len(pos16), 'positive_blocks': pos16,
        'other_active_blocks_at_relaxed_floor': all(2 * len(b['active_units']) * per_slack_floor_mwh <= b['S_mwh'] <= 0.0
                                                    for b in act16 if b['S_mwh'] <= 0.0),
        'n_active_blocks': len(act16),
        'printed_sum_mwh': 1.8e-6, 'printed_fraction': 5e-6,
    }
    if len(pos16) == 1:
        s, frac = pos16[0]['S_mwh'], pos16[0]['S_over_min_e_av']
        row16.update({'S_mwh': s, 'S_over_e_av': frac, 'e_av_mwh': pos16[0]['e_av_terminal_min_mwh'],
                      'block': pos16[0]['block'],
                      'S_rounds_to_printed_2sf': float(f'{s:.1e}') == 1.8e-6,
                      'fraction_rounds_to_printed_1sf': float(f'{frac:.0e}') == 5e-6,
                      'fraction_recomputed': s / pos16[0]['e_av_terminal_min_mwh']})

    # (v) the multi-scenario pair and the x = 0 continuation
    cont_meta = _read_json(W169_CONT)['cell_meta']
    cont_cl = os.path.join(cont_meta['eval_dir'], 'component_levels_terminal.json')
    cont = {'cell': cont_meta['cell'], 'eval_key': cont_meta['eval_key'], 'candidate_key': cont_meta['candidate_key'],
            'candidate_canonical': cont_meta['candidate_canonical'], 'point': cont_meta['point'],
            'x_equals_0': all(v == [0.0, 0.0] for v in cont_meta['candidate_canonical']['nodes'].values())}
    if os.path.exists(cont_cl):
        cl = _read_json(cont_cl)
        vals = [b['unweighted']['shared_ess_day_balance_slack'] for b in cl['blocks'].values()]
        cont.update({'component_levels_status': INPUTS[cont_cl]['status'], 'cycles_run': cl['cycles_run'],
                     'n_blocks': len(vals), 'every_block_exactly_zero': all(v == 0.0 for v in vals)})
    else:
        cont['component_levels_terminal'] = 'not present'
    ms_tab = [i + 1 for i, l in enumerate(lines) if r'\label{tab:res_multiscenario}' in l]
    ms_cycle_label = [i + 1 for i, l in enumerate(lines) if 'at operational stopping (cycle 72)' in l]
    multiscenario = {
        'table_label_lines': ms_tab, 'pair': {c: by_cell[c] for c in T11_POINTS},
        'pair_holds': all(by_cell[c]['verdict'] in ('exact_zero_no_storage', 'at_relaxed_lower_bound')
                          for c in T11_POINTS),
        'x0_continuation': cont,
        'table_value_label_cycle_72_lines': ms_cycle_label,
        'stopping_cycles': {c: per_cell[c]['point'] for c in T11_POINTS},
    }

    # IPOPT options in force
    logs = scan_ipopt_logs(w179c_manifest)
    code = check_code()

    # ------------------------------------------------------------------------------------------------ ratings
    e21_points = [x for x in rated if x['e21_no'] is not None]
    held = [x for x in rated if x['verdict'] in ('exact_zero_no_storage', 'at_relaxed_lower_bound')]
    with_storage = [x for x in rated if not x['x_equals_0']]
    x0_points = [x for x in rated if x['x_equals_0']]
    max_rel = max(x['max_dist_each_var_to_relaxed_floor_mwh'] for x in with_storage)
    max_rel_cell = max(with_storage, key=lambda x: x['max_dist_each_var_to_relaxed_floor_mwh'])['cell']
    with_storage_e21 = [x for x in with_storage if x['e21_no'] is not None]
    max_rel_e21 = max(x['max_dist_each_var_to_relaxed_floor_mwh'] for x in with_storage_e21)
    max_rel_e21_cell = max(with_storage_e21, key=lambda x: x['max_dist_each_var_to_relaxed_floor_mwh'])['cell']
    max_abs = max(x['max_abs_each_var_mwh'] for x in with_storage)
    min_abs = min(x['min_abs_each_var_mwh'] for x in with_storage)

    conversion = {
        'bound_relax_factor': BOUND_RELAX_FACTOR_DEFAULT, 'constr_viol_tol': CONSTR_VIOL_TOL_DEFAULT,
        'nominal_lb_pu': 0.0, 'relaxation_pu': min(CONSTR_VIOL_TOL_DEFAULT, BOUND_RELAX_FACTOR_DEFAULT * 1.0),
        'baseMVA': base, 'dt_h': dt_h, 'relaxation_mwh_per_slack': -per_slack_floor_mwh,
        'printed_mwh_per_slack': 1e-6,
        'conversion_correct': abs(-per_slack_floor_mwh - 1e-6) <= 1e-18,
        'honor_original_bounds_default': 'no (registered default false; not set in any options list)',
        'options_in_force_evidence': {'case_files_set_none': cfg['no_case_sets_bound_options'],
                                      'production_source_sets_none': code['no_production_file_sets_bound_options'],
                                      'logs_print_none': logs['bound_options_printed'] == [],
                                      'n_logs': logs['n_logs'], 'n_option_lists': logs['n_option_lists_total'],
                                      'ipopt_versions': logs['ipopt_versions']},
        'observed': {'max_dist_each_var_to_relaxed_floor_mwh_all_with_storage': max_rel,
                     'at_cell': max_rel_cell,
                     'max_dist_each_var_to_relaxed_floor_mwh_e21_with_storage': max_rel_e21,
                     'at_cell_e21': max_rel_e21_cell,
                     'max_abs_each_var_mwh': max_abs, 'min_abs_each_var_mwh': min_abs,
                     'every_var_within_relaxation_of_nominal_lb': max_abs <= -per_slack_floor_mwh},
    }

    code_ok_x0 = (code['closure_row_constructed']['unconditional_in_builder']
                  and code['closure_rule_def']['n_skip_returns'] == 1
                  and all(code[k].get('expected_line_holds', True) and code[k].get('expected_def_holds', True)
                          for k in ('inactive_fix_slacks', 'inactive_slack_bounds_00', 'inactive_deactivate_closure',
                                    'ops_constraints_closure', 'ops_vars_slack_up', 'ops_vars_slack_down')))
    x0_all_exact = all(x['verdict'] == 'exact_zero_no_storage' for x in x0_points)

    expert_expected = 52
    ratings = {
        '(i) population': {
            'rating': 'MATCH' if (len(held) == len(rated) and 16 not in cert_nos) else 'MISMATCH',
            'n_e21_current_certificates': len(cert_rows), 'n_points_incl_multiscenario': len(rated),
            'n_points_held': len(held), 'expert_expected_points': expert_expected,
            'count_equals_expert_expectation': len(rated) == expert_expected,
            'n_with_storage': len(with_storage), 'n_x0': len(x0_points),
            'n_distinct_certificates_excluding_twins': population['n_distinct_certificates_excluding_twins'],
        },
        '(ii) bound relaxation 1e-6 MWh per slack': {
            'rating': 'MATCH' if (conversion['conversion_correct']
                                  and all(conversion['options_in_force_evidence'][k] for k in
                                          ('case_files_set_none', 'production_source_sets_none', 'logs_print_none'))
                                  and conversion['observed']['every_var_within_relaxation_of_nominal_lb'])
            else 'MISMATCH'},
        '(iii) row 16': {
            'rating': 'MATCH' if (row16.get('S_rounds_to_printed_2sf') and row16.get('fraction_rounds_to_printed_1sf')
                                  and row16['n_positive_blocks'] == 1 and row16['kstar_agrees']
                                  and row16['plan_agrees'] and r16['status'].startswith('superseded'))
            else 'MISMATCH'},
        '(iv) without storage the closure row is not built': {
            'rating': 'L' if (code_ok_x0 and x0_all_exact) else 'MISMATCH',
            'mechanism': 'row constructed for every shared-ESS unit (network.py sess_soc_final, unconditional in the '
                         'builder; the rule skips only duplicate-scenario rows), then deactivated with the slack Vars '
                         'fixed at 0.0 and bounded [0, 0] by configure_shared_ess_operational_state when the unit is '
                         'inactive; values exactly 0.0 at every x = 0 point'},
        '(v) both multi-scenario evaluations': {
            'rating': 'MATCH' if multiscenario['pair_holds'] else 'MISMATCH'},
    }
    prediction = {
        'expert': 'the sentence holds at 52 of 52 points',
        'outcome_points': f'{len(held)} of {len(rated)}',
        'holds_at_every_point': len(held) == len(rated),
        'count_52_reproduced': len(rated) == expert_expected,
    }

    result = {
        'schema': 'p515_s53_w180d_closure_sentence_v1',
        'stage': 'P5.15 Addendum 79 W180d -- Section 2.3.3 closure sentence (round 6) re-rated at Overleaf e5322f8',
        'started_utc': started, 'git_HEAD': _git(['rev-parse', 'HEAD']).strip(), 'interpreter': sys.executable,
        'objective_convention': "closure level S = s+ + s- (MWh) summed over a block's units, as read by W179c from "
                                "production's penalty value / PENALTY_SHARED_ESS_BALANCE; no Q is quoted",
        'W179c_source': w179c_check, 'A_manuscript': ms,
        'E21': {'rows': e21, 'groups': groups, 'note': e21_note, 'diffs_vs_48099ab': e21_diffs,
                'generator_order_agrees_with_W179c_mapping': mapping_agrees},
        'E20_rows': e20, 'E20_note': e20_note, 'E22_rows': e22, 'E22_note': e22_note, 'E24_rows': e24,
        'population': population, 'points': rated, 'twins': twin_check, 'row16': row16,
        'multiscenario': multiscenario, 'constants': const, 'configuration': cfg, 'ipopt_logs': logs,
        'ipopt_source': IPOPT_SOURCE, 'conversion': conversion, 'code': code,
        'ratings': ratings, 'prediction_d': prediction,
    }
    bad_mods = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    if bad_mods or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        raise SystemExit(f'guard fault: production modules {bad_mods}; guard {gfail}; pickle {PICKLE_COUNTS}')
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts,
                                                'verify_0_failures': gfail},
                        'pickle_counts': PICKLE_COUNTS, 'production_modules_imported': bad_mods}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(result, indent=1, sort_keys=False) + '\n')
    script = os.path.basename(__file__)
    manifest = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'script': {script: _sha(script)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')

    # ------------------------------------------------------------------------------------------------ console
    print(f"W179c source verified: {w179c_check['ok']} (json {w179c_check['json_sha256'][:12]}, manifest "
          f"{w179c_check['manifest_sha256'][:12]}, script {w179c_check['script_sha256'][:12]})")
    print(f"clone {ms['clone_head'][:7]} sha ok {ms['sha_ok']} = git object {ms['main_tex_equals_git_object_at_head']}"
          f" = round6 file {ms['main_tex_equals_round6_file']} unmodified {ms['main_tex_unmodified_in_clone']}")
    print(f"sentence at l. {ms['sentence_line_no']}; clauses {ms['clauses_found']}; = STEP6 A.3 "
          f"{ms['sentence_equals_step6_A3_block']}; old sentence occurrences {ms['old_sentence_occurrences']}")
    print(f"E.21: {len(e21)} rows; diffs vs 48099ab {e21_diffs}; generator order = W179c mapping {mapping_agrees}")
    print(f"certified rows {len(cert_rows)}: {cert_nos}")
    print(f"non-certified: {population['e21_non_certified_rows']}")
    print(f"used by E.20 {used_e20}\n        E.22 {used_e22}\n        E.24 {used_e24}")
    print(f"E.20 note refs {e20_note_refs}; E.21 note refs {e21_note_refs}; E.22 note refs {e22_note_refs}")
    print(f"certified not in E.20/E.22/E.24: {not_in_tables_where}; year-ladder anchor lines {yl_hits}")
    print(f"twins {twins}: {twin_check}; distinct certificates {population['n_distinct_certificates_excluding_twins']}")
    print('points:')
    for x in rated:
        print(f"  {x['where']:<42} {x['cell']:<16} P={x['point']:<4} x0={str(x['x_equals_0']):<5} "
              f"act={x['n_active_blocks']:<3}/{x['n_blocks']:<3} {x['verdict']:<24} "
              f"d_relax={x['max_dist_each_var_to_relaxed_floor_mwh']} |s|max={x['max_abs_each_var_mwh']}")
    print(f"row 16: {row16['cell']} P {row16['w179c_point']} {row16.get('block')} S {row16.get('S_mwh')} "
          f"S/E^Av {row16.get('S_over_e_av')} E^Av {row16.get('e_av_mwh')} rounds 1.8e-6 "
          f"{row16.get('S_rounds_to_printed_2sf')} 5e-6 {row16.get('fraction_rounds_to_printed_1sf')}")
    print(f"multi-scenario: {multiscenario['stopping_cycles']} holds {multiscenario['pair_holds']}; continuation "
          f"{cont}; 'cycle 72' label lines {ms_cycle_label}")
    print(f"IPOPT logs {logs['n_logs']} hash-verified {logs['all_logs_hash_verified']} option lists "
          f"{logs['n_option_lists_total']} versions {logs['ipopt_versions']} bound options printed "
          f"{logs['bound_options_printed']}; options union {logs['options_printed_union']}")
    print(f"conversion: {conversion}")
    print(f"code: row constructed {code['closure_row_constructed']['lines']} governing "
          f"{code['closure_row_constructed']['governing_statement']} unconditional "
          f"{code['closure_row_constructed']['unconditional_in_builder']}; rule skips "
          f"{code['closure_rule_def']['n_skip_returns']} ({code['closure_rule_def']['skip_guard']}); fix "
          f"{code['inactive_fix_slacks']['lines']} {code['inactive_fix_slacks']['loop_header']}; setub "
          f"{code['inactive_slack_bounds_00']['lines']}; deactivate {code['inactive_deactivate_closure']['lines']} "
          f"{code['inactive_deactivate_closure']['loop_header']}")
    for k, v in code.items():
        if isinstance(v, dict) and ('expected_line_holds' in v or 'expected_def_holds' in v):
            print(f"  {k}: lines {v['lines']} line ok {v.get('expected_line_holds')} def ok {v.get('expected_def_holds')}")
    print('ratings:')
    for k, v in ratings.items():
        print(f'  {k}: {v}')
    print(f'prediction (d): {prediction}')
    print(f"inputs {len(INPUTS)} ({manifest['n_uncommitted_inputs']} uncommitted)")
    print(f"guards: {result['guards']}")
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')


if __name__ == '__main__':
    main()
