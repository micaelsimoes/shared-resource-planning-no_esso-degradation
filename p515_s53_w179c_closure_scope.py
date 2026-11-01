"""
P5.15 Addendum 78 order, Planner task W179c -- items (d) and (e). ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION
IMPORT. Static reading of committed records, uncommitted capture files (hash-recorded), production source text and the
Overleaf clone at 48099ab (read only).

(d) Section 2.3.3: "Both daily closure slacks were zero, within solver tolerance, at every certified evaluation
    reported in this study". W169 scored 45 of its 51 certified points; this script completes the other six and
    re-reads all of them from the records:
    D1  Table E.21 (main.tex) parsed; rows mapped to cells by the generator's own order (manuscript_review/gen_appE.py,
        parsed with `ast`, not executed); the certified rows (+ the two T11 production-exit points W169 counted).
    D2  Closure level per block at each cell's point P (E.21 k*): component_levels_terminal.json
        blocks.*.unweighted.shared_ess_day_balance_slack / PENALTY_SHARED_ESS_BALANCE (= sum over the block's units
        of s+ + s-, MWh) when its cycles_run == P; otherwise the committed per-cycle record
        creep_diagnostic_per_cycle.jsonl q_decomposition.blocks[*].ess_complementarity_penalties at cycle P, divided by
        the block's admm_block_weight and PENALTY_SHARED_ESS_BALANCE. That per-cycle field is production's
        `total_ess_complementarity_penalties` = local-ESS + shared-ESS day-balance penalties
        (model_construction_helpers.ess_complementarity_penalties); the local term is identically 0 here (no
        `energy_storages` in any case file; checked) -- and the identity is checked bitwise at every cell's terminal
        cycle where both records exist (positive control).
    D3  Classification per block: no active unit -> must be exactly 0.0; >= 1 active unit -> signed reading "at the
        relaxed lower bound" iff floor <= S <= 0 with floor = -2 x 1e-8 p.u. x baseMVA per active unit (IPOPT's
        default bound_relax_factor 1e-8 on lb = 0; the per-solve 'Variable bound violation' in the logs is checked).
    D4  d_c52e1670 at k* = 150: read (D2, committed per-cycle record) AND derived as a cross-check: s+ - s- = sum_t
        (eta_ch P^ch - P^dch / eta_dch) dt from the UNCOMMITTED ess_schedule_per_cycle.jsonl (network copies, MW),
        with its precision bound 25 x CV x baseMVA from the IPOPT logs (final unscaled 'Constraint violation' of each
        block's last attempt at that round, located by network_ipopt_solve_records.jsonl log_bytes). Validated at the
        cell's terminal cycle 198 (against component_levels_terminal.json) and on a positive control, the superseded
        pb_y2025_n5 at k* = 167, whose closure sum is positive.
    D5  The five x = 0 cells: candidate and published E^Av from evaluation_record.json; every block exactly 0.0; the
        code that makes it so (configure_shared_ess_operational_state: slack Vars fixed at 0.0, bounds [0, 0], the
        closure row deactivated when the unit is inactive), read as text at HEAD and at each run's git head.
(e) E1  Section 4.8 Scope / E.3: the fourteen box neighbours of x = 0 (W173 committed rows): which are in Table E.22
        (certification rule) and which were evaluated by the search only, with the search's own F difference over its
        resolution; the search campaign's code head against HEAD.
    E2  Section 2.2.7 Determinacy: which evaluations behind reported comparisons are absent from Table E.21 (the 3 x 3
        pair, the static arrangements, the discount re-weighting, the search-only neighbours of x = 0 and of F2).

Guards: `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0 at the end;
`pickle.load` / `pickle.loads` blocked for the whole run and counted (0 expected). No production module is imported
(asserted at exit). Refuses to run unless the output directory exists and holds nothing but launch.log.

Output: data/SRP1/Results/P515S53/w179c_closure_scope/w179c_closure_scope.json and manifest_sha256.json (every input
with sha256 and git-tracked flag).

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w179c_closure_scope && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w179c_closure_scope.py \\
      > data/SRP1/Results/P515S53/w179c_closure_scope/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W179c closure scope (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W179c: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W179c: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import ast  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w179c_closure_scope')
OUT_JSON = os.path.join(OUT_DIR, 'w179c_closure_scope.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
DECLARED_CLONE_HEAD = '48099ab'
DECLARED_MAIN_SHA256 = '28008ab0d79b477c4e8d86a0c3a4d1acf9c95ed9c4921f4757f1600cda1c60af'

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
W169_DIR = os.path.join(P53, 'w169_slack_inventory')
W169_JSON = os.path.join(W169_DIR, 'w169_slack_inventory.json')
W173_JSON = os.path.join(P53, 'w173_search_counts', 'w173_search_counts.json')
W178A_JSON = os.path.join(P53, 'w178a_section2', 'w178a_checks.json')
FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
GEN_APPE = os.path.join('manuscript_review', 'gen_appE.py')
T2_CSV = os.path.join('manuscript_review', 'export_W160', 'T2.csv')
T6_CSV = os.path.join('manuscript_review', 'export_W160', 'T6.csv')
S47_RESULTS = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b', 'campaign_results.json')
V6_CERT_D = os.path.join(P53, 'w142_resettle_v6', 'v6_from_records', 'd_c52e1670_v6_from_records_certificate.json')

CELL_D = 'd_c52e1670'
POINT_D = 150
NODE_D = 7
CONTROL_CELL = 'pb_y2025_n5'          # superseded certificate (E.21 row 16): positive closure sum at k* = 167
CONTROL_NODE = 5
X0_CELLS = ('ref:7aa017f0', 'h_aa8a76d7', 'h_x0_m175', 'h_50dea31c', 't11_3x3_x0')
T11_POINTS = ('t11_3x3_x0', 't11_3x3_unit')   # production-exit points W169 counted among its 51
N_PERIODS = 24
REL_THRESHOLD = 1e-6                   # W169's "1e-6 relative" reading
BOUND_RELAX_FACTOR = 1e-8              # IPOPT default (not in the options list printed: not user-set)

PRODUCTION_FILES = ('admm_anderson_acceleration.py', 'admm_parameters.py', 'admm_persistent_workers.py', 'branch.py',
                    'definitions.py', 'energy_storage.py', 'generator.py', 'helper_functions.py', 'load.py',
                    'model_construction_helpers.py', 'network.py', 'network_data.py', 'network_parameters.py',
                    'node.py', 'planning_parameters.py', 'shared_energy_storage.py', 'shared_energy_storage_data.py',
                    'shared_energy_storage_parameters.py', 'shared_resources_planning.py', 'solver_parameters.py')
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

MS_ANCHORS = {
    'claim_233': 'Both daily closure slacks were zero, within solver tolerance, at every certified evaluation reported in this study.',
    'scope_48': 'with every evaluated unit neighbour of the plan without storage determinately worse',
    'det_227': 'the bands, gaps and slacks of every evaluation used in a reported comparison are tabulated in',
    'reeval_227': 'every cell it proposed was re-evaluated under this rule before being reported',
    'sec41_fourteen': 'All fourteen admissible unit neighbours of this plan were evaluated, and none reduced the total discounted cost beyond the search resolution.',
    'e3_text': 'The planning search evaluated all fourteen admissible unit neighbours of the plan without storage at its resolution',
    'e3_eight': 'gives the eight of them that were re-evaluated under the certification rule',
    'e3_confirm': '% [CONFIRM -- W179] that the other six box neighbours',
    'appE_intro_multiscenario': 'The multi-scenario evaluations, stopped by the production exit rather than certified, are reported in Table',
    'e21_caption_superseded': '$^{s}$: superseded by evaluation~15',
    'e21_note_row16': 'Evaluation~16 is the first certificate of the 2025 node-5 neighbour',
    'e22_note_row16': 'The superseded first certificate of the 2025 node-5 neighbour (evaluation~16) gave',
    'tab13_caption': 'The multi-scenario evaluations stopped by the production exit',
    'tab13_resolution': 'Resolution of the value',
    'discount_reweight': 'rates other than 2\\,\\% re-weight the 2\\,\\% evaluations',
    'f2_seventeen': 'only seventeen of its sixty-one admissible unit neighbours were evaluated',
    'benchmark_band': 'approximately 1{,}264 times the larger numerical band of the comparison',
    'search_41': 'made 41 new recourse evaluations in all',
}

CODE_ANCHORS = (
    ('slack_vars_declared_up', 'network.py', 'model.slack_shared_es_soc_final_up = pe.Var('),
    ('slack_vars_declared_down', 'network.py', 'model.slack_shared_es_soc_final_down = pe.Var('),
    ('closure_row', 'model_construction_helpers.py',
     'return m.shared_es_soc[e, s_m0, s_o0, final_p] == final_soc + m.slack_shared_es_soc_final_up[e, s_m0, s_o0] - m.slack_shared_es_soc_final_down[e, s_m0, s_o0]'),
    ('soc_prev_initial', 'model_construction_helpers.py',
     'soc_prev = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC'),
    ('final_soc', 'model_construction_helpers.py',
     'final_soc = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC'),
    ('soc_delta_ch', 'model_construction_helpers.py', 'delta = (eff_ch * m.shared_es_pch[e, s_m0, s_o0, p] * dt'),
    ('soc_delta_dch', 'model_construction_helpers.py', '- m.shared_es_pdch[e, s_m0, s_o0, p] * dt / eff_dch)'),
    ('soc_recursion_row', 'model_construction_helpers.py', 'return m.shared_es_soc[e, s_m0, s_o0, p] == soc_prev + delta'),
    ('dt', 'model_construction_helpers.py', 'return HOURS_PER_REPRESENTATIVE_DAY / n_periods'),
    ('operational_vars_list_slack_up', 'model_construction_helpers.py', "    'slack_shared_es_soc_final_up',"),
    ('operational_vars_list_slack_down', 'model_construction_helpers.py', "    'slack_shared_es_soc_final_down',"),
    ('operational_constraints_closure', 'model_construction_helpers.py', "    'sess_soc_final',"),
    ('inactive_test', 'model_construction_helpers.py',
     'return abs(s_capacity) <= tolerance or abs(e_capacity) <= tolerance'),
    ('inactive_fix', 'model_construction_helpers.py', '                entry.fix(0.0)'),
    ('inactive_deactivate', 'model_construction_helpers.py', '                entry.deactivate()'),
    ('slack_ub', 'model_construction_helpers.py',
     'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE'),
    ('closure_penalty', 'model_construction_helpers.py',
     'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m0, s_o0] + model.slack_shared_es_soc_final_down[e, s_m0, s_o0])'),
    ('ess_comp_is_local_plus_shared_1', 'model_construction_helpers.py',
     '        local_ess_day_balance_slack_penalty(model, network, s_m, s_o, params)'),
    ('ess_comp_is_local_plus_shared_2', 'model_construction_helpers.py',
     '        + shared_ess_day_balance_slack_penalty(model, network, s_m, s_o, params)'),
    ('local_penalty_loop', 'model_construction_helpers.py', '        for e in model.energy_storages:'),
    ('component_ess_comp', 'shared_resources_planning.py',
     "'ess_complementarity_penalties': float(pe.value(model.total_ess_complementarity_penalties))"),
    ('detector_shared', 'shared_resources_planning.py',
     "components['shared_ess_day_balance_slack'] += probability * pe.value(shared_ess_day_balance_slack_penalty(model, network, s_m, s_o, params))"),
    ('local_ess_from_case_key', 'network.py', "    if 'energy_storages' in network_data:"),
    ('shared_ess_eff_ch', 'shared_energy_storage.py', 'self.eff_ch = 0.97'),
    ('shared_ess_eff_dch', 'shared_energy_storage.py', 'self.eff_dch = 0.96'),
    ('drain_round_meaning', 'shared_resources_planning.py', '(0 = initialisation, k = cycle k)'),
    ('schedule_reader_prob_base', 'p515_s53_w105_settling_extension_hooks.py', 'c_sum += prob * vc * base'),
)
ANCHOR_FILES_AT_RUN_HEADS = ('model_construction_helpers.py', 'network.py')
RUN_HEAD_NEEDLES = ('slack_ub = 0.0 if inactive else', "    'slack_shared_es_soc_final_up',", "    'sess_soc_final',",
                    '                entry.fix(0.0)', '                entry.deactivate()')

CONSTANTS = (('definitions.py', 'PENALTY_SHARED_ESS_BALANCE'), ('definitions.py', 'ENERGY_STORAGE_RELATIVE_INIT_SOC'),
             ('definitions.py', 'EQUALITY_TOLERANCE'), ('definitions.py', 'SHARED_ESS_ZERO_CAPACITY_TOLERANCE'),
             ('definitions.py', 'HOURS_PER_REPRESENTATIVE_DAY'),
             ('model_construction_helpers.py', 'ESS_DAY_BALANCE_SLACK_FRACTION'))

# ------------------------------------------------------------------------------------------------------ input registry
INPUTS = {}
_TRACKED = None


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout


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


def _read(path):
    _reg(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _jsonl_find(path, pred):
    _reg(path)
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if pred(r):
                out.append(r)
    return out


def _read_bytes(path, a, b):
    _reg(path)
    with open(path, 'rb') as f:
        f.seek(a)
        return f.read(b - a).decode('utf-8', errors='replace')


# ------------------------------------------------------------------------------------------------------- A manuscript
def check_manuscript():
    head = _git(['rev-parse', '--short=7', 'HEAD'], cwd=CLONE).strip()
    dirty = _git(['status', '--porcelain', '--', 'main.tex'], cwd=CLONE).strip()
    tex = _read(MAIN_TEX)
    sha = INPUTS[os.path.relpath(os.path.abspath(MAIN_TEX), REPO)]['sha256']
    lines = tex.split('\n')
    loc = {}
    for k, needle in MS_ANCHORS.items():
        hits = [i + 1 for i, l in enumerate(lines) if needle in l]
        loc[k] = {'needle': needle, 'lines': hits}
    return tex, lines, {'clone': CLONE, 'clone_head': head, 'declared_head': DECLARED_CLONE_HEAD,
                        'head_ok': head == DECLARED_CLONE_HEAD, 'main_tex_sha256': sha,
                        'sha_ok': sha == DECLARED_MAIN_SHA256, 'main_tex_unmodified_in_clone': dirty == '',
                        'n_lines': len(lines), 'anchors': loc,
                        'all_anchors_found': all(v['lines'] for v in loc.values())}


def e21_cell_order():
    """Cells of Table E.21 in printed order, from gen_appE.py's G.append((title, [(cell, plan, year, cond), ...]))
    calls, parsed with ast (the generator is NOT executed)."""
    tree = ast.parse(_read(GEN_APPE))
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'append'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'G'):
            calls.append(node)
    calls.sort(key=lambda n: n.lineno)
    order = []
    for c in calls:
        tup = c.args[0]
        lst = tup.elts[1]
        for row in lst.elts:
            order.append(row.elts[0].value)
    return order


def parse_e21(lines):
    i0 = next(i for i, l in enumerate(lines) if r'\label{tab:appE_evaluations}' in l)
    i1 = next(i for i in range(i0, len(lines)) if r'\end{longtable}' in lines[i])
    rows = []
    for i in range(i0, i1):
        m = re.match(r'^(\d+) & ', lines[i])
        if not m:
            continue
        cols = [c.strip() for c in lines[i].rstrip().rstrip('\\').split(' & ')]
        rows.append({'no': int(cols[0]), 'line': i + 1, 'plan': cols[1], 'year': cols[2], 'condition': cols[3],
                     'status': cols[4], 'k0': cols[5], 'kstar_end': cols[6], 'range_tau': cols[7], 'band': cols[8],
                     'gap': cols[9], 'slack': cols[10], 'nc': cols[11], 'replay': cols[12], 'rule': cols[13]})
    return rows


def parse_e22(lines):
    i0 = next(i for i, l in enumerate(lines) if r'\label{tab:appE_neighbours}' in l)
    i1 = next(i for i in range(i0, len(lines)) if r'\bottomrule' in lines[i])
    rows = []
    for i in range(i0, i1):
        l = lines[i].strip()
        m = re.match(r'^([0-9][0-9 and]*) & (\d{4}) & (\d+) & ([\d.]+) & ([\d.]+) & ([\d.]+) \\\\$', l)
        if m:
            rows.append({'line': i + 1, 'nodes': m.group(1), 'year': m.group(2), 'eval': int(m.group(3)),
                         'delta_keur': float(m.group(4)), 'threshold_keur': float(m.group(5)),
                         'multiple': float(m.group(6))})
    return rows


# -------------------------------------------------------------------------------------------------- code and config
def check_code_anchors():
    out = {}
    for name, fn, needle in CODE_ANCHORS:
        lines = _read(fn).split('\n')
        out[name] = {'file': fn, 'needle': needle, 'lines': [i + 1 for i, l in enumerate(lines) if needle in l]}
    return out


def read_constants():
    out = {}
    for fn, name in CONSTANTS:
        txt = _read(fn)
        m = re.search(r'^' + re.escape(name) + r'\s*=\s*([^#\n]+)', txt, flags=re.M)
        out[name] = {'file': fn, 'text': m.group(1).strip() if m else None,
                     'value': float(m.group(1).strip()) if m else None}
    return out


def read_case_config():
    out = {'srp1_case': {}, 'base_mva': {}, 'energy_storages_key_present': {}}
    case = _read_json(os.path.join('data', 'SRP1', 'SRP1.json'))
    out['srp1_case'] = {'NumMarketScenarios': case['NumMarketScenarios'],
                        'num_operation_scenarios': {dn['name']: dn['num_operation_scenarios']
                                                    for dn in case['DistributionNetworks']},
                        'tn_num_operation_scenarios': case['TransmissionNetwork']['num_operation_scenarios']}
    for net in ('case9', 'case33_1', 'case33_2', 'case33_3'):
        d = os.path.join('data', 'SRP1', net)
        for fn in sorted(os.listdir(d)):
            if re.match(re.escape(net) + r'_\d{4}\.json$', fn):
                j = _read_json(os.path.join(d, fn))
                out['base_mva'][f'{net}/{fn}'] = j.get('baseMVA')
                out['energy_storages_key_present'][f'{net}/{fn}'] = 'energy_storages' in j
        p = _read_json(os.path.join(d, f'{net}_params.json'))
        out.setdefault('params', {})[net] = {'es_reg': p.get('es_reg'), 'slacks': p.get('slacks')}
    out['all_base_100'] = all(v == 100.0 for v in out['base_mva'].values())
    out['no_local_ess_in_any_case_file'] = not any(out['energy_storages_key_present'].values())
    out['single_scenario_srp1'] = (out['srp1_case']['NumMarketScenarios'] == 1
                                   and out['srp1_case']['tn_num_operation_scenarios'] == 1
                                   and all(v == 1 for v in out['srp1_case']['num_operation_scenarios'].values()))
    return out


def code_identity(heads):
    out = {}
    for label, head in heads.items():
        try:
            stat = _git(['diff', '--stat', head, 'HEAD', '--'] + list(PRODUCTION_FILES)).strip()
            needles = {}
            for fn in ANCHOR_FILES_AT_RUN_HEADS:
                txt = _git(['show', f'{head}:{fn}'])
                needles[fn] = {n: (n in txt) for n in RUN_HEAD_NEEDLES if fn == 'model_construction_helpers.py'}
            out[label] = {'head': head, 'production_diff_stat_vs_HEAD': stat or 'NO CHANGE',
                          'needles_at_head': needles['model_construction_helpers.py']}
        except subprocess.CalledProcessError as e:
            out[label] = {'head': head, 'error': str(e)}
    return out


# ----------------------------------------------------------------------------------------------- closure per block
def _w169_meta(cell):
    p = os.path.join(W169_DIR, 'cells', cell.replace(':', '_') + '.json')
    return _read_json(p)['cell_meta']


def _active_units(storage, year, tol_mwh):
    return [n for n, v in storage.items()
            if v.get('has_storage') and float(v['published_available_capacity_terminal'].get(str(year), {})
                                              .get('e_available', 0.0)) > tol_mwh]


def _block_key_from_q(b):
    return f"TSO|{b['year']}|{b['day']}" if b['agent'] == 'TSO' else f"DSO|{b['node_id']}|{b['year']}|{b['day']}"


def closure_at_point(cell, point, penalty, base, zero_tol_pu):
    meta = _w169_meta(cell)
    ed = meta['eval_dir']
    er = _read_json(os.path.join(ed, 'evaluation_record.json'))
    cl_path = os.path.join(ed, 'component_levels_terminal.json')
    cl = _read_json(cl_path)
    storage = er['storage_per_node']
    floor_pair_mwh = -2.0 * BOUND_RELAX_FACTOR * base
    tol_mwh = zero_tol_pu * base
    creep_path = os.path.join(ed, 'creep_diagnostic_per_cycle.jsonl')
    creep_exists = os.path.exists(creep_path)
    res = {'cell': cell, 'point': point, 'eval_dir': ed, 'eval_key': er.get('eval_key'),
           'candidate_key': er.get('candidate_key'), 'candidate_canonical': er.get('candidate_canonical'),
           'component_levels_cycles_run': cl['cycles_run'], 'creep_file_exists': creep_exists}
    # positive control of the per-cycle source: q_decomposition ess_complementarity_penalties at the terminal cycle
    # against component_levels_terminal weighted shared + local (bitwise)
    local_all_zero = all(b['unweighted']['local_ess_day_balance_slack'] == 0.0 for b in cl['blocks'].values())
    res['local_ess_day_balance_slack_all_exactly_zero_at_terminal'] = local_all_zero
    if creep_exists:
        recs = _jsonl_find(creep_path, lambda r: r.get('cycle') in (cl['cycles_run'], point))
        by_cycle = {r['cycle']: r for r in recs}
        rt = by_cycle.get(cl['cycles_run'])
        if rt is not None:
            diffs = []
            for b in rt['q_decomposition']['blocks']:
                k = _block_key_from_q(b)
                w = cl['blocks'][k]['weighted']
                diffs.append(abs(b['ess_complementarity_penalties']
                                 - (w['shared_ess_day_balance_slack'] + w['local_ess_day_balance_slack'])))
            res['per_cycle_source_control_at_terminal'] = {'cycle': cl['cycles_run'], 'n_blocks': len(diffs),
                                                           'max_abs_diff_eur': max(diffs), 'bitwise': max(diffs) == 0.0}
        else:
            res['per_cycle_source_control_at_terminal'] = None
    else:
        by_cycle = {}
    if cl['cycles_run'] == point:
        src = {'record': cl_path, 'status': _reg(cl_path)['status'],
               'field': 'blocks.<b>.unweighted.shared_ess_day_balance_slack / PENALTY_SHARED_ESS_BALANCE',
               'kind': 'read'}
        levels = {k: b['unweighted']['shared_ess_day_balance_slack'] / penalty for k, b in cl['blocks'].items()}
    else:
        rp = by_cycle.get(point)
        if rp is None:
            res['source'] = None
            res['not_carried'] = f'no component_levels at P (cycles_run {cl["cycles_run"]}) and no per-cycle record at P'
            return res
        if not (local_all_zero):
            raise RuntimeError(f'{cell}: local ESS closure not identically zero; per-cycle field not separable')
        src = {'record': creep_path, 'status': _reg(creep_path)['status'],
               'field': 'cycle P: q_decomposition.blocks[*].ess_complementarity_penalties / admm_block_weight / '
                        'PENALTY_SHARED_ESS_BALANCE (weighted local + shared day-balance penalties; local = 0)',
               'kind': 'read', 'gross_at_P': rp['gross'], 'local_solves_ok_at_P': rp.get('local_solves_ok')}
        levels = {}
        for b in rp['q_decomposition']['blocks']:
            k = _block_key_from_q(b)
            levels[k] = b['ess_complementarity_penalties'] / cl['blocks'][k]['admm_block_weight'] / penalty
    res['source'] = src
    blocks = []
    for k in sorted(levels):
        parts = k.split('|')
        year = parts[1] if parts[0] == 'TSO' else parts[2]
        units = sorted(storage) if parts[0] == 'TSO' else [parts[1]]
        act = [n for n in _active_units(storage, year, tol_mwh) if n in units]
        s = levels[k]
        eav = [float(storage[n]['published_available_capacity_terminal'][year]['e_available']) for n in act]
        if not act:
            cls = 'exact_zero' if s == 0.0 else 'NONZERO_WITHOUT_ACTIVE_UNIT'
        elif s > 0.0:
            cls = 'POSITIVE'
        elif s >= floor_pair_mwh * len(act) * (1.0 + 1e-9):
            cls = 'at_relaxed_lower_bound'
        else:
            cls = 'BELOW_FLOOR'
        # each of the 2 n slack Vars of the block is >= the relaxed floor fv = -1e-8 p.u. x baseMVA, and they sum to S,
        # so each lies in [fv, S - (2 n - 1) fv] -- read from S alone, no derivation
        fv = -BOUND_RELAX_FACTOR * base
        each_var = [fv, s - (2 * len(act) - 1) * fv] if act else [0.0, 0.0]
        blocks.append({'block': k, 'active_units': act, 'S_mwh': s, 'floor_mwh': floor_pair_mwh * len(act),
                       'each_var_interval_mwh': each_var,
                       'class': cls, 'e_av_terminal_min_mwh': min(eav) if eav else None,
                       'S_over_min_e_av': (s / min(eav)) if eav else None})
    res['blocks'] = blocks
    act_blocks = [b for b in blocks if b['active_units']]
    res['n_blocks'] = len(blocks)
    res['n_blocks_with_active_unit'] = len(act_blocks)
    res['classes'] = {c: sum(1 for b in blocks if b['class'] == c) for c in sorted({b['class'] for b in blocks})}
    res['max_S_active_mwh'] = max((b['S_mwh'] for b in act_blocks), default=None)
    res['min_S_active_mwh'] = min((b['S_mwh'] for b in act_blocks), default=None)
    res['max_S_over_e_av_signed'] = max((b['S_over_min_e_av'] for b in act_blocks), default=None)
    res['max_each_var_upper_mwh'] = max((b['each_var_interval_mwh'][1] for b in act_blocks), default=None)
    res['inactive_blocks_all_exactly_zero'] = all(b['S_mwh'] == 0.0 for b in blocks if not b['active_units'])
    res['storage_terminal'] = {n: {'has_storage': v['has_storage'],
                                   'e_available': {y: x['e_available'] for y, x in
                                                   v['published_available_capacity_terminal'].items()},
                                   's_available': {y: x['s_available'] for y, x in
                                                   v['published_available_capacity_terminal'].items()}}
                               for n, v in storage.items()}
    ok_classes = {'exact_zero', 'at_relaxed_lower_bound'}
    res['signed_reading_holds'] = all(b['class'] in ok_classes for b in blocks)
    return res


# ------------------------------------------------------------------------------- derived signed closure (cross-check)
def _ipopt_final(path, a, b):
    seg = _read_bytes(path, a, b)
    cv = re.findall(r'Constraint violation\.+:\s+(\S+)\s+(\S+)', seg)
    bv = re.findall(r'Variable bound violation:\s+(\S+)\s+(\S+)', seg)
    ex = re.findall(r'EXIT: (.+)', seg)
    return {'constraint_violation_unscaled': float(cv[-1][1]) if cv else None,
            'variable_bound_violation_unscaled': float(bv[-1][1]) if bv else None,
            'exit': ex[-1].strip() if ex else None}


def derived_signed(cell, cycles, node, eta_ch, eta_dch, base, dt):
    meta = _w169_meta(cell)
    ed = meta['eval_dir']
    sched_path = os.path.join(ed, 'ess_schedule_per_cycle.jsonl')
    recs_path = os.path.join(ed, 'network_ipopt_solve_records.jsonl')
    header = None
    lines_at = {}
    _reg(sched_path)
    with open(sched_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('header'):
                header = r
            elif r.get('cycle') in cycles:
                lines_at[r['cycle']] = r
    solve_recs = _jsonl_find(recs_path, lambda r: r.get('round') in cycles)
    dso_agent = f'DSO{node}'
    out = {'schedule_file': sched_path, 'schedule_status': _reg(sched_path)['status'],
           'schedule_header': {k: header[k] for k in ('schema', 'units', 'read_at', 'families')},
           'solve_records_file': recs_path, 'solve_records_status': _reg(recs_path)['status'],
           'method': 'D = sum_t (eta_ch * charge_t - discharge_t / eta_dch) * dt over the 24 periods of the block, '
                     'network copy (tso / dso), MW from the schedule (prob = 1, single scenario; x baseMVA already '
                     'applied by the writer). s+ - s- = D + baseMVA * (sum_t r_t - r_closure), |r| <= CV (final unscaled '
                     'IPOPT constraint violation of the block solve whose values were loaded) -> |error| <= 25 CV '
                     'baseMVA + floating-point bound', 'eta_ch': eta_ch, 'eta_dch': eta_dch, 'dt_h': dt,
           'per_cycle': {}}
    blocks = header['blocks']
    for c in cycles:
        r = lines_at.get(c)
        if r is None:
            out['per_cycle'][str(c)] = None
            continue
        rows = []
        for i, (n, y, d) in enumerate(blocks):
            if int(n) != node:
                continue
            for side, agent in (('tso', 'TSO'), ('dso', dso_agent)):
                ch = r['charge'][side][i]
                dch = r['discharge'][side][i]
                assert len(ch) == N_PERIODS and len(dch) == N_PERIODS
                terms = [eta_ch * a * dt - b * dt / eta_dch for a, b in zip(ch, dch)]
                D = 0.0
                for t in terms:
                    D += t
                fp = len(terms) * 2.0 ** -52 * sum(abs(t) for t in terms) * 2.0
                throughput = sum(ch) * dt + sum(dch) * dt
                atts = [s for s in solve_recs if s['round'] == c and s['agent'] == agent and int(s['year']) == int(y)
                        and s['day'] == d]
                finals = []
                for s in atts:
                    fin = _ipopt_final(s['log_path'], s['log_bytes'][0], s['log_bytes'][1])
                    fin.update({'attempt': s['attempt'], 'record_exit': s['exit'],
                                'log': os.path.relpath(s['log_path'], REPO), 'log_bytes': s['log_bytes']})
                    finals.append(fin)
                last = finals[-1] if finals else None
                cv = last['constraint_violation_unscaled'] if last else None
                bound = (25.0 * cv * base + fp) if cv is not None else None
                rows.append({'block': (f'TSO|{y}|{d}' if side == 'tso' else f'DSO|{n}|{y}|{d}'), 'side': side,
                             'D_mwh': D, 'precision_bound_mwh': bound, 'fp_bound_mwh': fp,
                             'throughput_mwh': throughput, 'attempts': finals,
                             'cv_used': cv})
        out['per_cycle'][str(c)] = rows
    return out


def combine(read_res, derived_rows, base):
    """s+ = (S + D)/2, s- = (S - D)/2 per block (S read, D derived); each against the per-variable floor."""
    per_var_floor = -BOUND_RELAX_FACTOR * base
    S = {b['block']: b['S_mwh'] for b in read_res['blocks']}
    out = []
    for r in derived_rows:
        s = S.get(r['block'])
        if s is None:
            continue
        sp, sm = (s + r['D_mwh']) / 2.0, (s - r['D_mwh']) / 2.0
        half = r['precision_bound_mwh'] / 2.0 if r['precision_bound_mwh'] is not None else None
        out.append({'block': r['block'], 'S_read_mwh': s, 'D_derived_mwh': r['D_mwh'],
                    'D_precision_mwh': r['precision_bound_mwh'], 's_plus_mwh': sp, 's_minus_mwh': sm,
                    'pm_mwh': half, 'per_var_floor_mwh': per_var_floor,
                    's_plus_at_floor_within_1pct': (half is not None
                                                    and abs(sp - per_var_floor) <= half + 0.01 * abs(per_var_floor)),
                    's_minus_at_floor_within_1pct': (half is not None
                                                     and abs(sm - per_var_floor) <= half + 0.01 * abs(per_var_floor)),
                    'min_var_at_or_above_floor': (half is not None and min(sp, sm) >= per_var_floor - half - 1e-15)})
    return out


# ----------------------------------------------------------------------------------------------------------- main
def main():
    global OUT_DIR, OUT_JSON, OUT_MANIFEST
    if '--out-dir' in sys.argv:                       # trial runs only; the directory must hold nothing but launch.log
        OUT_DIR = sys.argv[sys.argv.index('--out-dir') + 1]
        OUT_JSON = os.path.join(OUT_DIR, 'w179c_closure_scope.json')
        OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'REFUSED: {OUT_DIR} does not exist (create it with mkdir; it must hold only launch.log)')
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        raise SystemExit(f'REFUSED: {OUT_DIR} already holds {extra}')
    started = datetime.now(timezone.utc).isoformat()

    # A. manuscript, Table E.21 / E.22
    tex, lines, ms = check_manuscript()
    order = e21_cell_order()
    t2 = {r['cell']: r for r in csv.DictReader(open(T2_CSV, encoding='utf-8'))}
    _reg(T2_CSV)
    e21 = parse_e21(lines)
    assert [r['no'] for r in e21] == list(range(1, 62)), 'E.21 rows are not 1..61'
    assert len(order) == 61 and set(order) == set(t2), 'gen_appE order does not match T2.csv'
    for r in e21:
        r['cell'] = order[r['no'] - 1]
        t = t2[r['cell']]
        r['t2_status'] = t['status']
        r['t2_kstar_end'] = t['k* / end']
        r['consistent'] = ((r['status'].startswith('cert.')) == (t['status'] == 'certified')
                           and r['kstar_end'] == t['k* / end'])
    e21_consistent = all(r['consistent'] for r in e21)
    certified_rows = [r for r in e21 if r['status'].startswith('cert.')]
    e22 = parse_e22(lines)

    # constants, configuration, code
    const = read_constants()
    penalty = const['PENALTY_SHARED_ESS_BALANCE']['value']
    zero_tol = const['SHARED_ESS_ZERO_CAPACITY_TOLERANCE']['value']
    dt = const['HOURS_PER_REPRESENTATIVE_DAY']['value'] / N_PERIODS
    cfg = read_case_config()
    base = 100.0
    assert cfg['all_base_100'] and cfg['no_local_ess_in_any_case_file'] and cfg['single_scenario_srp1']
    anchors = check_code_anchors()
    eff = _read('shared_energy_storage.py')
    eta_ch = float(re.search(r'self\.eff_ch = ([\d.]+)', eff).group(1))
    eta_dch = float(re.search(r'self\.eff_dch = ([\d.]+)', eff).group(1))

    # W169 coverage, for the record
    w169 = _read_json(W169_JSON)
    w169_cells = {}
    for fn in sorted(os.listdir(os.path.join(W169_DIR, 'cells'))):
        m = _read_json(os.path.join(W169_DIR, 'cells', fn))['cell_meta']
        w169_cells[m['cell']] = m
    w169_certified = sorted(c for c, m in w169_cells.items() if m.get('certified_point'))
    w169_unscored = w169['prediction']['rows'][0]['unscored_cells']

    # D2/D3: closure at the point, every certified E.21 row + the two T11 points
    targets = [(r['cell'], int(r['kstar_end']), f"E.21 row {r['no']}") for r in certified_rows]
    for c in T11_POINTS:
        targets.append((c, int(w169_cells[c]['point']), 'T11 / Table 13 (production exit)'))
    per_cell = {}
    for cell, point, where in targets:
        res = closure_at_point(cell, point, penalty, base, zero_tol)
        res['where'] = where
        res['w169_scored'] = cell not in w169_unscored
        per_cell[cell] = res

    # D4: d_c52e1670 derived cross-check (150) + validation at 198 + positive control pb_y2025_n5 at 167
    d_term = per_cell[CELL_D]['component_levels_cycles_run']
    der_d = derived_signed(CELL_D, (POINT_D, d_term), NODE_D, eta_ch, eta_dch, base, dt)
    term_res = closure_at_point(CELL_D, d_term, penalty, base, zero_tol)   # read at 198 (component levels)
    der_c = derived_signed(CONTROL_CELL, (int(per_cell[CONTROL_CELL]['point']),), CONTROL_NODE, eta_ch, eta_dch,
                           base, dt)
    v6 = _read_json(V6_CERT_D)
    d_block = {
        'certificate': {'path': V6_CERT_D, 'k_star': v6['k_star'], 'Q_k_star': v6['Q_k_star'],
                        'source_run': v6.get('source_run')},
        'gross_at_P_equals_certificate_Q': per_cell[CELL_D]['source'].get('gross_at_P') == v6['Q_k_star'],
        'read_at_P': {k: per_cell[CELL_D][k] for k in ('source', 'classes', 'max_S_active_mwh', 'min_S_active_mwh',
                                                        'max_S_over_e_av_signed', 'signed_reading_holds',
                                                        'inactive_blocks_all_exactly_zero')},
        'derived': der_d,
        'combined_at_P': combine(per_cell[CELL_D], der_d['per_cycle'][str(POINT_D)], base),
        'validation_at_terminal': {'read': {k: term_res[k] for k in ('source', 'classes', 'max_S_active_mwh',
                                                                      'min_S_active_mwh')},
                                   'combined': combine(term_res, der_d['per_cycle'][str(d_term)], base)},
        'positive_control': {'cell': CONTROL_CELL, 'point': per_cell[CONTROL_CELL]['point'],
                             'read': {k: per_cell[CONTROL_CELL][k] for k in ('source', 'classes', 'max_S_active_mwh')},
                             'combined': combine(per_cell[CONTROL_CELL],
                                                 der_c['per_cycle'][str(per_cell[CONTROL_CELL]['point'])], base),
                             'derived_attempts': {r['block']: r['attempts'] for r in
                                                  der_c['per_cycle'][str(per_cell[CONTROL_CELL]['point'])]}},
    }

    # D5: x = 0 cells + code identity at run heads
    heads = {}
    for cell in X0_CELLS + (CELL_D, CONTROL_CELL):
        root = os.path.dirname(os.path.dirname(w169_cells[cell]['eval_dir']))
        cr = os.path.join(root, 'campaign_results.json')
        if os.path.exists(cr):
            heads[cell] = _read_json(cr).get('git_head_at_run')
    s47 = _read_json(S47_RESULTS)
    heads['S47 search (x = 0 neighbours)'] = s47['git_head_at_run']
    ident = code_identity({k: v for k, v in heads.items() if v})
    x0 = {}
    for cell in X0_CELLS:
        r = per_cell[cell]
        x0[cell] = {'point': r['point'], 'source': r['source'], 'candidate_key': r['candidate_key'],
                    'candidate_canonical': r['candidate_canonical'],
                    'every_node_without_storage': all(not v['has_storage'] and
                                                      all(e == 0.0 for e in v['e_available'].values()) and
                                                      all(s == 0.0 for s in v['s_available'].values())
                                                      for v in r['storage_terminal'].values()),
                    'n_blocks': r['n_blocks'], 'classes': r['classes'],
                    'every_block_exactly_zero': all(b['S_mwh'] == 0.0 for b in r['blocks']),
                    'per_cycle_source_control_at_terminal': r.get('per_cycle_source_control_at_terminal'),
                    'git_head_at_run': heads.get(cell)}

    # coverage and re-rating
    rows_cov = []
    for cell, point, where in targets:
        r = per_cell[cell]
        rows_cov.append({'cell': cell, 'where': where, 'point': point, 'w169_scored': r['w169_scored'],
                         'source_kind': (r['source'] or {}).get('field', '')[:40],
                         'n_active_blocks': r['n_blocks_with_active_unit'], 'classes': r['classes'],
                         'max_S_active_mwh': r['max_S_active_mwh'],
                         'max_S_over_e_av_signed': r['max_S_over_e_av_signed'],
                         'max_each_var_upper_mwh': r['max_each_var_upper_mwh'],
                         'signed_reading_holds': r['signed_reading_holds'],
                         'superseded': bool(w169_cells[cell].get('superseded'))})
    e21_cells = [r['cell'] for r in certified_rows]
    hold_e21 = [c for c in e21_cells if per_cell[c]['signed_reading_holds']]
    fail_e21 = [c for c in e21_cells if not per_cell[c]['signed_reading_holds']]
    rel_fail_e21 = [c for c in e21_cells if (per_cell[c]['max_S_over_e_av_signed'] or -1.0) > REL_THRESHOLD]
    coverage = {
        'e21_certified_rows': len(certified_rows), 'e21_certified_cells': e21_cells,
        'e21_rows_consistent_with_T2': e21_consistent,
        'w169_certified_points': len(w169_certified), 'w169_certified_cells': w169_certified,
        'w169_set_equals_e21_certified_plus_T11': sorted(set(e21_cells) | set(T11_POINTS)) == w169_certified,
        'scored_here': len(targets), 'scored_from_records': sum(1 for c, _, _ in targets if per_cell[c]['source']),
        'signed_reading_holds_e21': len(hold_e21), 'signed_reading_fails_e21': fail_e21,
        'relative_1e-6_signed_fails_e21': rel_fail_e21,
        'T11': {c: per_cell[c]['signed_reading_holds'] for c in T11_POINTS},
        'max_positive_S_mwh_e21': max((per_cell[c]['max_S_active_mwh'] for c in e21_cells
                                       if per_cell[c]['max_S_active_mwh'] is not None
                                       and per_cell[c]['max_S_active_mwh'] > 0), default=None),
        'per_cycle_source_controls': {c: per_cell[c].get('per_cycle_source_control_at_terminal') for c, _, _ in targets},
        'rows': rows_cov,
    }

    # (e) E1: x = 0 neighbours
    w173 = _read_json(W173_JSON)
    x0rows = w173['item2']['x0']['rows']
    e22_cells = {order[r['eval'] - 1] for r in e22}
    neigh = []
    for r in x0rows:
        cells = [c for _, c in r['frozen_table_rows']]
        in_e22 = bool(set(cells) & e22_cells)
        neigh.append({'label': r['label'], 'z': r['z'], 'search_F_minus_F_inc_eur': r['search_F_minus_F_inc_eur'],
                      'search_resolution_eur': r['search_resolution_eur'],
                      'ratio': r['search_F_minus_F_inc_eur'] / r['search_resolution_eur'],
                      'search_class': r['search_symmetric_class'], 'frozen_table_cells': cells,
                      'in_table_E22': in_e22})
    reeval = [n for n in neigh if n['in_table_E22']]
    search_only = [n for n in neigh if not n['in_table_E22']]
    f2 = w173['item2']['F2']
    f2_eval = [r for r in f2['rows'] if r.get('search_F_minus_F_inc_eur') is not None]
    f2_in_tables = [r for r in f2_eval if r['frozen_table_rows']]
    e_block = {
        'x0_box_neighbours': len(neigh), 'in_table_E22': len(reeval), 'search_only': len(search_only),
        'search_only_labels': [n['label'] for n in search_only],
        'search_only_ratio_range': [min(n['ratio'] for n in search_only), max(n['ratio'] for n in search_only)],
        'all14_ratio_range': [min(n['ratio'] for n in neigh), max(n['ratio'] for n in neigh)],
        'e22_rows': e22, 'e22_cells': sorted(e22_cells), 'neighbours': neigh,
        'search_code_head': s47['git_head_at_run'], 'sigma_Q': s47.get('sigma_Q'),
        'F2': {'n_box': f2['b_box']['n_feasible'], 'n_evaluated_in_search': len(f2_eval),
               'n_in_frozen_tables': len(f2_in_tables),
               'search_only': [r['label'] for r in f2_eval if not r['frozen_table_rows']]},
    }
    # E2: section 2.2.7 coverage
    frozen = _read_json(FROZEN)
    t2_keys = {t['eval key'] for t in t2.values()}
    three = {c: w169_cells[c]['eval_key'][:16] for c in ('t11_3x3_x0', 't11_3x3_unit', 't11_3x3_x0_continuation')}
    _reg(T6_CSV)
    e2 = {
        'three_by_three_eval_keys': three,
        'three_by_three_in_T2_E21': {c: k in t2_keys for c, k in three.items()},
        'three_by_three_frozen_note': frozen['tables']['three_by_three']['note'],
        'three_by_three_resolution_eur': next(r['value'] for r in frozen['tables']['three_by_three']['rows']
                                              if r['quantity'] == '3 x 3 resolution of the value'),
        'static_arrangements': {'in_appendix_E': 'Table E.23 (T6)', 'coordinated_reproducibility_band':
                                frozen['tables']['benchmark']['coordinated_reproducibility_band'],
                                'coordinated_source': frozen['tables']['benchmark']['coordinated_source']},
        'discount': [{'rate': r['rate'], 'threshold': r['threshold_a61_conservative']}
                     for r in frozen['tables']['discount']],
        'e21_bands_rows_1_2_keur': {r['no']: r['band'] for r in e21 if r['no'] in (1, 2)},
        'x0_search_only': len(search_only), 'F2_search_only': len(e_block['F2']['search_only']),
    }

    result = {
        'schema': 'p515_s53_w179c_closure_scope_v1',
        'stage': 'P5.15 Addendum 78 W179c -- items (d) closure slack at the six missing points and (e) two scope '
                 'statements, at Overleaf 48099ab',
        'started_utc': started, 'git_HEAD': _git(['rev-parse', 'HEAD']).strip(), 'interpreter': sys.executable,
        'objective_convention': 'closure level S = s+ + s- (MWh) summed over the units of a block, read from '
                                'production\'s penalty value / PENALTY_SHARED_ESS_BALANCE; gross_operational_cost '
                                'where a Q is quoted (settlement excluded)',
        'A_manuscript': ms, 'E21_rows': e21, 'E21_consistent_with_T2': e21_consistent,
        'constants': const, 'configuration': cfg, 'eta': {'ch': eta_ch, 'dch': eta_dch, 'dt_h': dt},
        'code_anchors': anchors, 'code_identity_at_run_heads': ident,
        'W169': {'path': W169_JSON, 'unscored_cells': w169_unscored, 'certified_points': len(w169_certified)},
        'D_coverage': coverage, 'D_d_c52e1670': d_block, 'D_x0': x0, 'D_per_cell': per_cell,
        'E1_x0_neighbours': e_block, 'E2_section227': e2,
    }
    bad = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    gfail = GUARD.verify(0)
    if bad or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        raise SystemExit(f'guard fault: production modules {bad}; guard {gfail}; pickle {PICKLE_COUNTS}')
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                        'pickle_counts': PICKLE_COUNTS, 'production_modules_imported': bad}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=1, sort_keys=False)
        f.write('\n')
    script = os.path.basename(__file__)
    manifest = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'script': {script: _sha(script)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)},
                'n_inputs': len(INPUTS), 'n_uncommitted_inputs': sum(1 for v in INPUTS.values()
                                                                     if v['status'] == 'UNCOMMITTED')}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write('\n')

    # ------------------------------------------------------------------------------------------------ console summary
    print(f"manuscript {ms['clone_head']} (declared {DECLARED_CLONE_HEAD}) sha ok {ms['sha_ok']} unmodified "
          f"{ms['main_tex_unmodified_in_clone']} anchors found {ms['all_anchors_found']}")
    for k, v in ms['anchors'].items():
        print(f"  {k}: lines {v['lines']}")
    print(f"E.21: 61 rows, consistent with T2 {e21_consistent}; certified rows {len(certified_rows)}")
    print(f"W169 certified points {len(w169_certified)}; equals E.21 certified + T11: "
          f"{coverage['w169_set_equals_e21_certified_plus_T11']}; unscored by W169 {w169_unscored}")
    print('per cell (point, source, n active blocks, classes, max S active MWh, max S/E^Av, signed holds):')
    for r in rows_cov:
        print(f"  {r['cell']:<16} {r['where']:<34} P={r['point']:<4} W169 {str(r['w169_scored']):<5} "
              f"src={r['source_kind'][:12]:<12} act={r['n_active_blocks']:<3} {r['classes']} "
              f"maxS={r['max_S_active_mwh']} rel={r['max_S_over_e_av_signed']} holds={r['signed_reading_holds']}"
              f"{' SUPERSEDED' if r['superseded'] else ''}")
    print(f"signed reading holds at {len(hold_e21)} / {len(certified_rows)} E.21 certified rows; fails {fail_e21}; "
          f"relative > 1e-6 signed: {rel_fail_e21}; T11 {coverage['T11']}")
    print('per-cycle source controls (q_decomposition ess_comp at terminal vs component levels):')
    for c, v in coverage['per_cycle_source_controls'].items():
        print(f"  {c}: {v}")
    print(f"d_c52e1670: certificate k* {v6['k_star']} Q {v6['Q_k_star']}; gross at P equals certificate "
          f"{d_block['gross_at_P_equals_certificate_Q']}")
    for b in per_cell[CELL_D]['blocks']:
        if b['active_units']:
            print(f"  read {b['block']}: S {b['S_mwh']:.6e} MWh floor {b['floor_mwh']:.1e} {b['class']} "
                  f"S/E^Av {b['S_over_min_e_av']:.3e}")
    for tag, rows in (('combined at 150', d_block['combined_at_P']),
                      (f'validation at {d_term}', d_block['validation_at_terminal']['combined']),
                      (f'positive control {CONTROL_CELL}', d_block['positive_control']['combined'])):
        print(f'  {tag}:')
        for r in rows:
            if r['S_read_mwh'] == 0.0 and abs(r['D_derived_mwh']) < 1e-12:
                continue
            print(f"    {r['block']}: S {r['S_read_mwh']:.6e} D {r['D_derived_mwh']:.3e} +- {r['D_precision_mwh']:.1e}"
                  f" s+ {r['s_plus_mwh']:.4e} s- {r['s_minus_mwh']:.4e} (floor {r['per_var_floor_mwh']:.0e}) "
                  f"min>=floor {r['min_var_at_or_above_floor']}")
    print('x = 0 cells:')
    for c, v in x0.items():
        print(f"  {c}: P {v['point']} no storage {v['every_node_without_storage']} every block exactly 0 "
              f"{v['every_block_exactly_zero']} ({v['n_blocks']} blocks) head {v['git_head_at_run']}")
    print('code identity at run heads:')
    for k, v in ident.items():
        print(f"  {k}: {v.get('head')} -> {v.get('production_diff_stat_vs_HEAD', v.get('error'))[:200]} "
              f"needles {v.get('needles_at_head')}")
    print(f"(e) x = 0 box neighbours {len(neigh)}: in E.22 {len(reeval)}, search only {len(search_only)} "
          f"{e_block['search_only_labels']}; ratio search-only {e_block['search_only_ratio_range']}, all 14 "
          f"{e_block['all14_ratio_range']}; search head {s47['git_head_at_run']}")
    print(f"F2: box {e_block['F2']['n_box']}, evaluated {e_block['F2']['n_evaluated_in_search']}, in tables "
          f"{e_block['F2']['n_in_frozen_tables']}, search only {e_block['F2']['search_only']}")
    print(f"3 x 3 in E.21: {e2['three_by_three_in_T2_E21']}; discount thresholds {e2['discount']}")
    print(f"inputs {len(INPUTS)} ({manifest['n_uncommitted_inputs']} uncommitted)")
    print(f"guards: {result['guards']}")
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')


if __name__ == '__main__':
    main()
