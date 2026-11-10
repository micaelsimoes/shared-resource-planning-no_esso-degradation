"""P5.15 Addendum 89 order, Planner task W188 (a) -- SLACK INVENTORY (G10) OVER EVERY EVALUATION BEHIND THE MAIN-TEXT
RESULT TABLES. ZERO SOLVES, NO MODEL BUILD, NO PICKLE LOAD, NO PRODUCTION IMPORT. A NEW FILE; no production module,
record, spec, frozen JSON, manuscript file or `manuscript_review/` file is edited.

AUTHORITY. PLANNER_BRIEF_2026-09-13.md Addendum 89, Planner order W188 (a) and prediction P5 (expert): "every
penalised slack at its zero bound on every certified row (EUR content < 1 EUR); any non-zero content is listed with its
row". manuscript_review/REFOCUS_ANALYSIS_2026-11-10.md item G10.

MANUSCRIPT. main.tex at Overleaf HEAD a87e195, read with `git show a87e195:main.tex` in the clone
manuscript/6a67305f25e8348fb71380c3 (md5 declared 39f6fd86...). Tables by the expert's numbering (Addendum 89 /
refocus analysis): 7 = tab:res_value_ladder, 9 = tab:res_energy_ladder_m2, 10 = tab:res_ageing, 11 = tab:res_benchmark
(coordinated row only), 12 = tab:res_multiscenario, 13 = tab:res_year_ladder, 14 = tab:res_discount. The mapping is BY
CONTENT: every row is matched to its frozen-table entry and the printed numbers are asserted against the frozen JSON
(frozen_step6_tables_v1_590088fe.json) before any record is read.

WHICH PENALISED SLACKS ENTER Q (code read at HEAD; anchors located at run time, file:line in the output):
  Q = gross_operational_cost = sum over TSO + DSO blocks of w_b * objective_function_rule  (minus the contracted
  settlement and the voltage pin), shared_resources_planning._get_operational_recourse_components; the ESSO model is
  NOT summed into Q. objective_function_rule adds total_slack_penalties (voltage V^2 slacks, node-balance P/Q slacks,
  branch-flow slacks, flexibility P day-balance slacks) and total_ess_complementarity_penalties (local-ESS and shared-ESS
  daily-closure slacks s+/s-), each probability-weighted per scenario. The production detector
  `_get_local_detector_components` evaluates the SAME helper functions per block; component_levels_terminal.json
  records them per block ("unweighted" = probability-weighted block value, "weighted" = x w_b, the get_primal_value
  weight). The ESSO pair sigma+/sigma- = slack_es_pnet_up/down (shared_energy_storage_data.py) is penalised
  PENALTY_ESSO_SLACK * (sigma+ + sigma-) in the ESSO objective only: its EUR content in Q is 0 by construction; it is
  inventoried here from esso_feasibility_violation_D3 (= get_feasibility_violation, the sum of sigma+ + sigma- over all
  nodes and (y, d, t), MW) and element-wise from the esso_capture files where they exist.

QUANTITIES PER EVALUATION, AT ITS POINT P (certified: frozen k*; uncertified: frozen end; 3 x 3: production exit;
3 x 3 continuation: its terminal cycle):
  * signed EUR content in Q of each family F: C_F = sum_b w_b V_{F,b} (V = the recorded probability-weighted block
    penalty value). This is exactly what Q contains. It is NEGATIVE where IPOPT leaves a slack inside its relaxed bound.
  * positive-part bound (rigorous, given the floor premise): every slack element x_j >= -f, f = 1e-8 (IPOPT
    bound_relax_factor default on lb = 0, in the Var's own unit). With a_j = c_F * omega_s > 0 and sum_s omega_s = 1,
    V+ = sum_j a_j max(x_j, 0) <= V + f * c_F * N_F, N_F = number of non-fixed slack Vars of family F per scenario in
    the block (from the code and the case files; fixed Vars are exactly 0). Reported in EUR, weighted: B_F = sum_b w_b *
    max(0, V_{F,b} + f c_F N_{F,b}) (R1).
  * per-element upper bound (single-scenario cells, omega = 1): max_j x_j <= S_{F,b} + (N_{F,b} - 1) f, with
    S = V / c_F (the block's slack sum in the Var's unit). <= 0 means every element is in [-f, 0], i.e. at its zero
    bound within the relaxation, and the family's positive EUR content is exactly 0 ('proven'). A block sum S > 0 shows
    some element > 0 ('demonstrated_positive'). 3 x 3 cells: voltage element-wise from the run's results workbook,
    flexibility by a pair bound from the workbook nets and the committed block sum, closure as above.
  * P5 READING: a row holds when the refined positive-content bound (0 for proven families, B_F otherwise) is < 1 EUR.
  * implied count -S / f: must not exceed N (else the floor premise is contradicted -> reported).
  * block maximum of S (natural units) and of |w_b V| (EUR).
  * ESSO: D3 at P, N_esso = 2 x 3 nodes x |Y| x |D| x 24, positive-part bound D3 + f N_esso (MW), per-element bound
    D3 + (N_esso - 1) f, ESSO-objective value PENALTY_ESSO_SLACK x D3 (NOT in Q), element-wise maxima from esso_capture.
  * pickle-only: per-Var values (certified_models.pkl, esso_models_*.pkl) are LISTED with path and the sha256 from the
    eval dir's child_manifest_sha256.json; never opened.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0;
pickle.load / pickle.loads blocked for the whole run, counted (0 expected). No production module is imported
(asserted). All JSON is written through gate_result_io (the shared writer). The capture plan (every source file and
key, per cell) is asserted and written BEFORE any value is extracted (CLAUDE.md rule eleven). Refuses to run unless
the output directory exists and holds nothing but launch.log.

MODE (repo root, canonical interpreter; attached, both streams captured; mkdir refuses an existing directory):
    mkdir -p data/SRP1/Results/P515S55 && mkdir data/SRP1/Results/P515S55/w188a_slack_inventory && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w188a_slack_inventory.py \\
        > data/SRP1/Results/P515S55/w188a_slack_inventory/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W188a slack inventory (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W188a: pickle.load called -- no pickle loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W188a: pickle.loads called -- no pickle loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io  # noqa: E402  (the shared JSON writer; not a production module)

# ---------------------------------------------------------------------------------------------------------- constants
OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w188a_slack_inventory')
OUT_JSON_NAME = 'w188a_slack_inventory.json'
OUT_PLAN_NAME = 'capture_plan.json'
OUT_MANIFEST_NAME = 'manifest_sha256.json'

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
CLONE_COMMIT = 'a87e195'
DECLARED_MAIN_MD5_PREFIX = '39f6fd86'

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA256 = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
W169_PLAN = os.path.join(P53, 'w169_slack_inventory', 'capture_plan.json')
EXPORT_T1 = os.path.join('manuscript_review', 'export_W160', 'T1.csv')
RECON_CELLS = {   # D0 T0.1c reconciliation (records only): the W86 tight-tail pair T0.1c mapped
    'w86_x0_c132': os.path.join(P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals', '5cfe69a615ae3708_x0'),
    'w86_unit_c112': os.path.join(P53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert', 'evals',
                                  'ca8927e75d628bd1_n7_4h_e1'),
}
T01C_CORRECTION = {'w86_x0_c132': 13012.566792632126, 'w86_unit_c112': 13032.465347571979}   # P5_16_D0_T01C_REPORT.md

CASE_DIR = os.path.join('data', 'SRP1')
SRP1_JSON = os.path.join(CASE_DIR, 'SRP1.json')
ESS_PARAMS = os.path.join(CASE_DIR, 'SharedESS', 'SRP1_ESS_Params.json')
N_PERIODS = 24
FLOOR = 1e-8                       # IPOPT bound_relax_factor default; relaxed lb = -1e-8 * max(1, |0|) per Var
ONE_EUR = 1.0
INACTIVE_TOL = 1e-10               # shared_ess_capacity_is_inactive tolerance (model_construction_helpers.py)

FAMILIES = ('voltage_slack', 'node_balance_slack', 'branch_flow_slack', 'flexibility_p_day_balance_slack',
            'local_ess_day_balance_slack', 'shared_ess_day_balance_slack')
NETWORK_FAMILIES = ('voltage_slack', 'node_balance_slack', 'branch_flow_slack', 'flexibility_p_day_balance_slack')
CLOSURE_FAMILIES = ('local_ess_day_balance_slack', 'shared_ess_day_balance_slack')
ZERO_WEIGHT_TERMS = ('res_curtailment_penalty', 'ess_usage_cost', 'load_curtailment_cost')

FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions', 'node',
                             'branch', 'generator', 'load', 'energy_storage', 'planning_parameters',
                             'solver_parameters', 'uncoordinated_benchmark')

CODE_ANCHORS = (
    ('q_sum_tso', 'shared_resources_planning.py',
     "gross_operational_cost_including_settlement = planning_problem.transmission_network.get_primal_value(models['tso'])"),
    ('q_sum_dso', 'shared_resources_planning.py',
     "gross_operational_cost_including_settlement += distribution_network.get_primal_value(models['dso'][node_id])"),
    ('q_gross', 'shared_resources_planning.py', 'gross_operational_cost = (gross_operational_cost_including_settlement'),
    ('primal_is_objective_rule', 'network.py', 'primal_value = pe.value(objective_function_rule(model, params))'),
    ('obj_slack_penalties', 'model_construction_helpers.py', 'obj += model.total_slack_penalties'),
    ('obj_ess_closure_penalties', 'model_construction_helpers.py', 'obj += model.total_ess_complementarity_penalties'),
    ('obj_ess_usage', 'model_construction_helpers.py', 'obj += model.total_ess_utilization_cost_penalty'),
    ('pen_voltage', 'model_construction_helpers.py',
     'total += PENALTY_VOLTAGE_SQUARED * (model.slack_v_sqr_down[i, s_m, s_o, p] + model.slack_v_sqr_up[i, s_m, s_o, p])'),
    ('pen_node_balance_p', 'model_construction_helpers.py',
     'total += base * PENALTY_NODE_BALANCE * (model.slack_node_balance_p_up[i, s_m, s_o, p] + model.slack_node_balance_p_down[i, s_m, s_o, p])'),
    ('pen_branch_ij', 'model_construction_helpers.py',
     'total += base * PENALTY_CURRENT * (model.slack_flow_ij_sqr[b, s_m, s_o, p])'),
    ('pen_flex_p_day_balance', 'model_construction_helpers.py',
     'total += base * PENALTY_FLEXIBILITY * (model.slack_flex_p_balance_up[c, s_m, s_o] + model.slack_flex_p_balance_down[c, s_m, s_o])'),
    ('pen_local_ess_closure', 'model_construction_helpers.py',
     'total += base * PENALTY_ESS_BALANCE * (model.slack_es_soc_final_up[e, s_m, s_o] + model.slack_es_soc_final_down[e, s_m, s_o])'),
    ('pen_shared_ess_closure', 'model_construction_helpers.py',
     'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m0, s_o0] + model.slack_shared_es_soc_final_down[e, s_m0, s_o0])'),
    ('flex_skip_tso_interface', 'model_construction_helpers.py',
     'if load.fl_reg and not load_is_tso_adn_interface(network, load):'),
    ('voltage_slack_enabled_ref', 'model_construction_helpers.py', 'and node.type != BUS_REF'),
    ('voltage_slack_enabled_pv', 'model_construction_helpers.py', 'and not (node.type == BUS_PV and params.enforce_vg)'),
    ('voltage_slack_disabled_bounds', 'model_construction_helpers.py',
     'if not _voltage_magnitude_slack_enabled(node, params):'),
    ('tso_interface_voltage_slack_fixed', 'shared_resources_planning.py',
     'tso_model[year][day].slack_v_sqr_down[adn_node_idx, s_m, s_o, p].fix(0.00)'),
    ('dso_ref_voltage_slack_ub0', 'shared_resources_planning.py',
     'dso_model[year][day].slack_v_sqr_down[ref_node_idx, s_m, s_o, p].setub(0.00)'),
    ('tso_interface_flex_slack_fixed', 'network.py', 'model.slack_flex_p_balance_up[c, s_m, s_o].fix(0.0)'),
    ('flex_day_balance_pinned_true', 'network_parameters.py', 'self.day_balance = True'),
    ('local_ess_from_case_key', 'network.py', "if 'energy_storages' in network_data:"),
    ('detector_voltage', 'shared_resources_planning.py',
     "components['voltage_slack'] += probability * pe.value(voltage_slack_penalty(model, network, s_m, s_o, params))"),
    ('detector_shared_closure', 'shared_resources_planning.py',
     "components['shared_ess_day_balance_slack'] += probability * pe.value(shared_ess_day_balance_slack_penalty(model, network, s_m, s_o, params))"),
    ('detector_block_weight', 'shared_resources_planning.py',
     "blocks[('TSO', None, year, day)] = {name: weight * value for name, value in components.items()}"),
    ('esso_sigma_vars', 'shared_energy_storage_data.py',
     'model.slack_es_pnet_up = pe.Var(model.years, model.days, model.periods, domain=pe.NonNegativeReals, initialize=0.0)'),
    ('esso_sigma_row', 'shared_energy_storage_data.py',
     'model.energy_storage_operation_agg.add(model.es_pnet[y, d, p] == agg_pnet + model.slack_es_pnet_up[y, d, p] - model.slack_es_pnet_down[y, d, p])'),
    ('esso_sigma_penalty', 'shared_energy_storage_data.py',
     'slack_penalty += PENALTY_ESSO_SLACK * (model.slack_es_pnet_up[y_inv, d, p] + model.slack_es_pnet_down[y_inv, d, p])'),
    ('esso_feasibility_penalty_expr', 'shared_energy_storage_data.py',
     'model.feasibility_penalty = pe.Expression(expr=slack_penalty)'),
    ('esso_D3_sum', 'shared_energy_storage_data.py',
     'model.slack_es_pnet_up[y_inv, d, p] + model.slack_es_pnet_down[y_inv, d, p]'),
)
CONSTANT_NAMES = ('PENALTY_VOLTAGE_SQUARED', 'PENALTY_CURRENT', 'PENALTY_NODE_BALANCE', 'PENALTY_FLEXIBILITY',
                  'PENALTY_ESS_BALANCE', 'PENALTY_SHARED_ESS_BALANCE', 'PENALTY_ESSO_SLACK')

# ----------------------------------------------------------------------------------------- input registry and helpers
INPUTS = {}
_TRACKED = None
FAULTS = []


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
            if line.strip():
                r = json.loads(line)
                if pred(r):
                    out.append(r)
    return out


def _fault(msg):
    FAULTS.append(msg)
    print(f'FAULT: {msg}', flush=True)


def _k(x):
    """k EUR, 1 dp, sign dropped (the tables print the sign as $-$)."""
    return f'{abs(x) / 1e3:.1f}'


# -------------------------------------------------------------------------------------------------------- A manuscript
def read_manuscript():
    tex = _git(['show', f'{CLONE_COMMIT}:main.tex'], cwd=CLONE)
    blob = _git(['rev-parse', f'{CLONE_COMMIT}:main.tex'], cwd=CLONE).strip()
    full = _git(['rev-parse', CLONE_COMMIT], cwd=CLONE).strip()
    md5 = hashlib.md5(tex.encode('utf-8')).hexdigest()
    sha = hashlib.sha256(tex.encode('utf-8')).hexdigest()
    ok = md5.startswith(DECLARED_MAIN_MD5_PREFIX)
    if not ok:
        _fault(f'main.tex at {CLONE_COMMIT} md5 {md5} does not start with {DECLARED_MAIN_MD5_PREFIX}')
    return tex, {'clone': CLONE, 'commit': CLONE_COMMIT, 'commit_full': full, 'blob': blob, 'md5': md5,
                 'sha256': sha, 'md5_matches_declared_prefix': ok, 'read_via': f'git show {CLONE_COMMIT}:main.tex'}


def table_rows(tex, label):
    i = tex.index('\\label{' + label + '}')
    a = tex.index('\\midrule', i)
    b = tex.index('\\bottomrule', a)
    body = tex[a + len('\\midrule'):b]
    body = body.replace('\\midrule', '')
    rows = [r.strip() for r in body.split('\\\\')]
    rows = [re.sub(r'\s+', ' ', r) for r in rows if r.strip()]
    line = tex[:i].count('\n') + 1
    return rows, line


def _norm(s):
    return s.replace('\\,', '').replace('{,}', '').replace('$-$', '-').replace('~', ' ')


# ------------------------------------------------------------------------------------------- B table -> evaluation map
def build_table_map(tex, frozen):
    t = frozen['tables']
    claims = {c['claim_id']: c for c in t['claims']}
    a64 = {r['claim_id']: r for r in t['a64']['rows']}
    ageing = {r['cell']: r for r in t['ageing']['rows']}
    bench = t['benchmark']
    yl = t['year_ladder']['per_year']
    disc = t['discount']
    t33 = {r['quantity']: r for r in t['three_by_three']['rows']}

    def cl(cid):
        return claims[cid]

    def cells_of(c):
        return sorted(c['instance'].keys()) if 'instance' in c else sorted({c['ref_cell'], c['other_cell']})

    spec = []
    # Table 7: tab:res_value_ladder -- F(x) - F(0) and the threshold
    for cid in ('C:y2025__n5_p0.25_e0.5', 'B:n7_4h_e1', 'B:n7_2h_e1', 'B:n7_4h_e2',
                'C:y2025__n5_p0.25_e0.5__n9_p0.25_e0.5', 'C:y2030__n5_p0.25_e0.5__n7_p0.25_e0.5'):
        c = cl(cid)
        spec.append(('Table 7', 'tab:res_value_ladder', cid, [_k(c['d_gross']), _k(c['gross_threshold_or_bar'])],
                     cells_of(c)))
    # Table 9: tab:res_energy_ladder_m2 -- value - I and the increment, each with its threshold / bar
    t9 = (('H:m2:value_minus_I', None), ('I:m2:e2_value_minus_I', 'I:m2:second_MWh'),
          ('J:e3_value_minus_I', 'J:e2_to_e3'), ('J:e4_value_minus_I', 'J:e3_to_e4'),
          ('J:e5_value_minus_I', 'J:e4_to_e5'))
    for vid, iid in t9:
        c = cl(vid)
        bar = c['manuscript_figure_A64']['bar'] if 'manuscript_figure_A64' in c else c['gross_threshold_or_bar']
        exp = [_k(c['d_gross']), _k(bar)]
        cells = set(cells_of(c))
        if iid:
            ci = cl(iid)
            exp += [_k(ci['d_gross']), _k(ci['gross_threshold_or_bar'])]
            cells |= set(cells_of(ci))
        spec.append(('Table 9', 'tab:res_energy_ladder_m2', vid + (' + ' + iid if iid else ''), exp, sorted(cells)))
    # Table 10: tab:res_ageing -- value - I, threshold, AE, EFC/day
    for cell, cid in (('e_c2_calfade', 'E:n7_4h_e1_C2_calfade:value_minus_I'), ('e_c2', 'E:n7_4h_e1_C2:value_minus_I'),
                      ('e_c3_unit', 'E:C3_unit_value_minus_I'),
                      ('e_c3_midblock', 'E:n7_4h_e1_C3_midblock:value_minus_I'),
                      ('e_c4', 'E:n7_4h_e1_C4:value_minus_I'), ('e_no_ageing', 'E:n7_4h_e1_no_ageing:value_minus_I')):
        c, r = cl(cid), ageing[cell]
        spec.append(('Table 10', 'tab:res_ageing', cid,
                     [_k(r['value_minus_I']), _k(c['gross_threshold_or_bar']), f"{r['AE']:.3f}", f"{r['EFC']:.2f}"],
                     cells_of(c)))
    c = a64['E:soh050:value_minus_I']
    spec.append(('Table 10', 'tab:res_ageing', 'E:soh050:value_minus_I',
                 [_k(c['d_gross']), _k(c['gross_threshold_or_bar'])], ['e_soh050', 'ref:7aa017f0']))
    # Table 11: tab:res_benchmark -- coordinated row only
    spec.append(('Table 11', 'tab:res_benchmark', 'benchmark.coordinated_Q',
                 [f"{bench['coordinated_Q'] / 1e3:.1f}", _k(bench['coordinated_reproducibility_band'])],
                 ['ref:7aa017f0']))
    # Table 12: tab:res_multiscenario -- the 3 x 3 pair, the w98 continuation behind D, the single-scenario value
    q = {r['quantity']: r['value'] for r in t['three_by_three']['rows']}
    keys = list(q)
    v33 = q[[k for k in keys if k.startswith('3 x 3 storage value V')][0]]
    i33 = q[[k for k in keys if k.startswith('3 x 3 investment I')][0]]
    m33 = q[[k for k in keys if k.startswith('3 x 3 value - I')][0]]
    r33 = q[[k for k in keys if k.startswith('3 x 3 resolution of the value')][0]]
    vss = q[[k for k in keys if k.startswith('V_SRP1 settled')][0]]
    rlo, rhi = t['three_by_three']['R_range_rounded']
    for name, exp, cells in (('3x3 value V at stopping', [_k(v33)], ['t11_3x3_x0', 't11_3x3_unit']),
                             ('3x3 investment I', [_k(i33)], ['t11_3x3_unit']),
                             ('3x3 value - I', [_k(m33)], ['t11_3x3_x0', 't11_3x3_unit']),
                             ('3x3 resolution', [_k(r33)], ['t11_3x3_x0', 't11_3x3_unit']),
                             ('single-scenario value', [_k(vss)], ['ref:7aa017f0', 'ref:bd504ecf']),
                             ('R at stopping', [f'{rhi:.3f}'], ['t11_3x3_x0', 't11_3x3_unit', 'ref:7aa017f0',
                                                                 'ref:bd504ecf']),
                             ('R after continuation adjustment', [f'{rlo:.3f}'],
                              ['t11_3x3_x0', 't11_3x3_unit', 't11_3x3_x0_continuation', 'ref:7aa017f0',
                               'ref:bd504ecf'])):
        spec.append(('Table 12', 'tab:res_multiscenario', name, exp, cells))
    # Table 13: tab:res_year_ladder
    c = cl('B:n7_4h_e1')
    spec.append(('Table 13', 'tab:res_year_ladder', '2025 (B:n7_4h_e1)', [_k(c['d_gross'])],
                 ['ref:7aa017f0', 'ref:bd504ecf']))
    yl_cells = {'2030': 'yl_y2030', '2035': 'yl_y2035'}
    for y in ('2030', '2035'):
        r = yl[y]
        spec.append(('Table 13', 'tab:res_year_ladder', f'{y} (year_ladder.per_year)',
                     [_k(r['M_gross_w153_form']), _k(r['salvage_last']), _k(r['M_net_w153_form'])],
                     ['ref:7aa017f0', yl_cells[y]]))
    yt = t['year_ladder']
    spec.append(('Table 13', 'tab:res_year_ladder', '2035 - 2030', [_k(yt['D_gross']), _k(yt['D_net'])],
                 ['yl_y2030', 'yl_y2035']))
    # Table 14: tab:res_discount -- reweighting of the 2 % evaluations
    for r in disc:
        spec.append(('Table 14', 'tab:res_discount', f"rate {r['rate'] * 100:.0f} %",
                     [_k(r['V']), _k(r['value_minus_I']), _k(r['threshold_a61_conservative'])],
                     ['ref:7aa017f0', 'ref:bd504ecf']))

    # match every spec row to a printed row (all expected strings in that row, order-free)
    out = []
    rows_by_label = {}
    for tab, label, rid, exp, cells in spec:
        if label not in rows_by_label:
            rows_by_label[label] = table_rows(tex, label)
        rows, line = rows_by_label[label]
        hits = [r for r in rows if all(e in _norm(r) for e in exp)]
        ok = len(hits) >= 1
        if not ok:
            _fault(f'{tab} {label}: no printed row carries {exp} (row {rid})')
        out.append({'table': tab, 'label': label, 'label_line': line, 'row': rid, 'expected_strings': exp,
                    'printed_row_matched': hits[0] if hits else None, 'matched': ok, 'cells': cells})
    return out


# -------------------------------------------------------------------------------------------------- C code and case
def code_anchors():
    out = {}
    for key, fname, needle in CODE_ANCHORS:
        text = _read(fname)
        lines = [i + 1 for i, ln in enumerate(text.split('\n')) if needle in ln]
        if not lines:
            _fault(f'code anchor {key} not found in {fname}')
        out[key] = {'file': fname, 'lines': lines, 'needle': needle}
    consts = {}
    dtext = _read('definitions.py')
    for name in CONSTANT_NAMES:
        m = re.search(r'^' + name + r'\s*=\s*([0-9.eE+-]+)', dtext, re.M)
        if not m:
            _fault(f'constant {name} not found')
            continue
        consts[name] = float(m.group(1))
    return out, consts


def case_counts(years):
    """Non-fixed slack Var counts per scenario per block, from the code rules (anchored) and the case files."""
    srp = _read_json(SRP1_JSON)
    tso_case = srp['TransmissionNetwork']['name']
    dso = {str(d['connection_node_id']): d['name'] for d in srp['DistributionNetworks']}
    params = {}
    for name in [tso_case] + list(dso.values()):
        params[name] = _read_json(os.path.join(CASE_DIR, name, f'{name}_params.json'))
    ess = _read_json(ESS_PARAMS)
    interface = set(int(n) for n in dso)
    per_year = {}
    for y in years:
        rec = {}
        cj = _read_json(os.path.join(CASE_DIR, tso_case, f'{tso_case}_{y}.json'))
        p = params[tso_case]
        sl = p['slacks']
        enabled = [n['bus_i'] for n in cj['nodes']
                   if n['type'] != 3 and not (n['type'] == 2 and p.get('enforce_vg', False))]
        if not interface <= set(enabled):
            _fault(f'TSO interface nodes {interface} not all voltage-slack-enabled in {tso_case}_{y}')
        n_v = 2 * N_PERIODS * len([b for b in enabled if b not in interface]) if sl['grid_operation']['voltage'] else 0
        n_f = 2 * len([ld for ld in cj['loads'] if ld.get('fl_reg') and ld['bus'] not in interface]) \
            if p.get('fl_reg') else 0
        rec['TSO'] = {'case': tso_case, 'baseMVA': cj['baseMVA'], 'voltage_slack': n_v,
                      'flexibility_p_day_balance_slack': n_f,
                      'node_balance_slack': 0 if not (sl['node_balance']['active_power'] or
                                                     sl['node_balance']['reactive_power']) else None,
                      'branch_flow_slack': 0 if not sl['grid_operation']['branch_flow'] else None,
                      'local_ess_day_balance_slack': 0 if 'energy_storages' not in cj else None,
                      'n_nodes': len(cj['nodes']), 'voltage_enabled_nodes': enabled,
                      'interface_nodes_fixed': sorted(interface)}
        for node, name in dso.items():
            dj = _read_json(os.path.join(CASE_DIR, name, f'{name}_{y}.json'))
            p = params[name]
            sl = p['slacks']
            enabled = [n['bus_i'] for n in dj['nodes']
                       if n['type'] != 3 and not (n['type'] == 2 and p.get('enforce_vg', False))]
            rec[node] = {'case': name, 'baseMVA': dj['baseMVA'],
                         'voltage_slack': 2 * N_PERIODS * len(enabled) if sl['grid_operation']['voltage'] else 0,
                         'flexibility_p_day_balance_slack': 2 * len([ld for ld in dj['loads'] if ld.get('fl_reg')])
                         if p.get('fl_reg') else 0,
                         'node_balance_slack': 0 if not (sl['node_balance']['active_power'] or
                                                        sl['node_balance']['reactive_power']) else None,
                         'branch_flow_slack': 0 if not sl['grid_operation']['branch_flow'] else None,
                         'local_ess_day_balance_slack': 0 if 'energy_storages' not in dj else None,
                         'n_nodes': len(dj['nodes']), 'n_loads': len(dj['loads'])}
        per_year[y] = rec
    solver_opts = {name: params[name].get('solver', {}) for name in params}
    solver_opts['ESSO'] = ess.get('solver', {})
    brf = sorted(n for n, o in solver_opts.items() if 'bound_relax_factor' in json.dumps(o)
                 or 'honor_original_bounds' in json.dumps(o))
    return per_year, {'tso_case': tso_case, 'dso_cases': dso, 'esso_slacks_flag': ess.get('slacks'),
                      'case_files_setting_bound_relax_factor_or_honor_original_bounds': brf,
                      'n_periods_SRP1_json': srp.get('NumInstants')}


# ---------------------------------------------------------------------------------------------- D cell registry
def _find_key(obj, key):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out += _find_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out += _find_key(v, key)
    return out


def cell_registry(frozen, w169, named):
    t = frozen['tables']
    cells = dict(t['cells'])
    cells.update(t['cells_appended_w160'])
    plans = w169['plans']
    reg = {}
    order = list(named) + sorted(c for c in cells if c not in named)
    for c in order:
        p = plans[c]
        src = p['sources']
        rec_path = src['capacities']['path']
        clt_path = (src.get('closure') or src.get('closure_terminal_beside'))['path']
        eval_dir = os.path.dirname(rec_path)
        if c in cells:
            f = cells[c]
            status = f['status']
            point = f['k_star'] if status == 'certified' else f['end_cycle']
            point_kind = 'k*' if status == 'certified' else 'uncertified end'
            eval_key, cand_key = f['eval_key'], f['candidate_key']
            superseded = bool(f.get('superseded'))
            flagged = bool(f.get('flag_non_clean_turning_point'))
        else:   # the 3 x 3 cells (not in T2): production-exit point from W169's plan, checked against the record
            status = 'uncertified (production exit)' if c != 't11_3x3_x0_continuation' else \
                'uncertified (continuation terminal; post-hoc D source)'
            point = p['point']
            point_kind = p['point_kind']
            eval_key = cand_key = None
            superseded = flagged = False
        reg[c] = {'cell': c, 'named': c in named, 'status': status, 'point': point, 'point_kind': point_kind,
                  'eval_key_frozen': eval_key, 'candidate_key_frozen': cand_key, 'superseded': superseded,
                  'flagged_non_clean_turning_point': flagged, 'eval_dir': eval_dir, 'record': rec_path,
                  'component_levels': clt_path}
    return reg


# ---------------------------------------------------------------------------------------------- E capture plan
def capture_plan(reg):
    """Rule eleven: every quantity's source asserted to exist BEFORE extraction."""
    plan = {}
    for c, r in reg.items():
        need = {}
        for k in ('record', 'component_levels'):
            if not os.path.isfile(r[k]):
                _fault(f'{c}: {k} {r[k]} missing')
        rec = _read_json(r['record'])
        clt = _read_json(r['component_levels'])
        if r['eval_key_frozen'] and rec.get('eval_key') != r['eval_key_frozen']:
            _fault(f"{c}: record eval_key {rec.get('eval_key')} != frozen {r['eval_key_frozen']}")
        if r['candidate_key_frozen'] and rec.get('candidate_key') != r['candidate_key_frozen']:
            _fault(f"{c}: record candidate_key {rec.get('candidate_key')} != frozen {r['candidate_key_frozen']}")
        r['eval_key'] = rec.get('eval_key')
        r['candidate_key'] = rec.get('candidate_key')
        r['candidate_canonical'] = rec.get('candidate_canonical')
        r['cycles_run_record'] = rec.get('cycles_run')
        r['cycles_run_component_levels'] = clt.get('cycles_run')
        if r['eval_key_frozen'] is None and clt.get('cycles_run') != r['point']:
            _fault(f"{c}: production-exit point {r['point']} != component_levels cycles_run {clt.get('cycles_run')}")
        at_p = clt.get('cycles_run') == r['point']
        b0 = next(iter(clt['blocks'].values()))
        missing = [f for f in FAMILIES + ZERO_WEIGHT_TERMS if f not in b0['unweighted']]
        if at_p:
            if missing:
                _fault(f'{c}: component_levels lacks {missing}')
            if 'esso_feasibility_violation_D3' not in clt:
                _fault(f'{c}: component_levels lacks esso_feasibility_violation_D3')
            need['families'] = {'path': r['component_levels'], 'keys': 'blocks.*.{unweighted,weighted,'
                                'admm_block_weight}.' + '|'.join(FAMILIES + ZERO_WEIGHT_TERMS),
                                'at': f"cycle {r['point']} (cycles_run == P)"}
            need['esso_D3'] = {'path': r['component_levels'], 'keys': 'esso_feasibility_violation_D3',
                               'at': f"cycle {r['point']}"}
        else:
            per_cycle = os.path.join(r['eval_dir'], 'creep_diagnostic_per_cycle.jsonl')
            if not os.path.isfile(per_cycle):
                _fault(f'{c}: P {r["point"]} != component_levels cycles_run {clt.get("cycles_run")} and no '
                       f'per-cycle record')
            else:
                hit = _jsonl_find(per_cycle, lambda x: x.get('cycle') == r['point'])
                if len(hit) != 1 or 'q_decomposition' not in hit[0]:
                    _fault(f'{c}: per-cycle record has {len(hit)} lines at cycle {r["point"]}')
            need['network_and_closure_aggregate'] = {
                'path': per_cycle, 'keys': 'q_decomposition.blocks[*].{slack_penalties,ess_complementarity_penalties}'
                ' (weighted EUR; network families aggregated, closure local + shared aggregated) and '
                'q_decomposition.esso_side_terms_not_in_gross_Q.<node>.feasibility_penalty',
                'at': f"cycle {r['point']}",
                'why': f"component_levels_terminal.json is at cycle {clt.get('cycles_run')}, not P"}
            need['families_terminal_beside'] = {'path': r['component_levels'],
                                                'at': f"cycle {clt.get('cycles_run')} (terminal; NOT the point)"}
        need['block_weights'] = {'path': r['component_levels'], 'keys': 'blocks.*.admm_block_weight'}
        need['capacities'] = {'path': r['record'],
                              'keys': 'storage_per_node.<n>.published_available_capacity_terminal.<year>'}
        caps = {}
        cdir = os.path.join(r['eval_dir'], 'esso_capture')
        if os.path.isdir(cdir):
            for lab in sorted(os.listdir(cdir)):
                for n in ('5', '7', '9'):
                    fp = os.path.join(cdir, lab, f"node{n}_cycle{int(r['point']):03d}.jsonl")
                    if os.path.isfile(fp):
                        caps[n] = fp
        need['esso_elementwise'] = {'paths': caps, 'keys': 'slack_pnet_up, slack_pnet_down per (y, d, p) row '
                                    '(rows written for active cohort-periods only)', 'at': f"cycle {r['point']}"}
        man = os.path.join(r['eval_dir'], 'child_manifest_sha256.json')
        need['pickle_listing'] = {'path': man if os.path.isfile(man) else None,
                                  'keys': '<eval_dir>/*.pkl -> sha256 (listed, never opened)'}
        plan[c] = {'point': r['point'], 'point_kind': r['point_kind'], 'source_mode':
                   'component_levels_at_P' if at_p else 'per_cycle_at_P_aggregate', 'sources': need}
    return plan


# ---------------------------------------------------------------------------------------------- F extraction
def _scenario_count(rec):
    nm = _find_key(rec, 'num_market_scenarios')
    no = _find_key(rec, 'num_operation_scenarios')
    vals = [v for v in nm if isinstance(v, int)]
    for v in no:
        if isinstance(v, dict):
            vals += [x for x in v.values() if isinstance(x, int)]
        elif isinstance(v, int):
            vals.append(v)
    return max(vals) if vals else 1


def _active_units(rec, year):
    act = []
    for n, v in rec['storage_per_node'].items():
        cap = (v.get('published_available_capacity_terminal') or {}).get(str(year))
        if cap and abs(cap.get('s_available', 0.0)) > INACTIVE_TOL and abs(cap.get('e_available', 0.0)) > INACTIVE_TOL:
            act.append(n)
    return sorted(act)


def _coef(fam, base, consts):
    return {'voltage_slack': consts['PENALTY_VOLTAGE_SQUARED'],
            'node_balance_slack': base * consts['PENALTY_NODE_BALANCE'],
            'branch_flow_slack': base * consts['PENALTY_CURRENT'],
            'flexibility_p_day_balance_slack': base * consts['PENALTY_FLEXIBILITY'],
            'local_ess_day_balance_slack': base * consts['PENALTY_ESS_BALANCE'],
            'shared_ess_day_balance_slack': base * consts['PENALTY_SHARED_ESS_BALANCE']}[fam]


def _block_counts(block, counts, active):
    kind = block['kind']
    yr = str(block['year'])
    rec = counts[yr]['TSO'] if kind == 'TSO' else counts[yr][str(block['node_id'])]
    n = {f: rec[f] for f in NETWORK_FAMILIES + ('local_ess_day_balance_slack',)}
    if kind == 'TSO':
        n['shared_ess_day_balance_slack'] = 2 * len(active)
    else:
        n['shared_ess_day_balance_slack'] = 2 if str(block['node_id']) in active else 0
    return n, rec['baseMVA']


def extract_cell(c, r, plan_c, counts, consts):
    rec = _read_json(r['record'])
    clt = _read_json(r['component_levels'])
    n_scen = _scenario_count(rec)
    single = n_scen == 1
    out = {'cell': c, 'named': r['named'], 'status': r['status'], 'point': r['point'], 'point_kind': r['point_kind'],
           'eval_key': r['eval_key'], 'candidate_key': r['candidate_key'],
           'candidate_canonical': r['candidate_canonical'], 'superseded': r['superseded'],
           'flagged_non_clean_turning_point': r['flagged_non_clean_turning_point'],
           'scenario_count_max_per_dimension': n_scen, 'single_scenario': single,
           'source_mode': plan_c['source_mode'], 'eval_dir': r['eval_dir']}
    blocks = clt['blocks']
    years = sorted({str(b['year']) for b in blocks.values()})
    days = sorted({str(b['day']) for b in blocks.values()})
    act_by_year = {y: _active_units(rec, y) for y in years}
    out['active_units_by_year'] = act_by_year
    fam = {f: {'signed_eur_weighted': 0.0, 'positive_part_bound_eur_weighted': 0.0, 'max_block_abs_eur_weighted': 0.0,
               'max_block_S_natural': None, 'max_block_S_natural_at': None, 'per_element_upper_bound_max': None,
               'per_element_upper_bound_at': None, 'implied_count_over_declared_max': None, 'n_blocks_nonzero': 0,
               'nonzero_blocks_with_zero_declared_vars': [], 'unit': None}
           for f in FAMILIES}
    zero_terms = {z: 0.0 for z in ZERO_WEIGHT_TERMS}
    consistency = {'weighted_equals_unweighted_times_w_max_rel': 0.0}
    per_block = []
    if plan_c['source_mode'] == 'component_levels_at_P':
        for bk, b in blocks.items():
            w = b['admm_block_weight']
            n, base = _block_counts(b, counts, act_by_year[str(b['year'])])
            pb = {'block': bk, 'w': w}
            for f in FAMILIES:
                v = b['unweighted'][f]
                vw = b['weighted'][f]
                if v != 0.0:
                    rel = abs(vw - v * w) / max(abs(vw), 1e-300)
                    consistency['weighted_equals_unweighted_times_w_max_rel'] = max(
                        consistency['weighted_equals_unweighted_times_w_max_rel'], rel)
                cf = _coef(f, base, consts)
                N = n[f]
                F = fam[f]
                F['unit'] = 'p.u.^2 (V^2)' if f == 'voltage_slack' else 'p.u. (x baseMVA = MW or MWh)'
                F['signed_eur_weighted'] += v * w
                F['max_block_abs_eur_weighted'] = max(F['max_block_abs_eur_weighted'], abs(v * w))
                if v != 0.0:
                    F['n_blocks_nonzero'] += 1
                if N is None:
                    _fault(f'{c} {bk} {f}: family active in the case file; no count rule implemented')
                    continue
                if N == 0:
                    if v != 0.0:
                        F['nonzero_blocks_with_zero_declared_vars'].append({'block': bk, 'value': v})
                    continue
                S = v / cf
                if F['max_block_S_natural'] is None or S > F['max_block_S_natural']:
                    F['max_block_S_natural'], F['max_block_S_natural_at'] = S, bk
                bound = max(0.0, v + FLOOR * cf * N) * w
                F['positive_part_bound_eur_weighted'] += bound
                implied = (-S / FLOOR) / N
                if F['implied_count_over_declared_max'] is None or implied > F['implied_count_over_declared_max']:
                    F['implied_count_over_declared_max'] = implied
                if implied > 1.0 + 1e-9:
                    _fault(f'{c} {bk} {f}: implied count {-S / FLOOR:.4f} > declared {N} (floor premise contradicted)')
                if single:
                    ub = S + (N - 1) * FLOOR
                    if F['per_element_upper_bound_max'] is None or ub > F['per_element_upper_bound_max']:
                        F['per_element_upper_bound_max'], F['per_element_upper_bound_at'] = ub, bk
                pb[f] = {'V': v, 'N': N, 'c': cf, 'S': S, 'implied_count': -S / FLOOR, 'pos_bound_eur_w': bound}
            for z in ZERO_WEIGHT_TERMS:
                zero_terms[z] += b['weighted'][z]
            per_block.append(pb)
        detector_sum = sum(fam[f]['signed_eur_weighted'] for f in FAMILIES)
        rc = clt.get('recourse_components', {})
        out['detector_penalty_total_record'] = rc.get('detector_penalty_total')
        out['detector_penalty_total_component_levels'] = clt['totals_weighted']['detector_penalty_total']
        out['detector_sum_of_families'] = detector_sum
        out['detector_reconciles_abs'] = abs(detector_sum - clt['totals_weighted']['detector_penalty_total'])
        out['gross_operational_cost_at_P'] = rc.get('gross_operational_cost')
        # ESSO
        d3 = clt['esso_feasibility_violation_D3']
        n_esso = 2 * len(rec['storage_per_node']) * len(years) * len(days) * N_PERIODS
        out['esso'] = {'D3_sum_MW': d3, 'source': 'component_levels_terminal.json esso_feasibility_violation_D3',
                       'N_vars': n_esso, 'positive_part_bound_MW': max(0.0, d3 + FLOOR * n_esso),
                       'per_element_upper_bound_MW': d3 + (n_esso - 1) * FLOOR,
                       'implied_count_over_declared': (-d3 / FLOOR) / n_esso,
                       'esso_objective_value_c_sigma_x_D3': consts['PENALTY_ESSO_SLACK'] * d3,
                       'eur_content_in_Q': 0.0}
        if (-d3 / FLOOR) > n_esso * (1 + 1e-9):
            _fault(f'{c}: ESSO implied count {-d3 / FLOOR:.3f} > {n_esso}')
    else:
        # per-cycle aggregate at P (d_c52e1670): network families summed, closure local + shared summed
        pc = os.path.join(r['eval_dir'], 'creep_diagnostic_per_cycle.jsonl')
        rowp = _jsonl_find(pc, lambda x: x.get('cycle') == r['point'])[0]
        rowt = _jsonl_find(pc, lambda x: x.get('cycle') == clt['cycles_run'])
        qb = {(x['agent'], None if x['agent'] == 'TSO' else str(x['node_id']), str(x['year']), str(x['day'])): x
              for x in rowp['q_decomposition']['blocks']}
        agg = {'network': {'signed_eur_weighted': 0.0, 'positive_part_bound_eur_weighted': 0.0,
                           'max_block_abs_eur_weighted': 0.0, 'implied_fraction_max': None},
               'closure': {'signed_eur_weighted': 0.0, 'positive_part_bound_eur_weighted': 0.0,
                           'max_block_abs_eur_weighted': 0.0, 'implied_fraction_max': None,
                           'per_element_upper_bound_max_pu': None}}
        for bk, b in blocks.items():
            key = (b['kind'], None if b['kind'] == 'TSO' else str(b['node_id']), str(b['year']), str(b['day']))
            x = qb[key]
            w = b['admm_block_weight']
            n, base = _block_counts(b, counts, act_by_year[str(b['year'])])
            for grp, fams, field in (('network', NETWORK_FAMILIES, 'slack_penalties'),
                                     ('closure', CLOSURE_FAMILIES, 'ess_complementarity_penalties')):
                vw = x[field]
                sum_a = sum(_coef(f, base, consts) * n[f] for f in fams)
                A = agg[grp]
                A['signed_eur_weighted'] += vw
                A['max_block_abs_eur_weighted'] = max(A['max_block_abs_eur_weighted'], abs(vw))
                A['positive_part_bound_eur_weighted'] += max(0.0, vw + w * FLOOR * sum_a)
                if sum_a > 0 and single:
                    # any element j of the group: a_j x_j <= V - sum_{k != j} a_k x_k <= V + f (sum_a - a_j)
                    a_min = min(_coef(f, base, consts) for f in fams if n[f])
                    ub_mixed = (vw / w + FLOOR * sum_a) / a_min - FLOOR
                    key_ub = 'per_element_upper_bound_max_mixed'
                    A[key_ub] = ub_mixed if A.get(key_ub) is None else max(A[key_ub], ub_mixed)
                if sum_a > 0:
                    frac = (-vw / w) / (FLOOR * sum_a)
                    A['implied_fraction_max'] = frac if A['implied_fraction_max'] is None else max(
                        A['implied_fraction_max'], frac)
                    if frac > 1 + 1e-9:
                        _fault(f'{c} {bk} {grp}: implied |V| / (f sum a) = {frac} > 1')
                    if grp == 'closure' and single:
                        cf = _coef('shared_ess_day_balance_slack', base, consts)
                        N = n['shared_ess_day_balance_slack']
                        ub = (vw / w) / cf + (N - 1) * FLOOR
                        if A['per_element_upper_bound_max_pu'] is None or ub > A['per_element_upper_bound_max_pu']:
                            A['per_element_upper_bound_max_pu'] = ub
                elif vw != 0.0:
                    _fault(f'{c} {bk} {grp}: non-zero {vw} with zero declared Vars')
        out['aggregate_at_P'] = agg
        out['gross_operational_cost_at_P'] = rowp.get('gross')
        out['per_cycle_line_at_P'] = {'cycle': rowp['cycle'], 'local_solves_ok': rowp.get('local_solves_ok'),
                                      'reconciliation': rowp['q_decomposition'].get('reconciliation')}
        # positive control at the terminal cycle: per-cycle aggregates == component-level families
        ctl = None
        if len(rowt) == 1:
            qt = {(x['agent'], None if x['agent'] == 'TSO' else str(x['node_id']), str(x['year']), str(x['day'])): x
                  for x in rowt[0]['q_decomposition']['blocks']}
            dmax = 0.0
            for bk, b in blocks.items():
                key = (b['kind'], None if b['kind'] == 'TSO' else str(b['node_id']), str(b['year']), str(b['day']))
                net = sum(b['weighted'][f] for f in NETWORK_FAMILIES)
                clo = sum(b['weighted'][f] for f in CLOSURE_FAMILIES)
                dmax = max(dmax, abs(qt[key]['slack_penalties'] - net), abs(qt[key]['ess_complementarity_penalties']
                                                                           - clo))
            ctl = {'terminal_cycle': clt['cycles_run'], 'max_abs_diff_eur': dmax}
        out['positive_control_terminal'] = ctl
        # terminal per-family beside
        beside = {}
        for f in FAMILIES:
            beside[f] = sum(b['weighted'][f] for b in blocks.values())
        out['families_terminal_beside'] = {'cycle': clt['cycles_run'], 'signed_eur_weighted': beside}
        # ESSO at P: nodes without an active cohort -> feasibility_penalty / c_sigma (throughput fixed at 0, pure slack)
        es = rowp['q_decomposition'].get('esso_side_terms_not_in_gross_Q', {})
        act_any = sorted({n for y in years for n in act_by_year[y]})
        inactive_nodes = {n: es[n]['feasibility_penalty'] / consts['PENALTY_ESSO_SLACK'] for n in es
                          if n not in act_any}
        n_node = 2 * len(years) * len(days) * N_PERIODS
        inactive_ub = {n: v + (n_node - 1) * FLOOR for n, v in inactive_nodes.items()}
        out['esso'] = {'source': 'per-cycle esso_side_terms_not_in_gross_Q (inactive nodes: pure slack) + '
                                 'esso_capture (active nodes)', 'inactive_node_sigma_sum_MW': inactive_nodes,
                       'inactive_node_N_vars': n_node, 'inactive_node_per_element_upper_bound_MW': inactive_ub,
                       'eur_content_in_Q': 0.0, 'D3_terminal_beside_MW': clt['esso_feasibility_violation_D3']}
    out['families'] = fam if plan_c['source_mode'] == 'component_levels_at_P' else None
    out['zero_weight_terms_eur_weighted'] = zero_terms if plan_c['source_mode'] == 'component_levels_at_P' else None
    out['consistency'] = consistency
    out['per_block'] = per_block
    # x = 0 / inactive: closure exactly 0 in every block with no active unit
    if plan_c['source_mode'] == 'component_levels_at_P':
        bad = [bk for bk, b in blocks.items()
               if _block_counts(b, counts, act_by_year[str(b['year'])])[0]['shared_ess_day_balance_slack'] == 0
               and b['unweighted']['shared_ess_day_balance_slack'] != 0.0]
        out['closure_exactly_zero_where_no_active_unit'] = not bad
        if bad:
            _fault(f'{c}: closure non-zero with no active unit in {bad}')
    # ESSO element-wise
    ew = {}
    for n, fp in plan_c['sources']['esso_elementwise']['paths'].items():
        rows = _jsonl_find(fp, lambda x: True)
        if not rows:
            ew[n] = {'path': fp, 'rows': 0, 'status': INPUTS[os.path.relpath(os.path.abspath(fp), REPO)]['status']}
            continue
        up = [x['slack_pnet_up'] for x in rows]
        dn = [x['slack_pnet_down'] for x in rows]
        ydp = {(x['y'], x['d'], x['p']) for x in rows}
        ew[n] = {'path': fp, 'rows': len(rows), 'distinct_ydp': len(ydp), 'max_up_MW': max(up), 'min_up_MW': min(up),
                 'max_down_MW': max(dn), 'min_down_MW': min(dn),
                 'max_pair_sum_MW': max(a + b for a, b in zip(up, dn)),
                 'status': INPUTS[os.path.relpath(os.path.abspath(fp), REPO)]['status']}
    out['esso']['elementwise'] = ew
    # pickle-only listing
    pk = []
    man = plan_c['sources']['pickle_listing']['path']
    manifest = _read_json(man) if man else {}
    for fn in sorted(os.listdir(r['eval_dir'])):
        if fn.endswith('.pkl'):
            fp = os.path.join(r['eval_dir'], fn)
            pk.append({'path': fp, 'sha256_from_child_manifest': manifest.get(fp), 'exists': True,
                       'tracked': _tracked(fp), 'size_bytes': os.path.getsize(fp), 'opened': False})
    out['pickle_only'] = pk
    return out


WORKBOOK_SHEET = 'Relaxation Slacks TSO, DSOs'


def multiscenario_elementwise(c, r, counts, consts):
    """3 x 3 cells only (single-scenario cells are settled by the block sums). Sources: the run's UNCOMMITTED results
    workbook (results/*_terminal.xlsx, sheet 'Relaxation Slacks TSO, DSOs', written by
    shared_resources_planning.py from network.py `relaxation_slacks`: voltage slacks RAW per element; flexibility and
    closure only as the NET up - down x baseMVA) and the COMMITTED multiscenario_terminal.json (scenario
    probabilities). Voltage: the element-wise maximum. Flexibility P day balance: t = up + down >= |d| - 2f per pair
    (d = up - down, from the workbook), sum_s omega_s sum_j t_sj = T (committed block sum / c), so
    t_sj <= (T - sum_{other pairs} omega L) / omega_s and up, down <= (t_sj_max +- d) / 2. Closure: scenario-free,
    S + (N - 1) f per block."""
    import openpyxl   # third-party reader; not a production module
    wbs = sorted(os.path.join(r['eval_dir'], 'results', fn) for fn in os.listdir(os.path.join(r['eval_dir'], 'results'))
                 if fn.endswith('_terminal.xlsx'))
    ms_path = os.path.join(r['eval_dir'], 'multiscenario_terminal.json')
    if len(wbs) != 1 or not os.path.isfile(ms_path):
        return {'available': False, 'workbooks': wbs, 'multiscenario_terminal': os.path.isfile(ms_path)}
    wb_path = wbs[0]
    _reg(wb_path)
    ms = _read_json(ms_path)
    clt = _read_json(r['component_levels'])
    rec = _read_json(r['record'])
    prob = {}
    for bk, b in ms['blocks'].items():
        prob[bk] = {k: v['probability'] for k, v in b['per_scenario_costs'].items()}
    wb = openpyxl.load_workbook(wb_path, read_only=True, data_only=True)
    ws = wb[WORKBOOK_SHEET]
    vmax = {'TSO': -float('inf'), 'DSO': -float('inf')}
    vmax_at = {'TSO': None, 'DSO': None}
    n_v = {'TSO': [0, 0], 'DSO': [0, 0]}          # [non-zero elements, exactly-zero elements]
    flex = {}
    clos_net = {}
    hdr = None
    for row in ws.iter_rows(values_only=True):
        if hdr is None:
            hdr = row
            continue
        op, adn, res, yr, day, q, sm, so = row[:8]
        vals = [v for v in row[8:8 + N_PERIODS]]
        if q in ('Voltage squared slack, down [p.u.^2]', 'Voltage squared slack, up [p.u.^2]'):
            for t_, v in enumerate(vals):
                if v == 0:
                    n_v[op][1] += 1     # fixed / [0, 0] Vars (REF bus, TSO interface buses) are written as 0
                    continue
                n_v[op][0] += 1
                if v > vmax[op]:
                    vmax[op], vmax_at[op] = v, f'{op}|{adn}|bus{res}|{yr}|{day}|s{sm}_{so}|t{t_}|{q}'
        elif q == 'Flex. balance, p' and op == 'DSO':
            if len(set(vals)) != 1:
                _fault(f'{c}: workbook flex balance row not constant over periods ({op} {adn} {res} {yr} {day})')
            flex.setdefault(f'DSO|{adn}|{yr}|{day}', {})[(f'{sm}_{so}', res)] = vals[0]
        elif q == 'Shared Energy Storage, soc_final':
            clos_net[f'{op}|{adn}|{res}|{yr}|{day}|{sm}_{so}'] = vals[0]
    wb.close()
    # flexibility t-bound per DSO block
    fl_ub = -float('inf')
    fl_ub_at = None
    for bk, pairs in flex.items():
        b = clt['blocks'][bk]
        base = counts[str(b['year'])][str(b['node_id'])]['baseMVA']
        cf = _coef('flexibility_p_day_balance_slack', base, consts)
        T = b['unweighted']['flexibility_p_day_balance_slack'] / cf
        n_decl = counts[str(b['year'])][str(b['node_id'])]['flexibility_p_day_balance_slack']
        if 2 * len({k[1] for k in pairs}) != n_decl:
            _fault(f'{c} {bk}: workbook flex loads {len({k[1] for k in pairs})} != declared pairs {n_decl // 2}')
        pr = prob[bk]
        lsum = sum(pr[sk] * (abs(d / base) - 2 * FLOOR) for (sk, _ld), d in pairs.items())
        for (sk, ld), d in pairs.items():
            dd = d / base
            lo = abs(dd) - 2 * FLOOR
            tmax = (T - (lsum - pr[sk] * lo)) / pr[sk]
            ub = (tmax + abs(dd)) / 2
            if ub > fl_ub:
                fl_ub, fl_ub_at = ub, f'{bk}|s{sk}|load{ld}'
    # closure S + (N - 1) f (scenario-free Vars)
    cl_ub = -float('inf')
    for bk, b in clt['blocks'].items():
        act = _active_units(rec, b['year'])
        N = 2 * len(act) if b['kind'] == 'TSO' else (2 if str(b['node_id']) in act else 0)
        if N:
            base = counts[str(b['year'])]['TSO' if b['kind'] == 'TSO' else str(b['node_id'])]['baseMVA']
            S = b['unweighted']['shared_ess_day_balance_slack'] / _coef('shared_ess_day_balance_slack', base, consts)
            cl_ub = max(cl_ub, S + (N - 1) * FLOOR)
    return {'available': True, 'workbook': wb_path, 'workbook_status': INPUTS[os.path.relpath(
        os.path.abspath(wb_path), REPO)]['status'], 'multiscenario_terminal': ms_path,
            'voltage_elementwise_max_over_nonzero': vmax, 'voltage_elementwise_max_at': vmax_at,
            'voltage_counts_nonzero_zero': n_v,
            'flex_pair_upper_bound_max_pu': fl_ub, 'flex_pair_upper_bound_at': fl_ub_at,
            'flex_net_max_abs_MWh': max((abs(d) for pr_ in flex.values() for d in pr_.values()), default=None),
            'closure_per_element_upper_bound_max_pu': cl_ub if cl_ub > -float('inf') else None,
            'closure_net_workbook_max_abs_MWh': max((abs(v) for v in clos_net.values()), default=None),
            'proven_at_or_below_zero': {'voltage_slack': max(vmax.values()) <= 0.0,
                                        'flexibility_p_day_balance_slack': fl_ub <= 0.0,
                                        'shared_ess_day_balance_slack': (cl_ub <= 0.0) if cl_ub > -float('inf')
                                        else True},
            'every_element_at_or_below_zero': (max(vmax.values()) <= 0.0 and fl_ub <= 0.0 and
                                               (cl_ub <= 0.0 if cl_ub > -float('inf') else True))}


# ---------------------------------------------------------------------------------------------- G scoring
def score(cell_out):
    """P5 readings per evaluation.
    R3 per family: 'proven' when every slack element is shown <= 0 (i.e. in [-f, 0]); 'demonstrated_positive' when a
    block's (probability-weighted) slack sum is > 0, so some element is > 0; otherwise 'not_established'.
    Positive EUR content in Q: 0 for a proven family; else bounded by the family's positive-part bound (R1).
    P5 holds on a row when the refined bound on the positive content in Q is < 1 EUR.
    R2 (reported, not the prediction's reading): |signed EUR content in Q| < 1 EUR."""
    status = {}
    r1 = {}
    if cell_out['source_mode'] == 'component_levels_at_P':
        fam = cell_out['families']
        signed = sum(fam[f]['signed_eur_weighted'] for f in FAMILIES)
        me = cell_out.get('multiscenario_elementwise') or {}
        for f in FAMILIES:
            F = fam[f]
            r1[f] = F['positive_part_bound_eur_weighted']
            if F['max_block_S_natural'] is None and not F['nonzero_blocks_with_zero_declared_vars']:
                status[f] = 'no non-fixed Vars (identically 0)'
            elif F['nonzero_blocks_with_zero_declared_vars'] or (F['max_block_S_natural'] or 0.0) > 0.0:
                status[f] = 'demonstrated_positive'
            elif cell_out['single_scenario']:
                status[f] = 'proven' if F['per_element_upper_bound_max'] <= 0.0 else 'not_established'
            elif me.get('available') and f in me['proven_at_or_below_zero']:
                status[f] = 'proven' if me['proven_at_or_below_zero'][f] else 'not_established'
            else:
                status[f] = 'not_established'
    else:
        a = cell_out['aggregate_at_P']
        signed = a['network']['signed_eur_weighted'] + a['closure']['signed_eur_weighted']
        for grp in ('network', 'closure'):
            r1[grp] = a[grp]['positive_part_bound_eur_weighted']
            ubs = [u for u in (a[grp].get('per_element_upper_bound_max_mixed'),
                               a[grp].get('per_element_upper_bound_max_pu')) if u is not None]
            if not ubs:
                status[grp] = 'no non-fixed Vars (identically 0)' if a[grp]['signed_eur_weighted'] == 0.0 \
                    else 'not_established'
            else:
                status[grp] = 'proven' if (cell_out['single_scenario'] and max(ubs) <= 0.0) else 'not_established'
    refined = sum(0.0 if status[k] in ('proven', 'no non-fixed Vars (identically 0)') else r1[k] for k in status)
    every = all(v in ('proven', 'no non-fixed Vars (identically 0)') for v in status.values())
    demo = [k for k, v in status.items() if v == 'demonstrated_positive']
    e = cell_out['esso']
    if 'per_element_upper_bound_MW' in e:
        esso_ok = e['per_element_upper_bound_MW'] <= 0.0
    else:   # per-cycle mode: inactive nodes by their sums, active nodes element-wise from esso_capture
        ubs = list(e.get('inactive_node_per_element_upper_bound_MW', {}).values())
        ubs += [v['max_up_MW'] for v in e.get('elementwise', {}).values() if v.get('rows')]
        ubs += [v['max_down_MW'] for v in e.get('elementwise', {}).values() if v.get('rows')]
        esso_ok = (max(ubs) <= 0.0) if ubs else None
    return {'family_status': status, 'family_positive_part_bound_eur_R1': r1,
            'positive_part_bound_eur_in_Q_R1_all_families': sum(r1.values()),
            'positive_content_bound_eur_in_Q_refined': refined,
            'P5_row_holds_positive_content_below_1_eur': refined < ONE_EUR,
            'every_element_proven_at_or_below_zero': every, 'families_demonstrated_positive': demo,
            'signed_eur_in_Q': signed, 'R2_abs_signed_content_below_1_eur': abs(signed) < ONE_EUR,
            'esso_positive_part_bound_MW': e.get('positive_part_bound_MW'), 'esso_every_element_at_or_below_zero': esso_ok}


# ---------------------------------------------------------------------------------------------- H reconciliation
def reconcile_t01c():
    out = {}
    for k, d in RECON_CELLS.items():
        clt = _read_json(os.path.join(d, 'component_levels_terminal.json'))
        rec = _read_json(os.path.join(d, 'evaluation_record.json'))
        dpt = rec['recourse_components']['detector_penalty_total']
        out[k] = {'eval_dir': d, 'cycles_run': clt['cycles_run'], 'eval_key': rec.get('eval_key'),
                  'candidate_key': rec.get('candidate_key'), 'detector_penalty_total_record': dpt,
                  'families_weighted': {f: clt['totals_weighted'][f] for f in FAMILIES},
                  'T01C_correction_Qcert_plus_minus_gross': T01C_CORRECTION[k],
                  'T01C_plus_detector_abs': abs(T01C_CORRECTION[k] + dpt)}
    return out


# ---------------------------------------------------------------------------------------------- main
def main():
    out_dir = OUT_DIR
    if '--out-dir' in sys.argv:                 # trial runs only; the directory must hold nothing but launch.log
        out_dir = sys.argv[sys.argv.index('--out-dir') + 1]
    if not os.path.isdir(out_dir):
        raise SystemExit(f'REFUSED: {out_dir} does not exist (create it with mkdir; it must hold only launch.log)')
    extra = sorted(set(os.listdir(out_dir)) - {'launch.log'})
    if extra:
        raise SystemExit(f'REFUSED: {out_dir} already holds {extra}')
    out_json = os.path.join(out_dir, OUT_JSON_NAME)
    out_plan = os.path.join(out_dir, OUT_PLAN_NAME)
    out_manifest = os.path.join(out_dir, OUT_MANIFEST_NAME)
    started = datetime.now(timezone.utc).isoformat()
    head = _git(['rev-parse', 'HEAD']).strip()

    # A. manuscript + frozen JSON
    tex, ms = read_manuscript()
    fsha = _sha(FROZEN)
    if fsha != FROZEN_SHA256:
        _fault(f'frozen JSON sha256 {fsha} != {FROZEN_SHA256}')
    frozen = _read_json(FROZEN)
    _reg(EXPORT_T1)
    w169 = _read_json(W169_PLAN)

    # B. table -> evaluation map (asserted against the printed numbers)
    tmap = build_table_map(tex, frozen)
    named = []
    for r in tmap:
        for c in r['cells']:
            if c not in named:
                named.append(c)
    if FAULTS:
        raise SystemExit(f'REFUSED before extraction (table map): {FAULTS}')

    # C. code + case
    anchors, consts = code_anchors()
    reg = cell_registry(frozen, w169, named)
    years_all = set()
    for c, r in reg.items():
        clt = _read_json(r['component_levels'])
        years_all |= {str(b['year']) for b in clt['blocks'].values()}
    counts, case_meta = case_counts(sorted(years_all))
    if case_meta['case_files_setting_bound_relax_factor_or_honor_original_bounds']:
        _fault(f"case/ESS solver options set bound_relax_factor or honor_original_bounds: "
               f"{case_meta['case_files_setting_bound_relax_factor_or_honor_original_bounds']}")
    if case_meta['n_periods_SRP1_json'] != N_PERIODS:
        _fault(f"SRP1.json NumInstants {case_meta['n_periods_SRP1_json']} != {N_PERIODS}")
    # identical counts across years (structure does not change by year)
    sig = {y: json.dumps({k: {f: v.get(f) for f in NETWORK_FAMILIES} for k, v in counts[y].items()}, sort_keys=True)
           for y in counts}
    counts_identical_across_years = len(set(sig.values())) == 1
    if not counts_identical_across_years:
        _fault('slack Var counts differ across case years')
    rec_opt_hits = {}   # every bound option a record states, by value (a record that states one is evidence)
    for c, r in reg.items():
        rec = _read_json(r['record'])
        brf = [v.get('value') if isinstance(v, dict) else v for v in _find_key(rec, 'bound_relax_factor')]
        hob = [v.get('value') if isinstance(v, dict) else v for v in _find_key(rec, 'honor_original_bounds')]
        rec_opt_hits[c] = {'bound_relax_factor_values': sorted(set(map(str, brf))),
                           'honor_original_bounds_values': sorted(set(map(str, hob))), 'n_statements': len(brf) + len(hob)}
        if any(v != 1e-08 for v in brf) or any(v != 'no' for v in hob):
            _fault(f'{c}: record states bound options other than the default: brf {brf} hob {hob}')

    # D/E. capture plan, asserted and written BEFORE extraction
    plan = capture_plan(reg)
    if FAULTS:
        raise SystemExit(f'REFUSED before extraction (capture plan): {FAULTS}')
    with open(out_plan, 'w', encoding='utf-8') as f:
        gate_result_io.dump({'written_before_extraction': True, 'written_utc': datetime.now(timezone.utc).isoformat(),
                             'plans': plan}, f, indent=1)
        f.write('\n')
    print(f'capture plan written before extraction: {len(plan)} cells ({len(named)} named)', flush=True)

    # F. extraction
    cells_out = {}
    for c, r in reg.items():
        cells_out[c] = extract_cell(c, r, plan[c], counts, consts)
        if not cells_out[c]['single_scenario']:
            cells_out[c]['multiscenario_elementwise'] = multiscenario_elementwise(c, r, counts, consts)
        cells_out[c]['score'] = score(cells_out[c])
        cells_out[c]['tables'] = sorted({f"{t['table']}" for t in tmap if c in t['cells']})

    # G. P5 over the named set
    named_cert = [c for c in named if reg[c]['status'] == 'certified']
    named_unc = [c for c in named if reg[c]['status'] != 'certified']

    def _sel(cs, pred):
        return [c for c in cs if pred(cells_out[c]['score'])]
    p5 = {
        'prediction': 'every penalised slack at its zero bound on every certified row (EUR content < 1 EUR); any '
                      'non-zero content is listed with its row (expert, Addendum 89 P5)',
        'reading': 'at its zero bound = every slack element of every penalised family in Q shown in [-f, 0] '
                   '(f = 1e-8, the IPOPT relaxation), so its positive (penalty-paid) EUR content is 0; where an '
                   'element is not established <= 0, the positive content is bounded by the family positive-part '
                   'bound; the row holds when that refined bound is < 1 EUR',
        'reading_R2_reported_beside': '|signed EUR content in Q| < 1 EUR (counts the relaxed-bound residue)',
        'named_certified': named_cert, 'named_uncertified_final_iteration': named_unc,
        'certified_holds': _sel(named_cert, lambda x: x['P5_row_holds_positive_content_below_1_eur']),
        'certified_fails': _sel(named_cert, lambda x: not x['P5_row_holds_positive_content_below_1_eur']),
        'certified_every_element_proven': _sel(named_cert, lambda x: x['every_element_proven_at_or_below_zero']),
        'certified_demonstrated_positive': _sel(named_cert, lambda x: x['families_demonstrated_positive']),
        'certified_R2_holds': _sel(named_cert, lambda x: x['R2_abs_signed_content_below_1_eur']),
        'uncertified_holds': _sel(named_unc, lambda x: x['P5_row_holds_positive_content_below_1_eur']),
        'uncertified_fails': _sel(named_unc, lambda x: not x['P5_row_holds_positive_content_below_1_eur']),
        'uncertified_every_element_proven': _sel(named_unc, lambda x: x['every_element_proven_at_or_below_zero']),
        'uncertified_demonstrated_positive': _sel(named_unc, lambda x: x['families_demonstrated_positive']),
        'esso_every_element_at_or_below_zero_named': _sel(named, lambda x: x['esso_every_element_at_or_below_zero']
                                                          is True),
        'esso_not_established_named': _sel(named, lambda x: x['esso_every_element_at_or_below_zero'] is not True),
    }
    p5['P5_holds_on_every_named_certified_row'] = not p5['certified_fails']
    beside = [c for c in reg if not reg[c]['named']]
    p5['beside_all_other_frozen_cells'] = {
        'cells': beside,
        'holds': _sel(beside, lambda x: x['P5_row_holds_positive_content_below_1_eur']),
        'fails': _sel(beside, lambda x: not x['P5_row_holds_positive_content_below_1_eur']),
        'demonstrated_positive': _sel(beside, lambda x: x['families_demonstrated_positive']),
        'not_every_element_proven': _sel(beside, lambda x: not x['every_element_proven_at_or_below_zero'])}

    recon = reconcile_t01c()

    # compact per-evaluation rows (the report's table is read from here)
    def _row(c):
        o = cells_out[c]
        sc = o['score']
        row = {'cell': c, 'tables': o['tables'], 'status': o['status'], 'point': o['point'],
               'eval_key': o['eval_key'], 'candidate_key': o['candidate_key'],
               'candidate_canonical': o['candidate_canonical'], 'superseded': o['superseded'],
               'source_mode': o['source_mode'], 'signed_eur_in_Q_total': sc['signed_eur_in_Q'],
               'positive_content_bound_eur_in_Q_refined': sc['positive_content_bound_eur_in_Q_refined'],
               'positive_part_bound_eur_R1_all_families': sc['positive_part_bound_eur_in_Q_R1_all_families'],
               'family_status': sc['family_status'], 'P5_row_holds': sc['P5_row_holds_positive_content_below_1_eur'],
               'esso_every_element_at_or_below_zero': sc['esso_every_element_at_or_below_zero'],
               'pickle_only': [{'path': x['path'], 'sha256': x['sha256_from_child_manifest']} for x in o['pickle_only']]}
        if o['source_mode'] == 'component_levels_at_P':
            fam = o['families']
            row['signed_eur_in_Q'] = {f: fam[f]['signed_eur_weighted'] for f in FAMILIES}
            row['max_block_sum_natural'] = {f: fam[f]['max_block_S_natural'] for f in FAMILIES}
            row['per_element_upper_bound_max'] = {f: fam[f]['per_element_upper_bound_max'] for f in FAMILIES}
            row['esso_D3_sum_MW'] = o['esso']['D3_sum_MW']
            row['esso_per_element_upper_bound_MW'] = o['esso']['per_element_upper_bound_MW']
        else:
            a = o['aggregate_at_P']
            row['signed_eur_in_Q'] = {'network_families_aggregate': a['network']['signed_eur_weighted'],
                                      'closure_families_aggregate': a['closure']['signed_eur_weighted']}
            row['per_element_upper_bound_max'] = {
                'network_families_aggregate': a['network'].get('per_element_upper_bound_max_mixed'),
                'closure_families_aggregate': a['closure'].get('per_element_upper_bound_max_pu')}
            row['esso_inactive_node_per_element_upper_bound_MW'] = o['esso']['inactive_node_per_element_upper_bound_MW']
        me = o.get('multiscenario_elementwise')
        if me and me.get('available'):
            row['workbook_voltage_max_over_nonzero'] = me['voltage_elementwise_max_over_nonzero']
            row['workbook_flex_pair_upper_bound_max_pu'] = me['flex_pair_upper_bound_max_pu']
            row['workbook_flex_pair_upper_bound_at'] = me['flex_pair_upper_bound_at']
            row['closure_per_element_upper_bound_max_pu'] = me['closure_per_element_upper_bound_max_pu']
        ew = o['esso'].get('elementwise', {})
        row['esso_capture_max_up_down_MW'] = {n: (v.get('max_up_MW'), v.get('max_down_MW'), v['status'])
                                              for n, v in ew.items() if v.get('rows')}
        return row
    summary_rows = [_row(c) for c in reg]
    sig = [cells_out[c]['score']['signed_eur_in_Q'] for c in named_cert]
    p5['signed_content_named_certified'] = {'min': min(sig), 'max': max(sig), 'spread': max(sig) - min(sig)}

    result = {
        'stage': 'P5.15 Addendum 89 W188 (a) -- slack inventory (G10)',
        'authority': 'PLANNER_BRIEF_2026-09-13.md Addendum 89 (W188 (a), P5); manuscript_review/'
                     'REFOCUS_ANALYSIS_2026-11-10.md G10',
        'started_utc': started, 'repo_head': head,
        'objective_convention': 'Q = gross_operational_cost (settlement excluded; the frozen primary). EUR content in '
                                'Q of a family = sum_b w_b V_b with w_b = Y_y D_d (1+r)^-(y-y0) (get_primal_value '
                                'weight); SIGNED as recorded: negative where IPOPT leaves the slack inside its relaxed '
                                'lower bound. Q contains the signed content.',
        'sign_convention': 'signed_eur_weighted < 0 means the slack Vars sit below 0 inside the IPOPT bound '
                           'relaxation (Q lowered by that amount); positive_part_bound_eur_weighted is a rigorous '
                           'upper bound on sum_j a_j max(x_j, 0), the economic (penalty-paid) content',
        'floor_premise': {'f': FLOOR, 'basis': 'IPOPT bound_relax_factor default 1e-8 (relaxed lb = -1e-8 * max(1, '
                          '|lb|), lb = 0), honor_original_bounds no (IPOPT 3.14 default); interior-point iterates stay '
                          'inside the relaxed bounds; no case/ESS options file sets either option (scanned); a cell '
                          'record that states the options in force must state exactly these values (asserted; '
                          'record_option_mentions lists what each record states)',
                          'record_option_mentions': rec_opt_hits},
        'A_manuscript': ms, 'frozen_json': {'path': FROZEN, 'sha256': fsha, 'matches_declared': fsha == FROZEN_SHA256},
        'B_table_map': tmap, 'named_cells': named,
        'C_code_anchors': anchors, 'C_constants': consts, 'C_case_counts': counts, 'C_case_meta': case_meta,
        'C_counts_identical_across_years': counts_identical_across_years,
        'W169_capture_plan_used_as_path_index': {'path': W169_PLAN, **INPUTS[W169_PLAN]},
        'P5': p5, 'summary_rows': summary_rows, 'cells': cells_out, 'reconciliation_T01C': recon, 'faults': FAULTS,
    }
    bad = sorted(m for m in sys.modules if m.split('.')[0] in FORBIDDEN_MODULE_PREFIXES
                 or m.startswith(('shared_resources_planning', 'model_construction_helpers', 'admm_')))
    gfail = GUARD.verify(0)
    if bad or gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        raise SystemExit(f'guard fault: production modules {bad}; guard {gfail}; pickle {PICKLE_COUNTS}')
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts,
                                                'verify_0_failures': gfail},
                        'pickle_counts': PICKLE_COUNTS, 'production_modules_imported': bad}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(out_json, 'w', encoding='utf-8') as f:
        gate_result_io.dump(result, f, indent=1)
        f.write('\n')
    script = os.path.basename(__file__)
    manifest = {'outputs': {out_json: _sha(out_json), out_plan: _sha(out_plan)}, 'script': {script: _sha(script)},
                'script_tracked': _tracked(script), 'repo_head': head,
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(out_manifest, 'w', encoding='utf-8') as f:
        gate_result_io.dump(manifest, f, indent=1, sort_keys=True)
        f.write('\n')

    # ------------------------------------------------------------------------------------------- console summary
    print(f"manuscript {ms['commit']} blob {ms['blob'][:8]} md5 {ms['md5']} ok {ms['md5_matches_declared_prefix']}")
    print(f"frozen {fsha[:8]} ok {fsha == FROZEN_SHA256}; table rows matched {sum(t['matched'] for t in tmap)}/"
          f"{len(tmap)}; named cells {len(named)}")
    print(f"counts per scenario (2025): {json.dumps({k: {f: v[f] for f in NETWORK_FAMILIES} for k, v in counts[sorted(counts)[0]].items()})}")
    for c in list(named) + beside:
        o = cells_out[c]
        s = o['score']
        if o['source_mode'] == 'component_levels_at_P':
            fam = o['families']
            fs = ' '.join(f"{f.split('_')[0][:5]}={fam[f]['signed_eur_weighted']:.2f}/"
                          f"{fam[f]['positive_part_bound_eur_weighted']:.3g}" for f in
                          ('voltage_slack', 'flexibility_p_day_balance_slack', 'shared_ess_day_balance_slack'))
            ub = max([fam[f]['per_element_upper_bound_max'] for f in FAMILIES
                      if fam[f]['per_element_upper_bound_max'] is not None], default=None)
        else:
            a = o['aggregate_at_P']
            fs = (f"net={a['network']['signed_eur_weighted']:.2f}/{a['network']['positive_part_bound_eur_weighted']:.3g}"
                  f" clos={a['closure']['signed_eur_weighted']:.2f}/{a['closure']['positive_part_bound_eur_weighted']:.3g}")
            ub = max(u for u in (a['network'].get('per_element_upper_bound_max_mixed'),
                                 a['closure'].get('per_element_upper_bound_max_mixed'),
                                 a['closure'].get('per_element_upper_bound_max_pu')) if u is not None)
        print(f"{'N' if o['named'] else 'b'} {c:<24} {o['status'][:12]:<12} P={o['point']:<4} {o['source_mode'][:9]} "
              f"signed={s['signed_eur_in_Q']:.2f} pos<={s['positive_content_bound_eur_in_Q_refined']:.4g} "
              f"(R1 {s['positive_part_bound_eur_in_Q_R1_all_families']:.4g}) holds={s['P5_row_holds_positive_content_below_1_eur']} "
              f"status={s['family_status']} ub={ub} | {fs} | esso_ok={s['esso_every_element_at_or_below_zero']}")
    print('P5:', json.dumps({k: v for k, v in p5.items() if k.startswith(('uncert', 'P5', 'cert', 'esso'))}))
    for c in reg:
        me = cells_out[c].get('multiscenario_elementwise')
        if me:
            print(f"multiscenario {c}: {json.dumps({k: v for k, v in me.items() if k not in ('workbook', 'multiscenario_terminal')})}")
    print('beside:', json.dumps({k: v for k, v in p5['beside_all_other_frozen_cells'].items() if k not in ('cells', 'holds')}))
    print('T0.1c reconciliation:', json.dumps({k: (v['detector_penalty_total_record'], v['T01C_plus_detector_abs'])
                                               for k, v in recon.items()}))
    print(f"faults {FAULTS}")
    print(f"inputs {len(INPUTS)} ({manifest['n_uncommitted_inputs']} uncommitted)")
    print(f"guards: {result['guards']}")
    print(f'output {out_json} sha256 {_sha(out_json)}')
    print(f'plan {out_plan} sha256 {_sha(out_plan)}')
    print(f'manifest {out_manifest} sha256 {_sha(out_manifest)}')


if __name__ == '__main__':
    main()
