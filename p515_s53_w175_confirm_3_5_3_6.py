"""
P5.15 Addendum 70, Planner task W175 -- the two `% [CONFIRM -- W175]` comments of manuscript sections 3.5 and 3.6
(Overleaf clone manuscript/6a67305f25e8348fb71380c3/ at 260bd83, main.tex sha256 cb237b2d...). ZERO SOLVES, READ-ONLY.

Comment 1 (sec. 3.5, after the ageing table):
  (1a) the mid-block SoH point SoH_{y-1} e^{-D/2} phi^{Y/2} (shared_energy_storage_data.py:625-626): the code lines at
       HEAD (found by exact text), which capacity it sets (E^Av of the cohort-block, summed over cohorts and handed to
       the network models as the block's storage energy), what still uses the END-of-block SoH, the arm that runs it
       (C3_midblock in the ext spec v3 84775dc4) and the numerical read-back recorded by that run;
  (1b) the "no ageing" arm: its spec field (84775dc4 model_variant.arms.no_ageing), the code path (D == 0 row, phi = 1,
       floor row kept), the run's read-back and its terminal SoH per cohort-block (E^Av = E^Rated iff SoH = 1);
  (1c) the chemistry: every cell of data/SRP1/SharedESS/SRP1_ESS.xlsx (openpyxl, read-only), SRP1_ESS_Params.json and
       every SRP1 case JSON scanned for chemistry terms; a scoped `git grep` of the tracked tree outside
       data/SRP1/Results for LFP / iron phosphate / LiFePO / MB31.
Comment 2 (sec. 3.6, the benchmark paragraph): uncoordinated_benchmark.py and the W116 harness by exact text; the
  benchmark spec v5 bca69f97 (tie-breakers, objective convention, consistency convention); the six NRF arm records
  (decision / evaluation tie-breakers, phase A Q, consistency trigger, phase C Q, arm-cost source); report_v3 (tie
  breakers, claim, value independence); the v2 common-Q gate (status, bitwise fields); the frozen Step-6 table T6;
  the passive DSO decision objective per block with and without the no-reverse-flow rule (sweep_passive_cold vs
  nrf_arm_passive_cold_r2 per-solve records); the code identity of the benchmark modules at HEAD against the spec's
  sha256 binding.

GUARDS: SolveProfileGuard(permitted=()) installed before anything else is imported from the project and verified at
exactly 0 (solves and process launches); pickle.load / pickle.loads blocked for the whole run, counters verified at 0.
No production module is imported; code is read as text.

Output (refuses to overwrite): data/SRP1/Results/P515S53/w175_confirm/w175_confirm.json and manifest_sha256.json
(sha256 of every input read and of the output). Command (repo root, canonical interpreter, attached, both streams):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w175_confirm_3_5_3_6.py \\
      > data/SRP1/Results/P515S53/w175_confirm/launch.log 2>&1
Exit: 0 written and every integrity check holds; 3 written with integrity failures listed; 1 guard fault / refused.
"""
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W175 CONFIRM 3.5/3.6 (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W175: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W175: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import openpyxl  # noqa: E402  (read-only workbook scan; third-party, not project code)

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w175_confirm')
OUT_JSON = os.path.join(OUT_DIR, 'w175_confirm.json')
OUT_MAN = os.path.join(OUT_DIR, 'manifest_sha256.json')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
MAIN_TEX_SHA = 'cb237b2d129e689a94dd50abec9a9cca739567ef97a486ec012a901f0e04b6df'
CLONE_HEAD = '260bd83'

R = os.path.join('data', 'SRP1', 'Results')
EXT_SPEC = os.path.join(R, 'P515S53', 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json')
EXT_SPEC_SHA = '84775dc41428ba6a6b504c86de728f3d2b44d19354c977c18d3b33f197c76ee6'
_EXT = os.path.join(R, 'P515S53', 'w142_resettle_ext_v6')
AGEING_RECORDS = {
    'e_no_ageing': os.path.join(_EXT, 'campaign_s53_w142_resettle_ext_v6_e_no_ageing', 'evals',
                                '1b10e9e161523f46_e_no_ageing', 'evaluation_record.json'),
    'e_c3_midblock': os.path.join(_EXT, 'campaign_s53_w142_resettle_ext_v6_e_c3_midblock', 'evals',
                                  '8b1eace91df86cd5_e_c3_midblock', 'evaluation_record.json'),
}
FROZEN_TABLES = os.path.join(R, 'P515S53', 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_TABLES_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'

_B = os.path.join(R, 'P515S53', 'w116_benchmark_nrf')
BENCH_SPEC = os.path.join(_B, 'frozen_s53_benchmark_spec_v5_bca69f97.json')
BENCH_SPEC_SHA = 'bca69f97a6e60e92e3a7dc51d6bdc9f71fe463c50a840952d9d6fc78bac7d2c6'
REPORT_V3 = os.path.join(_B, 'report_v3', 'report_v3.json')
COMMON_Q_GATE = os.path.join(R, 'P515S53', 'w106_uncoordinated_settled', 'common_q_gate', 'common_q_gate.json')
COMMON_Q_GATE_SHA = '8e1a60f6c7e6e4b9e421004782b031a6813930a1a615f7d8a3e5b89bd9767f5e'
NRF_RUNS = [f'nrf_arm_{a}_{s}_r2' for a in ('passive', 'price_taker')
            for s in ('cold', 'warm_from_certified', 'perturbed')]
SWEEP_PASSIVE_SOLVES = os.path.join(_B, 'sweep_passive_cold', 'per_solve_record.jsonl')
NRF_PASSIVE_COLD_SOLVES = os.path.join(_B, 'nrf_arm_passive_cold_r2', 'per_solve_record.jsonl')

ESS_XLSX = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
CASE_JSONS = [os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SRP1_params.json')] + [
    os.path.join('data', 'SRP1', c, f'{c}_params.json') for c in ('case33_1', 'case33_2', 'case33_3', 'case9')]
CHEM_RE = re.compile(r'lfp|lithium|phosphate|lifepo|li-ion|li ion|nmc|chemistr|mb31|\beve\b', re.I)

# code locations, found by exact text (file, needle, tag); every needle must be found at least once
CODE_NEEDLES = [
    # (1a) mid-block SoH point
    ('shared_energy_storage_data.py',
     'soh_mid = prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y] / 2.00) * (phi_cal ** (num_years / 2.00))',
     'mid_soh_formula'),
    ('shared_energy_storage_data.py',
     'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == '
     'model.es_e_rated_per_unit[y_inv, y] * soh_mid)', 'mid_available_energy_row'),
    ('shared_energy_storage_data.py',
     'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == '
     'model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y])', 'end_available_energy_row'),
    ('shared_energy_storage_data.py', 'prev_soh = model.es_soh_per_unit_cumul[y_inv, y - 1]', 'prev_soh_end_of_y-1'),
    ('shared_energy_storage_data.py', "if soh_point == 'mid':", 'soh_point_switch'),
    ('shared_energy_storage_data.py', 'phi_cal = shared_energy_storage.phi_cal if ageing_enabled else 1.00',
     'mid_phi'),
    ('shared_energy_storage_data.py',
     '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))', 'end_soh_chain_row'),
    ('shared_energy_storage_data.py', 'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)',
     'soh_floor_row_on_end_soh'),
    ('shared_energy_storage_data.py',
     'e_available = e_rated * model.es_soh_per_unit_cumul[y_inv, terminal_year_idx]', 'salvage_uses_end_soh'),
    ('shared_energy_storage_data.py', 'SoH_mid_y = SoH_{y-1} * exp(-D_y / 2) * phi**(n / 2).', 'docstring_mid'),
    ('shared_energy_storage_data.py',
     "that curve at t = n / 2, which is also the GEOMETRIC mean of the block's", 'docstring_geometric_mean'),
    ('shared_energy_storage_data.py', 'e_available += pe.value(model.es_e_available_per_unit[y_inv, year_idx])',
     'block_E_av_sum_over_cohorts'),
    ('shared_resources_planning.py',
     "model[year][day].shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['e_available'] "
     "/ s_base)", 'E_av_to_network_models'),
    ('shared_resources_planning.py', 'shared_ess_capacity = self.shared_ess_data.get_available_capacity(',
     'available_capacity_read'),
    # (1b) no ageing
    ('shared_energy_storage_data.py', '# P5.15 Addenda 28-29 (task W20): ageing off -- no calendar loss.',
     'no_ageing_phi_1_comment'),
    ('shared_energy_storage_data.py', 'model.es_D_per_unit[y_inv, y] == 0.00)', 'no_ageing_D_eq_0_row'),
    ('shared_energy_storage_data.py', '# "infinite k"). Same row family and position, so the',
     'no_ageing_no_infinite_k'),
    ('shared_energy_storage_data.py', 'E_rated. The soh_min floor row is kept (trivially satisfied).',
     'no_ageing_floor_kept'),
    ('p515_s44_campaign_harness.py', 'def apply_model_variant(sed, model_variant):', 'apply_model_variant'),
    ('p515_s44_campaign_harness.py', "'d_row_form': 'D * 2kE == 365 n avg' if enabled else 'D == 0'}",
     'variant_expected_d_row'),
    ('p515_s44_campaign_harness.py',
     "'phi_cal_in_model': model_variant['calendar_retention_per_year'] if enabled else 1.0,",
     'variant_expected_phi'),
    # (2) benchmark
    ('uncoordinated_benchmark.py', 'every flexibility Var', 'passive_doc_flex_fixed'),
    ('uncoordinated_benchmark.py', 'minimum-curtailment selection rule in an otherwise empty objective).',
     'passive_doc_tie_breaker'),
    ('uncoordinated_benchmark.py', 'var_data.fix(0.0)', 'passive_flex_fixed_code'),
    ('uncoordinated_benchmark.py', 'block.penalty_gen_curtailment.set_value(float(curtailment_penalty))',
     'decision_tie_breaker_set'),
    ('uncoordinated_benchmark.py', '-- lambda = 0, rho = 0 because the augmented-Lagrangian terms are never built;',
     'price_taker_doc'),
    ('uncoordinated_benchmark.py', 'srp._prepare_distribution_objectives_for_admm(distribution_networks, dso_models)',
     'price_taker_pricing_code'),
    ('uncoordinated_benchmark.py', 'return m.pg_adn[s_m, s_o, p] >= 0.0', 'no_reverse_flow_row'),
    ('uncoordinated_benchmark.py',
     'return m.expected_interface_pf_p[dn, p] == getattr(m, FIXED_INTERFACE_P_TARGET)[dn, p]', 'tso_p_fixed_row'),
    ('uncoordinated_benchmark.py',
     'return m.expected_interface_pf_q[dn, p] == getattr(m, FIXED_INTERFACE_Q_TARGET)[dn, p]', 'tso_q_fixed_row'),
    ('uncoordinated_benchmark.py',
     'TSO_ARM_PIN_INTERFACE_VOLTAGE = False                    # ruling 2: the arm\'s interface voltage is within '
     'its bounds', 'tso_voltage_unpinned'),
    ('uncoordinated_benchmark.py', 'pin_interface_voltage=TSO_ARM_PIN_INTERFACE_VOLTAGE)', 'tso_arm_call_unpinned'),
    ('uncoordinated_benchmark.py', 'STARTS = (START_COLD, START_WARM, START_PERTURBED)', 'three_starts'),
    ('uncoordinated_benchmark.py',
     'block.penalty_gen_curtailment.set_value(float(evaluation_curtailment_penalty))', 'evaluation_tie_breaker_set'),
    ('uncoordinated_benchmark.py', 'components = srp._get_operational_recourse_components(planning_problem, models)',
     'common_q_production_components'),
    ('uncoordinated_benchmark.py', 'setpoint voltage), then the TSO solves its production subproblem with the '
     'interface P/Q hard-fixed (mutable', 'arm_sequence_doc'),
    ('p515_s53_w116_benchmark_nrf.py',
     "shift = UB.pin_dso_interface_voltage(planning, arm_out['models']['dso'], mismatch['v_actual_dn_pu'])",
     'seq_pass_dso_at_tn_voltage'),
    ('p515_s53_w116_benchmark_nrf.py',
     "UB.solve_tso_model(planning, arm_out['models']['tso'], phase='sequential_pass:tso', record_callback=sink)",
     'seq_pass_tso_resolve'),
    ('p515_s53_w116_benchmark_nrf.py', "arm_cost_source = 'phase_C_sequential_pass'", 'arm_cost_after_pass'),
    ('p515_s53_w116_benchmark_nrf.py', 'best_start = min(costs, key=costs.get)', 'best_of_three'),
    ('shared_resources_planning.py', 'dso_model[year][day].interface_settlement_weight.set_value(1.00)',
     'dso_settlement_weight_1'),
    ('shared_resources_planning.py', 'dso_model[year][day].penalty_gen_curtailment.set_value(0.00)',
     'dso_admm_tie_breaker_0'),
    ('model_construction_helpers.py',
     'settlement += probability * c_p[p] * network.baseMVA * model.pg_adn[s_m, s_o, p]', 'dso_settlement_at_pi'),
    ('model_construction_helpers.py', 'model.interface_settlement_weight = pe.Param(initialize=0.00, mutable=True)',
     'settlement_weight_build_default_0'),
    ('model_construction_helpers.py',
     'model.penalty_gen_curtailment = pe.Param(initialize=PENALTY_GENERATION_CURTAILMENT, mutable=True)',
     'tie_breaker_build_default'),
    ('definitions.py', 'PENALTY_GENERATION_CURTAILMENT = 1e0', 'tie_breaker_value'),
]

# manuscript locations (needle -> lines); every needle must be found
TEX_NEEDLES = {
    'sec_3_5_heading': r'\subsection{\textcolor{blue}{Shared Energy Storage Parameters}}',
    'sec_3_6_heading': r'\subsection{\textcolor{blue}{Evaluation and Certification Settings}}',
    'lfp_sentence': 'The shared ESS is a utility-scale lithium iron phosphate battery',
    'ageing_sensitivities_sentence': 'evaluated at the end of the block or at its midpoint',
    'table_caption_soh_point': "that sets a block's available energy (end of block, or its midpoint).",
    'table_row_midblock': r'Retention 50\,\%, mid-block',
    'table_row_no_ageing': r'& $\infty$ & 1.000 & --- & --- \\',
    'confirm_w175_1': r'% [CONFIRM — W175] the mid-block SoH point',
    'benchmark_paragraph_start': r'\textcolor{blue}{The value of coordination (Section~4.4)',
    'benchmark_paragraph_end': 'of three solver starts.}',
    'confirm_w175_2': r'% [CONFIRM — W175] the benchmark paragraph',
    'sec2_available_energy_sentence': 'reduced by its state of health at the end of the representative year',
    'sec2_soh_chain': r'\label{eq:soh_chain}',
}

# the benchmark code files whose sha256 the spec v5 binds
FAILED = []


def _log(msg):
    print(f'[{datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(*args, cwd=REPO, check=True):
    return subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True, check=check).stdout.strip()


def _load(path):
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _check(name, ok, detail=None):
    if not ok:
        FAILED.append({'check': name, 'detail': detail})
    return bool(ok)


def _tracked_clean(path):
    tracked = bool(_git('ls-files', '--', path))
    clean = _git('status', '--porcelain', '--', path) == ''
    return {'git_tracked': tracked, 'git_clean': clean}


def code_locations():
    out, cache = [], {}
    for rel, needle, tag in CODE_NEEDLES:
        if rel not in cache:
            with open(rel, encoding='utf-8') as fh:
                cache[rel] = fh.read().split('\n')
        lines = [i + 1 for i, ln in enumerate(cache[rel]) if needle in ln]
        _check(f'code needle found: {tag} ({rel})', bool(lines), needle)
        out.append({'tag': tag, 'file': rel, 'lines': lines, 'needle': needle})
    return out


def tex_locations(tex_lines):
    out = {}
    for tag, needle in TEX_NEEDLES.items():
        lines = [i + 1 for i, ln in enumerate(tex_lines) if needle in ln]
        _check(f'tex needle found: {tag}', bool(lines), needle)
        out[tag] = lines
    return out


def tex_block(tex_lines, first, last):
    return '\n'.join(tex_lines[first - 1:last])


def ageing_items(spec, records):
    mv = spec['model_variant']
    arms = mv['arms']
    cells = {k: v.get('model_variant') for k, v in spec['cells'].items()}
    out = {'spec': {'path': EXT_SPEC, 'sha256': EXT_SPEC_SHA, 'git_head': spec['git_head'],
                    'arms': arms, 'k_closed_form_by_arm': mv['k_closed_form_by_arm'],
                    'floor_row_lower_expected': mv['floor_row_lower_expected'], 'cells': cells,
                    'cell_order': spec['cell_order']}}
    _check('spec cell e_c3_midblock = arm C3_midblock', cells.get('e_c3_midblock') == arms['C3_midblock'])
    _check('spec cell e_no_ageing = arm no_ageing', cells.get('e_no_ageing') == arms['no_ageing'])
    _check('only C3_midblock uses the mid point',
           [a for a, v in arms.items() if v['available_energy_soh_point'] == 'mid'] == ['C3_midblock'])
    _check('only no_ageing has ageing off', [a for a, v in arms.items() if not v['ageing_enabled']] == ['no_ageing'])
    for name, rec in records.items():
        r = {'path': AGEING_RECORDS[name], 'status': rec.get('status'),
             'certified_cost_gross': rec.get('certified_cost'),
             'candidate_key': rec.get('candidate_key'), 'candidate_canonical': rec.get('candidate_canonical'),
             'model_variant': rec.get('model_variant'),
             'minimum_soh_in_configuration': rec.get('configuration', {}).get('ess_ageing_baseline', {})
             .get('minimum_soh')}
        for phase in ('model_variant_readback_pre_run', 'model_variant_readback_terminal'):
            rb = rec.get(phase) or {}
            r[phase] = {'all_match': rb.get('all_match'), 'expected': rb.get('expected'),
                        'per_node_readback': {n: v.get('readback') for n, v in (rb.get('per_node') or {}).items()},
                        'per_node_checks': {n: v.get('checks') for n, v in (rb.get('per_node') or {}).items()}}
            _check(f'{name} {phase} all_match is True', rb.get('all_match') is True, rb.get('all_match'))
        r['terminal_soh'] = {n: {'per_active_cohort_year': v.get('terminal_soh_per_active_cohort_year'),
                                 'min': v.get('terminal_soh_min_over_active_cohort_years'),
                                 'floor_rows_active_at_terminal': v.get('soh_floor_rows_active_at_terminal')}
                             for n, v in (rec.get('storage_per_node') or {}).items()}
        out[name] = r
    na = out['e_no_ageing']
    soh7 = (na['terminal_soh'].get('7') or {}).get('per_active_cohort_year') or {}
    _check('no_ageing: node-7 SoH = 1.0 at every active cohort-block', soh7 and all(v == 1.0 for v in soh7.values()),
           soh7)
    for phase in ('model_variant_readback_pre_run', 'model_variant_readback_terminal'):
        for n, rb in na[phase]['per_node_readback'].items():
            _check(f'no_ageing {phase} node {n}: D == 0, k None, phi 1.0, floor 0.70',
                   rb['d_row_form'] == 'D == 0' and rb['k'] is None and rb['phi_cal_in_model'] == 1.0
                   and rb['floor_row_lower'] == 0.7, rb)
        for n, rb in out['e_c3_midblock'][phase]['per_node_readback'].items():
            pr = rb['available_row_probe']
            _check(f'c3_midblock {phase} node {n}: mid point, X == X_mid_closed_form, k 11541.56, phi 1.0',
                   rb['available_energy_soh_point'] == 'mid' and pr['X'] == pr['X_mid_closed_form']
                   and abs(rb['k'] - 11541.560327111707) < 1e-6 and rb['phi_cal_in_model'] == 1.0, rb)
    return out


def chemistry_scan(xlsx_path, json_paths):
    out = {'xlsx': {'path': xlsx_path, 'sheets': {}, 'hits': [], 'cells_scanned': 0},
           'json': {}, 'git_grep': {}}
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=False)
    for ws in wb.worksheets:
        n = 0
        for row in ws.iter_rows():
            for c in row:
                n += 1
                v = c.value
                if isinstance(v, str) and CHEM_RE.search(v):
                    out['xlsx']['hits'].append({'sheet': ws.title, 'cell': c.coordinate, 'value': v})
        out['xlsx']['sheets'][ws.title] = n
        out['xlsx']['cells_scanned'] += n
    props = wb.properties
    out['xlsx']['document_properties'] = {k: getattr(props, k, None) for k in
                                          ('title', 'subject', 'description', 'keywords', 'category', 'creator')}
    wb.close()
    for p in json_paths:
        with open(p, encoding='utf-8') as fh:
            text = fh.read()
        out['json'][p] = [m.group(0) for m in CHEM_RE.finditer(text)]
    pat = ['-e', 'LFP', '-e', 'iron phosphate', '-e', 'LiFePO', '-e', 'MB31']
    res = subprocess.run(['git', 'grep', '-n', '-i', '-I', *pat, '--', '.', ':!data/SRP1/Results'], cwd=REPO,
                         capture_output=True, text=True)
    out['git_grep']['outside_results'] = {
        'scope': 'git grep -n -i -I over the TRACKED tree at HEAD (working tree), excluding data/SRP1/Results; '
                 'patterns LFP | iron phosphate | LiFePO | MB31 (case-insensitive)',
        'hits': [ln[:300] for ln in res.stdout.splitlines()]}
    res2 = subprocess.run(['git', 'grep', '-l', '-i', '-I', *pat, '--', 'data/SRP1/Results'], cwd=REPO,
                          capture_output=True, text=True)
    out['git_grep']['results_tree_files'] = {
        'scope': 'git grep -l (files only) over the tracked data/SRP1/Results tree, same patterns',
        'files': res2.stdout.splitlines()}
    return out


def _per_solve_dso_objectives(path):
    out = {}
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            if not r['block'].startswith('DSO|'):
                continue
            rnd = r.get('round') or (r.get('attempts') or [{}])[0].get('round')
            out.setdefault(r['block'], {})[rnd] = r.get('active_objective_value')
    return out


def benchmark_items(spec, report, gate, tables):
    runs = {}
    for rid in NRF_RUNS:
        p = os.path.join(_B, rid, f'{rid}.json')
        d = _load(p)
        pb = d.get('phase_B_consistency') or {}
        pc = d.get('phase_C_sequential_pass') or {}
        runs[rid] = {
            'path': p, 'sha256': _sha(p), 'arm': d['arm'], 'start': d['start'],
            'no_reverse_flow': d.get('no_reverse_flow'),
            'decision_tie_breaker': d['decision_tie_breaker'], 'evaluation_tie_breaker': d['evaluation_tie_breaker'],
            'phase_A_gross': d['phase_A']['evaluation'].get('gross_operational_cost'),
            'phase_B_max_abs_dv_dn_pu': pb.get('max_abs_dv_dn_pu'),
            'phase_B_trigger_sequential_pass': (pb.get('reevaluation') or {}).get('trigger_sequential_pass'),
            'phase_B_nrf_violations_at_actual_voltage': pb.get('no_reverse_flow_violations_at_actual_voltage'),
            'phase_C_gross': (pc.get('evaluation') or {}).get('gross_operational_cost'),
            'phase_C_effect_on_q_eur': pc.get('effect_on_q_eur'),
            'phase_C_max_abs_dv_dn_pu_after_pass': pc.get('max_abs_dv_dn_pu_after_pass'),
            'arm_cost': {k: v for k, v in d['arm_cost'].items() if k != 'objective_convention'},
            'instance': d.get('instance'),
        }
        st = _tracked_clean(p)
        _check(f'{rid} tracked and clean', st['git_tracked'] and st['git_clean'], st)
        want_dso = 1.0 if d['arm'] == 'passive' else 0.0
        _check(f'{rid} decision tie-breaker dso {want_dso}, tso 0; evaluation 0',
               d['decision_tie_breaker'] == {'dso': want_dso, 'tso': 0.0} and d['evaluation_tie_breaker'] == 0.0,
               [d['decision_tie_breaker'], d['evaluation_tie_breaker']])
    by_arm = {}
    for arm in ('passive', 'price_taker'):
        costs = {runs[r]['start']: runs[r]['arm_cost']['gross_operational_cost'] for r in runs
                 if runs[r]['arm'] == arm}
        best = min(costs, key=costs.get)
        by_arm[arm] = {'q_by_start': costs, 'best_start': best, 'q_best': costs[best],
                       'band': max(costs.values()) - min(costs.values()),
                       'sources': sorted({runs[r]['arm_cost']['source'] for r in runs if runs[r]['arm'] == arm}),
                       'all_triggered': all(runs[r]['phase_B_trigger_sequential_pass'] is True for r in runs
                                            if runs[r]['arm'] == arm)}
        t6 = tables['benchmark']['arms'][arm]
        _check(f'{arm}: best-of-three recomputed = T6', t6['Q_best'] == costs[best] and t6['best_start'] == best,
               [t6, costs])
    sweep = _per_solve_dso_objectives(SWEEP_PASSIVE_SOLVES)
    nrf = _per_solve_dso_objectives(NRF_PASSIVE_COLD_SOLVES)
    rows = []
    for blk in sorted(nrf):
        rows.append({'block': blk, 'unconstrained_obj_eur_rep_day': sweep.get(blk, {}).get('sweep:passive:cold:dso'),
                     'nrf_phase_A_obj_eur_rep_day': nrf[blk].get('passive:cold:dso'),
                     'nrf_seq_pass_obj_eur_rep_day': nrf[blk].get('sequential_pass:dso')})
    unc = [r['unconstrained_obj_eur_rep_day'] for r in rows]
    _check('passive comparison: 36 DSO blocks with both values', len(rows) == 36 and None not in unc, len(rows))
    passive_cmp = {
        'what': 'passive DSO DECISION objective value per block (EUR per representative day; the 1 EUR/MWh '
                'minimum-curtailment term is its only economic term: settlement weight 0, flexibility fixed at 0, '
                'l_curt false) -- without the no-reverse-flow rule (W120 sweep_passive_cold, spec-v2 arm, cold) and '
                'with it (nrf_arm_passive_cold_r2, phase A and the sequential pass)',
        'sources': {SWEEP_PASSIVE_SOLVES: _sha(SWEEP_PASSIVE_SOLVES), NRF_PASSIVE_COLD_SOLVES:
                    _sha(NRF_PASSIVE_COLD_SOLVES)},
        'unconstrained_min': min(unc), 'unconstrained_max': max(unc),
        'blocks_with_nrf_objective_above_1_eur': [r for r in rows if (r['nrf_phase_A_obj_eur_rep_day'] or 0) > 1.0],
        'n_blocks': len(rows), 'rows': rows}
    g = gate['gate']
    out = {
        'spec': {'path': BENCH_SPEC, 'sha256': BENCH_SPEC_SHA, 'git_head': spec['git_head'],
                 'tie_breaker': spec['tie_breaker'], 'objective_convention': spec['objective_convention'],
                 'arm_economies': spec['no_reverse_flow_addendum_57']['arm_economies'],
                 'no_reverse_flow_row': spec['no_reverse_flow_addendum_57']['row'],
                 'no_reverse_flow_not_in': spec['no_reverse_flow_addendum_57']['not_in'],
                 'consistency_convention': spec['consistency_convention'],
                 'consistency_convention_v3': spec['consistency_convention_v3'],
                 'claim_q_arm': spec['claim']['q_arm'], 'claim_gate_first': spec['claim']['gate_first'],
                 'instance': spec.get('instance')},
        'report_v3': {'path': REPORT_V3, 'sha256': _sha(REPORT_V3), 'tie_breaker': report['tie_breaker'],
                      'common_q_gate_status_v2': report['common_q_gate_status_v2'],
                      'claim': {k: report['claim'].get(k) for k in ('benefit_eur', 'benefit_relative', 'definition',
                                                                     'determinate', 'best_uncoordinated_arm',
                                                                     'objective_convention', 'measures')},
                      'passive_tie_breaker_value_independence_nrf': report['passive_tie_breaker_value_independence_nrf'],
                      'instance': report['instance']},
        'common_q_gate': {'path': COMMON_Q_GATE, 'sha256': COMMON_Q_GATE_SHA,
                          'status': g['status'], 'passed': g['passed'], 'gross_evaluated': g['gross_evaluated'],
                          'gross_certified': g['gross_certified'], 'difference': g['difference'],
                          'n_fields_bitwise_equal': g['n_fields_bitwise_equal'], 'n_fields': g['n_fields'],
                          'evaluation_curtailment_penalty': gate['evaluation_at_0']
                          .get('evaluation_curtailment_penalty')},
        'nrf_runs': runs, 'per_arm': by_arm, 'passive_with_vs_without_rule': passive_cmp,
        'frozen_T6': {k: tables['benchmark'][k] for k in ('arms', 'benefit', 'benefit_relative', 'coordinated_Q',
                                                           'definition', 'objective_convention', 'instance')},
    }
    _check('common-Q gate PASS_BITWISE at evaluation tie-breaker 0',
           g['status'] == 'PASS_BITWISE' and g['passed'] is True and g['difference'] == 0.0
           and out['common_q_gate']['evaluation_curtailment_penalty'] == 0.0, out['common_q_gate'])
    _check('report_v3 tie-breaker: decision passive 1 / price-taker 0 / tso 0; evaluation 0',
           report['tie_breaker']['decision'] == {'passive_dso': 1.0, 'price_taker_dso': 0.0, 'tso': 0.0}
           and report['tie_breaker']['evaluation'] == 0.0, report['tie_breaker'])
    return out


def code_identity(spec):
    binding = spec['code_sha256_binding']
    rows = {}
    for rel, want in binding.items():
        have = _sha(rel)
        rows[rel] = {'bound': want, 'head': have, 'equal': have == want}
    _check('benchmark code at HEAD = spec v5 binding (every file)', all(r['equal'] for r in rows.values()),
           {k: v for k, v in rows.items() if not v['equal']})
    ess_diff = _git('diff', '--stat', '13825451206ae9d2a313668c1c2eb17239d93892', 'HEAD', '--',
                    'shared_energy_storage_data.py', 'shared_energy_storage_parameters.py',
                    'data/SRP1/SharedESS/SRP1_ESS_Params.json')
    _check('ESS model code unchanged since the ext spec git_head', ess_diff == '', ess_diff)
    xlsx_blob_head = _git('rev-parse', 'HEAD:data/SRP1/SharedESS/SRP1_ESS.xlsx')
    xlsx_blob_7ce = _git('rev-parse', '7ce1d1ab:data/SRP1/SharedESS/SRP1_ESS.xlsx')
    _check('SRP1_ESS.xlsx at HEAD = 7ce1d1ab', xlsx_blob_head == xlsx_blob_7ce, [xlsx_blob_head, xlsx_blob_7ce])
    return {'benchmark_binding_vs_head': rows,
            'ess_code_diff_since_ext_spec_git_head': ess_diff or '(none)',
            'ess_xlsx_blob': {'HEAD': xlsx_blob_head, '7ce1d1ab': xlsx_blob_7ce}}


def main():
    t0 = datetime.now(timezone.utc)
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_MAN):
        _log(f'[W175] REFUSED: {OUT_DIR} already holds an output; not overwritten')
        sys.exit(1)
    nrf_paths = [os.path.join(_B, rid, f'{rid}.json') for rid in NRF_RUNS]
    inputs = ([MAIN_TEX, EXT_SPEC, FROZEN_TABLES, BENCH_SPEC, REPORT_V3, COMMON_Q_GATE, SWEEP_PASSIVE_SOLVES,
               NRF_PASSIVE_COLD_SOLVES, ESS_XLSX, ESS_PARAMS] + list(AGEING_RECORDS.values()) + nrf_paths
              + CASE_JSONS + sorted({rel for rel, _n, _t in CODE_NEEDLES}) + ['p513_solve_profile_guard.py'])
    before = {p: _sha(p) for p in inputs}
    _check('main.tex sha256 = declared', before[MAIN_TEX] == MAIN_TEX_SHA, before[MAIN_TEX])
    head = _git('-C', CLONE, 'rev-parse', '--short=7', 'HEAD')
    _check(f'clone HEAD = {CLONE_HEAD}', head == CLONE_HEAD, head)
    _check('ext spec sha256', before[EXT_SPEC] == EXT_SPEC_SHA, before[EXT_SPEC])
    _check('frozen tables sha256', before[FROZEN_TABLES] == FROZEN_TABLES_SHA, before[FROZEN_TABLES])
    _check('benchmark spec v5 sha256', before[BENCH_SPEC] == BENCH_SPEC_SHA, before[BENCH_SPEC])
    _check('common-Q gate sha256', before[COMMON_Q_GATE] == COMMON_Q_GATE_SHA, before[COMMON_Q_GATE])
    tracked = {}
    for p in [EXT_SPEC, FROZEN_TABLES, BENCH_SPEC, REPORT_V3, COMMON_Q_GATE, SWEEP_PASSIVE_SOLVES,
              NRF_PASSIVE_COLD_SOLVES, ESS_XLSX, ESS_PARAMS] + list(AGEING_RECORDS.values()):
        tracked[p] = _tracked_clean(p)
        _check(f'{p} tracked and clean', tracked[p]['git_tracked'] and tracked[p]['git_clean'], tracked[p])

    with open(MAIN_TEX, encoding='utf-8') as fh:
        tex_lines = fh.read().split('\n')
    tex = tex_locations(tex_lines)
    quotes = {}
    if tex['benchmark_paragraph_start'] and tex['benchmark_paragraph_end']:
        quotes['benchmark_paragraph'] = {'lines': [tex['benchmark_paragraph_start'][0],
                                                   tex['benchmark_paragraph_end'][0]],
                                         'text': tex_block(tex_lines, tex['benchmark_paragraph_start'][0],
                                                           tex['benchmark_paragraph_end'][0])}
    if tex['confirm_w175_1']:
        quotes['confirm_w175_1'] = {'lines': [tex['confirm_w175_1'][0], tex['confirm_w175_1'][0] + 2],
                                    'text': tex_block(tex_lines, tex['confirm_w175_1'][0],
                                                      tex['confirm_w175_1'][0] + 2)}
    if tex['confirm_w175_2']:
        quotes['confirm_w175_2'] = {'lines': [tex['confirm_w175_2'][0], tex['confirm_w175_2'][0] + 3],
                                    'text': tex_block(tex_lines, tex['confirm_w175_2'][0],
                                                      tex['confirm_w175_2'][0] + 3)}
    for tag in ('lfp_sentence', 'table_caption_soh_point', 'table_row_midblock', 'table_row_no_ageing',
                'ageing_sensitivities_sentence', 'sec2_available_energy_sentence'):
        if tex[tag]:
            quotes[tag] = {'line': tex[tag][0], 'text': tex_lines[tex[tag][0] - 1].strip()}
    # does sec. 3.5 print the mid-block formula? (any exp / e^ / SoH_{ inside 3.5)
    s35, s36 = tex['sec_3_5_heading'][0], tex['sec_3_6_heading'][0]
    sec35 = [(i, tex_lines[i - 1]) for i in range(s35, s36)
             if not tex_lines[i - 1].lstrip().startswith('%')
             and re.search(r'\\exp|e\^\{|SoH_\{|\\sqrt', tex_lines[i - 1])]
    quotes['sec_3_5_formula_lines_outside_comments'] = sec35

    ext_spec = _load(EXT_SPEC)
    records = {k: _load(p) for k, p in AGEING_RECORDS.items()}
    tables = _load(FROZEN_TABLES)['tables']
    ageing = ageing_items(ext_spec, records)
    ageing['frozen_T8_rows'] = [r for r in tables['ageing']['rows'] if r['arm'] in ('C3_midblock', 'no_ageing')]
    chem = chemistry_scan(ESS_XLSX, [ESS_PARAMS] + CASE_JSONS)
    bench = benchmark_items(_load(BENCH_SPEC), _load(REPORT_V3), _load(COMMON_Q_GATE), tables)
    ident = code_identity(_load(BENCH_SPEC))
    code = code_locations()

    after = {p: _sha(p) for p in inputs}
    _check('inputs unchanged during the run', after == before)

    gfail = GUARD.verify(0)
    if gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads']:
        _log(f'[W175 GUARD FAULT] {gfail} pickle {PICKLE_COUNTS}')
        sys.exit(1)

    result = {
        'stage': 'P5.15 Addendum 70 W175 -- the CONFIRM comments of manuscript sections 3.5 and 3.6 (mid-block SoH '
                 'point, no-ageing arm, chemistry, benchmark arm definitions)',
        'script': os.path.basename(__file__), 'script_sha256': _sha(__file__),
        'git_HEAD': _git('rev-parse', 'HEAD'), 'interpreter': sys.executable,
        'started_utc': t0.isoformat(), 'finished_utc': datetime.now(timezone.utc).isoformat(),
        'objective_convention': 'Q = gross_operational_cost, settlement excluded (production '
                                '_get_operational_recourse_components); benchmark arms evaluated at curtailment '
                                'tie-breaker 0; salvage not subtracted',
        'instances': {'ageing_arms': {'candidate_key': records['e_no_ageing'].get('candidate_key'),
                                      'canonical': records['e_no_ageing'].get('candidate_canonical')},
                      'benchmark': _load(REPORT_V3)['instance']},
        'manuscript': {'clone': CLONE, 'head': head, 'main_tex_sha256': before[MAIN_TEX], 'locations': tex,
                       'quotes': quotes},
        'ageing': ageing, 'chemistry': chem, 'benchmark': bench, 'code_identity': ident, 'code_locations': code,
        'integrity': {'failures': FAILED, 'inputs_sha256': before, 'tracked_clean': tracked},
        'guards': {'own': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                   'pickle_counts': PICKLE_COUNTS},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, 'x', encoding='utf-8') as fh:
        json.dump(result, fh, indent=1, sort_keys=False)
        fh.write('\n')
    man = {'outputs': {OUT_JSON: _sha(OUT_JSON)}, 'inputs': after}
    with open(OUT_MAN, 'x', encoding='utf-8') as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
        fh.write('\n')
    _log(f'[W175] ageing: C3_midblock readback all_match pre/terminal '
         f'{ageing["e_c3_midblock"]["model_variant_readback_pre_run"]["all_match"]}/'
         f'{ageing["e_c3_midblock"]["model_variant_readback_terminal"]["all_match"]}; no_ageing node-7 terminal SoH '
         f'{ageing["e_no_ageing"]["terminal_soh"].get("7")}')
    _log(f'[W175] chemistry: xlsx hits {chem["xlsx"]["hits"]}; json hits '
         f'{ {k: v for k, v in chem["json"].items() if v} }; git grep (outside Results) '
         f'{len(chem["git_grep"]["outside_results"]["hits"])} lines')
    for arm, v in bench['per_arm'].items():
        _log(f'[W175] benchmark {arm}: best {v["best_start"]} {v["q_best"]!r}; band {v["band"]!r}; sources '
             f'{v["sources"]}; all triggered {v["all_triggered"]}')
    _log(f'[W175] common-Q gate {bench["common_q_gate"]["status"]}; passive DSO objective without the rule '
         f'[{bench["passive_with_vs_without_rule"]["unconstrained_min"]!r}, '
         f'{bench["passive_with_vs_without_rule"]["unconstrained_max"]!r}], blocks > 1 EUR with the rule: '
         f'{[r["block"] for r in bench["passive_with_vs_without_rule"]["blocks_with_nrf_objective_above_1_eur"]]}')
    _log(f'[W175] guards: own {GUARD.counts}; pickle {PICKLE_COUNTS}; integrity failures {len(FAILED)}')
    for f in FAILED:
        _log(f'[W175 INTEGRITY FAILURE] {f}')
    sys.exit(3 if FAILED else 0)


if __name__ == '__main__':
    main()
