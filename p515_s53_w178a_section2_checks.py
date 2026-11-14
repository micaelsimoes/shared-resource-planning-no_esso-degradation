"""
P5.15 Addendum 77 order, Planner task W178a -- static checks behind the audit of sections 2.1-2.3 as rewritten after
Overleaf e10c104, against the code (P5_15_W178A_SECTION2.md). ZERO SOLVES, NO MODEL, NO PICKLE, NO PRODUCTION IMPORT.

What it does (static reading only):
  A. Manuscript identity: the Overleaf clone's HEAD (declared 9fd19e7), main.tex sha256 (declared, full hash), line
     count (2,511), main.tex unmodified in the clone, byte identity with manuscript_review/main_820ee1b_fixed.tex.
  B. Changed lines: `git diff -U0 e10c104 9fd19e7 -- main.tex` in the clone; the new-side line ranges that fall in
     section 2 (from \\section{Shared ESS Planning Framework} to the CASE STUDY banner) and in Appendix A.2
     (\\subsection Local Problems to the next \\subsection).
  C. Static equation numbering of the whole document (comment-stripped): equation (one number), subequations (one
     parent number, rows lettered), align rows (split at top-level \\\\, \\nonumber/\\notag honoured), equation*/align*
     unnumbered, aligned not numbered; after \\appendix the counter resets at each \\section and is printed A.n
     (elsarticle.dtx l. 1459-1469). Every numbered equation of section 2 with its line range, label, number and
     whether any of its lines is a changed line (B).
  D. Labels and references: every \\label (duplicates); every \\ref / \\eqref target in section 2 and A.2 and in the
     whole document (undefined); the two \\eqref ranges of the A.2 constraint list resolved to the label sequence they
     span.
  E. Code identity: `git diff --stat <git_head_at_run> HEAD -- <20 production files + SRP1 case data>` for every
     campaign_results.json behind the tables (v6, extension, A64, 3 x 3), one subprocess per head, argument list (no
     shell); positive control 353e094b must report changes (as W174b).
  F. Code anchors: exact-text search of each code location the report cites; line numbers at HEAD (every anchor must
     be found).
  G. Configuration values read as text (no import): constants in definitions.py and model_construction_helpers.py, the
     four network params files (shared_ess_model, slacks.shared_ess.day_balance), SRP1_ESS_Params.json, the SRP1 and
     3 x 3 case files (Years, DiscountFactor), the four frozen specs (sha256 prefix check).
  H. Active-cohort sets: equation (7) of the manuscript (y_inv <= y, y - y_inv < T^Cal, calendar years) against the
     code's window (shared_energy_storage_data.py l. 564-566: block index y in [y_inv, min(y_inv + round(T^Cal /
     Y_{y_inv}), |Y|)) ), both instances. The code window is transcribed from those lines (arithmetic only; the
     production function builds a Pyomo model and is not called here).

Guards: `SolveProfileGuard(permitted=())` installed before any project import and verified at exactly 0 at the end;
`pickle.load` / `pickle.loads` blocked for the whole run and counted (0 expected). No production module is imported
(asserted at exit). Refuses to run unless the output directory exists and holds nothing but launch.log.
`--out-dir DIR`: trial runs only (same refusal rule).

Output: data/SRP1/Results/P515S53/w178a_section2/w178a_checks.json and manifest_sha256.json.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w178a_section2 && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w178a_section2_checks.py \\
      > data/SRP1/Results/P515S53/w178a_section2/launch.log 2>&1
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W178a section-2 checks (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W178a: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W178a: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w178a_section2')
OUT_JSON = os.path.join(OUT_DIR, 'w178a_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
REVIEW_COPY = os.path.join('manuscript_review', 'main_820ee1b_fixed.tex')
DECLARED_CLONE_HEAD = '9fd19e7'
DECLARED_MAIN_SHA256 = 'd18384d669b7a4fbbcfb95e7ead5181e6ffdc5001f51f60d773053e1473df15d'
DECLARED_MAIN_LINES = 2511
DIFF_BASE = 'e10c104'

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
CAMPAIGN_ROOTS = (os.path.join(P53, 'w142_resettle_v6'), os.path.join(P53, 'w142_resettle_ext_v6'),
                  os.path.join(P53, 'w155_a64_cells'), os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair'))
FROZEN_SPECS = {'v6': (os.path.join(P53, 'w142_resettle_v6', 'frozen_s53_resettle_spec_v6_96c23404.json'), '96c23404'),
                'ext_v3': (os.path.join(P53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json'),
                           '84775dc4'),
                'a64': (os.path.join(P53, 'w155_a64_cells', 'frozen_s53_a64_cells_spec_v1_44a2dce8.json'), '44a2dce8'),
                '3x3': (os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair',
                                     'campaign_spec_s53_w91_3x3_pair_231558f0.json'), '231558f0')}
PRODUCTION_FILES = ('admm_anderson_acceleration.py', 'admm_parameters.py', 'admm_persistent_workers.py', 'branch.py',
                    'definitions.py', 'energy_storage.py', 'generator.py', 'helper_functions.py', 'load.py',
                    'model_construction_helpers.py', 'network.py', 'network_data.py', 'network_parameters.py',
                    'node.py', 'planning_parameters.py', 'shared_energy_storage.py', 'shared_energy_storage_data.py',
                    'shared_energy_storage_parameters.py', 'shared_resources_planning.py', 'solver_parameters.py')
CASE_DATA = (os.path.join('data', 'SRP1', 'SRP1.json'), os.path.join('data', 'SRP1', 'SRP1_params.json'),
             os.path.join('data', 'SRP1', 'SharedESS'), os.path.join('data', 'SRP1', 'case9'),
             os.path.join('data', 'SRP1', 'case33_1'), os.path.join('data', 'SRP1', 'case33_2'),
             os.path.join('data', 'SRP1', 'case33_3'), os.path.join('data', 'SRP1', 'MarketData'))
CONTROL_HEAD = '353e094b'
SRP1_CASE = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_3X3 = os.path.join(P53, 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
NETWORK_PARAMS = tuple(os.path.join('data', 'SRP1', n, f'{n}_params.json')
                       for n in ('case9', 'case33_1', 'case33_2', 'case33_3'))
FORBIDDEN_MODULE_PREFIXES = ('shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

SECTION2_START = r'\section{Shared ESS Planning Framework}'
SECTION2_END = 'CASE STUDY'                 # the banner comment that opens section 3
A2_START = r'\subsection{\textcolor{blue}{Local Problems}}'
A2_END = r'\subsection{\textcolor{blue}{Consensus, Dual and Penalty Updates}}'
A2_RANGES = (('eq:cohort_rated', 'eq:available_capacity'), ('eq:esso_net_power', 'eq:soh_floor'))

CONSTANTS = (('definitions.py', 'ENERGY_STORAGE_MAX_ENERGY_STORED'), ('definitions.py', 'ENERGY_STORAGE_MIN_ENERGY_STORED'),
             ('definitions.py', 'ENERGY_STORAGE_RELATIVE_INIT_SOC'), ('definitions.py', 'EQUALITY_TOLERANCE'),
             ('definitions.py', 'SMALL_TOLERANCE'), ('definitions.py', 'ESS_COMPLEMENTARITY_TOLERANCE'),
             ('definitions.py', 'PENALTY_SHARED_ESS_BALANCE'), ('definitions.py', 'PENALTY_ESSO_SLACK'),
             ('definitions.py', 'EPS_ESSO_THROUGHPUT'), ('definitions.py', 'HOURS_PER_REPRESENTATIVE_DAY'),
             ('definitions.py', 'ESS_MODEL_BILINEAR_RELAXATION'),
             ('model_construction_helpers.py', 'ESS_DAY_BALANCE_SLACK_FRACTION'))

ANCHORS = (
    # network storage block (section 2.3.3)
    ('na_scenario', 'model_construction_helpers.py', 'def sess_na_scenario(m):'),
    ('pnet_rule', 'model_construction_helpers.py',
     'return m.shared_es_pnet[e, s_m0, s_o0, p] == m.shared_es_pch[e, s_m0, s_o0, p] - m.shared_es_pdch[e, s_m0, s_o0, p]'),
    ('soc_prev_initial', 'model_construction_helpers.py',
     'soc_prev = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC'),
    ('soc_delta_ch', 'model_construction_helpers.py', 'delta = (eff_ch * m.shared_es_pch[e, s_m0, s_o0, p] * dt'),
    ('soc_delta_dch', 'model_construction_helpers.py', '- m.shared_es_pdch[e, s_m0, s_o0, p] * dt / eff_dch)'),
    ('soc_recursion_row', 'model_construction_helpers.py', 'return m.shared_es_soc[e, s_m0, s_o0, p] == soc_prev + delta'),
    ('dt', 'model_construction_helpers.py', 'return HOURS_PER_REPRESENTATIVE_DAY / n_periods'),
    ('soc_min', 'model_construction_helpers.py', 'soc_min = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MIN_ENERGY_STORED'),
    ('soc_max', 'model_construction_helpers.py', 'soc_max = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_MAX_ENERGY_STORED'),
    ('final_soc', 'model_construction_helpers.py',
     'final_soc = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC'),
    ('closure_row', 'model_construction_helpers.py',
     'return m.shared_es_soc[e, s_m0, s_o0, final_p] == final_soc + m.slack_shared_es_soc_final_up[e, s_m0, s_o0] '
     '- m.slack_shared_es_soc_final_down[e, s_m0, s_o0]'),
    ('closure_slack_fraction', 'model_construction_helpers.py', 'ESS_DAY_BALANCE_SLACK_FRACTION = 0.05'),
    ('closure_slack_ub', 'model_construction_helpers.py',
     'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE'),
    ('closure_penalty', 'model_construction_helpers.py',
     'total += base * PENALTY_SHARED_ESS_BALANCE * (model.slack_shared_es_soc_final_up[e, s_m0, s_o0] + '
     'model.slack_shared_es_soc_final_down[e, s_m0, s_o0])'),
    ('active_sum_rule', 'model_construction_helpers.py', 'def sess_active_sum_limit_rule(m, e, s_m, s_o, p):'),
    ('active_sum_row', 'model_construction_helpers.py',
     '+ m.shared_es_pdch[e, s_m0, s_o0, p]) <= m.shared_es_s_rated_fixed[e]'),
    ('circle_rule', 'model_construction_helpers.py', 'def sess_converter_capability_rule(m, e, s_m, s_o, p):'),
    ('circle_row', 'model_construction_helpers.py',
     '+ m.shared_es_qnet[e, s_m0, s_o0, p] ** 2) <= m.shared_es_s_rated_fixed[e] ** 2'),
    ('pch_hat_link', 'model_construction_helpers.py',
     '- m.shared_es_s_rated_fixed[e] * m.shared_es_pch_hat[e, s_m0, s_o0, p]) == 0'),
    ('pdch_hat_link', 'model_construction_helpers.py',
     '- m.shared_es_s_rated_fixed[e] * m.shared_es_pdch_hat[e, s_m0, s_o0, p]) == 0'),
    ('comp_row', 'model_construction_helpers.py',
     '* m.shared_es_pdch_hat[e, s_m0, s_o0, p]) <= ESS_COMPLEMENTARITY_TOLERANCE'),
    ('set_s_capacity', 'model_construction_helpers.py', 'model.shared_es_s_rated_fixed[shared_ess_idx].set_value(s_capacity)'),
    ('set_e_capacity', 'model_construction_helpers.py', 'model.shared_es_e_rated_fixed[shared_ess_idx].set_value(e_capacity)'),
    ('node_balance_pnet', 'model_construction_helpers.py', 'Pd += model.shared_es_pnet[e, s_m0, s_o0, p]'),
    ('usage_penalty_term', 'model_construction_helpers.py',
     'cost += model.penalty_shared_ess_usage * network.baseMVA * (model.shared_es_pch[e, s_m0, s_o0, p] + '
     'model.shared_es_pdch[e, s_m0, s_o0, p])'),
    ('pch_var', 'network.py',
     'model.shared_es_pch = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, '
     'model.periods, domain=pe.NonNegativeReals, initialize=0.0)'),
    ('pch_hat_var', 'network.py',
     'model.shared_es_pch_hat = pe.Var(model.shared_energy_storages, model.scenarios_market, model.scenarios_operation, '
     'model.periods, domain=pe.NonNegativeReals, bounds=(0.0, 1.0), initialize=0.0)'),
    ('wire_pnet', 'network.py', 'model.sess_pnet_def = pe.Constraint('),
    ('wire_circle', 'network.py', 'model.sess_converter_capability = pe.Constraint('),
    ('wire_active_sum', 'network.py', 'model.sess_active_sum_limit = pe.Constraint('),
    ('wire_soc_def', 'network.py', 'model.sess_soc_def = pe.Constraint('),
    ('wire_soc_upper', 'network.py', 'model.sess_soc_limit_upper = pe.Constraint('),
    ('wire_soc_lower', 'network.py', 'model.sess_soc_limit_lower = pe.Constraint('),
    ('wire_soc_final', 'network.py', 'model.sess_soc_final = pe.Constraint('),
    ('wire_comp', 'network.py', 'model.sess_comp = pe.Constraint('),
    # agent: cohort and available-capacity rows (section 2.3.2) and the A.2 list
    ('tcal_norm', 'shared_energy_storage_data.py',
     'tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y_inv]]))'),
    ('tcal_window', 'shared_energy_storage_data.py', 'max_tcal_norm = min(y_inv + tcal_norm, len(shared_ess_data.years))'),
    ('rated_unit_fixed_zero', 'shared_energy_storage_data.py', 'model.es_s_rated_per_unit.fix(0.00)'),
    ('rated_s_unit_row', 'shared_energy_storage_data.py',
     'model.rated_s_capacity_unit.add(model.es_s_rated_per_unit[y_inv, y] == model.es_s_investment_fixed[y_inv])'),
    ('rated_e_unit_row', 'shared_energy_storage_data.py',
     'model.rated_e_capacity_unit.add(model.es_e_rated_per_unit[y_inv, y] == model.es_e_investment_fixed[y_inv])'),
    ('rated_s_agg_row', 'shared_energy_storage_data.py', 'model.rated_s_capacity.add(model.es_s_rated[y] == total_s_capacity)'),
    ('rated_e_agg_row', 'shared_energy_storage_data.py', 'model.rated_e_capacity.add(model.es_e_rated[y] == total_e_capacity)'),
    ('available_s_unit_row', 'shared_energy_storage_data.py',
     'model.available_s_capacity_unit.add(model.es_s_available_per_unit[y_inv, y] == model.es_s_rated_per_unit[y_inv, y])'),
    ('available_e_unit_row_end', 'shared_energy_storage_data.py',
     'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] '
     '* model.es_soh_per_unit_cumul[y_inv, y])'),
    ('available_e_unit_row_mid', 'shared_energy_storage_data.py',
     'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] '
     '* soh_mid)'),
    ('soh_point_default', 'shared_energy_storage_data.py', "AVAILABLE_ENERGY_SOH_POINT_DEFAULT = 'end'"),
    ('published_s_available', 'shared_energy_storage_data.py',
     's_available += pe.value(model.es_s_available_per_unit[y_inv, year_idx])'),
    ('published_e_available', 'shared_energy_storage_data.py',
     'e_available += pe.value(model.es_e_available_per_unit[y_inv, year_idx])'),
    ('cohort_pch_var', 'shared_energy_storage_data.py',
     'model.es_pch_per_unit = pe.Var(model.years, model.years, model.days, model.periods, domain=pe.NonNegativeReals, '
     'initialize=0.00)'),
    ('cell_throughput', 'shared_energy_storage_data.py',
     'avg_ch_dch += (num_days / 365.00) * (eff_ch * pch * dt + pdch * dt / eff_dch)'),
    ('cycling_loss', 'shared_energy_storage_data.py', '== 365.00 * num_years * model.es_avg_ch_dch_per_unit[y_inv, y])'),
    ('soh_chain', 'shared_energy_storage_data.py',
     '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))'),
    ('soh_floor', 'shared_energy_storage_data.py', 'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'),
    ('cohort_power_bounds', 'shared_energy_storage_data.py',
     "_add_esso_cohort_constraint(model, 'energy_storage_limits', y_inv, y, pch <= s_max)"),
    ('esso_net_power', 'shared_energy_storage_data.py',
     'model.energy_storage_operation_agg.add(model.es_pnet[y, d, p] == agg_pnet + model.slack_es_pnet_up[y, d, p] - '
     'model.slack_es_pnet_down[y, d, p])'),
    ('esso_circle', 'shared_energy_storage_data.py',
     'model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2)'),
    ('cohort_allocation', 'shared_energy_storage_data.py', '== model.es_pnet_cohort_share_h3[y_inv, y] * agg_pnet)'),
    ('esso_throughput_reg', 'shared_energy_storage_data.py', 'slack_penalty += EPS_ESSO_THROUGHPUT * throughput'),
    # publication of S^Av / E^Av to the network blocks
    ('publish_in_loop', 'shared_resources_planning.py',
     'sess_available_capacities = shared_ess_data.get_updated_capacities(esso_model)'),
    ('publish_set_s', 'shared_resources_planning.py',
     "model[year][day].shared_es_s_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['s_available'] "
     "/ s_base)"),
    ('publish_set_e', 'shared_resources_planning.py',
     "model[year][day].shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['e_available'] "
     "/ s_base)"),
    ('usage_penalty_zeroed', 'shared_resources_planning.py', 'penalty_shared_ess_usage.set_value(0.00)'),
    # section 2.2 (master constraints, Q(x), weights) and section 2.1 (search stop rules)
    ('block_weight_def', 'shared_resources_planning.py', 'def _get_admm_block_weight(network_data, year, day):'),
    ('block_weight', 'shared_resources_planning.py',
     'return float(network_data.years[year]) * float(network_data.days[day]) * annualization'),
    ('gross_cost', 'shared_resources_planning.py', 'gross_operational_cost = (gross_operational_cost_including_settlement'),
    ('settlement_excluded', 'shared_resources_planning.py', '- interface_settlement_contracted_total'),
    ('voltage_pin_excluded', 'shared_resources_planning.py', '- voltage_pin_total)'),
    ('res_curtailment_zero', 'shared_resources_planning.py', 'penalty_gen_curtailment.set_value(0.00)'),
    ('discount_from_case', 'shared_resources_planning.py', 'shared_ess_data.discount_factor = planning_problem.discount_factor'),
    ('expected_price', 'shared_resources_planning.py', 'def _expected_market_price(model, network, p):'),
    ('first_stage_ratio_min', 'shared_resources_planning.py',
     'if e_value + tolerance < params.min_energy_to_power_ratio * s_value:'),
    ('first_stage_ratio_max', 'shared_resources_planning.py',
     'if e_value > params.max_energy_to_power_ratio * s_value + tolerance:'),
    ('first_stage_emax', 'shared_resources_planning.py', "if total_capacity['e'] > params.max_capacity + tolerance:"),
    ('first_stage_budget', 'shared_resources_planning.py', 'if expected_investment_cost > params.budget + tolerance:'),
    ('first_stage_discount', 'shared_resources_planning.py',
     '(1.00 + shared_ess_data.discount_factor) ** (int(year) - int(years[0]))'),
    ('harness_certified_cost', 'p515_s44_campaign_harness.py',
     "'certified_cost': report.get('gross_operational_cost') if certified else None,"),
    ('t_sum_term', 'p515_s53_w118_resettle_hooks.py', 'term = w * pi * (xd[p] - zt[p])'),
    ('search_granule_p', 'p515_s47_phase_b_record.py', 'GRANULE_P_MVA = 0.25'),
    ('search_granule_e', 'p515_s47_phase_b_record.py', 'GRANULE_E_MWH = 0.5'),
    ('search_emax', 'p515_s47_phase_b_record.py', 'if ze > ZE_MAX:'),
    ('search_duration', 'p515_s47_phase_b_record.py', 'if zp > 0 and not (zp <= ze <= 2 * zp):'),
    ('search_budget', 'p515_s47_phase_b_record.py', 'if not out and self.investment_cost(z) > self.budget:'),
    ('search_F', 'p515_s47_phase_b_record.py', "f_x = c['I_x_eur'] + rec['Q']"),
    ('search_cap_value', 'p515_s47_phase_b_record.py', 'COMPLETION_CAP = 30'),
    ('search_cap_stop', 'p515_s47_phase_b_record.py', "termination = {'reason': 'STOP_FOR_REVIEW_completion_cap'"),
    ('search_budget_stop', 'p515_s47_phase_b_record.py', "termination = {'reason': 'evaluation_budget_exhausted'"),
    ('search_budget_test', 'p515_s47_phase_b_record.py', 'if n_new + len(new) > max_new_evaluations:'),
    ('search_barrier_stop', 'p515_s47_phase_b_record.py',
     "if barrier_this_poll >= BARRIER_STOP_PER_POLL or n_barrier_new >= BARRIER_STOP_OVERALL:"),
    ('s53_budget', 'p515_s53_f2_certificate.py', 'MAX_NEW_EVALUATIONS = 60'),
    ('s53_cap', 'p515_s53_f2_certificate.py', 'COMPLETION_CAP = PB.COMPLETION_CAP'),
)


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


INPUTS = set()


def _read(path):
    INPUTS.add(path)
    with open(path, encoding='utf-8') as f:
        return f.read()


def _read_json(path):
    return json.loads(_read(path))


def _git(args, cwd=REPO):
    return subprocess.run(['git'] + list(args), cwd=cwd, check=True, capture_output=True, text=True).stdout


def _walk(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            else:
                yield from _walk(v, key)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)


def _line_of(text, needle):
    """1-based line numbers of every line containing needle (exact)."""
    return [i for i, line in enumerate(text.splitlines(), start=1) if needle in line]


# ---------------------------------------------------------------------------------------------------------------------
#  A. manuscript identity
# ---------------------------------------------------------------------------------------------------------------------
def check_manuscript(tex):
    head = _git(['rev-parse', '--short=7', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', 'main.tex'], cwd=CLONE)
    sha = _sha(MAIN_TEX)
    n_lines = len(tex.splitlines())
    review_sha = _sha(REVIEW_COPY)
    INPUTS.add(REVIEW_COPY)
    out = {'clone_head': head, 'main_tex_sha256': sha, 'main_tex_lines': n_lines,
           'main_tex_unmodified_in_clone': status.strip() == '',
           'review_copy': REVIEW_COPY, 'review_copy_sha256': review_sha, 'byte_identical_to_review_copy': sha == review_sha}
    if head != DECLARED_CLONE_HEAD or sha != DECLARED_MAIN_SHA256 or n_lines != DECLARED_MAIN_LINES \
            or not out['main_tex_unmodified_in_clone'] or not out['byte_identical_to_review_copy']:
        raise SystemExit(f'manuscript identity failed: {out}')
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  B. changed lines (e10c104 -> 9fd19e7)
# ---------------------------------------------------------------------------------------------------------------------
def scope_lines(tex):
    lines = tex.splitlines()

    def first(needle, start=0):
        for i in range(start, len(lines)):
            if needle in lines[i]:
                return i + 1
        raise SystemExit(f'scope marker not found: {needle!r}')
    s2a = first(SECTION2_START)
    s2b = first(SECTION2_END, s2a) - 2          # the banner's opening rule line is one above
    a2a = first(A2_START)
    a2b = first(A2_END, a2a) - 1
    return {'section2': [s2a, s2b], 'appendix_A2': [a2a, a2b]}


def changed_lines(scope):
    diff = _git(['diff', '-U0', DIFF_BASE, DECLARED_CLONE_HEAD, '--', 'main.tex'], cwd=CLONE)
    new_lines, hunks = set(), []
    for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', diff, flags=re.M):
        start, count = int(m.group(3)), int(m.group(4)) if m.group(4) is not None else 1
        old_start, old_count = int(m.group(1)), int(m.group(2)) if m.group(2) is not None else 1
        rng = list(range(start, start + count))
        new_lines.update(rng)
        hunks.append({'old': [old_start, old_count], 'new': [start, count]})
    out = {'base': DIFF_BASE, 'head': DECLARED_CLONE_HEAD, 'n_hunks_whole_file': len(hunks), 'by_scope': {}}
    for name, (a, b) in scope.items():
        hs = [h for h in hunks if (h['new'][1] > 0 and h['new'][0] <= b and h['new'][0] + h['new'][1] - 1 >= a)
              or (h['new'][1] == 0 and a <= h['new'][0] <= b)]
        out['by_scope'][name] = {'lines': [a, b], 'n_hunks': len(hs), 'hunks': hs,
                                 'n_changed_new_lines': sum(1 for x in new_lines if a <= x <= b)}
    return out, new_lines


# ---------------------------------------------------------------------------------------------------------------------
#  C. static equation numbering
# ---------------------------------------------------------------------------------------------------------------------
def strip_comments(tex):
    out = []
    for line in tex.splitlines():
        m = re.search(r'(?<!\\)%', line)
        out.append(line[:m.start()] if m else line)
    return out


TOKEN = re.compile(r'\\begin\{(equation\*?|subequations|align\*?|aligned|gather\*?|multline\*?)\}'
                   r'|\\end\{(equation\*?|subequations|align\*?|aligned|gather\*?|multline\*?)\}'
                   r'|\\label\{([^}]*)\}|\\appendix\b|\\section\*?\{|\\nonumber\b|\\notag\b|\\\\')


def number_equations(lines):
    """Walk the comment-stripped lines; return (equations, labels). equations: list of dicts with start/end line,
    number, labels; labels: {label: {'line', 'number' or None, 'kind'}} (kind 'equation' for equation labels,
    'other' otherwise; duplicates kept in a separate list)."""
    eq_counter, appendix, app_section = 0, False, 0
    stack = []          # env frames: {'env', 'start', 'number', 'row', 'row_tagged', 'labels', 'sub_letter'}
    equations, label_list = [], []
    in_doc = False
    for ln, line in enumerate(lines, start=1):
        if r'\begin{document}' in line:
            in_doc = True
        if not in_doc:
            for m in re.finditer(r'\\label\{([^}]*)\}', line):
                label_list.append({'label': m.group(1), 'line': ln, 'number': None, 'kind': 'preamble'})
            continue
        for m in TOKEN.finditer(line):
            tok = m.group(0)
            if m.group(1):
                env = m.group(1)
                frame = {'env': env, 'start': ln, 'labels': [], 'numbers': []}
                if env == 'equation' or env == 'gather' or env == 'multline':
                    sub = next((f for f in reversed(stack) if f['env'] == 'subequations'), None)
                    if sub is not None:
                        sub['letter'] += 1
                        frame['number'] = f"{sub['number']}{chr(96 + sub['letter'])}"
                    else:
                        eq_counter += 1
                        frame['number'] = f"{'%s.' % chr(64 + app_section) if appendix else ''}{eq_counter}"
                elif env == 'subequations':
                    eq_counter += 1
                    frame['number'] = f"{'%s.' % chr(64 + app_section) if appendix else ''}{eq_counter}"
                    frame['letter'] = 0
                elif env == 'align':
                    frame['row_open'] = True
                    frame['row_tagged'] = True
                    frame['row_labels'] = []
                stack.append(frame)
            elif m.group(2):
                env = m.group(2)
                if not stack or stack[-1]['env'] != env:
                    raise SystemExit(f'environment mismatch at line {ln}: end {env}, stack {[f["env"] for f in stack]}')
                frame = stack.pop()
                if env == 'align':
                    _close_align_row(frame, stack, ln, lambda: None)
                if env in ('equation', 'subequations', 'align', 'gather', 'multline'):
                    rec = {'env': env, 'start': frame['start'], 'end': ln,
                           'number': frame.get('number'), 'labels': frame['labels'],
                           'row_numbers': frame.get('numbers', [])}
                    equations.append(rec)
            elif m.group(3) is not None:
                label = m.group(3)
                frame = next((f for f in reversed(stack) if f['env'] in
                              ('equation', 'subequations', 'align', 'gather', 'multline', 'equation*', 'align*')), None)
                if frame is None:
                    label_list.append({'label': label, 'line': ln, 'number': None, 'kind': 'other'})
                elif frame['env'] in ('equation*', 'align*'):
                    label_list.append({'label': label, 'line': ln, 'number': None, 'kind': 'unnumbered_env'})
                elif frame['env'] == 'align':
                    frame['row_labels'].append((label, ln))
                else:
                    frame['labels'].append(label)
                    label_list.append({'label': label, 'line': ln, 'number': frame['number'], 'kind': 'equation'})
            elif tok == r'\appendix':
                appendix, app_section = True, 0
                eq_counter = 0
            elif tok.startswith(r'\section'):
                if appendix and not tok.startswith(r'\section*'):
                    app_section += 1
                    eq_counter = 0
            elif tok in (r'\nonumber', r'\notag'):
                fr = next((f for f in reversed(stack) if f['env'] == 'align'), None)
                if fr is not None:
                    fr['row_tagged'] = False
            elif tok == '\\\\':
                if stack and stack[-1]['env'] == 'align':
                    _close_align_row(stack[-1], stack[:-1], ln, None)

    # attach align row labels
    return equations, label_list


ALIGN_ROW_LABELS = []


def _close_align_row(frame, outer, ln, _cb):
    if frame['row_tagged']:
        sub = next((f for f in reversed(outer) if f['env'] == 'subequations'), None)
        if sub is not None:
            sub['letter'] += 1
            num = f"{sub['number']}{chr(96 + sub['letter'])}"
        else:
            raise SystemExit(f'align outside subequations at line {ln}: not handled by this static numberer')
        frame['numbers'].append(num)
        for label, lline in frame['row_labels']:
            ALIGN_ROW_LABELS.append({'label': label, 'line': lline, 'number': num, 'kind': 'equation'})
    frame['row_tagged'] = True
    frame['row_labels'] = []


# ---------------------------------------------------------------------------------------------------------------------
#  D. labels and references
# ---------------------------------------------------------------------------------------------------------------------
REF = re.compile(r'\\(eqref|ref|autoref|cref|Cref)\{([^}]*)\}')


def check_labels_refs(lines, label_list, scope):
    all_labels = label_list + ALIGN_ROW_LABELS
    by_label = {}
    for rec in all_labels:
        by_label.setdefault(rec['label'], []).append(rec)
    duplicates = {k: v for k, v in by_label.items() if len(v) > 1}
    refs = []
    for ln, line in enumerate(lines, start=1):
        for m in REF.finditer(line):
            for target in m.group(2).split(','):
                refs.append({'cmd': m.group(1), 'target': target.strip(), 'line': ln})
    undefined_all = [r for r in refs if r['target'] not in by_label]
    out = {'n_labels': len(all_labels), 'n_distinct_labels': len(by_label), 'duplicate_labels': duplicates,
           'n_refs_whole_document': len(refs), 'undefined_refs_whole_document': undefined_all, 'by_scope': {}}
    for name, (a, b) in scope.items():
        rs = [r for r in refs if a <= r['line'] <= b]
        labels_in = [rec for rec in all_labels if a <= rec['line'] <= b]
        out['by_scope'][name] = {
            'n_refs': len(rs), 'refs': [dict(r, number=(by_label.get(r['target']) or [{}])[0].get('number'),
                                             defined=r['target'] in by_label) for r in rs],
            'undefined': [r for r in rs if r['target'] not in by_label],
            'labels_defined_here': [rec['label'] for rec in labels_in],
            'duplicate_labels_defined_here': sorted({rec['label'] for rec in labels_in if rec['label'] in duplicates})}
    return out, by_label


def resolve_ranges(equations):
    """The label sequence of numbered body equations; each A.2 range resolved to the labels it spans."""
    seq = [(e['number'], e['labels'], e['start']) for e in equations if e['number'] is not None]
    seq.sort(key=lambda r: r[2])
    order = []
    for number, labels, start in seq:
        order.append({'number': number, 'labels': labels, 'line': start})
    out = []
    for a, b in A2_RANGES:
        ia = next((i for i, r in enumerate(order) if a in r['labels']), None)
        ib = next((i for i, r in enumerate(order) if b in r['labels']), None)
        if ia is None or ib is None or ib < ia:
            out.append({'range': [a, b], 'resolved': False})
            continue
        span = order[ia:ib + 1]
        out.append({'range': [a, b], 'resolved': True, 'numbers': [r['number'] for r in span],
                    'labels': [r['labels'] for r in span], 'lines': [r['line'] for r in span]})
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  E. code identity
# ---------------------------------------------------------------------------------------------------------------------
def check_code_identity():
    results_files = []
    for root in CAMPAIGN_ROOTS:
        results_files += sorted(glob.glob(os.path.join(root, '**', 'campaign_results.json'), recursive=True))
    heads = {}
    for path in results_files:
        for h in set(_walk(_read_json(path), 'git_head_at_run')):
            heads.setdefault(h, []).append(path)
    pathspec = list(PRODUCTION_FILES) + list(CASE_DATA)
    per_head = {}
    for h in sorted(heads):
        stat = _git(['diff', '--stat', h, 'HEAD', '--'] + pathspec).strip()
        per_head[h] = {'campaign_results': heads[h], 'diff_stat': stat or 'NO CHANGE'}
    control = _git(['diff', '--stat', CONTROL_HEAD, 'HEAD', '--'] + pathspec).strip().splitlines()
    out = {'repo_head': _git(['rev-parse', 'HEAD']).strip(), 'n_campaign_results': len(results_files),
           'n_distinct_heads': len(heads),
           'n_heads_with_change': sum(1 for v in per_head.values() if v['diff_stat'] != 'NO CHANGE'),
           'pathspec': pathspec, 'per_head': per_head,
           'control': {'head': CONTROL_HEAD, 'last_line': control[-1] if control else 'NO CHANGE'}}
    if out['n_heads_with_change'] or out['control']['last_line'] == 'NO CHANGE' or not results_files:
        raise SystemExit(f"code identity check failed: changed={out['n_heads_with_change']} control={out['control']}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  F. code anchors
# ---------------------------------------------------------------------------------------------------------------------
def check_anchors():
    out = {}
    for anchor_id, path, text in ANCHORS:
        hits = _line_of(_read(path), text)
        if not hits:
            raise SystemExit(f'anchor {anchor_id} not found in {path}: {text!r}')
        out[anchor_id] = {'file': path, 'text': text, 'lines': hits}
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  G. configuration values (text reads, no import)
# ---------------------------------------------------------------------------------------------------------------------
def check_configuration():
    consts = {}
    for path, name in CONSTANTS:
        src = _read(path)
        m = re.search(rf'^{name}\s*=\s*([^#\n]+)', src, flags=re.M)
        if not m:
            raise SystemExit(f'constant {name} not found in {path}')
        consts[name] = {'file': path, 'line': src[:m.start()].count('\n') + 1, 'text': m.group(1).strip()}
    networks = {}
    for p in NETWORK_PARAMS:
        d = _read_json(p)
        networks[p] = {'ess_model': d.get('ess_model'), 'shared_ess_model': d.get('shared_ess_model'),
                       'slacks_shared_ess': (d.get('slacks') or {}).get('shared_ess')}
    ess = _read_json(ESS_PARAMS)
    ess_out = {k: ess.get(k) for k in ('budget', 'max_capacity', 'min_energy_to_power_factor',
                                       'max_energy_to_power_factor', 'slacks')}
    ess_out['ageing'] = {k: (ess.get('ageing') or {}).get(k) for k in ('calendar_life_years', 'minimum_soh',
                                                                       'calendar_retention_per_year')}
    cases = {}
    for p in (SRP1_CASE, CASE_3X3):
        d = _read_json(p)
        cases[p] = {'Years': d.get('Years'), 'DiscountFactor': d.get('DiscountFactor'), 'NumInstants': d.get('NumInstants'),
                    'Days': d.get('Days'), 'NumMarketScenarios': d.get('NumMarketScenarios')}
    specs = {}
    for name, (p, prefix) in FROZEN_SPECS.items():
        sha = _sha(p)
        INPUTS.add(p)
        specs[name] = {'path': p, 'sha256': sha, 'prefix_matches': sha.startswith(prefix)}
        if not specs[name]['prefix_matches']:
            raise SystemExit(f'frozen spec {name} hash mismatch: {sha}')
    return {'constants': consts, 'network_params': networks, 'ess_params': ess_out, 'cases': cases,
            'frozen_specs': specs}


# ---------------------------------------------------------------------------------------------------------------------
#  H. active-cohort sets: manuscript eq. (7) vs the code window (shared_energy_storage_data.py l. 564-566)
# ---------------------------------------------------------------------------------------------------------------------
def check_active_sets(cases, t_cal):
    out = {}
    for path, c in cases.items():
        years = [int(y) for y in c['Years']]
        width = [int(c['Years'][str(y)]) for y in years]
        manuscript = sorted((i, j) for i in range(len(years)) for j in range(len(years))
                            if years[i] <= years[j] and years[j] - years[i] < t_cal)
        code = []
        for i in range(len(years)):
            tcal_norm = round(t_cal / width[i])                      # l. 564
            for j in range(i, min(i + tcal_norm, len(years))):       # l. 565-566
                code.append((i, j))
        code.sort()
        out[path] = {'years': years, 'block_widths': width, 't_cal': t_cal,
                     'manuscript_pairs': [list(p) for p in manuscript], 'code_pairs': [list(p) for p in code],
                     'identical': manuscript == code}
    return out


# ---------------------------------------------------------------------------------------------------------------------
def main():
    global OUT_DIR, OUT_JSON, OUT_MANIFEST
    if '--out-dir' in sys.argv:                       # trial runs only; the directory must hold nothing but launch.log
        OUT_DIR = sys.argv[sys.argv.index('--out-dir') + 1]
        OUT_JSON = os.path.join(OUT_DIR, 'w178a_checks.json')
        OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
    if not os.path.isdir(OUT_DIR):
        raise SystemExit(f'REFUSED: {OUT_DIR} does not exist (create it with mkdir; it must hold only launch.log)')
    extra = sorted(set(os.listdir(OUT_DIR)) - {'launch.log'})
    if extra:
        raise SystemExit(f'REFUSED: {OUT_DIR} already holds {extra}')
    started = datetime.now(timezone.utc).isoformat()
    tex = _read(MAIN_TEX)
    manuscript = check_manuscript(tex)
    scope = scope_lines(tex)
    changed, new_lines = changed_lines(scope)
    lines = strip_comments(tex)
    equations, label_list = number_equations(lines)
    labels_refs, by_label = check_labels_refs(lines, label_list, scope)
    s2a, s2b = scope['section2']
    s2_equations = []
    for e in equations:
        if s2a <= e['start'] <= s2b and e['number'] is not None:
            s2_equations.append(dict(e, changed_lines=sorted(x for x in new_lines if e['start'] <= x <= e['end']),
                                     changed=any(e['start'] <= x <= e['end'] for x in new_lines)))
    body_numbers = {lab: e['number'] for e in equations for lab in e['labels']}
    expert_numbers = {'eq:recourse_value': '13', 'eq:cohort_rated': '14', 'eq:cohort_rated_energy': '15',
                      'eq:storage_net_active_power': '20', 'eq:soc_limits': '22', 'eq:network_complementarity': '28',
                      'eq:esso_net_power': '29', 'eq:cycle_life_calibration': '35'}
    expert_check = {k: {'expert': v, 'static': body_numbers.get(k), 'match': body_numbers.get(k) == v}
                    for k, v in expert_numbers.items()}
    config = check_configuration()
    t_cal = int(config['ess_params']['ageing']['calendar_life_years'])
    result = {
        'schema': 'p515_s53_w178a_section2_checks_v1',
        'stage': 'P5.15 Addendum 77 W178a -- sections 2.1-2.3 as rewritten after Overleaf e10c104, against the code',
        'started_utc': started,
        'git_HEAD': _git(['rev-parse', 'HEAD']).strip(),
        'interpreter': sys.executable,
        'A_manuscript': manuscript,
        'B_scope_and_changed_lines': {'scope': scope, 'diff': changed},
        'C_equations_section2': s2_equations,
        'C_all_numbered_equations': [{'number': e['number'], 'labels': e['labels'], 'start': e['start'],
                                      'end': e['end'], 'env': e['env'], 'row_numbers': e['row_numbers']}
                                     for e in equations if e['number'] is not None],
        'C_expert_compile_numbers_vs_static': expert_check,
        'D_labels_refs': labels_refs,
        'D_A2_ranges': resolve_ranges(equations),
        'E_code_identity': check_code_identity(),
        'F_code_anchors': check_anchors(),
        'G_configuration': config,
        'H_active_cohort_sets': check_active_sets(config['cases'], t_cal),
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
                'inputs': {p: _sha(p) for p in sorted(INPUTS)}}
    with open(OUT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
        f.write('\n')
    print(f"W178a checks: manuscript {manuscript['clone_head']} sha256 {manuscript['main_tex_sha256'][:8]} "
          f"lines {manuscript['main_tex_lines']} identical-to-review-copy {manuscript['byte_identical_to_review_copy']}")
    print(f"scope {scope}; changed hunks: " + ', '.join(f"{k} {v['n_hunks']} ({v['n_changed_new_lines']} lines)"
                                                       for k, v in changed['by_scope'].items()))
    print('section 2 numbered equations (number, labels, lines, changed):')
    for e in s2_equations:
        print(f"  ({e['number']}) {e['labels']} l. {e['start']}-{e['end']} changed={e['changed']}")
    print(f"expert numbers vs static: {json.dumps(expert_check)}")
    print(f"labels {labels_refs['n_labels']} ({labels_refs['n_distinct_labels']} distinct); duplicates "
          f"{sorted(labels_refs['duplicate_labels'])}; undefined refs (whole document) "
          f"{[(r['target'], r['line']) for r in labels_refs['undefined_refs_whole_document']]}")
    for k, v in labels_refs['by_scope'].items():
        print(f"  {k}: {v['n_refs']} refs, undefined {v['undefined']}, duplicate labels defined here "
              f"{v['duplicate_labels_defined_here']}")
    for r in result['D_A2_ranges']:
        print(f"A.2 range {r['range']}: {r.get('numbers')} {r.get('labels')}")
    ci = result['E_code_identity']
    print(f"code identity: {ci['n_campaign_results']} campaign results, {ci['n_distinct_heads']} distinct heads, "
          f"{ci['n_heads_with_change']} changed; control {ci['control']}")
    print(f"anchors: {len(result['F_code_anchors'])} found")
    same = {k: v['identical'] for k, v in result['H_active_cohort_sets'].items()}
    print(f'active cohort sets identical: {same}')
    print(f"guards: {result['guards']}")
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')


if __name__ == '__main__':
    main()
