"""P5.15 Addendum 82 v2, Planner task W182e -- ITEMS Z1, Z6 AND C4 OF THE W182 ORDER: A CODE READING OF THE
NO-AGEING PATH, Q(0) AND THE ZERO-CAPACITY ROWS (Z1); THE PERTURBED START, rho_0 AND THE NO-REVERSE-FLOW ROW (Z6);
THE CONCURRENCY PATH (C4). ZERO SOLVES, NO MODEL BUILD, NO PRODUCTION IMPORT, READ ONLY.

WHAT IT DOES. Every claim of the W182e report is a CITATION: (file, line range at the repo HEAD, substrings that must
occur inside that range). The script reads each cited file AS TEXT from the working tree, refuses unless the file is
tracked, unmodified and byte-identical to its HEAD blob (git hash-object == git ls-files -s), asserts every required
substring inside the cited range (a citation that does not say what the report says it says is an INTEGRITY failure,
exit 3), and records the excerpt verbatim with its sha256 and the file's sha256 -- so each claim is preserved, not
paraphrased. It also reads committed JSON records (values recorded with the file's sha256), runs read-only git queries
(ancestry, last commit per file, numstat diff), and runs SCOPED text searches whose scope (file list) is recorded with
the result. Predictions (expert, Addendum 82 v2) are stated beside the outcome the citations support.

OBJECTIVE CONVENTION. Q = certified gross_operational_cost, settlement excluded; salvage reported, excluded.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked for the whole run (counters verified 0); no production module is imported (the only
project imports are the guard and the shared writer gate_result_io). git reads only. Refuses to run beside another copy
of itself. Every JSON output goes through gate_result_io (refuses string flags).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S53/w182e_code_paths && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182e_code_paths.py \\
        > data/SRP1/Results/P515S53/w182e_code_paths/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182e_code_paths.py --post-run
Write-once: every output is opened 'x' in a directory that holds nothing but its launch log.
Exit: 0 = written, every citation and integrity check holds and the guards are at 0 (failed predictions are FINDINGS
and never change the exit code); 3 = written, an integrity check failed (listed); 1 = precondition or guard fault
(nothing written).
"""
import argparse
import hashlib
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W182e code paths Z1/Z6/C4 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W182e: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W182e: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

pickle.load, pickle.loads = _blocked_load, _blocked_loads

SCRIPT_REL = os.path.basename(__file__)
S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(S53, 'w182e_code_paths')
OUT_NAMES = {'json': 'w182e_code_paths.json', 'md': 'w182e_code_paths.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'post': 'manifest_post_run_sha256.json'}
LAUNCH_LOG_RX = re.compile(r'launch(_attempt\d+)?\.log')
EXPECTED_HEAD_PREFIX = 'a17f6ba7'
LAST_PRODUCTION_COMMIT = '575bfec4'
T8_NO_AGEING_HEAD = '55bb1e7d'

SED = 'shared_energy_storage_data.py'
SEP = 'shared_energy_storage_parameters.py'
SRP = 'shared_resources_planning.py'
MCH = 'model_construction_helpers.py'
NET = 'network.py'
NETP = 'network_parameters.py'
ADMMP = 'admm_parameters.py'
APW = 'admm_persistent_workers.py'
DEFS = 'definitions.py'
UB = 'uncoordinated_benchmark.py'
H = 'p515_s44_campaign_harness.py'
G = 'p515_g_g1_g4_admm_gates.py'
ORA = 'p56a_oracle.py'
RHO = 'p59_rho.py'
PB = 'p515_s47_phase_b_record.py'
F2B = 'p515_s51_f2_phase_b.py'
S44G = 'p515_s44_gate.py'
W142L = 'p515_s53_w142_resettle_ext_v6_campaign.py'
BWG = 'p515_s53_srp1_bitwise_gate.py'
A46 = 'p515_s46_ageing_campaign.py'
A47 = 'p515_s47_baseline_campaign.py'
BRIEF = 'PLANNER_BRIEF_2026-09-13.md'

# ======================================================================================================================
#  THE CITATIONS: (id, item, file, first line, last line, substrings that must occur in the range, what it shows)
# ======================================================================================================================
CLAIMS = [
    # ---------------------------------------------------------------- Z1 (a) the no-ageing switch and its path
    ('Z1a-01', 'Z1a', SED, 58, 69,
     ['ageing_enabled: True (default); False -- D == 0 and phi_cal == 1', 'never by a case file',
      'self.ageing_enabled = True'],
     'No-ageing is an attribute of SharedEnergyStorageData (default True), changed only by a model-variant '
     'evaluation of the campaign harness, never by a case file.'),
    ('Z1a-02', 'Z1a', SED, 1529, 1563,
     ["`ageing_enabled` False: D_y == 0 (the D row becomes `D == 0`, so k is never",
      'The soh_min floor row is kept (trivially satisfied)',
      "ageing_enabled = getattr(shared_ess_data, 'ageing_enabled', True)"],
     'The production reader of the switch; documents D == 0, phi == 1, SoH == 1, floor row kept.'),
    ('Z1a-03', 'Z1a', SED, 676, 711,
     ['phi_cal = shared_energy_storage.phi_cal', 'if not ageing_enabled:', 'phi_cal = 1.00',
      '2 * shared_energy_storage.cl_eff * model.es_e_investment_fixed[y_inv]',
      'model.es_D_per_unit[y_inv, y] == 0.00)'],
     'Ageing off: the D row is D == 0 (k never consumed -- not a k -> infinity limit) and phi_cal is forced to 1.0 '
     'whatever calendar_retention_per_year says; ageing on: D * (2 k E_inv) == 365 n avg_ch_dch (product form).'),
    ('Z1a-04', 'Z1a', SED, 713, 732,
     ['prev_soh = 1.00', 'pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))',
      'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'],
     'SoH row SoH_y = SoH_{y-1} exp(-D_y) phi^n with SoH_{y_inv-1} = 1; the floor row SoH >= soh_min stays BUILT and '
     'ACTIVE under no-ageing (non-binding at SoH = 1), i.e. "floor inactive" holds as non-binding, not as deactivated.'),
    ('Z1a-05', 'Z1a', SED, 604, 628,
     ['soh_point, ageing_enabled = _esso_ageing_model_settings(shared_ess_data)',
      'phi_cal = shared_energy_storage.phi_cal if ageing_enabled else 1.00',
      'model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y])'],
     "Available energy E_av = E_rated * SoH ('end', the E1 setting); 'mid' uses the mid-block SoH with phi forced "
     "to 1.0 when ageing is off."),
    ('Z1a-06', 'Z1a', H, 336, 339, ["'model_variant'"],
     'model_variant is a per-entry evaluation option of the campaign harness.'),
    ('Z1a-07', 'Z1a', H, 366, 370, ["MODEL_VARIANT_KEYS = frozenset({'eol_retention_r', 'calendar_retention_per_year', "
                                    "'available_energy_soh_point',", "'ageing_enabled'})"],
     'Exactly four keys; eol_retention_r is REQUIRED even when ageing is off (it is then inert in the model).'),
    ('Z1a-08', 'Z1a', H, 646, 650,
     ["must lie in (0, 1), got", "must lie in (0, 1], got"],
     'eol_retention_r in (0,1) and calendar retention in (0,1]: k = N D/(-ln R) is always finite, so no case-file '
     'or variant value reaches k -> infinity; the only exact no-ageing route is ageing_enabled = False.'),
    ('Z1a-09', 'Z1a', H, 2302, 2346,
     ["needs an ACTIVE degradation calibration", "sed.ageing_enabled = model_variant['ageing_enabled']",
      'ageing.apply_to(ess)', "settings = SED._esso_ageing_model_settings(sed)"],
     'The variant is applied in the child to the deep-copied shared-ESS data BEFORE any ESSO model is built, through '
     'production\'s own apply_to and read back by production\'s own settings reader.'),
    ('Z1a-10', 'Z1a', H, 2998, 3010,
     ['if model_variant is not None:', 'applied_mv = apply_model_variant(sed, model_variant)'],
     'Where the child applies it: the configuration hook (run_admm_arm pre_solve_hook).'),
    ('Z1a-11', 'Z1a', H, 1126, 1133,
     ["payload['model_variant'] = model_variant"],
     'The model variant enters the evaluation key, so aged and no-ageing evaluations of one candidate never share a '
     'cache key (and two no-ageing declarations differing only in the inert eol_retention_r get different keys).'),
    ('Z1a-12', 'Z1a', PB, 874, 875, ["'no_model_variant': 'model_variant_label' not in spec and not any("],
     'The committed Phase-B search launcher (PB, imported by every later search) asserts NO model variant.'),
    ('Z1a-13', 'Z1a', F2B, 362, 363, ["'no_ageing_model_variant': ('model_variant_label' not in spec"],
     'The 2x (F2) Phase-B search launcher asserts NO ageing model variant on its cache sources ...'),
    ('Z1a-14', 'Z1a', F2B, 851, 852, ["'no_ageing_model_variant': ('model_variant_label' not in spec"],
     '... and on its own frozen spec: a no-ageing SEARCH is not runnable through the committed search launchers '
     'without a launcher change (harness-level; no production change).'),
    ('Z1a-15', 'Z1a', A46, 148, 154,
     ["('no_ageing', {'eol_retention_r': 0.50, 'calendar_retention_per_year': 1.0,",
      "'available_energy_soh_point': 'end', 'ageing_enabled': False}),"],
     'The original W20 no-ageing arm declaration (s46, C3-era): eol 0.50 is inherited from the C3 calibration.'),
    # ---------------------------------------------------------------- Z1 (b) Q(0) independence
    ('Z1b-01', 'Z1b', SED, 1566, 1570,
     ['model._esso_cohort_constraints[y_inv].append((constraint_name, constraint_idx, y))'],
     'Every row built through _add_esso_cohort_constraint is registered to its investment cohort.'),
    ('Z1b-02', 'Z1b', SED, 666, 666,
     ["_add_esso_cohort_constraint(model, 'energy_storage_charging_discharging', y_inv, y,"],
     'The throughput-definition row is cohort-registered.'),
    ('Z1b-03', 'Z1b', SED, 700, 732,
     ["model, 'energy_storage_capacity_degradation', y_inv, y,"],
     'The D row, the SoH row and the floor row (every ageing-parameter-bearing ESSO row: k, phi, n, soh_min) are '
     'cohort-registered.'),
    ('Z1b-04', 'Z1b', SED, 1586, 1636,
     ['inactive = s_is_zero and e_is_zero',
      '_set_esso_variable_state(model.es_D_per_unit[y_inv, y], active_pair, 0.0)',
      '_set_esso_variable_state(model.es_soh_per_unit_cumul[y_inv, y], active_pair, 1.0)',
      '_set_esso_variable_state(model.es_pch_per_unit[y_inv, y, d, p], active_pair, 0.0)',
      'constraint.deactivate()'],
     'At x = 0 every cohort is inactive: avg_ch_dch, D fixed 0, SoH fixed 1.0, pch/pdch fixed 0, and EVERY '
     'cohort-registered row (incl. D, SoH, floor) deactivated.'),
    ('Z1b-05', 'Z1b', SED, 1441, 1447,
     ['_configure_esso_cohort_state(model, y_inv, s_candidate, e_candidate, shared_ess_data.params.slacks)',
      '_configure_esso_cohort_pnet_share_rows(model)'],
     'Both are applied for every candidate (the production entry for the ESSO candidate).'),
    ('Z1b-06', 'Z1b', SED, 1709, 1714,
     ['if n_active <= 1:', 'model.energy_storage_cohort_pnet_share_h3[constraint_idx].deactivate()'],
     'Cohort-allocation (H3) rows: all deactivated, shares 0.0, when <= 1 cohort is active (x = 0: none).'),
    ('Z1b-07', 'Z1b', SED, 548, 550,
     ['initialize=1.00, bounds=(0.00, 1.00))', 'model.es_soh_per_unit_cumul.fix(1.00)'],
     'SoH initialisation: built fixed at 1.0 (unfixed only inside an active window).'),
    ('Z1b-08', 'Z1b', SED, 846, 854,
     ['salvage_value = _build_terminal_salvage_value_expression(shared_ess_data, model, shared_ess_idx)',
      'model.salvage_value = pe.Expression(expr=salvage_value)', 'expr=model.feasibility_penalty,'],
     'The ESSO objective is the feasibility penalty only; the salvage is an Expression outside every objective.'),
    ('Z1b-09', 'Z1b', SRP, 1302, 1306,
     ['gross_operational_cost = (gross_operational_cost_including_settlement',
      'net_operational_recourse = gross_operational_cost - terminal_salvage_value'],
     'Q (gross) is the network primal values less transfers; the salvage enters only net_operational_recourse.'),
    ('Z1b-10', 'Z1b', SRP, 1237, 1239,
     ['transmission_network.get_primal_value(models', 'distribution_network.get_primal_value(models'],
     'Q is summed over TSO and DSO blocks only (no ESSO term).'),
    ('Z1b-11', 'Z1b', SED, 615, 628,
     ['model.available_s_capacity_unit.add(model.es_s_available_per_unit[y_inv, y] == model.es_s_rated_per_unit[y_inv, y])',
      'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * soh_mid)'],
     "The available-capacity rows are NOT cohort-registered (active at x = 0). In 'end' mode they read "
     "E_av = E_rated * SoH with SoH FIXED at 1.0 (no ageing parameter); in 'mid' mode the active row carries the "
     "coefficient phi**(n/2) (with D fixed 0, prev SoH fixed 1) -- the only ageing parameter that can reach an "
     "active ESSO row at x = 0, on a variable forced to 0. Both E1 arms use 'end'."),
    ('Z1b-12', 'Z1b', SED, 562, 570,
     ['tcal_norm = round(shared_energy_storage.t_cal / (shared_ess_data.years[repr_years[y_inv]]))',
      'model.rated_e_capacity_unit.add(model.es_e_rated_per_unit[y_inv, y] == model.es_e_investment_fixed[y_inv])'],
     'The cohort window depends on t_cal (calendar_life_years); model_variant never changes t_cal (Z1a-07), so both '
     'E1 arms share the window.'),
    ('Z1b-13', 'Z1b', MCH, 1117, 1121,
     ['def configure_shared_ess_operational_state(', 'inactive = shared_ess_capacity_is_inactive(s_capacity, e_capacity)'],
     'Network side: shared-ESS rows read only the capacity Params; inactive when S or E is zero.'),
    ('Z1b-14', 'Z1b', APW, 333, 339,
     ["sess_capacity_year['s_available'] / s_base", "sess_capacity_year['e_available'] / s_base"],
     'The networks receive the ESSO AVAILABLE capacity (SoH-derated); at x = 0 it is 0 under any ageing model.'),
    ('Z1b-15', 'Z1b', SRP, 2944, 2947,
     ["if shared_ess_initialization_mode == 'price_taker':", '_initialize_shared_ess_from_price_taker('],
     'The price-taker initialisation (the other ageing-reading path, shared_ess_price_taker.py) runs only in mode '
     "'price_taker' ...'"),
    ('Z1b-16', 'Z1b', H, 2910, 2910,
     ["'shared_ess_initialization_is_standalone': a.shared_ess_initialization == 'standalone',"],
     "... and every campaign child refuses unless the mode is 'standalone'."),
    ('Z1b-17', 'Z1b', A47, 35, 38,
     ['It is ageing-independent: W21\'s case-file gate', 'NL file of', 'byte-identical before / after the edit'],
     'Prior empirical evidence (C3 -> baseline parameter change, ageing ON in both): x = 0 ESSO NL files byte-identical.'),
    # ---------------------------------------------------------------- Z1 (c) zero-capacity handling
    ('Z1c-01', 'Z1c', MCH, 906, 931,
     ['- m.shared_es_s_rated_fixed[e] * m.shared_es_pch_hat[e, s_m0, s_o0, p]) == 0',
      'that no expression ever divides by the rated-capacity decision variable'],
     'Normalised complementarity, network side: pch = S_av * pch_hat (product form, no division by S_av).'),
    ('Z1c-02', 'Z1c', MCH, 949, 951,
     ['elif params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:',
      '* m.shared_es_pdch_hat[e, s_m0, s_o0, p]) <= ESS_COMPLEMENTARITY_TOLERANCE'],
     'The complementarity row in force (BILINEAR_RELAXATION) is pch_hat * pdch_hat <= tol on the dimensionless pair.'),
    ('Z1c-03', 'Z1c', NET, 401, 402, ['bounds=(0.0, 1.0)'], 'pch_hat, pdch_hat in [0, 1].'),
    ('Z1c-04', 'Z1c', MCH, 1067, 1069,
     ['return abs(s_capacity) <= tolerance or abs(e_capacity) <= tolerance'],
     'Zero test on the network side: S_av OR E_av <= SHARED_ESS_ZERO_CAPACITY_TOLERANCE.'),
    ('Z1c-05', 'Z1c', DEFS, 93, 93, ['SHARED_ESS_ZERO_CAPACITY_TOLERANCE = 1e-10'], 'The tolerance.'),
    ('Z1c-06', 'Z1c', MCH, 1137, 1178,
     ['entry.fix(0.0)', 'entry.deactivate()'],
     'At zero capacity: every shared-ESS operational Var (incl. pch_hat/pdch_hat) fixed 0 and every row of '
     '_SHARED_ESS_OPERATIONAL_CONSTRAINTS (incl. sess_comp and the links) deactivated.'),
    ('Z1c-07', 'Z1c', MCH, 1050, 1064, ["'sess_pch_hat_link',", "'sess_comp',"],
     'The deactivated row list includes the complementarity and link rows.'),
    ('Z1c-08', 'Z1c', SED, 1588, 1593,
     ['s_is_zero = abs(s_capacity) <= tolerance', "raise ValueError(f'Inconsistent shared ESS cohort capacity"],
     'ESSO side: a cohort is inactive iff S_inv = E_inv = 0 (inconsistent zero raises).'),
    ('Z1c-09', 'Z1c', SED, 1716, 1726,
     ['total_e_rated = sum(', 'if y_inv in active_cohorts and total_e_rated > 0.00:',
      'share = pe.value(model.es_e_investment_fixed[y_inv]) / total_e_rated'],
     'Cohort allocation: the only division (share = E_inv / sum E_inv) is a Python float, guarded (> 0) and reached '
     'only with >= 2 active cohorts.'),
    ('Z1c-10', 'Z1c', SED, 785, 787,
     ['model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2'],
     'OBSERVATION: the ESSO aggregate capability row is not cohort-registered: at x = 0 it stays active as a '
     'radius-0 circle (es_s_rated forced 0 by active equalities) -- no division, ageing-independent, degenerate.'),
    ('Z1c-11', 'Z1c', SED, 2213, 2215,
     ['if not isclose(e_available, 0.00, abs_tol=SMALL_TOLERANCE):',
      "soh = e_available / pe.value(models[node_id].es_e_rated[y])"],
     'Reported SoH = E_av / E_rated is guarded at E_av = 0.'),
    ('Z1c-12', 'Z1c', SRP, 5024, 5028,
     ['if reference_mva is not None:', 'return float(reference_mva) / s_base'],
     'ESS-channel ADMM normalisation uses the declared reference rating (case file 2.5 MVA), not S_av.'),
    # ---------------------------------------------------------------- Z6 (a) perturbed start
    ('Z6a-01', 'Z6a', UB, 912, 945,
     ['def apply_perturbation(block, *, seed, delta, label):', 'new = value * (1.0 + delta * u)'],
     'The benchmark perturbation: per free Var of ONE static block, multiplicative, seeded by (seed, crc32(label)).'),
    ('Z6a-02', 'Z6a', UB, 948, 970,
     ['def apply_start(planning_problem, arm_models, *, start, warm_values=None, perturbation=None',
      "rec['warm'] = apply_warm_values(block, warm_values[label])"],
     'It is applied after a warm start from the CERTIFIED coordinated point, to the arm models before their one solve.'),
    ('Z6a-03', 'Z6a', UB, 1103, 1119,
     ['start_records = apply_start(planning_problem, _partial_models(planning_problem, dso_models=dso_models),',
      'tso_results = solve_tso_model(planning_problem, tso_model'],
     'Its only call sites: the uncoordinated arm runner.'),
    ('Z6a-04', 'Z6a', SRP, 2858, 2882,
     ['if initial_state is None:', 'dso_models, results[\'dso\'] = create_distribution_networks_models(',
      'esso_model, results[\'esso\'] = create_shared_energy_storage_model('],
     'Coordinated ADMM, fresh start: models are BUILT AND INITIALISATION-SOLVED inside _run_operational_planning; no '
     'callback exists between build and first solve.'),
    ('Z6a-05', 'Z6a', SRP, 2964, 2971,
     ["models = _clone_operational_models(initial_state['models'])", "dual_vars = deepcopy(initial_state['dual_vars'])"],
     'The only start alternative in production: a continuation from a full prior state (models, consensus, duals).'),
    ('Z6a-06', 'Z6a', G, 1194, 1198,
     ['`pre_solve_hook(planning=planning, sed=sed, candidate=candidate, report=report)`',
      'immediately after `_construct_arm_planning` returns and BEFORE'],
     'The campaign child\'s only hook runs before any model exists ...'),
    ('Z6a-07', 'Z6a', G, 1283, 1285,
     ["planning.run_operational_planning(", "type='distributed', candidate_solution=deepcopy(candidate),"],
     '... and the run is always a fresh (cold) start: no initial_state is passed.'),
    # ---------------------------------------------------------------- Z6 (b) rho_0
    ('Z6b-01', 'Z6b', ADMMP, 511, 513,
     ["admm_params.rho['v'] = params_data['rho']['v']", "admm_params.rho['ess'] = params_data['rho']['ess']"],
     'rho_0 is read from the case file (admm.rho.{v,pf,ess}.<network>).'),
    ('Z6b-02', 'Z6b', SRP, 5112, 5122,
     ["initialize=params.rho['v'][transmission_network.name])", "initialize=params.rho['ess'][transmission_network.name])"],
     'TSO: rho_0 initialises the mutable rho Params.'),
    ('Z6b-03', 'Z6b', SRP, 5378, 5388,
     ["initialize=params.rho['v'][distribution_network.network[year][day].name])"], 'DSO: likewise.'),
    ('Z6b-04', 'Z6b', SRP, 5483, 5483, ["initialize=params.rho['ess']['esso'])"], 'ESSO: likewise.'),
    ('Z6b-05', 'Z6b', SRP, 8275, 8284,
     ['elif not params.adaptive_penalty:', "factor = update_params['increase_factor']",
      "factor = 1.0 / update_params['decrease_factor']"],
     'The adaptive rule (case file adaptive_penalty true) moves rho from rho_0 by the declared factors.'),
    ('Z6b-06', 'Z6b', RHO, 89, 104,
     ['def apply_rho_to_params(planning, rho):', 'admm.rho[group][network_name] = float(rho[group])'],
     'An existing diagnostic helper that sets rho on a deep-copied planning (used by earlier G-gate arms).'),
    ('Z6b-07', 'Z6b', H, 327, 330,
     ["SUPPORTED_OVERRIDE_KEYS = frozenset({'anderson_acceleration'})"],
     'The campaign harness admits NO rho override ...'),
    ('Z6b-08', 'Z6b', H, 2897, 2899,
     ["'rho_v_matches_D': all(float(v) == G.S39_RHO_V for v in a.rho['v'].values()),",
      "'rho_ess_matches_D': all(float(v) == G.S39_RHO_ESS for v in a.rho['ess'].values()),"],
     '... and every child REFUSES unless rho equals the S39 constants ...'),
    ('Z6b-09', 'Z6b', H, 2924, 2925,
     ["raise RuntimeError(f'S44 campaign child: configuration not as frozen (case file D + cap): {missing}')"],
     '... (the refusal).'),
    ('Z6b-10', 'Z6b', G, 5878, 5880, ['S39_RHO_V = 0.0077', 'S39_RHO_PF = 0.198', 'S39_RHO_ESS = 0.01'],
     'The constants the child compares against (= the case file today).'),
    ('Z6b-11', 'Z6b', H, 1072, 1088,
     ['model_variant = validate_model_variant(model_variant)', "payload['ess_ageing_baseline'] = ess_ageing_baseline"],
     'The evaluation-key payload (read in full by the scoped search R-rho): rho is not a key component, so a '
     'case-file rho edit would not separate cache keys.'),
    # ---------------------------------------------------------------- Z6 (c) NRF
    ('Z6c-01', 'Z6c', UB, 192, 212,
     ['uncoord_no_reverse_flow[s_m, s_o, p]:   pg_adn[s_m, s_o, p] >= 0',
      'the coordinated path, production and the TSO arm never carry it.',
      "NO_REVERSE_FLOW_ROW = 'uncoord_no_reverse_flow'"],
     'The only NRF row in the repository: the uncoordinated DSO arm, DSO copy only; stated absent from the '
     'coordinated path.'),
    ('Z6c-02', 'Z6c', UB, 336, 338, ['return m.pg_adn[s_m, s_o, p] >= 0.0'], 'The row rule.'),
    ('Z6c-03', 'Z6c', UB, 405, 407, ['if no_reverse_flow:', 'block.add_component(NO_REVERSE_FLOW_ROW'],
     'Added only in build_dso_arm_models(no_reverse_flow=True).'),
    ('Z6c-04', 'Z6c', NET, 438, 445,
     ['model.pc_adn = pe.Expression(model.adn_nodes', 'model.pg_adn = pe.Expression(model.scenarios_market'],
     'The two interface copies in the coordinated model: TSO pc_adn (per ADN), DSO pg_adn.'),
    ('Z6c-05', 'Z6c', MCH, 1241, 1262,
     ['pc_adn = m.pc[adn_load_idx, s_m, s_o, p]', 'pc_adn += m.interface_delta_p[dn, s_m0, s_o0, p]'],
     'TSO copy = fixed anchor pc (the consensus request) + free signed interface_delta_p (scenario-free).'),
    ('Z6c-06', 'Z6c', MCH, 1287, 1297,
     ['return m.pg[ref_gen_idx, s_m, s_o, p] - shared_ess_p'],
     'DSO copy = reference-generator P MINUS the shared-ESS net power at the reference bus (equal to pg_ref at x = 0).'),
    ('Z6c-07', 'Z6c', SRP, 4233, 4242,
     ["interface_pf_p = consensus_vars['pf']['dso']['current'][adn_node_id][year][day]['p'][p] / s_base",
      'fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, s_o, p], interface_pf_p)'],
     'TSO builder: pc fixed to the DSO request (the anchor).'),
    ('Z6c-08', 'Z6c', SRP, 4267, 4273,
     ['tso_model[year][day].interface_delta_p[dn, s_m, s_o, p].setlb(-interface_transf_rating)'],
     'TSO builder: delta freed within +/- the interface rating -- a TSO-side NRF would bind here (pc + delta >= 0).'),
    ('Z6c-09', 'Z6c', SRP, 4277, 4282,
     ['tso_model[year][day].expected_interface_pf_p = pe.Var(', 'tso_model[year][day].expected_interface_pf_p_def = pe.Constraint('],
     'TSO builder site for a declared TSO-copy NRF row (create_transmission_network_model).'),
    ('Z6c-10', 'Z6c', SRP, 4372, 4377,
     ['dso_model[year][day].expected_interface_pf_p = pe.Var(dso_model[year][day].periods'],
     'DSO builder site, sequential (create_distribution_networks_models_sequential) ...'),
    ('Z6c-11', 'Z6c', SRP, 4488, 4488,
     ['dso_model[year][day].expected_interface_pf_p = pe.Var(dso_model[year][day].periods'],
     '... and parallel (create_distribution_network_model): a DSO-copy option must be wired in both builders.'),
    ('Z6c-12', 'Z6c', NETP, 168, 174,
     ["if 'shared_ess_model' in params_data:", "parameters.shared_ess_model = ESS_MODEL_BILINEAR_RELAXATION"],
     'Precedent for a declared per-network option read from the case33/case9 params files.'),
    ('Z6c-13', 'Z6c', SRP, 1154, 1157,
     ["p_int_tso_expected = pe.value(local_tso_model.expected_interface_pf_p[dn, p]) * s_base",
      "p_int_dso_expected = pe.value(local_dso_model.expected_interface_pf_p[p]) * s_base"],
     'What is recorded for interface P on each side: the expected interface P of the TSO copy and of the DSO copy ...'),
    ('Z6c-14', 'Z6c', SRP, 1209, 1220,
     ["'p_int_tso_expected_mw': p_int_tso_expected,", "'p_int_dso_expected_mw': p_int_dso_expected,",
      "'delta_p_mw': delta_p_by_scenario,", "'anchor_p_mw': anchor_p_by_scenario,", "'price_per_mwh': price,"],
     '... per (node, year, day, period), with the TSO delta and anchor per scenario and the price ...'),
    ('Z6c-15', 'Z6c', G, 2051, 2058,
     ['def write_interface_settlement_detail_s31c(', 'detail = _s31c_interface_detail(planning, models)'],
     '... written at TERMINAL only, to interface_settlement_detail_s31c.json in every evaluation directory.'),
    ('Z6c-16', 'Z6c', G, 1935, 1935, ['reporting_detail = srp._get_interface_reporting_detail(planning, models)'],
     'The writer reads production\'s reporting helper.'),
    # ---------------------------------------------------------------- C4 concurrency
    ('C4-01', 'C4', H, 11, 20,
     ['Every candidate', 'OWN OS process', 'At most `ctx.concurrency`'],
     'The harness runs N children, one process per evaluation, at most ctx.concurrency at once.'),
    ('C4-02', 'C4', H, 59, 65,
     ['Why concurrency is safe (checked, not assumed', 'p56a_oracle.fresh_planning',
      'pyomo\'s temporary NL/', 'workers are off, so no process pool is shared.'],
     'Its isolation argument (checked at Addendum 25).'),
    ('C4-03', 'C4', H, 312, 313,
     ["CAMPAIGN_LOCK_PATH = os.path.join(REPO, '.p515_s44_campaign.lock')"],
     'One campaign lock at the repo root: one campaign (one spec) per machine; slots live inside it.'),
    ('C4-04', 'C4', H, 1630, 1630, ["self.concurrency = int(spec['concurrency'])"], 'Concurrency is a spec field.'),
    ('C4-05', 'C4', H, 1872, 1872, ['while pending and len(running) < ctx.concurrency:'], 'The slot loop.'),
    ('C4-06', 'C4', ORA, 174, 186,
     ["logs = os.path.join(WORK_DIR, eval_id, 'logs')", 'planning.shared_ess_data.logs_dir = logs'],
     'Per-evaluation log directory for every TSO/DSO/ESSO holder (the IPOPT output_file path).'),
    ('C4-07', 'C4', SED, 29, 38,
     ['callers that never set it keep the pre-fix', 'behaviour of writing into the process cwd'],
     'The ESSO log fallback is the cwd -- harmless only because fresh_planning sets logs_dir (C4-06).'),
    ('C4-08', 'C4', S44G, 1, 21,
     ['Three concurrent evaluations through `p515_s44_campaign_harness.evaluate`', 'concurrency 3:',
      'reproduces arm D bitwise', '`data/SRP1/Results/P515S39_D_run/`'],
     'The ONLY concurrent-vs-serial bitwise gate: 3 slots, C* against the D run of P515S39 (pinned reference).'),
    ('C4-09', 'C4', S44G, 38, 39,
     ['Scalar checks (gating): c_star certified; cycles_run == 139;', 'gross_operational_cost == 650966975.2943751'],
     'Its gating scalars are literals of the pre-baseline D configuration.'),
    ('C4-10', 'C4', BRIEF, 1146, 1146, ['Harness gate ruling accepted (tie order from the sort fix).'],
     'Ruling on its one run (strict FAIL on tie-order diffs only).'),
    ('C4-11', 'C4', W142L, 134, 134, ['CONCURRENCY = 1'], 'The latest campaign launcher (W142 extension v6) fixes concurrency 1 ...'),
    ('C4-12', 'C4', W142L, 572, 572, ["'concurrency_1': spec.get('concurrency') == CONCURRENCY,"], '... and checks it.'),
    ('C4-13', 'C4', F2B, 855, 855, ["'concurrency_5': spec.get('concurrency') == CONCURRENCY == 5,"],
     'The 2x search launcher fixes concurrency 5 (Addendum 29 memory rule, pre-tight-tail).'),
    ('C4-14', 'C4', BWG, 75, 78,
     ['NOT COVERED, stated rather than implied', 'the parallel DSO builder (`create_distribution_network_model`) runs only with'],
     'The serial two-cycle bitwise gate does not cover the parallel DSO builder.'),
    ('C4-15', 'C4', G, 128, 128, ["SHARED_FROZEN_SMOPF_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF')"],
     'A shared repo directory every child HASHES before and after its run (read; a write by any child would be '
     'reported as an error by every concurrent sibling).'),
]

# committed records read for values (path, list of dotted keys)
RECORDS = {
    'T8_spec_v3': os.path.join(S53, 'w142_resettle_ext_v6', 'frozen_s53_resettle_ext_spec_v3_84775dc4.json'),
    'T8_no_ageing_eval_record': os.path.join(S53, 'w142_resettle_ext_v6', 'campaign_s53_w142_resettle_ext_v6_e_no_ageing',
                                             'evals', '1b10e9e161523f46_e_no_ageing', 'evaluation_record.json'),
    'T8_no_ageing_campaign_spec': os.path.join(S53, 'w142_resettle_ext_v6', 'campaign_s53_w142_resettle_ext_v6_e_no_ageing',
                                               'campaign_spec_s53_w142_resettle_ext_v6_e_no_ageing_10f4aa7a.json'),
    'x0_diff_analysis': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'case_file_baseline', 'x0_diff_analysis',
                                     'x0_diff_analysis.json'),
    'variant_checks_s46': os.path.join('data', 'SRP1', 'Results', 'P515S46', 'variant_checks', 'variant_checks.json'),
    's44_gate_results': os.path.join('data', 'SRP1', 'Results', 'P515S44', 'campaign_s44_gate', 'gate_results.json'),
    # cross-concurrency replays: an evaluation run at concurrency > 1, replayed serially with abort on divergence
    'w86_recert_spec': os.path.join(S53, 'tight_tail_w86', 'campaign_s53_w86_tail_recert',
                                    'campaign_spec_s53_w86_tail_recert_ddd6cd44.json'),
    'w101_cont_x0_results': os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0',
                                         'campaign_results.json'),
    'w101_cont_n7_4h_e1_results': os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_n7_4h_e1',
                                               'campaign_results.json'),
    'w101_cont_c_star_results': os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_c_star',
                                             'campaign_results.json'),
    'w101_cont_x0_spec': os.path.join(S53, 'w101_srp1_continuation', 'campaign_s53_w101_srp1_cont_x0',
                                      'campaign_spec_s53_w101_srp1_cont_x0_d1d5c3bc.json'),
    's47_phase_b_spec': os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_phase_b',
                                     'campaign_spec_s47_phase_b_8cfa264e.json'),
    'w142_pb_results': os.path.join(S53, 'w142_resettle_ext_v6', 'campaign_s53_w142_resettle_ext_v6_pb_y2025_n5_v6',
                                    'campaign_results.json'),
    'w142_pb_spec': os.path.join(S53, 'w142_resettle_ext_v6', 'campaign_s53_w142_resettle_ext_v6_pb_y2025_n5_v6',
                                 'campaign_spec_s53_w142_resettle_ext_v6_pb_y2025_n5_v6_dafacff3.json'),
    'bitwise_gate_w83': os.path.join(S53, 'tight_tail_w83', 'srp1_bitwise_gate', 'gate.json'),
    'bitwise_gate_w84': os.path.join(S53, 'tight_tail_w84', 'srp1_bitwise_gate', 'gate.json'),
    'case_file': os.path.join('data', 'SRP1', 'SRP1_params.json'),
    'ess_params': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
    'case33_1_params': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_params.json'),
    'case9_params': os.path.join('data', 'SRP1', 'case9', 'case9_params.json'),
}


# ======================================================================================================================
#  helpers
# ======================================================================================================================
def _log(msg):
    print(msg, flush=True)


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _read_bytes(rel):
    with open(os.path.join(REPO, rel), 'rb') as h:
        return h.read()


def _sha(rel):
    return _sha_bytes(_read_bytes(rel))


def _git(*args):
    r = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _file_at_head(rel):
    """tracked, unmodified, working-tree blob == index blob == HEAD blob."""
    _, ls, _ = _git('ls-files', '-s', '--', rel)
    _, st, _ = _git('status', '--porcelain', '--', rel)
    _, tree, _ = _git('ls-tree', 'HEAD', '--', rel)
    _, wt, _ = _git('hash-object', '--', rel)
    idx_blob = ls.split()[1] if ls else None
    head_blob = tree.split()[2] if tree else None
    return {'tracked': bool(ls), 'status_clean': st == '', 'head_blob': head_blob, 'worktree_blob': wt,
            'ok': bool(ls) and st == '' and head_blob is not None and head_blob == wt == idx_blob}


def _json(rel):
    with open(os.path.join(REPO, rel), 'r', encoding='utf-8') as h:
        return json.load(h)


def _dig(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def guards_state():
    blocked = pickle.load is _blocked_load and pickle.loads is _blocked_loads
    g = {'counts': dict(GUARD.counts), 'verify_0_failures': GUARD.verify(0)}
    pk = {'counts': dict(PICKLE_COUNTS), 'pickle_load_and_loads_blocked': blocked}
    pk['ok'] = blocked and PICKLE_COUNTS == {'load': 0, 'loads': 0}
    return g, pk, (not g['verify_0_failures']) and pk['ok']


def other_copies():
    r = subprocess.run(['ps', '-Ao', 'pid=,args='], capture_output=True, text=True).stdout.splitlines()
    me = os.getpid()
    out = []
    for ln in r:
        parts = ln.strip().split(None, 1)
        if len(parts) < 2 or int(parts[0]) == me:
            continue
        argv = parts[1].split()
        if argv and 'python' in os.path.basename(argv[0]) and any(a.endswith(SCRIPT_REL) for a in argv[1:3]) \
                and '--post-run' not in argv:
            out.append(ln.strip())
    return out


def production_modules():
    _, out, _ = _git('ls-files', '*.py')
    return sorted(f for f in out.splitlines() if '/' not in f and not re.match(r'p\d', f))


# ======================================================================================================================
#  sections
# ======================================================================================================================
def check_claims():
    files = {}
    out = []
    failures = []
    for cid, item, rel, a, b, needles, what in CLAIMS:
        if rel not in files:
            raw = _read_bytes(rel)
            files[rel] = {'sha256': _sha_bytes(raw), 'lines': raw.decode('utf-8').split('\n'),
                          'at_head': _file_at_head(rel)}
        lines = files[rel]['lines']
        excerpt_lines = lines[a - 1:b] if 1 <= a <= b <= len(lines) else []
        excerpt = '\n'.join(excerpt_lines)
        missing = [n for n in needles if n not in excerpt]
        ok = bool(excerpt_lines) and not missing and files[rel]['at_head']['ok']
        if not ok:
            failures.append({'claim': cid, 'file': rel, 'lines': [a, b], 'missing_substrings': missing,
                             'file_at_head': files[rel]['at_head']})
        out.append({'id': cid, 'item': item, 'file': rel, 'line_first': a, 'line_last': b, 'statement': what,
                    'required_substrings': needles, 'missing_substrings': missing, 'citation_holds': ok,
                    'excerpt': excerpt, 'excerpt_sha256': _sha_bytes(excerpt.encode('utf-8')),
                    'file_sha256': files[rel]['sha256']})
    file_table = {rel: {'sha256': d['sha256'], 'n_lines': len(d['lines']), **d['at_head']} for rel, d in files.items()}
    return out, file_table, failures


def read_records():
    out = {}
    for name, rel in RECORDS.items():
        out[name] = {'path': rel, 'sha256': _sha(rel), 'at_head': _file_at_head(rel)}
    spec = _json(RECORDS['T8_spec_v3'])
    cell = spec['cells']['e_no_ageing']
    out['T8_spec_v3']['values'] = {
        'cells.*.model_variant': {k: v.get('model_variant') for k, v in spec['cells'].items()},
        'cells.e_no_ageing.campaign_spec.git_head': cell['campaign_spec']['git_head'],
        'cells.e_no_ageing.campaign_spec.eval_key': cell['campaign_spec']['eval_key'],
        'cells.e_no_ageing.candidate_key': cell['candidate_key'],
        'cells.e_no_ageing.configuration.ess_ageing_baseline': cell['configuration']['ess_ageing_baseline'],
    }
    rec = _json(RECORDS['T8_no_ageing_eval_record'])
    traj = rec['ageing_trajectory_terminal']
    out['T8_no_ageing_eval_record']['values'] = {
        'model_variant': rec['model_variant'],
        'model_variant_readback_pre_run.expected': rec['model_variant_readback_pre_run']['expected'],
        'model_variant_readback_terminal.per_node.*.readback.d_row_form/phi_cal_in_model/floor_row_lower': {
            n: {k: v['readback'].get(k) for k in ('d_row_form', 'phi_cal_in_model', 'floor_row_lower', 'k')}
            for n, v in rec['model_variant_readback_terminal']['per_node'].items()},
        'ageing_trajectory_terminal.nodes.7.cells[*].{y,D,soh_end,soh_used_for_available_energy,e_available}': [
            {k: c.get(k) for k in ('y', 'D', 'soh_end', 'soh_used_for_available_energy', 'e_available')}
            for c in traj['nodes']['7']['cells']],
        'ageing_trajectory_terminal.nodes.{5,9}.cells': {n: traj['nodes'][n]['cells'] for n in ('5', '9')},
    }
    cs = _json(RECORDS['T8_no_ageing_campaign_spec'])
    out['T8_no_ageing_campaign_spec']['values'] = {'git_head': cs['git_head'], 'concurrency': cs['concurrency'],
                                                   'candidates[0].canonical': cs['candidates'][0]['canonical']}
    x0 = _json(RECORDS['x0_diff_analysis'])
    out['x0_diff_analysis']['values'] = {'checks': x0['checks'], 'all_ok': x0['all_ok'],
                                         'network_blocks_differing': x0['network_blocks_differing'],
                                         'per_esso_block.*.components_differing': {
                                             k: v['components_differing'] for k, v in x0['per_esso_block'].items()},
                                         'inputs': x0['inputs']}
    vc = _json(RECORDS['variant_checks_s46'])
    out['variant_checks_s46']['values'] = {'readback_table.no_ageing': vc['readback_table']['no_ageing'],
                                           'checks.V7_no_ageing_soh_identically_one':
                                               vc['checks'].get('V7_no_ageing_soh_identically_one'),
                                           'git_head_at_run': vc.get('git_head_at_run'),
                                           'D_per_instance.x0.settings_read_by_production':
                                               vc['detail']['D']['D_per_instance']['x0']['settings_read_by_production']}
    g = _json(RECORDS['s44_gate_results'])
    out['s44_gate_results']['values'] = {'gate_pass': g['gate_pass'], 'gate_scope': g['gate_scope'],
                                         'timestamp_utc': g['timestamp_utc'],
                                         'batch_info.max_concurrent_observed': g['batch_info']['max_concurrent_observed'],
                                         'c_star_vs_D.d_dir': g['c_star_vs_D']['d_dir']}
    _, c, _ = _git('log', '-1', '--format=%H %cI', '--', RECORDS['s44_gate_results'])
    out['s44_gate_results']['values']['committed_in'] = c
    for k in ('bitwise_gate_w83', 'bitwise_gate_w84'):
        d = _json(RECORDS[k])
        out[k]['values'] = {'gate_pass': d.get('gate_pass'), 'git_head_at_run': d.get('git_head_at_run')}
    cf = _json(RECORDS['case_file'])['admm']
    out['case_file']['values'] = {'admm.rho': cf['rho'], 'admm.adaptive_penalty': cf['adaptive_penalty'],
                                  'admm.penalty_update.min/max': [cf['penalty_update']['min'], cf['penalty_update']['max']],
                                  'admm.shared_ess_initialization': cf['shared_ess_initialization'],
                                  'admm.shared_ess_reference_rating_mva': cf['shared_ess_reference_rating_mva'],
                                  'admm.proximal_regularization.tso.tau':
                                      cf['proximal_regularization']['tso'].get('tau')}
    out['ess_params']['values'] = {'ageing': _json(RECORDS['ess_params'])['ageing'],
                                   'salvage_value.enabled': _json(RECORDS['ess_params'])['salvage_value']['enabled']}
    for k in ('case33_1_params', 'case9_params'):
        out[k]['values'] = {'shared_ess_model': _json(RECORDS[k]).get('shared_ess_model')}
    for k in ('w86_recert_spec', 'w101_cont_x0_spec', 's47_phase_b_spec', 'w142_pb_spec'):
        s = _json(RECORDS[k])
        out[k]['values'] = {'git_head': s.get('git_head'), 'concurrency': s.get('concurrency')}
    keys = ('replay_bitwise_through_cycle', 'replay_bitwise_through', 'replay_first_divergence',
            'replay_divergence_abort', 'replay_equal_every_gated_cycle', 'first_divergence_cycle')
    for k in ('w101_cont_x0_results', 'w101_cont_n7_4h_e1_results', 'w101_cont_c_star_results', 'w142_pb_results'):
        out[k]['values'] = {'found': _find_keys(_json(RECORDS[k]), keys)}
    return out


def _find_keys(obj, names, path=''):
    """Every (dotted path, value) whose key is in `names`, depth-first."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{path}.{k}' if path else str(k)
            if k in names and not isinstance(v, (dict, list)):
                found.append([p, v])
            found += _find_keys(v, names, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            found += _find_keys(v, names, f'{path}[{i}]')
    return found


def git_facts():
    _, head, _ = _git('rev-parse', 'HEAD')
    facts = {'HEAD': head, 'order_code_head': EXPECTED_HEAD_PREFIX}
    _, moved, _ = _git('diff', '--name-only', EXPECTED_HEAD_PREFIX, 'HEAD')
    facts['files_changed_order_head_to_HEAD'] = moved.splitlines()
    rc, _, _ = _git('merge-base', '--is-ancestor', LAST_PRODUCTION_COMMIT, T8_NO_AGEING_HEAD)
    facts[f'{LAST_PRODUCTION_COMMIT}_is_ancestor_of_{T8_NO_AGEING_HEAD}'] = rc == 0
    rc, _, _ = _git('merge-base', '--is-ancestor', T8_NO_AGEING_HEAD, 'HEAD')
    facts[f'{T8_NO_AGEING_HEAD}_is_ancestor_of_HEAD'] = rc == 0
    _, ns, _ = _git('diff', '--numstat', T8_NO_AGEING_HEAD, 'HEAD', '--', SED, SEP, 'shared_energy_storage.py', H,
                    RECORDS['ess_params'], RECORDS['case_file'], SRP, MCH, NET, 'network_data.py', ADMMP, APW, DEFS)
    facts[f'numstat_{T8_NO_AGEING_HEAD}_to_HEAD_model_and_harness_files'] = ns.splitlines()
    _, lg, _ = _git('log', '--format=%h %s', f'{T8_NO_AGEING_HEAD}..HEAD', '--', H)
    facts[f'harness_commits_{T8_NO_AGEING_HEAD}_to_HEAD'] = [l[:160] for l in lg.splitlines()]
    last = {}
    for f in production_modules():
        _, l, _ = _git('log', '-1', '--format=%h %cI', '--', f)
        last[f] = l
    facts['last_commit_per_production_module'] = last
    _, l, _ = _git('log', '-1', '--format=%h %cI %s', '--', *production_modules())
    facts['last_production_commit'] = l[:200]
    model_files = (SRP, NET, 'network_data.py', MCH, SED, SEP, ADMMP, APW, 'admm_anderson_acceleration.py', DEFS)
    for gate in ('bitwise_gate_w83', 'bitwise_gate_w84'):
        h = _json(RECORDS[gate]).get('git_head_at_run')
        _, ns2, _ = _git('diff', '--numstat', h, 'HEAD', '--', *model_files)
        facts[f'model_file_numstat_{gate}_head_to_HEAD'] = ns2.splitlines()
    # the two cross-concurrency replays: model-file changes between the concurrent original and the serial replay,
    # and between the replay and HEAD
    pairs = (('w86_recert_c3', _json(RECORDS['w86_recert_spec'])['git_head'],
              _json(RECORDS['w101_cont_x0_spec'])['git_head']),
             ('s47_phase_b_c5', _json(RECORDS['s47_phase_b_spec'])['git_head'], _json(RECORDS['w142_pb_spec'])['git_head']))
    for name, a, b in pairs:
        _, n1, _ = _git('diff', '--numstat', a, b, '--', *model_files)
        _, n2, _ = _git('diff', '--numstat', b, 'HEAD', '--', *model_files)
        facts[f'replay_{name}'] = {'original_head': a, 'replay_head': b,
                                   'model_file_numstat_original_to_replay': n1.splitlines(),
                                   'model_file_numstat_replay_to_HEAD': n2.splitlines()}
    return facts


def scoped_searches():
    out = {}
    prod = production_modules()
    texts = {f: _read_bytes(f).decode('utf-8') for f in prod}

    def grep(rx, files):
        hits = {}
        for f in files:
            for i, ln in enumerate(texts[f].split('\n'), 1):
                if re.search(rx, ln):
                    hits.setdefault(f, []).append(i)
        return hits
    out['S1_reverse_flow_in_production'] = {
        'scope': 'every tracked top-level non-p*.py module at HEAD (%d files)' % len(prod), 'files': prod,
        'regex': r'reverse_flow|no_reverse|NO_REVERSE|\bnrf\b', 'hits': grep(r'reverse_flow|no_reverse|NO_REVERSE|\bnrf\b', prod)}
    out['S2_perturb_in_production'] = {
        'scope': 'same', 'regex': r'perturb', 'hits': grep(r'perturb', prod)}
    out['S3_ageing_enabled_in_production'] = {
        'scope': 'same', 'regex': r'ageing_enabled', 'hits': grep(r'ageing_enabled', prod)}
    out['S4_initial_state_callers_in_production'] = {
        'scope': 'same', 'regex': r'initial_state\s*=', 'hits': grep(r'initial_state\s*=', prod)}
    # eval key region: every line of evaluation_key / _evaluation_key_payload mentioning rho
    htxt = _read_bytes(H).decode('utf-8').split('\n')
    out['R_rho_in_eval_key_region'] = {
        'scope': f'{H} lines 972-1140 (evaluation_key and its payload builder)',
        'hits': [i for i in range(972, 1141) if re.search(r'rho', htxt[i - 1], re.I)]}
    # committed campaign specs
    _, specs, _ = _git('ls-files', 'data/SRP1/Results/*campaign_spec_*.json')
    specs = specs.splitlines()
    combos, no_ageing_entries, conc = [], [], {}
    for rel in specs:
        try:
            s = _json(rel)
        except (ValueError, OSError):
            continue
        conc[str(s.get('concurrency'))] = conc.get(str(s.get('concurrency')), 0) + 1
        for e in s.get('candidates') or []:
            mv = e.get('model_variant')
            fm = e.get('flex_price_multiplier')
            if mv is not None and fm not in (None, 1.0):
                combos.append({'spec': rel, 'label': e.get('label')})
            if isinstance(mv, dict) and mv.get('ageing_enabled') is False:
                canon = e.get('canonical') or {}
                nodes = canon.get('nodes') if isinstance(canon, dict) else None
                is_x0 = isinstance(nodes, dict) and all(float(v[0]) == 0.0 and float(v[1]) == 0.0
                                                         for v in nodes.values())
                no_ageing_entries.append({'spec': rel, 'label': e.get('label'), 'canonical': canon, 'is_x0': is_x0,
                                          'model_variant': mv})
    out['S5_committed_campaign_specs'] = {
        'scope': f'git ls-files data/SRP1/Results/*campaign_spec_*.json at HEAD ({len(specs)} files), entries under '
                 '"candidates"',
        'n_specs': len(specs), 'concurrency_histogram': conc,
        'entries_with_model_variant_and_flex_multiplier_not_1': combos,
        'entries_with_ageing_enabled_false': no_ageing_entries,
        'any_no_ageing_entry_at_x0': any(e['is_x0'] for e in no_ageing_entries)}
    # search launchers asserting no model variant
    launchers = sorted(f for f in _git('ls-files', 'p515_*.py')[1].splitlines())
    lt = {f: _read_bytes(f).decode('utf-8') for f in launchers}
    out['S6_launchers_refusing_model_variant'] = {
        'scope': f'every tracked p515_*.py ({len(launchers)} files)',
        'regex': r"'no_(ageing_)?model_variant(_anywhere)?'",
        'hits': {f: [i for i, ln in enumerate(t.split('\n'), 1)
                     if re.search(r"'no_(ageing_)?model_variant(_anywhere)?'", ln)] for f, t in lt.items()
                 if re.search(r"'no_(ageing_)?model_variant(_anywhere)?'", t)}}
    # 'concurren...' within 12 lines of 'bitwise' in any tracked p515_*.py: the candidate list a reader inspects for a
    # concurrent-vs-serial bitwise gate (W182e inspected each hit by hand; the report states the outcome)
    win = {}
    for f, t in lt.items():
        L = t.split('\n')
        c = [i for i, ln in enumerate(L, 1) if re.search(r'concurren', ln, re.I)]
        b = [i for i, ln in enumerate(L, 1) if re.search(r'bitwise', ln, re.I)]
        pairs = [[i, j] for i in c for j in b if abs(i - j) <= 12]
        if pairs:
            win[f] = pairs
    out['S7_concurrent_near_bitwise'] = {
        'scope': f'every tracked p515_*.py ({len(launchers)} files): a line matching /concurren/i within 12 lines of '
                 'a line matching /bitwise/i', 'hits': win}
    # campaigns frozen on the final production code (git_head descends from the last production commit)
    by_c = {}
    for rel in specs:
        try:
            s = _json(rel)
        except (ValueError, OSError):
            continue
        h = s.get('git_head')
        if not h:
            continue
        rc, _, _ = _git('merge-base', '--is-ancestor', LAST_PRODUCTION_COMMIT, h)
        if rc == 0:
            by_c.setdefault(str(s.get('concurrency')), []).append(rel)
    out['S8_campaigns_on_final_production_code_by_concurrency'] = {
        'scope': f'the {len(specs)} committed campaign specs; git_head a descendant of {LAST_PRODUCTION_COMMIT}',
        'counts': {k: len(v) for k, v in by_c.items()}, 'specs': by_c}
    return out


PREDICTIONS = [
    {'item': 'Z1', 'prediction': 'Q(0) exactly independent of the ageing model (expert)',
     'outcome': 'HELD (by code reading; empirical NL identity on record only for an ageing-ON parameter change)',
     'basis': ['Z1b-01..Z1b-08', 'Z1b-09/10', 'Z1b-11', 'x0_diff_analysis', 'S5 any_no_ageing_entry_at_x0']},
    {'item': 'Z6', 'prediction': 'rho_0 is a configuration parameter (expert)',
     'outcome': 'HELD in production (case-file key, no production change); NOT runnable through the campaign harness '
                'without a harness change (child refuses rho != S39; rho absent from the eval key)',
     'basis': ['Z6b-01..05', 'Z6b-07..11', 'R_rho_in_eval_key_region']},
    {'item': 'Z6', 'prediction': 'the NRF row needs a new declared option (code change, both copies) (expert)',
     'outcome': 'HELD: no NRF option exists in the coordinated path; the only NRF row is the uncoordinated DSO arm\'s',
     'basis': ['Z6c-01..11', 'S1_reverse_flow_in_production']},
]


def md_summary(o):
    L = [f"# W182e — code paths Z1 / Z6 / C4 at HEAD {o['git_facts']['HEAD'][:8]}", '',
         f"Script `{SCRIPT_REL}`; zero solves; guards {'0/0 verified' if o['guards_ok'] else 'FAILED'}; "
         f"citations {o['n_claims_holding']}/{o['n_claims']} hold; integrity {'OK' if not o['integrity_failures'] else 'FAILED'}.",
         '', '## Predictions', '']
    for p in o['predictions']:
        L.append(f"- **{p['item']}** {p['prediction']}: **{p['outcome']}** (basis: {', '.join(p['basis'])})")
    L += ['', '## Citations', '', '| id | file:lines | holds | excerpt sha256 | statement |', '|---|---|---|---|---|']
    for c in o['claims']:
        st = c['statement'].replace('|', '\\|')
        L.append(f"| {c['id']} | `{c['file']}`:{c['line_first']}-{c['line_last']} | {c['citation_holds']} | "
                 f"`{c['excerpt_sha256'][:12]}` | {st} |")
    L += ['', '## Files cited (sha256 at HEAD)', '']
    for f, d in o['files'].items():
        L.append(f"- `{f}` `{d['sha256']}` ({d['n_lines']} lines; at HEAD {d['ok']})")
    return '\n'.join(L) + '\n'


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('man', 'json', 'md')]
    rels += [os.path.join(out_dir, f) for f in sorted(os.listdir(os.path.join(REPO, out_dir))) if LAUNCH_LOG_RX.fullmatch(f)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W182e post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W182e post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--post-run', action='store_true')
    args = ap.parse_args()
    if args.post_run:
        post_run(OUT_DIR)
    started = time.time()
    pre = []
    others = other_copies()
    if others:
        pre.append(f'another copy of {SCRIPT_REL} is running: {others}')
    od = os.path.join(REPO, OUT_DIR)
    if not os.path.isdir(od):
        pre.append(f'{OUT_DIR} does not exist (create it, with only the launch log, before the run)')
    else:
        extra = [f for f in sorted(os.listdir(od)) if not LAUNCH_LOG_RX.fullmatch(f)]
        if extra:
            pre.append(f'{OUT_DIR} is not fresh: {extra}')
    _, head, _ = _git('rev-parse', 'HEAD')
    # Six other W182 Workers commit in parallel: HEAD may move past the order's code head. Accept HEAD only if it
    # descends from it and no cited file and no production module differs between the two.
    rc, _, _ = _git('merge-base', '--is-ancestor', EXPECTED_HEAD_PREFIX, 'HEAD')
    if rc != 0:
        pre.append(f'HEAD {head} does not descend from {EXPECTED_HEAD_PREFIX}')
    else:
        cited = sorted({c[2] for c in CLAIMS} | set(RECORDS.values()) | set(production_modules()))
        _, moved, _ = _git('diff', '--name-only', EXPECTED_HEAD_PREFIX, 'HEAD', '--', *cited)
        if moved:
            pre.append(f'cited or production files changed since {EXPECTED_HEAD_PREFIX}: {moved.splitlines()}')
    if pre:
        for p in pre:
            _log(f'[W182e PRECONDITION FAILED] {p}')
        sys.exit(1)
    _log(f'[W182e] HEAD {head}; {len(CLAIMS)} citations')

    claims, files, failures = check_claims()
    for c in claims:
        _log(f"[W182e] {c['id']:8s} {c['file']}:{c['line_first']}-{c['line_last']} holds={c['citation_holds']}"
             + (f" MISSING {c['missing_substrings']}" if c['missing_substrings'] else ''))
    records = read_records()
    facts = git_facts()
    searches = scoped_searches()
    for k, v in searches.items():
        _log(f"[W182e] search {k}: " + json.dumps({kk: vv for kk, vv in v.items() if kk not in ('files',)})[:600])

    integrity = list(failures)
    for name, r in records.items():
        if not r['at_head']['ok']:
            integrity.append({'record': name, 'path': r['path'], 'at_head': r['at_head']})
    g, pk, guards_ok = guards_state()
    out = {
        'stage': 'P5.15 Addendum 82 v2 W182e -- Z1 / Z6 / C4 code paths (zero solves, code reading)',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 82 v2 (W182 order, items Z1, Z6, C4)',
                      'manuscript_review/manuscript_critical_review_2026-10-25.md section 3.7',
                      'manuscript_review/STEP7_PLAN_REVIEW_2026-10-25.md'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
        'objective_convention': 'Q = certified gross_operational_cost, settlement excluded; salvage reported, excluded',
        'git_facts': facts, 'files': files, 'claims': claims, 'n_claims': len(claims),
        'n_claims_holding': sum(1 for c in claims if c['citation_holds']),
        'records': records, 'scoped_searches': searches, 'predictions': PREDICTIONS,
        'integrity_failures': integrity,
        'solve_profile_guard': g, 'pickle_guard': pk, 'guards_ok': guards_ok,
        'production_modules_imported': False,
        'wall_clock_s': time.time() - started,
    }
    with open(os.path.join(od, OUT_NAMES['json']), 'x', encoding='utf-8') as h:
        GRIO.dump(out, h, indent=1, sort_keys=False)
        h.write('\n')
    with open(os.path.join(od, OUT_NAMES['md']), 'x', encoding='utf-8') as h:
        h.write(md_summary(out))
    man = {os.path.join(OUT_DIR, OUT_NAMES[k]): _sha(os.path.join(OUT_DIR, OUT_NAMES[k])) for k in ('json', 'md')}
    man[SCRIPT_REL] = _sha(SCRIPT_REL)
    with open(os.path.join(od, OUT_NAMES['man']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    g, pk, guards_ok = guards_state()
    _log(f"[W182e] guards: solve {g['counts']} verify0 failures {g['verify_0_failures']}; pickle {pk['counts']}")
    _log(f"[W182e] citations {out['n_claims_holding']}/{out['n_claims']} hold; integrity failures {len(integrity)}")
    if not guards_ok:
        _log('[W182e] GUARD FAULT')
        sys.exit(1)
    if integrity:
        for f in integrity:
            _log(f'[W182e INTEGRITY] {f}')
        sys.exit(3)
    sys.exit(0)


if __name__ == '__main__':
    main()
