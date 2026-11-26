"""
P5.15 Addendum 76 + its Amendment, Planner task W177b-2' -- record checks behind P5_15_W177B_PART2_E10C104.md:
Part 2 re-run at Overleaf e10c104, the Section 3 opening CONFIRM (which market / operation scenario the
single-scenario instance carries; the 3 x 3 prefix draw), and the claim audit of the letter
(response_to_reviewers_draft.tex at e10c104). ZERO SOLVES, NO MODEL BUILD, NO PICKLE.

What it does:
  A. Identity: clone HEAD (declared e10c104), main.tex sha256 / line count (declared), the letter's sha256 (declared)
     equal to manuscript_review/response_to_reviewers_v2.tex, clone *.tex unmodified; frozen tables sha256 590088fe...;
     the W177b-2 checks output (30002d43...) as declared.
  B. Diff 51a6fb5 -> e10c104 of main.tex: every hunk; whether old lines 1140-1560 (the Section 4 preamble, 4.1-4.7 and
     the Conclusions up to the changed future-work line) equal new lines 1172-1592 (offset +32); the line of every
     prior-row fragment and of the l. 1148 flag at both versions (exact substring, found exactly once).
  C. Section 3 opening CONFIRM. Production's own data reader (W22 `production_read`, imported unedited; model
     construction blocked, declared stop point SharedEnergyStorageData.read_parameters_from_file, 1 hit per read,
     2 reads) on data/SRP1/SRP1.json and on the committed 3 x 3 case (W89, sha 2a64e3c5...); the planning objects are
     captured by W52's pass-through Recorder (imported unedited). For every 2025 block (TSO + 3 DSOs x 4 days) every
     realized scenario array of SRP1 (market energy / flexibility prices; per load pd, qd, flexibility up / down; per
     generator pg, qg) is compared, as float64 bytes, with each of the three scenario rows of the same array in the
     3 x 3 read. The block seeds are compared too. SRP1's 2030 / 2035 blocks have no counterpart year in the 3 x 3
     instance (recorded). Code citations (file:line, found by exact substring) carry the HEAD blob hash of each file
     and whether the working-tree file equals it. W89's instance_record prefix verification and W52's index-level
     prefix check are read and summarised.
  D. Letter and manuscript record checks: every quoted letter / manuscript fragment located (exact substring, found
     once); value-in-size affinity of the frozen B ladders (least squares, residuals against 2 tau, slope against the
     per-MWh investment cost); uncertified causes in the frozen cells and the export T2; evaluation-record statuses
     (infeasibility scan) over every evaluation_record.json under data/SRP1/Results; text-presence checks in main.tex
     (abstract chemistry, the introduction's out-of-scope clause, a 5 x 5 / future-work statement, the
     "representative-day-weighted" wording, the Appendix E table label).

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked for the whole run and counted (0 expected). W22's model-construction blockers
installed for the two reads (0 blocked calls expected). Refuses to run concurrently with itself and to overwrite an
existing output; refuses if the clone HEAD or any declared hash differs.

Output (write-once, new directory) data/SRP1/Results/P515S53/w177b_part2_e10c104/:
    w177b2r_checks.json, production_stdout_capture.log, production_read/ (production's scenario plots; hashed in the
    manifest only, not committed), manifest_sha256.json (written by --manifest after the launch log is closed).

Command (repo root, canonical interpreter, attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S53/w177b_part2_e10c104
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b2r_checks.py \\
        > data/SRP1/Results/P515S53/w177b_part2_e10c104/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w177b2r_checks.py --manifest
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label="P5.15 W177b-2' checks (never solves)").install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError("W177b-2': pickle.load called -- no model loads are permitted")


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError("W177b-2': pickle.loads called -- no model loads are permitted")


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import csv  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_DIR = os.path.join(P53, 'w177b_part2_e10c104')
OUT_JSON = os.path.join(OUT_DIR, 'w177b2r_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
STDOUT_CAPTURE = os.path.join(OUT_DIR, 'production_stdout_capture.log')
LAUNCH_LOG = os.path.join(OUT_DIR, 'launch.log')

CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
LETTER = os.path.join(CLONE, 'response_to_reviewers_draft.tex')
LETTER_V2 = os.path.join('manuscript_review', 'response_to_reviewers_v2.tex')
S3_FIX = os.path.join('manuscript_review', 'section3_intro_conclusions_fixes.tex')
DECLARED_CLONE_HEAD = 'e10c104'
OLD_CLONE_HEAD = '51a6fb5'
DECLARED_MAIN_SHA256 = '1b532035abc65a7f8a08cea6304ac14fcd4ec2e6da553b9a1f250a0b09b5ba1b'
DECLARED_MAIN_LINES = 2345
DECLARED_LETTER_SHA256 = 'bc270fe2cf4b44f7e9547e17ccd85ecd94b56e8f99dd717486d9cf863c20a642'
OLD_MAIN_SHA256 = '61fe630cc427d622a855e127f33f96d4be1a7362fe23552b87d2d54318a39346'

FROZEN = os.path.join(P53, 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
K_PRIOR = os.path.join(P53, 'w177b_part2', 'w177b_part2_checks.json')
K_PRIOR_SHA = '30002d430cf96fc942d3f5e9feeef8698ce7f9d4e22a44bdf8e81cf6b7db6003'
EXPORT_T2 = os.path.join('manuscript_review', 'export_W160', 'T2.csv')
EXPORT_T9 = os.path.join('manuscript_review', 'export_W160', 'T9.csv')
W89_RECORD = os.path.join(P53, 'w89_3x3', 'instance', 'instance_record.json')
W52_SELECTION = os.path.join(P53, 'selection_3x3', 'selection_3x3.json')
W22_SPREADS = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'market_spreads', 'market_spreads.json')
W153_DISCOUNT = os.path.join(P53, 'w153_step5_rows', 'w153_discount_row.json')

SRP1_CASE = 'data/SRP1/SRP1.json'
CASE_3X3 = 'data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json'
CASE_3X3_SHA = '2a64e3c5bd06c30c8a015207f41b69257b001094cbd4b185a84394a6765c5871'
SRP1_CASE_SHA = '61a794a7ce7a3fb983f2e92128eec446dd7dbb17ad75a3b3b1c5735f8bd4e4ef'
COMMON_YEAR = 2025
DECLARED_STOPS = 2
TWO_TAU = 9078.14   # 2 tau, tau = 4,539.07 (main.tex Section 3.6; frozen constants)

# (Old line at 51a6fb5) -> found at e10c104. Section 4 preamble, 4.1-4.6 and Conclusions rows of P5_15_W177B_PART2.md.
PRIOR_ROWS = [
    ('flag_1148', 1148, 'Every reported number is a certified recourse evaluation or a difference between two of them'),
    ('4.2-1', 1284, 'the value of every evaluated 4~h configuration at node~7 exceeds its investment cost'),
    ('4.2-2', 1284, 'The returned plan is determinately better than seven of them and within resolution of the other '
                    'six.'),
    ('4.3-1', 1360, 'The negative investment margin persists across the tested ageing calibrations and both floor '
                    'settings.'),
    ('4.5-1', 1407, 'The planning search again returned the plan without shared storage'),
    ('4.6-1', 1467, 'so a higher rate lowers the value of the unit, from 277.1~k\\euro{} undiscounted to '
                    '203.4~k\\euro{} at 8\\,\\%. The unit does not pay at any rate between 0 and 8\\,\\%:'),
    ('4.6-2', 1488, 'Across the 60 pairwise comparisons reported in this section, applying the salvage credit changes '
                    'no sign. It changes two verdicts: the investment-year comparison above'),
    ('C-1', 1549, 'each evaluation is certified by an explicit stopping rule'),
    ('C-2', 1553, 'every other evaluated plan, from 0.5 to 5~MWh and at one or two nodes, also loses determinately'),
    ('C-3', 1557, 'the value of shared storage cannot be assessed apart from its ageing or from the coordination of '
                  'the operators who share it'),
    # the named items (a), (b) of Addendum 76 Part 2 (consistent rows at 51a6fb5)
    ('item_a', 1176, 'All fourteen admissible unit neighbours of this plan were evaluated, and none reduced the total '
                     'discounted cost beyond the search resolution.'),
    ('item_b', 1284, 'only seventeen of its sixty-one admissible unit neighbours were evaluated'),
]

# e10c104 manuscript fragments (new text, Section 3 opening, cross-checks for the letter)
MAIN_FRAGMENTS = [
    ('s3_two_instances', 'The 15-year planning horizon is represented in two instances.'),
    ('s3_single', 'uses three representative years, 2025, 2030 and 2035, each standing for a five-year block, with one '
                  'market scenario and one operation scenario of load and RES generation'),
    ('s3_multi', 'uses five representative years, 2025, 2028, 2031, 2034 and 2037, each standing for a three-year '
                 'block, with three market scenarios and three operation scenarios, nine operating scenarios per '
                 'representative day'),
    ('s3_days', 'In both instances each year is represented by four seasonal representative days.'),
    ('s3_single_year', 'the case-study search considers a single installation year common to all selected shared ESSs, '
                       'with one capacity installation per node'),
    ('s3_trajectories', 'Three ESS investment-cost trajectories are considered at the planning level (Subsection~3.1).'),
    ('s3_budget', 'The total investment budget for shared ESS deployment is set to 1~M\\euro, and the maximum '
                  'installable energy capacity at each interface node is limited to 5.00~MWh.'),
    ('s3_two_comparisons', 'Every candidate plan is compared with the plan without shared storage'),
    ('concl_future_alloc', 'and to an allocation of the degradation of a shared asset among the operators that use it.'),
    ('lim_inference', "the plan without storage is the search's answer under each of them, but this was not run"),
    ('lim_attribution', 'the attribution of degradation to the transmission and distribution operators is not unique'),
    ('s234_attribution', 'cannot be uniquely attributed to either the TSO or the DSOs'),
    ('s47_post_cert', 'Evaluations continued past a certificate under this rule moved by at most $0.9\\tau$.'),
    ('s47_causes', '5 refused by the gap clause, 2 reset by residual lapses, 3 failed by the growth test'),
    ('s36_frozen_cfg', 'under one frozen configuration'),
    ('a3_freeze', 'all $\\rho_g$ are frozen from the cycle after the first residual pass'),
    ('a_prodexit_scope', 'The single-scenario evaluations of this paper were stopped by the certification rule'),
    ('s45_prodexit', 'These evaluations were stopped by the production exit of the coordination procedure'),
    ('intro_uncertainty', 'Uncertainty is represented through alternative trajectories of future ESS investment costs '
                          'and operational scenarios of load, RES generation and market prices.'),
    ('res_ref_unit_def', 'The reference unit, comprising 0.25~MVA a'),
    ('s36_ref_unit_first', 'the value of the reference unit in force when the rule was adopted'),
    ('appE_submitted_table', '\\label{tab:cs3_results_summary_years_days}'),
    ('test_system_panel_a', 'IEEE 9-bus TN and ADN connection points.'),
    ('test_system_panel_b', 'IEEE 33-bus topology adopted for each ADN.'),
    ('graphical_abstract', '\\begin{graphicalabstract}'),
]

# text that the letter says the manuscript now contains; absence is recorded (no expectation encoded)
MAIN_PRESENCE_PROBES = [
    ('abstract_lithium', 'abstract', ['lithium', 'Li-ion', 'LFP']),
    ('intro_out_of_scope', 'all', ['outside the present formulation', 'infrastructure retirement', 'policy change']),
    ('five_by_five_future', 'all', ['5 \\times 5', '5\\times5', 'twenty-five', 'five market', '25 operating']),
    ('rep_day_weighted', 'all', ['representative-day-weighted', 'representative-day weighted']),
    ('appE_claims_table', 'appendixE', ['claims', 'consensus gap', 'settling slack']),
]

LETTER_FRAGMENTS = [
    ('open_each_candidate', 'coordinated TSO--DSO recourse under an explicit stopping and certification rule'),
    ('open_every_other', 'as does every other plan'),
    ('open_affine', 'at the baseline calibration the value is measured to be affine in storage size with a slope below '
                    'cost'),
    ('open_assumption', 'provided that this affinity holds under it, which we state as an assumption (Section~4.8)'),
    ('open_5x5', 'could not be evaluated under the certification rule within the memory available and is stated as '
                 'future work'),
    ('open_shapley', 'a marginal or Shapley-type allocation is identified as future work'),
    ('r12_chem', 'the abstract names the chemistry'),
    ('r12_conclusion', 'the planning conclusion is unchanged'),
    ('r12_abstract_size', 'and the abstract now states it'),
    ('r12_marginal', 'a marginal allocation is listed as future work (Section~5)'),
    ('r12_attr_where', '(Section~2.3.4, last paragraph, and Section~4.8)'),
    ('r14_intro', 'the introduction now states which of these sources of uncertainty are represented in this work'),
    ('r14_exact', 'The per-year operating costs do not depend on the rate; only the present-value weights do, which is '
                  'why the sensitivity is exact at a fixed plan.'),
    ('r15_two_panels', 'was redrawn as two panels'),
    ('r22_stop', 'or, for the evaluations behind the reported comparisons, when the certification rule of '
                 'Section~2.2.7'),
    ('r22_blocks', '48 blocks'),
    ('r23_feasible', 'every candidate evaluated in this study was feasible'),
    ('r23_certifies', 'what it certifies is each evaluation (Section~2.2.7), and every planning statement in the paper '
                      'is a pairwise comparison of certified evaluations with a stated determinacy threshold'),
    ('r25_most', 'the limitations section explains the mechanism behind most of them'),
    ('r25_freeze', 'frozen once the residuals first pass'),
    ('r25_noadjust', 'no parameter was adjusted in'),
    ('r25_post', '$\\le 0.9\\,\\tau$ on continued cells'),
    ('r25_appE', '(Section~4.7 and Appendix~E)'),
    ('r31_sixty', 'It changes no sign in the 60 reported comparisons and two verdicts'),
    ('r31_appE', 'the full claims table of Appendix~E'),
    ('r32_tau', '$\\tau = \\delta_R V/4$ with $\\delta_R = 0.07$ of the single-scenario storage value'),
    ('r32_fig', 'each results figure of the revision is discussed by'),
    ('r33_no_soc', 'so no state of charge is needed there and none appears'),
    ('r33_premium', "its deviation from the DSO's own expected import being charged at a premium on the hour's expected "
                    'market price'),
    ('r33_closure', 'closes there up to a penalised slack bounded by a small fraction of the available energy'),
    ('r33_chain', 'into the available energy of that year and the next'),
    ('r36_inference', 'this is stated as an inference, the search was not rerun per calibration (Section~4.8)'),
    ('r37_discount', 'one discount factor per representative year applied to'),
    ('r37_inv', 'discounted from their installation year (Section~2.2.1)'),
    ('r37_annual', 'representative-day-weighted annual'),
    ('fc_noload', 'a transmission network that has no load of its own'),
    ('fc_49', '49 predictions, with outcomes'),
    ('fc_single_thread', 'single machine, single thread, bitwise replays, solver and library versions'),
]

# code citations: (id, file, exact substring, expected hits or None for 'all hits listed')
CODE_CITES = [
    ('srp1_years', 'data/SRP1/SRP1.json', '"2030": 5', 1),
    ('srp1_nmarket', 'data/SRP1/SRP1.json', '"NumMarketScenarios": 1', 1),
    ('srp1_noper', 'data/SRP1/SRP1.json', '"num_operation_scenarios": 1', 4),
    ('srp1_seed', 'data/SRP1/SRP1.json', '"RandomSeed": 2026', 1),
    ('x3_nmarket', CASE_3X3, '"NumMarketScenarios": 3', 1),
    ('x3_noper', CASE_3X3, '"num_operation_scenarios": 3', 4),
    ('x3_seed', CASE_3X3, '"RandomSeed": 2026', 1),
    ('read_nmarket', 'shared_resources_planning.py', "planning_problem.num_market_scenarios = planning_data['NumMarketScenarios']", 1),
    ('market_seed', 'shared_resources_planning.py', "market_seed = derive_random_seed(planning_problem.random_seed, 'market')", 1),
    ('market_pool', 'shared_resources_planning.py', "random_seed=derive_random_seed(market_seed, 'synthetic_profiles'),", 1),
    ('market_sample', 'shared_resources_planning.py', "energy_selected_profiles = synthetic_profiles['energy'][day].sample(", 1),
    ('market_sample_seed', 'shared_resources_planning.py', "random_state=derive_random_seed(market_seed, 'selection', 'energy', year, str(day)),", 1),
    ('market_prob', 'shared_resources_planning.py', "planning_problem.prob_market_scenarios[year] = [(1 / planning_problem.num_market_scenarios)] * planning_problem.num_market_scenarios", 1),
    ('dso_seed', 'shared_resources_planning.py', 'distribution_network.random_seed = derive_random_seed(', 1),
    ('tso_seed', 'shared_resources_planning.py', 'transmission_network.random_seed = derive_random_seed(', 1),
    ('block_weight', 'shared_resources_planning.py', 'def _get_admm_block_weight(network_data, year, day):', 1),
    ('block_weight_disc', 'shared_resources_planning.py', 'annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - int(years[0])))', 1),
    ('net_pool', 'network_data.py', "random_seed=derive_random_seed(network_planning.random_seed, 'synthetic_profiles'),", 1),
    ('block_seed', 'network_data.py', "network_planning.network[year][day].random_seed = derive_random_seed(network_planning.random_seed, 'realization', int(year), str(day))", 1),
    ('oper_prob', 'network_data.py', "network_planning.network[year][day].prob_operation_scenarios = [(1 / network_planning.num_oper_scenarios)] * network_planning.num_oper_scenarios", 1),
    ('initial_year', 'network_data.py', "base_data['initial_year'] = list(network_planning.years)[0]", 1),
    ('load_sample', 'network.py', "pc = np.array(synthetic_profiles['consumption'][network.day][load_id]['pc'].sample(", 1),
    ('gen_sample', 'network.py', "pg = np.array(synthetic_profiles['generation'][network.day][gen_type]['pg'].sample(", 1),
    ('derive_seed', 'helper_functions.py', 'def derive_random_seed(base_seed, *labels):', 1),
    ('throughput', 'shared_energy_storage_data.py', 'avg_ch_dch += (num_days / 365.00) * (eff_ch * pch * dt + pdch * dt / eff_dch)', 1),
    ('soh_chain', 'shared_energy_storage_data.py', '== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))', 1),
    ('e_available', 'shared_energy_storage_data.py', 'model.available_e_capacity_unit.add(model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y])', 1),
    ('net_e_available', 'admm_persistent_workers.py', "model_yd.shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_capacity_year['e_available'] / s_base)", 1),
    ('soc_init', 'model_construction_helpers.py', 'soc_prev = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC', 1),
    ('soc_final', 'model_construction_helpers.py', 'final_soc = m.shared_es_e_rated_fixed[e] * ENERGY_STORAGE_RELATIVE_INIT_SOC', 1),
    ('soc_slack_bound', 'model_construction_helpers.py', 'slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE', 1),
    ('soc_slack_fraction', 'model_construction_helpers.py', 'ESS_DAY_BALANCE_SLACK_FRACTION = 0.05', 1),
    ('row18_price', 'model_construction_helpers.py', 'pibar = expected_market_price(network, p)', 1),
    ('row18_dev', 'model_construction_helpers.py', 'return (m.pg_adn[s_m, s_o, p] - m.expected_interface_pf_p[p]', 1),
    ('tn_loads_are_adn', 'model_construction_helpers.py', 'return network.is_transmission and load.bus in network.active_distribution_network_nodes', 1),
    ('case9_loads', 'data/SRP1/case9/case9_2025.json', '"loads"', 1),
]


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args, cwd=None):
    return subprocess.run(['git'] + args, cwd=cwd or REPO, capture_output=True, text=True, check=True).stdout


def _load(path):
    with open(path) as f:
        return json.load(f)


def _refuse_concurrent():
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me = os.getpid()
    others = []
    for ln in ps.splitlines():
        tok = ln.split()
        # an interpreter process whose own argv runs this script (not a shell whose command line mentions it)
        if (len(tok) >= 3 and int(tok[0]) != me and os.path.basename(tok[1]).startswith('python')
                and any(os.path.basename(t) == 'p515_s53_w177b2r_checks.py' for t in tok[2:4])
                and '--manifest' not in tok):
            others.append(ln)
    if others:
        raise SystemExit(f'refusing: another copy is running: {others}')


# ======================================================================================================================
#  A. identity
# ======================================================================================================================
def section_identity():
    head = _git(['rev-parse', '--short=7', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain'], cwd=CLONE)
    tex_dirty = [ln for ln in status.splitlines() if ln.strip().endswith('.tex')]
    with open(MAIN_TEX) as f:
        lines = f.read().split('\n')
    if lines and lines[-1] == '':
        lines = lines[:-1]
    old_text = _git(['show', f'{OLD_CLONE_HEAD}:main.tex'], cwd=CLONE)
    old_lines = old_text.split('\n')
    if old_lines and old_lines[-1] == '':
        old_lines = old_lines[:-1]
    ident = {
        'clone_head': head, 'clone_head_matches_declared': head == DECLARED_CLONE_HEAD,
        'clone_tex_dirty': tex_dirty, 'clone_tex_unmodified': not tex_dirty,
        'main_tex_sha256': _sha(MAIN_TEX), 'main_tex_sha256_matches_declared': _sha(MAIN_TEX) == DECLARED_MAIN_SHA256,
        'main_tex_lines': len(lines), 'main_tex_lines_match_declared': len(lines) == DECLARED_MAIN_LINES,
        'old_main_tex_sha256': hashlib.sha256(old_text.encode()).hexdigest(),
        'old_main_tex_sha256_matches': hashlib.sha256(old_text.encode()).hexdigest() == OLD_MAIN_SHA256,
        'letter_sha256': _sha(LETTER), 'letter_sha256_matches_declared': _sha(LETTER) == DECLARED_LETTER_SHA256,
        'letter_equals_manuscript_review_v2': _sha(LETTER) == _sha(LETTER_V2),
        'frozen_sha256_matches': _sha(FROZEN) == FROZEN_SHA,
        'prior_checks_sha256_matches': _sha(K_PRIOR) == K_PRIOR_SHA,
        'case_3x3_sha256_matches': _sha(CASE_3X3) == CASE_3X3_SHA,
        'srp1_case_sha256_matches': _sha(SRP1_CASE) == SRP1_CASE_SHA,
    }
    return ident, lines, old_lines


# ======================================================================================================================
#  B. diff and line map
# ======================================================================================================================
def _find(lines, frag):
    return [i + 1 for i, ln in enumerate(lines) if frag in ln]


def section_diff(lines, old_lines):
    raw = _git(['diff', '-U0', f'{OLD_CLONE_HEAD}', f'{DECLARED_CLONE_HEAD}', '--', 'main.tex'], cwd=CLONE)
    hunks = [ln.split(' @@')[0] + ' @@' for ln in raw.splitlines() if ln.startswith('@@')]
    old_range = old_lines[1139:1560]       # old l. 1140-1560
    new_range = lines[1171:1592]           # new l. 1172-1592
    region_identical = old_range == new_range
    rows = []
    for rid, old_ln, frag in PRIOR_ROWS:
        o, n = _find(old_lines, frag), _find(lines, frag)
        rows.append({'id': rid, 'fragment': frag, 'old_lines_51a6fb5': o, 'new_lines_e10c104': n,
                     'old_line_declared': old_ln, 'old_matches_declared': o == [old_ln],
                     'present_once_at_e10c104': len(n) == 1,
                     'offset': (n[0] - o[0]) if (len(n) == 1 and len(o) == 1) else None})
    in_scope_hunks = []
    for h in hunks:
        old_part = h.split()[1]  # -a[,b]
        a = int(old_part[1:].split(',')[0])
        if 1140 <= a <= 1563:
            in_scope_hunks.append(h)
    return {'hunks': hunks, 'n_hunks': len(hunks), 'hunks_in_old_1140_1563': in_scope_hunks,
            'old_1140_1560_equals_new_1172_1592': region_identical, 'prior_rows': rows}


def section_main_fragments(lines):
    out = []
    for fid, frag in MAIN_FRAGMENTS:
        hits = _find(lines, frag)
        out.append({'id': fid, 'fragment': frag, 'lines': hits, 'found_once': len(hits) == 1})
    abstract = '\n'.join(lines[_find(lines, '\\begin{abstract}')[0] - 1:_find(lines, '\\end{abstract}')[0]])
    app_e_start = _find(lines, '\\label{app:results}')[0]
    app_e = '\n'.join(lines[app_e_start - 1:])
    full = '\n'.join(lines)
    probes = []
    for pid, scope, needles in MAIN_PRESENCE_PROBES:
        text = abstract if scope == 'abstract' else (app_e if scope == 'appendixE' else full)
        probes.append({'id': pid, 'scope': scope, 'needles': needles,
                       'hits': {n: [i + 1 for i, ln in enumerate(text.split('\n')) if n in ln] for n in needles},
                       'any_present': any(n in text for n in needles)})
    return {'fragments': out, 'presence_probes': probes,
            'appendix_E_first_line': app_e_start}


def section_letter_fragments():
    with open(LETTER) as f:
        lines = f.read().split('\n')
    out = []
    for fid, frag in LETTER_FRAGMENTS:
        hits = _find(lines, frag)
        out.append({'id': fid, 'fragment': frag, 'lines': hits, 'found_once': len(hits) == 1})
    ref_unit = [i + 1 for i, ln in enumerate(lines) if 'reference unit' in ln or ln.rstrip().endswith('reference')]
    smallest = [i + 1 for i, ln in enumerate(lines) if 'smallest' in ln]
    return {'fragments': out, 'reference_unit_lines': ref_unit, 'smallest_lines': smallest}


# ======================================================================================================================
#  C. Section 3 opening CONFIRM
# ======================================================================================================================
def _cite(rel, needle, expected):
    with open(rel) as f:
        hits = [i + 1 for i, ln in enumerate(f) if needle in ln]
    try:
        blob_head = _git(['rev-parse', f'HEAD:{rel}']).strip()
        tracked = True
    except subprocess.CalledProcessError:
        blob_head, tracked = None, False
    blob_wt = _git(['hash-object', rel]).strip()
    return {'file': rel, 'needle': needle, 'lines': hits, 'expected_hits': expected,
            'hits_as_expected': (len(hits) == expected) if expected is not None else True,
            'blob_HEAD': blob_head, 'blob_worktree': blob_wt, 'tracked': tracked,
            'worktree_equals_HEAD': blob_head == blob_wt}


def _f64(v):
    import numpy as np
    return np.asarray(v, dtype=np.float64)


def _row_bytes(arr, i):
    a = _f64(arr)
    return a[i].tobytes() if a.ndim >= 1 and a.shape[0] > i else None


def section_confirm(sink):
    import p515_s47_market_spreads as W22       # imported unedited (W22 precedent)
    import p515_s53_3x3_selection as W52        # imported unedited (Recorder)
    W22.OUT_DIR = os.path.join(REPO, OUT_DIR)   # production's plots go under OUT_DIR/production_read
    blocks = W22.Blocks()
    blocks.install()
    recorder = W52.Recorder()
    recorder.install()
    try:
        srp_rd = W22.production_read(SRP1_CASE, 'srp1', sink)
        p1 = recorder.captured[-1]
        x3_rd = W22.production_read(CASE_3X3, 's53_3x3', sink)
        p3 = recorder.captured[-1]
    finally:
        recorder.uninstall()
        blocks.uninstall()
    res = {'reads': {k: {kk: rd[kk] for kk in ('case', 'case_sha256', 'num_market_scenarios', 'random_seed',
                                                'scenario_checksum', 'market_scenario_checksum',
                                                'stopped_at_declared_point')} | {'years': {str(y): n for y, n in rd['years'].items()}}
                     for k, rd in (('srp1', srp_rd), ('s53_3x3', x3_rd))},
           'blockers': {'targets': [f'{c.__name__}.{a}' for c, a in blocks.targets], 'blocked_calls': blocks.blocked_calls,
                        'stop_target': f'{blocks.stop_target[0].__name__}.{blocks.stop_target[1]}',
                        'stop_hits': blocks.stop_hits, 'declared_stops': DECLARED_STOPS}}
    years1, years3 = sorted(int(y) for y in p1.years), sorted(int(y) for y in p3.years)
    res['years_srp1'], res['years_3x3'] = years1, years3
    res['common_years'] = sorted(set(years1) & set(years3))
    res['srp1_years_without_3x3_counterpart'] = sorted(set(years1) - set(years3))
    # market
    fam = Counter()
    match_row = Counter()
    mism = []
    n_arrays = 0
    seeds = []
    y = COMMON_YEAR
    for d in p1.days:
        for name, a1, a3 in (('cost_energy_p', p1.cost_energy_p[y][d], p3.cost_energy_p[y][d]),
                             ('cost_flex', p1.cost_flex[y][d], p3.cost_flex[y][d])):
            n_arrays += 1
            fam[name] += 1
            r1 = _row_bytes(a1, 0)
            eq = [r1 == _row_bytes(a3, i) for i in range(3)]
            match_row[tuple(eq)] += 1
            if eq != [True, False, False]:
                mism.append({'block': f'market|{y}|{d}', 'family': name, 'eq_rows': eq,
                             'shape_srp1': list(_f64(a1).shape), 'shape_3x3': list(_f64(a3).shape)})
    holders = [('TSO', p1.transmission_network, p3.transmission_network)] + [
        (f'DSO{n}', p1.distribution_networks[n], p3.distribution_networks[n]) for n in sorted(p1.distribution_networks)]
    for tag, h1, h3 in holders:
        for d in h1.days:
            a, b = h1.network[y][d], h3.network[y][d]
            seeds.append({'block': f'{tag}|{y}|{d}', 'seed_srp1': a.random_seed, 'seed_3x3': b.random_seed,
                          'equal': a.random_seed == b.random_seed,
                          'n_oper_srp1': a.num_oper_scenarios, 'n_oper_3x3': b.num_oper_scenarios,
                          'pm_srp1': [float(v) for v in a.prob_market_scenarios],
                          'po_srp1': [float(v) for v in a.prob_operation_scenarios]})
            items = []
            loads3 = {ld.load_id: ld for ld in b.loads}
            for ld in a.loads:
                lb = loads3[ld.load_id]
                items += [('load.pd', ld.load_id, ld.pd, lb.pd), ('load.qd', ld.load_id, ld.qd, lb.qd),
                          ('load.flex_p_up', ld.load_id, ld.flexibility.active_power.upward,
                           lb.flexibility.active_power.upward),
                          ('load.flex_p_down', ld.load_id, ld.flexibility.active_power.downward,
                           lb.flexibility.active_power.downward)]
            gens3 = {g.gen_id: g for g in b.generators}
            for g in a.generators:
                gb = gens3[g.gen_id]
                items += [('gen.pg', g.gen_id, g.pg, gb.pg), ('gen.qg', g.gen_id, g.qg, gb.qg)]
            for name, ident, v1, v3 in items:
                n_arrays += 1
                fam[name] += 1
                x1 = _f64(v1)
                r1 = _row_bytes(v1, 0)
                eq = [r1 == _row_bytes(v3, i) for i in range(3)]
                # rows that are identical across scenarios (e.g. zero non-RES gen output) cannot identify an index
                degenerate = len({_row_bytes(v3, i) for i in range(3)}) < 3
                key = ('degenerate_rows_equal' if degenerate else tuple(eq))
                match_row[key] += 1
                if x1.shape[0] != 1 or not eq[0] or (not degenerate and eq != [True, False, False]):
                    mism.append({'block': f'{tag}|{y}|{d}', 'family': name, 'id': ident, 'eq_rows': eq,
                                 'degenerate': degenerate, 'shape_srp1': list(x1.shape),
                                 'shape_3x3': list(_f64(v3).shape)})
    res['comparison_2025'] = {
        'definition': ('for every 2025 block of SRP1 (market prices per day; TSO + 3 DSOs x 4 days: per load pd, qd, '
                       'flexibility up / down; per generator pg, qg): the single SRP1 row, as float64 bytes, against '
                       'each of the three scenario rows of the same array in the 3 x 3 read'),
        'n_arrays_compared': n_arrays, 'arrays_per_family': dict(sorted(fam.items())),
        'pattern_counts': {str(k): v for k, v in match_row.items()},
        'n_mismatches': len(mism), 'mismatches_first': mism[:20],
        'srp1_equals_3x3_row1_everywhere': not mism,
        'block_seeds_equal_all': all(s['equal'] for s in seeds), 'block_seeds': seeds}
    # 2030 / 2035 blocks: SRP1-only years (each its own first draw); record that the 3 x 3 has no such year
    res['note_other_years'] = ('SRP1 2030 / 2035 are not years of the 3 x 3 instance (2025 / 2028 / 2031 / 2034 / '
                               '2037); each SRP1 block draws its own row with its year-dependent seed')
    del p1, p3
    return res


def section_prefix_records():
    w89 = _load(W89_RECORD)
    w52 = _load(W52_SELECTION)
    w22 = _load(W22_SPREADS)
    pv = w89['prefix_verification']
    return {
        'w89_instance_record': {'path': W89_RECORD, 'sha256': _sha(W89_RECORD),
                                'n_arrays_compared': pv['n_arrays_compared'], 'n_mismatches': pv['n_mismatches'],
                                'prefix_holds': pv['prefix_holds'],
                                'realized_subset_market': pv['realized_subset_market'],
                                'realized_subset_operation': pv['realized_subset_operation'],
                                'prefix_draw': {k: w89['prefix_draw'][k] for k in ('market', 'operation', 'R_recorded')}},
        'w52_prefix_check': {'path': W52_SELECTION, 'sha256': _sha(W52_SELECTION),
                             **{k: w52['prefix_check'][k] for k in ('market_seeds_checked', 'market_prefix_true',
                                                                   'operation_seeds_checked', 'operation_prefix_true')}},
        'w22_correspondence_finding': {'path': W22_SPREADS, 'sha256': _sha(W22_SPREADS),
                                       'finding': w22['correspondence_finding'],
                                       'sample_prefix_market_n1_of_n5': {d: v['n1_is_prefix_of_n5'] for d, v in
                                                                         w22['sample_prefix_check']['per_day'].items()}},
    }


# ======================================================================================================================
#  D. record checks for the letter
# ======================================================================================================================
def section_affinity(fz):
    out = {}
    claims = {c['claim_id']: c for c in fz['tables']['claims']}
    for lad, ids, cost_per_mwh_key in (('n7_4h', [f'B:n7_4h_e{e}' for e in range(1, 6)], 1),
                                       ('n7_2h', [f'B:n7_2h_e{e}' for e in range(1, 6)], 1)):
        es = [float(i.split('_e')[-1]) for i in ids]
        V = [claims[i]['I_other'] - claims[i]['d_gross'] for i in ids]
        I = [claims[i]['I_other'] for i in ids]
        n = len(es)
        mx, my = sum(es) / n, sum(V) / n
        sxx = sum((x - mx) ** 2 for x in es)
        slope = sum((x - mx) * (v - my) for x, v in zip(es, V)) / sxx
        icpt = my - slope * mx
        resid = [v - (icpt + slope * x) for x, v in zip(es, V)]
        out[lad] = {'E_MWh': es, 'V_eur': V, 'I_eur': I, 'slope_eur_per_MWh': slope, 'intercept_eur': icpt,
                    'residuals_eur': resid, 'max_abs_residual_eur': max(abs(r) for r in resid),
                    'max_abs_residual_over_2tau': max(abs(r) for r in resid) / TWO_TAU,
                    'increments_eur': [V[k + 1] - V[k] for k in range(n - 1)],
                    'investment_per_MWh_eur': I[0] / es[0],
                    'slope_below_cost': slope < I[0] / es[0]}
    return out


def section_uncertified(fz):
    cells = fz['tables']['cells']
    lab = Counter(v.get('label') for v in cells.values() if v.get('status') == 'uncertified')
    t2 = list(csv.DictReader(open(EXPORT_T2)))
    t2c = Counter((r['status'], r['cause (uncertified)']) for r in t2)
    t9 = list(csv.DictReader(open(EXPORT_T9)))
    return {'frozen_cells_uncertified_labels': dict(lab),
            'export_T2_rows': len(t2), 'export_T2_status_cause': {f'{a} | {b}': v for (a, b), v in t2c.items()},
            'export_T9_rows': [{k: r[k] for k in ('cell', 'cause', 't_sum at cap (k€)', 'pf_primal last')} for r in t9]}


def section_record_statuses():
    fs = sorted(glob.glob(os.path.join('data', 'SRP1', 'Results', '**', 'evaluation_record.json'), recursive=True))
    st, bar, err = Counter(), Counter(), 0
    infeasible_words = []
    for f in fs:
        try:
            d = _load(f)
        except Exception:
            err += 1
            continue
        st[str(d.get('status'))] += 1
        bar[f"{d.get('barrier')} | {d.get('barrier_cause')}"] += 1
        blob = json.dumps({k: d.get(k) for k in ('status', 'barrier_cause', 'stopped_by_trajectory')}).lower()
        if 'infeasib' in blob:
            infeasible_words.append(f)
    return {'n_records': len(fs), 'n_read_errors': err, 'status_counts': dict(st), 'barrier_counts': dict(bar),
            'records_with_infeasib_in_status_or_barrier_cause': infeasible_words}


# ======================================================================================================================
def main():
    if '--manifest' in sys.argv:
        return write_manifest()
    _refuse_concurrent()
    for p in (OUT_JSON, OUT_MANIFEST, STDOUT_CAPTURE, os.path.join(OUT_DIR, 'production_read')):
        if os.path.exists(p):
            raise SystemExit(f'refusing to overwrite an existing output: {p}')
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    repo_head = _git(['rev-parse', 'HEAD']).strip()
    ident, lines, old_lines = section_identity()
    hard = ['clone_head_matches_declared', 'clone_tex_unmodified', 'main_tex_sha256_matches_declared',
            'main_tex_lines_match_declared', 'letter_sha256_matches_declared', 'frozen_sha256_matches',
            'prior_checks_sha256_matches', 'case_3x3_sha256_matches', 'srp1_case_sha256_matches']
    bad = [k for k in hard if not ident[k]]
    if bad:
        raise SystemExit(f'refusing: identity check failed: {bad}')
    fz = _load(FROZEN)
    result = {
        'stage': "P5.15 W177b-2' (Addendum 76 Part 2 at e10c104 + Amendment: Section 3 opening CONFIRM, letter claims)",
        'objective_convention': fz['objective_convention'],
        'started_utc': started, 'repo_head_at_run': repo_head,
        'identity': ident,
        'diff': section_diff(lines, old_lines),
        'main_fragments': section_main_fragments(lines),
        'letter': section_letter_fragments(),
        'code_citations': {cid: _cite(rel, needle, exp) for cid, rel, needle, exp in CODE_CITES},
        'prefix_records': section_prefix_records(),
        'affinity': section_affinity(fz),
        'uncertified': section_uncertified(fz),
        'record_statuses': section_record_statuses(),
        'w153_discount_method': {'path': W153_DISCOUNT, 'sha256': _sha(W153_DISCOUNT)},
    }
    print('[read] production data reads (SRP1, 3 x 3) ...', flush=True)
    with open(STDOUT_CAPTURE, 'w') as sink:
        result['confirm'] = section_confirm(sink)
    post_status = _git(['status', '--porcelain', '--untracked-files=no'])
    result['post_git_status_tracked_changes'] = post_status.splitlines()
    c = result['confirm']
    integrity = {
        **{k: ident[k] for k in hard},
        'old_main_tex_sha256_matches': ident['old_main_tex_sha256_matches'],
        'letter_equals_manuscript_review_v2': ident['letter_equals_manuscript_review_v2'],
        'every_prior_row_old_line_as_declared': all(r['old_matches_declared'] for r in result['diff']['prior_rows']),
        'every_letter_fragment_found_once': all(f['found_once'] for f in result['letter']['fragments']),
        'every_main_fragment_found_once': all(f['found_once'] for f in result['main_fragments']['fragments']),
        'every_code_citation_as_expected': all(v['hits_as_expected'] for v in result['code_citations'].values()),
        'every_cited_file_equals_HEAD': all(v['worktree_equals_HEAD'] for v in result['code_citations'].values()),
        'reads_stopped_at_declared_point': c['blockers']['stop_hits'] == DECLARED_STOPS,
        'no_model_construction_call': not c['blockers']['blocked_calls'],
        'no_record_read_error': result['record_statuses']['n_read_errors'] == 0,
        'frozen_unchanged_at_end': _sha(FROZEN) == FROZEN_SHA,
        'main_tex_unchanged_at_end': _sha(MAIN_TEX) == DECLARED_MAIN_SHA256,
        'no_tracked_file_changed': not post_status.strip(),
    }
    result['integrity'] = integrity
    gfail = GUARD.verify(0)
    result['guards'] = {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts,
                                                'verify_0_failures': gfail},
                        'pickle': dict(PICKLE_COUNTS)}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True, default=str)
    ok = all(integrity.values()) and not gfail and PICKLE_COUNTS == {'load': 0, 'loads': 0}
    cc = c['comparison_2025']
    print(json.dumps({'integrity': integrity, 'guards': result['guards'], 'ok': ok}, indent=1))
    print(f"diff: {result['diff']['n_hunks']} hunks; in old 1140-1563: {result['diff']['hunks_in_old_1140_1563']}; "
          f"old 1140-1560 == new 1172-1592: {result['diff']['old_1140_1560_equals_new_1172_1592']}")
    for r in result['diff']['prior_rows']:
        print(f"  {r['id']}: old {r['old_lines_51a6fb5']} -> new {r['new_lines_e10c104']}")
    print(f"confirm: years srp1 {c['years_srp1']}, 3x3 {c['years_3x3']}; 2025 arrays {cc['n_arrays_compared']} "
          f"{cc['arrays_per_family']}; patterns {cc['pattern_counts']}; mismatches {cc['n_mismatches']}; "
          f"seeds equal {cc['block_seeds_equal_all']}")
    for lad, a in result['affinity'].items():
        print(f"affinity {lad}: slope {a['slope_eur_per_MWh']:.1f} vs I/MWh {a['investment_per_MWh_eur']:.1f}; "
              f"max |resid| {a['max_abs_residual_eur']:.1f} ({a['max_abs_residual_over_2tau']:.2f} x 2tau)")
    print('main presence probes:', {p['id']: p['any_present'] for p in result['main_fragments']['presence_probes']})
    print('record statuses:', result['record_statuses']['status_counts'],
          'infeasib:', len(result['record_statuses']['records_with_infeasib_in_status_or_barrier_cause']))
    print('exit', 0 if ok else 1)
    return 0 if ok else 1


def write_manifest():
    if os.path.exists(OUT_MANIFEST):
        raise SystemExit('refusing to overwrite the manifest')
    script = os.path.basename(__file__)
    inputs = [MAIN_TEX, LETTER, LETTER_V2, S3_FIX, FROZEN, K_PRIOR, EXPORT_T2, EXPORT_T9, W89_RECORD, W52_SELECTION,
              W22_SPREADS, W153_DISCOUNT, SRP1_CASE, CASE_3X3, 'p515_s47_market_spreads.py',
              'p515_s53_3x3_selection.py', 'p513_solve_profile_guard.py']
    inputs += sorted({rel for _cid, rel, _n, _e in CODE_CITES})
    produced = {}
    for dirpath, _d, files in os.walk(OUT_DIR):
        for fn in sorted(files):
            p = os.path.join(dirpath, fn)
            if p != OUT_MANIFEST:
                produced[p] = _sha(p)
    manifest = {'outputs': produced, 'script': {script: _sha(script)},
                'inputs': {p: _sha(p) for p in sorted(set(inputs))}}
    with open(OUT_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    print(f'manifest: {len(produced)} outputs, {len(manifest["inputs"])} inputs -> {OUT_MANIFEST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
