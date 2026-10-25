"""
P5.15 Addendum 82 v2 order, Planner task W182d (W182 items A7 and A8) -- a LISTING of claims at Overleaf edd7a63,
not a rewrite. ZERO SOLVES, NO MODEL, NO PICKLE. Read-only on the clone (every clone file via `git show edd7a63:`).

A7  Every sentence of the abstract, the highlights (main.tex and the standalone highlights.tex; the staging copy
    manuscript_review/highlights_keywords_v2.tex is compared, not re-listed), the contributions (the "To address this
    gap" paragraph and the contribution list), Section 4.5, Section 5, the response letter (v9, the clone's
    response_to_reviewers_draft.tex, reviewer quotes excluded) and the cover letter that claims or implies
      (i)   a multi-scenario search,
      (ii)  a value of stochastic selection,
      (iii) an installation-timing preference,
      (iv)  that "certification" bounds error or resolves a ratio (iv-b), or certifies what it did not (iv-s),
    each with the manuscript passage or code line that limits it. Every quoted sentence and every evidence passage is
    asserted at its line. Coverage: every in-scope sentence fragment that matches the keyword pattern KW is either a
    curated entry (listed / borderline / other) or an exclusion with a stated reason -- the run refuses otherwise.
A8  (1) the budget arithmetic of every evaluated plan of Table E.21 (tab:appE_evaluations) from the case and cost
        files through the production readers and the production first-stage check, cross-checked against the W2
        table and Table 1; every place an out-of-budget plan appears, and whether it is labelled as outside the
        admissible set; (2) every occurrence of 13.9 % and whether it is called a saving; (3) the abstract's
        technology and cost-uncertainty sentences against what the case models.

Guards: SolveProfileGuard(permitted=()) installed before any project import and verified at exactly 0; pickle.load /
loads blocked and counted; production modules imported ONLY for the three readers and the first-stage check named in
PRODUCTION_CALLS (no model is built); JSON written only through gate_result_io. Refuses to run unless the output
directory exists and holds nothing but launch.log, and when another copy of this script is running.

Command (repo root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S53/w182d_claims && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w182d_claims.py \\
      > data/SRP1/Results/P515S53/w182d_claims/launch.log 2>&1
Optional: --out-dir <dir> (scratch trials only); --list-uncovered (prints uncovered fragments, writes nothing).
"""
import pickle
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W182d claims listing (never solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W182d: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W182d: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import argparse  # noqa: E402
import copy  # noqa: E402
import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import gate_result_io as GRIO  # noqa: E402

# production: readers and the closed-form first-stage check only (no model is built, nothing is solved)
import shared_energy_storage_data as SED  # noqa: E402
import shared_energy_storage_parameters as SEP  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402

PRODUCTION_CALLS = ('shared_energy_storage_parameters.SharedEnergyStorageParameters.read_parameters_from_file',
                    'shared_energy_storage_data._get_operational_scenarios_info_from_excel_file',
                    'shared_energy_storage_data._get_investment_costs_from_excel_file',
                    'shared_resources_planning._check_candidate_first_stage_feasibility')

SCRIPT = os.path.basename(os.path.abspath(__file__))
DEFAULT_OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w182d_claims')

# ------------------------------------------------------------------------------------------------------- identity
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
DECLARED_HEAD = 'edd7a63'
MAIN_SHA_PREFIX = 'b08a3365607e56f2'           # audit l. 7: "main.tex content fingerprint is b08a3365607e56f2"
CLONE_FILES = ('main.tex', 'highlights.tex', 'cover_letter.tex', 'response_to_reviewers_draft.tex')
REPO_TEX = {'highlights_v2': os.path.join('manuscript_review', 'highlights_keywords_v2.tex'),
            'submitted_highlights': os.path.join('manuscript_submitted', 'highlights.tex'),
            'submitted_main': os.path.join('manuscript_submitted', 'main.tex'),
            'submitted_cover': os.path.join('manuscript_submitted', 'cover_letter.tex')}
AUDIT = os.path.join('manuscript_review', 'manuscript_critical_review_2026-10-25.md')

# ------------------------------------------------------------------------------------------------------- case files
CASE_JSON = os.path.join('data', 'SRP1', 'SRP1.json')
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
COST_XLSX = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx')
COST_XLSX_SHA = 'e17bd5887e1d0738005ae17c3144593527081c9a0776e19cfaa50aafefe39cd6'   # p515_s47_phase_b_record COST_FILE
W2_TABLE = os.path.join('data', 'SRP1', 'Results', 'P515S45', 'investment_cost', 'investment_cost_results.json')
W2_SHA = '28152120f5c7acc57655d40871f764a5e797b93428f5a9f87b7eed55fbe4790b'        # p515_s47_phase_b_record
S51_LADDER = os.path.join('data', 'SRP1', 'Results', 'P515S51', 'campaign_s51_f2_ladder', 'campaign_results.json')
FROZEN = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w160_step6_frozen', 'frozen_step6_tables_v1_590088fe.json')
FROZEN_SHA = '590088fe6b364c265c97998c5491edea7ca70baad0dbe2ba5150273656d9b6f4'
CODE_SRC = {'shared_energy_storage_data.py': None, 'shared_resources_planning.py': None,
            'p515_s47_phase_b_record.py': None}
FIG_FILES = ('fig1_breakeven.pdf', 'fig2_flexibility_ladder.pdf', 'fig_lattice_case.pdf')

# ------------------------------------------------------------------------------------------------------- A7 scope
KW = re.compile(r'stochastic|uncertain|scenario|trajector|probabilit|expected|timing|installation year|investment year|'
                r'install|staged|multi-year|defer|certif|resolution|resolv|determinat|precision|bound|break-even|'
                r'breaks even|guarantee|optimal', re.I)

# Evidence passages (main.tex unless a file is given): key -> (file, line, exact substring)
EV = {
    'search_single_scenario': ('main.tex', 1203, 'Planning searches use the single-scenario configuration and consider '
                               'one installation per selected node, with a common installation year.'),
    'ms_search_not_repeated': ('main.tex', 1636, 'The planning search was not repeated.'),
    'ms_not_certified': ('main.tex', 1636, 'were not certified under the rule of Subsection'),
    'scope_timing': ('main.tex', 1781, 'Node-specific timing and staged expansion were not explored, although the '
                     'formulation allows both.'),
    'scope_fixed_plans': ('main.tex', 1781, 'The degradation sensitivities and multi-scenario analysis evaluated fixed '
                          'investment plans without repeating the planning search.'),
    'expected_objective': ('main.tex', 541, 'The planning search seeks to minimise the expected discounted investment '
                           'cost plus the evaluated expected discounted operating cost of the investment plan:'),
    'expected_budget': ('main.tex', 690, 'The constraint applies to the probability-weighted expenditure over the '
                        'investment-cost trajectories.'),
    'code_budget_row_expected': ('shared_energy_storage_data.py', 376,
                                 'investment_cost_total += annualization * omega_m * model.es_s_investment[e, y] * c_inv_s'),
    'code_search_expected_coeff': ('p515_s47_phase_b_record.py', 385,
                                   'return sum(self.c_p[y] * GRANULE_P_MVA * zp + self.c_e[y] * GRANULE_E_MWH * ze '
                                   'for zp, ze in self.node_pairs(z))'),
    'cert_definition': ('main.tex', 2003, 'The term certified denotes satisfaction of these criteria; it does not '
                        'establish global optimality or a rigorous bound on subsequent objective movement.'),
    'cert_no_bound': ('main.tex', 1777, 'Certification therefore does not provide a guaranteed bound on the remaining '
                      'objective movement'),
    'within_resolution_defined': ('main.tex', 2037, 'Comparisons that do not meet the applicable criterion are reported '
                                  'as within numerical resolution.'),
    'ten_uncertified': ('main.tex', 1739, '32 of 42 recourse evaluations satisfied the numerical acceptance criteria'),
    'search_41_residual': ('main.tex', 1760, 'The three planning searches performed 41 new recourse evaluations using '
                           'the residual-based stopping rule.'),
    'returned_plan_uncertified': ('main.tex', 1507, 'Its operational evaluation remains uncertified because of the '
                                  'interface-consensus gap.'),
    'tau_definition': ('main.tex', 1993, '\\tau = \\frac{\\delta_R\\mathcal{V}}{4}'),
    'no_ageing_row': ('main.tex', 2676, '46 & No ageing & 1 / 54 & $-$4.1 & 12.9 & 0.32 & w.r.'),
    'salvage_post_evaluation': ('main.tex', 1363, 'The credit is calculated after the operational evaluation and does '
                                'not affect dispatch or investment selection.'),
    'year_neither_preferable': ('main.tex', 1677, 'so neither year can be identified as preferable on this basis.'),
    'ms_not_certified_interval': ('main.tex', 1640, 'is not a certified interval.'),
    'no_global_optimality': ('main.tex', 1739, 'These checks assess numerical stability and do not establish global '
                             'optimality.'),
    'datasheet_row': ('main.tex', 2679, '49 & Datasheet 8\\,000 cycles & 1 / 57 & $-$45.2'),
    'baseline_row': ('main.tex', 2678, '48 & Baseline$^{t}$ & 1 / 56 & $-$64.4'),
    'search_only_six': ('main.tex', 2808, 'were evaluated by the search only, on an earlier state of the code and '
                        'stopped by the production exit'),
    'floor_row': ('main.tex', 2688, '58 & Baseline: 0.50 $-$ 0.70 floor & 2 / 60 & 4.9 & 13.1 & 0.38 & w.r.'),
}

# Curated entries. status: listed (counts toward the A7 prediction) / borderline / other (outside the four categories)
# Each: id, file, lines (first, last), quote (lines joined by '\n'), categories, status, why, evidence keys.
A7 = [
    # ---- main.tex abstract
    {'id': 'M-ABS-1', 'file': 'main.tex', 'lines': (145, 145), 'region': 'abstract', 'status': 'listed',
     'quote': 'This paper proposes a two-stage stochastic framework for shared battery ESS siting, sizing and '
              'installation timing under investment-cost uncertainty.',
     'cat': ['ii', 'iii'], 'strength': 'implied',
     'why': 'Investment-cost uncertainty enters only as probability-weighted coefficients of I(x) and of the expected-'
            'budget row (l. 541, 690; code: the master row and the search lattice use one expected coefficient per '
            'year), so with one common plan the selection is that of the expected-cost problem and no value of '
            'stochastic selection is demonstrated or measured. "Installation timing": the searches used one cohort per '
            'node and a common year (l. 1203); node-specific timing and staging not explored (l. 1781).',
     'ev': ['expected_objective', 'expected_budget', 'code_budget_row_expected', 'code_search_expected_coeff',
            'search_single_scenario', 'scope_timing']},
    {'id': 'M-ABS-2', 'file': 'main.tex', 'lines': (145, 145), 'region': 'abstract', 'status': 'listed',
     'quote': 'A derivative-free search explores discrete investment plans, and an Alternating Direction Method of '
              'Multipliers (ADMM)-based coordination procedure evaluates their operation under load, '
              'renewable-generation and market-price uncertainty.',
     'cat': ['i'], 'strength': 'implied',
     'why': 'The plans the search explored were evaluated in the single-scenario configuration (one market x one '
            'load-RES scenario, l. 1203); only two fixed plans were evaluated on the 3x3 instance and the search was '
            'not repeated there (l. 1636, 1781).',
     'ev': ['search_single_scenario', 'ms_search_not_repeated', 'scope_fixed_plans']},
    # ---- main.tex highlights
    {'id': 'M-HL-1', 'file': 'main.tex', 'lines': (163, 163), 'region': 'highlights (main.tex)', 'status': 'listed',
     'quote': 'A stochastic framework plans shared storage at transmission--distribution interfaces',
     'cat': ['i', 'ii'], 'strength': 'implied',
     'why': '"Plans" under a stochastic framework: the planning searches ran single-scenario (l. 1203) and the cost '
            'uncertainty acts through expected coefficients only (l. 690).',
     'ev': ['search_single_scenario', 'expected_budget']},
    # ---- main.tex contributions
    {'id': 'M-CON-1', 'file': 'main.tex', 'lines': (387, 387), 'region': 'contributions', 'status': 'listed',
     'quote': 'A two-stage stochastic planning framework for shared ESS deployment at TN--DN interface nodes that '
              'determines a multi-year investment plan and evaluates it against scenario-dependent coordinated '
              'TSO--DSO operation.',
     'cat': ['i', 'iii'], 'strength': 'implied',
     'why': 'The plan the framework "determines" (the search incumbent) was evaluated single-scenario; scenario-'
            'dependent evaluation exists only for two fixed plans (l. 1203, 1636). "Multi-year plan": one cohort, '
            'common year (l. 1781).',
     'ev': ['search_single_scenario', 'ms_search_not_repeated', 'scope_timing']},
    {'id': 'M-CON-3', 'file': 'main.tex', 'lines': (389, 389), 'region': 'contributions', 'status': 'borderline',
     'quote': 'together with explicit numerical acceptance criteria for assessing the resolution of planning '
              'comparisons.',
     'cat': ['iv-b'], 'strength': 'implied (read with l. 368: "its stopping error is seldom quantified")',
     'why': '"Resolution" is a defined term (l. 2037) and the bullet does not assert a bound; read with l. 368 it '
            'implies the paper quantifies stopping error, which l. 2003 and l. 1777 deny.',
     'ev': ['within_resolution_defined', 'cert_definition', 'cert_no_bound']},
    {'id': 'M-CON-4', 'file': 'main.tex', 'lines': (390, 390), 'region': 'contributions', 'status': 'other',
     'quote': 'and the role of degradation in the investment decision.',
     'cat': ['other: audit 3.1 "degradation matters" row'], 'strength': 'explicit',
     'why': 'Degradation results are fixed-plan re-evaluations (l. 1781); no degradation-neutral planning search was run.',
     'ev': ['scope_fixed_plans']},
    # ---- main.tex Section 4.5
    {'id': 'M-45-1', 'file': 'main.tex', 'lines': (1636, 1636), 'region': 'Section 4.5', 'status': 'borderline',
     'quote': 'The resulting shortfall of 81.3~k\\euro{} exceeds the reported numerical resolution of 18.0~k\\euro{}.',
     'cat': ['iv-b'], 'strength': 'implied',
     'why': 'A "numerical resolution" is quoted for two evaluations that the same paragraph says were not certified; '
            'l. 2037 defines resolution through the comparison rules of Appendix A.5. The 18.0 figure\'s rule is not '
            'stated in Section 4.5 (not checked here).',
     'ev': ['ms_not_certified', 'within_resolution_defined']},
    {'id': 'M-45-2', 'file': 'main.tex', 'lines': (1640, 1640), 'region': 'Section 4.5', 'status': 'borderline',
     'quote': 'This range reflects estimated remaining movement in the no-storage evaluation and is not a certified '
              'interval.',
     'cat': ['iv-b'], 'strength': 'implied (by contrast)',
     'why': '"Not a certified interval" implies that certification would yield an interval on the value ratio; '
            'l. 2003 says certification gives no rigorous bound.',
     'ev': ['cert_definition']},
    # ---- main.tex Section 5
    {'id': 'M-5-1', 'file': 'main.tex', 'lines': (1795, 1795), 'region': 'Section 5', 'status': 'borderline',
     'quote': 'The two-stage stochastic formulation links investment decisions with coordinated TSO--DSO operation '
              'under uncertainty, while battery degradation couples storage utilisation with available capacity over '
              'the planning horizon.',
     'cat': ['i'], 'strength': 'formulation statement',
     'why': 'True of the formulation; in the case the investment decisions were searched single-scenario (l. 1203).',
     'ev': ['search_single_scenario']},
    # ---- standalone highlights.tex
    {'id': 'H-3', 'file': 'highlights.tex', 'lines': (12, 12), 'region': 'highlights (standalone)', 'status': 'listed',
     'quote': 'Every recourse evaluation is certified; every planning difference has a resolution',
     'cat': ['iv-s', 'iv-b'], 'strength': 'explicit',
     'why': '10 of 42 campaign evaluations uncertified (l. 1739); the 41 search evaluations ran under the residual-'
            'based exit (l. 1760); the 3x3 evaluations and the returned 2x plan are uncertified (l. 1636, 1507); '
            'certification gives no rigorous bound (l. 2003, 1777).',
     'ev': ['ten_uncertified', 'search_41_residual', 'ms_not_certified', 'returned_plan_uncertified',
            'cert_definition', 'cert_no_bound']},
    {'id': 'H-4', 'file': 'highlights.tex', 'lines': (13, 13), 'region': 'highlights (standalone)', 'status': 'listed',
     'quote': 'Without ageing the reference storage unit breaks even; with ageing it loses',
     'cat': ['iv-b'], 'strength': 'explicit',
     'why': 'Converts "within numerical resolution" (row 46, w.r., 0.32x) into a demonstrated break-even; a w.r. '
            'verdict is the failure of the determinacy rule (l. 2037), not a bound placing the margin at zero.',
     'ev': ['no_ageing_row', 'within_resolution_defined', 'cert_definition']},
    # ---- response letter v9
    {'id': 'L-01', 'file': 'response_to_reviewers_draft.tex', 'lines': (42, 43), 'region': 'letter, opening',
     'status': 'listed',
     'quote': 'in which each candidate plan is evaluated by the\ncoordinated TSO--DSO recourse under an explicit '
              'stopping and certification rule',
     'cat': ['iv-s'], 'strength': 'explicit',
     'why': 'Search candidates ran under the production (residual) exit, not the certification rule (l. 1760; E.6).',
     'ev': ['search_41_residual']},
    {'id': 'L-00', 'file': 'response_to_reviewers_draft.tex', 'lines': (48, 49), 'region': 'letter, opening',
     'status': 'borderline',
     'quote': 'loses 64.4~k\\euro{} over the horizon, determinately, as does every other plan\nevaluated',
     'cat': ['iv-s'], 'strength': 'implied',
     'why': 'Six of the fourteen neighbours of x = 0 were evaluated by the search only, under the production exit, '
            'and carry no determinacy verdict under the rule (l. 2808, 1760); every certified baseline comparison of '
            'Table E.20 (a) is determinate.',
     'ev': ['search_only_six', 'search_41_residual']},
    {'id': 'L-02', 'file': 'response_to_reviewers_draft.tex', 'lines': (54, 55), 'region': 'letter, opening',
     'status': 'borderline',
     'quote': 'and we have tried to make every number it\nreports traceable and to state the resolution at which each '
              'difference is claimed.',
     'cat': ['iv-b'], 'strength': 'implied',
     'why': '"Resolution" as a property of each claimed difference; l. 2003 / 1777 state it is not a bound.',
     'ev': ['cert_definition', 'cert_no_bound']},
    {'id': 'L-02b', 'file': 'response_to_reviewers_draft.tex', 'lines': (81, 81), 'region': 'letter, R1.2',
     'status': 'borderline',
     'quote': 'and the resolution at which the difference is claimed (Section~2.2.7, the opening of Section~4, and the '
              'captions and notes of Tables~7--15).',
     'cat': ['iv-b'], 'strength': 'implied', 'why': 'As L-02.', 'ev': ['cert_definition', 'cert_no_bound']},
    {'id': 'L-03', 'file': 'response_to_reviewers_draft.tex', 'lines': (58, 58), 'region': 'letter, opening',
     'status': 'borderline',
     'quote': 'under the no-ageing calibration the unit is at break-even within resolution and the plan is '
              'indeterminate at the precision of this study.',
     'cat': ['iv-b'], 'strength': 'implied ("precision of this study")',
     'why': 'Row 46 is w.r.; a "precision" of the study is not established by the acceptance rule (l. 2003).',
     'ev': ['no_ageing_row', 'cert_definition']},
    {'id': 'L-04', 'file': 'response_to_reviewers_draft.tex', 'lines': (78, 78), 'region': 'letter, R1.2',
     'status': 'listed',
     'quote': 'on the latter, run on a finer five-block horizon, the reference unit again loses determinately to the '
              'plan without storage, and its value is 0.909--0.934 of its single-scenario value, against 0.933 '
              'predicted from the mean-profile price spread (Section~4.5).',
     'cat': ['iv-s', 'iv-b'], 'strength': 'explicit',
     'why': '"Determinately" for two evaluations Section 4.5 calls not certified; the ratio range is set against a '
            'three-decimal prediction as if resolved; the 0.933 prediction is not in main.tex (searched).',
     'ev': ['ms_not_certified', 'ms_not_certified_interval']},
    {'id': 'L-05', 'file': 'response_to_reviewers_draft.tex', 'lines': (102, 102), 'region': 'letter, R1.5',
     'status': 'listed',
     'quote': 'a derivative-free search driving certified recourse evaluations, with no cuts.',
     'cat': ['iv-s'], 'strength': 'explicit',
     'why': 'The search drove residual-exit evaluations (l. 1760); certification was applied afterwards to reported '
            'comparisons.',
     'ev': ['search_41_residual']},
    {'id': 'L-06', 'file': 'response_to_reviewers_draft.tex', 'lines': (114, 115), 'region': 'letter, R2.1',
     'status': 'borderline',
     'quote': 'stage selects one scenario-independent multi-year plan; the second stage evaluates it under\n'
              'scenario-dependent coordinated operation.',
     'cat': ['i'], 'strength': 'formulation statement',
     'why': 'True of the formulation; the case search evaluated single-scenario (l. 1203).',
     'ev': ['search_single_scenario']},
    {'id': 'L-07', 'file': 'response_to_reviewers_draft.tex', 'lines': (137, 137), 'region': 'letter, R2.3',
     'status': 'listed',
     'quote': 'The method claims no optimality certificate; what it certifies is each evaluation (Section~2.2.7), and '
              'every planning statement in the paper is a pairwise comparison of certified evaluations with a stated '
              'determinacy threshold.',
     'cat': ['iv-s'], 'strength': 'explicit',
     'why': 'Ten campaign evaluations, the 41 search evaluations, the 3x3 pair and the returned 2x plan are not '
            'certified; Section 4.2 / 4.5 statements rest on them (l. 1739, 1760, 1636, 1507).',
     'ev': ['ten_uncertified', 'search_41_residual', 'ms_not_certified', 'returned_plan_uncertified']},
    {'id': 'L-08', 'file': 'response_to_reviewers_draft.tex', 'lines': (168, 169), 'region': 'letter, R2.5',
     'status': 'listed',
     'quote': 'On sensitivity to the penalty schedule: the certification bounds the\nstopping error of each '
              'evaluation, not the basin the nonconvex recourse converges to',
     'cat': ['iv-b'], 'strength': 'explicit',
     'why': 'main.tex says the opposite: no rigorous bound on subsequent objective movement (l. 2003, 1777).',
     'ev': ['cert_definition', 'cert_no_bound']},
    {'id': 'L-09', 'file': 'response_to_reviewers_draft.tex', 'lines': (229, 230), 'region': 'letter, R3.2',
     'status': 'listed',
     'quote': 'what is certified is each recourse evaluation, and each reported difference carries the\nthreshold '
              'above which it is called determinate (Section~2.2.7).',
     'cat': ['iv-s'], 'strength': 'explicit',
     'why': 'Not every recourse evaluation is certified (l. 1739, 1760, 1636).',
     'ev': ['ten_uncertified', 'search_41_residual', 'ms_not_certified']},
    {'id': 'L-10', 'file': 'response_to_reviewers_draft.tex', 'lines': (230, 232), 'region': 'letter, R3.2',
     'status': 'listed',
     'quote': 'The resolution is set by\n$\\tau = \\delta_R V/4$ with $\\delta_R = 0.07$ of the single-scenario '
              'storage value, so that a ratio of\ntwo values is resolved to $\\delta_R$.',
     'cat': ['iv-b'], 'strength': 'explicit',
     'why': 'Eq. (acceptance_tolerance) only prescribes tau; nothing in main.tex establishes that a ratio is resolved '
            'to delta_R, and l. 2003 denies a rigorous bound.',
     'ev': ['tau_definition', 'cert_definition']},
    {'id': 'L-11', 'file': 'response_to_reviewers_draft.tex', 'lines': (221, 221), 'region': 'letter, R3.1',
     'status': 'borderline',
     'quote': '2035 is worse than 2030 in gross terms (+42.5~k\\euro{}, determinate) and indistinguishable net of '
              'salvage ($-2.3$~k\\euro{}, within resolution)',
     'cat': ['iii'], 'strength': 'implied (offered in reply to a request to reassess timing with salvage)',
     'why': 'A fixed-plan year comparison with a post-evaluation credit (l. 1363); the paper itself says neither year '
            'is preferable (l. 1677). Hedged, but it answers a salvage-aware timing request.',
     'ev': ['salvage_post_evaluation', 'year_neither_preferable']},
    {'id': 'L-12', 'file': 'response_to_reviewers_draft.tex', 'lines': (263, 263), 'region': 'letter, R3.6',
     'status': 'listed',
     'quote': 'without ageing $-4.1$~k\\euro{} (within resolution, i.e.\\ break-even)',
     'cat': ['iv-b'], 'strength': 'explicit',
     'why': '"i.e. break-even" turns a w.r. verdict (row 46) into a zero-margin result (audit 6).',
     'ev': ['no_ageing_row', 'within_resolution_defined']},
    {'id': 'L-13', 'file': 'response_to_reviewers_draft.tex', 'lines': (264, 264), 'region': 'letter, R3.6',
     'status': 'listed',
     'quote': 'Under the no-ageing calibration the unit is at break-even and the plan is indeterminate at this '
              'precision.',
     'cat': ['iv-b'], 'strength': 'explicit',
     'why': 'As L-12, plus "this precision" as an established property of the rule (l. 2003).',
     'ev': ['no_ageing_row', 'cert_definition']},
    {'id': 'L-14', 'file': 'response_to_reviewers_draft.tex', 'lines': (264, 264), 'region': 'letter, R3.6',
     'status': 'borderline',
     'quote': 'Lowering the end-of-life floor from 0.70 to 0.50 changes the value by $+4.9$~k\\euro{}, within '
              'resolution, so the floor is not what makes storage uneconomic here (Table~11, last row).',
     'cat': ['iv-b'], 'strength': 'implied',
     'why': 'A w.r. verdict (row 58, 0.38x) read as the absence of an effect; w.r. means only that the determinacy '
            'rule was not met (l. 2037).',
     'ev': ['floor_row', 'within_resolution_defined']},
    {'id': 'L-O1', 'file': 'response_to_reviewers_draft.tex', 'lines': (58, 58), 'region': 'letter, opening',
     'status': 'other',
     'quote': 'so the plan without storage would be the search\'s answer under each aged calibration provided that '
              'this affinity holds under it',
     'cat': ['other: audit 7 extrapolation of the baseline search to unsearched calibrations; uses ladders with '
             'out-of-budget points (A8)'], 'strength': 'explicit (hedged as inference)',
     'why': 'No search was run under any other calibration (l. 1781).', 'ev': ['scope_fixed_plans']},
    {'id': 'L-O2', 'file': 'response_to_reviewers_draft.tex', 'lines': (249, 249), 'region': 'letter, R3.4',
     'status': 'other',
     'quote': 'storage does not pay under an optimistic cycle life, and it loses more under the datasheet calibration '
              '(Section~4.3).',
     'cat': ['other: audit 7 (datasheet loses 45.2 k EUR, baseline 64.4 k EUR: it loses less)'], 'strength': 'explicit',
     'why': 'Rows 48 and 49 of tab:appE_claims.', 'ev': ['baseline_row', 'datasheet_row']},
    {'id': 'L-O3', 'file': 'response_to_reviewers_draft.tex', 'lines': (264, 264), 'region': 'letter, R3.6',
     'status': 'other',
     'quote': 'since the value is affine in size on the evaluated ladders with a slope below cost, the plan without '
              'storage would be the search\'s answer under each of them',
     'cat': ['other: audit 3.2 / 7 (fixed-plan losses offered as making re-optimisation unnecessary); ladders include '
             'out-of-budget points (A8)'], 'strength': 'explicit (hedged as inference)',
     'why': 'Fixed-plan re-evaluations only (l. 1781).', 'ev': ['scope_fixed_plans']},
    # ---- cover letter
    {'id': 'C-1', 'file': 'cover_letter.tex', 'lines': (14, 14), 'region': 'cover letter', 'status': 'listed',
     'quote': 'The proposed approach is formulated as a bi-level stochastic optimization model in which the upper level '
              'determines siting, sizing, and staged investment decisions under investment-cost uncertainty, while the '
              'lower level evaluates these decisions through coordinated operation.',
     'cat': ['ii', 'iii'], 'strength': 'explicit (submitted-version text, unrevised)',
     'why': 'Staged decisions were not searched (l. 1781); cost uncertainty enters as expected coefficients '
            '(l. 690); "bi-level" withdrawn in the revision.',
     'ev': ['scope_timing', 'expected_budget']},
    {'id': 'C-O1', 'file': 'cover_letter.tex', 'lines': (24, 24), 'region': 'cover letter', 'status': 'other',
     'quote': 'The analysis further shows that battery degradation materially affects ESS valuation and that '
              'planning-horizon discretization can influence siting and sizing decisions.',
     'cat': ['other: withdrawn result (audit 7: cover letter unrevised)'], 'strength': 'explicit',
     'why': 'The discretization sensitivity was removed (letter l. 281).', 'ev': []},
]

# Outside the named scope, listed for the Planner (not counted)
A7_SUPPLEMENT = [
    {'id': 'S-INTRO-GAP', 'file': 'main.tex', 'lines': (368, 368), 'region': 'Introduction, gap paragraph',
     'quote': 'its stopping error is seldom quantified, so that planning differences are reported without a stated '
              'resolution', 'cat': ['iv-b'],
     'why': 'Implies the paper quantifies the stopping error (contrast l. 2003, 1777).'},
    {'id': 'S-INTRO-TIMING', 'file': 'main.tex', 'lines': (361, 361), 'region': 'Introduction',
     'quote': 'linking shared ESS siting, sizing and installation timing with coordinated TSO--DSO operation',
     'cat': ['iii'], 'why': 'Formulation capability; case search: common year, one cohort (l. 1781).'},
    {'id': 'S-MASTER-STAGED', 'file': 'main.tex', 'lines': (536, 536), 'region': 'Section 2.2 opening',
     'quote': 'including siting, sizing, and staged capacity expansion', 'cat': ['iii'],
     'why': 'Formulation capability; staged expansion not explored (l. 1781).'},
]

# Keyword fragments in scope that are not claims of the four kinds: (file, line, fragment prefix) -> reason
INC = 'keyword incidental: no claim about what was run'
FORM = 'formulation or scope statement, true of the model; not a claim about the experiments'
ACC = 'accurate statement of what was run (states the restriction, the status or a recorded value)'
FIX = ('fixed-plan result labelled as such; "within numerical resolution" / "determinate" used as defined (l. 2037, '
       'A.5), not converted into break-even')
HEAD = 'heading, label, caption or pointer list'
A8REF = 'listed under A8 instead (out-of-budget fit or abstract wording)'
RULE = 'describes the acceptance rule ("bound" = a threshold of the rule), not an error bound'
TAB = 'table row of the quantity of M-45-1 (borderline there)'
REV = "restates the reviewer's point in the letter's agreement"
EXCLUDED = {
    ('main.tex', 144, 'Shared Energy Storage Systems (ESSs) ins'): INC,
    ('main.tex', 144, 'Their planning must account for coordina'): INC,
    ('main.tex', 146, 'For a fixed reference 0.25~MVA/1~MWh uni'): FIX,
    ('main.tex', 146, 'Its estimated margin remains negative in'): FIX,
    ('main.tex', 381, 'To address this gap, this paper formulat'): FORM,
    ('main.tex', 381, 'The first stage determines a single, sce'): FORM,
    ('main.tex', 381, 'Uncertainty is represented through alter'): FORM,
    ('main.tex', 382, 'Demand growth and RES capacity follow pr'): FORM,
    ('main.tex', 388, '\\item \\textcolor{blue}{A degradation-awa'): INC,
    ('main.tex', 1632, '\\subsection{\\textcolor{blue}{Multi-Scena'): HEAD,
    ('main.tex', 1633, '\\label{subsec:res_multiscenario}'): HEAD,
    ('main.tex', 1636, 'The reference unit and the plan without '): ACC,
    ('main.tex', 1636, 'As shown in Table~\\ref{tab:res_multiscen'): ACC,
    ('main.tex', 1636, 'These evaluations used the residual-base'): ACC,
    ('main.tex', 1640, 'A fit to the continued objective traject'): INC,
    ('main.tex', 1640, 'Accounting for this decrease alone would'): ACC,
    ('main.tex', 1640, 'The configurations also differ in tempor'): ACC,
    ('main.tex', 1645, '\\caption{\\textcolor{blue}{Multi-scenario'): HEAD,
    ('main.tex', 1646, '\\label{tab:res_multiscenario}'): HEAD,
    ('main.tex', 1655, 'Reported numerical resolution & 18.0~k\\e'): TAB,
    ('main.tex', 1656, 'Single-scenario operating savings & 253.'): INC,
    ('main.tex', 1665, 'Operating savings and investment margin '): ACC,
    ('main.tex', 1667, 'Ratios use the single-scenario operating'): INC,
    ('main.tex', 1803, 'Fixed-plan sensitivities showed that the'): FIX,
    ('main.tex', 1803, 'The estimated marginal break-even energy'): A8REF,
    ('main.tex', 1803, 'Its estimated investment margin remained'): FIX,
    ('main.tex', 1807, 'Future work will extend the planning sea'): ACC,
    ('main.tex', 1807, 'Further developments will consider riche'): INC,
    ('cover_letter.tex', 14, 'This manuscript proposes a degradation-a'): INC,
    ('response_to_reviewers_draft.tex', 40, '0.50\\,\\% gap was a stopping tolerance ra'): REV,
    ('response_to_reviewers_draft.tex', 58, 'The largest instance of the submitted ve'): ACC,
    ('response_to_reviewers_draft.tex', 63, '\\subsection*{R1.2 --- Uncertainty, syste'): HEAD,
    ('response_to_reviewers_draft.tex', 75, 'On (i): the investment-cost uncertainty '): A8REF,
    ('response_to_reviewers_draft.tex', 75, 'Operational uncertainty is represented t'): A8REF,
    ('response_to_reviewers_draft.tex', 77, 'The revised results are given on the sin'): ACC,
    ('response_to_reviewers_draft.tex', 85, '\\rchanges{Abstract; Section~3 (instance '): HEAD,
    ('response_to_reviewers_draft.tex', 86, 'Section~2.3.4, last paragraph, Section~4'): HEAD,
    ('response_to_reviewers_draft.tex', 88, '\\subsection*{R1.4 --- Uncertainty and di'): HEAD,
    ('response_to_reviewers_draft.tex', 97, '\\rresponse{On (i): the introduction now '): FORM,
    ('response_to_reviewers_draft.tex', 97, 'Fuel costs are not represented separatel'): INC,
    ('response_to_reviewers_draft.tex', 97, 'On (ii): a discount-rate sensitivity was'): FIX,
    ('response_to_reviewers_draft.tex', 108, '\\subsection*{R2.1 and R2.9 --- Not a tru'): HEAD,
    ('response_to_reviewers_draft.tex', 113, 'The problem is a two-stage stochastic pr'): FORM,
    ('response_to_reviewers_draft.tex', 116, 'investment plan is a single vector commo'): ACC,
    ('response_to_reviewers_draft.tex', 117, 'nonanticipative by construction; the inv'): ACC,
    ('response_to_reviewers_draft.tex', 118, 'probability-weighted expected cost of th'): ACC,
    ('response_to_reviewers_draft.tex', 125, '\\rresponse{Algorithm~1 and Section~2.1 n'): ACC,
    ('response_to_reviewers_draft.tex', 125, 'One recourse evaluation is made per cand'): ACC,
    ('response_to_reviewers_draft.tex', 125, 'The number of candidates evaluated is re'): ACC,
    ('response_to_reviewers_draft.tex', 142, 'new 2.2.7 ``Recourse evaluation and cert'): HEAD,
    ('response_to_reviewers_draft.tex', 154, 'measured turning points, non-growing swi'): RULE,
    ('response_to_reviewers_draft.tex', 155, 'least one period long, a bound on the pr'): RULE,
    ('response_to_reviewers_draft.tex', 158, 'certified, the 10 uncertified by cause, '): ACC,
    ('response_to_reviewers_draft.tex', 159, 'post-certification movement ($\\le 0.9\\,\\'): ACC,
    ('response_to_reviewers_draft.tex', 160, 'The uncertified evaluations are reported'): ACC,
    ('response_to_reviewers_draft.tex', 162, 'storage discharge drives transmission co'): INC,
    ('response_to_reviewers_draft.tex', 166, 'On failure handling: a local solve that '): ACC,
    ('response_to_reviewers_draft.tex', 167, 'under a documented sequence of solver se'): ACC,
    ('response_to_reviewers_draft.tex', 182, 'These studies take the installed capacit'): INC,
    ('response_to_reviewers_draft.tex', 203, 'The state-of-charge recursion with charg'): INC,
    ('response_to_reviewers_draft.tex', 204, 'The paper now also says where each const'): INC,
    ('response_to_reviewers_draft.tex', 221, 'Among the 55 comparisons of the full cla'): ACC,
    ('response_to_reviewers_draft.tex', 224, '\\subsection*{R3.2 --- The 0.50\\,\\% gap i'): HEAD,
    ('response_to_reviewers_draft.tex', 228, 'The method is described as a nested deco'): ACC,
    ('response_to_reviewers_draft.tex', 241, 'On continuity: within each block (one re'): INC,
    ('response_to_reviewers_draft.tex', 241, 'Scenario deviations are absorbed on the '): INC,
    ('response_to_reviewers_draft.tex', 241, 'The state of charge evolves over the hou'): INC,
    ('response_to_reviewers_draft.tex', 241, 'Cumulative discharge beyond the stored e'): INC,
    ('response_to_reviewers_draft.tex', 256, 'Their values differ materially: the refe'): FIX,
    ('response_to_reviewers_draft.tex', 264, 'On installed capacity over time: Table~1'): ACC,
    ('response_to_reviewers_draft.tex', 273, 'the years of its block, representative-d'): ACC,
    ('response_to_reviewers_draft.tex', 274, 'discounted from their installation year '): ACC,
    ('response_to_reviewers_draft.tex', 283, '\\item The limitations section is new: se'): INC,
    ('response_to_reviewers_draft.tex', 285, 'In the submitted version it held per-blo'): ACC,
    ('response_to_reviewers_draft.tex', 288, '\\item The method description now focuses'): HEAD,
}

# ------------------------------------------------------------------------------------------------------- A8 places
# (id, file, line, exact substring, element, over-budget plans shown, label: none / partial / labelled, note)
A8_PLACES = [
    ('P01', 'main.tex', 1424, 'An affine fit, $\\text{value}=a+bE+cP$, is obtained from the evaluated 2~h and 4~h '
     'configurations', 'text, Section 4.2', 'fit points = evaluations 2-11 (5 of 10 over budget)', 'none', ''),
    ('P02', 'main.tex', 1424, 'The six certified evaluations give $e^{*}=183{,}080$~\\euro/MWh.', 'text, Section 4.2',
     'certified fit points 2, 3, 5, 7, 10, 11 (3 of 6 over budget: 5, 10, 11)', 'none', ''),
    ('P03', 'main.tex', 1425, 'Including all ten evaluations, with uncertified and flagged evaluations represented by '
     'intervals, gives 171{,}310--192{,}560~\\euro/MWh.', 'text, Section 4.2', 'evaluations 2-11', 'none',
     'the next sentence restricts the estimate to "the fitted range", not to the admissible set'),
    ('P04', 'main.tex', 1431, 'Storage investment margins at node~7 in 2025 for (a) 4~h and (b) 2~h configurations',
     'Figure 1 caption (fig1_breakeven.pdf)', 'panels (a), (b): E = 1-5 MWh, 4 h and 2 h', 'none',
     'figure inspected visually: no budget line or legend entry'),
    ('P05', 'main.tex', 1437, 'Estimated marginal break-even energy cost for 4~h storage at node~7',
     'Table tab:res_breakeven caption', 'n = 6 / 10 fit (rows l. 1446-1447)', 'none', ''),
    ('P06', 'main.tex', 1467, 'all five evaluated 4~h configurations at node~7, installed in 2025, have positive '
     'investment margins', 'text, Section 4.2', '7: 1/4 and 7: 1.25/5 at m = 2', 'none', ''),
    ('P07', 'main.tex', 1467, 'Each additional 1~MWh yields a determinate increase in margin of 41.0--51.8~k\\euro{}.',
     'text, Section 4.2', 'increments 3->4 and 4->5 MWh', 'none', ''),
    ('P08', 'main.tex', 1472, 'Investment margins and incremental gains for 4~h storage at node~7 in 2025, at twice the '
     'base flexibility price.', 'Table tab:res_energy_ladder_m2 caption', 'rows 4 and 5 MWh', 'none',
     'row marks u / f are certification marks only'),
    ('P08a', 'main.tex', 1487, '4$^{f}$ & 176.6', 'Table tab:res_energy_ladder_m2 row', '7: 1/4, m = 2', 'none', ''),
    ('P08b', 'main.tex', 1488, '5$^{u}$ & 228.4', 'Table tab:res_energy_ladder_m2 row', '7: 1.25/5, m = 2', 'none', ''),
    ('P09', 'main.tex', 1803, 'The estimated marginal break-even energy costs for 4~h storage at node~7 were below the '
     'baseline energy-capacity cost.', 'text, Section 5', 'derived from the P01-P03 fit', 'none', ''),
    ('P10a', 'main.tex', 2628, '4 & 7: 1/4 & 1 / 5 & 274.0', 'tab:appE_claims (a) row 4', '7: 1/4', 'none', ''),
    ('P10b', 'main.tex', 2629, '5 & 7: 1.25/5 & 1 / 6$^{u}$', 'tab:appE_claims (a) row 5', '7: 1.25/5', 'none', ''),
    ('P10c', 'main.tex', 2632, '8 & 7: 1.5/3 & 1 / 9$^{u}$', 'tab:appE_claims (a) row 8', '7: 1.5/3', 'none', ''),
    ('P10d', 'main.tex', 2633, '9 & 7: 2/4 & 1 / 10', 'tab:appE_claims (a) row 9', '7: 2/4', 'none', ''),
    ('P10e', 'main.tex', 2634, '10 & 7: 2.5/5 & 1 / 11', 'tab:appE_claims (a) row 10', '7: 2.5/5', 'none', ''),
    ('P10f', 'main.tex', 2652, '26 & 7: 1/4, $m=2$', 'tab:appE_claims (b) row 26', '7: 1/4, m = 2', 'none', ''),
    ('P10g', 'main.tex', 2653, '27 & 7: 1.25/5, $m=2$', 'tab:appE_claims (b) row 27', '7: 1.25/5, m = 2', 'none', ''),
    ('P10h', 'main.tex', 2656, '30 & 7: 0.75/3 $\\to$ 1/4, $m=2$', 'tab:appE_claims (b) row 30', '-> 7: 1/4', 'none', ''),
    ('P10i', 'main.tex', 2657, '31 & 7: 1/4 $\\to$ 1.25/5, $m=2$', 'tab:appE_claims (b) row 31', '7: 1/4 -> 1.25/5',
     'none', ''),
    ('P11a', 'main.tex', 2716, '(a) Baseline price: the plan without storage and single-node plans in 2025',
     'tab:appE_evaluations group (a) header', 'evaluations 5, 6, 9, 10, 11', 'none', ''),
    ('P11d', 'main.tex', 2721, '5 & 7: 1/4 & 2025 & base', 'tab:appE_evaluations row 5', '7: 1/4', 'none', ''),
    ('P11e', 'main.tex', 2722, '6 & 7: 1.25/5 & 2025 & base', 'tab:appE_evaluations row 6', '7: 1.25/5', 'none', ''),
    ('P11f', 'main.tex', 2725, '9 & 7: 1.5/3 & 2025 & base', 'tab:appE_evaluations row 9', '7: 1.5/3', 'none', ''),
    ('P11g', 'main.tex', 2726, '10 & 7: 2/4 & 2025 & base', 'tab:appE_evaluations row 10', '7: 2/4', 'none', ''),
    ('P11h', 'main.tex', 2727, '11 & 7: 2.5/5 & 2025 & base', 'tab:appE_evaluations row 11', '7: 2.5/5', 'none', ''),
    ('P11b', 'main.tex', 2760, '38 & 7: 1/4 & 2025 & $m=2$', 'tab:appE_evaluations row 38', '7: 1/4, m = 2', 'none', ''),
    ('P11c', 'main.tex', 2761, '39 & 7: 1.25/5 & 2025 & $m=2$', 'tab:appE_evaluations row 39', '7: 1.25/5, m = 2',
     'none', ''),
    ('P12a', 'main.tex', 2899, '39 & 7: 1.25/5 & 2025 & $m=2$ & gap clause', 'tab:appE_gap row', '7: 1.25/5, m = 2',
     'none', ''),
    ('P12b', 'main.tex', 2906, '6 & 7: 1.25/5 & 2025 & base & growth test', 'tab:appE_gap row', '7: 1.25/5', 'none', ''),
    ('P13', 'main.tex', 2921, 'the cache of the baseline search was seeded with the single-node ladders of the initial '
     'design, including ladder points over the budget', 'text, Appendix E.7', 'unnamed', 'labelled (over budget)',
     'the only sentence that says so; it does not name the points or reach Section 4.2, Fig. 1 or the tables'),
    ('P14', 'main.tex', 2797, 'twelve are admissible, the smallest being 0.25~MVA / 0.5~MWh',
     'text + fig_lattice_case.pdf, Appendix E.3', 'all 23 over-budget single-node 2025 plans (legend: "within limits, '
     'over budget")', 'partial', 'labels the lattice, not the evaluated ladder; no cross-reference from Section 4.2'),
    ('P15', 'response_to_reviewers_draft.tex', 58, 'at the baseline calibration the value is measured to be affine in '
     'storage size with a slope below cost', 'letter, opening', 'the ladders (evaluations 2-11)', 'none', ''),
    ('P16', 'response_to_reviewers_draft.tex', 264, 'since the value is affine in size on the evaluated ladders with a '
     'slope below cost', 'letter, R3.6', 'the ladders (evaluations 2-11)', 'none', ''),
]
LABEL_PATTERNS = re.compile(r'over (the )?budget|outside the admissible|inadmissible|exploratory|probe|exceeds? the '
                            r'budget|budget-infeasible|not admissible', re.I)

# A8 (2): every 13.9 occurrence; (file, line) -> (framing, note)
P139 = re.compile(r'13[.,]9')
P139_EXPECTED = {
    ('highlights.tex', 14): ('saving', '"Coordination is worth 13.9 % of operating cost against static interface '
                             'practice": a benefit of 13.9 % of operating cost; denominator unstated'),
    ('response_to_reviewers_draft.tex', 53): ('saving', 'the value of coordination "90.9 M EUR (13.9 %)"'),
    ('response_to_reviewers_draft.tex', 80): ('saving', '"the abstract reports instead the value of coordination, '
                                              '13.9 % of the discounted operating cost" -- the abstract reports 12.2 %'),
    ('main.tex', 1583): ('cost increase (correct)', '"incur 13.9 % ... higher ... than coordinated operation"'),
    ('main.tex', 1597): ('cost increase (correct)', 'table column Delta Q (%); note: "percentages are relative to the '
                         'coordinated cost"'),
    ('highlights_v2', 12): ('saving', 'staging copy of the standalone bullet (identical text)'),
}

# A8 (3)
ABS_TECH_TERMS = ('lithium', 'li-ion', 'lfp', 'chemistr')
ABS_COST_TERMS = ('technology', 'trajector', 'future', 'converter')


# ------------------------------------------------------------------------------------------------------- helpers
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


def _ck(name, ok, detail=None):
    return {'check': name, 'ok': bool(ok), 'detail': detail}


def clone_files(head):
    out = {}
    for f in CLONE_FILES:
        data = _git(['show', f'{head}:{f}'], cwd=CLONE, binary=True)
        INPUTS[f'{CLONE}@{head}:{f}'] = {'sha256': _sha_bytes(data), 'status': 'overleaf clone (git show; read only)'}
        out[f] = data.decode('utf-8')
    for f in FIG_FILES:
        data = _git(['show', f'{head}:{f}'], cwd=CLONE, binary=True)
        INPUTS[f'{CLONE}@{head}:{f}'] = {'sha256': _sha_bytes(data), 'status': 'overleaf clone (git show; read only)'}
    return out


def locate(lines, line_range, quote):
    """True when `quote` (parts joined by newline) sits on lines first..last, part k on line first + k."""
    first, last = line_range
    parts = quote.split('\n')
    if len(parts) != last - first + 1 or first < 1 or last > len(lines):
        return False
    return all(p in lines[first - 1 + k] for k, p in enumerate(parts))


# ------------------------------------------------------------------------------------------------------- identity
def identity():
    checks = []
    head_full = _git(['rev-parse', 'HEAD'], cwd=CLONE).strip()
    checks.append(_ck('clone HEAD is edd7a63', head_full.startswith(DECLARED_HEAD), head_full))
    status = _git(['status', '--porcelain', '--untracked-files=no'], cwd=CLONE).strip()
    checks.append(_ck('clone tracked files clean (read-only use)', status == '', status))
    files = clone_files(DECLARED_HEAD)
    main_sha = _sha_bytes(files['main.tex'].encode('utf-8'))
    checks.append(_ck('main.tex sha256 prefix = audit fingerprint b08a3365607e56f2', main_sha.startswith(MAIN_SHA_PREFIX),
                      main_sha))
    letter = files['response_to_reviewers_draft.tex']
    checks.append(_ck('letter is v9 (header line)', letter.splitlines()[0].startswith('% Response to reviewers — v9'),
                      letter.splitlines()[0][:60]))
    repo = {k: _read(v) for k, v in REPO_TEX.items()}
    return {'ok': all(c['ok'] for c in checks), 'checks': checks, 'clone_head': head_full, 'main_sha256': main_sha,
            'repo_head': _git(['rev-parse', 'HEAD']).strip()}, files, repo


# ------------------------------------------------------------------------------------------------------- A7
def scopes(files):
    """In-scope line numbers per file, comments excluded."""
    m = files['main.tex'].splitlines()

    def idx(lines, needle, start=0):
        for i in range(start, len(lines)):
            if needle in lines[i]:
                return i + 1
        raise KeyError(needle)

    sc = {}
    a0, a1 = idx(m, '\\begin{abstract}'), idx(m, '\\end{abstract}')
    h0, h1 = idx(m, '\\begin{highlights}'), idx(m, '\\end{highlights}')
    c0 = idx(m, 'To address this gap, this paper formulates')
    c1 = idx(m, '\\end{itemize}', idx(m, 'The main contributions of this paper are as follows:'))
    s0 = idx(m, '\\subsection{\\textcolor{blue}{Multi-Scenario Evaluation}}')
    s1 = idx(m, '\\subsection', s0) - 1
    k0, k1 = idx(m, '\\section{Conclusions}'), idx(m, '\\appendix') - 1
    regions = {'abstract': (a0, a1), 'highlights (main.tex)': (h0, h1), 'contributions': (c0, c1),
               'Section 4.5': (s0, s1), 'Section 5': (k0, k1)}
    sc['main.tex'] = {r: [n for n in range(lo, hi + 1) if not m[n - 1].lstrip().startswith('%')]
                      for r, (lo, hi) in regions.items()}
    hl = files['highlights.tex'].splitlines()
    sc['highlights.tex'] = {'highlights (standalone)': list(range(idx(hl, '\\begin{itemize}'), idx(hl, '\\end{itemize}') + 1))}
    cl = files['cover_letter.tex'].splitlines()
    sc['cover_letter.tex'] = {'cover letter': list(range(idx(cl, '\\begin{letter}'), idx(cl, '\\end{letter}') + 1))}
    le = files['response_to_reviewers_draft.tex'].splitlines()
    begin = idx(le, '\\begin{document}')
    keep, in_comment = [], False
    for n in range(begin, len(le) + 1):
        t = le[n - 1]
        if '\\rcomment{' in t:          # a reviewer quote runs from \rcomment{ to the line before \rresponse{
            in_comment = True
        if '\\rresponse{' in t:
            in_comment = False
        if not in_comment and not t.lstrip().startswith('%'):
            keep.append(n)
    sc['response_to_reviewers_draft.tex'] = {'letter (v9), reviewer quotes excluded': keep}
    return sc, regions


_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z$\\(])')


def fragments(text):
    return [f for f in _SPLIT.split(text) if f.strip()]


def _overlap(a, b):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sm.find_longest_match(0, len(a), 0, len(b)).size


def a7_run(files, entries=A7, excluded=EXCLUDED):
    lines = {f: files[f].splitlines() for f in CLONE_FILES}
    sc, regions = scopes(files)
    located = []
    for e in entries:
        ok = locate(lines[e['file']], e['lines'], e['quote'])
        evs = []
        for k in e['ev']:
            f, ln, sub = EV[k]
            src = lines[f] if f in lines else CODE_SRC[f]
            evs.append({'key': k, 'file': f, 'line': ln, 'ok': 0 < ln <= len(src) and sub in src[ln - 1]})
        in_scope = any(ln in v for v in sc[e['file']].values() for ln in range(e['lines'][0], e['lines'][1] + 1))
        located.append({**{k: v for k, v in e.items() if k != 'ev'}, 'quote_at_lines': ok, 'in_scope': in_scope,
                        'evidence': evs, 'evidence_ok': all(x['ok'] for x in evs)})
    supp = [{**s, 'quote_at_lines': locate(lines[s['file']], s['lines'], s['quote'])} for s in A7_SUPPLEMENT]
    # coverage of every keyword fragment in scope
    covered, excl, uncovered = [], [], []
    for f, regs in sc.items():
        for reg, nums in regs.items():
            for n in nums:
                for frag in fragments(lines[f][n - 1]):
                    hits = sorted({h.group(0).lower() for h in KW.finditer(frag)})
                    if not hits:
                        continue
                    ent = [e['id'] for e in entries if e['file'] == f and e['lines'][0] <= n <= e['lines'][1]
                           and _overlap(frag, e['quote'].split('\n')[n - e['lines'][0]]) >=
                           min(25, len(e['quote'].split('\n')[n - e['lines'][0]]), len(frag.strip()))]
                    key = (f, n, frag.strip()[:40])
                    rec = {'file': f, 'line': n, 'region': reg, 'fragment': frag.strip(), 'keywords': hits}
                    if ent:
                        covered.append({**rec, 'entries': ent})
                    elif key in excluded:
                        excl.append({**rec, 'reason': excluded[key]})
                    else:
                        uncovered.append(rec)
    unused_excl = [list(k) for k in excluded if not any((x['file'], x['line'], x['fragment'][:40]) == k for x in excl)]
    counts = {}
    for st in ('listed', 'borderline', 'other'):
        counts[st] = sum(1 for e in located if e['status'] == st)
    by_file = {}
    for e in located:
        if e['status'] == 'listed':
            by_file[e['file']] = by_file.get(e['file'], 0) + 1
    by_cat = {}
    for e in located:
        if e['status'] == 'listed':
            for c in e['cat']:
                by_cat[c] = by_cat.get(c, 0) + 1
    return {'entries': located, 'supplement_outside_scope': supp, 'scope_lines': sc,
            'regions_main_tex': {k: list(v) for k, v in regions.items()},
            'keyword_pattern': KW.pattern, 'coverage': {'covered': covered, 'excluded': excl, 'uncovered': uncovered,
                                                        'unused_exclusions': unused_excl},
            'counts': counts, 'listed_by_file': by_file, 'listed_by_category': by_cat,
            'all_quotes_located': all(e['quote_at_lines'] for e in located) and all(s['quote_at_lines'] for s in supp),
            'all_evidence_located': all(e['evidence_ok'] for e in located),
            'all_entries_in_scope': all(e['in_scope'] for e in located)}


# ------------------------------------------------------------------------------------------------------- A8 (1)
def _parse_plan(plan, year):
    """Table E.21 plan text -> {node: (S, E)} and the year (None for x = 0)."""
    plan = plan.strip()
    if 'boldsymbol{x}=\\boldsymbol{0}' in plan:
        return {}, None
    out = {}
    for part in plan.split(';'):
        part = part.strip()
        nodes, size = part.split(':')
        size = size.replace('each', '').strip()
        s, e = (float(v) for v in size.split('/'))
        for n in nodes.split(','):
            out[int(n.strip())] = (s, e)
    return out, int(year)


def evaluations_table(main_lines):
    start = next(i for i, ln in enumerate(main_lines) if '\\label{tab:appE_evaluations}' in ln)
    end = next(i for i in range(start, len(main_lines)) if '\\end{longtable}' in main_lines[i])
    rows = []
    for i in range(start, end):
        m = re.match(r'^(\d+) & (.+?) & (\d{4}|---) & (.+?) & (.+?) & ', main_lines[i])
        if m:
            nodes, year = _parse_plan(m.group(2), m.group(3) if m.group(3) != '---' else 0)
            rows.append({'eval': int(m.group(1)), 'line': i + 1, 'plan_text': m.group(2), 'year': year,
                         'condition': m.group(4), 'status': m.group(5), 'nodes': nodes})
    return rows


def case_inputs():
    case = json.loads(_read(CASE_JSON))
    years = {int(y): int(n) for y, n in case['Years'].items()}
    params = SEP.SharedEnergyStorageParameters()
    _reg(ESS_PARAMS)
    params.read_parameters_from_file(ESS_PARAMS)
    _reg(COST_XLSX)
    n, probs = SED._get_operational_scenarios_info_from_excel_file(COST_XLSX, 'Scenarios')
    costs = {'power': SED._get_investment_costs_from_excel_file(COST_XLSX, 'Investment Cost, Power', n, list(years)),
             'energy': SED._get_investment_costs_from_excel_file(COST_XLSX, 'Investment Cost, Energy', n, list(years))}
    return {'years': years, 'discount_rate': float(case['DiscountFactor']), 'budget': params.budget,
            'max_capacity': params.max_capacity, 'min_ratio': params.min_energy_to_power_ratio,
            'max_ratio': params.max_energy_to_power_ratio, 'n_trajectories': n, 'probabilities': probs,
            'costs': costs, 'params': params}


def expected_unit_costs(ci):
    y0 = min(ci['years'])
    out = {}
    for y in ci['years']:
        disc = 1.0 / (1.0 + ci['discount_rate']) ** (y - y0)
        p = sum(w * ci['costs']['power'][k][y] for k, w in enumerate(ci['probabilities']))
        e = sum(w * ci['costs']['energy'][k][y] for k, w in enumerate(ci['probabilities']))
        out[y] = {'discount_multiplier': disc, 'power_undiscounted': p, 'energy_undiscounted': e,
                  'power_discounted': p * disc, 'energy_discounted': e * disc}
    return out


def production_check(ci, nodes, year):
    """shared_resources_planning._check_candidate_first_stage_feasibility on a stub carrying the case data."""
    sed = SimpleNamespace(params=ci['params'], years=ci['years'], discount_factor=ci['discount_rate'],
                          prob_market_scenarios=ci['probabilities'], cost_investment=ci['costs'])
    yrs = list(ci['years'])
    inv = {n: {y: {'s': (s if y == year else 0.0), 'e': (e if y == year else 0.0)} for y in yrs}
           for n, (s, e) in nodes.items()}
    tot = {n: {y: {'e': (e if year is not None and y >= year else 0.0)} for y in yrs} for n, (s, e) in nodes.items()}
    ok, reasons = SRP._check_candidate_first_stage_feasibility(SimpleNamespace(shared_ess_data=sed),
                                                                {'investment': inv, 'total_capacity': tot})
    return ok, reasons


def a8_budget(ci, main_lines, w2, s51):
    uc = expected_unit_costs(ci)
    checks = []
    for y in ci['years']:
        w = w2['expected_unit_costs_per_case_year']['new'][str(y)]
        checks.append(_ck(f'{y}: expected discounted unit costs = W2 table (|diff| <= 1e-6 EUR)',
                          abs(uc[y]['power_discounted'] - w['power_eur_per_mva_discounted']) <= 1e-6 and
                          abs(uc[y]['energy_discounted'] - w['energy_eur_per_mwh_discounted']) <= 1e-6,
                          {'power': uc[y]['power_discounted'], 'energy': uc[y]['energy_discounted']}))
    # Table 1 of main.tex against the cost file (k EUR, two decimals)
    t1 = {}
    for kind, sym in (('power', 'c^\\text{Inv,S}'), ('energy', 'c^\\text{Inv,E}')):
        i0 = next(i for i, ln in enumerate(main_lines) if sym in ln and '\\left' not in ln and '&' in ln)
        for k, name in enumerate(('Low', 'Medium', 'High')):
            ln = main_lines[i0 + k]
            m = re.search(name + r'\s*&\s*([\d.]+)\\%\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)', ln)
            t1[(kind, name)] = (float(m.group(1)), [float(m.group(j)) for j in (2, 3, 4)], i0 + k + 1)
    t1_ok = []
    for (kind, name), (prob, vals, ln) in t1.items():
        k = ('Low', 'Medium', 'High').index(name)
        file_vals = [round(ci['costs'][kind][k][y] / 1e3, 2) for y in sorted(ci['years'])]
        t1_ok.append({'row': f'{kind} {name}', 'line': ln, 'table': vals, 'file': file_vals,
                      'prob_table': prob, 'prob_file': ci['probabilities'][k] * 100,
                      'ok': vals == file_vals and abs(prob - ci['probabilities'][k] * 100) < 1e-9})
    checks.append(_ck('Table 1 equals the cost file (k EUR, 2 dp; probabilities)', all(r['ok'] for r in t1_ok), t1_ok))
    printed = [ln for ln in main_lines if 'The probability-weighted unit costs in 2025 are' in ln]
    checks.append(_ck('printed 2025 expected costs 256.32 / 253.88 = file rounded',
                      bool(printed) and '256.32' in printed[0] and '253.88' in printed[0] and
                      round(uc[2025]['power_undiscounted'] / 1e3, 2) == 256.32 and
                      round(uc[2025]['energy_undiscounted'] / 1e3, 2) == 253.88))
    budget_line = [i + 1 for i, ln in enumerate(main_lines) if 'The baseline expected discounted investment budget is '
                   '1~M\\euro{}, with a maximum rated energy capacity of 5~MWh per interface node.' in ln]
    checks.append(_ck('main.tex states budget 1 M EUR and 5 MWh per node = case file', bool(budget_line) and
                      ci['budget'] == 1e6 and ci['max_capacity'] == 5.0, budget_line))
    # every evaluation of Table E.21
    rows = evaluations_table(main_lines)
    for r in rows:
        y = r['year']
        r['I_eur'] = 0.0 if not r['nodes'] else sum(uc[y]['power_discounted'] * s + uc[y]['energy_discounted'] * e
                                                    for s, e in r['nodes'].values())
        ok, reasons = production_check(ci, r['nodes'], y) if r['nodes'] else (True, '')
        r['production_first_stage_ok'] = ok
        r['production_reasons'] = reasons
        r['budget_slack_eur'] = ci['budget'] - r['I_eur']
        r['over_budget'] = r['I_eur'] > ci['budget']
        r['consistent'] = (not r['over_budget']) == ok or (r['over_budget'] and 'budget' in reasons)
        r['nodes'] = {str(k): list(v) for k, v in r['nodes'].items()}
    checks.append(_ck('61 evaluations parsed from tab:appE_evaluations', len(rows) == 61, len(rows)))
    checks.append(_ck('closed-form verdict = production check for every evaluation', all(r['consistent'] for r in rows)))
    over = [r for r in rows if r['over_budget']]
    # the 4 h ladder at node 7 in 2025 (audit 3.6 table)
    ladder = []
    for e in (1, 2, 3, 4, 5):
        s = e / 4.0
        i_x = uc[2025]['power_discounted'] * s + uc[2025]['energy_discounted'] * e
        ok, reasons = production_check(ci, {7: (s, float(e))}, 2025)
        ladder.append({'S_MVA': s, 'E_MWh': e, 'I_eur': i_x, 'budget_slack_eur': ci['budget'] - i_x,
                       'production_first_stage_ok': ok, 'reasons': reasons})
    audit_vals = {1: 318.0, 2: 635.9, 3: 953.9, 4: 1271.8, 5: 1589.8}
    checks.append(_ck('audit 3.6 table reproduced (k EUR, 1 dp)',
                      all(round(x['I_eur'] / 1e3, 1) == audit_vals[x['E_MWh']] for x in ladder),
                      {x['E_MWh']: round(x['I_eur'] / 1e3, 1) for x in ladder}))
    s51_stmt = s51['budget_corner_statement']
    checks.append(_ck('s51 F2 record: 0.75/3 I = 953871.0255 EUR, E = 4 and 5 budget-infeasible',
                      abs(ladder[2]['I_eur'] - 953871.0255106756) < 1e-3 and 'both budget-infeasible' in s51_stmt,
                      s51_stmt[:200]))
    # single-node 2025 lattice (fig_lattice_case): admissible / over budget counts with the production check
    adm, ovb = [], []
    for i in range(1, 12):
        for j in range(1, 12):
            s, e = i * 0.25, j * 0.5
            if not (2 * s - 1e-9 <= e <= 4 * s + 1e-9) or e > 5.0 + 1e-9:
                continue
            ok, reasons = production_check(ci, {7: (s, e)}, 2025)
            (adm if ok else ovb).append(f'{s:g}/{e:g}')
            if not ok and 'budget' not in reasons:
                raise SystemExit(f'harness fault: lattice point {s}/{e} refused for {reasons}')
    checks.append(_ck('single-node 2025 lattice: 12 admissible (main.tex l. 2797), 23 over budget (W177b)',
                      len(adm) == 12 and len(ovb) == 23, {'admissible': adm, 'over_budget': ovb}))
    returned = next(r for r in rows if r['eval'] == 40)
    checks.append(_ck('returned 2x plan (evaluation 40, 2030) I ~ 992.3 k EUR, within budget (audit 3.6)',
                      round(returned['I_eur'] / 1e3, 1) == 992.3 and not returned['over_budget'], returned['I_eur']))
    return {'case': {'years': ci['years'], 'discount_rate': ci['discount_rate'], 'budget_eur': ci['budget'],
                     'max_capacity_mwh': ci['max_capacity'], 'ratio_bounds_h': [ci['min_ratio'], ci['max_ratio']],
                     'trajectory_probabilities': ci['probabilities'],
                     'unit_costs_eur': {kind: {str(k): {str(y): v for y, v in d.items()} for k, d in ci['costs'][kind].items()}
                                        for kind in ('power', 'energy')}},
            'expected_unit_costs': {str(y): v for y, v in uc.items()}, 'formula': (
                'I(x) = sum_nodes (1+r)^-(y-2025) * sum_c w_c (c^S_{y,c} S + c^E_{y,c} E), r = SRP1.json DiscountFactor, '
                'w_c = SRP1_ESS.xlsx Scenarios row; admissible iff I(x) <= budget (SRP1_ESS_Params.json) -- the formula '
                'of shared_resources_planning._check_candidate_first_stage_feasibility and of the master budget row '
                '(shared_energy_storage_data.py l. 366-378)'),
            'checks': checks, 'ladder_4h_node7_2025': ladder, 'evaluations': rows,
            'over_budget_evaluations': [{k: r[k] for k in ('eval', 'line', 'plan_text', 'year', 'condition', 'I_eur',
                                                           'budget_slack_eur')} for r in over],
            'lattice_single_node_2025': {'admissible': adm, 'over_budget': ovb}}


def a8_places(files, over):
    lines = {f: files[f].splitlines() for f in CLONE_FILES}
    out = []
    for pid, f, ln, sub, elem, plans, label, note in A8_PLACES:
        ok = 0 < ln <= len(lines[f]) and sub in lines[f][ln - 1]
        out.append({'id': pid, 'file': f, 'line': ln, 'quote': sub, 'element': elem, 'over_budget_content': plans,
                    'label': label, 'note': note, 'quote_at_line': ok})
    # completeness: every line of main.tex naming an over-budget evaluation's plan in its year is a listed place
    tokens = sorted({r['plan_text'] for r in over})
    hits = []
    for i, t in enumerate(lines['main.tex']):
        for tok in tokens:
            if tok in t and ('2025' in t or '$m=2$' in t or 'm=2' in t or re.match(r'^\d+ & ', t)):
                hits.append((i + 1, tok))
    listed_lines = {p['line'] for p in out if p['file'] == 'main.tex'}
    # rows of the 2030 neighbourhood that merely share a token with a 2025 plan are not out-of-budget appearances
    hits_2025 = [(n, tok) for n, tok in hits if '2030' not in lines['main.tex'][n - 1]
                 and not re.match(r'^4[01] & ', lines['main.tex'][n - 1])]
    unlisted = sorted({n for n, _ in hits_2025} - listed_lines)
    # every line in any submission file using a budget-label phrase
    label_hits = []
    for f in CLONE_FILES:
        for i, t in enumerate(lines[f]):
            for m in LABEL_PATTERNS.finditer(t):
                label_hits.append({'file': f, 'line': i + 1, 'match': m.group(0),
                                   'context': t[max(0, m.start() - 100): m.end() + 100].strip()})
    return {'places': out, 'all_quotes_at_lines': all(p['quote_at_line'] for p in out),
            'token_scan_main_tex': {'tokens': tokens, 'hits': hits_2025, 'unlisted_lines': unlisted},
            'label_phrase_scan': {'pattern': LABEL_PATTERNS.pattern, 'files': list(CLONE_FILES), 'hits': label_hits},
            'counts': {'places': len(out), 'labelled': sum(1 for p in out if p['label'].startswith('labelled')),
                       'partial': sum(1 for p in out if p['label'] == 'partial'),
                       'none': sum(1 for p in out if p['label'] == 'none')}}


# ------------------------------------------------------------------------------------------------------- A8 (2)
def a8_139(files, repo, frozen):
    srcs = {f: files[f] for f in CLONE_FILES}
    srcs.update({k: v for k, v in repo.items()})
    hits = []
    for f, txt in srcs.items():
        for i, t in enumerate(txt.splitlines()):
            if P139.search(t):
                exp = P139_EXPECTED.get((f, i + 1))
                hits.append({'file': f, 'line': i + 1, 'text': t.strip()[:400],
                             'framing': exp[0] if exp else 'UNCLASSIFIED', 'note': exp[1] if exp else ''})
    b = frozen['tables']['benchmark']
    q_c, q_pt = b['coordinated_Q'], b['arms']['price_taker']['Q_best']
    ratios = {'benefit_eur': b['benefit'], 'coordinated_Q_eur': q_c, 'price_taker_Q_eur': q_pt,
              'benefit_over_coordinated': b['benefit'] / q_c, 'benefit_over_price_taker': b['benefit'] / q_pt,
              'frozen_field_benefit_relative': b['benefit_relative'],
              'frozen_benefit_relative_is_over_coordinated': abs(b['benefit_relative'] - b['benefit'] / q_c) < 1e-12,
              'pct_saving_vs_price_taker_1dp': round(100 * b['benefit'] / q_pt, 1),
              'pct_increase_over_coordinated_1dp': round(100 * b['benefit'] / q_c, 1)}
    m = files['main.tex']
    ratios['main_table_values_match_frozen'] = ('653\\,873.7' in m and '744\\,770.3' in m and '90\\,896.6' in m and
                                                round(q_c / 1e3, 1) == 653873.7 and round(q_pt / 1e3, 1) == 744770.3 and
                                                round(b['benefit'] / 1e3, 1) == 90896.6)
    cover = files['cover_letter.tex']
    return {'pattern': P139.pattern, 'files_searched': list(srcs), 'hits': hits,
            'unclassified': [h for h in hits if h['framing'] == 'UNCLASSIFIED'],
            'missing_expected': [list(k) for k in P139_EXPECTED if not any((h['file'], h['line']) == k for h in hits)],
            'called_saving': [h for h in hits if h['framing'] == 'saving'], 'ratios': ratios,
            'cover_letter_has_13_9': bool(P139.search(cover)),
            'cover_letter_operating_cost_claim': [i + 1 for i, t in enumerate(cover.splitlines())
                                                  if 'significantly reduce operating costs' in t]}


# ------------------------------------------------------------------------------------------------------- A8 (3)
def a8_abstract(files, ci, xlsx_breakdown_first):
    m = files['main.tex'].splitlines()
    a0 = next(i for i, t in enumerate(m) if '\\begin{abstract}' in t)
    a1 = next(i for i, t in enumerate(m) if '\\end{abstract}' in t)
    abstract = '\n'.join(m[a0:a1 + 1])
    tech_sentence = 'This paper proposes a two-stage stochastic framework for shared battery ESS siting, sizing and ' \
                    'installation timing under investment-cost uncertainty.'
    op_sentence = 'evaluates their operation under load, renewable-generation and market-price uncertainty.'
    low = abstract.lower()
    li_line = [i + 1 for i, t in enumerate(m) if 'Shared ESSs are modelled as utility-scale lithium-ion batteries' in t]
    traj_line = [i + 1 for i, t in enumerate(m) if 'Three investment-cost trajectories -- low, medium and high -- '
                 'represent uncertainty in future converter and battery energy-capacity costs' in t]
    le = files['response_to_reviewers_draft.tex'].splitlines()
    letter_claims = {
        'abstract_now_says_both': {'line': 76, 'ok_at_line': 'the abstract now says both.' in le[75]},
        'abstract_names_the_chemistry': {'line': 77, 'ok_at_line': 'and the abstract names the chemistry.' in le[76]},
    }
    return {'abstract_lines': [a0 + 1, a1 + 1],
            'technology_sentence': {'line': 145, 'quote': tech_sentence, 'at_line': tech_sentence in m[144]},
            'operational_uncertainty_sentence': {'line': 145, 'quote': op_sentence, 'at_line': op_sentence in m[144]},
            'abstract_technology_terms_present': {t: (t in low) for t in ABS_TECH_TERMS},
            'abstract_says_battery': 'battery' in low,
            'abstract_cost_terms_present': {t: (t in low) for t in ABS_COST_TERMS},
            'abstract_says_investment_cost_uncertainty': 'investment-cost uncertainty' in low,
            'case_technology_line': li_line, 'case_trajectories_line': traj_line,
            'cost_file_breakdown_first_row': xlsx_breakdown_first,
            'trajectories_in_file': {'n': ci['n_trajectories'], 'probabilities': ci['probabilities']},
            'letter_claims_about_abstract': letter_claims}


def xlsx_breakdown_first_row():
    import openpyxl
    wb = openpyxl.load_workbook(COST_XLSX, read_only=True, data_only=True)
    ws = wb['Cost breakdown NREL']
    rows = list(ws.iter_rows(min_row=1, max_row=2, values_only=True))
    wb.close()
    return [list(r) for r in rows]


# ------------------------------------------------------------------------------------------------------- controls
def negative_controls(files, ci, main_lines):
    out = []

    def rec(name, refused, detail):
        out.append({'control': name, 'refused': bool(refused), 'detail': detail})

    e = copy.deepcopy(A7[0])
    e['quote'] += ' X'
    r = a7_run(files, entries=[e], excluded={})
    rec('NC1 an altered A7 quote is not located', not r['entries'][0]['quote_at_lines'], None)
    f2 = dict(files)
    f2['main.tex'] = files['main.tex'].replace('\\end{abstract}', 'We certify every bound.\n\\end{abstract}', 1)
    r = a7_run(f2, entries=A7, excluded=EXCLUDED)
    rec('NC2 a planted keyword sentence in the abstract is uncovered', any('We certify every bound' in u['fragment']
                                                                           for u in r['coverage']['uncovered']), None)
    ci2 = dict(ci)
    ci2['params'] = copy.deepcopy(ci['params'])
    ci2['params'].budget = 1.6e6
    ok_hi, _ = production_check(ci2, {7: (1.25, 5.0)}, 2025)
    ok_lo, _ = production_check(ci, {7: (1.25, 5.0)}, 2025)
    rec('NC3 the production check is live: 7: 1.25/5 refused at 1e6, admitted at 1.6e6', (not ok_lo) and ok_hi,
        {'at_1e6': ok_lo, 'at_1_6e6': ok_hi})
    ok30, _ = production_check(ci, {7: (1.0, 4.0)}, 2030)
    ok25, _ = production_check(ci, {7: (1.0, 4.0)}, 2025)
    rec('NC4 7: 1/4 is over budget in 2025 and within it in 2030 (discounting is applied)', (not ok25) and ok30,
        {'2025': ok25, '2030': ok30})
    f3 = dict(files)
    f3['main.tex'] = files['main.tex'].replace('shared battery ESS siting', 'shared lithium-ion battery ESS siting', 1)
    r = a8_abstract(f3, ci, None)
    rec('NC5 a planted "lithium-ion" in the abstract is detected', r['abstract_technology_terms_present']['lithium'],
        None)
    f4 = dict(files)
    f4['highlights.tex'] = files['highlights.tex'].replace('13.9', '12.2')
    hits = [h for h in a8_139(f4, {}, json.loads(_read(FROZEN)))['hits'] if h['file'] == 'highlights.tex']
    rec('NC6 removing 13.9 from highlights.tex removes its hit', not hits, None)
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
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    ap.add_argument('--list-uncovered', action='store_true')
    args = ap.parse_args()
    out_dir = args.out_dir
    started = datetime.now(timezone.utc).isoformat()
    if not args.list_uncovered:
        entries = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else None
        if entries is None or any(e != 'launch.log' for e in entries):
            raise SystemExit(f'refusing [fresh-dir check]: {out_dir} must exist and hold nothing but launch.log '
                             f'(found {entries})')
    others = _other_copy_running()
    if others:
        raise SystemExit(f'refusing [single-copy check]: another copy is running: {others}')
    required = [CASE_JSON, ESS_PARAMS, COST_XLSX, W2_TABLE, S51_LADDER, FROZEN, AUDIT] + list(REPO_TEX.values()) + \
        list(CODE_SRC)
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise SystemExit(f'refusing [capture-path check]: missing {missing}')
    pv = {'cost_xlsx': _reg(COST_XLSX)['sha256'] == COST_XLSX_SHA, 'w2_table': _reg(W2_TABLE)['sha256'] == W2_SHA,
          'frozen': _reg(FROZEN)['sha256'] == FROZEN_SHA}
    if not all(pv.values()) or any(INPUTS[p]['status'] != 'committed' for p in (COST_XLSX, W2_TABLE, FROZEN)):
        raise SystemExit(f'refusing [input hash check]: {pv} {[(p, INPUTS[p]) for p in (COST_XLSX, W2_TABLE, FROZEN)]}')
    _read(AUDIT)
    for p in CODE_SRC:
        CODE_SRC[p] = _read(p).splitlines()
    ident, files, repo = identity()
    if not ident['ok']:
        raise SystemExit(f'refusing [identity check]: {[c for c in ident["checks"] if not c["ok"]]}')
    main_lines = files['main.tex'].splitlines()

    a7 = a7_run(files)
    if args.list_uncovered:
        for u in a7['coverage']['uncovered']:
            print(f"{u['file']}|{u['line']}|{u['fragment'][:40]!r}|{u['keywords']}|{u['fragment'][:220]}")
        print(f"uncovered {len(a7['coverage']['uncovered'])}; unused exclusions {a7['coverage']['unused_exclusions']}; "
              f"quotes located {a7['all_quotes_located']}; evidence {a7['all_evidence_located']}; "
              f"in scope {a7['all_entries_in_scope']}")
        for e in a7['entries']:
            if not (e['quote_at_lines'] and e['evidence_ok'] and e['in_scope']):
                print('  ENTRY', e['id'], e['quote_at_lines'], e['in_scope'], [x for x in e['evidence'] if not x['ok']])
        for s in a7['supplement_outside_scope']:
            if not s['quote_at_lines']:
                print('  SUPP', s['id'])
        return

    ci = case_inputs()
    w2 = json.loads(_read(W2_TABLE))
    s51 = json.loads(_read(S51_LADDER))
    frozen = json.loads(_read(FROZEN))
    budget = a8_budget(ci, main_lines, w2, s51)
    places = a8_places(files, [r for r in budget['evaluations'] if r['over_budget']])
    p139 = a8_139(files, repo, frozen)
    abstract = a8_abstract(files, ci, xlsx_breakdown_first_row())
    controls = negative_controls(files, ci, main_lines)

    hl_main = [ln.strip()[6:] for ln in main_lines[a7['regions_main_tex']['highlights (main.tex)'][0]:
                                                   a7['regions_main_tex']['highlights (main.tex)'][1] - 1]
               if ln.strip().startswith('\\item')]
    hl_std = [ln.strip()[6:] for ln in files['highlights.tex'].splitlines() if ln.strip().startswith('\\item')]
    hl_v2 = [re.sub(r'\s+%\s*\d+\s*$', '', ln.strip()[6:]).strip() for ln in repo['highlights_v2'].splitlines()
             if ln.strip().startswith('\\item')]
    hl_sub = [ln.strip()[6:] for ln in repo['submitted_highlights'].splitlines() if ln.strip().startswith('\\item')]
    highlights_copies = {'main_tex': hl_main, 'standalone': hl_std, 'review_v2_items': hl_v2, 'submitted': hl_sub,
                         'standalone_equals_main': [s.strip() for s in hl_std] == [s.strip() for s in hl_main],
                         'standalone_equals_review_v2_first5': [s.strip() for s in hl_std] ==
                         [s.strip() for s in hl_v2[:5]],
                         'standalone_equals_submitted': [s.strip() for s in hl_std] == [s.strip() for s in hl_sub]}

    harness_faults = []
    if not a7['all_quotes_located']:
        harness_faults.append('A7 quote not located')
    if not a7['all_evidence_located']:
        harness_faults.append('A7 evidence not located')
    if not a7['all_entries_in_scope']:
        harness_faults.append('A7 entry outside scope')
    if a7['coverage']['uncovered'] or a7['coverage']['unused_exclusions']:
        harness_faults.append('A7 coverage incomplete')
    if not all(c['ok'] for c in budget['checks']):
        harness_faults.append(f"A8 budget checks {[c['check'] for c in budget['checks'] if not c['ok']]}")
    if not places['all_quotes_at_lines'] or places['token_scan_main_tex']['unlisted_lines']:
        harness_faults.append(f"A8 places {places['token_scan_main_tex']['unlisted_lines']} "
                              f"{[p['id'] for p in places['places'] if not p['quote_at_line']]}")
    if p139['unclassified'] or p139['missing_expected']:
        harness_faults.append('A8 13.9 scan')
    if not all(c['refused'] for c in controls):
        harness_faults.append('negative control')
    gfail = GUARD.verify(0)
    if gfail or PICKLE_COUNTS['load'] or PICKLE_COUNTS['loads'] or harness_faults:
        raise SystemExit(f'harness fault: guard {gfail}; pickle {PICKLE_COUNTS}; {harness_faults}')

    listed = a7['counts']['listed']
    pred = {'A7': {'expected': '>= 6 sentences listed', 'listed': listed, 'borderline': a7['counts']['borderline'],
                   'held': listed >= 6, 'held_on_main_tex_alone': a7['listed_by_file'].get('main.tex', 0) >= 6}}
    result = {
        'schema': 'w182d_claims/1', 'stage': 'P5.15 W182d (Addendum 82 v2, W182 items A7 and A8)',
        'started_utc': started, 'interpreter': sys.executable, 'identity': ident,
        'objective_convention': 'no Q computed; I(x) is the expected discounted investment cost of the master '
                                '(EUR, discounted to 2025); 13.9 / 12.2 % from the frozen benchmark (gross '
                                'operational cost, settlement excluded)',
        'production_calls': list(PRODUCTION_CALLS),
        'A7': a7, 'A8_1_budget': budget, 'A8_1_places': places, 'A8_2_13_9': p139, 'A8_3_abstract': abstract,
        'highlights_copies': highlights_copies, 'negative_controls': controls, 'predictions': pred,
        'input_hash_checks': pv,
        'guards': {'solve_profile_guard': {'label': GUARD.label, 'counts': GUARD.counts, 'verify_0_failures': gfail},
                   'pickle_counts': PICKLE_COUNTS},
        'finished_utc': datetime.now(timezone.utc).isoformat(),
    }
    out_json = os.path.join(out_dir, 'w182d_claims.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(result, indent=1, sort_keys=False) + '\n')
    manifest = {'outputs': {out_json: _sha(out_json)}, 'script': {SCRIPT: _sha(SCRIPT)},
                'inputs': {p: INPUTS[p] for p in sorted(INPUTS)}, 'n_inputs': len(INPUTS),
                'n_uncommitted_inputs': sum(1 for v in INPUTS.values() if v['status'] == 'UNCOMMITTED')}
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'w', encoding='utf-8') as f:
        f.write(GRIO.dumps(manifest, indent=1, sort_keys=True) + '\n')

    print(f"repo HEAD {ident['repo_head']}; clone HEAD {ident['clone_head']}; main.tex sha256 {ident['main_sha256']}")
    print(f"A7: listed {listed}, borderline {a7['counts']['borderline']}, other {a7['counts']['other']}; by file "
          f"{a7['listed_by_file']}; by category {a7['listed_by_category']}")
    print(f"A7 coverage: covered {len(a7['coverage']['covered'])}, excluded {len(a7['coverage']['excluded'])}, "
          f"uncovered {len(a7['coverage']['uncovered'])}")
    for e in a7['entries']:
        print(f"  {e['id']:8s} {e['status']:10s} {e['file']}:{e['lines'][0]}-{e['lines'][1]} {e['cat']}")
    print('A8(1) expected discounted unit costs (EUR):')
    for y, v in budget['expected_unit_costs'].items():
        print(f"  {y}: S {v['power_discounted']:.2f} /MVA, E {v['energy_discounted']:.2f} /MWh (x {v['discount_multiplier']:.6f})")
    for x in budget['ladder_4h_node7_2025']:
        print(f"  4 h ladder 2025 {x['S_MVA']:g}/{x['E_MWh']:g}: I = {x['I_eur']:.2f}, slack {x['budget_slack_eur']:.2f}, "
              f"production ok {x['production_first_stage_ok']} {x['reasons']}")
    for r in budget['over_budget_evaluations']:
        print(f"  over budget: eval {r['eval']} (l. {r['line']}) {r['plan_text']} {r['year']} {r['condition']}: "
              f"I = {r['I_eur']:.2f}, slack {r['budget_slack_eur']:.2f}")
    for c in budget['checks']:
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"A8(1) places {places['counts']}; token-scan unlisted {places['token_scan_main_tex']['unlisted_lines']}")
    for h in places['label_phrase_scan']['hits']:
        print(f"  label phrase {h['file']}:{h['line']} '{h['match']}'")
    print(f"A8(2) 13.9 hits {[(h['file'], h['line'], h['framing']) for h in p139['hits']]}; ratios {p139['ratios']}")
    print(f"A8(3) abstract tech terms {abstract['abstract_technology_terms_present']}; cost terms "
          f"{abstract['abstract_cost_terms_present']}; letter claims {abstract['letter_claims_about_abstract']}")
    print(f"highlights: standalone = main {highlights_copies['standalone_equals_main']}; = review v2 "
          f"{highlights_copies['standalone_equals_review_v2_first5']}; = submitted "
          f"{highlights_copies['standalone_equals_submitted']}")
    for c in controls:
        print(f"  control {c['control']}: refused {c['refused']}")
    print(f"prediction A7: {pred['A7']}")
    print(f"guards: {result['guards']}")
    print(f"inputs {len(INPUTS)} ({manifest['n_uncommitted_inputs']} uncommitted)")
    print(f"output {out_json} sha256 {_sha(out_json)}")


if __name__ == '__main__':
    main()
