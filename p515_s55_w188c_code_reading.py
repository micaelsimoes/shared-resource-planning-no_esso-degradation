"""P5.15 Addendum 89, Planner task W188 (c) -- CODE READING FOR RULING 5 (zero-capacity conventions, G1 of
`manuscript_review/REFOCUS_ANALYSIS_2026-11-10.md`): the production rows for the storage complementarity condition
(network side and ESSO side), the cohort allocation and the cycling exponent, and the deactivation of inactive cohorts
and zero-capacity nodes. ZERO SOLVES, NO MODEL BUILD, NO PICKLE, NO PRODUCTION IMPORT, READ ONLY.

TWO HEADS. The production code changed after spec v1: commit f6c23977 (P5.16 convention change C1 + C3 + C4,
switch-controlled, defaults bitwise) edited shared_energy_storage_data.py, shared_energy_storage_parameters.py and
shared_resources_planning.py. Every citation is therefore read at BOTH heads:
    V1   = 6f43479e (spec v1, Addendum 85; the head on which the frozen-convention runs were made), via `git show`;
    HEAD = the repository HEAD (descending from 75d8e12a), via `git show` AND the working tree, which must be
           byte-identical to the HEAD blob (tracked, unmodified).
A citation is (id, item, file, line range at V1, line range at HEAD, substrings that must occur inside each range).
A range of None at V1 marks a HEAD-only component (a P5.16 switch); its substrings must then be ABSENT from the whole
V1 file. For every citation present at both heads the script records both excerpts verbatim with their sha256, whether
they are byte-identical, and whether either range intersects a hunk of `git diff -U0 V1 HEAD` for that file. A
citation that does not say what the report says it says is an INTEGRITY failure (exit 3).

It also records: the blobs of the SRP1 case/parameter files at both heads (the frozen instance's files must be
identical), the manuscript file the rows are compared with (`manuscript_review/main_edd7a63_round8.tex`, md5 must
start 39f6fd86 = Overleaf a87e195 per Addendum 89), and the verdicts of ruling 5 with the citations they rest on.

GUARDS. SolveProfileGuard(permitted=()) installed BEFORE any other project import and verified at exactly 0 at the end;
pickle.load / pickle.loads blocked for the whole run (counters verified 0); no production module is imported (the only
project imports are the guard and the shared writer gate_result_io). git reads only. Refuses to run beside another copy
of itself. Every JSON output goes through gate_result_io (refuses string flags).

MODE (repo root, canonical interpreter; attached, both streams captured):
    mkdir data/SRP1/Results/P515S55/w188c_code_reading && \\
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w188c_code_reading.py \\
        > data/SRP1/Results/P515S55/w188c_code_reading/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w188c_code_reading.py --post-run
Write-once: every output is opened 'x' in a directory that holds nothing but its launch log.
Exit: 0 = written, every citation and integrity check holds and the guards are at 0; 3 = written, an integrity check
failed (listed); 1 = precondition or guard fault (nothing written).
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

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W188c code reading ruling 5 (never solves)').install()

PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('W188c: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('W188c: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402  (stdlib only; the shared writer that refuses string flags)

pickle.load, pickle.loads = _blocked_load, _blocked_loads

SCRIPT_REL = os.path.basename(__file__)
S55 = os.path.join('data', 'SRP1', 'Results', 'P515S55')
OUT_DIR = os.path.join(S55, 'w188c_code_reading')
OUT_NAMES = {'json': 'w188c_code_reading.json', 'md': 'w188c_code_reading.md', 'man': 'manifest_sha256.json',
             'log': 'launch.log', 'post': 'manifest_post_run_sha256.json'}
LAUNCH_LOG_RX = re.compile(r'launch(_attempt\d+)?\.log')
V1 = '6f43479e'                    # spec v1 (Addendum 85): head of the frozen-convention runs
EXPECTED_HEAD_PREFIX = '75d8e12a'  # Addendum 89 commit; HEAD may move (parallel Workers) if no cited file changes
CONVENTION_COMMIT = 'f6c23977'     # P5.16 C1 + C3 + C4 (switch-controlled; defaults bitwise)
MANUSCRIPT = os.path.join('manuscript_review', 'main_edd7a63_round8.tex')
MANUSCRIPT_MD5_PREFIX = '39f6fd86'
# Citations whose excerpt is EXPECTED to differ between V1 and HEAD by leading whitespace only (code moved into the
# else branch of a P5.16 switch). They must be identical line by line after stripping leading whitespace.
EXPECTED_INDENT_ONLY = {'EC-07'}

SED = 'shared_energy_storage_data.py'
SEP = 'shared_energy_storage_parameters.py'
SRP = 'shared_resources_planning.py'
MCH = 'model_construction_helpers.py'
NET = 'network.py'
NETP = 'network_parameters.py'
APW = 'admm_persistent_workers.py'
DEFS = 'definitions.py'
HF = 'helper_functions.py'
C9P = 'data/SRP1/case9/case9_params.json'
C331P = 'data/SRP1/case33_1/case33_1_params.json'
C332P = 'data/SRP1/case33_2/case33_2_params.json'
C333P = 'data/SRP1/case33_3/case33_3_params.json'
SRP1J = 'data/SRP1/SRP1.json'
ESSP = 'data/SRP1/SharedESS/SRP1_ESS_Params.json'
INSTANCE_FILES = [SRP1J, ESSP, C9P, C331P, C332P, C333P, 'data/SRP1/SRP1_params.json']

# ======================================================================================================================
#  THE CITATIONS: (id, item, file, (V1 first, last) or None, (HEAD first, last), substrings, what it shows)
#  item: N = network side, E = ESSO side, C = complementarity, A = allocation, X = cycling exponent,
#        Z = zero capacity / inactive, S = P5.16 switches, M = manuscript (HEAD file only, compared, not code)
# ======================================================================================================================
CLAIMS = [
    # ------------------------------------------------------------------ (1) complementarity, network side
    ('NC-01', 'C-net', NET, (393, 396), (393, 396),
     ['model.shared_es_s_rated_fixed = pe.Param(model.shared_energy_storages, mutable=True',
      'model.shared_es_pch = pe.Var(', 'model.shared_es_pdch = pe.Var(', 'domain=pe.NonNegativeReals'],
     'Network copy: capacity S is a mutable Param (shared_es_s_rated_fixed); P^Ch, P^Dch are non-negative Vars.'),
    ('NC-02', 'C-net', NET, (397, 402), (397, 402),
     ['used ONLY by the complementarity row',
      'model.shared_es_pch_hat = pe.Var(', 'model.shared_es_pdch_hat = pe.Var(', 'bounds=(0.0, 1.0)'],
     'Dimensionless pair pch_hat, pdch_hat in [0, 1], used only by the complementarity row.'),
    ('NC-03', 'C-net', NET, (500, 503), (500, 503),
     ['model.sess_pch_hat_link = pe.Constraint(', 'model.sess_pdch_hat_link = pe.Constraint(',
      'model.sess_converter_capability = pe.Constraint(', 'model.sess_active_sum_limit = pe.Constraint('],
     'Rows wired in every network model (TSO and DSO): the two hat links, the converter circle, the active sum limit.'),
    ('NC-04', 'C-net', NET, (512, 512), (512, 512),
     ['model.sess_comp = pe.Constraint(', 'rule=partial(sess_comp_rule, network=network, params=params)'],
     'The complementarity row sess_comp, one per (e, s_m, s_o, t), built from the network parameters.'),
    ('NC-05', 'C-net', MCH, (874, 887), (874, 887),
     ['def sess_active_sum_limit_rule(m, e, s_m, s_o, p):',
      '+ m.shared_es_pdch[e, s_m0, s_o0, p]) <= m.shared_es_s_rated_fixed[e]'],
     'Power bound P^Ch + P^Dch <= S (eq. storage_active_power_limit).'),
    ('NC-06', 'C-net', MCH, (906, 930), (906, 930),
     ['def sess_pch_hat_link_rule(m, e, s_m, s_o, p):', 'no expression ever divides by the rated-capacity',
      '- m.shared_es_s_rated_fixed[e] * m.shared_es_pch_hat[e, s_m0, s_o0, p]) == 0',
      'def sess_pdch_hat_link_rule(m, e, s_m, s_o, p):',
      '- m.shared_es_s_rated_fixed[e] * m.shared_es_pdch_hat[e, s_m0, s_o0, p]) == 0'],
     'Linear links P^Ch - S * pch_hat = 0 and P^Dch - S * pdch_hat = 0 (no division by S).'),
    ('NC-07', 'C-net', MCH, (933, 955), (933, 955),
     ['def sess_comp_rule(m, e, s_m, s_o, p, network, params):',
      '`pch * pdch <= eps * S_rated^2` -- substituting the link equalities',
      'reproduces it identically',
      'elif params.shared_ess_model == ESS_MODEL_BILINEAR_RELAXATION:',
      '* m.shared_es_pdch_hat[e, s_m0, s_o0, p]) <= ESS_COMPLEMENTARITY_TOLERANCE'],
     'The complementarity row in force (BILINEAR_RELAXATION branch): pch_hat * pdch_hat <= ESS_COMPLEMENTARITY_TOLERANCE.'),
    ('NC-08', 'C-net', DEFS, (34, 36), (34, 36),
     ['ESS_MODEL_EXACT = 1', 'ESS_MODEL_BILINEAR_RELAXATION = 2', 'ESS_MODEL_POLYNOMIAL_COMPLEMENTARITY = 3'],
     'Storage model codes.'),
    ('NC-09', 'C-net', DEFS, (91, 93), (91, 93),
     ['SMALL_TOLERANCE = 1e-4', 'ESS_COMPLEMENTARITY_TOLERANCE = SMALL_TOLERANCE',
      'SHARED_ESS_ZERO_CAPACITY_TOLERANCE = 1e-10'],
     'eps^C = 1e-4; the zero-capacity tolerance 1e-10 (network and ESSO tests).'),
    ('NC-10', 'C-net', NETP, (172, 178), (172, 178),
     ["if 'shared_ess_model' in params_data:",
      "elif params_data['shared_ess_model'] == 'BILINEAR_RELAXATION':",
      'parameters.shared_ess_model = ESS_MODEL_BILINEAR_RELAXATION'],
     'The shared-storage model is read per network from its params file.'),
    ('NC-11', 'C-net', C9P, (11, 11), (11, 11), ['"shared_ess_model": "BILINEAR_RELAXATION"'],
     'TSO (case9) uses the BILINEAR_RELAXATION branch.'),
    ('NC-12', 'C-net', C331P, (11, 11), (11, 11), ['"shared_ess_model": "BILINEAR_RELAXATION"'],
     'DSO case33_1 uses the BILINEAR_RELAXATION branch.'),
    ('NC-13', 'C-net', C332P, (11, 11), (11, 11), ['"shared_ess_model": "BILINEAR_RELAXATION"'],
     'DSO case33_2 uses the BILINEAR_RELAXATION branch.'),
    ('NC-14', 'C-net', C333P, (11, 11), (11, 11), ['"shared_ess_model": "BILINEAR_RELAXATION"'],
     'DSO case33_3 uses the BILINEAR_RELAXATION branch.'),
    ('NC-15', 'C-net', APW, (333, 339), (333, 339),
     ["shared_es_s_rated_fixed[shared_ess_idx].set_value(sess_capacity_year['s_available'] / s_base)",
      "shared_es_e_rated_fixed[shared_ess_idx].set_value(sess_capacity_year['e_available'] / s_base)",
      'configure_shared_ess_operational_state('],
     'The S in the network rows is the ESSO available capacity S^Av (per node and year, p.u.), set per ADMM cycle.'),
    ('NC-16', 'C-net', SRP, (6114, 6121), (6132, 6139),
     ["shared_es_s_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['s_available'] / s_base)",
      'configure_shared_ess_operational_state('],
     'Same assignment of S^Av on the in-process path (shifted +18 lines at HEAD by f6c23977).'),
    ('NC-17', 'C-net', SED, (274, 279), (278, 283),
     ['def get_available_capacities(self, model, year_idx):',
      's_available += pe.value(model.es_s_available_per_unit[y_inv, year_idx])'],
     'S^Av sent to the networks = sum over cohorts of es_s_available_per_unit.'),
    ('NC-18', 'C-net', SED, (615, 619), (619, 623),
     ['model.available_s_capacity_unit.add(model.es_s_available_per_unit[y_inv, y] == model.es_s_rated_per_unit[y_inv, y])'],
     'Available S per cohort-year = rated S per cohort-year (no SoH on power).'),
    # ------------------------------------------------------------------ (1) complementarity, ESSO side
    ('EC-01', 'C-esso', SED, (470, 478), (474, 482),
     ['model.es_pch_per_unit = pe.Var(', 'model.es_pdch_per_unit = pe.Var(',
      'complementarity rows (`energy_storage_complementarity` and the aggregate',
      'all retired. Complementarity is no longer enforced by a constraint'],
     'ESSO side: NO complementarity row. The per-cohort and aggregate complementarity rows and the hat pair are retired.'),
    ('EC-02', 'C-esso', SED, (735, 738), (739, 742),
     ['`energy_storage_complementarity`', 'retired'],
     'ESSO side: energy_storage_complementarity, its normalization and the hat links are retired (second statement).'),
    ('EC-03', 'C-esso', SED, (807, 812), (811, 816),
     ['the aggregate complementarity link', 'is retired along with',
      'the LP throughput regularization in `feasibility_penalty`'],
     'ESSO side: the aggregate complementarity link is retired too.'),
    ('EC-04', 'C-esso', SED, (831, 844), (835, 848),
     ['throughput += model.es_pch_per_unit[y_inv, y, d, p] + model.es_pdch_per_unit[y_inv, y, d, p]',
      'slack_penalty += EPS_ESSO_THROUGHPUT * throughput'],
     'ESSO side: complementarity is replaced by the objective term eps^E * sum(P^Ch + P^Dch).'),
    ('EC-05', 'C-esso', DEFS, (74, 74), (74, 74), ['EPS_ESSO_THROUGHPUT = 1e-5'], 'eps^E = 1e-5.'),
    ('EC-06', 'C-esso', SED, (739, 749), (743, 753),
     ['model.energy_storage_limits = pe.ConstraintList()', 's_max = model.es_s_rated_per_unit[y_inv, y]',
      "_add_esso_cohort_constraint(model, 'energy_storage_limits', y_inv, y, pch <= s_max)",
      "_add_esso_cohort_constraint(model, 'energy_storage_limits', y_inv, y, pdch <= s_max)"],
     'ESSO power bounds per cohort: P^Ch <= S^Rated,Unit, P^Dch <= S^Rated,Unit (cohort-registered).'),
    ('EC-07', 'C-esso', SED, (851, 854), (875, 878),
     ['model.objective = pe.Objective(', 'sense=pe.minimize,', 'expr=model.feasibility_penalty,'],
     'ESSO objective with the C1 switch off (default; the only form at V1): feasibility_penalty alone. At HEAD it '
     'sits in the else branch of the C1 switch, so it differs from V1 by indentation only (EXPECTED_INDENT_ONLY).'),
    # ------------------------------------------------------------------ (2) cohort allocation
    ('A-01', 'A', SED, (560, 570), (564, 574),
     ['for y in range(y_inv, max_tcal_norm):',
      'model.rated_e_capacity_unit.add(model.es_e_rated_per_unit[y_inv, y] == model.es_e_investment_fixed[y_inv])'],
     'Within the calendar-life window E^Rated,Unit[y_inv, y] = E^Inv[y_inv] (a Param).'),
    ('A-02', 'A', SED, (466, 468), (470, 472),
     ['model.es_s_rated_per_unit.fix(0.00)', 'model.es_e_rated_per_unit.fix(0.00)'],
     'Outside the window E^Rated,Unit and S^Rated,Unit stay fixed at 0 (the lifetime test reads this fixing).'),
    ('A-03', 'A', SED, (760, 761), (764, 765),
     ['model.es_pnet_cohort_share_h3 = pe.Param(model.years, model.years, mutable=True, initialize=0.00)',
      'model.energy_storage_cohort_pnet_share_h3 = pe.ConstraintList()'],
     'Allocation share is a mutable Param (initialised 0), not an expression in Vars.'),
    ('A-04', 'A', SED, (778, 784), (782, 788),
     ['agg_pnet += (model.es_pch_per_unit[y_inv, y, d, p] - model.es_pdch_per_unit[y_inv, y, d, p])',
      'model.es_pnet[y, d, p] == agg_pnet + model.slack_es_pnet_up[y, d, p] - model.slack_es_pnet_down[y, d, p]'],
     'agg_pnet = sum over cohorts of (P^Ch - P^Dch), without the sigma slacks; es_pnet carries the slacks.'),
    ('A-05', 'A', SED, (801, 805), (805, 809),
     ["model, 'energy_storage_cohort_pnet_share_h3', y_inv, y,",
      'model.es_pch_per_unit[y_inv, y, d, p] - model.es_pdch_per_unit[y_inv, y, d, p]',
      '== model.es_pnet_cohort_share_h3[y_inv, y] * agg_pnet)'],
     'Allocation row: P^Ch[y_inv] - P^Dch[y_inv] = share[y_inv, y] * agg_pnet, built for every y_inv, cohort-registered.'),
    ('A-06', 'A', SED, (1427, 1447), (1523, 1543),
     ['def _update_model_with_candidate_solution(shared_ess_data, models, candidate_solution):',
      'model.es_e_investment_fixed[y_inv].set_value(e_candidate)',
      '_configure_esso_cohort_state(model, y_inv, s_candidate, e_candidate, shared_ess_data.params.slacks)',
      '_configure_esso_cohort_pnet_share_rows(model)'],
     'Per candidate: each cohort state is set first, then the allocation rows and shares once per node.'),
    ('A-07', 'A', SED, (1639, 1673), (1735, 1769),
     ['def _configure_esso_cohort_pnet_share_rows(model):',
      'exactly\n    N_active(y)-1 rows are left ACTIVE per (y, d, p) -- never N.',
      'the one with the LARGEST y_inv among that year\'s active cohorts',
      'every\n    share written here is a plain PYTHON FLOAT'],
     'Allocation docstring: N_active - 1 rows active (largest y_inv omitted; sum-to-one supplies it); shares are floats.'),
    ('A-08', 'A', SED, (1702, 1734), (1798, 1830),
     ['active_cohorts = [', "if (not model._esso_cohort_inactive.get(y_inv, False))",
      'and _esso_cohort_pair_is_within_lifetime(model, y_inv, y)',
      'if n_active <= 1:', 'model.energy_storage_cohort_pnet_share_h3[constraint_idx].deactivate()',
      'pe.value(model.es_e_investment_fixed[y_inv]) for y_inv in active_cohorts',
      'omitted = max(active_cohorts)',
      'if y_inv in active_cohorts and total_e_rated > 0.00:',
      'share = pe.value(model.es_e_investment_fixed[y_inv]) / total_e_rated',
      'enable_row = (y_inv in active_cohorts) and (y_inv != omitted)'],
     'Guard: the only division E^Inv / sum_active E^Inv is a float, reached only with >= 2 active cohorts and a '
     'positive sum; with <= 1 active cohort every allocation row of the year is deactivated and every share set 0.'),
    # ------------------------------------------------------------------ (2) cycling exponent
    ('X-01', 'X', SED, (652, 666), (656, 670),
     ['avg_ch_dch += (num_days / 365.00) * (eff_ch * pch * dt + pdch * dt / eff_dch)',
      "_add_esso_cohort_constraint(model, 'energy_storage_charging_discharging', y_inv, y, "
      "model.es_avg_ch_dch_per_unit[y_inv, y] == avg_ch_dch)"],
     'Cell throughput row (eq. cell_throughput), cohort-registered, built for every (y_inv, y).'),
    ('X-02', 'X', SED, (675, 689), (679, 693),
     ['model.energy_storage_capacity_degradation = pe.ConstraintList()',
      'num_years = shared_ess_data.years[repr_years[y_inv]]',
      'for y in range(y_inv, max_tcal_norm):', 'model.es_soh_per_unit_cumul[y_inv, y].fixed = False'],
     'Degradation rows exist only inside the calendar window; Y is read from the INVESTMENT block of the cohort.'),
    ('X-03', 'X', SED, (699, 711), (703, 715),
     ['if ageing_enabled:',
      'model.es_D_per_unit[y_inv, y] * (2 * shared_energy_storage.cl_eff * model.es_e_investment_fixed[y_inv])',
      '== 365.00 * num_years * model.es_avg_ch_dch_per_unit[y_inv, y])',
      'model.es_D_per_unit[y_inv, y] == 0.00)'],
     'Cycling exponent in product form: D * (2 k E^Inv) = 365 Y E^Cell (ageing on); D == 0 (ageing off).'),
    ('X-04', 'X', SED, (721, 732), (725, 736),
     ['== prev_soh * pe.exp(-model.es_D_per_unit[y_inv, y]) * (phi_cal ** num_years))',
      'model.es_soh_per_unit_cumul[y_inv, y] >= shared_energy_storage.soh_min)'],
     'SoH row and floor row: cohort-registered, same family as the D row.'),
    # ------------------------------------------------------------------ (3) inactive cohorts, ESSO side
    ('ZE-01', 'Z-esso', SED, (431, 437), (435, 441),
     ['model._esso_cohort_constraints = {', 'model._esso_cohort_inactive = {'],
     'Per-cohort registry of rows and of the inactive flag.'),
    ('ZE-02', 'Z-esso', SED, (1566, 1570), (1662, 1666),
     ['def _add_esso_cohort_constraint(model, constraint_name, y_inv, y, expr):',
      'model._esso_cohort_constraints[y_inv].append((constraint_name, constraint_idx, y))'],
     'Every row added through _add_esso_cohort_constraint is registered to its cohort.'),
    ('ZE-03', 'Z-esso', SED, (1581, 1583), (1677, 1679),
     ['def _esso_cohort_pair_is_within_lifetime(model, y_inv, y):',
      'return (not model.es_s_rated_per_unit[y_inv, y].fixed and not model.es_e_rated_per_unit[y_inv, y].fixed)'],
     'Window test: a (y_inv, y) pair is in the window iff its rated per-unit Vars were unfixed at build.'),
    ('ZE-04', 'Z-esso', SED, (1586, 1636), (1682, 1732),
     ['tolerance = SHARED_ESS_ZERO_CAPACITY_TOLERANCE', 'if s_is_zero != e_is_zero:',
      "raise ValueError(f'Inconsistent shared ESS cohort capacity: S={s_capacity}, E={e_capacity}.')",
      'inactive = s_is_zero and e_is_zero', 'active_pair = (not inactive and within_lifetime)',
      '_set_esso_variable_state(model.es_avg_ch_dch_per_unit[y_inv, y], active_pair, 0.0)',
      '_set_esso_variable_state(model.es_D_per_unit[y_inv, y], active_pair, 0.0)',
      '_set_esso_variable_state(model.es_soh_per_unit_cumul[y_inv, y], active_pair, 1.0)',
      '_set_esso_variable_state(model.es_pch_per_unit[y_inv, y, d, p], active_pair, 0.0)',
      '_set_esso_variable_state(model.es_pdch_per_unit[y_inv, y, d, p], active_pair, 0.0)',
      'for constraint_name, constraint_idx, y in (model._esso_cohort_constraints[y_inv]):',
      'active_constraint = (not inactive and _esso_cohort_pair_is_within_lifetime(model, y_inv, y))',
      'constraint.deactivate()'],
     'Cohort inactive iff |S^Inv| <= 1e-10 and |E^Inv| <= 1e-10 (one without the other raises). For an inactive cohort '
     'or a pair outside the window: throughput, D, P^Ch, P^Dch fixed 0, SoH fixed 1; every registered row deactivated.'),
    ('ZE-05', 'Z-esso', SED, (550, 550), (554, 554), ['model.es_soh_per_unit_cumul.fix(1.00)'],
     'SoH is built fixed at 1.0 (unfixed only inside the window).'),
    ('ZE-06', 'Z-esso', SED, (573, 582), (577, 586),
     ['model.rated_s_capacity.add(model.es_s_rated[y] == total_s_capacity)',
      'model.rated_e_capacity.add(model.es_e_rated[y] == total_e_capacity)'],
     'NOT cohort-registered: node-level rated S and E rows (stay active at zero capacity; they then read 0 = 0 sums).'),
    ('ZE-07', 'Z-esso', SED, (615, 628), (619, 632),
     ['model.available_s_capacity_unit = pe.ConstraintList()', 'model.available_e_capacity_unit = pe.ConstraintList()',
      'model.es_e_available_per_unit[y_inv, y] == model.es_e_rated_per_unit[y_inv, y] * model.es_soh_per_unit_cumul[y_inv, y]'],
     'NOT cohort-registered: available S/E per cohort-year rows (built for every pair; stay active).'),
    ('ZE-08', 'Z-esso', SED, (764, 787), (768, 791),
     ['model.energy_storage_operation_agg = pe.ConstraintList()',
      'model.es_pnet[y, d, p] ** 2 + model.es_qnet[y, d, p] ** 2 <= model.es_s_rated[y] ** 2)'],
     'NOT cohort-registered: node-level net-power definition (with sigma slacks) and converter circle; they stay '
     'active at zero capacity, where the circle has radius es_s_rated[y] = 0.'),
    ('ZE-09', 'Z-esso', SED, (157, 158), (157, 158),
     ['def update_model_with_candidate_solution(self, models, candidate_solution):',
      '_update_model_with_candidate_solution(self, models, candidate_solution)'],
     'Public entry that applies the cohort states for a candidate.'),
    ('ZE-10', 'Z-esso', HF, (10, 10), (10, 10), ['from definitions import *'],
     'SHARED_ESS_ZERO_CAPACITY_TOLERANCE reaches shared_energy_storage_data through helper_functions (import *).'),
    # ------------------------------------------------------------------ (3) zero-capacity nodes, network side
    ('ZN-01', 'Z-net', MCH, (1002, 1013), (1002, 1013),
     ['_SHARED_ESS_OPERATIONAL_VARIABLES = (', "'shared_es_pch',", "'shared_es_pdch',", "'shared_es_pch_hat',",
      "'shared_es_pdch_hat',", "'shared_es_pnet',", "'shared_es_qnet',", "'shared_es_soc',",
      "'slack_shared_es_soc_final_up',", "'slack_shared_es_soc_final_down',"],
     'The nine network storage variables gated at zero capacity (hats included).'),
    ('ZN-02', 'Z-net', MCH, (1043, 1048), (1043, 1048),
     ['_SHARED_ESS_ZERO_GATED_BOUND_VARIABLES = (', "('shared_es_pch', 0.0, None)", "('shared_es_qnet', None, None)"],
     'Variables whose box collapses to [0, 0] at zero capacity.'),
    ('ZN-03', 'Z-net', MCH, (1050, 1064), (1050, 1064),
     ['_SHARED_ESS_OPERATIONAL_CONSTRAINTS = (', "'sess_pnet_def',", "'sess_pch_hat_link',", "'sess_pdch_hat_link',",
      "'sess_converter_capability',", "'sess_active_sum_limit',", "'sess_soc_def',", "'sess_soc_limit_upper',",
      "'sess_soc_limit_lower',", "'sess_soc_final',", "'sess_comp',"],
     'The ten network storage rows gated at zero capacity (sess_comp included).'),
    ('ZN-04', 'Z-net', MCH, (1067, 1078), (1067, 1078),
     ['def shared_ess_capacity_is_inactive(s_capacity, e_capacity):',
      'return abs(s_capacity) <= tolerance or abs(e_capacity) <= tolerance',
      'def normalize_shared_ess_capacity(capacity):', 'return 0.0 if abs(capacity) <= tolerance else capacity'],
     'Network node inactive iff S^Av <= 1e-10 OR E^Av <= 1e-10; values within the tolerance are normalised to 0.'),
    ('ZN-05', 'Z-net', MCH, (1117, 1202), (1117, 1202),
     ['def configure_shared_ess_operational_state(', 'inactive = shared_ess_capacity_is_inactive(s_capacity, e_capacity)',
      'entry.fix(0.0)', "slack_ub = 0.0 if inactive else e_capacity * ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE",
      'for constraint_name in _SHARED_ESS_OPERATIONAL_CONSTRAINTS:', 'entry.deactivate()',
      'entry.setlb(0.0)\n                entry.setub(0.0)', '_configure_shared_ess_expected_schedule(',
      'return inactive'],
     'At an inactive node: the nine variables fixed 0, the closure slacks bounded [0, 0], the ten rows deactivated, '
     'the gated boxes [0, 0], the expected-schedule variables fixed 0 and their rows deactivated.'),
    ('ZN-06', 'Z-net', MCH, (1088, 1114), (1088, 1114),
     ['def _configure_shared_ess_expected_schedule(model, shared_ess_idx, inactive):',
      "for variable_name in ('expected_shared_ess_p', 'expected_shared_ess_q'):", 'variable.fix(0.0)',
      "for constraint_name in ('expected_shared_ess_p_def', 'expected_shared_ess_q_def'):", 'entry.deactivate()'],
     'Expected storage schedule (consensus copy) fixed at 0 and its definition rows deactivated at an inactive node.'),
    # ------------------------------------------------------------------ P5.16 switches (HEAD only) and their defaults
    ('S-01', 'S', SEP, None, (19, 24),
     ['P5.16 convention change C1', 'self.in_objective = False'],
     'C1 switch salvage_value.in_objective: default False (HEAD only).'),
    ('S-02', 'S', SEP, None, (246, 249),
     ['P5.16 convention change C3', 'self.available_energy_soh_point = None'],
     'C3 switch available_energy_soh_point: default None = not declared (HEAD only).'),
    ('S-03', 'S', SED, None, (176, 179),
     ['P5.16 convention change C3', 'if self.params.available_energy_soh_point is not None:',
      'self.available_energy_soh_point = self.params.available_energy_soh_point'],
     'C3 is applied only when the ESS parameters file declares it (HEAD only).'),
    ('S-04', 'S', SED, None, (855, 873),
     ["if getattr(shared_ess_data.params.salvage_value, 'in_objective', False):",
      'model.salvage_objective_kappa = pe.Param(mutable=True, initialize=0.00)',
      'expr=model.feasibility_penalty + model.salvage_objective_kappa * model.salvage_credit_obj,'],
     'C1 changes the ESSO OBJECTIVE only (no row); off by default (HEAD only).'),
    # ------------------------------------------------------------------ manuscript (compared, HEAD file only)
    ('M-01', 'M', MANUSCRIPT, None, (946, 952),
     ['normalised complementarity condition relaxed with tolerance $\\varepsilon^{\\text{C}}$',
      '\\frac{P^{\\text{Ch}}_{e,y,d,t}}{S^{\\text{Av}}_{e,y}} \\; \\frac{P^{\\text{Dch}}_{e,y,d,t}}{S^{\\text{Av}}_{e,y}} \\le \\varepsilon^{\\text{C}}',
      '\\label{eq:network_complementarity}'],
     'Manuscript at a87e195: the normalised complementarity (divides by S^Av).'),
    ('M-02', 'M', MANUSCRIPT, None, (918, 925),
     ['P^{\\text{Ch}}_{e,y,d,t}', '+ P^{\\text{Dch}}_{e,y,d,t}', '\\le S^{\\text{Av}}_{e,y}',
      '\\label{eq:storage_active_power_limit}'],
     'Manuscript: the power bound that makes the product form vacuous at S^Av = 0.'),
    ('M-03', 'M', MANUSCRIPT, None, (984, 996),
     ['\\frac{E^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y}}{\\sum_{y\'} E^{\\text{Rated,Unit}}_{e,y\',y}}',
      '\\label{eq:cohort_allocation}', 'The sums extend over $\\mathcal{Y}^{\\text{Act}}_{e,y}$.'],
     'Manuscript: the cohort allocation (divides by the sum of E^Rated,Unit), stated for all active cohorts.'),
    ('M-04', 'M', MANUSCRIPT, None, (1019, 1026),
     ['{2 \\, k_e \\, E^{\\text{Rated,Unit}}_{e,y^\\text{Inv},y}}', '\\label{eq:cycling_loss}',
      '\\frac{365 \\, Y_y \\, \\bar{E}^{\\text{Cell}}_{e,y^\\text{Inv},y}}'],
     'Manuscript: the cycling exponent (divides by E^Rated,Unit), with Y_y.'),
    ('M-05', 'M', MANUSCRIPT, None, (1914, 1914),
     ['complementarity in the agent follows from the problem structure and needs no nonconvex constraint'],
     'Manuscript (Appendix A): ESSO-side complementarity is structural, not a row -- consistent with EC-01..04.'),
    ('M-06', 'M', MANUSCRIPT, None, (2117, 2117),
     ['The normalized network complementarity condition uses $\\varepsilon^{\\text{C}} = 10^{-4}$.',
      '$\\varepsilon^{\\text{E}} = 10^{-5}$'],
     'Manuscript: eps^C = 1e-4, eps^E = 1e-5 (= NC-09, EC-05).'),
]

# Verdicts of ruling 5, each with the citations it rests on (code reading; no solve).
VERDICTS = [
    {'id': 'V1', 'question': '(1) network complementarity row vs P^Ch P^Dch <= eps^C (S^Av)^2',
     'verdict': 'POSITIVE SCALING ONLY. The coded row is pch_hat * pdch_hat <= eps^C (eps^C = 1e-4) with the linear '
                'links P^Ch = S^Av pch_hat, P^Dch = S^Av pdch_hat and pch_hat, pdch_hat in [0, 1]. Substituting the '
                'links gives P^Ch P^Dch = (S^Av)^2 pch_hat pdch_hat, so for S^Av > 0 the coded row is the product form '
                'divided by (S^Av)^2 (a positive constant: S^Av is a Param in every network solve). The hat bounds '
                'add nothing beyond P^Ch + P^Dch <= S^Av. The scaling is carried by two auxiliary variables and two '
                'linear rows, not by multiplying the product row by a constant; the projection onto (P^Ch, P^Dch) is '
                'identical. At S^Av <= 1e-10 (or E^Av <= 1e-10) the row is not merely vacuous: it is deactivated and '
                'P^Ch, P^Dch, the hats and every other storage variable are fixed at 0.',
     'basis': ['NC-01', 'NC-02', 'NC-05', 'NC-06', 'NC-07', 'NC-09', 'NC-11', 'NC-12', 'NC-13', 'NC-14', 'NC-15',
               'NC-17', 'NC-18', 'ZN-03', 'ZN-04', 'ZN-05']},
    {'id': 'V2', 'question': '(1) ESSO-side complementarity row',
     'verdict': 'NONE EXISTS. The ESSO has no complementarity row at either head (retired in P5.15-1); simultaneous '
                'charging and discharging is discouraged by eps^E sum(P^Ch + P^Dch) in the objective. The product form '
                'of ruling 5 applies to eq. network_complementarity only; it must not be printed for the agent.',
     'basis': ['EC-01', 'EC-02', 'EC-03', 'EC-04', 'EC-05', 'M-05']},
    {'id': 'V3', 'question': '(2) cohort allocation and its guard',
     'verdict': 'The row is P^Ch[y_inv] - P^Dch[y_inv] = share * sum_cohorts (P^Ch - P^Dch) (sigma slacks excluded), '
                'with share = E^Inv[y_inv] / sum_active E^Inv a Python float (E^Inv = E^Rated,Unit inside the window). '
                'Division only when >= 2 cohorts are active and the sum is > 0; with <= 1 active cohort every '
                'allocation row of the year is deactivated and the shares are 0. Exactly N_active - 1 rows are active '
                '(largest y_inv omitted, supplied by the sum-to-one identity) where the manuscript prints N; '
                'equivalent. Inert on SRP1 (one cohort).',
     'basis': ['A-01', 'A-02', 'A-03', 'A-04', 'A-05', 'A-06', 'A-07', 'A-08', 'M-03']},
    {'id': 'V4', 'question': '(2) cycling exponent and its guard',
     'verdict': 'Coded in product form D * (2 k E^Inv) = 365 Y E^Cell (ageing on) or D = 0 (ageing off): the '
                'manuscript equation multiplied by the positive constant 2 k E^Rated,Unit for an active cohort; no '
                'division is coded. Built only inside the calendar window; deactivated (D fixed 0, SoH fixed 1) for '
                'an inactive cohort. Y is the investment block width of the cohort, which equals Y_y when all blocks '
                'have equal width (SRP1: 5-year blocks).',
     'basis': ['X-01', 'X-02', 'X-03', 'X-04', 'A-01', 'ZE-04', 'M-04']},
    {'id': 'V5', 'question': '(3) deactivation of inactive cohorts / zero-capacity nodes',
     'verdict': 'Network side: complete -- the ten storage rows deactivated and the storage variables fixed at 0 when '
                'S^Av or E^Av <= 1e-10. ESSO side: every cohort-registered row (throughput, D, SoH, floor, power '
                'bounds, allocation) is deactivated and its variables fixed for an inactive cohort; the node-level '
                'rows (rated S/E sums, available S/E per pair, net-power definition with sigma slacks, converter '
                'circle with radius es_s_rated = 0) are NOT deactivated -- they stay active and are satisfied at zero.',
     'basis': ['ZN-01', 'ZN-02', 'ZN-03', 'ZN-04', 'ZN-05', 'ZN-06', 'ZE-01', 'ZE-02', 'ZE-03', 'ZE-04', 'ZE-05',
               'ZE-06', 'ZE-07', 'ZE-08']},
    {'id': 'V6', 'question': 'differences between V1 (6f43479e) and HEAD for the cited rows',
     'verdict': 'Computed below (head_comparison): every row excerpt present at both heads is byte-identical '
                '(EC-07, the ESSO objective, identical modulo indentation, moved into the else branch of C1); '
                'f6c23977 shifts line numbers (+4 / +24 / +96 in shared_energy_storage_data.py, +18 in '
                'shared_resources_planning.py past l. 5533) and adds the C1/C3 switches, which with their defaults '
                '(in_objective False, available_energy_soh_point None) construct nothing.',
     'basis': ['S-01', 'S-02', 'S-03', 'S-04', 'EC-07']},
]


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


def _git_bytes(*args):
    r = subprocess.run(['git', *args], cwd=REPO, capture_output=True)
    return r.returncode, r.stdout


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


def _blob(rev, rel):
    rc, out, _ = _git('rev-parse', f'{rev}:{rel}')
    return out if rc == 0 else None


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


def diff_hunks(rel):
    """[(v1_first, v1_last), (head_first, head_last)] for every hunk of `git diff -U0 V1 HEAD -- rel`
    (an empty side is recorded as (n, n-1), i.e. a zero-length range after line n)."""
    _, out, _ = _git('diff', '-U0', V1, 'HEAD', '--', rel)
    hunks = []
    for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', out, flags=re.M):
        a, b = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
        c, d = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
        hunks.append({'v1': [a, a + b - 1], 'head': [c, c + d - 1], 'v1_len': b, 'head_len': d})
    return hunks


def _intersects(rng, hunk_rng, length):
    if length == 0:
        return False
    return not (rng[1] < hunk_rng[0] or rng[0] > hunk_rng[1])


# ======================================================================================================================
#  sections
# ======================================================================================================================
def check_claims():
    texts = {}
    out = []
    failures = []

    def text(rev, rel):
        key = (rev, rel)
        if key not in texts:
            rc, raw = _git_bytes('show', f'{rev}:{rel}')
            texts[key] = {'rc': rc, 'sha256': _sha_bytes(raw) if rc == 0 else None,
                          'lines': raw.decode('utf-8').split('\n') if rc == 0 else []}
        return texts[key]

    hunks_cache = {}
    at_head = {}
    for cid, item, rel, r1, rh, needles, what in CLAIMS:
        if rel not in at_head:
            at_head[rel] = _file_at_head(rel)
            hunks_cache[rel] = diff_hunks(rel)
            wt = _read_bytes(rel)
            th = text('HEAD', rel)
            at_head[rel]['worktree_sha256_equals_head_show'] = (_sha_bytes(wt) == th['sha256'])
        th = text('HEAD', rel)
        t1 = text(V1, rel)
        ex_h_lines = th['lines'][rh[0] - 1:rh[1]] if 1 <= rh[0] <= rh[1] <= len(th['lines']) else []
        ex_h = '\n'.join(ex_h_lines)
        miss_h = [n for n in needles if n not in ex_h]
        rec = {'id': cid, 'item': item, 'file': rel, 'statement': what, 'required_substrings': needles,
               'head': {'line_first': rh[0], 'line_last': rh[1], 'excerpt': ex_h,
                        'excerpt_sha256': _sha_bytes(ex_h.encode('utf-8')), 'missing_substrings': miss_h,
                        'file_sha256': th['sha256']}}
        ok = bool(ex_h_lines) and not miss_h and at_head[rel]['ok'] and at_head[rel]['worktree_sha256_equals_head_show']
        if r1 is None:
            # HEAD-only component: none of its substrings may occur anywhere in the V1 file
            v1_full = '\n'.join(t1['lines'])
            present = [n for n in needles if t1['rc'] == 0 and n in v1_full]
            rec['v1'] = {'present': False, 'file_exists_at_v1': t1['rc'] == 0, 'file_sha256': t1['sha256'],
                         'substrings_found_in_v1_file': present}
            rec['identical_at_both_heads'] = None
            rec['range_touches_diff_hunk'] = None
            if rel != MANUSCRIPT:
                # a P5.16 switch must be absent from the whole V1 file (the manuscript is not code: not checked)
                ok = ok and present == []
        else:
            ex_1_lines = t1['lines'][r1[0] - 1:r1[1]] if 1 <= r1[0] <= r1[1] <= len(t1['lines']) else []
            ex_1 = '\n'.join(ex_1_lines)
            miss_1 = [n for n in needles if n not in ex_1]
            rec['v1'] = {'present': True, 'line_first': r1[0], 'line_last': r1[1], 'excerpt': ex_1,
                         'excerpt_sha256': _sha_bytes(ex_1.encode('utf-8')), 'missing_substrings': miss_1,
                         'file_sha256': t1['sha256']}
            identical = ex_1 == ex_h
            rec['identical_modulo_leading_whitespace'] = ([ln.lstrip() for ln in ex_1_lines]
                                                         == [ln.lstrip() for ln in ex_h_lines])
            touches = [h for h in hunks_cache[rel]
                       if _intersects(r1, h['v1'], h['v1_len']) or _intersects(rh, h['head'], h['head_len'])]
            rec['identical_at_both_heads'] = identical
            rec['range_touches_diff_hunk'] = touches
            rec['line_shift_head_minus_v1'] = rh[0] - r1[0]
            ok = ok and bool(ex_1_lines) and not miss_1
        rec['citation_holds'] = ok
        if not ok:
            failures.append({'claim': cid, 'file': rel, 'head_missing': miss_h,
                             'v1': {k: v for k, v in rec['v1'].items() if k != 'excerpt'},
                             'file_at_head': at_head[rel]})
        out.append(rec)
    files = {}
    for (rev, rel), d in texts.items():
        files.setdefault(rel, {})['v1' if rev == V1 else 'head'] = {'sha256': d['sha256'], 'n_lines': len(d['lines']),
                                                                    'exists': d['rc'] == 0}
    for rel in files:
        files[rel]['at_head'] = at_head[rel]
        files[rel]['diff_hunks_v1_to_head'] = hunks_cache[rel]
        files[rel]['identical_v1_head'] = (files[rel].get('v1', {}).get('sha256') == files[rel]['head']['sha256'])
    return out, files, failures


def head_comparison(claims):
    both = [c for c in claims if c['v1']['present']]
    differ = [c['id'] for c in both if not c['identical_at_both_heads']]
    touched = [c['id'] for c in both if c['range_touches_diff_hunk']]
    indent_only = [c['id'] for c in both if c['id'] in EXPECTED_INDENT_ONLY]
    indent_ok = all(c['identical_modulo_leading_whitespace'] for c in both if c['id'] in EXPECTED_INDENT_ONLY)
    unexpected_differ = [i for i in differ if i not in EXPECTED_INDENT_ONLY]
    unexpected_touch = [i for i in touched if i not in EXPECTED_INDENT_ONLY]
    return {'n_claims_at_both_heads': len(both), 'excerpts_differing': differ, 'ranges_touching_a_diff_hunk': touched,
            'expected_indentation_only': indent_only, 'expected_indentation_only_hold': indent_ok,
            'unexpected_differences': unexpected_differ, 'unexpected_hunk_contacts': unexpected_touch,
            'head_only_claims': [c['id'] for c in claims if not c['v1']['present']],
            'all_rows_identical': not unexpected_differ and not unexpected_touch and indent_ok}


def git_facts():
    _, head, _ = _git('rev-parse', 'HEAD')
    _, v1full, _ = _git('rev-parse', V1)
    rc_anc, _, _ = _git('merge-base', '--is-ancestor', V1, 'HEAD')
    rc_cc, _, _ = _git('merge-base', '--is-ancestor', CONVENTION_COMMIT, 'HEAD')
    _, names, _ = _git('diff', '--name-only', V1, 'HEAD', '--', '*.py')
    prod = sorted(f for f in names.splitlines() if '/' not in f and not re.match(r'p\d', f))
    _, cc_files, _ = _git('show', '--name-only', '--format=', CONVENTION_COMMIT)
    _, log_prod, _ = _git('log', '--format=%h %ad %s', '--date=short', f'{V1}..HEAD', '--', *prod) if prod else (0, '', '')
    inst = {}
    for rel in INSTANCE_FILES:
        b1, bh = _blob(V1, rel), _blob('HEAD', rel)
        inst[rel] = {'blob_v1': b1, 'blob_head': bh, 'identical': b1 is not None and b1 == bh,
                     'at_head': _file_at_head(rel)}
    essp = _read_bytes(ESSP).decode('utf-8')
    return {'HEAD': head, 'V1': v1full, 'v1_is_ancestor_of_head': rc_anc == 0,
            'convention_commit': CONVENTION_COMMIT, 'convention_commit_is_ancestor_of_head': rc_cc == 0,
            'production_modules_changed_v1_to_head': prod,
            'convention_commit_files': [f for f in cc_files.splitlines() if f.endswith('.py')],
            'commits_touching_those_modules_v1_to_head': log_prod.splitlines(),
            'srp1_instance_files': inst,
            'srp1_ess_params_declares_in_objective': '"in_objective"' in essp,
            'srp1_ess_params_declares_available_energy_soh_point': '"available_energy_soh_point"' in essp}


def manuscript_facts():
    raw = _read_bytes(MANUSCRIPT)
    md5 = hashlib.md5(raw).hexdigest()
    return {'path': MANUSCRIPT, 'md5': md5, 'sha256': _sha_bytes(raw),
            'md5_matches_addendum89_a87e195': md5.startswith(MANUSCRIPT_MD5_PREFIX), 'at_head': _file_at_head(MANUSCRIPT)}


def md_summary(o):
    hc = o['head_comparison']
    L = [f"# W188c — code reading for ruling 5 at V1 {o['git_facts']['V1'][:8]} and HEAD {o['git_facts']['HEAD'][:8]}",
         '', f"Script `{SCRIPT_REL}`; zero solves; guards {'0/0 verified' if o['guards_ok'] else 'FAILED'}; "
         f"citations {o['n_claims_holding']}/{o['n_claims']} hold; integrity "
         f"{'OK' if not o['integrity_failures'] else 'FAILED'}; rows identical at both heads: {hc['all_rows_identical']} "
         f"({hc['n_claims_at_both_heads']} citations at both heads; differing {hc['excerpts_differing']}; "
         f"touching a diff hunk {hc['ranges_touching_a_diff_hunk']}).", '', '## Verdicts', '']
    for v in o['verdicts']:
        L.append(f"- **{v['id']}** {v['question']}: {v['verdict']} (basis: {', '.join(v['basis'])})")
    L += ['', '## Citations', '', '| id | file | V1 lines | HEAD lines | identical | holds | statement |',
          '|---|---|---|---|---|---|---|']
    for c in o['claims']:
        st = c['statement'].replace('|', '\\|')
        v1 = f"{c['v1']['line_first']}-{c['v1']['line_last']}" if c['v1']['present'] else 'absent'
        L.append(f"| {c['id']} | `{c['file']}` | {v1} | {c['head']['line_first']}-{c['head']['line_last']} | "
                 f"{c['identical_at_both_heads']} | {c['citation_holds']} | {st} |")
    L += ['', '## Files (sha256 at V1 / HEAD)', '']
    for f, d in o['files'].items():
        L.append(f"- `{f}` V1 `{(d.get('v1') or {}).get('sha256')}` HEAD `{d['head']['sha256']}` "
                 f"identical {d['identical_v1_head']}; at HEAD {d['at_head']['ok']}")
    return '\n'.join(L) + '\n'


def post_run(out_dir):
    rels = [os.path.join(out_dir, OUT_NAMES[k]) for k in ('man', 'json', 'md')]
    rels += [os.path.join(out_dir, f) for f in sorted(os.listdir(os.path.join(REPO, out_dir))) if LAUNCH_LOG_RX.fullmatch(f)]
    for rel in rels:
        if not os.path.exists(os.path.join(REPO, rel)):
            _log(f'[W188c post-run PRECONDITION FAILED] {rel} missing')
            sys.exit(1)
    man = {rel: _sha(rel) for rel in rels}
    with open(os.path.join(REPO, out_dir, OUT_NAMES['post']), 'x', encoding='utf-8') as h:
        h.write(GRIO.dumps(man, indent=1, sort_keys=True) + '\n')
    _log(f"[W188c post-run] wrote {os.path.join(out_dir, OUT_NAMES['post'])}")
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
    rc, _, _ = _git('merge-base', '--is-ancestor', EXPECTED_HEAD_PREFIX, 'HEAD')
    if rc != 0:
        pre.append(f'HEAD {head} does not descend from {EXPECTED_HEAD_PREFIX}')
    else:
        cited = sorted({c[2] for c in CLAIMS} | set(INSTANCE_FILES))
        _, moved, _ = _git('diff', '--name-only', EXPECTED_HEAD_PREFIX, 'HEAD', '--', *cited)
        if moved:
            pre.append(f'cited files changed since {EXPECTED_HEAD_PREFIX}: {moved.splitlines()}')
    rc, _, _ = _git('cat-file', '-e', f'{V1}^{{commit}}')
    if rc != 0:
        pre.append(f'V1 {V1} is not a commit')
    if pre:
        for p in pre:
            _log(f'[W188c PRECONDITION FAILED] {p}')
        sys.exit(1)
    _log(f'[W188c] HEAD {head}; V1 {V1}; {len(CLAIMS)} citations')

    claims, files, failures = check_claims()
    for c in claims:
        v1 = f"{c['v1']['line_first']}-{c['v1']['line_last']}" if c['v1']['present'] else 'absent'
        _log(f"[W188c] {c['id']:6s} {c['file']} V1 {v1} HEAD {c['head']['line_first']}-{c['head']['line_last']} "
             f"identical={c['identical_at_both_heads']} holds={c['citation_holds']}"
             + (f" MISSING head {c['head']['missing_substrings']}" if c['head']['missing_substrings'] else '')
             + (f" MISSING v1 {c['v1'].get('missing_substrings')}" if c['v1'].get('missing_substrings') else '')
             + (f" V1-HAS {c['v1'].get('substrings_found_in_v1_file')}" if c['v1'].get('substrings_found_in_v1_file') else ''))
    hc = head_comparison(claims)
    facts = git_facts()
    msf = manuscript_facts()
    integrity = list(failures)
    if not hc['all_rows_identical']:
        integrity.append({'head_comparison': hc})
    if not facts['v1_is_ancestor_of_head']:
        integrity.append({'git': 'V1 is not an ancestor of HEAD'})
    for rel, d in facts['srp1_instance_files'].items():
        if not d['identical'] or not d['at_head']['ok']:
            integrity.append({'instance_file': rel, 'detail': d})
    if facts['srp1_ess_params_declares_in_objective'] or facts['srp1_ess_params_declares_available_energy_soh_point']:
        integrity.append({'srp1_ess_params_declares_a_p516_switch': True})
    if not msf['md5_matches_addendum89_a87e195'] or not msf['at_head']['ok']:
        integrity.append({'manuscript': msf})
    _log(f"[W188c] head comparison: {json.dumps(hc)}")
    _log(f"[W188c] production modules changed V1->HEAD: {facts['production_modules_changed_v1_to_head']}")
    g, pk, guards_ok = guards_state()
    out = {
        'stage': 'P5.15 Addendum 89 W188 (c) -- code reading for ruling 5 (zero-capacity conventions; zero solves)',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 89 ruling 5 and Planner order W188 (c)',
                      'manuscript_review/REFOCUS_ANALYSIS_2026-11-10.md G1',
                      'P5_15_W182E_CODE_PATHS.md Z1 (c)'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'script': SCRIPT_REL, 'script_sha256': _sha(SCRIPT_REL),
        'objective_convention': 'not applicable (no objective value is reported); the frozen convention of '
                                'Addendum 85 is the one the V1 rows were run under',
        'problem_instance': 'SRP1 (data/SRP1/SRP1.json); instance files hashed at both heads in git_facts',
        'git_facts': facts, 'manuscript': msf, 'files': files, 'claims': claims, 'n_claims': len(claims),
        'n_claims_holding': sum(1 for c in claims if c['citation_holds']),
        'head_comparison': hc, 'verdicts': VERDICTS, 'integrity_failures': integrity,
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
    _log(f"[W188c] guards: solve {g['counts']} verify0 failures {g['verify_0_failures']}; pickle {pk['counts']}")
    _log(f"[W188c] citations {out['n_claims_holding']}/{out['n_claims']} hold; integrity failures {len(integrity)}")
    if not guards_ok:
        _log('[W188c] GUARD FAULT')
        sys.exit(1)
    if integrity:
        for f in integrity:
            _log(f'[W188c INTEGRITY] {json.dumps(f, default=str)[:2000]}')
        sys.exit(3)
    sys.exit(0)


if __name__ == '__main__':
    main()
