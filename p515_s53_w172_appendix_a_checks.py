"""
P5.15 Addendum 69 order, Planner task W172 -- record and static checks behind the Appendix A audit
(P5_15_W172_APPENDIX_A_AUDIT.md). ZERO SOLVES, NO MODEL, NO PRODUCTION MODULE IMPORTED.

What it does (static reading only; no production module is imported, so no solve path exists to guard):
  A. Manuscript identity: the Overleaf clone's HEAD and the sha256 / line count of main.tex (declared 407f8df,
     9891057c..., 2,169 lines); the bibliography key walker_ni_2011 (present? how many entries?).
  B. Code identity: `git diff --stat <git_head_at_run> HEAD -- <production files + SRP1 case data>` for every
     `campaign_results.json` behind the tables (w142_resettle_v6, w142_resettle_ext_v6, w155_a64_cells, the 3 x 3
     pair) -- one subprocess per head, argument list (no shell, no globbing); control: the same pathspec against
     353e094b must report changes, so the pathspec is live.
  C. Interface ratings R^I_i (Task B). `network.Network.get_interface_branch_rating` (network.py) is a pure read:
     the sum of `branch.rate` over the in-service branches incident to the DN's reference node (type 3), with
     `branch.rate` = the case JSON field `rating` (MVA, `_read_network_from_json_file`; a parallel-branch merge sums the
     ratings, so the sum is unchanged). Replicated here on every DN year file (case33_{1,2,3}_<year>.json), with the
     git blob hash of each file at HEAD; the years each instance loads are flagged (SRP1.json, the 3 x 3 instance
     file). Cross-check: every `worst_pf_primal_rating` (MVA) recorded per cycle in the committed g_s39_D.json of the
     v6 and 3 x 3 campaigns, by node.
  D. Configuration readbacks from committed records (g_s39_D.json): sigma_fixed, sigma_computed, al_scale_esso
     (kappa_E), gamma after, S_ref, backstop, initial rho; recomputation of kappa_E = sigma / median block weight from
     SRP1.json and the 3 x 3 instance file (weights Y_y D_d (1 + r)^-(y - y0), the multiset over the TSO and the three
     DSO block sets, as `_compute_median_admm_block_weight`).
  E. Solver settings from the case files: options and recovery_options of case9 / case33_{1,2,3} / the ESS file.
  F. Code anchors: exact-text search of each code location the audit cites, line numbers at HEAD (every anchor must be
     found; the hit count is recorded).

Guards: `pickle` is blocked before any import; at exit the script asserts that no production module (pyomo,
shared_resources_planning, network*, shared_energy_storage*, model_construction_helpers, admm_*, helper_functions,
definitions) was imported. Refuses to overwrite an existing output.

Output: data/SRP1/Results/P515S53/w172_appendix_a_audit/w172_checks.json and manifest_sha256.json (sha256 of the
output, this script and every input read).

Command (repo root, canonical interpreter, attached, both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w172_appendix_a_checks.py \\
      > data/SRP1/Results/P515S53/w172_appendix_a_audit/launch.log 2>&1
"""
import sys

sys.modules['pickle'] = None  # block pickle before anything else is imported

import glob  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
os.chdir(REPO)

OUT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w172_appendix_a_audit')
OUT_JSON = os.path.join(OUT_DIR, 'w172_checks.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'manifest_sha256.json')
CLONE = os.path.join('manuscript', '6a67305f25e8348fb71380c3')
MAIN_TEX = os.path.join(CLONE, 'main.tex')
BIB = os.path.join(CLONE, 'bibliography.bib')
DECLARED_CLONE_HEAD = '407f8df'
DECLARED_MAIN_SHA256 = '9891057c03437cda16da749885001dbdedddde2d917d4a11deeeb2d68ade3062'
DECLARED_MAIN_LINES = 2169

P53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
CAMPAIGN_ROOTS = (os.path.join(P53, 'w142_resettle_v6'), os.path.join(P53, 'w142_resettle_ext_v6'),
                  os.path.join(P53, 'w155_a64_cells'), os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair'))
G_RECORD_ROOTS = (os.path.join(P53, 'w142_resettle_v6'), os.path.join(P53, 'w90_3x3', 'campaign_s53_w91_3x3_pair'))
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
INSTANCE_3X3 = os.path.join(P53, 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
SRP1_PARAMS = os.path.join('data', 'SRP1', 'SRP1_params.json')
NETWORK_PARAM_FILES = {
    'case9': os.path.join('data', 'SRP1', 'case9', 'case9_params.json'),
    'case33_1': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_params.json'),
    'case33_2': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_params.json'),
    'case33_3': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_params.json'),
}
ESS_PARAMS = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
FORBIDDEN_MODULE_PREFIXES = ('pyomo', 'shared_resources_planning', 'network', 'shared_energy_storage',
                             'model_construction_helpers', 'admm_', 'helper_functions', 'definitions')

# (id, file, exact text) -- every anchor must be found at least once; hit lines recorded.
ANCHORS = (
    ('block_weight_def', 'shared_resources_planning.py', 'def _get_admm_block_weight(network_data, year, day):'),
    ('block_weight_discount', 'shared_resources_planning.py',
     'annualization = 1.0 / ((1.0 + network_data.discount_factor) ** (int(year) - int(years[0])))'),
    ('sigma_resolve_def', 'shared_resources_planning.py', 'def _resolve_common_admm_objective_scale(objective_scale_computed, admm_parameters):'),
    ('sigma_assert', 'shared_resources_planning.py', 'if not (isfinite(ratio) and (1.0 / factor) <= ratio <= factor):'),
    ('median_weight', 'shared_resources_planning.py', 'median_weight = float(np.median(weights))'),
    ('kappa', 'shared_resources_planning.py', 'al_scale_esso = objective_scale_used / median_block_weight'),
    ('assert_factor_default', 'admm_parameters.py', 'self.objective_scale_assert_factor = 3.0'),
    ('R_int_tso_init', 'shared_resources_planning.py',
     'interface_transf_rating = distribution_network.network[year][day].get_interface_branch_rating() / s_base'),
    ('tso_obj_scaled', 'shared_resources_planning.py', 'obj = copy(model[year][day].objective.expr) / effective_scale'),
    ('dso_obj_scaled', 'shared_resources_planning.py', 'obj = copy(dso_model[year][day].objective.expr) / effective_scale'),
    ('tso_al_v', 'shared_resources_planning.py',
     'constraint_v_req = (model[year][day].expected_interface_vmag[dn, p] - model[year][day].vmag_req[dn, p])'),
    ('tso_al_p', 'shared_resources_planning.py',
     'constraint_p_req = (model[year][day].expected_interface_pf_p[dn, p] - model[year][day].p_pf_req[dn, p]) / interface_transf_rating'),
    ('dso_al_v', 'shared_resources_planning.py',
     'constraint_vmag_req = (dso_model[year][day].expected_interface_vmag[p] - dso_model[year][day].vmag_req[p])'),
    ('dso_al_p', 'shared_resources_planning.py',
     'constraint_p_req = (dso_model[year][day].expected_interface_pf_p[p] - dso_model[year][day].p_pf_req[p]) / interface_transf_rating'),
    ('tso_al_ess', 'shared_resources_planning.py',
     'constraint_ess_p_req = (model[year][day].expected_shared_ess_p[e, p] - model[year][day].p_ess_req[e, p]) / (2 * shared_ess_rating)'),
    ('dso_al_ess', 'shared_resources_planning.py',
     'constraint_ess_p_req = (dso_model[year][day].expected_shared_ess_p[p] - dso_model[year][day].p_ess_req[p]) / (2 * shared_ess_rating)'),
    ('esso_al', 'shared_resources_planning.py',
     'constraint_p_req = (models[node_id].es_pnet[y, d, p] - models[node_id].p_req[y, d, p]) / (2 * shared_ess_rating)'),
    ('esso_al_kappa', 'shared_resources_planning.py',
     'obj += models[node_id].admm_esso_al_scale * (models[node_id].dual_p_req[y, d, p] * constraint_p_req)'),
    ('tso_gamma_tied', 'shared_resources_planning.py',
     "model[year][day].prox_gamma_v = pe.Param(mutable=True, initialize=tso_gamma_tau * params.rho['v'][transmission_network.name])"),
    ('settlement_weight_tso', 'shared_resources_planning.py', 'model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('settlement_weight_dso', 'shared_resources_planning.py', 'dso_model[year][day].interface_settlement_weight.set_value(1.00)'),
    ('row18_inactive_def', 'shared_resources_planning.py', 'def _set_row18_inactive_for_initialisation(model):'),
    ('row18_activate_call', 'shared_resources_planning.py', '_activate_row18_with_settlement(dso_model[year][day])'),
    ('gross_q', 'shared_resources_planning.py', 'gross_operational_cost = (gross_operational_cost_including_settlement'),
    ('settlement_blocks', 'shared_resources_planning.py',
     'settlement_blocks = _get_operational_interface_settlement_blocks(planning_problem, models)'),
    ('loop', 'shared_resources_planning.py', 'for iter in range(1, admm_parameters.num_max_iters + 1):'),
    ('init_dso_flags', 'shared_resources_planning.py', 'update_flags={"update_tn": False, "update_dns": True, "update_sess": False},'),
    ('init_tso_flags', 'shared_resources_planning.py', 'update_flags={"update_tn": True, "update_dns": False, "update_sess": False},'),
    ('convert_dso', 'shared_resources_planning.py',
     'update_distribution_models_to_admm(planning_problem, dso_models, admm_parameters, objective_scale)'),
    ('z_init_call', 'shared_resources_planning.py', '_initialize_shared_ess_consensus(planning_problem, consensus_vars)'),
    ('z_init_mean', 'shared_resources_planning.py', 'z_value = (tso_value + dso_value + esso_value) / 3.0'),
    ('interface_dual_init', 'shared_resources_planning.py', 'planning_problem.update_interface_power_flow_variables('),
    ('esso_init_fix', 'shared_resources_planning.py', 'fix_or_set(esso_model[node_id].es_pnet[y, d, p], p_req)'),
    ('tso_init_pc_fix', 'shared_resources_planning.py',
     'fix_or_set(tso_model[year][day].pc[adn_load_idx, s_m, s_o, p], interface_pf_p)'),
    ('tso_delta_bound', 'shared_resources_planning.py',
     'tso_model[year][day].interface_delta_p[dn, s_m, s_o, p].setlb(-interface_transf_rating)'),
    ('plan_into_network_init', 'shared_resources_planning.py', 'update_data_with_candidate_solution(candidate_solution)'),
    ('capacities_publish', 'shared_resources_planning.py',
     'sess_available_capacities = shared_ess_data.get_updated_capacities(esso_model)'),
    ('capacities_set', 'shared_resources_planning.py',
     "model[year][day].shared_es_s_rated_fixed[shared_ess_idx].set_value(sess_estimated_capacity[year]['s_available'] / s_base)"),
    ('boyd_metrics_call', 'shared_resources_planning.py',
     'boyd_metrics = get_admm_boyd_residual_metrics(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters)'),
    ('aa_step_call', 'shared_resources_planning.py', 'aa_record = _anderson_acceleration_cycle_step('),
    ('aa_skip_failure', 'shared_resources_planning.py', 'aa_record = aa_state.skip_on_failure(iter)'),
    ('cycle_convergence', 'shared_resources_planning.py', 'cycle_convergence = boyd_all_pass and local_solves_ok'),
    ('production_exit', 'shared_resources_planning.py',
     'convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)'),
    ('tail_next_call', 'shared_resources_planning.py',
     'convergence_depth_tail_active_next = _convergence_depth_tail_next_state('),
    ('penalty_update_call', 'shared_resources_planning.py',
     'penalty_actions, penalties_before, penalties_after, gamma_before, gamma_after, rho_freeze_active, freeze_state = _update_admm_penalties('),
    ('aa_clear_rho', 'shared_resources_planning.py', 'aa_rho_change_record = aa_state.clear_for_rho_change(iter, aa_rho_changed)'),
    ('exit_break', 'shared_resources_planning.py', 'ADMM converged in {iter} iteration(s).'),
    ('boyd_def', 'shared_resources_planning.py',
     'def get_admm_boyd_residual_metrics(planning_problem, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_parameters):'),
    ('boyd_r_v', 'shared_resources_planning.py', 'r_v = (x_dso_v - z_tso_v) / v_base'),
    ('boyd_r_pf', 'shared_resources_planning.py', 'r_pf = (x_dso_pf - z_tso_pf) / interface_rating'),
    ('boyd_s_pf', 'shared_resources_planning.py', 's_pf = sqrt(rho_tso_pf ** 2 + gamma_pf ** 2) * dz_pf'),
    ('boyd_y_pf_dso_only', 'shared_resources_planning.py', 'y_pf = lambda_dso_pf / s_base_dso'),
    ('boyd_r_ess', 'shared_resources_planning.py', 'r_agent = a[agent] * (x_agent - z_new)'),
    ('boyd_s_ess_per_agent', 'shared_resources_planning.py', 's_agent_rho = rho_ess[agent] * a[agent] * dz_ess'),
    ('boyd_eps_pri', 'shared_resources_planning.py', 'eps_pri = sqrt(n) * eps_abs + eps_rel * max(norm_x, norm_z)'),
    ('boyd_eps_dual', 'shared_resources_planning.py', 'eps_dual = sqrt(n) * eps_abs + eps_rel * norm_y'),
    ('tail_apply_def', 'shared_resources_planning.py', 'def _apply_convergence_depth_tail(planning_problem, admm_parameters, active, baseline, cycle):'),
    ('tail_next_def', 'shared_resources_planning.py', 'def _convergence_depth_tail_next_state(cycle_convergence, aa_enabled, aa_record):'),
    ('tail_refuses_recovery_key', 'shared_resources_planning.py',
     'if CONVERGENCE_DEPTH_TAIL_OPTION in (solver_params.recovery_options or {}):'),
    ('local_solves_def', 'shared_resources_planning.py', 'def _admm_local_solves_succeeded(planning_problem, results):'),
    ('penalties_def', 'shared_resources_planning.py', 'def _update_admm_penalties('),
    ('rho_increase', 'shared_resources_planning.py', 'elif boyd_primal_ratio > increase_balance_ratio * boyd_dual_ratio_balance:'),
    ('rho_decrease', 'shared_resources_planning.py', 'elif boyd_dual_ratio_balance > decrease_balance_ratio * boyd_primal_ratio:'),
    ('rho_backstop', 'shared_resources_planning.py',
     'backstop_frozen_global = bool(freeze_backstop_cycle is not None and iter is not None and iter >= freeze_backstop_cycle)'),
    ('rho_failure_hold', 'shared_resources_planning.py', "action = 'held after solver failure'"),
    ('rho_exempt_until', 'shared_resources_planning.py', 'if full_dual_ratio < dual_ratio_below:'),
    ('rho_streak_freeze', 'shared_resources_planning.py',
     "group_state['unchanged_streak'] >= freeze_after_unchanged_cycles"),
    ('rho_clamp', 'shared_resources_planning.py', "penalty.set_value(min(max(value, params['min']), params['max']))"),
    ('interface_dual_tso_v', 'shared_resources_planning.py',
     "dual_vars['vmag']['tso']['current'][node_id][year][day][p] += rho_v_tso * error_v_req_tso"),
    ('interface_dual_tso_p', 'shared_resources_planning.py',
     "dual_vars['pf']['tso']['current'][node_id][year][day]['p'][p] += rho_pf_tso * error_p_pf_req_tso / interface_rating * tso_s_base"),
    ('interface_dual_gate', 'shared_resources_planning.py', 'if update_tn and tso_succeeded and dso_succeeded:'),
    ('ess_failure_gate', 'shared_resources_planning.py', '# Do not update the consensus or multipliers from a'),
    ('z_update', 'shared_resources_planning.py', 'z_new = numerator / denominator'),
    ('ess_dual_update', 'shared_resources_planning.py',
     "dual_vars[agent]['current'][node_id][year][day][power_type][p] += rhos[agent] * normalized_residual"),
    ('solver_result_succeeded', 'helper_functions.py', 'def solver_result_succeeded(result):'),
    ('aa_memory_deque', 'admm_anderson_acceleration.py', 'self._history = deque(maxlen=self.memory + 1)'),
    ('aa_tikhonov', 'admm_anderson_acceleration.py', 'gram = gram + self.regularization * np.eye(gram.shape[0])'),
    ('aa_safeguard', 'admm_anderson_acceleration.py', 'if combined_residual < self.last_accepted_residual:'),
    ('aa_keep_memory', 'admm_anderson_acceleration.py', "elif self.reject_policy == 'keep_memory':"),
    ('aa_off_branch', 'admm_anderson_acceleration.py', "action='off (all channels within Boyd tolerance)',"),
    ('aa_clear_rho_def', 'admm_anderson_acceleration.py', 'def clear_for_rho_change(self, cycle, channels_changed):'),
    ('aa_skip_failure_def', 'admm_anderson_acceleration.py', 'def skip_on_failure(self, cycle):'),
    ('aa_combined_residual', 'admm_anderson_acceleration.py',
     "total += boyd_metrics[group]['r'] ** 2 + boyd_metrics[group]['s'] ** 2"),
    ('aa_interface_z_is_tso_copy', 'admm_anderson_acceleration.py',
     "raw = consensus_vars['pf']['tso']['current'][node_id][year][day][pt][p]"),
    ('rating_def', 'network.py', 'def get_interface_branch_rating(self):'),
    ('rating_sum', 'network.py', 'interface_branch_rating += branch.rate'),
    ('rating_read_transformer', 'network.py', "branch.rate = float(transf_data['rating'])"),
    ('rating_parallel_merge', 'network.py', 'processed_branch.rate = sum([branch.rate for branch in connected_parallel_branches])'),
    ('network_max_iter', 'network.py', "options['max_iter'] = 500"),
    ('network_recoverable_maxiter', 'network.py', 'po.TerminationCondition.maxIterations,'),
    ('network_recovery_cold', 'network.py', "recovery_options['warm_start_init_point'] = 'no'"),
    ('network_tier2', 'network.py', "tier2_options['mu_strategy'] = 'adaptive'"),
    ('esso_tol_overrides', 'shared_energy_storage_data.py', "ESSO_TOL_OVERRIDES = {'tol': 1e-10, 'acceptable_tol': 1e-9}"),
    ('esso_recovery_override', 'shared_energy_storage_data.py', 'recovery_options.update(esso_option_overrides)'),
    ('esso_get_updated_capacities', 'shared_energy_storage_data.py', 'def get_updated_capacities(self, model):'),
    ('mch_settlement_weight', 'model_construction_helpers.py', 'model.interface_settlement_weight = pe.Param(initialize=0.00, mutable=True)'),
    ('mch_settlement_in_obj', 'model_construction_helpers.py', 'obj += model.interface_settlement_weight * model.interface_settlement'),
    ('mch_settlement_def', 'model_construction_helpers.py', 'def interface_energy_settlement(model, network):'),
    ('mch_single_scenario_guard', 'model_construction_helpers.py', 'if n_scenarios == 1:'),
    ('mch_voltage_pin', 'model_construction_helpers.py',
     'model.scenario_voltage_pin_weight = pe.Param(initialize=PENALTY_SCENARIO_DEVIATION, mutable=True)'),
    ('mch_ref_voltage_fixed', 'model_construction_helpers.py', 'return (vg - SMALL_TOLERANCE, vg + SMALL_TOLERANCE)'),
    ('def_voltage_pin_weight', 'definitions.py', 'PENALTY_SCENARIO_DEVIATION = 9e4'),
    ('hooks_held', 'p515_s53_w118_resettle_hooks.py', 'return self.first_pass is not None and c > self.first_pass'),
    ('hooks_k0_boyd', 'p515_s53_w118_resettle_hooks.py', "st.cur['boyd_k'] = bool(boyd_metrics['all_boyd_pass'])"),
    ('hooks_aa_forced_off', 'p515_s53_w118_resettle_hooks.py', "forced['all_boyd_pass'] = True"),
)

INPUTS = set()


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path):
    INPUTS.add(path)
    with open(path) as f:
        return json.load(f)


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


def check_manuscript():
    head = _git(['rev-parse', '--short=7', 'HEAD'], cwd=CLONE).strip()
    status = _git(['status', '--porcelain', '--', 'main.tex', 'bibliography.bib'], cwd=CLONE)
    with open(MAIN_TEX) as f:
        n_lines = sum(1 for _ in f)
    INPUTS.update({MAIN_TEX, BIB})
    with open(BIB) as f:
        bib = f.read()
    entry_start = bib.find('@article{walker_ni_2011,')
    entry = bib[entry_start: bib.find('}}', entry_start) + 2] if entry_start >= 0 else None
    out = {'clone_head': head, 'clone_head_matches_declared': head == DECLARED_CLONE_HEAD,
           'main_tex_sha256': _sha(MAIN_TEX), 'main_tex_sha256_matches_declared': _sha(MAIN_TEX) == DECLARED_MAIN_SHA256,
           'main_tex_lines': n_lines, 'main_tex_lines_match_declared': n_lines == DECLARED_MAIN_LINES,
           'main_or_bib_modified_in_clone': bool(status.strip()), 'bibliography_sha256': _sha(BIB),
           'walker_ni_2011_entries': bib.count('{walker_ni_2011,'), 'walker_ni_2011_entry': entry}
    if not (out['clone_head_matches_declared'] and out['main_tex_sha256_matches_declared']
            and out['main_tex_lines_match_declared'] and not out['main_or_bib_modified_in_clone']):
        raise SystemExit(f'manuscript identity check failed: {out}')
    return out


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
           'n_heads': len(heads), 'n_heads_with_change': sum(1 for v in per_head.values() if v['diff_stat'] != 'NO CHANGE'),
           'pathspec': pathspec, 'per_head': per_head,
           'control': {'head': CONTROL_HEAD, 'last_line': control[-1] if control else 'NO CHANGE'}}
    if out['n_heads_with_change'] or out['control']['last_line'] == 'NO CHANGE':
        raise SystemExit(f"code identity check failed: changed={out['n_heads_with_change']} control={out['control']}")
    return out


def _dn_rating(path):
    d = _read_json(path)
    nodes = {int(n['bus_i']): int(n['type']) for n in d['nodes']}
    refs = [b for b, t in nodes.items() if t == 3]
    if len(refs) != 1:
        raise SystemExit(f'{path}: expected one reference node, found {refs}')
    ref = refs[0]
    branches = []
    for kind in ('lines', 'transformers'):
        for b in d.get(kind, []):
            if not bool(b['status']):
                continue
            if nodes.get(int(b['fbus'])) == 4 or nodes.get(int(b['tbus'])) == 4:
                continue  # isolated end: removed by _pre_process_network
            if int(b['fbus']) == ref or int(b['tbus']) == ref:
                branches.append({'kind': kind, 'branch_id': b['branch_id'], 'fbus': b['fbus'], 'tbus': b['tbus'],
                                 'rating_mva': float(b['rating'])})
    rating = sum(b['rating_mva'] for b in branches)
    return {'baseMVA': float(d['baseMVA']), 'ref_node': ref, 'incident_branches': branches, 'rating_mva': rating,
            'rating_pu': rating / float(d['baseMVA']), 'git_blob_at_HEAD': _git(['rev-parse', f'HEAD:{path}']).strip()}


def check_interface_ratings():
    srp1 = _read_json(SRP1_CASE)
    inst = _read_json(INSTANCE_3X3)
    years_srp1 = sorted(srp1['Years'])
    years_3x3 = sorted(inst['Years'])
    node_of = {dn['name']: dn['connection_node_id'] for dn in srp1['DistributionNetworks']}
    node_of_3x3 = {dn['name']: dn['connection_node_id'] for dn in inst['DistributionNetworks']}
    if node_of != node_of_3x3:
        raise SystemExit(f'DN-to-node mapping differs between instances: {node_of} vs {node_of_3x3}')
    per_dn = {}
    for name, node in sorted(node_of.items()):
        files = sorted(glob.glob(os.path.join('data', 'SRP1', name, f'{name}_20*.json')))
        per_year = {os.path.basename(f)[len(name) + 1:-5]: _dn_rating(f) for f in files}
        values = sorted({(v['rating_mva'], v['baseMVA']) for v in per_year.values()})
        per_dn[name] = {'tn_node': node, 'per_year': per_year, 'distinct_(rating_mva, baseMVA)': values,
                        'year_dependent': len(values) > 1,
                        'years_loaded_srp1': years_srp1, 'years_loaded_3x3': years_3x3,
                        'all_loaded_years_present': all(y in per_year for y in years_srp1 + years_3x3)}
    # cross-check against committed per-cycle records
    recorded = {}
    for root in G_RECORD_ROOTS:
        for path in sorted(glob.glob(os.path.join(root, '**', 'g_s39_D.json'), recursive=True)):
            if _git(['ls-files', path]).strip() == '':
                continue  # uncommitted record: not used
            d = _read_json(path)
            stack = [d]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    if 'worst_pf_primal_rating' in o and o.get('worst_pf_primal_rating') is not None:
                        key = f"{o.get('worst_pf_primal_node')}|{o.get('worst_pf_primal_rating')}"
                        recorded[key] = recorded.get(key, 0) + 1
                    stack.extend(o.values())
                elif isinstance(o, list):
                    stack.extend(o)
    return {'per_dn': per_dn, 'recorded_worst_pf_primal_rating_node|mva_counts': recorded}


def _median_block_weight(case):
    years = list(case['Years'])
    r = float(case['DiscountFactor'])
    weights = []
    n_agents = 1 + len(case['DistributionNetworks'])
    for y in years:
        for d, days in case['Days'].items():
            w = float(case['Years'][y]) * float(days) / ((1.0 + r) ** (int(y) - int(years[0])))
            weights += [w] * n_agents
    return statistics.median(weights), r


def check_records_configuration():
    params = _read_json(SRP1_PARAMS)['admm']
    sigma = float(params['objective_scale'])
    out = {'case_file_admm': {k: params[k] for k in ('objective_scale', 'shared_ess_reference_rating_mva',
                                                     'esso_al_scale', 'penalty_update', 'rho', 'proximal_regularization',
                                                     'anderson_acceleration', 'tol', 'minimum_consecutive_converged_cycles',
                                                     'num_max_iters', 'adaptive_penalty')}}
    for label, path in (('SRP1', SRP1_CASE), ('3x3', INSTANCE_3X3)):
        med, r = _median_block_weight(_read_json(path))
        out[f'kappa_recomputed_{label}'] = {'discount_rate': r, 'median_block_weight': med, 'kappa': sigma / med}
    seen = {}
    for root in G_RECORD_ROOTS:
        for path in sorted(glob.glob(os.path.join(root, '**', 'g_s39_D.json'), recursive=True)):
            if _git(['ls-files', path]).strip() == '':
                continue
            d = _read_json(path)
            stack = [d]
            while stack:
                o = stack.pop()
                if isinstance(o, dict):
                    if 'sigma_fixed' in o and 'al_scale_esso' in o and o.get('cycle') == 1:
                        key = json.dumps({k: o.get(k) for k in (
                            'sigma_fixed', 'al_scale_esso', 'gamma_v_after', 'gamma_pf_after', 'gamma_ess_after',
                            'shared_ess_reference_rating_mva', 'freeze_backstop_cycle', 'freeze_after_unchanged_cycles',
                            'rho_v_before', 'rho_pf_before', 'rho_ess_before', 'boyd_eps_abs', 'boyd_eps_rel',
                            'required_consecutive_cycles')}, sort_keys=True)
                        seen.setdefault(key, {'n': 0, 'sigma_computed': []})
                        seen[key]['n'] += 1
                        seen[key]['sigma_computed'].append(o.get('sigma_computed'))
                    stack.extend(o.values())
                elif isinstance(o, list):
                    stack.extend(o)
    out['recorded_cycle1_configuration'] = [{'values': json.loads(k), 'n_records': v['n'],
                                             'sigma_computed_min': min(v['sigma_computed']),
                                             'sigma_computed_max': max(v['sigma_computed'])} for k, v in seen.items()]
    return out


def check_solver_settings():
    out = {}
    for name, path in NETWORK_PARAM_FILES.items():
        s = _read_json(path)['solver']
        out[name] = {'options': s.get('options'), 'recovery_options_key_present': 'recovery_options' in s,
                     'recovery_options': s.get('recovery_options'), 'recovery_block': s.get('recovery'),
                     'git_blob_at_HEAD': _git(['rev-parse', f'HEAD:{path}']).strip()}
    s = _read_json(ESS_PARAMS)['solver']
    out['esso'] = {'options': s.get('options'), 'recovery_options': s.get('recovery_options'),
                   'recovery_block': s.get('recovery'), 'git_blob_at_HEAD': _git(['rev-parse', f'HEAD:{ESS_PARAMS}']).strip()}
    return out


def check_anchors():
    out = {}
    for anchor_id, path, text in ANCHORS:
        INPUTS.add(path)
        with open(path) as f:
            hits = [i for i, line in enumerate(f, start=1) if text in line]
        if not hits:
            raise SystemExit(f'anchor {anchor_id} not found in {path}: {text!r}')
        out[anchor_id] = {'file': path, 'text': text, 'lines': hits}
    return out


def main():
    if os.path.exists(OUT_JSON) or os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'refusing to overwrite {OUT_JSON} / {OUT_MANIFEST}')
    os.makedirs(OUT_DIR, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    result = {
        'schema': 'p515_s53_w172_appendix_a_checks_v1',
        'started_utc': started,
        'A_manuscript': check_manuscript(),
        'B_code_identity': check_code_identity(),
        'C_interface_ratings': check_interface_ratings(),
        'D_records_configuration': check_records_configuration(),
        'E_solver_settings': check_solver_settings(),
        'F_code_anchors': check_anchors(),
    }
    bad = sorted(m for m in sys.modules if m.startswith(FORBIDDEN_MODULE_PREFIXES))
    if bad:
        raise SystemExit(f'production modules imported: {bad}')
    if sys.modules.get('pickle', 'absent') is not None:
        raise SystemExit('pickle block was lifted')
    result['guards'] = {'zero_solves': True, 'no_model_built': True, 'pickle_blocked': True,
                        'production_modules_imported': bad}
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=1, sort_keys=True)
    script = os.path.basename(__file__)
    manifest = {'files': {OUT_JSON: _sha(OUT_JSON), script: _sha(script)},
                'inputs': {p: _sha(p) for p in sorted(INPUTS)}}
    with open(OUT_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    rating = {dn: (v['tn_node'], v['distinct_(rating_mva, baseMVA)'], v['year_dependent'])
              for dn, v in result['C_interface_ratings']['per_dn'].items()}
    print(f"W172 checks: OK | heads {result['B_code_identity']['n_heads']} changed "
          f"{result['B_code_identity']['n_heads_with_change']} | R^I {rating} | anchors {len(ANCHORS)} found")
    print(f'output {OUT_JSON} sha256 {_sha(OUT_JSON)}')


if __name__ == '__main__':
    main()
