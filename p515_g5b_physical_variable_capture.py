"""
P5.15 Gate G5-B -- physical-variable discriminator for the G5 A1/A3/A4
disagreement (follow-up diagnostic, NOT a re-specification of G5).

Authority: Planner follow-up task, 2026-09-13, dispatched because G5 FAILED
(node 7: A1 vs A3 relative objective difference 2.86e-04, 286x the 1e-6 gate;
`data/SRP1/Results/P5151/g5_gate_summary.json`) and the failure admits two
competing interpretations that G5's own capture did not distinguish:

  I1 -- artifact, not nonconvexity. `es_pnet` is hard-fixed on this fixture, so
        `model.objective` (== `feasibility_penalty`) has NO physical content:
        it is entirely `PENALTY_ESSO_SLACK*(slack_up+slack_down) +
        EPS_ESSO_THROUGHPUT*sum(pch+pdch)`, both barrier-residual terms. A3
        differs from A1 only in `mu_strategy`, which changed `mu_final` by
        2.06x. Under I1 the arms reach the SAME solution and differ only in
        artifact terms.
  I2 -- genuine nonconvexity. The arms reach DIFFERENT physical solutions.

DISCRIMINATOR (predeclared by the Planner, not chosen by this harness):
physical solution variables -- `es_soh_per_unit_cumul`, `es_D_per_unit`,
`es_avg_ch_dch_per_unit` (per cohort-year), the per-cohort NET power
`es_pch_per_unit - es_pdch_per_unit` (per cohort-period), and `es_pnet`
(control, expected identical since it is fixed) -- plus the slack pair
`slack_es_pnet_{up,down}` and the barrier-identity arithmetic test on that
slack pair. G5 captured NONE of these; this harness captures them for the
SAME three arms on the SAME two nodes, reusing G5's own construction and arm
definitions BY IMPORT.

VERDICT RULE (declared in advance, stated in the dispatching task):
  - physical variables agree across A1/A3/A4 to 1e-6 relative on both nodes
    -> I1 supported, no nonconvexity demonstrated.
  - any physical variable disagrees by more than 1e-6 relative on either node
    -> I2 supported (genuine finding, regardless of what the slack arithmetic
    shows).
The physical-variable test decides; the slack-arithmetic test is reported
alongside but does not override it.

REUSE, NOT REIMPLEMENTATION. `_build_fresh_node_models`, `_run_arm`,
`ARM_SPECS`, `ARM_NAMES`, `GATE_NODES`, `PERMITTED`, `_pairwise_discriminator`
are imported from `p515_g5_reformulated_gate.py` and called UNMODIFIED --
`p515_g5_reformulated_gate.py` is not edited by this harness at all.

`_run_arm` (as written) returns diagnostics but not the solved model object
itself, and this harness needs the model object for physical-variable
extraction. Rather than reimplement `_run_arm`'s dispatch (deepcopy + option
overrides + call `SED._optimize`), this harness temporarily wraps
`SED._optimize` (a module-attribute lookup `_run_arm` already performs at call
time: `SED._optimize(...)`) with a thin recorder that calls through to the
REAL, unmodified `SED._optimize` and then stashes the exact `model` object
`_run_arm` is operating on. Since `SED._optimize` already loads the solved
values into that same `model` object on success (`_run_arm`'s own comment,
confirmed by reading `shared_energy_storage_data.py` `_optimize`/
`_run_solver_attempt`), the stashed reference is the fully-solved model after
`_run_arm` returns -- with zero changes to `_run_arm`'s or `_optimize`'s code,
and zero change to solve arguments, options, or arm definitions. The wrapper
is installed only for the duration of this run and restored in a `finally`.

GUARD. `SolveProfileGuard` armed for the whole run, same permitted call site
as G5 (`shared_energy_storage_data.py:_run_solver_attempt`). Declared count in
advance: 6 (2 nodes x 3 arms) -- IDENTICAL solve profile to G5, since this
harness re-solves the same instance under the same three arms; it adds no new
solves, only new post-solve extraction from the same solved models.

OUTPUT. Only NEW files under `data/SRP1/Results/P5151/` (asserted before
writing that none pre-exist, and that G5's own artifacts are untouched):
  - g5b_physical_variable_capture_summary.json
  - g5b_physical_variable_capture_full.json
  - g5b_gate_logs/node{7,9}_{A1,A3,A4}[_recovery].txt

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_g5b_physical_variable_capture.py
"""

import hashlib
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import shared_energy_storage_data as SED  # noqa: E402
import p515_g5_reformulated_gate as G5  # noqa: E402 -- reused, not reimplemented, not edited
from helper_functions import solver_result_succeeded  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_1b_eps_sensitivity_check import sha256_of  # noqa: E402

OUT_DIR = G5.OUT_DIR
NEW_LOG_ROOT = os.path.join(OUT_DIR, 'g5b_gate_logs')
SUMMARY_PATH = os.path.join(OUT_DIR, 'g5b_physical_variable_capture_summary.json')
FULL_PATH = os.path.join(OUT_DIR, 'g5b_physical_variable_capture_full.json')

GATE_NODES = G5.GATE_NODES
ARM_NAMES = G5.ARM_NAMES
PERMITTED = G5.PERMITTED
EXPECTED_SOLVES = len(GATE_NODES) * len(ARM_NAMES)  # 2 nodes x 3 arms = 6, declared in advance,
                                                      # IDENTICAL solve profile to G5 (same instance, same arms)

PHYSICAL_AGREEMENT_TOLERANCE = 1e-6  # the discriminator's own predeclared threshold (from the dispatching task)

REQUIRED_MODEL_VARS = (
    'es_soh_per_unit_cumul', 'es_D_per_unit', 'es_avg_ch_dch_per_unit',
    'es_pch_per_unit', 'es_pdch_per_unit', 'es_pnet',
    'slack_es_pnet_up', 'slack_es_pnet_down',
)

VARIABLE_FAMILIES = (
    'es_soh_per_unit_cumul', 'es_D_per_unit', 'es_avg_ch_dch_per_unit',
    'net_power_pch_minus_pdch_per_cohort_period', 'es_pnet',
)


def _assert_capture_paths_exist(fresh_models):
    """Preflight, fail fast BEFORE any solve."""
    problems = []
    if not hasattr(SED, 'ESSO_TOL_OVERRIDES'):
        problems.append('SED.ESSO_TOL_OVERRIDES missing (remedy (h) not in working tree)')
    elif SED.ESSO_TOL_OVERRIDES != {'tol': 1e-8, 'acceptable_tol': 1e-7}:
        problems.append(f'SED.ESSO_TOL_OVERRIDES unexpected: {SED.ESSO_TOL_OVERRIDES}')
    if not hasattr(SED, '_get_esso_complementarity_diagnostics'):
        problems.append('SED._get_esso_complementarity_diagnostics missing (mu_final/s_obj/bound capture path)')
    if not hasattr(SED, '_optimize'):
        problems.append('SED._optimize missing')
    if not hasattr(SED, '_parse_ipopt_barrier_terms'):
        problems.append('SED._parse_ipopt_barrier_terms missing (mu_final/s_obj log parser)')
    if SED.PENALTY_ESSO_SLACK != 1e3:
        problems.append(f'SED.PENALTY_ESSO_SLACK unexpected: {SED.PENALTY_ESSO_SLACK} (task assumes 1e3)')
    reference_log = os.path.join(OUT_DIR, 'tol_check_logs', 'remedy_h', 'optim_log_node_7.txt')
    if not os.path.exists(reference_log):
        problems.append(f'reference IPOPT log for capture-path check not found: {reference_log}')
    else:
        mu_final, s_obj, reason = SED._parse_ipopt_barrier_terms(reference_log)
        if mu_final is None or s_obj is None:
            problems.append(f'mu_final/s_obj capture path FAILED on reference log: {reason}')
    for node_id, model in fresh_models.items():
        for var_name in REQUIRED_MODEL_VARS:
            if not hasattr(model, var_name):
                problems.append(f'node {node_id}: fresh model missing required variable {var_name}')
    for path in (SUMMARY_PATH, FULL_PATH):
        if os.path.exists(path):
            problems.append(f'output path already exists, refusing to overwrite: {path}')
    if os.path.isdir(NEW_LOG_ROOT) and os.listdir(NEW_LOG_ROOT):
        problems.append(f'log directory already exists and is non-empty: {NEW_LOG_ROOT}')
    for path in (G5.SUMMARY_PATH, G5.FULL_PATH):
        if not os.path.exists(path):
            problems.append(f'expected pre-existing G5 artifact not found (unexpected environment): {path}')
    if problems:
        raise RuntimeError('G5-B capture-path preflight FAILED:\n' + '\n'.join(f'  - {p}' for p in problems))


def _extract_physical_variables(model):
    years = list(model.years)
    days = list(model.days)
    periods = list(model.periods)

    soh, d_deg, avg_ch_dch = {}, {}, {}
    for y_inv in years:
        for y in years:
            soh[f'{y_inv}_{y}'] = pe.value(model.es_soh_per_unit_cumul[y_inv, y])
            d_deg[f'{y_inv}_{y}'] = pe.value(model.es_D_per_unit[y_inv, y])
            avg_ch_dch[f'{y_inv}_{y}'] = pe.value(model.es_avg_ch_dch_per_unit[y_inv, y])

    net_power = {}
    for y_inv in years:
        for y in years:
            for d in days:
                for p in periods:
                    pch = pe.value(model.es_pch_per_unit[y_inv, y, d, p])
                    pdch = pe.value(model.es_pdch_per_unit[y_inv, y, d, p])
                    net_power[f'{y_inv}_{y}_{d}_{p}'] = pch - pdch

    es_pnet, slack_up, slack_down, slack_min = {}, {}, {}, {}
    for y in years:
        for d in days:
            for p in periods:
                key = f'{y}_{d}_{p}'
                es_pnet[key] = pe.value(model.es_pnet[y, d, p])
                su = pe.value(model.slack_es_pnet_up[y, d, p])
                sd = pe.value(model.slack_es_pnet_down[y, d, p])
                slack_up[key] = su
                slack_down[key] = sd
                slack_min[key] = min(su, sd)

    return {
        'n_years': len(years), 'n_days': len(days), 'n_periods_per_day': len(periods),
        'n_y_d_p_total': len(years) * len(days) * len(periods),
        'es_soh_per_unit_cumul': soh,
        'es_D_per_unit': d_deg,
        'es_avg_ch_dch_per_unit': avg_ch_dch,
        'net_power_pch_minus_pdch_per_cohort_period': net_power,
        'es_pnet': es_pnet,
        'slack_es_pnet_up': slack_up,
        'slack_es_pnet_down': slack_down,
        'slack_min_up_down': slack_min,
    }


def _slack_summary(phys):
    up = list(phys['slack_es_pnet_up'].values())
    down = list(phys['slack_es_pnet_down'].values())
    mins = list(phys['slack_min_up_down'].values())
    return {
        'n_periods': len(up),
        'slack_up_max': max(up), 'slack_up_min': min(up),
        'slack_down_max': max(down), 'slack_down_min': min(down),
        'min_up_down_max': max(mins), 'min_up_down_min': min(mins),
        'min_up_down_mean': statistics.mean(mins),
        'min_up_down_stdev': statistics.pstdev(mins) if len(mins) > 1 else 0.0,
    }


def _compare_family(vals_a, vals_b):
    """Same relative-difference convention G5's own `_pairwise_discriminator`
    uses for the objective (denom = max(abs(a), abs(b), 1e-300)), applied
    per-key across a whole variable family. Both the raw (1e-300 floor) and a
    floored (1e-9 floor) max-relative-diff are reported: the raw form is the
    literal gate convention; the floored form is reported alongside so a
    roundoff-level difference between two entries that are both ~0 (e.g. an
    inactive cohort fixed at 0.0) is not allowed to silently dominate the
    reported number without being visible as such. Neither replaces the
    other; both are shown."""
    assert set(vals_a.keys()) == set(vals_b.keys())
    max_abs = 0.0
    argmax_abs = None
    max_rel_raw = 0.0
    argmax_rel_raw = None
    max_rel_floored = 0.0
    argmax_rel_floored = None
    for k in vals_a:
        a = vals_a[k]
        b = vals_b[k]
        abs_diff = abs(a - b)
        if abs_diff > max_abs:
            max_abs = abs_diff
            argmax_abs = k
        denom_raw = max(abs(a), abs(b), 1e-300)
        rel_raw = abs_diff / denom_raw
        if rel_raw > max_rel_raw:
            max_rel_raw = rel_raw
            argmax_rel_raw = k
        denom_floored = max(abs(a), abs(b), 1e-9)
        rel_floored = abs_diff / denom_floored
        if rel_floored > max_rel_floored:
            max_rel_floored = rel_floored
            argmax_rel_floored = k
    return {
        'n_keys_compared': len(vals_a),
        'max_abs_diff': max_abs, 'argmax_abs_diff_key': argmax_abs,
        'value_a_at_argmax_abs': vals_a.get(argmax_abs), 'value_b_at_argmax_abs': vals_b.get(argmax_abs),
        'max_rel_diff_raw_1e-300_floor': max_rel_raw, 'argmax_rel_diff_raw_key': argmax_rel_raw,
        'value_a_at_argmax_rel_raw': vals_a.get(argmax_rel_raw), 'value_b_at_argmax_rel_raw': vals_b.get(argmax_rel_raw),
        'max_rel_diff_1e-9_floor': max_rel_floored, 'argmax_rel_diff_floored_key': argmax_rel_floored,
        'value_a_at_argmax_rel_floored': vals_a.get(argmax_rel_floored), 'value_b_at_argmax_rel_floored': vals_b.get(argmax_rel_floored),
        'agrees_to_1e-6_relative_raw': max_rel_raw <= PHYSICAL_AGREEMENT_TOLERANCE,
        'agrees_to_1e-6_relative_floored': max_rel_floored <= PHYSICAL_AGREEMENT_TOLERANCE,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shared_ess_data, candidate_solution, fresh_models = G5._build_fresh_node_models()
    _assert_capture_paths_exist(fresh_models)
    os.makedirs(NEW_LOG_ROOT, exist_ok=True)

    instance_descriptor = {
        'construction_source': 'p515_g5_reformulated_gate.py:_build_fresh_node_models (imported, not reimplemented)',
        'shared_energy_storage_data_py_sha256_working_tree': hashlib.sha256(
            open(os.path.join(REPO, 'shared_energy_storage_data.py'), 'rb').read()
        ).hexdigest(),
        'p515_g5_reformulated_gate_py_sha256_working_tree': hashlib.sha256(
            open(os.path.join(REPO, 'p515_g5_reformulated_gate.py'), 'rb').read()
        ).hexdigest(),
        'g5_gate_summary_reference': G5.SUMMARY_PATH,
        'gate_nodes': list(GATE_NODES),
        'candidate_solution': {str(k): v for k, v in candidate_solution.items()},
        'eps_esso_throughput_in_force': SED.EPS_ESSO_THROUGHPUT,
        'penalty_esso_slack_in_force': SED.PENALTY_ESSO_SLACK,
        'esso_tol_overrides_in_force': SED.ESSO_TOL_OVERRIDES,
    }
    instance_hash = sha256_of(instance_descriptor)

    report = {
        'stage': 'P5.15 Gate G5-B (physical-variable discriminator, follow-up to failed G5)',
        'authority': 'Planner follow-up task, 2026-09-13 (gate G5 FAILED; this captures what separates I1/I2)',
        'instance': instance_descriptor,
        'instance_hash_sha256': instance_hash,
        'arm_specs_reused_from_g5': {name: dict(spec) for name, spec in G5.ARM_SPECS.items()},
        'physical_agreement_tolerance': PHYSICAL_AGREEMENT_TOLERANCE,
    }

    # Redirect G5's own `_run_arm` IPOPT log output to a NEW log directory.
    # `_run_arm` reads `LOG_ROOT` from its own module's globals at call time
    # (this is how `p515_g5_reformulated_gate.py` directs its own output when
    # run standalone) -- overriding it here redirects OUTPUT LOCATION only.
    G5.LOG_ROOT = NEW_LOG_ROOT

    # Capture the exact model object `_run_arm` solves, WITHOUT altering
    # `_run_arm`'s code: wrap `SED._optimize` (the module attribute `_run_arm`
    # looks up at call time) with a thin recorder that calls straight through
    # to the real, unmodified function.
    captured = {}
    original_optimize = SED._optimize

    def _optimize_and_capture(model, *args, **kwargs):
        result = original_optimize(model, *args, **kwargs)
        captured['model'] = model
        captured['result'] = result
        return result

    guard = SolveProfileGuard(PERMITTED, label='P5.15 Gate G5-B').install()
    all_arms = {}
    physical = {}
    try:
        SED._optimize = _optimize_and_capture
        for node_id in GATE_NODES:
            all_arms[node_id] = {}
            physical[node_id] = {}
            for arm_name in ARM_NAMES:
                print(f'[G5-B] node={node_id} arm={arm_name} solving...', flush=True)
                captured.clear()
                arm_result = G5._run_arm(shared_ess_data, node_id, arm_name, fresh_models[node_id])
                all_arms[node_id][arm_name] = arm_result
                if 'model' not in captured:
                    raise RuntimeError(f'node={node_id} arm={arm_name}: SED._optimize wrapper never fired -- capture path broken')
                solved_model = captured['model']
                succeeded = solver_result_succeeded(captured['result'])
                if not succeeded:
                    raise RuntimeError(
                        f'node={node_id} arm={arm_name}: solve did not succeed '
                        f'(termination_condition={arm_result["termination_condition"]}); '
                        'physical-variable extraction requires a loaded solution, refusing to extract from an unsolved model.'
                    )
                phys = _extract_physical_variables(solved_model)
                physical[node_id][arm_name] = phys
                diag = arm_result['diagnostics']
                print(
                    f'  -> objective={arm_result["objective"]} mu_final={diag["mu_final"]} s_obj={diag["s_obj"]} '
                    f'slack_summary={_slack_summary(phys)}',
                    flush=True,
                )
    finally:
        SED._optimize = original_optimize
        G5.LOG_ROOT = os.path.join(OUT_DIR, 'g5_gate_logs')  # restore G5's own default, defence in depth
        guard.uninstall()

    guard_failures = guard.verify(EXPECTED_SOLVES)
    report['guard'] = {
        'permitted_call_sites': PERMITTED,
        'expected_solves_declared_in_advance': EXPECTED_SOLVES,
        'observed_counts': guard.counts,
        'permitted_sites_hit': guard.permitted_sites,
        'verify_failures': guard_failures,
    }

    # --- Physical-variable cross-arm comparison table -----------------------
    physical_comparison = {}
    node_agrees_raw = {}
    node_agrees_floored = {}
    for node_id in GATE_NODES:
        physical_comparison[node_id] = {}
        node_agrees_raw[node_id] = True
        node_agrees_floored[node_id] = True
        pairs = [('A1', 'A3'), ('A1', 'A4'), ('A3', 'A4')]
        for family in VARIABLE_FAMILIES:
            physical_comparison[node_id][family] = {}
            for a, b in pairs:
                cmp = _compare_family(physical[node_id][a][family], physical[node_id][b][family])
                physical_comparison[node_id][family][f'{a}_vs_{b}'] = cmp
                if not cmp['agrees_to_1e-6_relative_raw']:
                    node_agrees_raw[node_id] = False
                if not cmp['agrees_to_1e-6_relative_floored']:
                    node_agrees_floored[node_id] = False

    verdict_raw = 'I1' if all(node_agrees_raw.values()) else 'I2'
    verdict_floored = 'I1' if all(node_agrees_floored.values()) else 'I2'

    report['physical_variable_comparison_by_node'] = {str(k): v for k, v in physical_comparison.items()}
    report['node_agrees_to_1e-6_relative_raw'] = {str(k): v for k, v in node_agrees_raw.items()}
    report['node_agrees_to_1e-6_relative_floored'] = {str(k): v for k, v in node_agrees_floored.items()}
    report['verdict_raw_1e-300_floor_convention'] = verdict_raw
    report['verdict_floored_1e-9_floor_convention'] = verdict_floored

    # --- Slack barrier-identity arithmetic test ------------------------------
    slack_summaries = {}
    slack_predicted_vs_measured = {}
    penalty = SED.PENALTY_ESSO_SLACK
    for node_id in GATE_NODES:
        slack_summaries[node_id] = {}
        slack_predicted_vs_measured[node_id] = {}
        for arm_name in ARM_NAMES:
            phys = physical[node_id][arm_name]
            summary = _slack_summary(phys)
            slack_summaries[node_id][arm_name] = summary
            diag = all_arms[node_id][arm_name]['diagnostics']
            mu_final = diag['mu_final']
            s_obj = diag['s_obj']
            predicted_min_up_down = None
            if mu_final is not None and s_obj not in (None, 0.0):
                predicted_min_up_down = mu_final / (2.0 * s_obj * penalty)
            entry = {
                'mu_final': mu_final, 's_obj': s_obj,
                'predicted_min_up_down': predicted_min_up_down,
                'measured_min_up_down_max': summary['min_up_down_max'],
                'measured_min_up_down_mean': summary['min_up_down_mean'],
                'measured_min_up_down_min': summary['min_up_down_min'],
                'measured_over_predicted_at_max': (
                    summary['min_up_down_max'] / predicted_min_up_down if predicted_min_up_down else None
                ),
                'measured_over_predicted_at_mean': (
                    summary['min_up_down_mean'] / predicted_min_up_down if predicted_min_up_down else None
                ),
            }
            slack_predicted_vs_measured[node_id][arm_name] = entry

    report['slack_summary_by_node_arm'] = {str(k): v for k, v in slack_summaries.items()}
    report['slack_barrier_identity_test_by_node_arm'] = {str(k): v for k, v in slack_predicted_vs_measured.items()}

    # --- Residual-closure test: does the slack term close the A1-vs-A3 gap
    # left unexplained by the pch/pdch leak (G5's own discriminator)? --------
    residual_closure = {}
    for node_id in GATE_NODES:
        pairs = G5._pairwise_discriminator(node_id, all_arms[node_id])
        pair_by_name = {p['pair']: p for p in pairs}
        p13 = pair_by_name.get('A1_vs_A3')
        n_periods_diag = all_arms[node_id]['A1']['diagnostics']['n_active_cohort_periods']
        pred_min_a1 = slack_predicted_vs_measured[node_id]['A1']['predicted_min_up_down']
        pred_min_a3 = slack_predicted_vs_measured[node_id]['A3']['predicted_min_up_down']
        meas_min_a1 = slack_predicted_vs_measured[node_id]['A1']['measured_min_up_down_mean']
        meas_min_a3 = slack_predicted_vs_measured[node_id]['A3']['measured_min_up_down_mean']
        predicted_slack_term_diff_from_predicted_mins = (
            penalty * 2.0 * (pred_min_a3 - pred_min_a1) * n_periods_diag
            if pred_min_a1 is not None and pred_min_a3 is not None else None
        )
        predicted_slack_term_diff_from_measured_means = (
            penalty * 2.0 * (meas_min_a3 - meas_min_a1) * n_periods_diag
            if meas_min_a1 is not None and meas_min_a3 is not None else None
        )
        pch_pdch_residual = p13['residual_observed_minus_predicted'] if p13 else None
        residual_after_slack_term_pred = (
            pch_pdch_residual - predicted_slack_term_diff_from_predicted_mins
            if pch_pdch_residual is not None and predicted_slack_term_diff_from_predicted_mins is not None else None
        )
        residual_closure[node_id] = {
            'n_periods_used': n_periods_diag,
            'observed_diff_A1_vs_A3': p13['observed_diff'] if p13 else None,
            'predicted_diff_from_pch_pdch_leak': p13['predicted_diff_from_leak'] if p13 else None,
            'residual_after_pch_pdch_leak_only': pch_pdch_residual,
            'predicted_min_up_down_A1': pred_min_a1,
            'predicted_min_up_down_A3': pred_min_a3,
            'measured_mean_min_up_down_A1': meas_min_a1,
            'measured_mean_min_up_down_A3': meas_min_a3,
            'predicted_slack_term_objective_diff_using_predicted_mins': predicted_slack_term_diff_from_predicted_mins,
            'predicted_slack_term_objective_diff_using_measured_means': predicted_slack_term_diff_from_measured_means,
            'residual_after_pch_pdch_leak_and_slack_term_using_predicted_mins': residual_after_slack_term_pred,
            'ratio_residual_over_predicted_slack_term': (
                abs(pch_pdch_residual) / abs(predicted_slack_term_diff_from_predicted_mins)
                if pch_pdch_residual and predicted_slack_term_diff_from_predicted_mins else None
            ),
        }

    report['residual_closure_test_A1_vs_A3_by_node'] = {str(k): v for k, v in residual_closure.items()}

    with open(FULL_PATH, 'w') as fh:
        json.dump({
            'meta': {k: v for k, v in report.items() if k not in ('physical_variable_comparison_by_node',)},
            'physical_raw_by_node_arm': {
                str(node_id): {arm_name: physical[node_id][arm_name] for arm_name in ARM_NAMES}
                for node_id in GATE_NODES
            },
            'arms_full': {
                str(node_id): {arm_name: all_arms[node_id][arm_name] for arm_name in ARM_NAMES}
                for node_id in GATE_NODES
            },
        }, fh, indent=2, default=str)

    with open(SUMMARY_PATH, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps({k: v for k, v in report.items() if k != 'physical_variable_comparison_by_node'}, indent=2, default=str))
    print(f'\n[G5-B] full data (incl. raw physical variables) written to {FULL_PATH}')
    print(f'[G5-B] summary written to {SUMMARY_PATH}')
    print(f'[G5-B] guard verify failures: {guard_failures}')
    print(f'[G5-B] verdict (raw 1e-300 floor convention): {verdict_raw}')
    print(f'[G5-B] verdict (floored 1e-9 convention): {verdict_floored}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
