"""
P5.15 s32 mapping diagnostic -- bounded, NO production-code changes.

Explains the inconsistency the Planner found in the committed cycle-1 s32
preflight (`data/SRP1/Results/P515S32/preflight/g_preflight.json`): under
rho = gamma = 1 with lambda^0 assumed zero, ||y_v|| should equal
rho*||r_v|| = 0.101 but is observed as 0.0375 (matching rho*||Delta z_v||
instead).

Runs ONE ADMM cycle through the SAME s32 arm machinery as
`p515_s32_preflight.py` (`p515_g_g1_g4_admm_gates.run_admm_arm`, imported,
never through that module's own CLI), candidate C*, case-file rho in force
(`apply_rho=False`), smoke cap override 1. Output goes to a NEW directory,
`data/SRP1/Results/P515S32/diag_mapping/`; refuses if it exists; never
writes into `preflight/`.

Read-only instrumentation only: `shared_resources_planning._update_interface_
power_flow_variables` and `shared_resources_planning.get_admm_boyd_residual_
metrics` are monkeypatched, in THIS script only, with wrappers that deep-copy
snapshot the consensus/dual dictionaries before and after calling the ORIGINAL
function UNCHANGED, then delegate to it. No production file is edited.

    python p515_s32_mapping_diag.py
"""

import json
import math
import os
import pickle
import sys
from copy import deepcopy

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'diag_mapping')
PREFLIGHT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'preflight')


# ======================================================================================
# Instrumentation (read-only wrappers around the two production functions in question).
# Wrappers call the saved originals UNCHANGED; nothing here mutates production state
# beyond what the originals themselves already do.
# ======================================================================================

_ORIG_UPDATE_IFPFV = srp._update_interface_power_flow_variables
_ORIG_BOYD = srp.get_admm_boyd_residual_metrics
_ORIG_UACC = srp.update_and_check_convergence

CALL_LOG = []      # every call to _update_interface_power_flow_variables, in order
BOYD_LOG = []      # every call to get_admm_boyd_residual_metrics
UACC_LOG = []      # every call to update_and_check_convergence (order/flags marker only)


def _snapshot(consensus_vars, dual_vars):
    """Deep-copy the vmag/pf/ess consensus and dual sub-dictionaries."""
    return {
        'consensus_vmag': deepcopy(consensus_vars['vmag']),
        'consensus_pf': deepcopy(consensus_vars['pf']),
        'consensus_ess': deepcopy(consensus_vars['ess']),
        'dual_vmag': deepcopy(dual_vars['vmag']),
        'dual_pf': deepcopy(dual_vars['pf']),
        'dual_ess': deepcopy(dual_vars['ess']),
    }


def _call_tag(update_tn, update_dns):
    if update_tn and update_dns:
        return 'pre_admm_init'
    if update_dns and not update_tn:
        return 'cycle1_dso'
    if update_tn and not update_dns:
        return 'cycle1_tso'
    return 'cycle1_esso'


def _wrapped_update_ifpfv(planning_problem, tso_model, dso_models, interface_vars, dual_vars,
                           results, params, update_tn=True, update_dns=True):
    idx = len(CALL_LOG)
    pre = _snapshot(interface_vars, dual_vars)

    # rho values in force AT THIS CALL (captured before the original runs; the
    # original does not itself mutate rho, but capture pre-call to remove any
    # ambiguity about a LATER, unrelated penalty update changing the same
    # mutable Pyomo Param in place).
    rho_snapshot = {}
    try:
        for node_id in planning_problem.active_distribution_network_nodes:
            for year in planning_problem.years:
                for day in planning_problem.days:
                    rho_snapshot[f'{node_id}|{year}|{day}'] = {
                        'rho_v_tso': pe.value(tso_model[year][day].rho_v),
                        'rho_v_dso': pe.value(dso_models[node_id][year][day].rho_v),
                        'rho_pf_tso': pe.value(tso_model[year][day].rho_pf),
                        'rho_pf_dso': pe.value(dso_models[node_id][year][day].rho_pf),
                    }
    except Exception as error:  # pragma: no cover -- diagnostic only, never swallow silently
        rho_snapshot = {'error': f'{type(error).__name__}: {error}'}

    result = _ORIG_UPDATE_IFPFV(planning_problem, tso_model, dso_models, interface_vars,
                                 dual_vars, results, params, update_tn=update_tn,
                                 update_dns=update_dns)
    post = _snapshot(interface_vars, dual_vars)
    CALL_LOG.append({
        'index': idx,
        'tag': _call_tag(update_tn, update_dns),
        'update_tn': update_tn,
        'update_dns': update_dns,
        'rho_snapshot': rho_snapshot,
        'pre': pre,
        'post': post,
    })
    return result


def _wrapped_boyd(planning_problem, tso_model, dso_models, esso_model, consensus_vars,
                   dual_vars, admm_parameters):
    args_snapshot = _snapshot(consensus_vars, dual_vars)
    result = _ORIG_BOYD(planning_problem, tso_model, dso_models, esso_model, consensus_vars,
                         dual_vars, admm_parameters)
    BOYD_LOG.append({'index': len(BOYD_LOG), 'args_snapshot': args_snapshot,
                      'result': deepcopy(result)})
    return result


def _wrapped_uacc(planning_problem, tso_model, dso_models, esso_model, consensus_vars,
                   dual_vars, results, admm_parameters, primal_evolution, update_flags,
                   debug_flag=False, check_convergence=True):
    idx = len(UACC_LOG)
    result = _ORIG_UACC(planning_problem, tso_model, dso_models, esso_model, consensus_vars,
                         dual_vars, results, admm_parameters, primal_evolution, update_flags,
                         debug_flag=debug_flag, check_convergence=check_convergence)
    UACC_LOG.append({'index': idx, 'update_flags': dict(update_flags)})
    return result


def _install_patches():
    """Installed AFTER the capture-path precheck runs (that precheck reads
    `inspect.getsource(srp.get_admm_boyd_residual_metrics)` and asserts the
    literal field names are present in the PRODUCTION source -- it must see
    the original function, not this diagnostic's wrapper)."""
    srp._update_interface_power_flow_variables = _wrapped_update_ifpfv
    srp.get_admm_boyd_residual_metrics = _wrapped_boyd
    srp.update_and_check_convergence = _wrapped_uacc


# ======================================================================================
# Analysis helpers -- flatten the captured snapshots into flat lists, per channel,
# in the SAME node/year/day/period order the production Boyd function iterates in.
# ======================================================================================

def _iter_nodes_years_days(planning):
    for node_id in planning.active_distribution_network_nodes:
        for year in planning.years:
            for day in planning.days:
                yield node_id, year, day


def _flatten_vmag(d, planning):
    out = []
    for node_id, year, day in _iter_nodes_years_days(planning):
        out.extend(d[node_id][year][day])
    return out


def _flatten_pf(d, planning, power_type):
    out = []
    for node_id, year, day in _iter_nodes_years_days(planning):
        out.extend(d[node_id][year][day][power_type])
    return out


def _l2(values):
    return math.sqrt(sum(v * v for v in values))


def _max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def _elementwise(a, b, op):
    return [op(x, y) for x, y in zip(a, b)]


def _rho_series_for_channel(call_entry, planning, channel, which):
    """which in {'rho_v_dso','rho_v_tso','rho_pf_dso','rho_pf_tso'}; returns one
    scalar per (node,year,day) triple, in the same order _flatten_vmag/_flatten_pf use,
    broadcast over periods (rho is constant across periods within a (node,year,day))."""
    out = []
    for node_id, year, day in _iter_nodes_years_days(planning):
        key = f'{node_id}|{year}|{day}'
        rho = call_entry['rho_snapshot'][key][which]
        n_periods = len(call_entry['pre']['consensus_vmag']['dso']['current'][node_id][year][day]) \
            if channel == 'v' else len(call_entry['pre']['consensus_pf']['dso']['current'][node_id][year][day]['p'])
        out.extend([rho] * n_periods)
    return out


def _dso_pf_lambda_scale_series(planning):
    """The DSO-side PF lambda update (`_update_interface_power_flow_variables`,
    'Update Lambdas' block) is `dual += rho_pf_dso * error / interface_rating *
    dso_s_base` -- i.e. an EXTRA (dso_s_base / interface_rating) factor the V
    channel does not have. Returns that factor, one scalar per (node,year,day),
    broadcast over periods, static network data (no per-call capture needed)."""
    out = []
    for node_id, year, day in _iter_nodes_years_days(planning):
        dso_network = planning.distribution_networks[node_id].network[year][day]
        interface_rating = dso_network.get_interface_branch_rating()
        dso_s_base = dso_network.baseMVA
        n_periods = planning.num_instants
        out.extend([dso_s_base / interface_rating] * n_periods)
    return out


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to start: output directory already exists: {OUT_DIR}')
    os.makedirs(OUT_DIR)

    # ---- same precheck p515_s32_preflight.py runs (zero-solve capture-path check) ----
    precheck_eval_id = 'p515s32_diag_mapping_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s32_capture_paths(precheck_planning)
    del precheck_planning
    print(f'[S32 mapping-diag] capture-path pre-flight passed: {checklist}')

    _install_patches()

    analysis = {}

    def hook(planning, sed, models, rows, report, out_dir, label):
        analysis['planning'] = planning
        analysis['models'] = models
        analysis['rows'] = rows

    report, report_path = G.run_admm_arm(
        'diag_mapping', OUT_DIR, k_override=None, eval_id='p515s32_diag_mapping',
        num_max_iters_override=1, apply_rho=False, full_diagnostics_in_rows=True,
        post_run_hook=hook)

    print(f'[S32 mapping-diag] arm report: {report_path}')
    solve_counts = report['solve_profile']['observed']
    permitted_solve = solve_counts['permitted_solve']
    expected_solve = 102
    if permitted_solve != expected_solve:
        raise RuntimeError(
            f'solve-profile guard mismatch: expected exactly {expected_solve} permitted '
            f'solves (same bound as the preflight), observed {permitted_solve}; '
            f'counts={solve_counts}')
    if solve_counts['blocked_solve'] or solve_counts['blocked_exec']:
        raise RuntimeError(f'blocked solve/exec observed: {solve_counts}')
    print(f'[S32 mapping-diag] solve-profile guard: {solve_counts} (checked exactly against 102)')

    planning = analysis['planning']

    if len(CALL_LOG) != 4:
        raise RuntimeError(f'expected exactly 4 calls to _update_interface_power_flow_variables '
                            f'(pre-init + DSO + TSO + ESSO), got {len(CALL_LOG)}: '
                            f"{[c['tag'] for c in CALL_LOG]}")
    if len(BOYD_LOG) != 1:
        raise RuntimeError(f'expected exactly 1 call to get_admm_boyd_residual_metrics, got {len(BOYD_LOG)}')
    if len(UACC_LOG) != 3:
        raise RuntimeError(f'expected exactly 3 calls to update_and_check_convergence, got {len(UACC_LOG)}')

    by_tag = {c['tag']: c for c in CALL_LOG}
    preinit = by_tag['pre_admm_init']
    cyc_dso = by_tag['cycle1_dso']
    cyc_tso = by_tag['cycle1_tso']
    cyc_esso = by_tag['cycle1_esso']
    boyd = BOYD_LOG[0]

    # ==================================================================================
    # Channel-by-channel numeric verification (V, PF p, PF q). Raw ("model") units:
    # consensus values in kV / MW-MVAr (as stored, no Boyd normalization), dual values
    # as stored by the lambda update (also un-normalized by Boyd's v_base/s_base).
    # ==================================================================================
    channel_report = {}

    pf_scale = _dso_pf_lambda_scale_series(planning)  # dso_s_base/interface_rating, PF only

    def _analyze(channel_name, flatten_x, flatten_z, flatten_dual, rho_which):
        lambda_scale = pf_scale if channel_name.startswith('pf') else [1.0] * len(
            _flatten_vmag(preinit['pre']['dual_vmag']['dso']['current'], planning))
        lam0 = flatten_dual(preinit['pre'])
        x0 = flatten_x(preinit['post'])
        z0 = flatten_z(preinit['post'])
        r0_raw = _elementwise(x0, z0, lambda a, b: a - b)
        lam_after_preinit = flatten_dual(preinit['post'])
        delta_lam_preinit = _elementwise(lam_after_preinit, lam0, lambda a, b: a - b)
        rho_preinit = _rho_series_for_channel(preinit, planning, 'v' if channel_name == 'v' else 'pf', rho_which)

        # cycle-1 lambda-update time: x_DSO at cycle1_tso pre (== post, DSO untouched by
        # this call) vs z_TSO at cycle1_tso post (freshly refreshed inside this same call,
        # BEFORE the lambda block runs).
        x1 = flatten_x(cyc_tso['pre'])
        x1_post = flatten_x(cyc_tso['post'])
        z1 = flatten_z(cyc_tso['post'])
        r1_raw = _elementwise(x1, z1, lambda a, b: a - b)
        lam_before_cyc1 = flatten_dual(cyc_tso['pre'])
        lam_after_cyc1 = flatten_dual(cyc_tso['post'])
        delta_lam_cyc1 = _elementwise(lam_after_cyc1, lam_before_cyc1, lambda a, b: a - b)
        rho_cyc1 = _rho_series_for_channel(cyc_tso, planning, 'v' if channel_name == 'v' else 'pf', rho_which)

        # Boyd-time state (state passed into get_admm_boyd_residual_metrics)
        x_boyd = flatten_x(boyd['args_snapshot'])
        z_boyd = flatten_z(boyd['args_snapshot'])
        lam_boyd = flatten_dual(boyd['args_snapshot'])

        # cross-checks required by the method
        max_abs_diff_x_boyd_vs_lambda_time = _max_abs_diff(x_boyd, x1_post)
        max_abs_diff_z_boyd_vs_lambda_time = _max_abs_diff(z_boyd, z1)
        max_abs_diff_lambda_boyd_vs_lambda_time = _max_abs_diff(lam_boyd, lam_after_cyc1)

        # dual_vars['vmag'|'pf']['dso'] is untouched by the DSO-only call (update_tn=False):
        dso_call_lambda_unchanged = _max_abs_diff(
            flatten_dual(cyc_dso['pre']), flatten_dual(cyc_dso['post']))
        esso_call_vmag_pf_unchanged_x = _max_abs_diff(
            flatten_x(cyc_esso['pre']), flatten_x(cyc_esso['post']))
        esso_call_vmag_pf_unchanged_z = _max_abs_diff(
            flatten_z(cyc_esso['pre']), flatten_z(cyc_esso['post']))

        # entrywise identities (the core of the method). PF carries an extra
        # (dso_s_base/interface_rating) factor the V channel does not (F4,
        # `_update_interface_power_flow_variables`'s 'Update Lambdas' block).
        predicted_delta_lam_cyc1 = [rho * r * scale for rho, r, scale in zip(rho_cyc1, r1_raw, lambda_scale)]
        entrywise_id_cyc1 = _max_abs_diff(delta_lam_cyc1, predicted_delta_lam_cyc1)
        predicted_delta_lam_preinit = [rho * r * scale for rho, r, scale in zip(rho_preinit, r0_raw, lambda_scale)]
        entrywise_id_preinit = _max_abs_diff(delta_lam_preinit, predicted_delta_lam_preinit)
        predicted_lam_boyd = _elementwise(
            lam0, _elementwise(delta_lam_preinit, delta_lam_cyc1, lambda a, b: a + b),
            lambda a, b: a + b)
        identity_y_vs_lambda0_plus_two_deltas = _max_abs_diff(lam_boyd, predicted_lam_boyd)

        return {
            'n_entries': len(lam0),
            'norm_lambda0': _l2(lam0),
            'max_abs_lambda0': max((abs(v) for v in lam0), default=0.0),
            'norm_delta_lambda_preinit': _l2(delta_lam_preinit),
            'norm_r0_raw': _l2(r0_raw),
            'norm_delta_lambda_cycle1': _l2(delta_lam_cyc1),
            'norm_r1_raw': _l2(r1_raw),
            'norm_lambda_boyd_time': _l2(lam_boyd),
            'norm_lambda_after_preinit_plus_cycle1': _l2(predicted_lam_boyd),
            'rho_at_preinit': rho_preinit[0] if rho_preinit else None,
            'rho_at_cycle1': rho_cyc1[0] if rho_cyc1 else None,
            'max_abs_diff__x_boyd_time_vs_lambda_update_time': max_abs_diff_x_boyd_vs_lambda_time,
            'max_abs_diff__z_boyd_time_vs_lambda_update_time': max_abs_diff_z_boyd_vs_lambda_time,
            'max_abs_diff__lambda_boyd_time_vs_lambda_update_time': max_abs_diff_lambda_boyd_vs_lambda_time,
            'dso_only_call_leaves_lambda_unchanged_max_abs_diff': dso_call_lambda_unchanged,
            'esso_only_call_leaves_x_unchanged_max_abs_diff': esso_call_vmag_pf_unchanged_x,
            'esso_only_call_leaves_z_unchanged_max_abs_diff': esso_call_vmag_pf_unchanged_z,
            'entrywise_identity_max_abs_diff__delta_lambda_cycle1_vs_rho_times_r1': entrywise_id_cyc1,
            'entrywise_identity_max_abs_diff__delta_lambda_preinit_vs_rho_times_r0': entrywise_id_preinit,
            'identity_max_abs_diff__y_boyd_vs_lambda0_plus_two_deltas': identity_y_vs_lambda0_plus_two_deltas,
        }

    # ---- V ----
    # NB: `create_admm_variables` shapes consensus_vars[channel][agent] as
    # {'current': {...}, 'prev': {...}} and dual_vars[channel][agent] as
    # {'current': {...}} ONLY (no 'prev' key at all for vmag/pf duals) -- both
    # need an explicit ['current'] before the node_id level.
    channel_report['v'] = _analyze(
        'v',
        lambda snap: _flatten_vmag(snap['consensus_vmag']['dso']['current'], planning),
        lambda snap: _flatten_vmag(snap['consensus_vmag']['tso']['current'], planning),
        lambda snap: _flatten_vmag(snap['dual_vmag']['dso']['current'], planning),
        'rho_v_dso',
    )

    # ---- PF p, q ----
    for pt in ('p', 'q'):
        key = f'pf_{pt}'
        channel_report[key] = _analyze(
            key,
            lambda snap, pt=pt: _flatten_pf(snap['consensus_pf']['dso']['current'], planning, pt),
            lambda snap, pt=pt: _flatten_pf(snap['consensus_pf']['tso']['current'], planning, pt),
            lambda snap, pt=pt: _flatten_pf(snap['dual_pf']['dso']['current'], planning, pt),
            'rho_pf_dso',
        )

    # ==================================================================================
    # Cross-check against the OFFICIAL production Boyd result for cycle 1 (must match
    # exactly -- same dict objects, just re-read).
    # ==================================================================================
    official = boyd['result']
    official_summary = {
        g: {k: official[g][k] for k in ('r', 's', 's_rho_part', 's_proximal_part',
                                         'proximal_share', 'norm_x', 'norm_z', 'norm_y')}
        for g in ('v', 'pf', 'ess')
    }

    # ==================================================================================
    # ESS channel (agent-weighted; single dual-update round -- expect y == r exactly,
    # per the design comment "Shared-ESS dual variables must remain zero before the
    # first consensus-ADMM cycle").
    # ==================================================================================
    ess_dual_before_first_uacc = preinit['pre']['dual_ess']
    ess_norms_zero = {}
    for agent in ('tso', 'dso', 'esso'):
        vals = []
        for node_id, year, day in _iter_nodes_years_days(planning):
            vals.extend(ess_dual_before_first_uacc[agent]['current'][node_id][year][day]['p'])
            vals.extend(ess_dual_before_first_uacc[agent]['current'][node_id][year][day]['q'])
        ess_norms_zero[agent] = {'norm': _l2(vals), 'max_abs': max((abs(v) for v in vals), default=0.0)}

    # ==================================================================================
    # Write outputs
    # ==================================================================================
    call_log_summary = [
        {'index': c['index'], 'tag': c['tag'], 'update_tn': c['update_tn'], 'update_dns': c['update_dns']}
        for c in CALL_LOG
    ]
    uacc_summary = UACC_LOG

    payload = {
        'stage': 'P5.15 s32 mapping diagnostic (bounded, no production-code changes)',
        'authority': [
            'Planner task: bounded diagnostic, s32 Boyd residual implementation '
            '(commits 2d765573 production, 7305a476 harness)',
            'data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json',
            'data/SRP1/Results/P515S32/preflight/g_preflight.json (observation source)',
        ],
        'capture_path_checklist': checklist,
        'call_log_order': call_log_summary,
        'update_and_check_convergence_call_order': uacc_summary,
        'solve_profile_observed': solve_counts,
        'solve_profile_expected_permitted_solve': expected_solve,
        'channel_report': channel_report,
        'official_boyd_result_cycle1': official_summary,
        'ess_dual_norms_before_first_update_and_check_convergence_call': ess_norms_zero,
        'row_cycle1_selected_fields': {
            k: analysis['rows'][0].get(k) for k in (
                'boyd_v_r', 'boyd_v_s', 'boyd_v_s_rho_part', 'boyd_v_norm_y',
                'boyd_pf_r', 'boyd_pf_s', 'boyd_pf_s_rho_part', 'boyd_pf_norm_y',
                'boyd_ess_r', 'boyd_ess_s', 'boyd_ess_norm_y',
                'rho_v_before', 'rho_pf_before', 'rho_ess_before',
            )
        },
    }

    payload_path = os.path.join(OUT_DIR, 'mapping_diag_summary.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S32 mapping-diag] wrote {payload_path}')

    # Full snapshot pickle (call log + boyd log), for reproducibility; hash-recorded in
    # the manifest (see commit step), not necessarily committed if large.
    snapshots_path = os.path.join(OUT_DIR, 'raw_snapshots.pkl')
    G._refuse_overwrite(snapshots_path)
    with open(snapshots_path, 'wb') as handle:
        pickle.dump({'call_log': CALL_LOG, 'boyd_log': BOYD_LOG, 'uacc_log': UACC_LOG}, handle)
    print(f'[S32 mapping-diag] wrote {snapshots_path} ({os.path.getsize(snapshots_path)} bytes)')

    # ---- print headline numbers ----
    for g in ('v', 'pf_p', 'pf_q'):
        cr = channel_report[g]
        print(
            f"[S32 mapping-diag] {g.upper()} | "
            f"||lambda0||={cr['norm_lambda0']:.6e} | "
            f"||r0_raw||={cr['norm_r0_raw']:.6e} | "
            f"||delta_lambda_preinit||={cr['norm_delta_lambda_preinit']:.6e} | "
            f"||r1_raw||={cr['norm_r1_raw']:.6e} | "
            f"||delta_lambda_cycle1||={cr['norm_delta_lambda_cycle1']:.6e} | "
            f"||lambda_boyd_time||={cr['norm_lambda_boyd_time']:.6e} | "
            f"entrywise_id_cycle1_maxdiff={cr['entrywise_identity_max_abs_diff__delta_lambda_cycle1_vs_rho_times_r1']:.3e} | "
            f"entrywise_id_preinit_maxdiff={cr['entrywise_identity_max_abs_diff__delta_lambda_preinit_vs_rho_times_r0']:.3e} | "
            f"identity_y_vs_lambda0+2deltas_maxdiff={cr['identity_max_abs_diff__y_boyd_vs_lambda0_plus_two_deltas']:.3e}"
        )
    print(f"[S32 mapping-diag] official production Boyd cycle-1 result: {official_summary}")
    print(f"[S32 mapping-diag] ESS dual norms before first update_and_check_convergence call: {ess_norms_zero}")


if __name__ == '__main__':
    main()
