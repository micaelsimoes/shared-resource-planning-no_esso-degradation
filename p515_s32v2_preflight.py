"""
P5.15 Step 3.2 + 3.3(a) v2 (s32 worker task, residual balancing on s_rho_part)
-- one-cycle preflight.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a); binding
specification data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json
(supersedes v1 frozen_s32_spec_v1_14a18674.json). Runs `run_admm_arm` BY
IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s own `__main__` -- the
Planner launches the 150-cycle gate, not this script) for ONE ADMM cycle
through the (now v2) s32 arm machinery (case-file rho in force,
`apply_rho=False`; full per-cycle Boyd diagnostics captured in the
trajectory), into a fresh smoke root `data/SRP1/Results/P515S32/preflight_v2/`
and a fresh eval id (distinct from v1's `preflight/` and the mapping
diagnostic's `diag_mapping/` -- this script refuses to write into either),
then runs the s32 terminal writer (`write_boyd_terminal_s32`, zero extra
solves, now spec-v2-aware via the harness module's updated
`S32_SPEC_PATH`/`S32_SPEC_SHA256`) on the final models.

    python p515_s32v2_preflight.py

Verifies every field the frozen spec's `report_per_cycle_channel` requires
(now including `dual_ratio_balance`) is populated and finite for cycle 1 (or
None with a stated reason -- only `gap_proxy_G`/`gap_proxy_G_over_Q`),
recomputes the v2 3.3(a) balancing decision from the logged ratios
(`primal_ratio`, `dual_ratio_balance` -- NOT the full `dual_ratio`) to
confirm rho-after-cycle-1 follows the v2 rule, checks the Planner's
prediction (rho HELD on V, PF and ESS at cycle 1), and diffs cycle-1
residuals against the v1 preflight's committed
`data/SRP1/Results/P515S32/preflight/preflight_verification.json` (READ
ONLY -- never re-run onto that path) to confirm they are identical (v2
changes only the end-of-cycle balancing decision, not the residuals
themselves).
"""

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'preflight_v2')
FORBIDDEN_OUT_DIRS = (
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'preflight'),
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'diag_mapping'),
)
V1_PREFLIGHT_VERIFICATION_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S32', 'preflight', 'preflight_verification.json')

REQUIRED_FIELDS_PER_CHANNEL = (
    'r', 's', 's_rho_part', 's_proximal_part', 'eps_pri', 'eps_dual',
    'norm_x', 'norm_z', 'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance',
)
# Fields allowed to be None, with the reason this stage documents.
ALLOWED_NONE_FIELDS = {'gap_proxy_G', 'gap_proxy_G_over_Q'}

# Bounded solve-profile guard: `run_admm_arm` arms its OWN SolveProfileGuard
# internally (`SolveProfileGuard(N.PERMITTED, ...)`, see
# `p515_g_g1_g4_admm_gates.run_admm_arm`) and reports
# `report['solve_profile'] = {'observed': dict(guard.counts), 'identity_holds':
# guard.counts['permitted_solve'] == 51*len(rows)+51}`. For a ONE-cycle run
# that identity is 51*1+51 = 102 exactly -- the same count the v1 preflight
# measured (`solve_profile.identity_holds=True`, `permitted_solve=102`).
# This script checks the reported counts EXACTLY against 102 (not just
# `identity_holds`), per this task's explicit instruction; it does NOT
# install a second, outer guard (call-site declarations belong to the arm
# machinery, not to this script).
PERMITTED_SOLVES = 102


def _expected_action(primal_ratio, dual_ratio_balance, increase_ratio, decrease_ratio):
    if primal_ratio > increase_ratio * dual_ratio_balance:
        return 'increased'
    if dual_ratio_balance > decrease_ratio * primal_ratio:
        return 'decreased'
    return 'held'


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to start: preflight_v2 output root already exists: {OUT_DIR}')
    for forbidden in FORBIDDEN_OUT_DIRS:
        if os.path.abspath(OUT_DIR) == os.path.abspath(forbidden):
            raise RuntimeError(f'refusing to write into a v1/diagnostic output root: {forbidden}')
    G._require_fresh_output_root(OUT_DIR)

    if not os.path.exists(V1_PREFLIGHT_VERIFICATION_PATH):
        raise RuntimeError(
            f'v1 preflight comparison artifact not found (needed read-only, NOT re-run): '
            f'{V1_PREFLIGHT_VERIFICATION_PATH}')
    with open(V1_PREFLIGHT_VERIFICATION_PATH) as handle:
        v1_preflight = json.load(handle)
    v1_row = v1_preflight['cycle_1_row']

    precheck_eval_id = 'p515s32v2_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s32_capture_paths(precheck_planning)
    precheck_admm_params = precheck_planning.params.admm
    del precheck_planning
    print(f'[S32v2 preflight] capture-path pre-flight passed (spec v2): {checklist}')
    print(
        f'[S32v2 preflight] initial rho: v={precheck_admm_params.rho["v"]}, '
        f'pf={precheck_admm_params.rho["pf"]}, ess={precheck_admm_params.rho["ess"]}; '
        f'boyd_eps_source={precheck_admm_params.boyd_eps_source}'
    )
    print(f'[S32v2 preflight] spec_file={G.S32_SPEC_PATH} spec_sha256={G.S32_SPEC_SHA256}')

    def hook(planning, sed, models, rows, report, out_dir, label):
        G.write_boyd_terminal_s32(planning, sed, models, rows, report, out_dir, label)

    report, report_path = G.run_admm_arm(
        'preflight_v2', OUT_DIR, k_override=None, eval_id='p515s32v2_preflight',
        num_max_iters_override=1, apply_rho=False, full_diagnostics_in_rows=True,
        post_run_hook=hook)

    print(f'[S32v2 preflight] arm report: {report_path}')
    print(f"[S32v2 preflight] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']}")

    inner_observed = report['solve_profile'].get('observed', {})
    inner_counts_exact = (
        inner_observed.get('permitted_solve') == PERMITTED_SOLVES
        and inner_observed.get('permitted_exec') == PERMITTED_SOLVES
        and inner_observed.get('blocked_solve') == 0
        and inner_observed.get('blocked_exec') == 0
    )
    print(
        f'[S32v2 preflight] solve counts checked EXACTLY against {PERMITTED_SOLVES} '
        f"(run_admm_arm's own internal SolveProfileGuard, N.PERMITTED call sites): "
        f'observed={inner_observed} exact={inner_counts_exact} '
        f'identity_holds={report["solve_profile"].get("identity_holds")}'
    )

    rows = report['cycle_trajectory']
    if len(rows) != 1:
        raise RuntimeError(f'expected exactly one cycle, got {len(rows)}')
    row = rows[0]

    # ---- capture-field completeness / finiteness --------------------------
    def _finite(x):
        return isinstance(x, (int, float)) and math.isfinite(x)

    missing_or_nonfinite = []
    field_report = {}
    for group in ('v', 'pf', 'ess'):
        for field in REQUIRED_FIELDS_PER_CHANNEL:
            key = f'boyd_{group}_{field}'
            value = row.get(key)
            ok = _finite(value)
            field_report[key] = {'value': value, 'finite': ok}
            if not ok:
                missing_or_nonfinite.append(key)
        for key in (f'rho_{group}_before', f'rho_{group}_after', f'rho_{group}_action'):
            value = row.get(key)
            present = value is not None
            field_report[key] = {'value': value, 'present': present}
            if not present:
                missing_or_nonfinite.append(key)

    for key in ('objective_change_abs', 'objective_tolerance', 'objective_change_ratio',
                'recourse', 'gross_operational_cost'):
        value = row.get(key)
        if key in ('recourse', 'gross_operational_cost'):
            ok = _finite(value)
            field_report[key] = {'value': value, 'finite': ok}
            if not ok:
                missing_or_nonfinite.append(key)
        else:
            field_report[key] = {'value': value}

    for key in ('gap_proxy_G', 'gap_proxy_Q', 'gap_proxy_G_over_Q', 'gap_proxy_G_reason'):
        value = row.get(key)
        field_report[key] = {'value': value}
        if key in ALLOWED_NONE_FIELDS:
            if value is not None:
                missing_or_nonfinite.append(f'{key}_expected_None_got_{value}')
        elif key == 'gap_proxy_Q':
            if not _finite(value):
                missing_or_nonfinite.append(key)
        elif key == 'gap_proxy_G_reason':
            if not value:
                missing_or_nonfinite.append(key)

    all_fields_ok = (len(missing_or_nonfinite) == 0)
    print(f'[S32v2 preflight] all required per-cycle-channel fields populated: {all_fields_ok}')
    if not all_fields_ok:
        print(f'[S32v2 preflight] MISSING/NONFINITE: {missing_or_nonfinite}')

    # ---- print cycle-1 values per channel ----------------------------------
    for group in ('v', 'pf', 'ess'):
        print(
            f"[S32v2 preflight] {group.upper()} | "
            f"r={row.get(f'boyd_{group}_r'):.6e} | "
            f"s={row.get(f'boyd_{group}_s'):.6e} | "
            f"s_rho_part={row.get(f'boyd_{group}_s_rho_part'):.6e} | "
            f"eps_pri={row.get(f'boyd_{group}_eps_pri'):.6e} | "
            f"eps_dual={row.get(f'boyd_{group}_eps_dual'):.6e} | "
            f"primal_ratio={row.get(f'boyd_{group}_primal_ratio'):.6e} | "
            f"dual_ratio={row.get(f'boyd_{group}_dual_ratio'):.6e} | "
            f"dual_ratio_balance={row.get(f'boyd_{group}_dual_ratio_balance'):.6e} | "
            f"proximal_share={row.get(f'boyd_{group}_proximal_share'):.6e} | "
            f"action={row.get(f'rho_{group}_action')} | "
            f"rho_before={row.get(f'rho_{group}_before'):.6e} | "
            f"rho_after={row.get(f'rho_{group}_after'):.6e}"
        )

    # ---- recompute the v2 3.3(a) balancing decision from the logged ratios
    penalty_update = precheck_admm_params.penalty_update
    increase_ratio = penalty_update['residual_balance_ratio']
    decrease_ratio_pf = penalty_update.get('residual_balance_ratio_pf_decrease', increase_ratio)
    increase_factor = penalty_update['increase_factor']
    decrease_factor = penalty_update['decrease_factor']
    clamp_min, clamp_max = penalty_update['min'], penalty_update['max']

    balancing_recheck = {}
    balancing_all_ok = True
    for group in ('v', 'pf', 'ess'):
        primal_ratio = row.get(f'boyd_{group}_primal_ratio')
        dual_ratio_balance = row.get(f'boyd_{group}_dual_ratio_balance')
        decrease_ratio = decrease_ratio_pf if group == 'pf' else increase_ratio
        expected_action = _expected_action(primal_ratio, dual_ratio_balance, increase_ratio, decrease_ratio)
        observed_action = row.get(f'rho_{group}_action')
        rho_before = row.get(f'rho_{group}_before')
        rho_after = row.get(f'rho_{group}_after')
        if expected_action == 'increased':
            expected_after = min(max(rho_before * increase_factor, clamp_min), clamp_max)
        elif expected_action == 'decreased':
            expected_after = min(max(rho_before / decrease_factor, clamp_min), clamp_max)
        else:
            expected_after = rho_before
        ok = (
            expected_action == observed_action
            and abs(expected_after - rho_after) < 1e-9
        )
        balancing_all_ok = balancing_all_ok and ok
        balancing_recheck[group] = {
            'primal_ratio': primal_ratio, 'dual_ratio_balance': dual_ratio_balance,
            'increase_ratio': increase_ratio, 'decrease_ratio': decrease_ratio,
            'expected_action': expected_action, 'observed_action': observed_action,
            'expected_rho_after': expected_after, 'observed_rho_after': rho_after,
            'matches': ok,
        }
    print(f'[S32v2 preflight] v2 3.3(a) balancing recheck (dual_ratio_balance-gated) matches production: {balancing_all_ok}')

    # ---- Planner's prediction (from the v1 preflight values): rho HELD on
    #      V, PF and ESS at cycle 1 -----------------------------------------
    prediction = {
        group: (balancing_recheck[group]['observed_action'] == 'held')
        for group in ('v', 'pf', 'ess')
    }
    prediction_holds = all(prediction.values())
    print(f'[S32v2 preflight] Planner prediction (rho HELD on V, PF, ESS at cycle 1): '
          f'holds={prediction_holds} per_channel={prediction}')

    # ---- diff cycle-1 residuals against the v1 preflight (READ ONLY) ------
    residual_fields_to_compare = (
        'r', 's', 's_rho_part', 's_proximal_part', 'eps_pri', 'eps_dual',
        'norm_x', 'norm_z', 'norm_y', 'primal_ratio', 'dual_ratio',
    )
    v1_vs_v2_residual_diff = {}
    residuals_identical = True
    for group in ('v', 'pf', 'ess'):
        group_diff = {}
        for field in residual_fields_to_compare:
            key = f'boyd_{group}_{field}'
            v1_value = v1_row.get(key)
            v2_value = row.get(key)
            diff = None
            if isinstance(v1_value, (int, float)) and isinstance(v2_value, (int, float)):
                diff = v2_value - v1_value
                if abs(diff) > 1e-9:
                    residuals_identical = False
            elif v1_value != v2_value:
                residuals_identical = False
            group_diff[field] = {'v1': v1_value, 'v2': v2_value, 'diff': diff}
        v1_vs_v2_residual_diff[group] = group_diff
    print(f'[S32v2 preflight] cycle-1 residuals identical to v1 preflight (balancing acts only at '
          f'end of cycle, so residuals must match exactly): {residuals_identical}')

    payload = {
        'stage': 'P5.15 Step 3.2 + 3.3(a) v2 (s32 worker task, balance on s_rho_part) -- one-cycle preflight',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a)',
            'data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json',
        ],
        'spec_file': os.path.relpath(G.S32_SPEC_PATH, REPO),
        'spec_file_sha256': G.S32_SPEC_SHA256,
        'checklist': checklist,
        'cycle_1_row': row,
        'field_completeness': {
            'all_ok': all_fields_ok,
            'missing_or_nonfinite': missing_or_nonfinite,
            'per_field': field_report,
        },
        'balancing_recheck': balancing_recheck,
        'balancing_recheck_all_match': balancing_all_ok,
        'planner_prediction_rho_held_v_pf_ess': {
            'per_channel': prediction, 'holds': prediction_holds,
        },
        'v1_vs_v2_cycle1_residual_diff': v1_vs_v2_residual_diff,
        'v1_vs_v2_residuals_identical': residuals_identical,
        'v1_preflight_source_path_read_only': os.path.relpath(V1_PREFLIGHT_VERIFICATION_PATH, REPO),
        'solve_profile': report['solve_profile'],
        'solve_profile_permitted_exact': PERMITTED_SOLVES,
        'solve_profile_counts_exact': inner_counts_exact,
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_v2_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S32v2 preflight] wrote {payload_path}')

    if not (all_fields_ok and balancing_all_ok and inner_counts_exact):
        sys.exit(1)


if __name__ == '__main__':
    main()
