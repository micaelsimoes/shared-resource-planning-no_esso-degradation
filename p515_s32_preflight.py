"""
P5.15 Step 3.2 + 3.3(a) (s32 worker task) -- one-cycle preflight.

Runs `run_admm_arm` BY IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s own
`__main__` -- the Planner launches the gate, not this script) for ONE ADMM cycle
through the s32 arm machinery (case-file rho in force, `apply_rho=False`; full
per-cycle Boyd diagnostics captured in the trajectory), into a fresh smoke root
`data/SRP1/Results/P515S32/preflight/` and a fresh eval id, then runs the s32
terminal writer (`write_boyd_terminal_s32`, zero extra solves) on the final
models.

    python p515_s32_preflight.py

Verifies every field the frozen spec's `report_per_cycle_channel` requires is
populated and finite for cycle 1 (or None with a stated reason -- only
`gap_proxy_G`/`gap_proxy_G_over_Q`, per the ESSO-sigma-unavailable reason
recorded in `shared_resources_planning.py`), and recomputes the 3.3(a)
balancing decision from the logged ratios to confirm rho-after-cycle-1
follows the new (freeze-clause-removed) rule.
"""

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'preflight')

REQUIRED_FIELDS_PER_CHANNEL = (
    'r', 's', 's_rho_part', 's_proximal_part', 'eps_pri', 'eps_dual',
    'norm_x', 'norm_z', 'norm_y', 'primal_ratio', 'dual_ratio',
)
# Fields allowed to be None, with the reason this stage documents.
ALLOWED_NONE_FIELDS = {'gap_proxy_G', 'gap_proxy_G_over_Q'}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _expected_action(primal_ratio, dual_ratio, increase_ratio, decrease_ratio):
    if primal_ratio > increase_ratio * dual_ratio:
        return 'increased'
    if dual_ratio > decrease_ratio * primal_ratio:
        return 'decreased'
    return 'held'


def main():
    G._require_fresh_output_root(OUT_DIR)

    precheck_eval_id = 'p515s32_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s32_capture_paths(precheck_planning)
    precheck_admm_params = precheck_planning.params.admm
    del precheck_planning
    print(f'[S32 preflight] capture-path pre-flight passed: {checklist}')
    print(
        f'[S32 preflight] initial rho: v={precheck_admm_params.rho["v"]}, '
        f'pf={precheck_admm_params.rho["pf"]}, ess={precheck_admm_params.rho["ess"]}; '
        f'boyd_eps_source={precheck_admm_params.boyd_eps_source}'
    )

    def hook(planning, sed, models, rows, report, out_dir, label):
        G.write_boyd_terminal_s32(planning, sed, models, rows, report, out_dir, label)

    report, report_path = G.run_admm_arm(
        'preflight', OUT_DIR, k_override=None, eval_id='p515s32_preflight',
        num_max_iters_override=1, apply_rho=False, full_diagnostics_in_rows=True,
        post_run_hook=hook)

    print(f'[S32 preflight] arm report: {report_path}')
    print(f"[S32 preflight] cycles_run={report['cycles_run']} "
          f"local_solve_failures={report['local_solve_failures']} "
          f"solve_profile={report['solve_profile']} "
          f"network_failures={report['network_failures_summary']}")

    rows = report['cycle_trajectory']
    if len(rows) != 1:
        raise RuntimeError(f'expected exactly one cycle, got {len(rows)}')
    row = rows[0]

    # ---- capture-field completeness / finiteness -------------------------
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
        # objective_change_abs/ratio/tolerance are legitimately None on cycle 1
        # (no previous-cycle recourse yet); recourse/gross_operational_cost
        # must be finite.
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
    print(f'[S32 preflight] all required per-cycle-channel fields populated: {all_fields_ok}')
    if not all_fields_ok:
        print(f'[S32 preflight] MISSING/NONFINITE: {missing_or_nonfinite}')

    # ---- print cycle-1 values per channel ---------------------------------
    for group in ('v', 'pf', 'ess'):
        print(
            f"[S32 preflight] {group.upper()} | "
            f"r={row.get(f'boyd_{group}_r'):.6e} | "
            f"s={row.get(f'boyd_{group}_s'):.6e} | "
            f"eps_pri={row.get(f'boyd_{group}_eps_pri'):.6e} | "
            f"eps_dual={row.get(f'boyd_{group}_eps_dual'):.6e} | "
            f"primal_ratio={row.get(f'boyd_{group}_primal_ratio'):.6e} | "
            f"dual_ratio={row.get(f'boyd_{group}_dual_ratio'):.6e} | "
            f"proximal_share={row.get(f'boyd_{group}_proximal_share'):.6e} | "
            f"action={row.get(f'rho_{group}_action')} | "
            f"rho_before={row.get(f'rho_{group}_before'):.6e} | "
            f"rho_after={row.get(f'rho_{group}_after'):.6e}"
        )

    # ---- recompute the 3.3(a) balancing decision from the logged ratios --
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
        dual_ratio = row.get(f'boyd_{group}_dual_ratio')
        decrease_ratio = decrease_ratio_pf if group == 'pf' else increase_ratio
        expected_action = _expected_action(primal_ratio, dual_ratio, increase_ratio, decrease_ratio)
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
            'primal_ratio': primal_ratio, 'dual_ratio': dual_ratio,
            'increase_ratio': increase_ratio, 'decrease_ratio': decrease_ratio,
            'expected_action': expected_action, 'observed_action': observed_action,
            'expected_rho_after': expected_after, 'observed_rho_after': rho_after,
            'matches': ok,
        }
    print(f'[S32 preflight] 3.3(a) balancing recheck from logged ratios matches production: {balancing_all_ok}')

    payload = {
        'stage': 'P5.15 Step 3.2 + 3.3(a) (s32 worker task) -- one-cycle preflight',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a)',
            'data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json',
        ],
        'checklist': checklist,
        'cycle_1_row': row,
        'field_completeness': {
            'all_ok': all_fields_ok,
            'missing_or_nonfinite': missing_or_nonfinite,
            'per_field': field_report,
        },
        'balancing_recheck': balancing_recheck,
        'balancing_recheck_all_match': balancing_all_ok,
        'solve_profile': report['solve_profile'],
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S32 preflight] wrote {payload_path}')

    if not (all_fields_ok and balancing_all_ok):
        sys.exit(1)


if __name__ == '__main__':
    main()
