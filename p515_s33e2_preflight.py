"""
P5.15 Step 3.2 E2 (s33e2 worker task) -- two-cycle preflight.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 14; binding specification
data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json (supersedes v2
data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json). Runs
`run_admm_arm` BY IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s own
`__main__` -- the Planner launches the 150-cycle gate, not this script) for
EXACTLY TWO ADMM cycles through the (real) `s33e2` arm machinery (case-file
rho in force, `apply_rho=False`; full per-cycle Boyd/gamma diagnostics
captured in the trajectory), into a fresh smoke root
`data/SRP1/Results/P515S33/preflight_e2/`, then runs the s33e2 terminal
writer (`write_boyd_terminal_s33e2`, zero extra solves) on the final models.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s33e2_preflight.py

Tests the spec's `predictions.identity_cycles_1_2`: cycles 1 and 2
reproduce s32 EXACTLY (rho = gamma = 1 in both runs until the first rho
change, which in s32 is the V decrease at the end of cycle 2). Compares
against the COMMITTED `data/SRP1/Results/P515S32_run/g_baseline.json`
cycles 1-2 (READ ONLY -- never re-run onto that path) on
`gross_operational_cost`, every `boyd_*` field, and the rho actions/values.
Expects, at cycle 2 (as in s32): V decreased, `rho_v_after`=0.6667 and,
under the tied policy (new in e2), `gamma_v_after`=0.6667 too. If anything
differs BEFORE the end of cycle 2's update, this script reports it and
exits nonzero rather than declaring the harness ready to commit.

Also verifies every capture field is populated, and writes/verifies
`interface_voltage_terminal.json`.
"""

import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33', 'preflight_e2')
S32_G_BASELINE_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32_run', 'g_baseline.json')

NUM_CYCLES = 2
PERMITTED_SOLVES = 51 * NUM_CYCLES + 51  # 153 = 51 * 3, per the arm's own identity (see run_admm_arm)

REQUIRED_FIELDS_PER_CHANNEL = (
    'r', 's', 's_rho_part', 's_proximal_part', 'eps_pri', 'eps_dual',
    'norm_x', 'norm_z', 'norm_y', 'primal_ratio', 'dual_ratio', 'dual_ratio_balance',
)
ALLOWED_NONE_FIELDS = {'gap_proxy_G', 'gap_proxy_G_over_Q'}


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def _expected_action(primal_ratio, dual_ratio_balance, increase_ratio, decrease_ratio):
    if primal_ratio > increase_ratio * dual_ratio_balance:
        return 'increased'
    if dual_ratio_balance > decrease_ratio * primal_ratio:
        return 'decreased'
    return 'held'


def _field_completeness(row):
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
        for key in (f'gamma_{group}_before', f'gamma_{group}_after'):
            value = row.get(key)
            present = value is not None
            field_report[key] = {'value': value, 'present': present}
            if not present:
                missing_or_nonfinite.append(key)

    for key in ('objective_change_abs', 'objective_tolerance', 'objective_change_ratio',
                'recourse', 'gross_operational_cost', 'rho_freeze_active', 'freeze_after_cycle',
                'gamma_policy', 'gamma_tau'):
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

    return (len(missing_or_nonfinite) == 0), missing_or_nonfinite, field_report


def main():
    G._require_fresh_output_root(OUT_DIR)

    if not os.path.exists(S32_G_BASELINE_PATH):
        raise RuntimeError(
            f's32 reference g_baseline.json not found (needed READ ONLY, NOT re-run): {S32_G_BASELINE_PATH}')
    with open(S32_G_BASELINE_PATH) as handle:
        s32_report = json.load(handle)
    s32_rows_by_cycle = {r['cycle']: r for r in s32_report.get('cycle_trajectory', []) if r.get('cycle') is not None}
    for cycle in (1, 2):
        if cycle not in s32_rows_by_cycle:
            raise RuntimeError(f's32 reference is missing cycle {cycle} -- cannot test identity_cycles_1_2')

    precheck_eval_id = 'p515s33e2_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist = G.assert_s33e2_capture_paths(precheck_planning)
    precheck_admm_params = precheck_planning.params.admm
    del precheck_planning
    print(f'[S33E2 preflight] capture-path pre-flight passed (spec v3): {checklist}')
    print(
        f'[S33E2 preflight] initial rho: v={precheck_admm_params.rho["v"]}, '
        f'pf={precheck_admm_params.rho["pf"]}, ess={precheck_admm_params.rho["ess"]}; '
        f'boyd_eps_source={precheck_admm_params.boyd_eps_source}; '
        f'gamma_policy={precheck_admm_params.proximal_regularization["tso"]["gamma_policy"]}, '
        f'tau={precheck_admm_params.proximal_regularization["tso"]["tau"]}; '
        f'freeze_after_cycle={precheck_admm_params.penalty_update["freeze_after_cycle"]}; '
        f'minimum_consecutive_converged_cycles={precheck_admm_params.minimum_consecutive_converged_cycles}'
    )
    print(f'[S33E2 preflight] spec_file={G.S33E2_SPEC_PATH} spec_sha256={G.S33E2_SPEC_SHA256}')

    def hook(planning, sed, models, rows, report, out_dir, label):
        G.write_boyd_terminal_s33e2(planning, sed, models, rows, report, out_dir, label)

    report, report_path = G.run_admm_arm(
        'preflight_e2', OUT_DIR, k_override=None, eval_id='p515s33e2_preflight',
        num_max_iters_override=NUM_CYCLES, apply_rho=False, full_diagnostics_in_rows=True,
        post_run_hook=hook)

    print(f'[S33E2 preflight] arm report: {report_path}')
    print(f"[S33E2 preflight] cycles_run={report['cycles_run']} "
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
        f'[S33E2 preflight] solve counts checked EXACTLY against {PERMITTED_SOLVES} '
        f"(run_admm_arm's own internal SolveProfileGuard, N.PERMITTED call sites): "
        f'observed={inner_observed} exact={inner_counts_exact} '
        f'identity_holds={report["solve_profile"].get("identity_holds")}'
    )

    rows = report['cycle_trajectory']
    if len(rows) != NUM_CYCLES:
        raise RuntimeError(f'expected exactly {NUM_CYCLES} cycles, got {len(rows)}')

    # ---- capture-field completeness / finiteness (both cycles) -----------
    all_fields_ok = True
    field_completeness_by_cycle = {}
    for row in rows:
        ok, missing, field_report = _field_completeness(row)
        all_fields_ok = all_fields_ok and ok
        field_completeness_by_cycle[row['cycle']] = {
            'all_ok': ok, 'missing_or_nonfinite': missing, 'per_field': field_report,
        }
        print(f"[S33E2 preflight] cycle {row['cycle']}: all required fields populated: {ok}"
              + ('' if ok else f' MISSING/NONFINITE: {missing}'))

    # ---- print cycle values per channel ------------------------------------
    for row in rows:
        for group in ('v', 'pf', 'ess'):
            print(
                f"[S33E2 preflight] cycle {row['cycle']} {group.upper()} | "
                f"r={row.get(f'boyd_{group}_r'):.6e} | "
                f"s={row.get(f'boyd_{group}_s'):.6e} | "
                f"action={row.get(f'rho_{group}_action')} | "
                f"rho_before={row.get(f'rho_{group}_before'):.6e} | "
                f"rho_after={row.get(f'rho_{group}_after'):.6e} | "
                f"gamma_before={row.get(f'gamma_{group}_before'):.6e} | "
                f"gamma_after={row.get(f'gamma_{group}_after'):.6e}"
            )

    # ---- identity_cycles_1_2: compare cycles 1-2 against s32 (READ ONLY) --
    compare_fields = (
        'gross_operational_cost',
        'rho_v_before', 'rho_v_after', 'rho_v_action',
        'rho_pf_before', 'rho_pf_after', 'rho_pf_action',
        'rho_ess_before', 'rho_ess_after', 'rho_ess_action',
    )
    boyd_compare_fields = REQUIRED_FIELDS_PER_CHANNEL
    max_abs_diff_per_field = {}
    identity_holds = True
    first_divergence = None
    identity_detail = {}

    e2_rows_by_cycle = {r['cycle']: r for r in rows}
    for cycle in (1, 2):
        e2_row = e2_rows_by_cycle[cycle]
        s32_row = s32_rows_by_cycle[cycle]
        cycle_detail = {}
        for field in compare_fields:
            e2_v = e2_row.get(field)
            s32_v = s32_row.get(field)
            if isinstance(e2_v, (int, float)) and isinstance(s32_v, (int, float)):
                diff = abs(e2_v - s32_v)
                max_abs_diff_per_field[field] = max(max_abs_diff_per_field.get(field, 0.0), diff)
                matches = diff < 1e-6
            else:
                diff = None
                matches = (e2_v == s32_v)
            cycle_detail[field] = {'e2': e2_v, 's32': s32_v, 'abs_diff': diff, 'matches': matches}
            if not matches and first_divergence is None:
                first_divergence = {'cycle': cycle, 'field': field, 'e2': e2_v, 's32': s32_v}
            identity_holds = identity_holds and matches
        for group in ('v', 'pf', 'ess'):
            for field in boyd_compare_fields:
                key = f'boyd_{group}_{field}'
                e2_v = e2_row.get(key)
                s32_v = s32_row.get(key)
                if isinstance(e2_v, (int, float)) and isinstance(s32_v, (int, float)):
                    diff = abs(e2_v - s32_v)
                    max_abs_diff_per_field[key] = max(max_abs_diff_per_field.get(key, 0.0), diff)
                    matches = diff < 1e-6
                else:
                    diff = None
                    matches = (e2_v == s32_v)
                cycle_detail[key] = {'e2': e2_v, 's32': s32_v, 'abs_diff': diff, 'matches': matches}
                if not matches and first_divergence is None:
                    first_divergence = {'cycle': cycle, 'field': key, 'e2': e2_v, 's32': s32_v}
                identity_holds = identity_holds and matches
        identity_detail[cycle] = cycle_detail

    print(f'[S33E2 preflight] identity_cycles_1_2 holds: {identity_holds}')
    if not identity_holds:
        print(f'[S33E2 preflight] FIRST DIVERGENCE: {first_divergence}')

    # ---- expectation at cycle 2 (as in s32): V decreased, rho_v_after and,
    #      under the tied policy, gamma_v_after == 0.6667 -------------------
    cycle2 = e2_rows_by_cycle[2]
    rho_v_after_c2 = cycle2.get('rho_v_after')
    gamma_v_after_c2 = cycle2.get('gamma_v_after')
    cycle2_expectation = {
        'rho_v_action': cycle2.get('rho_v_action'),
        'rho_v_after': rho_v_after_c2,
        'gamma_v_after': gamma_v_after_c2,
        'rho_v_after_matches_0p6667': (rho_v_after_c2 is not None and abs(rho_v_after_c2 - (2.0 / 3.0)) < 1e-6),
        'gamma_v_after_matches_0p6667': (gamma_v_after_c2 is not None and abs(gamma_v_after_c2 - (2.0 / 3.0)) < 1e-6),
        'gamma_tracks_rho_at_c2': (
            rho_v_after_c2 is not None and gamma_v_after_c2 is not None
            and abs(gamma_v_after_c2 - rho_v_after_c2) < 1e-9
        ),
    }
    cycle2_expectation['pass'] = (
        cycle2_expectation['rho_v_action'] == 'decreased'
        and cycle2_expectation['rho_v_after_matches_0p6667']
        and cycle2_expectation['gamma_v_after_matches_0p6667']
        and cycle2_expectation['gamma_tracks_rho_at_c2']
    )
    print(f'[S33E2 preflight] cycle-2 expectation (V decreased, rho_v_after=gamma_v_after=0.6667): '
          f'{cycle2_expectation}')

    # ---- interface_voltage_terminal.json (also run here, per the task) ----
    # run_admm_arm does not return `models` to this caller (only via the
    # post_run_hook closure); write_interface_voltage_terminal was already
    # called by `hook` above (through write_boyd_terminal_s33e2), writing
    # data/SRP1/Results/P515S33/preflight_e2/interface_voltage_terminal.json.
    # Verify it here (read only) rather than re-deriving it -- zero extra solves.
    voltage_path = os.path.join(OUT_DIR, 'interface_voltage_terminal.json')
    voltage_written = os.path.exists(voltage_path)
    voltage_summary = None
    voltage_sane = False
    if voltage_written:
        with open(voltage_path) as handle:
            voltage_payload = json.load(handle)
        voltage_summary = voltage_payload.get('summary')
        entries = voltage_payload.get('entries', [])
        voltage_sane = (
            isinstance(voltage_payload, dict)
            and set(voltage_payload.keys()) == {'entries', 'summary'}
            and len(entries) == voltage_summary.get('n_entries')
            and all(
                (e['v_min_pu'] <= e['tso_pu']) or (e['tso_pu'] <= e['v_max_pu'])
                for e in entries
            )
        )
    print(f'[S33E2 preflight] interface_voltage_terminal.json written={voltage_written} sane={voltage_sane}')

    payload = {
        'stage': 'P5.15 Step 3.2 E2 (s33e2 worker task) -- two-cycle preflight',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 14',
            'data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json',
        ],
        'spec_file': os.path.relpath(G.S33E2_SPEC_PATH, REPO),
        'spec_file_sha256': G.S33E2_SPEC_SHA256,
        'checklist': checklist,
        'num_cycles': NUM_CYCLES,
        'cycles_rows': rows,
        'field_completeness_by_cycle': field_completeness_by_cycle,
        'field_completeness_all_ok': all_fields_ok,
        's32_reference_path_read_only': os.path.relpath(S32_G_BASELINE_PATH, REPO),
        'identity_cycles_1_2': {
            'holds': identity_holds,
            'first_divergence': first_divergence,
            'max_abs_diff_per_field': max_abs_diff_per_field,
            'detail_by_cycle': identity_detail,
        },
        'cycle2_expectation_v_decreased_rho_gamma_0p6667': cycle2_expectation,
        'interface_voltage_terminal': {
            'path': os.path.relpath(voltage_path, REPO) if voltage_written else None,
            'written': voltage_written, 'sane': voltage_sane, 'summary': voltage_summary,
        },
        'solve_profile': report['solve_profile'],
        'solve_profile_permitted_exact': PERMITTED_SOLVES,
        'solve_profile_counts_exact': inner_counts_exact,
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_e2_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S33E2 preflight] wrote {payload_path}')

    if not identity_holds:
        print('[S33E2 preflight] STOPPING: cycles 1-2 do not reproduce s32 exactly -- see first_divergence above. '
              'Per the task instruction, this must be reported rather than proceeding to commit the harness.')
        sys.exit(2)

    if not (all_fields_ok and inner_counts_exact and cycle2_expectation['pass'] and voltage_sane):
        sys.exit(1)


if __name__ == '__main__':
    main()
