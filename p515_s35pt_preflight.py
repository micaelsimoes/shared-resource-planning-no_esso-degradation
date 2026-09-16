"""
P5.15 Addendum 16 items 2-3, PHASE 2 -- TWO-cycle preflight for the
`s35pt` arm (price-taker shared-ESS initialization), NOT the 150-cycle
gate itself.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 16 items 2-3; binding
specification `data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json`
(`preflight`: "cycles: 2", `expected_signatures`).

Runs `run_admm_arm` BY IMPORT (never through `p515_g_g1_g4_admm_gates.py`'s
own `__main__` -- the Planner launches the 150-cycle gate, THIS script
never does) for EXACTLY TWO ADMM cycles through the (real) `s35pt` arm
machinery (case-file rho in force, `apply_rho=False`; case-file
`admm.shared_ess_initialization = "price_taker"` in force; full per-cycle
diagnostics captured; `s35pt_capture_hooks` wired exactly as the gate wires
them), into a fresh smoke root `data/SRP1/Results/P515S35/preflight_pt/`,
then runs the s35pt terminal writer (`write_boyd_terminal_s35pt`, zero
extra solves) on the final models.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s35pt_preflight.py

Reference numbers (run 1 / s35ref cycles 1-2) are READ from run 1's own
committed `g_baseline.json`, by path, hashed -- never typed in as literals
(same discipline the `s35pt` gate arm's gate-3 uses).

Checks spec v6's `preflight.expected_signatures`:
  1. EFC near the LP harness-definition value at cycle 1 (not ~0.0003 as in
     run 1 cycle 1); change in EFC from cycle 1 to 2 below 1%.
  2. Small ESS Boyd DUAL residual RATIO (`boyd_ess_dual_ratio`) at cycles
     1-2, compared against run 1's own cycles 1-2 (read from
     `g_baseline.json`, not the literal 19.4/17.8 the task quotes from
     memory -- see `_read_reference_rows`); per-cell ESS lambda (TSO + DSO
     + ESSO, per node/year/day/period) summing to zero to machine
     precision at every captured cycle (captured via an ADDITIONAL
     monkeypatch of `get_admm_boyd_residual_metrics`, layered on top of
     `s35pt_capture_hooks`'s own wrapper, storing `dual_vars['ess']` --
     already an argument of that function -- once per cycle; zero extra
     solves); small TSO proximal movement on ESS
     (`boyd_ess_s_proximal_part`, the production-computed proximal
     contribution to the dual residual), compared against run 1.
  3. No solve failures / network-failure restorations attributable to the
     initialization (`local_solve_failures == 0`,
     `network_failures_summary.n_blocks == 0`).
  4. PF primal residual MAY exceed run 1 cycle 1's (interface flows carry
     storage from the start) -- reported, not a failure either way.
  5. Diagnostics (STOP and report if seen, do not commit): EFC ~ 0 at
     cycle 1 would mean the injection is not wired -- subsumed by check 1
     and by Z3 (`p515_s35pt_phase2_checks.py`, already machine-precision
     PASS at t=0, cross-referenced here); TSO ESS copy ~ z/2 and ESSO pnet
     falling away from z are likewise already excluded at t=0 by Z3/Z4
     (machine precision) -- this preflight is the LIVE, post-solve
     corroboration that nothing regresses once real IPOPT cycles run.
"""

import hashlib
import json
import math
import os
import sys
from copy import deepcopy

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402 -- reused BY CALLING, not copied
import p515_s34_preflight as S34PF  # noqa: E402 -- reused BY CALLING, not copied

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35', 'preflight_pt')

NUM_CYCLES = 2
# 51 solves per cycle + 51 for initialization (same identity `run_admm_arm`
# itself checks internally, `report['solve_profile']['identity_holds']`,
# and Z3/Z4 (p515_s35pt_phase2_checks.py) independently confirmed the
# initialization count of 51 by a bounded dry run): 51*2 + 51 = 153, same
# as s34's own two-cycle preflight.
PERMITTED_SOLVES = 51 * NUM_CYCLES + 51

EFC_NEAR_LP_FRACTION_MIN = 0.5   # cycle-1 EFC must be at least half the LP target (not ~0)
EFC_CYCLE1_TO_2_REL_CHANGE_MAX = 0.01
ESS_LAMBDA_SUM_TOL = 1e-9


def _read_reference_rows_1_2():
    """Run 1 (s35ref) cycles 1-2, read BY PATH from its own committed
    `g_baseline.json`, hashed -- never typed in as literals."""
    with open(G.S35PT_S35REF_G_PATH, 'rb') as handle:
        raw = handle.read()
    data = json.loads(raw)
    rows = data.get('cycle_trajectory', [])
    by_cycle = {r['cycle']: r for r in rows if r.get('cycle') is not None}
    return by_cycle.get(1, {}), by_cycle.get(2, {}), {
        'path': os.path.relpath(G.S35PT_S35REF_G_PATH, REPO),
        'sha256': hashlib.sha256(raw).hexdigest(),
    }


def _lp_max_efc(lp_result):
    best = None
    for node_id, node_result in (lp_result or {}).items():
        for year, val in (node_result.get('efc_per_day_harness') or {}).items():
            if best is None or val > best:
                best = val
    return best


def main():
    G._require_fresh_output_root(OUT_DIR)

    precheck_eval_id = 'p515s35pt_preflight_precheck'
    precheck_eval_dir = os.path.join(G.O.WORK_DIR, precheck_eval_id)
    if os.path.exists(precheck_eval_dir):
        raise RuntimeError(f'refusing to start: precheck eval dir already exists: {precheck_eval_dir}')
    precheck_planning = G.O.fresh_planning(precheck_eval_id)
    checklist, floor_rows_by_node = G.assert_s35pt_capture_paths(precheck_planning)
    precheck_admm_params = precheck_planning.params.admm
    del precheck_planning
    print(f'[S35PT preflight] capture-path pre-flight passed (spec v6): {checklist}')
    print(
        f'[S35PT preflight] initial rho: v={precheck_admm_params.rho["v"]}, '
        f'pf={precheck_admm_params.rho["pf"]}, ess={precheck_admm_params.rho["ess"]}; '
        f'shared_ess_initialization={precheck_admm_params.shared_ess_initialization} '
        f'(source={precheck_admm_params.shared_ess_initialization_source})'
    )
    print(f'[S35PT preflight] spec_file={G.S35PT_SPEC_PATH} spec_sha256={G.S35PT_SPEC_SHA256}')

    rho_ess_ok = all(float(v) == G.S35PT_INITIAL_RHO_ESS for v in precheck_admm_params.rho['ess'].values())
    print(f'[S35PT preflight] rho_ess == {G.S35PT_INITIAL_RHO_ESS} on every network and the ESSO: {rho_ess_ok}')

    ref_row1, ref_row2, ref_source = _read_reference_rows_1_2()
    print(f'[S35PT preflight] run-1 (s35ref) reference rows 1-2 read from {ref_source}')

    recourse_jump_path = os.path.join(OUT_DIR, 'recourse_jump_sidecar_preflight.jsonl')
    ess_stride_path = os.path.join(OUT_DIR, 'ess_entry_stride_preflight.jsonl')
    floor_sidecar_path = os.path.join(OUT_DIR, 'soh_floor_sidecar_preflight.jsonl')

    price_taker_capture = {}
    ess_dual_snapshots = []  # one entry per cycle: {'cycle': int, 'dual_ess': deepcopy(dual_vars['ess'])}

    def hook(planning, sed, models, rows, report, out_dir, label):
        report['s34_recourse_jump_sidecar_path'] = os.path.relpath(recourse_jump_path, G.REPO)
        report['s34_ess_entry_stride_sidecar_path'] = os.path.relpath(ess_stride_path, G.REPO)
        report['s35ref_soh_floor_sidecar_path'] = os.path.relpath(floor_sidecar_path, G.REPO)
        G.write_boyd_terminal_s35pt(planning, sed, models, rows, report, out_dir, label,
                                     floor_rows_by_node=floor_rows_by_node,
                                     floor_sidecar_path=floor_sidecar_path,
                                     price_taker_capture=price_taker_capture)

    with G.s35pt_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                floor_rows_by_node, price_taker_capture, stride=1):
        # Layer a SECOND monkeypatch of get_admm_boyd_residual_metrics on
        # top of s35pt_capture_hooks's own (itself layered on
        # s34_capture_hooks's) -- same pattern s35ref_capture_hooks used to
        # layer onto s34_capture_hooks. Captures dual_vars['ess'] (already
        # an argument of this function) once per cycle, zero extra solves.
        inner_fn = G.srp.get_admm_boyd_residual_metrics

        def dual_snapshot_wrapper(planning_problem, tso_model, dso_models, esso_model,
                                   consensus_vars, dual_vars, admm_parameters):
            result = inner_fn(planning_problem, tso_model, dso_models, esso_model,
                               consensus_vars, dual_vars, admm_parameters)
            ess_dual_snapshots.append({
                'cycle': len(ess_dual_snapshots) + 1,
                'dual_ess': deepcopy(dual_vars['ess']),
            })
            return result

        G.srp.get_admm_boyd_residual_metrics = dual_snapshot_wrapper
        try:
            report, report_path = G.run_admm_arm(
                'preflight', OUT_DIR, k_override=None, eval_id='p515s35pt_preflight',
                num_max_iters_override=NUM_CYCLES, apply_rho=False, full_diagnostics_in_rows=True,
                post_run_hook=hook)
        finally:
            G.srp.get_admm_boyd_residual_metrics = inner_fn

    print(f'[S35PT preflight] arm report: {report_path}')
    print(f"[S35PT preflight] cycles_run={report['cycles_run']} "
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
        f'[S35PT preflight] solve counts checked EXACTLY against {PERMITTED_SOLVES} '
        f"(run_admm_arm's own internal SolveProfileGuard, N.PERMITTED call sites; "
        f"2 cycles plus initialization = 51*3): "
        f'observed={inner_observed} exact={inner_counts_exact} '
        f'identity_holds={report["solve_profile"].get("identity_holds")}'
    )

    rows = report['cycle_trajectory']
    if len(rows) != NUM_CYCLES:
        raise RuntimeError(f'expected exactly {NUM_CYCLES} cycles, got {len(rows)}')
    if len(ess_dual_snapshots) != NUM_CYCLES:
        raise RuntimeError(
            f'expected exactly {NUM_CYCLES} ESS dual snapshots, got {len(ess_dual_snapshots)}')

    all_fields_ok = True
    field_completeness_by_cycle = {}
    for row in rows:
        ok, missing, field_report = S34PF._field_completeness(row)
        all_fields_ok = all_fields_ok and ok
        field_completeness_by_cycle[row['cycle']] = {
            'all_ok': ok, 'missing_or_nonfinite': missing, 'per_field': field_report,
        }
        print(f"[S35PT preflight] cycle {row['cycle']}: all required fields populated: {ok}"
              + ('' if ok else f' MISSING/NONFINITE: {missing}'))

    cycle1, cycle2 = rows[0], rows[1]

    # ---- Signature 1: EFC near the LP value at cycle 1, not ~0.0003 -------
    lp_result = price_taker_capture.get('lp_result') or {}
    lp_max_efc = _lp_max_efc(lp_result)
    efc_cycle1 = cycle1.get('efc_per_day_max')
    efc_cycle2 = cycle2.get('efc_per_day_max')
    ref_efc_cycle1 = ref_row1.get('efc_per_day_max')

    efc_near_lp = (
        lp_max_efc is not None and efc_cycle1 is not None and lp_max_efc > 0
        and (efc_cycle1 / lp_max_efc) >= EFC_NEAR_LP_FRACTION_MIN
    )
    efc_far_from_ref_cycle1 = (
        efc_cycle1 is not None and ref_efc_cycle1 is not None
        and efc_cycle1 > 50 * ref_efc_cycle1  # run 1 cycle 1 was ~0.0003; s35pt should be orders of magnitude larger
    )
    efc_change_1_to_2 = (
        abs(efc_cycle2 - efc_cycle1) / efc_cycle1
        if (efc_cycle1 is not None and efc_cycle2 is not None and efc_cycle1 != 0) else None
    )
    efc_change_below_1pct = (efc_change_1_to_2 is not None and efc_change_1_to_2 < EFC_CYCLE1_TO_2_REL_CHANGE_MAX)

    signature_1_pass = bool(efc_near_lp and efc_far_from_ref_cycle1 and efc_change_below_1pct)
    print(
        f'[S35PT preflight] SIGNATURE 1 (EFC near LP at cycle 1, not ~0.0003; change < 1%): '
        f'lp_max_efc={lp_max_efc} efc_cycle1={efc_cycle1} efc_cycle1/lp_max={efc_cycle1 / lp_max_efc if (lp_max_efc and efc_cycle1 is not None) else None} '
        f'ref_s35ref_efc_cycle1={ref_efc_cycle1} efc_cycle2={efc_cycle2} '
        f'rel_change_1_to_2={efc_change_1_to_2} PASS={signature_1_pass}'
    )

    # ---- Signature 2a: small ESS Boyd dual residual RATIO vs run 1 --------
    dual_ratio_c1 = cycle1.get('boyd_ess_dual_ratio')
    dual_ratio_c2 = cycle2.get('boyd_ess_dual_ratio')
    ref_dual_ratio_c1 = ref_row1.get('boyd_ess_dual_ratio')
    ref_dual_ratio_c2 = ref_row2.get('boyd_ess_dual_ratio')
    print(
        f'[S35PT preflight] SIGNATURE 2a (ESS Boyd dual residual ratio, s35pt vs run 1): '
        f'cycle1={dual_ratio_c1} (ref={ref_dual_ratio_c1}) cycle2={dual_ratio_c2} (ref={ref_dual_ratio_c2})'
    )

    # ---- Signature 2b: per-cell ESS lambda sums to zero, every cycle ------
    lambda_sum_report = {}
    lambda_sum_max_abs = 0.0
    for snap in ess_dual_snapshots:
        cyc = snap['cycle']
        dual_ess = snap['dual_ess']
        node_ids = list(dual_ess.get('tso', {}).get('current', {}).keys())
        max_abs = 0.0
        n_cells = 0
        for node_id in node_ids:
            years_map = dual_ess['tso']['current'][node_id]
            for year in years_map:
                days_map = years_map[year]
                for day in days_map:
                    n_periods = len(days_map[day]['p'])
                    for p in range(n_periods):
                        for power_type in ('p', 'q'):
                            total = sum(
                                dual_ess[agent]['current'][node_id][year][day][power_type][p]
                                for agent in ('tso', 'dso', 'esso')
                            )
                            n_cells += 1
                            max_abs = max(max_abs, abs(total))
        lambda_sum_report[cyc] = {'n_cells': n_cells, 'max_abs_sum': max_abs}
        lambda_sum_max_abs = max(lambda_sum_max_abs, max_abs)
        print(f'[S35PT preflight] SIGNATURE 2b (per-cell ESS lambda sum to zero) cycle {cyc}: '
              f'n_cells={n_cells} max_abs_sum={max_abs:.3e}')
    lambda_sum_ok = (lambda_sum_max_abs <= ESS_LAMBDA_SUM_TOL)

    # ---- Signature 2c: small TSO proximal movement on ESS -----------------
    prox_c1 = cycle1.get('boyd_ess_s_proximal_part')
    prox_c2 = cycle2.get('boyd_ess_s_proximal_part')
    ref_prox_c1 = ref_row1.get('boyd_ess_s_proximal_part')
    ref_prox_c2 = ref_row2.get('boyd_ess_s_proximal_part')
    print(
        f'[S35PT preflight] SIGNATURE 2c (TSO proximal movement on ESS, boyd_ess_s_proximal_part): '
        f'cycle1={prox_c1} (ref={ref_prox_c1}) cycle2={prox_c2} (ref={ref_prox_c2})'
    )

    # ---- Signature 3: no solve failures / restorations attributable ------
    no_failures = (report['local_solve_failures'] == 0
                   and report['network_failures_summary'].get('n_blocks', 0) == 0)
    print(f"[S35PT preflight] SIGNATURE 3 (no solve failures/restorations): "
          f"local_solve_failures={report['local_solve_failures']} "
          f"n_blocks={report['network_failures_summary'].get('n_blocks')} PASS={no_failures}")

    # ---- Signature 4: PF primal residual MAY exceed run 1 cycle 1 (reported only) ----
    pf_primal_c1 = cycle1.get('primal_pf')
    ref_pf_primal_c1 = ref_row1.get('primal_pf')
    pf_exceeds_ref = (
        pf_primal_c1 is not None and ref_pf_primal_c1 is not None and pf_primal_c1 > ref_pf_primal_c1
    )
    print(f'[S35PT preflight] SIGNATURE 4 (PF primal residual, may exceed run 1 cycle 1 -- expected, '
          f'not a failure): s35pt_cycle1={pf_primal_c1} ref_cycle1={ref_pf_primal_c1} '
          f'exceeds_ref={pf_exceeds_ref}')

    # ---- Diagnostics red flags -------------------------------------------
    efc_approx_zero_red_flag = (efc_cycle1 is not None and efc_cycle1 < 1e-3)
    print(f'[S35PT preflight] DIAGNOSTIC red flag check: efc_per_day_max cycle1 ~ 0 '
          f'(injection not wired): {efc_approx_zero_red_flag} (value={efc_cycle1}); '
          f'TSO-copy~z/2 and ESSO-pnet-drift are excluded at t=0 by Z3/Z4 '
          f'(p515_s35pt_phase2_checks.py, machine precision) -- see that report.')

    overall_ok = (
        all_fields_ok and inner_counts_exact and rho_ess_ok
        and signature_1_pass and lambda_sum_ok and no_failures
        and not efc_approx_zero_red_flag
    )

    payload = {
        'stage': 'P5.15 Addendum 16 items 2-3, PHASE 2 -- two-cycle preflight (spec v6, s35pt arm)',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 16 items 2 and 3',
            'data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json',
        ],
        'spec_file': os.path.relpath(G.S35PT_SPEC_PATH, REPO),
        'spec_file_sha256': G.S35PT_SPEC_SHA256,
        'checklist': checklist,
        'num_cycles': NUM_CYCLES,
        'permitted_solves': PERMITTED_SOLVES,
        'solve_profile': report['solve_profile'],
        'solve_profile_counts_exact': inner_counts_exact,
        'rho_ess_matches_0p1125': rho_ess_ok,
        'reference_source_run1_s35ref': ref_source,
        'cycles_rows': rows,
        'field_completeness_by_cycle': field_completeness_by_cycle,
        'field_completeness_all_ok': all_fields_ok,
        'lp_result_max_efc_per_day_harness': lp_max_efc,
        'price_taker_initialization_capture': {
            'clipped_q_cells': price_taker_capture.get('clipped_q_cells'),
        },
        'signature_1_efc_near_lp_at_cycle1': {
            'lp_max_efc': lp_max_efc, 'efc_cycle1': efc_cycle1, 'efc_cycle2': efc_cycle2,
            'efc_cycle1_over_lp_max': (efc_cycle1 / lp_max_efc) if (lp_max_efc and efc_cycle1 is not None) else None,
            'ref_s35ref_efc_cycle1': ref_efc_cycle1,
            'rel_change_1_to_2': efc_change_1_to_2,
            'efc_near_lp': efc_near_lp, 'efc_far_from_ref_cycle1': efc_far_from_ref_cycle1,
            'efc_change_below_1pct': efc_change_below_1pct,
            'pass': signature_1_pass,
        },
        'signature_2a_ess_boyd_dual_ratio': {
            's35pt_cycle1': dual_ratio_c1, 's35pt_cycle2': dual_ratio_c2,
            'ref_s35ref_cycle1': ref_dual_ratio_c1, 'ref_s35ref_cycle2': ref_dual_ratio_c2,
        },
        'signature_2b_per_cell_ess_lambda_sum_to_zero': {
            'tolerance': ESS_LAMBDA_SUM_TOL, 'by_cycle': lambda_sum_report,
            'max_abs_sum_any_cycle': lambda_sum_max_abs, 'pass': lambda_sum_ok,
        },
        'signature_2c_tso_proximal_movement_on_ess': {
            's35pt_cycle1': prox_c1, 's35pt_cycle2': prox_c2,
            'ref_s35ref_cycle1': ref_prox_c1, 'ref_s35ref_cycle2': ref_prox_c2,
        },
        'signature_3_no_failures': {
            'local_solve_failures': report['local_solve_failures'],
            'network_failures_n_blocks': report['network_failures_summary'].get('n_blocks'),
            'pass': no_failures,
        },
        'signature_4_pf_primal_residual_reported_only': {
            's35pt_cycle1': pf_primal_c1, 'ref_s35ref_cycle1': ref_pf_primal_c1,
            'exceeds_ref': pf_exceeds_ref,
        },
        'diagnostic_red_flags': {
            'efc_approx_zero_at_cycle1': efc_approx_zero_red_flag,
            'tso_copy_and_esso_pnet_drift_excluded_at_t0_by': 'p515_s35pt_phase2_checks.py (Z3/Z4)',
        },
        'overall_ok': overall_ok,
        'local_solve_failures': report['local_solve_failures'],
    }
    payload_path = os.path.join(OUT_DIR, 'preflight_verification.json')
    G._refuse_overwrite(payload_path)
    with open(payload_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S35PT preflight] wrote {payload_path}')
    print(f'[S35PT preflight] OVERALL_OK={overall_ok}')

    if not overall_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
