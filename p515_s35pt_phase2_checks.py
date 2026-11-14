"""
P5.15 Addendum 16 items 2-3 -- PHASE 2 bounded-solve checks (frozen spec
v6, `data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json`).

Runs Z3, Z4, and the IPOPT half of Z7 -- the three `zero_solve_checks`
PHASE 1 explicitly deferred (`WORKER_REPORT_S35PT_PHASE1.md`, "Remaining
issues"), each needing the standalone initialization solves (BOUNDED, not
zero -- the standalone per-(network,year,day) SMOPF solves and the ESSO
standalone `optimize` call that `create_distribution_networks_models`,
`create_transmission_network_model` and `create_shared_energy_storage_model`
perform unconditionally as part of model construction).

Reuses production/harness functions BY CALLING THEM, never reimplementing:
  * `p515_g_g1_g4_admm_gates._construct_arm_planning` -- constructs the
    planning object EXACTLY as the `s35pt` gate arm does (same
    `O.fresh_planning`, same `apply_rho=False` case-file rho, same
    S_INV/E_INV candidate at N.INVEST_YEAR, same
    `_rebuild_candidate_total_capacities`).
  * `shared_resources_planning` module-level functions
    (`create_admm_variables`, `create_distribution_networks_models`,
    `create_transmission_network_model`, `create_shared_energy_storage_model`,
    `update_distribution_coordination_models_and_solve`,
    `_initialize_shared_ess_from_price_taker`) -- monkeypatched to CAPTURE
    (call through to the real function, store its return value/arguments),
    never reimplemented.

## How the pre-cycle-1 point is reached (without running any ADMM cycle)

A monkeypatched stop on the first cycle-1 DSO solve
(`update_distribution_coordination_models_and_solve`) that RAISES
`_PreCycle1Stop` BEFORE calling through -- so it contributes zero solves and
zero ADMM cycles. The four ADMM-model constructors above are separately
monkeypatched (call-through wrappers) to capture their OWN return values --
the SAME mutable objects (`consensus_vars`, `dual_vars`, `dso_models`,
`tso_model`, `esso_model`) `_run_operational_planning` continues to mutate
IN PLACE (never reconstructed) between construction and the loop. Reading
these captured references out only AFTER `_PreCycle1Stop` is caught (i.e.
only once execution has reached the exact pre-cycle-1 point) means every
pre-loop step between construction and the loop -- in particular
`_initialize_shared_ess_from_price_taker` and
`update_interface_power_flow_variables` -- has already run and is reflected
in the captured state, satisfying the task's "not merely right after the
wrapper returns" requirement.

`shared_ess_price_taker.solve_price_taker_schedule` is ALSO monkeypatched
for the Z3 (price-taker) construction only, to capture the ONE LP result
the real wrapper computes internally -- NOT a second, redundant LP solve.

## Bounded guards

Two independent `SolveProfileGuard` instances (Z3, Z4), each declared and
verified at EXACTLY 51 permitted IPOPT solves: 3 DSOs x 3 years x 4 days =
36 SMOPF solves (`network.py:_run_smopf_solver_attempt`), TSO x 3 x 4 = 12
(same call site), ESSO standalone `optimize` x 3 active nodes = 3
(`shared_energy_storage_data.py:_run_solver_attempt`) -- discovered by one
dry run against `p514_n_instrumented_cstar.PERMITTED`'s own declared call
sites (reported verbatim under `solve_profile_discovery` below) before this
script was written, per "declare the permitted IPOPT count in advance and
check it exactly". Z7's IPOPT half is exactly
`z3['ipopt_solve_count'] == z4['ipopt_solve_count'] == 51`.

The LP call counter (`shared_ess_price_taker.get_lp_call_count`) is
declared and checked at EXACTLY 120 for Z3 (3 active nodes x 40 damped
outer iterations, `wear_on=True` default -- the same count Phase 1's own
Z2p check found) and EXACTLY 0 for Z4 (the standalone flag never calls the
wrapper, so `shared_ess_price_taker.solve_price_taker_schedule` is never
reached).

Refuses to run if its own output directory already exists. Does NOT touch
`data/SRP1/SRP1_params.json`, and does NOT invoke
`p515_g_g1_g4_admm_gates.py`'s `__main__` gate dispatch (its module-level
helper `_construct_arm_planning` is called directly; the module is only
imported, never run as `__main__`) -- no harness output directory, lock
file, or campaign artifact is touched.

Usage:
    python p515_s35pt_phase2_checks.py
"""

import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35', 'pt_phase2_checks')

Z3_EVAL_ID = 'p515s35pt_phase2_z3'
Z4_EVAL_ID = 'p515s35pt_phase2_z4'

# Declared IN ADVANCE (dry run against N.PERMITTED's own call sites, before
# this script was written -- see module docstring "Bounded guards"; also
# re-derived and cross-checked live below via `_discover_permitted_sites`).
DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION = 51
DECLARED_LP_CALLS_Z3 = 120
DECLARED_LP_CALLS_Z4 = 0
NON_ESS_TOLERANCE = 1e-9
ESS_DIFFERS_FRACTION_MAX = 0.5  # Z4's z must match the LP schedule in fewer than half its cells


class _PreCycle1Stop(Exception):
    """Raised by the monkeypatched `update_distribution_coordination_models_and_solve`
    BEFORE calling through -- the pre-cycle-1 capture point."""


def _pv(x):
    return float(pe.value(x))


def _max_abs_diff_nested(a, b, path=''):
    if isinstance(a, dict):
        max_diff, worst_path = 0.0, None
        for key in a:
            d, p = _max_abs_diff_nested(a[key], b[key], f'{path}/{key}')
            if d > max_diff:
                max_diff, worst_path = d, p
        return max_diff, worst_path
    if isinstance(a, (list, tuple)):
        max_diff, worst_path = 0.0, None
        for i, (av, bv) in enumerate(zip(a, b)):
            d, p = _max_abs_diff_nested(av, bv, f'{path}[{i}]')
            if d > max_diff:
                max_diff, worst_path = d, p
        return max_diff, worst_path
    return abs(float(a) - float(b)), path


# ======================================================================================================================
#  Pre-cycle-1 construction, shared by Z3 and Z4
# ======================================================================================================================
def _run_precycle1_capture(G, srp, N, label, out_dir, eval_id, force_standalone=False, capture_lp=False):
    """Constructs the planning object EXACTLY as the `s35pt` arm does
    (`G._construct_arm_planning`, `apply_rho=False` -- case-file rho/flag in
    force), optionally forcing `admm.shared_ess_initialization = 'standalone'`
    on a DEEP-COPIED `planning.params` (Z4; the case file itself is never
    touched), then runs `planning.run_operational_planning` up to (not
    including) cycle 1's first DSO solve. Returns
    (planning, sed, candidate, capture, guard).
    """
    import p56a_oracle as O

    report = {}
    planning, sed, candidate = G._construct_arm_planning(
        label, out_dir, report, k_override=None, investment_map=None,
        eval_id=eval_id, num_max_iters_override=G.S35PT_CAP, apply_rho=False)

    if force_standalone:
        # Deep-copy `planning.params` (NOT the case file) before mutating,
        # per task instruction -- isolates this override from anything else
        # `O.fresh_planning`'s own deepcopy might still share.
        planning.params = deepcopy(planning.params)
        planning.params.admm.shared_ess_initialization = 'standalone'
        planning.params.admm.shared_ess_initialization_source = 'phase2_check_override_not_case_file'

    capture = {}
    real_admm_vars = srp.create_admm_variables
    real_dso = srp.create_distribution_networks_models
    real_tso = srp.create_transmission_network_model
    real_esso = srp.create_shared_energy_storage_model
    real_dso_solve = srp.update_distribution_coordination_models_and_solve
    real_pt_init = srp._initialize_shared_ess_from_price_taker

    real_solve_pt = None
    sept = None
    if capture_lp:
        import shared_ess_price_taker as sept
        real_solve_pt = sept.solve_price_taker_schedule

    def w_admm_vars(pp):
        cv, dv = real_admm_vars(pp)
        capture['consensus_vars'] = cv
        capture['dual_vars'] = dv
        return cv, dv

    def w_dso(distribution_networks, consensus_vars, total_capacity, parallel_execution=False):
        m, r = real_dso(distribution_networks, consensus_vars, total_capacity, parallel_execution=parallel_execution)
        capture['dso_models'] = m
        return m, r

    def w_tso(pp, consensus_vars, total_capacity):
        m, r = real_tso(pp, consensus_vars, total_capacity)
        capture['tso_model'] = m
        return m, r

    def w_esso(sed_, consensus_vars, investment):
        m, r = real_esso(sed_, consensus_vars, investment)
        capture['esso_model'] = m
        return m, r

    def w_pt_init(planning_problem, candidate_solution, tso_model, esso_model, consensus_vars):
        clipped = real_pt_init(planning_problem, candidate_solution, tso_model, esso_model, consensus_vars)
        capture['clipped_q_cells'] = clipped
        return clipped

    def w_solve_pt(*args, **kwargs):
        result = real_solve_pt(*args, **kwargs)
        capture['lp_result'] = result
        return result

    def w_dso_solve(*args, **kwargs):
        raise _PreCycle1Stop()

    srp.create_admm_variables = w_admm_vars
    srp.create_distribution_networks_models = w_dso
    srp.create_transmission_network_model = w_tso
    srp.create_shared_energy_storage_model = w_esso
    srp.update_distribution_coordination_models_and_solve = w_dso_solve
    srp._initialize_shared_ess_from_price_taker = w_pt_init
    if capture_lp:
        sept.solve_price_taker_schedule = w_solve_pt

    guard = SolveProfileGuard(N.PERMITTED, label=f'P5.15-S35PT-PHASE2 {label}').install()
    stopped = False
    try:
        try:
            planning.run_operational_planning(
                type='distributed', candidate_solution=deepcopy(candidate),
                print_results=False, debug_flag=False, return_state=True)
        except _PreCycle1Stop:
            stopped = True
    finally:
        guard.uninstall()
        srp.create_admm_variables = real_admm_vars
        srp.create_distribution_networks_models = real_dso
        srp.create_transmission_network_model = real_tso
        srp.create_shared_energy_storage_model = real_esso
        srp.update_distribution_coordination_models_and_solve = real_dso_solve
        srp._initialize_shared_ess_from_price_taker = real_pt_init
        if capture_lp:
            sept.solve_price_taker_schedule = real_solve_pt

    if not stopped:
        raise RuntimeError(
            f'{label}: expected _PreCycle1Stop to be raised before any cycle-1 DSO solve; it was not')

    return planning, sed, candidate, capture, guard


# ======================================================================================================================
#  Z3 -- price-taker construction: z/copies/proximal centres/ESSO state == LP; ESS duals == 0
# ======================================================================================================================
def _run_z3(G, srp, N, sept):
    out_dir = os.path.join(OUT_DIR, 'z3_construction')
    sept.reset_lp_call_count()
    planning, sed, candidate, capture, guard = _run_precycle1_capture(
        G, srp, N, 'z3', out_dir, Z3_EVAL_ID, force_standalone=False, capture_lp=True)

    problems = []
    consensus_vars = capture['consensus_vars']
    dual_vars = capture['dual_vars']
    tso_model = capture['tso_model']
    esso_model = capture['esso_model']
    lp_result = capture['lp_result']
    clipped_q_cells_reported = capture.get('clipped_q_cells')

    years = list(sed.years)
    days = list(sed.days)
    n_periods = planning.num_instants
    node_ids = list(sed.active_distribution_network_nodes)

    z = consensus_vars['ess']['z']
    clipped_q_recount = 0
    cells_checked = 0
    for node_id in node_ids:
        s_by_year = lp_result[node_id]['s_nameplate_per_year']
        for year in years:
            s_avail = s_by_year[year]
            for day in days:
                lp_p = lp_result[node_id]['p'][year][day]
                for p in range(n_periods):
                    cells_checked += 1
                    z_cur_p = z['current'][node_id][year][day]['p'][p]
                    z_prev_p = z['prev'][node_id][year][day]['p'][p]
                    if abs(z_cur_p - float(lp_p[p])) > 1e-9 or abs(z_prev_p - float(lp_p[p])) > 1e-9:
                        problems.append(f'Z3: z p != LP p at {node_id}/{year}/{day}/{p}')

                    z_cur_q = z['current'][node_id][year][day]['q'][p]
                    if (z_cur_q ** 2 + z_cur_p ** 2) > s_avail ** 2 + 1e-6:
                        clipped_q_recount += 1
                        problems.append(f'Z3: q outside converter circle at {node_id}/{year}/{day}/{p}')

                    for agent in ('tso', 'dso', 'esso'):
                        for tag in ('current', 'prev'):
                            copy_p = consensus_vars['ess'][agent][tag][node_id][year][day]['p'][p]
                            copy_q = consensus_vars['ess'][agent][tag][node_id][year][day]['q'][p]
                            z_p_tag = z[tag][node_id][year][day]['p'][p]
                            z_q_tag = z[tag][node_id][year][day]['q'][p]
                            if abs(copy_p - z_p_tag) > 1e-12 or abs(copy_q - z_q_tag) > 1e-12:
                                problems.append(f'Z3: {agent}/{tag} copy != z at {node_id}/{year}/{day}/{p}')

                    tso_network = planning.transmission_network.network[year][day]
                    s_base = tso_network.baseMVA
                    idx = tso_network.get_shared_energy_storage_idx(node_id)
                    prox_p = _pv(tso_model[year][day].prox_ess_p_prev[idx, p])
                    prox_q = _pv(tso_model[year][day].prox_ess_q_prev[idx, p])
                    if abs(prox_p - z_cur_p / s_base) > 1e-9 or abs(prox_q - z_cur_q / s_base) > 1e-9:
                        problems.append(f'Z3: TSO prox centre != z/s_base at {node_id}/{year}/{day}/{p}')

        esso_m = esso_model[node_id]
        active_cohort_year = lp_result[node_id]['active_cohort_year']
        y_inv_idx = years.index(active_cohort_year) if active_cohort_year is not None else None
        for y in esso_m.years:
            year = years[y]
            if y_inv_idx is not None:
                soh_val = _pv(esso_m.es_soh_per_unit_cumul[y_inv_idx, y])
                soh_lp = lp_result[node_id]['soh_per_year'][year]
                if not esso_m.es_soh_per_unit_cumul[y_inv_idx, y].fixed and abs(soh_val - soh_lp) > 1e-9:
                    problems.append(f'Z3: SoH mismatch node={node_id} y_inv={y_inv_idx} y={y}')
                e_avail_val = _pv(esso_m.es_e_available_per_unit[y_inv_idx, y])
                e_avail_lp = lp_result[node_id]['e_available_per_year'][year]
                if abs(e_avail_val - e_avail_lp) > 1e-9:
                    problems.append(f'Z3: e_available mismatch node={node_id} y_inv={y_inv_idx} y={y}')
            for d in esso_m.days:
                day = days[d]
                p_lp_day = lp_result[node_id]['p'][year][day]
                for p in esso_m.periods:
                    pnet_val = _pv(esso_m.es_pnet[y, d, p])
                    if abs(pnet_val - float(p_lp_day[p])) > 1e-9:
                        problems.append(f'Z3: es_pnet mismatch node={node_id} y={y} d={d} p={p}')

    for agent in ('tso', 'dso', 'esso'):
        for node_id in node_ids:
            for year in years:
                for day in days:
                    for power_type in ('p', 'q'):
                        vals = dual_vars['ess'][agent]['current'][node_id][year][day][power_type]
                        if any(v != 0.0 for v in vals):
                            problems.append(f'Z3: nonzero ESS dual {agent}/{node_id}/{year}/{day}/{power_type}')

    lp_call_count = sept.get_lp_call_count()
    guard_failures = guard.verify(DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION)
    lp_ok = (lp_call_count == DECLARED_LP_CALLS_Z3)

    report = {
        'passed': (len(problems) == 0) and not guard_failures and lp_ok,
        'n_cells_checked': cells_checked,
        'problems': problems,
        'clipped_q_cells_reported_by_wrapper': clipped_q_cells_reported,
        'clipped_q_cells_recomputed': clipped_q_recount,
        'lp_call_count': lp_call_count,
        'lp_call_count_declared': DECLARED_LP_CALLS_Z3,
        'lp_call_count_matches_declared': lp_ok,
        'ipopt_solve_count': guard.counts['permitted_solve'],
        'ipopt_solve_count_declared': DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION,
        'guard_counts': dict(guard.counts),
        'guard_verify_failures': guard_failures,
        'shared_ess_initialization_mode': planning.params.admm.shared_ess_initialization,
        'shared_ess_initialization_source': planning.params.admm.shared_ess_initialization_source,
    }
    return report, capture, planning, sed


# ======================================================================================================================
#  Z4 -- standalone construction (flag forced off on a deep-copied params object):
#  non-ESS pre-loop state bit-identical to Z3; ESS state differs from the LP
# ======================================================================================================================
def _run_z4(G, srp, N, sept, z3_capture, z3_planning, z3_sed):
    out_dir = os.path.join(OUT_DIR, 'z4_construction')
    sept.reset_lp_call_count()
    planning, sed, candidate, capture, guard = _run_precycle1_capture(
        G, srp, N, 'z4', out_dir, Z4_EVAL_ID, force_standalone=True, capture_lp=False)

    problems = []
    consensus_vars = capture['consensus_vars']
    dual_vars = capture['dual_vars']

    z3_consensus = z3_capture['consensus_vars']
    z3_dual = z3_capture['dual_vars']
    z3_lp_result = z3_capture['lp_result']

    vmag_diff, vmag_path = _max_abs_diff_nested(consensus_vars['vmag'], z3_consensus['vmag'])
    pf_diff, pf_path = _max_abs_diff_nested(consensus_vars['pf'], z3_consensus['pf'])
    dual_vmag_diff, _ = _max_abs_diff_nested(dual_vars['vmag'], z3_dual['vmag'])
    dual_pf_diff, _ = _max_abs_diff_nested(dual_vars['pf'], z3_dual['pf'])

    non_ess_identical = (vmag_diff <= NON_ESS_TOLERANCE and pf_diff <= NON_ESS_TOLERANCE
                          and dual_vmag_diff <= NON_ESS_TOLERANCE and dual_pf_diff <= NON_ESS_TOLERANCE)
    if not non_ess_identical:
        problems.append(
            f'Z4: non-ESS pre-loop state NOT identical to Z3 (vmag_diff={vmag_diff} @ {vmag_path}, '
            f'pf_diff={pf_diff} @ {pf_path}, dual_vmag_diff={dual_vmag_diff}, dual_pf_diff={dual_pf_diff})')

    years = list(sed.years)
    days = list(sed.days)
    n_periods = planning.num_instants
    node_ids = list(sed.active_distribution_network_nodes)
    z4 = consensus_vars['ess']['z']

    cells_total = 0
    cells_matching_lp = 0
    for node_id in node_ids:
        for year in years:
            for day in days:
                lp_p = z3_lp_result[node_id]['p'][year][day]
                for p in range(n_periods):
                    cells_total += 1
                    z4_p = z4['current'][node_id][year][day]['p'][p]
                    if abs(z4_p - float(lp_p[p])) <= 1e-9:
                        cells_matching_lp += 1

    fraction_matching_lp = (cells_matching_lp / cells_total) if cells_total else None
    ess_differs_as_expected = (fraction_matching_lp is not None and fraction_matching_lp < ESS_DIFFERS_FRACTION_MAX)
    if not ess_differs_as_expected:
        problems.append(
            f'Z4: standalone z unexpectedly matches the LP schedule in {cells_matching_lp}/{cells_total} '
            f'cells (fraction={fraction_matching_lp}) -- expected the price-taker injection to be ABSENT '
            f'under the standalone flag')

    for agent in ('tso', 'dso', 'esso'):
        for node_id in node_ids:
            for year in years:
                for day in days:
                    for power_type in ('p', 'q'):
                        vals = dual_vars['ess'][agent]['current'][node_id][year][day][power_type]
                        if any(v != 0.0 for v in vals):
                            problems.append(f'Z4: nonzero ESS dual {agent}/{node_id}/{year}/{day}/{power_type}')

    price_taker_wrapper_called = ('clipped_q_cells' in capture)
    if price_taker_wrapper_called:
        problems.append('Z4: _initialize_shared_ess_from_price_taker was called under the standalone flag')

    lp_calls_during_z4 = sept.get_lp_call_count()
    lp_ok = (lp_calls_during_z4 == DECLARED_LP_CALLS_Z4)
    guard_failures = guard.verify(DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION)

    report = {
        'passed': (len(problems) == 0) and not guard_failures and lp_ok,
        'problems': problems,
        'non_ess_pre_loop_state_identical_to_z3': non_ess_identical,
        'vmag_max_abs_diff_vs_z3': vmag_diff, 'vmag_worst_path': vmag_path,
        'pf_max_abs_diff_vs_z3': pf_diff, 'pf_worst_path': pf_path,
        'dual_vmag_max_abs_diff_vs_z3': dual_vmag_diff, 'dual_pf_max_abs_diff_vs_z3': dual_pf_diff,
        'tolerance': NON_ESS_TOLERANCE,
        'ess_z_cells_total': cells_total, 'ess_z_cells_matching_lp_schedule': cells_matching_lp,
        'ess_z_fraction_matching_lp_schedule': fraction_matching_lp,
        'ess_z_fraction_threshold': ESS_DIFFERS_FRACTION_MAX,
        'ess_differs_from_lp_as_expected': ess_differs_as_expected,
        'price_taker_wrapper_called_under_standalone_flag': price_taker_wrapper_called,
        'lp_calls_during_z4_construction': lp_calls_during_z4,
        'lp_calls_declared': DECLARED_LP_CALLS_Z4,
        'lp_call_count_matches_declared': lp_ok,
        'ipopt_solve_count': guard.counts['permitted_solve'],
        'ipopt_solve_count_declared': DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION,
        'guard_counts': dict(guard.counts),
        'guard_verify_failures': guard_failures,
        'shared_ess_initialization_mode': planning.params.admm.shared_ess_initialization,
        'shared_ess_initialization_source': planning.params.admm.shared_ess_initialization_source,
    }
    return report, capture


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    import p56a_oracle as O
    import shared_resources_planning as srp
    import shared_ess_price_taker as sept
    import p514_n_instrumented_cstar as N
    import p515_g_g1_g4_admm_gates as G

    report = {
        'stage': 'P5.15-S35PT-PHASE2',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 16 items 2 and 3, PHASE 2',
            'data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json',
        ],
        'precycle1_capture_method': (
            'A monkeypatched stop on the first cycle-1 DSO solve '
            '(shared_resources_planning.update_distribution_coordination_models_and_solve) '
            'that raises _PreCycle1Stop BEFORE calling through (zero solves, zero ADMM '
            'cycles), combined with call-through capture wrappers on the four ADMM-model '
            'constructors (create_admm_variables, create_distribution_networks_models, '
            'create_transmission_network_model, create_shared_energy_storage_model) that '
            'store references to their OWN return values -- the SAME mutable objects '
            '_run_operational_planning continues to mutate in place (never reconstructed) '
            'up to the loop, so state is read out only once execution reaches EXACTLY the '
            'pre-cycle-1 point, catching any later pre-loop step that overwrites it.'
        ),
    }

    z3_report, z3_capture, z3_planning, z3_sed = _run_z3(G, srp, N, sept)
    z4_report, z4_capture = _run_z4(G, srp, N, sept, z3_capture, z3_planning, z3_sed)

    z7_ipopt_report = {
        'z3_ipopt_solve_count': z3_report['ipopt_solve_count'],
        'z4_ipopt_solve_count': z4_report['ipopt_solve_count'],
        'equal': (z3_report['ipopt_solve_count'] == z4_report['ipopt_solve_count']),
        'both_match_declared': (
            z3_report['ipopt_solve_count'] == DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION
            and z4_report['ipopt_solve_count'] == DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION
        ),
        'declared': DECLARED_IPOPT_SOLVES_PER_CONSTRUCTION,
    }
    z7_ipopt_report['passed'] = bool(z7_ipopt_report['equal'] and z7_ipopt_report['both_match_declared'])

    report['Z3'] = z3_report
    report['Z4'] = z4_report
    report['Z7_ipopt_part'] = z7_ipopt_report

    all_passed = bool(z3_report['passed'] and z4_report['passed'] and z7_ipopt_report['passed'])
    report['all_passed'] = all_passed
    report['summary'] = {
        'Z3_passed': z3_report['passed'],
        'Z4_passed': z4_report['passed'],
        'Z7_ipopt_part_passed': z7_ipopt_report['passed'],
        'ALL_PASSED': all_passed,
    }

    # OUT_DIR may already exist at this point as a side effect of
    # `_construct_arm_planning`'s results_dir redirection (creates
    # OUT_DIR/{z3,z4}_construction/results under it) -- the freshness
    # refusal above already ran BEFORE anything was constructed or solved,
    # which is the check that matters.
    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, 'phase2_checks_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    manifest = {}
    for fname in ('phase2_checks_results.json',):
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'rb') as handle:
            manifest[fname] = hashlib.sha256(handle.read()).hexdigest()
    for src_name in ('p515_s35pt_phase2_checks.py', 'shared_ess_price_taker.py',
                      'shared_resources_planning.py', 'p515_g_g1_g4_admm_gates.py'):
        src_path = os.path.join(REPO, src_name)
        with open(src_path, 'rb') as handle:
            manifest[src_name] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, 'sha256_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(json.dumps(report['summary'], indent=1))
    print(f'wrote: {results_path}')
    print(f'wrote: {manifest_path}')
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
