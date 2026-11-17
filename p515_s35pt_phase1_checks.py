"""
P5.15 Addendum 16 items 2-3 -- PHASE 1 zero-solve checks (frozen spec v6,
`data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json`).

Runs ONLY the checks from spec v6 `zero_solve_checks` that need no
Pyomo/IPOPT solve: Z1, Z2p, Z5, Z6, the LP half of Z7, an injection unit
test of `_initialize_shared_ess_from_price_taker`, and a flag-absent
identity check at the ADMM-parameter level. Z3, Z4 (need the standalone
initialization solve) and the IPOPT-count half of Z7 are explicitly
DEFERRED to PHASE 2 (Worker task, "Explicitly deferred to PHASE 2").

`SolveProfileGuard` is armed with an EMPTY permitted list for the whole
run (`guard.verify(0)` checked before anything is written) -- ZERO
Pyomo/IPOPT solves anywhere. The `shared_ess_price_taker` module's own
`scipy.optimize.linprog` call counter (`_LP_CALL_COUNTER`, via
`get_lp_call_count`/`reset_lp_call_count`) is read directly to check Z7's
LP-count claim exactly (RULE SIX: declare and count).

This script does NOT touch `data/SRP1/SRP1_params.json` (never writes to
it; the reference run `s35ref` reads it in a live process it must not
observe any change in) and does NOT import or run
`p515_g_g1_g4_admm_gates.py`. It refuses to run if its own output
directory already exists.

Usage:
    python p515_s35pt_phase1_checks.py
"""

import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35', 'pt_phase1_checks')
EFC_BENCHMARK_JSON = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34', 'EFC_benchmark', 'efc_benchmark_results.json')
Z2_JSON = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35', 'Z2', 'z2_floor_slackness_results.json')

S_INV, E_INV, INVEST_YEAR = 0.96875, 3.875, 2025   # p514_n_instrumented_cstar.py:42, same C* instance as S34/Z2
NODES = [5, 7, 9]
YEARS = [2025, 2030, 2035]
Z2P_TOL = 5e-4     # LP tolerance for Z2p (matches Z2's own committed decimal precision, 1e-4 level)


# ======================================================================================================================
#  Z1 / Z2p / Z5 -- against the real C* baseline
# ======================================================================================================================
def _build_cstar_candidate(planning, srp):
    candidate = planning.get_initial_candidate_solution()
    for node_id in planning.active_distribution_network_nodes:
        candidate['investment'][node_id][INVEST_YEAR]['s'] = S_INV
        candidate['investment'][node_id][INVEST_YEAR]['e'] = E_INV
    srp._rebuild_candidate_total_capacities(planning, candidate)
    return candidate


def _check_z1(sept, planning, candidate):
    with open(EFC_BENCHMARK_JSON) as handle:
        committed = json.load(handle)

    sept.reset_lp_call_count()
    res_v1 = sept.solve_price_taker_schedule(planning, candidate['investment'], efficiency_on=True, wear_on=False)
    res_v3 = sept.solve_price_taker_schedule(planning, candidate['investment'], efficiency_on=False, wear_on=False)
    lp_calls_z1 = sept.get_lp_call_count()

    per_cell = {}
    max_diff_v1 = 0.0
    max_diff_v3 = 0.0
    for node in NODES:
        for year in YEARS:
            committed_cell = committed['results'][str(node)][str(year)]
            e_nameplate = res_v1[node]['e_nameplate_per_year'][year]
            for day in committed_cell:
                our_v1 = float((res_v1[node]['pch'][year][day] + res_v1[node]['pdch'][year][day]).sum()
                                / (2.0 * e_nameplate))
                our_v3 = float((res_v3[node]['pch'][year][day] + res_v3[node]['pdch'][year][day]).sum()
                                / (2.0 * e_nameplate))
                diff_v1 = abs(our_v1 - committed_cell[day]['efc_star_energy'])
                diff_v3 = abs(our_v3 - committed_cell[day]['efc_star_efficiency1'])
                max_diff_v1 = max(max_diff_v1, diff_v1)
                max_diff_v3 = max(max_diff_v3, diff_v3)
                per_cell[f'{node}/{year}/{day}'] = {
                    'our_v1_efficiency_on': our_v1, 'committed_v1_efc_star_energy': committed_cell[day]['efc_star_energy'],
                    'diff_v1': diff_v1,
                    'our_v3_efficiency_off': our_v3, 'committed_v3_efc_star_efficiency1': committed_cell[day]['efc_star_efficiency1'],
                    'diff_v3': diff_v3,
                }
    n_cells = len(NODES) * len(YEARS) * 4
    passed = (max_diff_v1 < 1e-8) and (max_diff_v3 < 1e-8)
    return {
        'passed': bool(passed), 'n_cells_checked': n_cells,
        'max_abs_diff_variant1_efficiency_on_wear_off': max_diff_v1,
        'max_abs_diff_variant3_efficiency_off_wear_off': max_diff_v3,
        'lp_calls': lp_calls_z1, 'per_cell': per_cell,
    }, res_v1, res_v3, lp_calls_z1


def _check_z2p(sept, planning, candidate):
    with open(Z2_JSON) as handle:
        committed = json.load(handle)
    # Read the committed per-(node,year) EFC/day and per-node terminal SoH
    # directly from the Z2 artifact (not retyped literals), node 5's chain
    # used as the reference (Z2 confirmed all three nodes identical).
    committed_efc_by_year = {
        year: committed['part_A_unconstrained_price_taker']['5'][str(year)]['efc_per_day']
        for year in YEARS
    }
    committed_terminal_soh_by_node = committed['part_A_summary']['terminal_soh_2035_by_node']
    committed_floor_active_by_node = committed['part_A_summary']['floor_active_by_node']

    sept.reset_lp_call_count()
    res_full = sept.solve_price_taker_schedule(planning, candidate['investment'], efficiency_on=True, wear_on=True)
    lp_calls_z2p = sept.get_lp_call_count()

    per_node = {}
    max_efc_diff = 0.0
    max_soh_diff = 0.0
    all_floor_slack = True
    all_multiplier_zero = True
    for node in NODES:
        r = res_full[node]
        node_report = {'efc_per_day_harness': r['efc_per_day_harness'], 'soh_per_year': r['soh_per_year'],
                        'floor_multiplier_per_year': r['floor_multiplier_per_year'],
                        'converged': r['converged'], 'final_rel_change': r['final_rel_change'],
                        'outer_iterations_run': r['outer_iterations_run']}
        for year in YEARS:
            diff = abs(r['efc_per_day_harness'][year] - committed_efc_by_year[year])
            max_efc_diff = max(max_efc_diff, diff)
        terminal_soh = r['soh_per_year'][2035]
        committed_terminal_soh_value = committed_terminal_soh_by_node[str(node)]
        soh_diff = abs(terminal_soh - committed_terminal_soh_value)
        max_soh_diff = max(max_soh_diff, soh_diff)
        if terminal_soh <= 0.50 or committed_floor_active_by_node[str(node)]:
            all_floor_slack = False
        for year in YEARS:
            if abs(r['floor_multiplier_per_year'][year]) > 1e-9:
                all_multiplier_zero = False
        node_report['terminal_soh_2035'] = terminal_soh
        node_report['committed_terminal_soh_2035'] = committed_terminal_soh_value
        node_report['terminal_soh_diff_vs_committed'] = soh_diff
        per_node[str(node)] = node_report

    all_converged = all(res_full[node]['converged'] for node in NODES)
    passed = (max_efc_diff < Z2P_TOL and max_soh_diff < Z2P_TOL
              and all_floor_slack and all_multiplier_zero and all_converged)

    return {
        'passed': bool(passed), 'lp_calls': lp_calls_z2p,
        'max_abs_efc_diff_vs_committed_Z2': max_efc_diff,
        'max_abs_terminal_soh_diff_vs_committed_Z2': max_soh_diff,
        'floor_slack_everywhere': all_floor_slack, 'floor_multiplier_zero_everywhere': all_multiplier_zero,
        'all_converged': all_converged, 'per_node': per_node,
        'committed_reference': {'efc_per_day_by_year': committed_efc_by_year,
                                 'terminal_soh_by_node': committed_terminal_soh_by_node},
    }, res_full, lp_calls_z2p


def _check_z5(res_full, sed_cost_energy_p, days_list):
    from definitions import ENERGY_STORAGE_MIN_ENERGY_STORED, ENERGY_STORAGE_MAX_ENERGY_STORED
    violations = []
    n_cells_checked = 0
    for node, node_result in res_full.items():
        for year in YEARS:
            e_avail = node_result['e_available_per_year'][year]
            s_avail = node_result['s_nameplate_per_year'][year]
            soc_min = e_avail * ENERGY_STORAGE_MIN_ENERGY_STORED - 1e-6
            soc_max = e_avail * ENERGY_STORAGE_MAX_ENERGY_STORED + 1e-6
            for day in days_list:
                n_cells_checked += 1
                prices = np.asarray(sed_cost_energy_p[year][day][0], dtype=float)
                day_mean = float(prices.mean())
                pch = node_result['pch'][year][day]
                pdch = node_result['pdch'][year][day]
                soc = node_result['soc'][year][day]
                p = node_result['p'][year][day]

                if np.any(np.abs(p) > s_avail + 1e-6):
                    violations.append(f'{node}/{year}/{day}: |p| exceeds s_avail={s_avail}')
                if np.any(soc < soc_min) or np.any(soc > soc_max):
                    violations.append(f'{node}/{year}/{day}: soc outside [{soc_min},{soc_max}]')

                charge_mask = pch > 1e-9
                dischg_mask = pdch > 1e-9
                if charge_mask.any():
                    avg_charge_price = float(np.average(prices[charge_mask], weights=pch[charge_mask]))
                    if avg_charge_price > day_mean + 1e-6:
                        violations.append(
                            f'{node}/{year}/{day}: charge-weighted price {avg_charge_price:.4f} '
                            f'> day mean {day_mean:.4f}')
                if dischg_mask.any():
                    avg_dischg_price = float(np.average(prices[dischg_mask], weights=pdch[dischg_mask]))
                    if avg_dischg_price < day_mean - 1e-6:
                        violations.append(
                            f'{node}/{year}/{day}: discharge-weighted price {avg_dischg_price:.4f} '
                            f'< day mean {day_mean:.4f}')
    return {'passed': len(violations) == 0, 'n_cells_checked': n_cells_checked, 'violations': violations}


# ======================================================================================================================
#  Z6 -- synthetic-input guard checks
# ======================================================================================================================
class _StubSharedEnergyStorage:
    def __init__(self, t_cal=15.0, eff_ch=0.97, eff_dch=0.96, cl_eff=11541.560327111707, phi_cal=1.0, soh_min=0.50):
        self.t_cal = t_cal
        self.eff_ch = eff_ch
        self.eff_dch = eff_dch
        self.cl_eff = cl_eff
        self.phi_cal = phi_cal
        self.soh_min = soh_min


class _StubNetwork:
    """Per-(year,day) network stand-in exposing only what
    `shared_ess_price_taker`'s guards and LP rows read."""

    def __init__(self, node_ids, prices_row, prob_market=1, prob_operation=1):
        self.prob_market_scenarios = [1.0 / prob_market] * prob_market
        self.prob_operation_scenarios = [1.0 / prob_operation] * prob_operation
        self.cost_energy_p = prices_row
        self.baseMVA = 1.0
        self._node_ids = list(node_ids)
        self.shared_energy_storages = [SimpleNamespace(bus=n, s=1.0) for n in node_ids]

    def get_shared_energy_storage_idx(self, node_id):
        return self._node_ids.index(node_id)


class _StubNetworkContainer:
    def __init__(self, years, days, node_ids, prices_by_year_day, prob_market=1, prob_operation=1):
        self.network = {
            year: {day: _StubNetwork(node_ids, prices_by_year_day[year][day], prob_market, prob_operation)
                   for day in days}
            for year in years
        }


class _StubSharedEssData:
    def __init__(self, node_ids, years, days, n_periods, prices_by_year_day, t_cal=15.0):
        self.years = {y: 5 for y in years}
        self.days = {d: 91 for d in days}
        self.num_instants = n_periods
        self.active_distribution_network_nodes = list(node_ids)
        self.cost_energy_p = prices_by_year_day
        self.shared_energy_storages = {y: [_StubSharedEnergyStorage(t_cal=t_cal) for _ in node_ids] for y in years}

    def get_shared_energy_storage_idx(self, node_id):
        return list(self.active_distribution_network_nodes).index(node_id)


class _StubPlanning:
    def __init__(self, node_ids, years, days, n_periods, prices_by_year_day,
                 t_cal=15.0, tso_prob_market=1, tso_prob_operation=1,
                 perturbed_dso_prices=None):
        self.shared_ess_data = _StubSharedEssData(node_ids, years, days, n_periods, prices_by_year_day, t_cal=t_cal)
        self.transmission_network = _StubNetworkContainer(
            years, days, node_ids, prices_by_year_day, tso_prob_market, tso_prob_operation)
        self.distribution_networks = {}
        for i, node_id in enumerate(node_ids):
            dso_prices = prices_by_year_day
            if perturbed_dso_prices is not None and node_id == perturbed_dso_prices[0]:
                dso_prices = perturbed_dso_prices[1]
            self.distribution_networks[node_id] = _StubNetworkContainer(years, days, [node_id], dso_prices, 1, 1)


def _make_prices(years, days, n_periods, seed=0):
    rng = np.random.default_rng(seed)
    return {year: {day: rng.uniform(20.0, 80.0, size=(1, n_periods)) for day in days} for year in years}


def _run_z6(sept):
    years = [2025, 2030, 2035]
    days = ['Spring']
    n_periods = 4
    node_ids = [5]
    base_prices = _make_prices(years, days, n_periods, seed=1)
    base_investment = {5: {year: {'s': 0.0, 'e': 0.0} for year in years}}
    base_investment[5][2025] = {'s': 1.0, 'e': 2.0}

    results = {}

    # (a) more than one market/operation scenario
    stub = _StubPlanning(node_ids, years, days, n_periods, base_prices, t_cal=15.0,
                          tso_prob_market=2, tso_prob_operation=1)
    results['scenario_guard'] = _expect_not_implemented(sept, stub, base_investment)

    # (b) more than one active cohort
    stub = _StubPlanning(node_ids, years, days, n_periods, base_prices, t_cal=15.0)
    two_cohort_investment = {5: {year: dict(base_investment[5][year]) for year in years}}
    two_cohort_investment[5][2030] = {'s': 1.0, 'e': 2.0}
    results['cohort_count_guard'] = _expect_not_implemented(sept, stub, two_cohort_investment)

    # (c) prices differ across networks
    perturbed = deepcopy(base_prices)
    perturbed[2025]['Spring'] = perturbed[2025]['Spring'] + 1.0
    stub = _StubPlanning(node_ids, years, days, n_periods, base_prices, t_cal=15.0,
                          perturbed_dso_prices=(5, perturbed))
    results['prices_differ_guard'] = _expect_not_implemented(sept, stub, base_investment)

    # (d) cohort window does not cover all blocks (t_cal too short)
    stub = _StubPlanning(node_ids, years, days, n_periods, base_prices, t_cal=5.0)
    results['cohort_window_guard'] = _expect_not_implemented(sept, stub, base_investment)

    return results


def _expect_not_implemented(sept, stub_planning, investment):
    calls_before = sept.get_lp_call_count()
    try:
        sept.solve_price_taker_schedule(stub_planning, investment)
        return {'passed': False, 'raised': False, 'lp_calls_during_call': sept.get_lp_call_count() - calls_before}
    except NotImplementedError as exc:
        return {'passed': True, 'raised': True, 'message': str(exc),
                'lp_calls_during_call': sept.get_lp_call_count() - calls_before}
    except Exception as exc:  # pragma: no cover - report, do not hide
        return {'passed': False, 'raised': True, 'wrong_exception_type': type(exc).__name__, 'message': str(exc)}


# ======================================================================================================================
#  Injection unit test (synthetic route -- see WORKER_REPORT_S35PT_PHASE1.md
#  for why the real-unsolved-model route is not available)
# ======================================================================================================================
class _FakeParamEntry:
    def __init__(self):
        self.value = 0.0

    def set_value(self, v):
        self.value = v


class _FakeIndexedParam(dict):
    def __missing__(self, key):
        entry = _FakeParamEntry()
        self[key] = entry
        return entry


class _FakeVarEntry:
    def __init__(self, fixed=False):
        self.value = 0.0
        self.fixed = fixed

    def set_value(self, v):
        self.value = v


class _FakeIndexedVar(dict):
    def __missing__(self, key):
        entry = _FakeVarEntry()
        self[key] = entry
        return entry


def _make_fake_tso_model(years, days):
    return {year: {day: SimpleNamespace(prox_ess_p_prev=_FakeIndexedParam(), prox_ess_q_prev=_FakeIndexedParam())
                   for day in days}
            for year in years}


def _make_fake_esso_model(n_years, n_days, n_periods):
    return SimpleNamespace(
        years=range(n_years), days=range(n_days), periods=range(n_periods),
        es_soh_per_unit_cumul=_FakeIndexedVar(), es_e_available_per_unit=_FakeIndexedVar(),
        es_pnet=_FakeIndexedVar(), es_pch_per_unit=_FakeIndexedVar(), es_pdch_per_unit=_FakeIndexedVar(),
    )


def _run_injection_unit_test(srp, planning, candidate):
    years = list(planning.years)
    days = list(planning.days)
    n_periods = planning.num_instants
    node_ids = list(planning.active_distribution_network_nodes)

    consensus_vars, dual_vars = srp.create_admm_variables(planning)
    srp._initialize_shared_ess_consensus(planning, consensus_vars)

    # Snapshot vmag/pf consensus and every dual before the wrapper call.
    vmag_before = deepcopy(consensus_vars['vmag'])
    pf_before = deepcopy(consensus_vars['pf'])
    dual_vmag_before = deepcopy(dual_vars['vmag'])
    dual_pf_before = deepcopy(dual_vars['pf'])
    dual_ess_before = deepcopy(dual_vars['ess'])

    fake_tso_model = _make_fake_tso_model(years, days)
    fake_esso_model = {node_id: _make_fake_esso_model(len(years), len(days), n_periods) for node_id in node_ids}

    clipped_q_cells = srp._initialize_shared_ess_from_price_taker(
        planning, candidate, fake_tso_model, fake_esso_model, consensus_vars)

    lp_result = shared_ess_price_taker_ref.solve_price_taker_schedule(planning, candidate['investment'])

    problems = []

    # z == LP p (current and prev); q inside converter circle
    for node_id in node_ids:
        s_avail_by_year = lp_result[node_id]['s_nameplate_per_year']
        for year in years:
            for day in days:
                lp_p = lp_result[node_id]['p'][year][day]
                for p in range(n_periods):
                    z_cur = consensus_vars['ess']['z']['current'][node_id][year][day]['p'][p]
                    z_prev = consensus_vars['ess']['z']['prev'][node_id][year][day]['p'][p]
                    if abs(z_cur - lp_p[p]) > 1e-9 or abs(z_prev - lp_p[p]) > 1e-9:
                        problems.append(f'z p mismatch at {node_id}/{year}/{day}/{p}')

                    z_q = consensus_vars['ess']['z']['current'][node_id][year][day]['q'][p]
                    s_avail = s_avail_by_year[year]
                    if (z_cur ** 2 + z_q ** 2) > s_avail ** 2 + 1e-6:
                        problems.append(f'q outside converter circle at {node_id}/{year}/{day}/{p}')

                    for agent in ('tso', 'dso', 'esso'):
                        for tag in ('current', 'prev'):
                            copy_p = consensus_vars['ess'][agent][tag][node_id][year][day]['p'][p]
                            copy_q = consensus_vars['ess'][agent][tag][node_id][year][day]['q'][p]
                            z_p_tag = consensus_vars['ess']['z'][tag][node_id][year][day]['p'][p]
                            z_q_tag = consensus_vars['ess']['z'][tag][node_id][year][day]['q'][p]
                            if abs(copy_p - z_p_tag) > 1e-12 or abs(copy_q - z_q_tag) > 1e-12:
                                problems.append(f'{agent}/{tag} copy != z at {node_id}/{year}/{day}/{p}')

                    tso_network = planning.transmission_network.network[year][day]
                    s_base = tso_network.baseMVA
                    idx = tso_network.get_shared_energy_storage_idx(node_id)
                    prox_p = fake_tso_model[year][day].prox_ess_p_prev[idx, p].value
                    prox_q = fake_tso_model[year][day].prox_ess_q_prev[idx, p].value
                    if abs(prox_p - z_cur / s_base) > 1e-9 or abs(prox_q - z_q / s_base) > 1e-9:
                        problems.append(f'proximal centre mismatch at {node_id}/{year}/{day}/{p}')

    # ESS duals untouched (exactly zero)
    for agent in ('tso', 'dso', 'esso'):
        for node_id in node_ids:
            for year in years:
                for day in days:
                    for power_type in ('p', 'q'):
                        before = dual_ess_before[agent]['current'][node_id][year][day][power_type]
                        after = dual_vars['ess'][agent]['current'][node_id][year][day][power_type]
                        if before != after or any(v != 0.0 for v in after):
                            problems.append(f'ESS dual touched or nonzero: {agent}/{node_id}/{year}/{day}/{power_type}')

    # V/PF consensus and duals bit-identical
    if consensus_vars['vmag'] != vmag_before:
        problems.append('vmag consensus changed')
    if consensus_vars['pf'] != pf_before:
        problems.append('pf consensus changed')
    if dual_vars['vmag'] != dual_vmag_before:
        problems.append('vmag duals changed')
    if dual_vars['pf'] != dual_pf_before:
        problems.append('pf duals changed')

    return {
        'passed': len(problems) == 0, 'problems': problems, 'clipped_q_cells': clipped_q_cells,
        'route': 'synthetic (fake tso_model/esso_model Params/Vars; real create_admm_variables, '
                 '_initialize_shared_ess_consensus, _initialize_shared_ess_from_price_taker, and real '
                 'planning_problem/candidate_solution). Building REAL unsolved Pyomo ADMM models is not '
                 'possible without a solve: create_shared_energy_storage_model calls '
                 'shared_ess_data.optimize(esso_model) unconditionally (shared_resources_planning.py), so '
                 'the synthetic route was used, per the task fallback instruction.',
    }


# ======================================================================================================================
#  Flag-absent identity at the parameter level
# ======================================================================================================================
def _run_flag_absent_check():
    import admm_parameters as ap
    results = {}
    for case_name, params_path in (('SRP1', os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')),
                                    ('CS1', os.path.join(REPO, 'data', 'CS1', 'CS1_params.json'))):
        with open(params_path) as handle:
            case_json = json.load(handle)
        admm_raw = case_json['admm']
        if 'shared_ess_initialization' in admm_raw:
            results[case_name] = {'passed': False, 'reason': 'case file already has shared_ess_initialization'}
            continue
        params = ap.ADMMParameters()
        params.read_parameters_from_file(admm_raw)
        ok = (params.shared_ess_initialization == 'standalone'
              and params.shared_ess_initialization_source == 'default'
              and params.num_max_iters == int(admm_raw['num_max_iters'])
              and params.rho['v'] == admm_raw['rho']['v'])
        results[case_name] = {
            'passed': bool(ok),
            'shared_ess_initialization': params.shared_ess_initialization,
            'shared_ess_initialization_source': params.shared_ess_initialization_source,
            'num_max_iters_matches': params.num_max_iters == int(admm_raw['num_max_iters']),
            'rho_v_matches': params.rho['v'] == admm_raw['rho']['v'],
        }
    overall = all(v['passed'] for v in results.values())
    return {'passed': overall, 'by_case': results}


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    guard = SolveProfileGuard([], label='P5.15-S35pt Phase 1 checks').install()
    report = {'stage': 'P5.15-S35PT-PHASE1', 'timestamp_utc': datetime.now(timezone.utc).isoformat()}
    try:
        global shared_ess_price_taker_ref
        import p56a_oracle as O  # noqa: E402  (import after guard install; no solves on import)
        import shared_resources_planning as srp  # noqa: E402
        import shared_ess_price_taker as sept  # noqa: E402
        shared_ess_price_taker_ref = sept

        baseline = O.load_baseline()
        planning = baseline['planning']
        candidate = _build_cstar_candidate(planning, srp)

        z1_report, res_v1, res_v3, lp_calls_z1 = _check_z1(sept, planning, candidate)
        z2p_report, res_full, lp_calls_z2p = _check_z2p(sept, planning, candidate)
        z5_report = _check_z5(res_full, planning.shared_ess_data.cost_energy_p, list(planning.days))
        z6_report = _run_z6(sept)
        injection_report = _run_injection_unit_test(srp, planning, candidate)
        flag_absent_report = _run_flag_absent_check()

        declared_lp_total = lp_calls_z1 + lp_calls_z2p
        z7_lp_report = {
            'declared_z1': lp_calls_z1, 'declared_z2p': lp_calls_z2p, 'declared_total': declared_lp_total,
            'note': 'counts read directly from shared_ess_price_taker.get_lp_call_count() after each '
                    'check; the module counter is reset (reset_lp_call_count) between Z1 and Z2p, so each '
                    'count is exact for its own check. Z6 synthetic-guard calls raise before any linprog '
                    'call (see per-guard lp_calls_during_call, expected 0 for all four).',
            'z6_lp_calls_during_each_guard_call': {k: v.get('lp_calls_during_call') for k, v in z6_report.items()},
            'passed': all(v.get('lp_calls_during_call') == 0 for v in z6_report.values()),
        }

    finally:
        failures = guard.verify(0)
        guard.uninstall()

    if failures:
        raise AssertionError('RULE SIX: Pyomo/IPOPT solve guard was not exactly zero -> ' + '; '.join(failures))

    report['solve_profile'] = {'observed': dict(guard.counts), 'permitted_declared': 0, 'guard_verify_failures': failures}
    report['Z1'] = z1_report
    report['Z2p'] = z2p_report
    report['Z5'] = z5_report
    report['Z6'] = z6_report
    report['Z7_lp_part'] = z7_lp_report
    report['injection_unit_test'] = injection_report
    report['flag_absent_identity'] = flag_absent_report

    all_checks = [z1_report['passed'], z2p_report['passed'], z5_report['passed'],
                  all(v['passed'] for v in z6_report.values()), z7_lp_report['passed'],
                  injection_report['passed'], flag_absent_report['passed']]
    report['all_passed'] = bool(all(all_checks))
    report['summary'] = {
        'Z1_passed': z1_report['passed'], 'Z2p_passed': z2p_report['passed'], 'Z5_passed': z5_report['passed'],
        'Z6_passed': all(v['passed'] for v in z6_report.values()),
        'Z7_lp_part_passed': z7_lp_report['passed'],
        'injection_unit_test_passed': injection_report['passed'],
        'flag_absent_identity_passed': flag_absent_report['passed'],
        'ALL_PASSED': report['all_passed'],
    }

    os.makedirs(OUT_DIR)
    results_path = os.path.join(OUT_DIR, 'phase1_checks_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    manifest = {}
    for fname in ('phase1_checks_results.json',):
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'rb') as handle:
            manifest[fname] = hashlib.sha256(handle.read()).hexdigest()
    for src_name in ('p515_s35pt_phase1_checks.py', 'shared_ess_price_taker.py'):
        src_path = os.path.join(REPO, src_name)
        with open(src_path, 'rb') as handle:
            manifest[src_name] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, 'sha256_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(json.dumps(report['summary'], indent=1))
    print(f"wrote: {results_path}")
    print(f"wrote: {manifest_path}")
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
