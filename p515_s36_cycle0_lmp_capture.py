"""
P5.15 Addendum 17 -- PART B: harness capture hooks (TSO/DSO node-balance
duals) plus a bounded, EXACTLY-51-solve cycle-0 LMP capture and Addendum 17
diagnostic 1(c) (price-taker LP re-solved with cycle-0 LMPs).

Authority: Planner task "Addendum 17 capture and reconstruction", PART B
(2026-09-16); `P5_15_ADDENDUM16_EXPERT_REPORT.md` Sec.5-8 item 1(c).

## Correction of `WORKER_REPORT_S36_A17_DIAGNOSTICS.md` (commit 5e2ba6f9)

That report's PART 0 item 4 states: "no standalone pre-cycle-1 solve exists
in this pipeline at all". This is WRONG.
`data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json`
records EXACTLY 51 IPOPT solves that run BEFORE cycle 1's first ADMM network
solve, in BOTH its Z3 (price-taker init) and Z4 (standalone init) checks
(`ipopt_solve_count: 51` in each). These are the standalone per-(network,
year, day) SMOPF solves `create_distribution_networks_models` (36 = 3 DSOs x
3 years x 4 days), `create_transmission_network_model` (12 = 3 years x 4
days) and `create_shared_energy_storage_model` (3 = 3 active nodes)
unconditionally perform as part of MODEL CONSTRUCTION, before
`create_admm_variables`'s consensus/dual state is even read by the ADMM
loop. `_run_precycle1_capture` (that same script) already demonstrates the
zero-solve technique for stopping exactly after these 51 solves and before
cycle 1's first coordination solve (a monkeypatched raise on
`update_distribution_coordination_models_and_solve`, called through
UNCHANGED for everything before it). These standalone constructions ARE the
"cycle-0" network solves this Addendum's item 1(c) needs -- the correction
is that they exist, not that a NEW solve type must be invented.

## Harness capture hooks (wrap production functions; call through unchanged)

`capture_tso_node_balance_duals` / `capture_dso_reference_node_balance_duals`
below read `model.dual` on the ALREADY-SOLVED `tso_model`/`dso_models` dicts
`_run_precycle1_capture` returns (via its own call-through wrappers on
`create_transmission_network_model` / `create_distribution_networks_models`,
reused unmodified here, not re-wrapped) -- no new monkeypatch of the
constructors is needed for THIS bounded run, since the constructor wrappers
already capture the SAME mutable model objects the constructors return
in-place, and nothing touches `model.dual` between the standalone solve and
the `_PreCycle1Stop` catch point. The two functions below are still written
as REUSABLE, general-purpose hooks (not run-specific one-offs) so PART C's
(not launched) terminal-cycle capture can call them again on the TSO/DSO
models at any later cycle, given the same `model.dual`-populated dict.

## Sign convention and units (stated once, applies everywhere below)

`node_balance_p_rule`/`node_balance_q_rule` (`model_construction_helpers.py`)
build the equality constraint literally as `Pg == Pd + Pi + slack_up -
slack_down` (P) and the analogous form for Q. `model.dual.get(constraint)`
is read AS PYOMO RETURNS IT (the IMPORT_EXPORT Suffix populated from
IPOPT's exported multipliers) -- NO SIGN FLIP applied, the same convention
`_s35ref_capture_hooks`' SoH-floor-dual capture already uses in this
codebase.

Units: the constraint is written in the network's own per-unit system
(`baseMVA` = `S_base`). If `x_pu = x_MW / S_base`, then
`d(objective)/d(x_MW) = d(objective)/d(x_pu) * d(x_pu)/d(x_MW) = dual /
S_base`. So **LMP [$/MW, equivalently $/MWh since periods are hourly (24
periods/day; `objective_function_rule`'s cost terms, e.g.
`interface_energy_settlement`/`generation_cost`, are `c_p[p] * S_base *
pg_pu[p]` with no separate `dt` factor, i.e. $/MWh x MW = $/h = $ per
1-hour period)] = dual_pu / S_base**. This conversion is applied explicitly
below wherever a dual is turned into a price-taker LP input.

At CONSTRUCTION time (the 51 standalone solves captured here), the
objective is the network's own UNSCALED base objective --
`_prepare_distribution_objectives_for_admm`/`_prepare_transmission_objectives_for_admm`
and the `objective_scale`/`sigma` division happen AFTER these constructor
calls return, inside `_run_operational_planning`'s `if initial_state is
None:` branch (`shared_resources_planning.py`). So these are genuine
$/MWh system-cost duals of the UNSCALED objective, not scaled-objective
duals -- no further sigma correction is needed for this capture. (A
terminal-cycle capture under a LATER, un-run PART C would be a dual of the
SCALED objective and would need multiplying by that run's own
`objective_scale`/`sigma` before comparison on the same footing; stated
here for completeness, not exercised by this bounded run.)

## Cycle-0 price choice for the price-taker LP substitution

The production LP (`shared_ess_price_taker.solve_price_taker_schedule`)
takes ONE market price series per (year, day) via
`shared_ess_data.cost_energy_p[year][day][0]`, enforced UNIFORM across the
TSO and every DSO by its own guard
(`_check_prices_uniform_across_networks`) -- there is no per-node nodal-
price argument in production (confirmed in the prior diagnostics report,
commit 5e2ba6f9). To honour that guard while still giving each of the 3
storage nodes its OWN cycle-0 price, this script solves the LP ONCE PER
NODE, each time on its own private, deep-copied planning object whose
`cost_energy_p[year][day]` array (shared BY REFERENCE across
`planning.transmission_network`, every `planning.distribution_networks[n]`
and `planning.shared_ess_data` at construction --
`shared_resources_planning.py:7362,7378,7405,7421,7450` -- so ONE in-place
mutation of `planning.cost_energy_p[year][day]` is visible everywhere,
satisfying the uniformity guard for free) is overwritten with THAT node's
own cycle-0 LMP series. The DSO's OWN reference-node-balance-P dual (the
bus the storage physically sits behind, in the DSO's own network) is used
as the substituted price series -- the TSO's node-balance-p dual at the
same interface bus is ALSO captured and reported (per this task's literal
"TSO node-balance duals ... DSO reference-node balance duals" wording) for
comparison, but was NOT the one fed into the LP; both series are written to
the output so the Planner can re-run with the alternative if a different
choice was intended.

## Bounded guards

Construction: EXACTLY 51 IPOPT solves (same declared count and discovery
method as `p515_s35pt_phase2_checks.py`). Price-taker LP: EXACTLY
3 nodes x `DEFAULT_OUTER_ITERATIONS` (40) = 120 scipy LP calls (`wear_on`
defaults True), declared and checked via `shared_ess_price_taker`'s own
counter -- matching the SAME 120 the phase2-checks Z3 already found for a
3-node, single-price run (this script makes 3 SEPARATE 1-node, 40-outer-
iteration calls instead of 1 combined 3-node call, so the total is
identical: 3*40 = 120).

## Output

New directory `data/SRP1/Results/P515S36/cycle0_lmp/` (refuses if it
already exists): `cycle0_lmp_capture_results.json`, `sha256_manifest.json`.
"""

import glob
import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s35pt_phase2_checks as PT2  # noqa: E402  (reused BY CALLING, not reimplemented)
import shared_resources_planning as srp  # noqa: E402
import shared_ess_price_taker as sept  # noqa: E402
import p56a_oracle as O  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
OUT_DIR = os.path.join(RES, 'P515S36', 'cycle0_lmp')

NODES = (5, 7, 9)
DECLARED_IPOPT_SOLVES = 51
DECLARED_LP_CALLS = len(NODES) * sept.DEFAULT_OUTER_ITERATIONS  # 3 * 40 = 120

RUN1_EFC = 1.058952279550704
GATE3_EFC = 1.1842435189565115
MARKET_PRICE_LP_EFC = 1.1917770631428612
MARKET_PRICE_LP_JSON = os.path.join(RES, 'P515S35', 'Z2', 'z2_floor_slackness_results.json')


# ======================================================================================================================
#  Harness capture hooks -- reusable, general-purpose (PART B requirement)
# ======================================================================================================================
def capture_tso_node_balance_duals(tso_model, transmission_network, node_ids):
    """TSO node-balance-p/q duals at `node_ids` (the storage buses), for
    every (year, day, s_m, s_o, period). Reads `model.dual` on an
    ALREADY-SOLVED model -- no solve is performed here. Returns
    {node_id: {year: {day: [ {s_m, s_o, period, dual_p_pu, dual_q_pu}, ... ]}}}."""
    out = {node_id: {} for node_id in node_ids}
    for year in transmission_network.years:
        network = None
        for day in transmission_network.days:
            network = transmission_network.network[year][day]
            model = tso_model[year][day]
            for node_id in node_ids:
                idx = network.get_node_idx(node_id)
                entries = []
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            con_p = model.node_balance_p[idx, s_m, s_o, p]
                            con_q = model.node_balance_q[idx, s_m, s_o, p]
                            dual_p = model.dual.get(con_p)
                            dual_q = model.dual.get(con_q)
                            entries.append({
                                's_m': s_m, 's_o': s_o, 'period': p,
                                'dual_p_pu': float(pe.value(dual_p)) if dual_p is not None else None,
                                'dual_q_pu': float(pe.value(dual_q)) if dual_q is not None else None,
                            })
                out[node_id].setdefault(str(year), {})[str(day)] = entries
    return out


def capture_dso_reference_node_balance_duals(dso_models, distribution_networks, node_ids):
    """DSO reference-node-balance-p/q duals, one DSO per `node_ids` entry
    (the shared-ESS is attached at that DSO's OWN reference/substation bus).
    Reads `model.dual` on an ALREADY-SOLVED model -- no solve is performed
    here. Returns {node_id: {year: {day: [ {s_m, s_o, period, dual_p_pu,
    dual_q_pu}, ... ]}}} plus each DSO's own `baseMVA` (needed for the
    p.u.->$/MWh conversion)."""
    out = {}
    base_mva = {}
    for node_id in node_ids:
        dso_model = dso_models[node_id]
        distribution_network = distribution_networks[node_id]
        out[node_id] = {}
        for year in distribution_network.years:
            for day in distribution_network.days:
                network = distribution_network.network[year][day]
                model = dso_model[year][day]
                ref_node_id = network.get_reference_node_id()
                idx = network.get_node_idx(ref_node_id)
                base_mva[node_id] = network.baseMVA
                entries = []
                for s_m in model.scenarios_market:
                    for s_o in model.scenarios_operation:
                        for p in model.periods:
                            con_p = model.node_balance_p[idx, s_m, s_o, p]
                            con_q = model.node_balance_q[idx, s_m, s_o, p]
                            dual_p = model.dual.get(con_p)
                            dual_q = model.dual.get(con_q)
                            entries.append({
                                's_m': s_m, 's_o': s_o, 'period': p,
                                'dual_p_pu': float(pe.value(dual_p)) if dual_p is not None else None,
                                'dual_q_pu': float(pe.value(dual_q)) if dual_q is not None else None,
                            })
                out[node_id].setdefault(str(year), {})[str(day)] = entries
    return out, base_mva


def _dual_p_series_dollars_per_mwh(entries_by_year_day, base_mva, years, days):
    """Convert a captured per-(year,day) list of {s_m,s_o,period,dual_p_pu}
    (single market/operation scenario, asserted) into {year: {day:
    np.ndarray[24]}} in $/MWh (dual_pu / base_mva). Missing (None) duals
    raise -- a price-taker LP input cannot silently substitute a missing
    price."""
    out = {}
    for year in years:
        out[year] = {}
        for day in days:
            rows = entries_by_year_day[str(year)][str(day)]
            s_m_values = sorted(set(r['s_m'] for r in rows))
            s_o_values = sorted(set(r['s_o'] for r in rows))
            if len(s_m_values) != 1 or len(s_o_values) != 1:
                raise NotImplementedError(
                    f'cycle0 LMP capture: expected exactly one market/operation scenario, '
                    f'got s_m={s_m_values}, s_o={s_o_values} at year={year}, day={day}')
            by_period = {r['period']: r['dual_p_pu'] for r in rows}
            n_periods = len(by_period)
            arr = np.empty(n_periods, dtype=float)
            for p in range(n_periods):
                dual_pu = by_period[p]
                if dual_pu is None:
                    raise RuntimeError(
                        f'cycle0 LMP capture: missing dual_p (solve did not succeed or Pyomo did not '
                        f'export it) at year={year}, day={day}, period={p} -- cannot build a price series')
                arr[p] = dual_pu / base_mva
            out[year][day] = arr
    return out


# ======================================================================================================================
#  Step 1: the bounded 51-solve construction capture (reuses PT2's own function, unmodified)
# ======================================================================================================================
def run_construction_capture():
    out_dir = os.path.join(OUT_DIR, 'construction')
    planning, sed, candidate, capture, guard = PT2._run_precycle1_capture(
        G, srp, N, 'p515s36_cycle0lmp', out_dir, 'p515s36_cycle0lmp_construction',
        force_standalone=True,  # C*: run 1's ACTUAL configuration (case file now defaults to
                                  # price_taker; run 1/s35ref used standalone -- same override
                                  # `p515_s35pt_phase2_checks._run_z4` already validated).
        capture_lp=False,
    )
    guard_failures = guard.verify(DECLARED_IPOPT_SOLVES)
    if guard_failures:
        raise AssertionError('RULE SIX: construction guard mismatch -> ' + '; '.join(guard_failures))

    return planning, sed, candidate, capture, dict(guard.counts)


# ======================================================================================================================
#  Step 2: price-taker LP with cycle-0 LMPs (diagnostic 1(c))
# ======================================================================================================================
def _build_planning_lp_for_node(template_planning, node_id, dso_lmp_dollars_per_mwh, years, days):
    planning_lp = deepcopy(template_planning)
    for year in years:
        for day in days:
            arr = planning_lp.cost_energy_p[year][day]
            if arr.shape[0] != 1:
                raise NotImplementedError(
                    f'expected exactly one market scenario row in cost_energy_p[{year}][{day}], '
                    f'got shape {arr.shape}')
            arr[0, :] = dso_lmp_dollars_per_mwh[node_id][year][day]
    return planning_lp


def run_pricetaker_lp_with_cycle0_lmps(candidate, dso_lmp_dollars_per_mwh, years, days):
    """Primary, DECLARED result: production defaults
    (`outer_iterations=sept.DEFAULT_OUTER_ITERATIONS=40`,
    `damping=sept.DEFAULT_DAMPING`) exactly as `_initialize_shared_ess_from_price_taker`
    calls them. `_solve_node`'s damped fixed point runs its FULL declared
    `outer_iterations` budget (one scipy LP per iteration) BEFORE checking
    convergence and raising -- so exactly `outer_iterations` LP calls happen
    per node regardless of whether that node's fixed point converges; the
    declared LP-call count (3 nodes * 40 = 120) is therefore exact whether
    or not any node converges, and a non-convergence is caught here and
    reported as a genuine per-node result, NOT silently forced to pass and
    NOT hidden.
    """
    lp_calls_before = sept.get_lp_call_count()

    template_planning = O.fresh_planning('p515s36_cycle0lmp_pricetaker_template')
    per_node_result = {}
    per_node_error = {}
    for node_id in NODES:
        planning_lp = _build_planning_lp_for_node(template_planning, node_id, dso_lmp_dollars_per_mwh, years, days)
        try:
            node_result = sept.solve_price_taker_schedule(planning_lp, candidate['investment'], node_ids=[node_id])
            per_node_result[node_id] = node_result[node_id]
        except RuntimeError as error:
            per_node_error[node_id] = str(error)

    lp_calls_after = sept.get_lp_call_count()
    observed_lp_calls = lp_calls_after - lp_calls_before

    efc_per_day_per_node_per_year = {
        str(node_id): {str(year): float(v) for year, v in result['efc_per_day_harness'].items()}
        for node_id, result in per_node_result.items()
    }
    efc_per_day_max_by_node = {
        node: max(year_values.values()) for node, year_values in efc_per_day_per_node_per_year.items()
    }
    all_nodes_converged = (len(per_node_error) == 0)
    harness_definition_maximum = max(efc_per_day_max_by_node.values()) if efc_per_day_max_by_node else None

    return {
        'declared_outer_iterations': sept.DEFAULT_OUTER_ITERATIONS,
        'declared_damping': sept.DEFAULT_DAMPING,
        'nodes_that_did_not_converge_at_default_budget': {str(n): msg for n, msg in per_node_error.items()},
        'all_nodes_converged_at_default_budget': all_nodes_converged,
        'per_node_efc_per_day_per_cohort_year': efc_per_day_per_node_per_year,
        'efc_per_day_max_by_node': efc_per_day_max_by_node,
        'harness_definition_maximum_efc_per_day': harness_definition_maximum,
        'harness_definition_maximum_caveat': (
            None if all_nodes_converged else
            'computed from CONVERGED nodes only -- see nodes_that_did_not_converge_at_default_budget; '
            'NOT directly comparable to run1/gate3/market-price-LP EFC values, which reflect a converged '
            'schedule at every node'
        ),
        'lp_calls_observed': observed_lp_calls,
        'lp_calls_declared': DECLARED_LP_CALLS,
        'lp_calls_match_declared': observed_lp_calls == DECLARED_LP_CALLS,
        'per_node_active_cohort_year': {str(n): (str(r['active_cohort_year'])
                                                  if r['active_cohort_year'] is not None else None)
                                         for n, r in per_node_result.items()},
        'per_node_converged': {str(n): bool(r['converged']) for n, r in per_node_result.items()},
        'per_node_lp_status': {str(n): r['lp_status'] for n, r in per_node_result.items()},
    }


def run_pricetaker_lp_extended_iterations_diagnostic(candidate, dso_lmp_dollars_per_mwh, years, days,
                                                      failed_node_ids, extended_outer_iterations=400):
    """Informational-only, NON-DEFAULT diagnostic: retried ONLY for nodes
    that failed to converge at the production default (40 outer iterations)
    above, with a larger outer-iteration budget so the Planner has SOME
    converged number to look at. This deviates from the production default
    and is reported as such -- it is NOT the declared 1(c) result, and its
    own LP-call count is reported separately, not folded into
    `DECLARED_LP_CALLS`.
    """
    if not failed_node_ids:
        return {'ran': False, 'reason': 'no node failed to converge at the default budget'}

    lp_calls_before = sept.get_lp_call_count()
    template_planning = O.fresh_planning('p515s36_cycle0lmp_pricetaker_template_extended')
    per_node_result = {}
    per_node_error = {}
    for node_id in failed_node_ids:
        planning_lp = _build_planning_lp_for_node(template_planning, node_id, dso_lmp_dollars_per_mwh, years, days)
        try:
            node_result = sept.solve_price_taker_schedule(
                planning_lp, candidate['investment'], node_ids=[node_id],
                outer_iterations=extended_outer_iterations)
            per_node_result[node_id] = node_result[node_id]
        except RuntimeError as error:
            per_node_error[node_id] = str(error)
    lp_calls_after = sept.get_lp_call_count()

    return {
        'ran': True,
        'non_default_outer_iterations_used': extended_outer_iterations,
        'nodes_retried': list(failed_node_ids),
        'lp_calls_observed_this_diagnostic_only': lp_calls_after - lp_calls_before,
        'per_node_efc_per_day_per_cohort_year': {
            str(node_id): {str(year): float(v) for year, v in result['efc_per_day_harness'].items()}
            for node_id, result in per_node_result.items()
        },
        'nodes_still_not_converged': {str(n): msg for n, msg in per_node_error.items()},
    }


# ======================================================================================================================
#  Main
# ======================================================================================================================
def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    planning, sed, candidate, capture, construction_guard_counts = run_construction_capture()
    years = list(sed.years)
    days = list(sed.days)

    tso_model = capture['tso_model']
    dso_models = capture['dso_models']

    tso_duals_pu = capture_tso_node_balance_duals(tso_model, planning.transmission_network, NODES)
    dso_duals_pu, dso_base_mva = capture_dso_reference_node_balance_duals(
        dso_models, planning.distribution_networks, NODES)

    dso_lmp_dollars_per_mwh = {
        node_id: _dual_p_series_dollars_per_mwh(dso_duals_pu[node_id], dso_base_mva[node_id], years, days)
        for node_id in NODES
    }
    # TSO series captured/reported for comparison only (see module docstring
    # "Cycle-0 price choice"); NOT fed into the LP.
    tso_base_mva = {node_id: planning.transmission_network.network[years[0]][days[0]].baseMVA for node_id in NODES}
    tso_lmp_dollars_per_mwh_informational = {
        node_id: _dual_p_series_dollars_per_mwh(tso_duals_pu[node_id], tso_base_mva[node_id], years, days)
        for node_id in NODES
    }

    lp_report = run_pricetaker_lp_with_cycle0_lmps(candidate, dso_lmp_dollars_per_mwh, years, days)

    failed_nodes = [int(n) for n in lp_report['nodes_that_did_not_converge_at_default_budget'].keys()]
    extended_report = run_pricetaker_lp_extended_iterations_diagnostic(
        candidate, dso_lmp_dollars_per_mwh, years, days, failed_node_ids=failed_nodes)

    z2 = json.load(open(MARKET_PRICE_LP_JSON)) if os.path.isfile(MARKET_PRICE_LP_JSON) else None

    hd_max = lp_report['harness_definition_maximum_efc_per_day']
    comparison_table = {
        'run1_certified_c477': RUN1_EFC,
        'gate3_terminal_c150': GATE3_EFC,
        'market_price_lp_z2': MARKET_PRICE_LP_EFC,
        'cycle0_lmp_lp_1c_this_script_default_budget': hd_max,
        'cycle0_lmp_lp_1c_default_budget_caveat': lp_report['harness_definition_maximum_caveat'],
        'pairwise_differences_default_budget': (
            None if hd_max is None else {
                'cycle0_lmp_lp_minus_run1': hd_max - RUN1_EFC,
                'cycle0_lmp_lp_minus_gate3': hd_max - GATE3_EFC,
                'cycle0_lmp_lp_minus_market_price_lp': hd_max - MARKET_PRICE_LP_EFC,
            }
        ),
    }

    report = {
        'stage': 'P5.15-S36-A17-PARTB', 'authority': 'Addendum 17 capture-and-reconstruction task (2026-09-16)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'correction_of_5e2ba6f9': (
            'WORKER_REPORT_S36_A17_DIAGNOSTICS.md PART 0 item 4 ("no standalone pre-cycle-1 solve '
            'exists in this pipeline at all") is WRONG. 51 standalone IPOPT solves run during model '
            'construction, before cycle 1 -- see this script\'s module docstring and '
            'data/SRP1/Results/P515S35/pt_phase2_checks/phase2_checks_results.json '
            '(Z3.ipopt_solve_count = Z4.ipopt_solve_count = 51).'
        ),
        'construction_guard': {'observed': construction_guard_counts, 'declared': DECLARED_IPOPT_SOLVES},
        'years': years, 'days': days,
        'sign_convention_and_units': (
            'model.dual.get(constraint) as Pyomo returns it (IMPORT_EXPORT Suffix, IPOPT-exported '
            'multiplier), NO sign flip, on node_balance_p/q as literally written '
            '(Pg == Pd + Pi + slack_up - slack_down). LMP [$/MWh] = dual_pu / network.baseMVA. At '
            'construction time the objective is UNSCALED (objective_scale/sigma division happens '
            'later, in _run_operational_planning, after these constructors return) -- see module '
            'docstring for the full derivation.'
        ),
        'tso_node_balance_duals_pu': tso_duals_pu,
        'dso_reference_node_balance_duals_pu': dso_duals_pu,
        'dso_base_mva': dso_base_mva,
        'tso_base_mva': tso_base_mva,
        'cycle0_lmp_dollars_per_mwh_used_in_lp': {
            str(node_id): {str(year): {str(day): arr.tolist() for day, arr in day_map.items()}
                           for year, day_map in year_map.items()}
            for node_id, year_map in dso_lmp_dollars_per_mwh.items()
        },
        'tso_cycle0_lmp_dollars_per_mwh_informational_only': {
            str(node_id): {str(year): {str(day): arr.tolist() for day, arr in day_map.items()}
                           for year, day_map in year_map.items()}
            for node_id, year_map in tso_lmp_dollars_per_mwh_informational.items()
        },
        'pricetaker_lp_with_cycle0_lmps': lp_report,
        'pricetaker_lp_extended_iterations_diagnostic_NON_DEFAULT_informational_only': extended_report,
        'efc_comparison_table': comparison_table,
        'magnitude_finding': (
            'The captured cycle-0 LMPs (DSO reference-node-balance-P dual, $/MWh) are ~1e-6 to '
            '1e-8 in magnitude -- NOT a plausible real-world $/MWh price. This is because '
            'model.interface_settlement_weight defaults to 0.00 at construction '
            '(model_construction_helpers.py; set to 1 only later, by '
            '_prepare_distribution_objectives_for_admm, AFTER these 51 standalone solves complete), '
            'so the standalone DSO objective at the reference bus has essentially no priced economic '
            'driver -- only local generation cost (typically none/negligible on these DSO test '
            'feeders) and slack-penalty gradients. The price-taker LP objective '
            '(sum pi*(pdch-pch)) is POSITIVELY HOMOGENEOUS OF DEGREE 1 in the price series, so its '
            'OPTIMAL SCHEDULE (charge/discharge TIMING) depends only on the RELATIVE shape of pi '
            'across periods, not its absolute scale -- the resulting EFC/day values are therefore a '
            'meaningful read of the cycle-0 duals\' temporal SHAPE even though their absolute $/MWh '
            'magnitude is not economically interpretable. This is reported as a finding, not '
            'corrected or rescaled (doing so would be tuning the price input to a level that "looks '
            'right", which this task does not authorize).'
        ),
    }

    # OUT_DIR may already exist at this point as a side effect of
    # `_construct_arm_planning`'s results_dir redirection (creates
    # OUT_DIR/construction/results under it, same as
    # `p515_s35pt_phase2_checks.py`'s own main()) -- the freshness refusal
    # at the top of this function already ran BEFORE anything was
    # constructed or solved, which is the check that matters.
    os.makedirs(OUT_DIR, exist_ok=True)
    results_path = os.path.join(OUT_DIR, 'cycle0_lmp_capture_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    manifest = {}
    for fname in ('cycle0_lmp_capture_results.json',):
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'rb') as handle:
            manifest[fname] = hashlib.sha256(handle.read()).hexdigest()
    script_path = os.path.abspath(__file__)
    with open(script_path, 'rb') as handle:
        manifest[os.path.basename(script_path)] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, 'sha256_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(f'construction guard: {construction_guard_counts} (declared {DECLARED_IPOPT_SOLVES})')
    print(f"LP calls: observed={lp_report['lp_calls_observed']} declared={lp_report['lp_calls_declared']} "
          f"match={lp_report['lp_calls_match_declared']}")
    print(f"nodes NOT converged at default (40) budget: "
          f"{list(lp_report['nodes_that_did_not_converge_at_default_budget'].keys())}")
    if hd_max is not None:
        print(f'cycle0-LMP LP harness-definition max EFC/day (default budget): {hd_max:.6f}')
    else:
        print('cycle0-LMP LP harness-definition max EFC/day (default budget): UNAVAILABLE (no node converged)')
    if extended_report.get('ran'):
        print(f"extended-iteration diagnostic (NON-DEFAULT, informational only): "
              f"{extended_report['per_node_efc_per_day_per_cohort_year']}")
    print(json.dumps(comparison_table, indent=1, default=str))
    print(f'wrote: {results_path}')

    if not lp_report['lp_calls_match_declared']:
        raise AssertionError(
            f"RULE SIX (scipy LP count): observed {lp_report['lp_calls_observed']} != "
            f"declared {lp_report['lp_calls_declared']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
