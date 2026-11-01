"""
P5.15 Step 3.4 diagnostic -- price-taking storage arbitrage benchmark EFC*.

Bounded diagnostic task (Planner instruction 2026-09-16). ZERO production solves:
`SolveProfileGuard` is armed with an EMPTY permitted list for the whole script, so any
Pyomo `OptSolver.solve` / `SystemCallSolver._execute_command` call (Pyomo, IPOPT) anywhere
in the call tree raises immediately. Only production DATA is read (`p56a_oracle.load_baseline`
calls `SharedResourcesPlanning.read_planning_problem`, which parses the case files and builds
no Pyomo model). The LP itself is solved with `scipy.optimize.linprog` (HiGHS), never Pyomo.

## Why (Addendum 15 / Step 3.4 gate)

Step 3.4's gate predicts "EFC/day rises from 0.067 to order 1" (`REVISION_CONTEXT.md`
Amendment 2026-09-16). The only order-1 EFC/day figures in the programme (~1.1) were measured
on a solver objective that still contained the row-3 TSO ADN-interface flexibility charge,
later judged a transfer payment and removed from the objective (Addendum 10/11). "EFC/day was
1.1 before" is therefore not a defensible target for what a rational, unconstrained arbitrage
schedule against the ACTUAL market price would look like. This script computes that
DERIVED benchmark directly: the profit-maximising price-taking schedule for C*'s own storage,
given the scenario's hourly market prices and the storage's own physical limits and
efficiencies (no coordination, no degradation cost, no shared-ESS usage price -- none of
these are priced in production either; see `REVISION_CONTEXT.md` Amendment 2026-09-16 point
2 of "Decisions in force").

## Formal problem, per (node, year, representative day)

Price-taker LP, maximising

    sum_p pi_p * (pdch_p - pch_p)

subject to (every row cites the production constraint it reproduces; SI: p.u. per network
convention, but here worked directly in the PHYSICAL units the parameters are recorded in --
MW / MWh / $-per-MWh -- since this is a standalone LP, not a Pyomo network model):

  - power limits          0 <= pch_p <= S,  0 <= pdch_p <= S                    [1]
  - active-power envelope pch_p + pdch_p <= S                                    [2]
  - SoC recursion         soc_p = soc_{p-1} + eff_ch*pch_p*dt - pdch_p*dt/eff_dch [3]
  - SoC bounds            E*ENERGY_STORAGE_MIN_ENERGY_STORED <= soc_p
                                <= E*ENERGY_STORAGE_MAX_ENERGY_STORED            [4]
  - day balance           soc_{P-1} = E*ENERGY_STORAGE_RELATIVE_INIT_SOC          [5]
                          (soc_{-1} := E*ENERGY_STORAGE_RELATIVE_INIT_SOC, same constant)

File:line for each row (production; this script reproduces the math, it does not import
these Pyomo rules, since a fresh scipy LP is required and must not touch Pyomo):
  [1] network.py:390-391 (`shared_es_pch`/`shared_es_pdch`, domain=NonNegativeReals) and
      model_construction_helpers.py:827-836 (`sess_active_sum_limit_rule`: the docstring
      there derives `pch<=S`, `pdch<=S` from the retired sch/sdch feasible set).
  [2] model_construction_helpers.py:827-836 (`sess_active_sum_limit_rule`,
      `pch + pdch <= S_rated`).
  [3] model_construction_helpers.py:893-912 (`sess_soc_rule`); the time step
      `dt = period_duration_hours(m)` is model_construction_helpers.py:797-808
      (`HOURS_PER_REPRESENTATIVE_DAY / len(periods)`, `HOURS_PER_REPRESENTATIVE_DAY = 24.0`
      at definitions.py:94; with the standard 24-instant day dt = 1 h exactly).
  [4] model_construction_helpers.py:840-847 (`sess_soc_lower_limit`, `sess_soc_upper_limit`);
      `ENERGY_STORAGE_MIN_ENERGY_STORED = 0.10`, `ENERGY_STORAGE_MAX_ENERGY_STORED = 0.90`
      at definitions.py:38-39.
  [5] model_construction_helpers.py:915-921 (`sess_soc_final_rule`). Production allows this
      row a small PENALIZED slack when `params.slacks.shared_ess.day_balance` is True
      (network_parameters.py:76, the Candidate-4-pinned default) -- this script enforces it
      as a hard equality, the tighter and economically correct choice for a price-taker
      upper bound (a price-taking storage would never leave uncompensated slack on a
      constraint that, if binding, only costs it foregone arbitrage the LP already prices).
      `ENERGY_STORAGE_RELATIVE_INIT_SOC = 0.50` at definitions.py:40.

Efficiencies `eff_ch`, `eff_dch` and rating `S`, `E` are read from the SAME production data
objects the ESSO/network models read (`SharedEnergyStorageData.shared_energy_storages`,
class defaults at shared_energy_storage.py:14-15, eff_ch=0.97, eff_dch=0.96 -- confirmed
un-overridden for C* by a zero-solve read, see Worker report), not hard-coded blind. `S` and
`E` are C*'s instance values (`p514_n_instrumented_cstar.py:42`, S_INV=0.96875 MVA,
E_INV=3.875 MWh, INVEST_YEAR=2025), taken as the nameplate rating/capacity for every
evaluated year (2025/2030/2035) exactly as the Planner's task specifies ("uniform across
nodes 5/7/9"); this benchmark does not apply an SoH derating, matching the fact that
production does not price degradation as a benchmark cost either (Amendment 2026-09-16,
"no shared-ESS usage price; ESSO carries no degradation cost").

Market price pi_p: `SharedEnergyStorageData.cost_energy_p[year][day][0]` (the single
market scenario, `NumMarketScenarios: 1`, `data/SRP1/SRP1.json:17`), the SAME array
`shared_resources_planning.py:6976` copies from `planning_problem.cost_energy_p`
(populated at `shared_resources_planning.py:7118`, `generation_cost` /
`interface_energy_settlement` in model_construction_helpers.py both read this identical
array via `network.cost_energy_p`). Cross-checked below against the serialized
`price_per_mwh` field in
`data/SRP1/Results/P515S33_E2_run/interface_settlement_detail_s31c.json`
(`interface_reporting_detail[node][year][day]['periods'][p]['price_per_mwh']`).

## EFC/day definition (preserved verbatim per task)

    EFC/day = sum_p (pch_p + pdch_p) / (2 * E_rated)

over the day's periods -- the numerator is a POWER sum here, not the ESSO's active-energy
throughput (`es_avg_ch_dch_per_unit`, shared_energy_storage_data.py:608-622, which sums
`eff_ch*pch*dt + pdch*dt/eff_dch`, annual-day-weighted). At dt = 1 h the two coincide
numerically in the POWER-vs-ENERGY sense (MW == MWh per period) but NOT in the efficiency
weighting; this is the formula the campaign harness's EFC block actually computes
(`p514_n_instrumented_cstar.py:133`, `value = avg / (2.0 * rated)` where `avg` there is
`es_avg_ch_dch_per_unit`) restricted to a single representative day and with the simple
(pch+pdch) numerator the Planner's task text specifies. Both readings are reported where
they differ (see script output field `efc_per_day_esso_style` for the throughput-based
reading using the same eff_ch/eff_dch weights, alongside the task-specified
`efc_per_day` = simple power-sum reading).

## Variants

  1. `EFC*_energy` -- arbitrage on pi alone (S, E as above, eff_ch=0.97, eff_dch=0.96).
  2. `EFC*_row3` -- SKIPPED. See `ROW3_SKIP_REASON` below: the mapping from `cost_flex` to
     a per-unit value earned by the shared ESS is not well defined from committed sources.
  3. Sensitivity -- variant 1 repeated with eff_ch = eff_dch = 1.0 (round-trip loss removed).

Usage:
    python p515_s34_efc_benchmark.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import linprog

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34', 'EFC_benchmark')
CROSS_CHECK_JSON = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S33_E2_run', 'interface_settlement_detail_s31c.json')

S_INV, E_INV, INVEST_YEAR = 0.96875, 3.875, 2025   # p514_n_instrumented_cstar.py:42
SOC_MIN_FRAC = 0.10                                 # definitions.py:39
SOC_MAX_FRAC = 0.90                                 # definitions.py:38
SOC_INIT_FRAC = 0.50                                # definitions.py:40
NODES = [5, 7, 9]
YEARS = [2025, 2030, 2035]

ROW3_SKIP_REASON = (
    "Row 3 (the TSO's ADN-interface flexibility charge, historically the source of the "
    "order-1 EFC/day figures the gate cites) is NOT part of any current production solver "
    "objective: `flexibility_cost` (model_construction_helpers.py:1687-1701) explicitly "
    "`continue`s over TSO ADN-interface loads (`load_is_tso_adn_interface`, "
    "model_construction_helpers.py:1682-1685), and the counterfactual value it WOULD have "
    "had is computed only by `adn_interface_flexibility_cost` "
    "(model_construction_helpers.py:1712-1732), whose own docstring states it is "
    "'NOT part of any solver objective... used only by reporting/harness code' "
    "(REVISION_CONTEXT.md Amendment 2026-09-15, row 3 removal per the signed penalty table, "
    "Addendum 10/11). The shared ESS's own dispatch variables (`shared_es_pch`/"
    "`shared_es_pdch` at the DSO's bus, network.py:390-391) are physically distinct from the "
    "TSO's ADN-interface load flexibility variables (`flex_p_down`/`flex_q_down` on loads at "
    "the TSO's ADN-interface buses, model_construction_helpers.py:1693-1699); any mapping "
    "from 'one unit of shared-ESS discharge at node e' to 'one unit of displaced TSO-side "
    "interface flexibility' runs through the coupled AC power flow of both the TSO and DSO "
    "networks and the ADMM interface consensus -- there is no committed algebraic identity "
    "for it, and recovering one would require re-solving the coupled AC OPF, which this "
    "bounded diagnostic (zero production solves) is barred from doing. Per the task's own "
    "instruction ('If the mapping is not well defined from committed sources, say so and "
    "skip this variant rather than inventing one'), variant 2 is skipped. "
    "`cost_flex` itself IS readable from committed production data "
    "(`SharedResourcesPlanning.cost_flex[year][day][0]`, shared_resources_planning.py:7119, "
    "populated exactly like `cost_energy_p`) -- e.g. cost_flex[2025]['Spring'][0][:6] = "
    "[50.39, 48.65, 46.73, 43.06, 45.28, 47.58] $/MWh at the time of this run -- so the "
    "skip is about the MAPPING onto the storage, not about data availability."
)


def solve_price_taker(prices, s_max, e_max, eff_ch, eff_dch, dt=1.0,
                       soc_min_frac=SOC_MIN_FRAC, soc_max_frac=SOC_MAX_FRAC,
                       soc_init_frac=SOC_INIT_FRAC):
    """Profit-maximising price-taking storage LP for one representative day.

    Variables x = [pch_0..pch_{n-1}, pdch_0..pdch_{n-1}, soc_0..soc_{n-1}].
    See module docstring for the constraint provenance ([1]-[5]).
    """
    n = len(prices)
    soc_min = e_max * soc_min_frac
    soc_max = e_max * soc_max_frac
    soc_init = e_max * soc_init_frac

    n_vars = 3 * n
    c = np.zeros(n_vars)
    c[0:n] = prices          # minimize +price*pch  (== maximize -price*pch)
    c[n:2 * n] = -prices     # minimize -price*pdch (== maximize +price*pdch)

    # [2] active-power envelope: pch_p + pdch_p <= s_max
    A_ub = np.zeros((n, n_vars))
    b_ub = np.full(n, s_max)
    for p in range(n):
        A_ub[p, p] = 1.0
        A_ub[p, n + p] = 1.0

    # [3] SoC recursion + [5] day balance
    A_eq = np.zeros((n + 1, n_vars))
    b_eq = np.zeros(n + 1)
    for p in range(n):
        A_eq[p, 2 * n + p] = 1.0
        if p == 0:
            b_eq[p] = soc_init
        else:
            A_eq[p, 2 * n + p - 1] = -1.0
        A_eq[p, p] = -eff_ch * dt
        A_eq[p, n + p] = dt / eff_dch
    A_eq[n, 2 * n + n - 1] = 1.0
    b_eq[n] = soc_init

    bounds = ([(0.0, s_max)] * n) + ([(0.0, s_max)] * n) + ([(soc_min, soc_max)] * n)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                   method='highs')
    if not res.success:
        raise RuntimeError(f'LP did not solve to optimality: {res.message}')

    x = res.x
    pch = x[0:n]
    pdch = x[n:2 * n]
    soc = x[2 * n:3 * n]
    profit = -res.fun

    tol_power = max(1e-9, 1e-6 * s_max)
    tol_energy = max(1e-9, 1e-6 * e_max)
    power_bound_periods = [p for p in range(n)
                            if pch[p] >= s_max - tol_power or pdch[p] >= s_max - tol_power]
    energy_bound_periods = [p for p in range(n)
                             if soc[p] <= soc_min + tol_energy or soc[p] >= soc_max - tol_energy]

    if power_bound_periods and energy_bound_periods:
        binding = 'power+energy'
    elif power_bound_periods:
        binding = 'power'
    elif energy_bound_periods:
        binding = 'energy'
    else:
        binding = 'day-balance-only (no power/energy bound active)'

    efc_per_day = float(np.sum(pch + pdch) / (2.0 * e_max))
    esso_style_throughput = float(np.sum(eff_ch * pch * dt + pdch * dt / eff_dch))
    efc_per_day_esso_style = esso_style_throughput / (2.0 * e_max)

    return {
        'status': res.status, 'success': bool(res.success), 'message': res.message,
        'profit': float(profit),
        'pch': pch.tolist(), 'pdch': pdch.tolist(), 'soc': soc.tolist(),
        'efc_per_day': efc_per_day,
        'efc_per_day_esso_style': float(efc_per_day_esso_style),
        'binding_constraint': binding,
        'power_bound_period_count': len(power_bound_periods),
        'energy_bound_period_count': len(energy_bound_periods),
        'power_bound_periods': power_bound_periods,
        'energy_bound_periods': energy_bound_periods,
        's_max': s_max, 'e_max': e_max, 'eff_ch': eff_ch, 'eff_dch': eff_dch, 'dt': dt,
        'soc_min': soc_min, 'soc_max': soc_max, 'soc_init': soc_init,
    }


def price_stats(prices):
    prices = np.asarray(prices, dtype=float)
    mean = float(prices.mean())
    spread = float(prices.max() - prices.min())
    mad = float(np.mean(np.abs(prices - mean)))
    return {'mean': mean, 'max_minus_min': spread, 'mean_abs_deviation': mad,
            'min': float(prices.min()), 'max': float(prices.max())}


def cross_check_prices(planning, nodes, years, days):
    """Compare `sed.cost_energy_p[year][day][0]` against the serialized
    `price_per_mwh` in interface_settlement_detail_s31c.json. Report per-cell max abs diff."""
    if not os.path.isfile(CROSS_CHECK_JSON):
        return {'available': False, 'reason': f'{CROSS_CHECK_JSON} not found'}
    with open(CROSS_CHECK_JSON) as handle:
        serialized = json.load(handle)
    ird = serialized['interface_reporting_detail']
    sed = planning.shared_ess_data
    results = {}
    max_abs_diff = 0.0
    for node in nodes:
        node_key = str(node)
        for year in years:
            year_key = str(year)
            for day in days:
                if node_key not in ird or year_key not in ird[node_key] or day not in ird[node_key][year_key]:
                    continue
                periods = ird[node_key][year_key][day]['periods']
                production_prices = sed.cost_energy_p[year][day][0]
                diffs = []
                for p_str, entry in periods.items():
                    p = int(p_str)
                    serialized_price = entry['price_per_mwh']
                    production_price = float(production_prices[p])
                    diffs.append(abs(serialized_price - production_price))
                cell_max = max(diffs) if diffs else None
                if cell_max is not None:
                    max_abs_diff = max(max_abs_diff, cell_max)
                results[f'{node}/{year}/{day}'] = {'max_abs_diff': cell_max, 'n_periods_checked': len(diffs)}
    return {'available': True, 'per_cell': results, 'max_abs_diff_overall': max_abs_diff}


def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    guard = SolveProfileGuard([], label='P5.15-S34 EFC benchmark').install()
    try:
        import p56a_oracle as O  # noqa: E402  (import after guard install; no solves on import)
        baseline = O.load_baseline()
        planning = baseline['planning']
        sed = planning.shared_ess_data

        active_nodes = list(sed.active_distribution_network_nodes)
        years = list(sed.years)
        days = list(sed.days)
        if sorted(active_nodes) != sorted(NODES):
            raise AssertionError(f'active nodes {active_nodes} != expected {NODES}')
        if sorted(years) != sorted(YEARS):
            raise AssertionError(f'years {years} != expected {YEARS}')

        idx0 = sed.get_shared_energy_storage_idx(active_nodes[0])
        ess0 = sed.shared_energy_storages[YEARS[0]][idx0]
        eff_ch, eff_dch = ess0.eff_ch, ess0.eff_dch
        for node in active_nodes:
            idx = sed.get_shared_energy_storage_idx(node)
            for year in years:
                ess = sed.shared_energy_storages[year][idx]
                if ess.eff_ch != eff_ch or ess.eff_dch != eff_dch:
                    raise AssertionError(
                        f'eff_ch/eff_dch not uniform across node/year: node={node} year={year} '
                        f'got ({ess.eff_ch},{ess.eff_dch}) expected ({eff_ch},{eff_dch})')

        price_cross_check = cross_check_prices(planning, active_nodes, years, days)

        report = {
            'stage': 'P5.15-S34', 'authority': 'Planner task 2026-09-16 (EFC* price-taking benchmark)',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'instance': {'s_mva': S_INV, 'e_mwh': E_INV, 'invest_year': INVEST_YEAR,
                         'eff_ch': eff_ch, 'eff_dch': eff_dch,
                         'soc_min_frac': SOC_MIN_FRAC, 'soc_max_frac': SOC_MAX_FRAC,
                         'soc_init_frac': SOC_INIT_FRAC, 'dt_hours': 1.0},
            'row3_variant': {'computed': False, 'skip_reason': ROW3_SKIP_REASON},
            'price_cross_check_vs_s31c_json': price_cross_check,
            'results': {},
        }

        efc_energy_2025 = []
        efc_efficiency1_2025 = []

        for node in active_nodes:
            report['results'][str(node)] = {}
            for year in years:
                report['results'][str(node)][str(year)] = {}
                for day in days:
                    prices = np.asarray(sed.cost_energy_p[year][day][0], dtype=float)
                    stats = price_stats(prices)

                    variant1 = solve_price_taker(prices, S_INV, E_INV, eff_ch, eff_dch)
                    variant3 = solve_price_taker(prices, S_INV, E_INV, 1.0, 1.0)

                    cell = {
                        'price_stats': stats,
                        'efc_star_energy': variant1['efc_per_day'],
                        'efc_star_energy_esso_style': variant1['efc_per_day_esso_style'],
                        'profit_variant1': variant1['profit'],
                        'binding_constraint_variant1': variant1['binding_constraint'],
                        'power_bound_period_count_variant1': variant1['power_bound_period_count'],
                        'energy_bound_period_count_variant1': variant1['energy_bound_period_count'],
                        'efc_star_efficiency1': variant3['efc_per_day'],
                        'profit_variant3': variant3['profit'],
                        'binding_constraint_variant3': variant3['binding_constraint'],
                        'efc_star_row3': None,
                    }
                    report['results'][str(node)][str(year)][day] = cell

                    if year == 2025:
                        efc_energy_2025.append((node, day, variant1['efc_per_day']))
                        efc_efficiency1_2025.append((node, day, variant3['efc_per_day']))

        max_energy_2025 = max(efc_energy_2025, key=lambda t: t[2])
        max_eff1_2025 = max(efc_efficiency1_2025, key=lambda t: t[2])
        report['summary'] = {
            'efc_star_energy_2025_max_across_nodes': {
                'value': max_energy_2025[2], 'node': max_energy_2025[0], 'day': max_energy_2025[1]},
            'efc_star_efficiency1_2025_max_across_nodes': {
                'value': max_eff1_2025[2], 'node': max_eff1_2025[0], 'day': max_eff1_2025[1]},
            'efc_star_row3_2025_max_across_nodes': None,
            'harness_reference_values': {
                's33e2_efc_per_day': 0.067, 's31c_efc_per_day': 0.0285,
                'pre_table_runs_efc_per_day_order1': 1.1,
                'efc_binding_threshold': 1.4612,
            },
        }
        order_of_magnitude = 'order 1' if max_energy_2025[2] >= 0.3 else 'order 0.1'
        report['verdict'] = (
            f"EFC*_energy (2025, max across nodes 5/7/9) = {max_energy_2025[2]:.4f} "
            f"(node {max_energy_2025[0]}, {max_energy_2025[1]}) -> {order_of_magnitude}."
        )
    finally:
        failures = guard.verify(0)
        guard.uninstall()

    if failures:
        raise AssertionError('RULE SIX: solve guard was not exactly zero -> ' + '; '.join(failures))
    report['solve_profile'] = {'observed': dict(guard.counts), 'permitted_declared': 0,
                               'guard_verify_failures': failures}

    os.makedirs(OUT_DIR)
    results_path = os.path.join(OUT_DIR, 'efc_benchmark_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    manifest = {}
    for fname in ('efc_benchmark_results.json',):
        fpath = os.path.join(OUT_DIR, fname)
        with open(fpath, 'rb') as handle:
            manifest[fname] = hashlib.sha256(handle.read()).hexdigest()
    script_path = os.path.abspath(__file__)
    with open(script_path, 'rb') as handle:
        manifest[os.path.basename(script_path)] = hashlib.sha256(handle.read()).hexdigest()
    manifest_path = os.path.join(OUT_DIR, 'sha256_manifest.json')
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(report['verdict'])
    print(f"solve_profile: {report['solve_profile']}")
    print(f"price cross-check max abs diff vs s31c JSON: "
          f"{price_cross_check.get('max_abs_diff_overall')}")
    print(f"wrote: {results_path}")
    print(f"wrote: {manifest_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
