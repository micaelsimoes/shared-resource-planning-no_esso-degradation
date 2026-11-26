"""
P5.15 Step 3.5 -- bounded ZERO-SOLVE diagnostic Z2.

Question (Planner task, 2026-09-16): is the SoH floor active or slack at
candidate C* under the price-taker schedule, measured with the PRODUCTION EFC
and degradation formulas? No Pyomo/IPOPT solves anywhere in this script
(`SolveProfileGuard` is armed with an EMPTY permitted list for the whole run
and `guard.verify(0)` is checked before anything is written). A declared,
counted number of `scipy.optimize.linprog` (HiGHS) calls is used instead --
see DECLARED LP COUNT below.

## Why (verbatim from task)

Addendum 16 defines success as "equilibrium at the SoH threshold with the
degradation constraint active". `EFC_BINDING_THRESHOLD = 1.4612`
(`p514_n_instrumented_cstar.py:46`) is the EFC/day at which the floor
`es_soh_per_unit_cumul[y_inv, y] >= soh_min` (0.50) binds by 2035. The
price-taker benchmark EFC* = 1.852 (`data/SRP1/Results/P515S34/EFC_benchmark/
efc_benchmark_results.json`) is a SINGLE-DAY maximum (2025 Winter, node 5),
not the harness's own day-weighted annual EFC/day. This script computes the
harness-definition quantity directly and checks the floor.

## 1. The harness EFC/day formula (file:line)

`p514_n_instrumented_cstar.py:118-144` (`capture_esso`), specifically line 133:

    value = avg / (2.0 * rated)          # EFC/day = throughput / (2 * E_rated)

where `avg = es_avg_ch_dch_per_unit[y_inv, y]` (a Pyomo Var/value) and
`rated = es_e_rated_per_unit[y_inv, y]` (also captured there). `efc_per_day_max`
(line 139) is the MAX over the cohort-years captured (over `y` for the model's
one active cohort `y_inv=0`, i.e. over the three modelled calendar years).

`es_avg_ch_dch_per_unit[y_inv, y]` is defined by the constraint
`energy_storage_charging_discharging` (`shared_energy_storage_data.py:587-622`):

    avg_ch_dch[y_inv, y] = sum_d (num_days[d] / 365) * sum_p
                                (eff_ch * pch[y_inv,y,d,p] * dt
                                 + pdch[y_inv,y,d,p] * dt / eff_dch)

-- EFFICIENCY-WEIGHTED (eta_ch*pch + pdch/eta_dch, matching the SOC recursion,
`sess_soc_rule`/`shared_energy_storage_data.py`'s own SOC row), DAY-WEIGHTED
(`num_days[d]/365`, SRP1 day weights Spring/Summer/Autumn/Winter =
92/91/91/91, `data/SRP1/SRP1.json:"Days"`), summed over the representative
day's periods with `dt = period_duration_hours(model)` = 1 h exactly at the
standard 24-instant day (`model_construction_helpers.py:797-808`,
`shared_energy_storage_data.py:608`). `rated` = `es_e_rated_per_unit[y_inv,y]`,
which is constrained equal to `es_e_investment_fixed[y_inv]`
(`shared_energy_storage_data.py:552`, `rated_e_capacity_unit`) -- the NAMEPLATE
investment, NOT the SoH-derated `es_e_available_per_unit`
(`es_e_rated_per_unit * es_soh_per_unit_cumul`, line 584). So the harness
EFC/day denominator is always nameplate E, never derated E, for every cohort
year within the cohort's active window.

## 2. The degradation chain (file:line)

`shared_energy_storage_data.py:631-676` (`energy_storage_capacity_degradation`):

    D[y_inv, y] * (2 * cl_eff * E_inv[y_inv]) == 365 * num_years * avg_ch_dch[y_inv, y]   (line 652-655)
    soh_cumul[y_inv, y] == prev_soh * exp(-D[y_inv, y]) * phi_cal**num_years              (line 665-668)
    soh_cumul[y_inv, y] >= soh_min                                                        (line 674-676, EVERY y, not only terminal)

with `prev_soh = 1.00` at `y == y_inv` and `prev_soh = soh_cumul[y_inv, y-1]`
otherwise (line 658-660). `E_inv[y_inv]` is `es_e_investment_fixed[y_inv]`, the
SAME nameplate Param as the EFC denominator (so D[y_inv,y] and EFC/day[y_inv,y]
are exactly proportional: D = 365*num_years*EFC_per_day / cl_eff, independent
of any SoH derating -- confirmed algebraically below and numerically in the
output).

`cl_eff`: with the C3 calibration ACTIVE (`data/SRP1/SharedESS/
SRP1_ESS_Params.json`, `ageing.calibration.status="ACTIVE"`,
`shared_energy_storage_parameters.py:188-201`),
`cl_eff = cycles_n * reference_dod_d / (-ln(eol_retention_r))
        = 10000 * 0.80 / (-ln(0.50)) = 11541.560327111707`
(confirmed by a zero-solve read of `shared_energy_storage.cl_eff` below, see
`instance_parameters` in the output JSON).
`phi_cal = calendar_retention_per_year`, absent from `SRP1_ESS_Params.json`'s
`ageing` block, so it takes the class default 1.00
(`shared_energy_storage_parameters.py:144`) -- NEUTRAL, no calendar ageing
beyond cycling. `soh_min = 0.50`, `t_cal = 15` years
(`SRP1_ESS_Params.json:"ageing"`).

`num_years` (block width) = 5 for every one of 2025/2030/2035
(`data/SRP1/SRP1.json:"Years"`). The cohort's active window is
`y_inv=0` (2025) through `min(y_inv + round(t_cal/num_years), len(years)) - 1
= min(0 + round(15/5), 3) - 1 = 2` (2035), i.e. ALL THREE modelled years share
one SoH chain (`shared_energy_storage_data.py:636-637`).

Linear equivalent of the floor (task's algebra, checked here): for the
cumulative chain starting at `prev_soh=1`,

    soh_cumul[y_inv, Y] = exp(-sum_{y=y_inv}^{Y} D[y_inv,y]) * phi_cal**(sum_{y=y_inv}^{Y} num_years[y])

so `soh_cumul[y_inv,Y] >= soh_min`

    <=>  -sum D + (sum num_years) * ln(phi_cal) >= ln(soh_min)
    <=>  sum_{y<=Y} D[y_inv,y]  <=  -ln(soh_min) + (sum_{y<=Y} num_years[y]) * ln(phi_cal)

which is exactly the task's stated form (sign check: ln(soh_min) is negative
for soh_min<1, so `-ln(soh_min)>0`; `ln(phi_cal)<=0` for phi_cal<=1, so the
RHS is non-increasing in calendar ageing, as physically expected). With
phi_cal=1.00 the RHS is IDENTICAL for every Y (`-ln(0.50) = 0.6931472...`), so
since D>=0 (NonNegativeReals) the cumulative sum is non-decreasing in Y and the
terminal row (Y=2035) is always at least as tight as the earlier ones -- the
terminal SoH is the correct single quantity to check for "floor active by
2035".

## 3. Capacity wear -- what the networks and the ESSO actually see

`es_s_available_per_unit[y_inv,y] == es_s_rated_per_unit[y_inv,y]`
(`shared_energy_storage_data.py:583`, NO SoH factor -- power rating never
derates). `es_e_available_per_unit[y_inv,y] == es_e_rated_per_unit[y_inv,y] *
es_soh_per_unit_cumul[y_inv,y]` (line 584, SoH-derated energy capacity).
These per-unit available capacities are aggregated into `es_e_rated[y]` /
`es_s_rated[y]` -- NO: aggregated into `sess_estimated_capacity[year]
['e_available'/'s_available']` via `SharedEnergyStorageData.get_updated_capacities`
/ `get_available_capacities` (`shared_energy_storage_data.py:246-262`, summing
`es_e_available_per_unit`/`es_s_available_per_unit` over investment cohorts),
and THAT is what is pushed into the TSO/DSO network models' own physical
dispatch bound `shared_es_e_rated_fixed` / `shared_es_s_rated_fixed`
(`network.py:388-389`, mutable Params; set at
`shared_resources_planning.py:5004-5005`, `:5201-5202`, `:5326-5327`), which
in turn bound `sess_soc_lower_limit`/`sess_soc_upper_limit`/
`sess_active_sum_limit_rule`/`sess_converter_capability_rule`
(`model_construction_helpers.py:809-847`). So: **S stays nameplate for every
year; E for the physical SOC/dispatch bounds IS SoH-derated (E*SoH) in every
year, while the DEGRADATION LAW'S OWN DENOMINATOR stays nameplate** (section 1
above) -- a deliberate asymmetry preserved from the original law (comment at
`shared_energy_storage_data.py:309`: "considering calendar life, not
degradation").

Within one ESSO solve, `es_e_available_per_unit[y_inv,y]` and
`es_soh_per_unit_cumul[y_inv,y]` (which drives it) are co-optimised
SIMULTANEOUSLY with the dispatch that produces `avg_ch_dch[y_inv,y]` (all are
Vars in the SAME IPOPT NLP) -- a genuine same-year self-consistency this
zero-solve diagnostic cannot reproduce by solving an NLP. Instead (per the
task's explicit instruction) it is approximated by FIXED-POINT ITERATION: for
each (node, year), solve the day-LPs at a trial available capacity, recompute
D/SoH/available-capacity from the result, and repeat to convergence with a
fixed, declared iteration count -- see `FIXED_POINT_ITERS` below. This is an
approximation of the joint NLP's self-consistency, not a reproduction of it;
stated as an assumption, not a fact about production.

## Price-taker LP (per day) -- reused from `p515_s34_efc_benchmark.py`

Same rows [1]-[5] as that script's docstring (not re-derived here; see that
file for the file:line citations for each row). Read-only reuse: this script
does not import `p515_s34_efc_benchmark.py`'s functions (Planner instruction:
"you may ... or reimplement it -- state which"); it REIMPLEMENTS
`solve_price_taker` verbatim (same math) so this script owns its own LP-count
bookkeeping without touching that committed script or its output directory.

Efficiencies, S, E, prices, day weights: read from the SAME production data
objects `p515_s34_efc_benchmark.py` reads (`SharedEnergyStorageData`,
`SharedResourcesPlanning.cost_energy_p`), confirmed uniform across
node/year in the output JSON (`instance_parameters` block).

## DECLARED LP COUNT (checked exactly at the end, RULE SIX/"declare and count")

  Part A (unconstrained fixed point): NODES(3) * YEARS(3) * DAYS(4) *
      FIXED_POINT_ITERS(20) = 720. Iterations are NOT stopped early (every
      declared iteration always runs, for a deterministic count); convergence
      is CHECKED after the fact, not enforced by early exit.
  Cross-check variant3 (eff_ch=eff_dch=1.0, nameplate E, uncoupled -- the ONE
      cross-check reading Part A's own iteration-0 results cannot supply,
      since Part A always uses the real efficiencies): NODES(3) * YEARS(3) *
      DAYS(4) = 36.
      (The eff-on / nameplate-E / uncoupled cross-check reading is read
      directly off Part A's OWN iteration-0 per-day results -- iteration 0 of
      the fixed point always starts from the nameplate guess -- at ZERO
      extra LP calls.)
  BASE TOTAL = 720 + 36 = 756, unconditional.
  Part D (coupled re-solve with the linear floor row): ONE additional LP per
      node whose Part-A UNCONSTRAINED terminal (2035) SoH is AT OR BELOW
      soh_min (the floor would bind there); ZERO for a node whose
      unconstrained terminal SoH already clears the floor (the floor is
      provably slack for that node without re-solving anything -- adding a
      constraint that is already satisfied cannot change the optimum). This
      count is a DETERMINISTIC function of Part A's own output, computed and
      printed BEFORE Part D executes, then checked exactly after.
  TOTAL DECLARED = 756 + (0 to 3, per the rule above, printed before Part D runs).

Usage:
    python p515_s35_z2_floor_slackness.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from math import exp, log

import numpy as np
from scipy.optimize import linprog

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35', 'Z2')
CROSS_CHECK_JSON = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S34', 'EFC_benchmark', 'efc_benchmark_results.json')

S_INV, E_INV, INVEST_YEAR = 0.96875, 3.875, 2025          # p514_n_instrumented_cstar.py:42
SOC_MIN_FRAC = 0.10                                         # definitions.py:39 (ENERGY_STORAGE_MIN_ENERGY_STORED)
SOC_MAX_FRAC = 0.90                                         # definitions.py:38 (ENERGY_STORAGE_MAX_ENERGY_STORED)
SOC_INIT_FRAC = 0.50                                        # definitions.py:40 (ENERGY_STORAGE_RELATIVE_INIT_SOC)
NODES = [5, 7, 9]
YEARS = [2025, 2030, 2035]
EFC_BINDING_THRESHOLD = 1.4612                              # p514_n_instrumented_cstar.py:46

FIXED_POINT_ITERS = 20                                       # declared, fixed, no early exit
FIXED_POINT_CONVERGENCE_TOL_REL = 1e-9                        # checked AFTER the fixed iterations, not enforced

# ---- scipy LP call counter (declared and verified exactly; see module docstring) ----
_LP_CALLS = {'count': 0}


def solve_price_taker(prices, s_max, e_max, eff_ch, eff_dch, dt=1.0,
                       soc_min_frac=SOC_MIN_FRAC, soc_max_frac=SOC_MAX_FRAC,
                       soc_init_frac=SOC_INIT_FRAC):
    """Profit-maximising price-taking storage LP for one representative day.

    Reimplements `p515_s34_efc_benchmark.py:solve_price_taker` verbatim (same
    rows [1]-[5], same objective); see that script's docstring for the
    file:line provenance of every row. This script does not import that
    function (read-only reuse boundary stated in the Planner task).
    """
    _LP_CALLS['count'] += 1
    n = len(prices)
    soc_min = e_max * soc_min_frac
    soc_max = e_max * soc_max_frac
    soc_init = e_max * soc_init_frac

    n_vars = 3 * n
    c = np.zeros(n_vars)
    c[0:n] = prices
    c[n:2 * n] = -prices

    A_ub = np.zeros((n, n_vars))
    b_ub = np.full(n, s_max)
    for p in range(n):
        A_ub[p, p] = 1.0
        A_ub[p, n + p] = 1.0

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

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'LP did not solve to optimality: {res.message}')

    x = res.x
    pch = x[0:n]
    pdch = x[n:2 * n]
    soc = x[2 * n:3 * n]
    profit = -res.fun
    throughput_energy = float(np.sum(eff_ch * pch * dt + pdch * dt / eff_dch))  # what avg_ch_dch sums

    return {'success': bool(res.success), 'profit': float(profit),
            'pch': pch.tolist(), 'pdch': pdch.tolist(), 'soc': soc.tolist(),
            'throughput_energy': throughput_energy,
            'simple_power_efc': float(np.sum(pch + pdch) / (2.0 * e_max)),
            's_max': s_max, 'e_max': e_max, 'eff_ch': eff_ch, 'eff_dch': eff_dch}


def year_avg_ch_dch(prices_by_day, day_weights, s_max, e_max, eff_ch, eff_dch):
    """avg_ch_dch[y_inv,y] -- shared_energy_storage_data.py:614-622 -- for ONE
    year, day-weighted over the representative days, at trial capacity e_max."""
    total = 0.0
    per_day = {}
    for day, prices in prices_by_day.items():
        result = solve_price_taker(prices, s_max, e_max, eff_ch, eff_dch)
        per_day[day] = result
        total += (day_weights[day] / 365.0) * result['throughput_energy']
    return total, per_day


def solve_year_fixed_point(prices_by_day, day_weights, s_max, e_inv_nameplate,
                            eff_ch, eff_dch, cl_eff, num_years, prev_soh, phi_cal,
                            iters=FIXED_POINT_ITERS):
    """Fixed-point self-consistency for ONE cohort-year (section 3 of the module
    docstring). ALWAYS runs exactly `iters` LP-solving rounds (no early exit),
    for a deterministic declared LP count; convergence is checked afterward."""
    e_avail_guess = e_inv_nameplate
    history = []
    iter0_per_day = None
    for it in range(iters):
        avg_ch_dch, per_day = year_avg_ch_dch(prices_by_day, day_weights, s_max,
                                               e_avail_guess, eff_ch, eff_dch)
        if it == 0:
            iter0_per_day = per_day       # nameplate-capacity cross-check reading
        d_y = 365.0 * num_years * avg_ch_dch / (2.0 * cl_eff * e_inv_nameplate)
        soh_y = prev_soh * exp(-d_y) * (phi_cal ** num_years)
        e_avail_new = e_inv_nameplate * soh_y
        rel_change = abs(e_avail_new - e_avail_guess) / max(1e-12, e_inv_nameplate)
        history.append({'iter': it, 'e_avail_guess': e_avail_guess, 'avg_ch_dch': avg_ch_dch,
                         'D_y': d_y, 'soh_y': soh_y, 'e_avail_new': e_avail_new,
                         'rel_change': rel_change})
        e_avail_guess = e_avail_new

    final = history[-1]
    return {'avg_ch_dch': final['avg_ch_dch'], 'D_y': final['D_y'], 'soh_y': final['soh_y'],
            'e_avail': final['e_avail_new'], 'e_avail_guess_used_for_last_lp': history[-1]['e_avail_guess'],
            'per_day_last_iteration': per_day, 'per_day_iteration0_nameplate': iter0_per_day,
            'iterations': iters, 'final_rel_change': final['rel_change'],
            'converged': final['rel_change'] < FIXED_POINT_CONVERGENCE_TOL_REL,
            'history': history}


def solve_coupled_price_taker_with_floor(node, prices_by_year_day, day_weights, s_max,
                                          e_inv_nameplate, eff_ch, eff_dch, cl_eff,
                                          num_years_by_year, phi_cal, soh_min,
                                          e_avail_by_year_from_part_a, years):
    """Part D -- one JOINT LP per node across all (year, day, period), with the
    LINEAR floor row `sum_{y'<=Y} D[y'] <= -ln(soh_min) + (sum n) * ln(phi_cal)`
    added for every Y (shared_energy_storage_data.py:674-676, task's linear
    equivalent, section 2 above). SOC-bound capacity per year is held FIXED at
    Part A's converged (unconstrained) available capacity -- re-coupling
    capacity to the constrained dispatch would reintroduce the same-year
    bilinearity a scipy LP cannot represent; stated here as an explicit
    simplification of Part D, not a claim that this exactly reproduces the
    coupled NLP under a binding floor.
    """
    _LP_CALLS['count'] += 1
    n_periods = 24
    days = list(day_weights)
    n_days = len(days)
    n_years = len(years)
    # variable layout: for y in years, for d in days, for p in periods: pch, pdch, soc
    block = n_days * n_periods
    n_vars = n_years * block * 3

    def idx_pch(y, d, p):
        return y * block * 3 + d * n_periods + p

    def idx_pdch(y, d, p):
        return y * block * 3 + block + d * n_periods + p

    def idx_soc(y, d, p):
        return y * block * 3 + 2 * block + d * n_periods + p

    c = np.zeros(n_vars)
    A_ub_rows = []
    b_ub = []
    A_eq_rows = []
    b_eq = []
    bounds = [(0.0, 0.0)] * n_vars

    for yi, year in enumerate(years):
        e_max = e_avail_by_year_from_part_a[year]
        soc_min = e_max * SOC_MIN_FRAC
        soc_max = e_max * SOC_MAX_FRAC
        soc_init = e_max * SOC_INIT_FRAC
        for di, day in enumerate(days):
            prices = prices_by_year_day[year][day]
            for p in range(n_periods):
                pch_i, pdch_i, soc_i = idx_pch(yi, di, p), idx_pdch(yi, di, p), idx_soc(yi, di, p)
                c[pch_i] = prices[p]
                c[pdch_i] = -prices[p]
                bounds[pch_i] = (0.0, s_max)
                bounds[pdch_i] = (0.0, s_max)
                bounds[soc_i] = (soc_min, soc_max)

                # [2] active-power envelope
                row = np.zeros(n_vars)
                row[pch_i] = 1.0
                row[pdch_i] = 1.0
                A_ub_rows.append(row)
                b_ub.append(s_max)

                # [3]/[5] SoC recursion and day balance
                row = np.zeros(n_vars)
                row[soc_i] = 1.0
                row[pch_i] = -eff_ch
                row[pdch_i] = 1.0 / eff_dch
                if p == 0:
                    b_eq.append(soc_init)
                else:
                    row[idx_soc(yi, di, p - 1)] = -1.0
                    b_eq.append(0.0)
                A_eq_rows.append(row)
            row = np.zeros(n_vars)
            row[idx_soc(yi, di, n_periods - 1)] = 1.0
            A_eq_rows.append(row)
            b_eq.append(soc_init)

    # Floor rows: sum_{y'<=Y} D[y'] <= -ln(soh_min) + (sum n) * ln(phi_cal), for every Y.
    # D[y'] = 365 * num_years[y'] * avg_ch_dch[y'] / (2 * cl_eff * E_inv), and
    # avg_ch_dch[y'] = sum_d (num_days[d]/365) * throughput_energy[y',d]
    # = sum_d (num_days[d]/365) * sum_p (eff_ch*pch + pdch/eff_dch).
    rhs_base = -log(soh_min)
    cum_n = 0.0
    for yi, year in enumerate(years):
        cum_n += num_years_by_year[year]
        row = np.zeros(n_vars)
        for yj in range(yi + 1):
            coeff = 365.0 * num_years_by_year[years[yj]] / (2.0 * cl_eff * e_inv_nameplate)
            for di, day in enumerate(days):
                w = day_weights[day] / 365.0
                for p in range(n_periods):
                    row[idx_pch(yj, di, p)] += coeff * w * eff_ch
                    row[idx_pdch(yj, di, p)] += coeff * w / eff_dch
        A_ub_rows.append(row)
        b_ub.append(rhs_base + cum_n * log(phi_cal))

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq_rows)
    b_eq = np.array(b_eq)

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'Coupled floor LP for node {node} did not solve: {res.message}')

    n_floor_rows = n_years
    floor_row_offset = len(A_ub_rows) - n_floor_rows
    floor_multipliers = res.ineqlin.marginals[floor_row_offset:floor_row_offset + n_floor_rows]
    terminal_multiplier = float(-floor_multipliers[-1])   # sign: <= row, marginal <=0 by scipy convention; report as a non-negative shadow price

    return {'success': bool(res.success), 'profit_total': float(-res.fun),
            'floor_multiplier_per_year': {str(y): float(-m) for y, m in zip(years, floor_multipliers)},
            'terminal_floor_multiplier': terminal_multiplier,
            'x': res.x.tolist()}


def price_stats(prices):
    prices = np.asarray(prices, dtype=float)
    return {'mean': float(prices.mean()), 'max_minus_min': float(prices.max() - prices.min()),
            'min': float(prices.min()), 'max': float(prices.max())}


def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    guard = SolveProfileGuard([], label='P5.15-S35 Z2 floor slackness').install()
    try:
        import p56a_oracle as O  # noqa: E402  (import after guard install; no solves on import)
        baseline = O.load_baseline()
        planning = baseline['planning']
        sed = planning.shared_ess_data

        active_nodes = list(sed.active_distribution_network_nodes)
        years = list(sed.years)
        days = list(sed.days)
        day_weights = {d: sed.days[d] for d in days}
        num_years_by_year = {y: sed.years[y] for y in years}
        if sorted(active_nodes) != sorted(NODES):
            raise AssertionError(f'active nodes {active_nodes} != expected {NODES}')
        if sorted(years) != sorted(YEARS):
            raise AssertionError(f'years {years} != expected {YEARS}')

        # ---- read + assert uniformity of ageing/efficiency params across node/year (as S34 did for eff) ----
        idx0 = sed.get_shared_energy_storage_idx(active_nodes[0])
        ess0 = sed.shared_energy_storages[years[0]][idx0]
        eff_ch, eff_dch = ess0.eff_ch, ess0.eff_dch
        cl_eff, phi_cal, soh_min, t_cal = ess0.cl_eff, ess0.phi_cal, ess0.soh_min, ess0.t_cal
        for node in active_nodes:
            idx = sed.get_shared_energy_storage_idx(node)
            for year in years:
                ess = sed.shared_energy_storages[year][idx]
                for attr, ref in (('eff_ch', eff_ch), ('eff_dch', eff_dch), ('cl_eff', cl_eff),
                                   ('phi_cal', phi_cal), ('soh_min', soh_min), ('t_cal', t_cal)):
                    got = getattr(ess, attr)
                    if got != ref:
                        raise AssertionError(f'{attr} not uniform: node={node} year={year} got={got} ref={ref}')

        # cohort window (shared_energy_storage_data.py:636-637), computed generally, not hardcoded
        y_inv_idx = years.index(INVEST_YEAR)
        tcal_norm = round(t_cal / num_years_by_year[INVEST_YEAR])
        max_tcal_norm = min(y_inv_idx + tcal_norm, len(years))
        cohort_years = years[y_inv_idx:max_tcal_norm]
        if cohort_years != years:
            raise AssertionError(f'cohort window {cohort_years} != all modelled years {years}; '
                                  f'script assumes a single cohort spanning every modelled year')

        # ---- cross-check LP-3 solve count sanity: prices per (node,year,day) ----
        prices_by_node_year_day = {}
        for node in active_nodes:
            prices_by_node_year_day[node] = {}
            for year in years:
                prices_by_node_year_day[node][year] = {
                    day: np.asarray(sed.cost_energy_p[year][day][0], dtype=float) for day in days}

        instance_parameters = {
            's_mva': S_INV, 'e_mwh': E_INV, 'invest_year': INVEST_YEAR,
            'eff_ch': eff_ch, 'eff_dch': eff_dch, 'cl_eff': cl_eff, 'phi_cal': phi_cal,
            'soh_min': soh_min, 't_cal': t_cal, 'num_years_by_year': num_years_by_year,
            'day_weights': day_weights, 'cohort_years': cohort_years,
            'soc_min_frac': SOC_MIN_FRAC, 'soc_max_frac': SOC_MAX_FRAC, 'soc_init_frac': SOC_INIT_FRAC,
        }

        declared_base = len(active_nodes) * len(years) * len(days) * FIXED_POINT_ITERS \
            + len(active_nodes) * len(years) * len(days)
        print(f'[Z2] declared BASE scipy LP count = {declared_base} '
              f'(Part A {len(active_nodes)}*{len(years)}*{len(days)}*{FIXED_POINT_ITERS} + '
              f'cross-check eff=1 {len(active_nodes)}*{len(years)}*{len(days)})')

        # ================================================================================
        # Part A -- unconstrained price-taker, sequential fixed point per (node, year)
        # ================================================================================
        results_by_node = {}
        for node in active_nodes:
            prev_soh = 1.00
            year_results = {}
            for year in years:
                num_years = num_years_by_year[year]
                fp = solve_year_fixed_point(
                    prices_by_node_year_day[node][year], day_weights, S_INV, E_INV,
                    eff_ch, eff_dch, cl_eff, num_years, prev_soh, phi_cal)
                efc_per_day = fp['avg_ch_dch'] / (2.0 * E_INV)          # p514_n_instrumented_cstar.py:133
                year_results[year] = {
                    'efc_per_day': efc_per_day,
                    'avg_ch_dch': fp['avg_ch_dch'], 'D_y': fp['D_y'], 'soh_cumul': fp['soh_y'],
                    'e_avail_converged': fp['e_avail'], 'prev_soh': prev_soh,
                    'fixed_point_iterations': fp['iterations'],
                    'fixed_point_final_rel_change': fp['final_rel_change'],
                    'fixed_point_converged': fp['converged'],
                    'fixed_point_history_first_last': [fp['history'][0], fp['history'][-1]],
                    'per_day_last_iteration_summary': {
                        d: {'profit': r['profit'], 'simple_power_efc': r['simple_power_efc'],
                            'throughput_energy': r['throughput_energy']}
                        for d, r in fp['per_day_last_iteration'].items()},
                    '_iteration0_per_day': fp['per_day_iteration0_nameplate'],  # for cross-check, stripped before dump
                }
                prev_soh = fp['soh_y']
            results_by_node[node] = year_results

        efc_per_day_max_by_node = {node: max(v['efc_per_day'] for v in year_results.values())
                                    for node, year_results in results_by_node.items()}
        terminal_soh_by_node = {node: year_results[years[-1]]['soh_cumul']
                                 for node, year_results in results_by_node.items()}
        floor_active_by_node = {node: (terminal_soh_by_node[node] <= soh_min)
                                 for node in active_nodes}
        floor_margin_by_node = {node: (terminal_soh_by_node[node] - soh_min) for node in active_nodes}

        # ================================================================================
        # Cross-check -- reproduce the committed P5.15-S34 benchmark per-cell EFC values
        # ================================================================================
        cross_check = {'available': False}
        if os.path.isfile(CROSS_CHECK_JSON):
            with open(CROSS_CHECK_JSON) as handle:
                committed = json.load(handle)
            per_cell = {}
            max_abs_diff_variant1 = 0.0
            max_abs_diff_variant3 = 0.0
            for node in active_nodes:
                for year in years:
                    committed_cell_by_day = committed['results'][str(node)][str(year)]
                    it0 = results_by_node[node][year]['_iteration0_per_day']
                    for day in days:
                        our_v1 = it0[day]['simple_power_efc']
                        committed_v1 = committed_cell_by_day[day]['efc_star_energy']
                        diff_v1 = abs(our_v1 - committed_v1)
                        max_abs_diff_variant1 = max(max_abs_diff_variant1, diff_v1)

                        prices = prices_by_node_year_day[node][year][day]
                        eff1_result = solve_price_taker(prices, S_INV, E_INV, 1.0, 1.0)  # cross-check LP
                        our_v3 = eff1_result['simple_power_efc']
                        committed_v3 = committed_cell_by_day[day]['efc_star_efficiency1']
                        diff_v3 = abs(our_v3 - committed_v3)
                        max_abs_diff_variant3 = max(max_abs_diff_variant3, diff_v3)

                        per_cell[f'{node}/{year}/{day}'] = {
                            'variant1_ours': our_v1, 'variant1_committed': committed_v1, 'diff': diff_v1,
                            'variant3_ours': our_v3, 'variant3_committed': committed_v3, 'diff3': diff_v3}
            cross_check = {'available': True, 'per_cell': per_cell,
                            'max_abs_diff_variant1_efficiency_on_wear_off': max_abs_diff_variant1,
                            'max_abs_diff_variant3_efficiency_off_wear_off': max_abs_diff_variant3}
        else:
            print(f'[Z2] WARNING: cross-check file not found: {CROSS_CHECK_JSON}')

        # strip the private per-iteration-0 cache before dumping Part A results
        for node in active_nodes:
            for year in years:
                results_by_node[node][year].pop('_iteration0_per_day', None)

        # ================================================================================
        # (C) equivalent constant EFC/day that makes the floor exactly bind by 2035
        # ================================================================================
        sum_num_years = sum(num_years_by_year[y] for y in cohort_years)
        equivalent_constant_efc_threshold = (cl_eff * (-log(soh_min))) / (365.0 * sum_num_years)
        # (with phi_cal == 1.00; general form below folds it in via the RHS derivation, section 2)

        # ================================================================================
        # (D) coupled re-solve with the linear floor row -- ONLY for nodes where it binds
        # ================================================================================
        binding_nodes = [node for node in active_nodes if floor_active_by_node[node]]
        declared_part_d = len(binding_nodes)
        print(f'[Z2] Part A terminal SoH by node: {terminal_soh_by_node}')
        print(f'[Z2] floor ACTIVE (binds) for nodes: {binding_nodes if binding_nodes else "NONE (floor slack everywhere)"}')
        print(f'[Z2] declared Part D LP count = {declared_part_d} (one coupled LP per binding node)')

        part_d_results = {}
        for node in active_nodes:
            if node in binding_nodes:
                e_avail_by_year = {y: results_by_node[node][y]['e_avail_converged'] for y in years}
                coupled = solve_coupled_price_taker_with_floor(
                    node, prices_by_node_year_day[node], day_weights, S_INV, E_INV, eff_ch, eff_dch,
                    cl_eff, num_years_by_year, phi_cal, soh_min, e_avail_by_year, years)
                part_d_results[node] = {
                    'floor_active': True,
                    'floor_multiplier_terminal_2035': coupled['terminal_floor_multiplier'],
                    'floor_multiplier_per_year': coupled['floor_multiplier_per_year'],
                    'profit_total_constrained': coupled['profit_total'],
                }
            else:
                part_d_results[node] = {'floor_active': False, 'floor_multiplier_terminal_2035': 0.0,
                                         'floor_multiplier_per_year': {str(y): 0.0 for y in years},
                                         'note': 'floor provably slack under the unconstrained Part-A schedule; '
                                                 'no re-solve performed (adding a satisfied constraint cannot '
                                                 'change the optimum)'}

        declared_total = declared_base + declared_part_d

    finally:
        failures = guard.verify(0)
        guard.uninstall()

    if failures:
        raise AssertionError('RULE SIX: Pyomo/IPOPT solve guard was not exactly zero -> ' + '; '.join(failures))

    observed_lp_calls = _LP_CALLS['count']
    if observed_lp_calls != declared_total:
        raise AssertionError(
            f'RULE SIX (scipy LP count): observed {observed_lp_calls} != declared {declared_total} '
            f'(base {declared_base} + Part D {declared_part_d})')

    report = {
        'stage': 'P5.15-S35-Z2', 'authority': 'Planner task 2026-09-16 (bounded ZERO-SOLVE diagnostic Z2)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'instance_parameters': instance_parameters,
        'lp_call_accounting': {
            'declared_base': declared_base, 'declared_part_d': declared_part_d,
            'declared_total': declared_total, 'observed': observed_lp_calls,
            'exact_match': observed_lp_calls == declared_total,
            'pyomo_ipopt_solve_guard': {'observed': dict(guard.counts), 'permitted_declared': 0,
                                        'guard_verify_failures': failures},
        },
        'part_A_unconstrained_price_taker': results_by_node,
        'part_A_summary': {
            'efc_per_day_max_by_node': efc_per_day_max_by_node,
            'terminal_soh_2035_by_node': terminal_soh_by_node,
            'floor_active_by_node': floor_active_by_node,
            'floor_margin_by_node_soh_minus_soh_min': floor_margin_by_node,
        },
        'part_B_floor_activity': {
            'criterion': 'floor active iff unconstrained terminal (2035) SoH <= soh_min; slack otherwise',
            'soh_min': soh_min,
            'by_node': {str(node): {'terminal_soh': terminal_soh_by_node[node],
                                     'active': floor_active_by_node[node],
                                     'margin': floor_margin_by_node[node]}
                        for node in active_nodes},
        },
        'part_C_equivalent_constant_efc_threshold': {
            'computed': equivalent_constant_efc_threshold,
            'reference_EFC_BINDING_THRESHOLD': EFC_BINDING_THRESHOLD,
            'abs_diff': abs(equivalent_constant_efc_threshold - EFC_BINDING_THRESHOLD),
            'formula': 'cl_eff * (-ln(soh_min)) / (365 * sum(num_years over cohort_years)), phi_cal == 1.00',
        },
        'part_D_coupled_floor_resolve': part_d_results,
        'cross_check_vs_p515_s34_committed_benchmark': cross_check,
    }

    order_of_magnitude_note = (
        f"harness EFC/day (day-weighted annual, max over cohort-years) by node/year: " +
        "; ".join(f"node {node}: " + ", ".join(
            f"{y}={results_by_node[node][y]['efc_per_day']:.4f}" for y in years)
            for node in active_nodes)
    )
    report['verdict'] = (
        order_of_magnitude_note +
        f" | terminal (2035) SoH by node: " +
        ", ".join(f"{node}={terminal_soh_by_node[node]:.6f}" for node in active_nodes) +
        f" | soh_min={soh_min} | floor active nodes: {binding_nodes if binding_nodes else 'NONE (slack everywhere)'}"
    )

    os.makedirs(OUT_DIR)
    results_path = os.path.join(OUT_DIR, 'z2_floor_slackness_results.json')
    with open(results_path, 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    manifest = {}
    for fname in ('z2_floor_slackness_results.json',):
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
    print(f"scipy LP calls: declared_total={declared_total} observed={observed_lp_calls} "
          f"exact_match={observed_lp_calls == declared_total}")
    print(f"Pyomo/IPOPT solve guard: {dict(guard.counts)} (0 permitted, 0 blocked expected)")
    if cross_check.get('available'):
        print(f"cross-check vs committed S34 benchmark: "
              f"max_abs_diff (eff on, wear off) = {cross_check['max_abs_diff_variant1_efficiency_on_wear_off']:.3e}, "
              f"max_abs_diff (eff off, wear off) = {cross_check['max_abs_diff_variant3_efficiency_off_wear_off']:.3e}")
    print(f"wrote: {results_path}")
    print(f"wrote: {manifest_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
