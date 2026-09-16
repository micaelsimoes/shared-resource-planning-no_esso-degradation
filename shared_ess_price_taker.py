"""
shared_ess_price_taker.py

P5.15 Addendum 16 items 2-3, PHASE 1 (frozen spec v6,
`data/SRP1/Results/P515S35/frozen_s35pt_spec_v6_651a9d84.json`,
`initialization.lp`).

Production module computing a HISTORY-FREE, price-taking storage schedule
for the shared ESS at each active distribution-network node, given a
candidate investment. The result is used ONLY to build an INITIAL VALUE
for the ADMM shared-ESS consensus (`z`, agent copies, TSO proximal
centres, ESSO warm state) via
`_initialize_shared_ess_from_price_taker` in `shared_resources_planning.py`
-- never to fix anything, never inside the ADMM loop itself.

NO PYOMO IMPORT ANYWHERE IN THIS MODULE (spec v6 requirement). The LP is
solved with `scipy.optimize.linprog` (HiGHS) through the single
module-level call point `_solve_lp` below, so a check harness can count
the exact number of solver calls by patching/reading
`_LP_CALL_COUNTER['count']` (spec v6 `initialization.lp.solver`: "its own
armed counter with the exact count declared in advance").

This module reimplements (does not import) the LP row algebra already
validated against production data in the two committed zero-solve
diagnostics `p515_s34_efc_benchmark.py` (committed benchmark,
`data/SRP1/Results/P515S34/EFC_benchmark/`) and
`p515_s35_z2_floor_slackness.py` (committed `Z2` results,
`data/SRP1/Results/P515S35/Z2/`) -- those two scripts are diagnostics and
are never imported by production; this is production code with its own,
independent implementation, cross-checked against both committed outputs
by `p515_s35pt_phase1_checks.py` (Z1, Z2p).

## Formulation (per node, coupled across (year, day); frozen spec v6
   `initialization.lp.formulation`)

Variables, per (year y, representative day d, period p):
    pch[y,d,p] >= 0    -- charging power, MW
    pdch[y,d,p] >= 0   -- discharging power, MW
    soc[y,d,p]         -- state of charge, MWh

Objective (day-weighted, per spec v6: "maximise sum over (year, day,
period) of day-weighted pi*(pdch - pch)"):

    max  sum_{y,d,p} (num_days[d]/365) * pi[y,d,p] * (pdch[y,d,p] - pch[y,d,p])

Rows, each reproducing a specific production constraint (file:line as in
`p515_s34_efc_benchmark.py`'s module docstring, reproduced here):

  [1] power limits        0 <= pch <= S,  0 <= pdch <= S
      network.py:390-391 (`shared_es_pch`/`shared_es_pdch`, NonNegativeReals);
      `S` = the candidate's nameplate rated power for that year (never
      SoH-derated -- `shared_energy_storage_data.py:583`,
      `es_s_available_per_unit == es_s_rated_per_unit`).
  [2] active-power envelope   pch + pdch <= S
      model_construction_helpers.py:827-836 (`sess_active_sum_limit_rule`).
  [3] SoC recursion       soc[p] = soc[p-1] + eff_ch*pch[p]*dt - pdch[p]*dt/eff_dch
      model_construction_helpers.py:893-912 (`sess_soc_rule`);
      dt = HOURS_PER_REPRESENTATIVE_DAY / n_periods
      (model_construction_helpers.py:797-808, `period_duration_hours`,
      reimplemented here as one line to avoid importing that Pyomo-bearing
      module -- see NOTE ON IMPORTS below).
  [4] SoC band            E_soc*MIN <= soc <= E_soc*MAX
      model_construction_helpers.py:840-847 (`sess_soc_lower_limit`,
      `sess_soc_upper_limit`); MIN/MAX = `ENERGY_STORAGE_MIN_ENERGY_STORED`/
      `ENERGY_STORAGE_MAX_ENERGY_STORED` (`definitions.py:38-39`).
      `E_soc` is the candidate's nameplate energy capacity for that year
      when wear is OFF, or the current outer-iteration's SoH-derated
      available energy when wear is ON (`es_e_available_per_unit`,
      `shared_energy_storage_data.py:584`).
  [5] day balance         soc[last_period] == E_soc*INIT
      model_construction_helpers.py:915-921 (`sess_soc_final_rule`);
      INIT = `ENERGY_STORAGE_RELATIVE_INIT_SOC` (`definitions.py:40`).
      Enforced as a hard equality (production allows a small penalized
      slack under `params.slacks.shared_ess.day_balance`; a price-taker
      upper bound never leaves uncompensated slack on a row that, if
      binding, only costs it foregone arbitrage the LP already prices --
      same choice as `p515_s34_efc_benchmark.py`).

  Capacity wear (SoH floor rows, spec v6, only when `wear_on=True`):
      D[y] = 365 * num_years[y] * avg_ch_dch[y] / (2 * cl_eff * E_inv)
      avg_ch_dch[y] = sum_d (num_days[d]/365) * sum_p
                          (eff_ch*pch[y,d,p]*dt + pdch[y,d,p]*dt/eff_dch)
      (`shared_energy_storage_data.py:587-622`,
      `energy_storage_charging_discharging`; `E_inv` = the candidate's
      NAMEPLATE investment for the single active cohort -- the
      degradation law's own denominator never derates,
      `shared_energy_storage_data.py:654`, comment at line 309).
      Linear floor row, for every modelled year Y (task algebra,
      cross-checked algebraically and numerically in
      `p515_s35_z2_floor_slackness.py`, Part B/C):
          sum_{y<=Y} D[y] <= -ln(soh_min) + (sum_{y<=Y} num_years[y]) * ln(phi_cal)
      (`shared_energy_storage_data.py:631-676`,
      `energy_storage_capacity_degradation`).

  Capacity-wear fixed point: `E_soc[y]` in rows [4]/[5] depends on
  SoH[y], which depends on the LP's own dispatch through D[y] above --
  the same same-year self-consistency `p515_s35_z2_floor_slackness.py`
  approximates by fixed-point iteration (its module docstring, section 3),
  preserved here as a FIXED, declared number `outer_iterations` of DAMPED
  outer iterations (spec v6: "capacity wear by a FIXED, declared number K
  of damped outer iterations over SoH_y with convergence asserted (raise
  otherwise)"). Each outer iteration re-solves ONE joint LP per node
  (all years/days/periods at once, floor rows included) at the current
  `E_soc[y]` guess, recomputes `D[y]` / `SoH[y]` / a NEW `E_soc[y]` from
  that solve's own dispatch, and damps the update
  (`E_soc <- damping*E_new + (1-damping)*E_soc`) before the next
  iteration. Every declared iteration always runs (no early exit, for a
  deterministic LP-call count); convergence (`max_y |E_new-E_soc|/E_nameplate
  < convergence_rel_tol`) is asserted AFTER the fixed iteration budget,
  raising `RuntimeError` if not met -- never silently accepted.

## NOTE ON IMPORTS

`definitions.py` has no imports of its own (confirmed by inspection) and
is the only production module this file imports from, for the four named
physical constants above. `model_construction_helpers.py` and
`helper_functions.py` both import `pyomo.environ`
(`helper_functions.py:6`) and are therefore NOT imported here, even
though they own the Pyomo constraint rules this module's math mirrors;
`period_duration_hours` is a one-line function
(`HOURS_PER_REPRESENTATIVE_DAY / n_periods`) and is reproduced inline
instead.

## Guards (spec v6 `initialization.guards_raise_NotImplementedError_if`)

`solve_price_taker_schedule` raises `NotImplementedError` before solving
anything if:
  - more than one market or operation scenario is configured in any
    network (TSO or any DSO), year or day;
  - more than one investment cohort is active (nonzero `s` or `e`) for a
    node's candidate investment;
  - a node's active cohort's calendar-life window does not cover every
    modelled year (block);
  - the market price series differs across networks (TSO, any DSO, the
    ESSO's own copy) for any (year, day) -- a structural invariant in the
    current codebase (`shared_ess_data.cost_energy_p`,
    `transmission_network.cost_energy_p` and every
    `distribution_network.cost_energy_p` are literally the SAME object,
    `shared_resources_planning.py:7225,7268,7313`), checked defensively
    rather than assumed.

Usage (illustrative; see `_initialize_shared_ess_from_price_taker` in
`shared_resources_planning.py` for the actual production call site):

    import shared_ess_price_taker as sept
    result = sept.solve_price_taker_schedule(planning_problem, candidate_solution['investment'])
    result[node_id]['p'][year][day]   # MW, load convention (pch - pdch)
"""

from math import exp, log

import numpy as np
from scipy.optimize import linprog

from definitions import (
    ENERGY_STORAGE_MAX_ENERGY_STORED,
    ENERGY_STORAGE_MIN_ENERGY_STORED,
    ENERGY_STORAGE_RELATIVE_INIT_SOC,
    HOURS_PER_REPRESENTATIVE_DAY,
    SMALL_TOLERANCE,
)

DEFAULT_OUTER_ITERATIONS = 40      # declared, fixed; see module docstring
DEFAULT_DAMPING = 0.5              # damped outer-iteration relaxation factor
DEFAULT_CONVERGENCE_REL_TOL = 1e-9  # checked AFTER the fixed iteration budget

# ---- module-level LP call point, counted (spec v6: "its own armed counter
#      with the exact count declared in advance") ----
_LP_CALL_COUNTER = {'count': 0}


def _solve_lp(*args, **kwargs):
    """Single call point for every `scipy.optimize.linprog` invocation in
    this module. A check harness counts calls by reading
    `_LP_CALL_COUNTER['count']` before/after (or by monkeypatching this
    function's `__wrapped__`/replacing the module attribute)."""
    _LP_CALL_COUNTER['count'] += 1
    return linprog(*args, **kwargs)


def get_lp_call_count():
    return _LP_CALL_COUNTER['count']


def reset_lp_call_count():
    _LP_CALL_COUNTER['count'] = 0


def _period_duration_hours(n_periods):
    """Reproduces `model_construction_helpers.py:797-808`
    (`period_duration_hours`) inline -- see NOTE ON IMPORTS above."""
    if n_periods <= 0:
        raise ValueError('n_periods must be positive.')
    return HOURS_PER_REPRESENTATIVE_DAY / n_periods


# ======================================================================================================================
#  Guards
# ======================================================================================================================
def _iter_all_networks(planning_problem):
    """Yield (role, NetworkData-like container) for the TSO and every DSO."""
    yield 'tso', planning_problem.transmission_network
    for node_id, distribution_network in planning_problem.distribution_networks.items():
        yield f'dso[{node_id}]', distribution_network


def _check_single_market_and_operation_scenario(planning_problem, years, days):
    for role, container in _iter_all_networks(planning_problem):
        for year in years:
            for day in days:
                net = container.network[year][day]
                if len(net.prob_market_scenarios) != 1:
                    raise NotImplementedError(
                        f'shared_ess_price_taker: more than one market scenario '
                        f'({role}, year={year}, day={day}, '
                        f'n={len(net.prob_market_scenarios)}) -- unsupported (spec v6 guard).')
                if len(net.prob_operation_scenarios) != 1:
                    raise NotImplementedError(
                        f'shared_ess_price_taker: more than one operation scenario '
                        f'({role}, year={year}, day={day}, '
                        f'n={len(net.prob_operation_scenarios)}) -- unsupported (spec v6 guard).')


def _check_prices_uniform_across_networks(planning_problem, years, days):
    reference = planning_problem.shared_ess_data.cost_energy_p
    for role, container in _iter_all_networks(planning_problem):
        for year in years:
            for day in days:
                net_prices = np.asarray(container.network[year][day].cost_energy_p)
                ref_prices = np.asarray(reference[year][day])
                if net_prices.shape != ref_prices.shape or not np.array_equal(net_prices, ref_prices):
                    raise NotImplementedError(
                        f'shared_ess_price_taker: market prices differ between network {role} and '
                        f'the shared-ESS reference copy at year={year}, day={day} -- unsupported (spec v6 guard).')


def _active_cohort_index(candidate_investment, node_id, years):
    """Return the (single) 0-based index into `years` of the node's only
    active investment cohort, or None if the candidate invests nothing at
    this node. Raises NotImplementedError if more than one cohort is
    active (spec v6 guard)."""
    active = []
    for y_inv_idx, year in enumerate(years):
        investment = candidate_investment[node_id][year]
        if investment['s'] > SMALL_TOLERANCE or investment['e'] > SMALL_TOLERANCE:
            active.append(y_inv_idx)
    if len(active) > 1:
        raise NotImplementedError(
            f'shared_ess_price_taker: node {node_id} has more than one active investment cohort '
            f'({[years[i] for i in active]}) -- unsupported (spec v6 guard).')
    return active[0] if active else None


def _cohort_window(t_cal, num_years_by_year, years, y_inv_idx):
    """Reproduces the cohort calendar-life window exactly as production
    computes it (`shared_energy_storage_data.py:636-637`,
    `shared_resources_planning.py:2236-2239`)."""
    num_years = num_years_by_year[years[y_inv_idx]]
    tcal_norm = round(t_cal / num_years)
    max_tcal_norm = min(y_inv_idx + tcal_norm, len(years))
    return range(y_inv_idx, max_tcal_norm)


def _check_cohort_window_covers_all_blocks(window, years):
    if list(window) != list(range(len(years))):
        raise NotImplementedError(
            f'shared_ess_price_taker: active cohort calendar window {list(window)} does not cover '
            f'every modelled year/block {list(range(len(years)))} -- unsupported (spec v6 guard).')


# ======================================================================================================================
#  Joint per-node LP (single scipy call; see module docstring, section
#  "Formulation")
# ======================================================================================================================
def _build_and_solve_joint_lp(years, days, n_periods, day_weights, prices_by_year_day,
                               s_nameplate_by_year, e_soc_bound_by_year,
                               eff_ch, eff_dch, dt, include_floor, floor_ctx=None):

    n_years = len(years)
    n_days = len(days)
    block = n_days * n_periods
    n_vars = n_years * block * 3

    def idx_pch(yi, di, p):
        return yi * block * 3 + di * n_periods + p

    def idx_pdch(yi, di, p):
        return yi * block * 3 + block + di * n_periods + p

    def idx_soc(yi, di, p):
        return yi * block * 3 + 2 * block + di * n_periods + p

    c = np.zeros(n_vars)
    A_ub_rows = []
    b_ub = []
    A_eq_rows = []
    b_eq = []
    bounds = [(0.0, 0.0)] * n_vars

    for yi, year in enumerate(years):
        s_max = s_nameplate_by_year[year]
        e_soc = e_soc_bound_by_year[year]
        soc_min = e_soc * ENERGY_STORAGE_MIN_ENERGY_STORED
        soc_max = e_soc * ENERGY_STORAGE_MAX_ENERGY_STORED
        soc_init = e_soc * ENERGY_STORAGE_RELATIVE_INIT_SOC
        for di, day in enumerate(days):
            w = day_weights[day] / 365.0
            prices = prices_by_year_day[year][day]
            for p in range(n_periods):
                pch_i, pdch_i, soc_i = idx_pch(yi, di, p), idx_pdch(yi, di, p), idx_soc(yi, di, p)
                c[pch_i] = w * prices[p]     # minimize +w*price*pch (== maximize -w*price*pch)
                c[pdch_i] = -w * prices[p]   # minimize -w*price*pdch (== maximize +w*price*pdch)
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

    floor_row_offset = None
    if include_floor:
        cl_eff = floor_ctx['cl_eff']
        phi_cal = floor_ctx['phi_cal']
        soh_min = floor_ctx['soh_min']
        num_years_by_year = floor_ctx['num_years_by_year']
        e_inv_nameplate = floor_ctx['e_inv_nameplate']
        rhs_base = -log(soh_min)
        floor_row_offset = len(A_ub_rows)
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

    res = _solve_lp(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError(f'shared_ess_price_taker: joint LP did not solve to optimality: {res.message}')

    pch, pdch, soc = {}, {}, {}
    for yi, year in enumerate(years):
        pch[year], pdch[year], soc[year] = {}, {}, {}
        for di, day in enumerate(days):
            start = idx_pch(yi, di, 0)
            pch[year][day] = np.asarray(res.x[start:start + n_periods])
            start = idx_pdch(yi, di, 0)
            pdch[year][day] = np.asarray(res.x[start:start + n_periods])
            start = idx_soc(yi, di, 0)
            soc[year][day] = np.asarray(res.x[start:start + n_periods])

    floor_multiplier_per_year = {year: 0.0 for year in years}
    if include_floor:
        marginals = res.ineqlin.marginals[floor_row_offset:floor_row_offset + n_years]
        floor_multiplier_per_year = {year: float(-m) for year, m in zip(years, marginals)}

    return {
        'success': bool(res.success), 'status': int(res.status), 'message': res.message,
        'objective': float(-res.fun), 'pch': pch, 'pdch': pdch, 'soc': soc,
        'floor_multiplier_per_year': floor_multiplier_per_year,
    }


def _year_avg_ch_dch(lp, year, days, day_weights, eff_ch, eff_dch, dt):
    """`es_avg_ch_dch_per_unit[y_inv,y]`, `shared_energy_storage_data.py:608-622`."""
    total = 0.0
    for day in days:
        pch = lp['pch'][year][day]
        pdch = lp['pdch'][year][day]
        total += (day_weights[day] / 365.0) * float(np.sum(eff_ch * pch * dt + pdch * dt / eff_dch))
    return total


# ======================================================================================================================
#  Per-node solve (single day-LP when wear is off; damped outer fixed
#  point over a joint LP when wear is on)
# ======================================================================================================================
def _solve_node(years, days, n_periods, day_weights, prices_by_year_day,
                 s_nameplate_by_year, e_nameplate_by_year, eff_ch, eff_dch, dt,
                 efficiency_on, wear_on, cl_eff, phi_cal, soh_min, num_years_by_year,
                 e_inv_nameplate, outer_iterations, damping, convergence_rel_tol):

    use_eff_ch = eff_ch if efficiency_on else 1.0
    use_eff_dch = eff_dch if efficiency_on else 1.0

    if not wear_on:
        # No floor rows couple years/days when wear is off, so the joint
        # formulation is mathematically separable into one LP per (year,
        # day); solved that way explicitly rather than as one large
        # combined LP. This matters numerically, not just for speed: at
        # efficiency_on=False (eff_ch=eff_dch=1.0), simultaneous same-period
        # charge/discharge is a genuine ZERO-COST degenerate direction of
        # the LP (buying and selling the same MWh at the same price nets
        # zero, and cancels in the SoC recursion too), so a large combined
        # LP and a small per-day LP over the SAME feasible region can land
        # HiGHS on different, equally-optimal vertices with different
        # raw (pch+pdch) sums even though the objective and net dispatch
        # (pch-pdch) agree. Solving per (year, day) reproduces the exact
        # LP size/row order `p515_s34_efc_benchmark.py` and
        # `p515_s35_z2_floor_slackness.py` used (verified: this choice
        # reproduces the committed EFC* benchmark to 0.0 max abs diff,
        # `p515_s35pt_phase1_checks.py` Z1; the combined-LP alternative did
        # not, for the same physical dispatch and objective).
        pch, pdch, soc = {}, {}, {}
        status = message = None
        objective_total = 0.0
        for year in years:
            pch[year], pdch[year], soc[year] = {}, {}, {}
            for day in days:
                day_lp = _build_and_solve_joint_lp(
                    [year], [day], n_periods, day_weights, prices_by_year_day,
                    {year: s_nameplate_by_year[year]}, {year: e_nameplate_by_year[year]},
                    use_eff_ch, use_eff_dch, dt, include_floor=False)
                pch[year][day] = day_lp['pch'][year][day]
                pdch[year][day] = day_lp['pdch'][year][day]
                soc[year][day] = day_lp['soc'][year][day]
                status, message = day_lp['status'], day_lp['message']
                objective_total += day_lp['objective']
        lp = {
            'success': True, 'status': status, 'message': message, 'objective': objective_total,
            'pch': pch, 'pdch': pdch, 'soc': soc,
            'floor_multiplier_per_year': {year: 0.0 for year in years},
        }
        soh_per_year = {year: 1.0 for year in years}
        e_available_per_year = dict(e_nameplate_by_year)
        avg_ch_dch_per_year = {
            year: _year_avg_ch_dch(lp, year, days, day_weights, use_eff_ch, use_eff_dch, dt)
            for year in years
        }
        efc_per_day_harness = {
            year: avg_ch_dch_per_year[year] / (2.0 * e_nameplate_by_year[year]) for year in years
        }
        return {
            'lp': lp, 'soh_per_year': soh_per_year, 'e_available_per_year': e_available_per_year,
            'avg_ch_dch_per_year': avg_ch_dch_per_year, 'efc_per_day_harness': efc_per_day_harness,
            'converged': True, 'outer_iterations_run': 1, 'final_rel_change': 0.0,
        }

    if not efficiency_on:
        raise NotImplementedError(
            'shared_ess_price_taker: wear_on=True with efficiency_on=False is not a supported '
            'configuration (the committed Z2 reference only models wear with production efficiencies).')

    e_guess = dict(e_nameplate_by_year)
    lp = None
    avg_ch_dch_per_year = None
    soh_per_year = None
    e_new = None
    rel_change_history = []

    for _ in range(outer_iterations):
        lp = _build_and_solve_joint_lp(
            years, days, n_periods, day_weights, prices_by_year_day,
            s_nameplate_by_year, e_guess, use_eff_ch, use_eff_dch, dt,
            include_floor=True,
            floor_ctx={'cl_eff': cl_eff, 'phi_cal': phi_cal, 'soh_min': soh_min,
                       'num_years_by_year': num_years_by_year, 'e_inv_nameplate': e_inv_nameplate})

        avg_ch_dch_per_year = {
            year: _year_avg_ch_dch(lp, year, days, day_weights, use_eff_ch, use_eff_dch, dt)
            for year in years
        }

        soh_per_year = {}
        prev_soh = 1.0
        for year in years:
            num_years = num_years_by_year[year]
            d_y = 365.0 * num_years * avg_ch_dch_per_year[year] / (2.0 * cl_eff * e_inv_nameplate)
            soh_y = prev_soh * exp(-d_y) * (phi_cal ** num_years)
            soh_per_year[year] = soh_y
            prev_soh = soh_y

        e_new = {year: e_nameplate_by_year[year] * soh_per_year[year] for year in years}
        rel_change = max(
            abs(e_new[year] - e_guess[year]) / max(1e-12, e_nameplate_by_year[year])
            for year in years
        )
        rel_change_history.append(rel_change)

        e_guess = {year: damping * e_new[year] + (1.0 - damping) * e_guess[year] for year in years}

    converged = rel_change_history[-1] < convergence_rel_tol
    if not converged:
        raise RuntimeError(
            f'shared_ess_price_taker: capacity-wear damped fixed point did not converge within '
            f'{outer_iterations} outer iterations (final rel_change={rel_change_history[-1]:.3e}, '
            f'tol={convergence_rel_tol:.1e}). Not silently accepted per spec v6.')

    efc_per_day_harness = {
        year: avg_ch_dch_per_year[year] / (2.0 * e_nameplate_by_year[year]) for year in years
    }

    return {
        'lp': lp, 'soh_per_year': soh_per_year, 'e_available_per_year': e_new,
        'avg_ch_dch_per_year': avg_ch_dch_per_year, 'efc_per_day_harness': efc_per_day_harness,
        'converged': converged, 'outer_iterations_run': outer_iterations,
        'final_rel_change': rel_change_history[-1], 'rel_change_history': rel_change_history,
    }


# ======================================================================================================================
#  Public entry point
# ======================================================================================================================
def solve_price_taker_schedule(planning_problem, candidate_investment, node_ids=None,
                                efficiency_on=True, wear_on=True,
                                outer_iterations=DEFAULT_OUTER_ITERATIONS,
                                damping=DEFAULT_DAMPING,
                                convergence_rel_tol=DEFAULT_CONVERGENCE_REL_TOL):
    """Per-node, coupled-across-(year,day) price-taker schedule. See module
    docstring for the formulation and every parameter's provenance.

    Parameters
    ----------
    planning_problem : SharedResourcesPlanning
        The live planning object (`planning_problem.shared_ess_data`,
        `.transmission_network`, `.distribution_networks`,
        `.active_distribution_network_nodes`, `.years`, `.days` are read;
        nothing is mutated).
    candidate_investment : dict
        `candidate_solution['investment']` exactly as produced by
        `SharedEnergyStorageData.get_candidate_solution` /
        `SharedResourcesPlanning.get_initial_candidate_solution`
        (`node_id -> year -> {'s': MVA, 'e': MWh}`), i.e. the PER-COHORT
        nameplate investment, not the aggregated `total_capacity`.
    node_ids : list or None
        Nodes to solve for; defaults to every active distribution-network
        node (`shared_ess_data.active_distribution_network_nodes`).
    efficiency_on : bool
        False sets eff_ch=eff_dch=1.0 (round-trip loss removed).
    wear_on : bool
        False skips the SoH floor rows and the outer capacity-wear fixed
        point entirely (SoH=1, E_available=nameplate for every year); one
        INDEPENDENT LP per (year, day) is solved (no coupling needed once
        the floor rows are absent -- see `_solve_node` for why this is not
        merely an optimization but a numerically necessary choice at
        efficiency_on=False).

    Returns
    -------
    dict node_id -> {
        'p': {year: {day: np.ndarray[n_periods]}},   # MW, load convention pch-pdch
        'pch', 'pdch', 'soc': same shape,
        'soh_per_year': {year: float}, 'e_available_per_year': {year: float},
        's_nameplate_per_year': {year: float}, 'e_nameplate_per_year': {year: float},
        'efc_per_day_harness': {year: float},         # p514_n_instrumented_cstar.py:133 definition
        'floor_multiplier_per_year': {year: float},
        'lp_status': int, 'lp_message': str, 'lp_objective': float,
        'active_cohort_year': year or None,
        'outer_iterations_run': int, 'converged': bool, 'final_rel_change': float,
        'efficiency_on': bool, 'wear_on': bool,
    }
    """
    shared_ess_data = planning_problem.shared_ess_data
    years = list(shared_ess_data.years)
    days = list(shared_ess_data.days)
    day_weights = {day: shared_ess_data.days[day] for day in days}
    num_years_by_year = {year: shared_ess_data.years[year] for year in years}
    n_periods = shared_ess_data.num_instants
    dt = _period_duration_hours(n_periods)

    if outer_iterations < 1:
        raise ValueError('outer_iterations must be at least 1.')
    if not (0.0 < damping <= 1.0):
        raise ValueError('damping must be in (0, 1].')

    _check_single_market_and_operation_scenario(planning_problem, years, days)
    _check_prices_uniform_across_networks(planning_problem, years, days)

    nodes = list(node_ids) if node_ids is not None else list(shared_ess_data.active_distribution_network_nodes)

    results = {}
    for node_id in nodes:

        y_inv_idx = _active_cohort_index(candidate_investment, node_id, years)
        idx = shared_ess_data.get_shared_energy_storage_idx(node_id)

        s_nameplate_by_year = {year: 0.0 for year in years}
        e_nameplate_by_year = {year: 0.0 for year in years}
        active_cohort_year = None
        e_inv_nameplate = None
        cl_eff = phi_cal = soh_min = t_cal = eff_ch = eff_dch = None

        if y_inv_idx is not None:
            active_cohort_year = years[y_inv_idx]
            ess_obj = shared_ess_data.shared_energy_storages[active_cohort_year][idx]
            t_cal = ess_obj.t_cal
            eff_ch, eff_dch = ess_obj.eff_ch, ess_obj.eff_dch
            cl_eff, phi_cal, soh_min = ess_obj.cl_eff, ess_obj.phi_cal, ess_obj.soh_min

            window = _cohort_window(t_cal, num_years_by_year, years, y_inv_idx)
            _check_cohort_window_covers_all_blocks(window, years)

            s_candidate = candidate_investment[node_id][active_cohort_year]['s']
            e_candidate = candidate_investment[node_id][active_cohort_year]['e']
            e_inv_nameplate = e_candidate
            for y in window:
                s_nameplate_by_year[years[y]] = s_candidate
                e_nameplate_by_year[years[y]] = e_candidate
        else:
            # Zero-investment candidate at this node: trivial all-zero
            # schedule; efficiencies/ageing constants are still needed for
            # bookkeeping, read from the first modelled year uniformly
            # (never used since S/E are zero -> pch=pdch=0 everywhere).
            ess_obj = shared_ess_data.shared_energy_storages[years[0]][idx]
            eff_ch, eff_dch = ess_obj.eff_ch, ess_obj.eff_dch
            cl_eff, phi_cal, soh_min = ess_obj.cl_eff, ess_obj.phi_cal, ess_obj.soh_min
            e_inv_nameplate = 1.0  # placeholder; never divides a nonzero throughput (S=E=0)

        prices_by_year_day = {
            year: {day: np.asarray(shared_ess_data.cost_energy_p[year][day][0], dtype=float) for day in days}
            for year in years
        }

        solved = _solve_node(
            years, days, n_periods, day_weights, prices_by_year_day,
            s_nameplate_by_year, e_nameplate_by_year, eff_ch, eff_dch, dt,
            efficiency_on, wear_on, cl_eff, phi_cal, soh_min, num_years_by_year,
            e_inv_nameplate, outer_iterations, damping, convergence_rel_tol)

        lp = solved['lp']
        p_schedule = {
            year: {day: lp['pch'][year][day] - lp['pdch'][year][day] for day in days}
            for year in years
        }

        results[node_id] = {
            'p': p_schedule, 'pch': lp['pch'], 'pdch': lp['pdch'], 'soc': lp['soc'],
            'soh_per_year': solved['soh_per_year'], 'e_available_per_year': solved['e_available_per_year'],
            's_nameplate_per_year': s_nameplate_by_year, 'e_nameplate_per_year': e_nameplate_by_year,
            'efc_per_day_harness': solved['efc_per_day_harness'],
            'floor_multiplier_per_year': lp['floor_multiplier_per_year'],
            'lp_status': lp['status'], 'lp_message': lp['message'], 'lp_objective': lp['objective'],
            'active_cohort_year': active_cohort_year,
            'outer_iterations_run': solved['outer_iterations_run'], 'converged': solved['converged'],
            'final_rel_change': solved['final_rel_change'],
            'efficiency_on': efficiency_on, 'wear_on': wear_on,
        }

    return results
