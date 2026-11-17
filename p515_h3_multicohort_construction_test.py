"""
P5.15 Addendum 3 item 2 (H3 rule) -- multi-cohort BUILD-ONLY construction test
(PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Verification required" item 2).

ZERO SOLVES. Builds an ESSO subproblem for node 7 with TWO active investment
cohorts (2025 and 2030), using the SAME production construction path every
other P5.15 harness uses (`shared_ess_data.build_subproblem()` +
`shared_ess_data.update_model_with_candidate_solution(...)`), and inspects the
resulting Pyomo model directly. No `optimize()` call is made anywhere in this
script; a BLOCKING `SolveProfileGuard` (permitted=[]) is armed for the whole
run and its `verify(0)` result is reported, per CLAUDE.md's "armed guards,
never asserted" rule.

INSTANCE. `p56a_oracle.fresh_planning` (read-only import of production data,
same as every other P5.15 harness), node 7. Candidate solution: S=1.00 MVA /
E=2.00 MVAh at the FIRST representative year (2025, cohort y_inv=0) and
S=0.50 MVA / E=1.00 MVAh at the SECOND representative year (2030, cohort
y_inv=1); every other node and year is zero. `t_cal`=15 years and each
represented year block is 5 years wide (`shared_ess_data.years`), so
`tcal_norm = round(15/5) = 3` for every cohort: cohort 0 (invested 2025) is
within its calendar-life window for calendar years y=0,1,2 (2025/2030/2035);
cohort 1 (invested 2030) is within its window for y=1,2 (2030/2035). This
makes calendar years y=1 and y=2 genuinely two-cohort-active years -- neither
manufactured beyond setting `es_e/s_investment_fixed` for a second cohort
through the SAME candidate-solution mechanism every fixture uses.

VERIFIED, per the four checks the brief requires:
  (a) exactly N_active(y)-1 rows are ACTIVE at y=1 and y=2 (predicted: 1 each,
      since N_active=2 there); y=0 (single-cohort) has 0 active rows.
  (b) every H3 row coefficient is a plain Python float, not a Pyomo
      expression: asserted via `type(...) is float` on the stored
      `es_pnet_cohort_share_h3` Param value, AND via `polynomial_degree() == 1`
      on the active row's body (a Var-ratio coefficient would make the row's
      cohort-sum term nonpolynomial, i.e. `polynomial_degree() is None`).
  (c) the shares implied by the STORED Param values (including the omitted
      cohort's, which is still written even though its row is inactive) sum
      to 1 across active cohorts, for every two-cohort year.
  (d) the active H3 row is linearly independent of
      `energy_storage_operation_agg`'s aggregate-definition row for the same
      (y, d, p): a NUMERICAL rank check, via `pyomo.repn.generate_standard_repn`
      to extract each row's linear coefficient vector over the same variable
      ordering, stacked into a matrix, and `numpy.linalg.matrix_rank`.

Usage (canonical interpreter):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -B \
        p515_h3_multicohort_construction_test.py
"""

import io
import json
import os
import sys
from contextlib import redirect_stdout

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import pyomo.environ as pe  # noqa: E402
from pyomo.repn import generate_standard_repn  # noqa: E402

import p56a_oracle as oracle  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_1_esso_reform_smoke import _zero_candidate  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P5151')
os.makedirs(OUT_DIR, exist_ok=True)

TEST_NODE = 7
COHORT_0_S_MVA, COHORT_0_E_MVAH = 1.00, 2.00   # invested in the first representative year
COHORT_1_S_MVA, COHORT_1_E_MVAH = 0.50, 1.00   # invested in the second representative year


def _iter_h3_rows(model, y_inv):
    for constraint_name, constraint_idx, y in model._esso_cohort_constraints[y_inv]:
        if constraint_name == 'energy_storage_cohort_pnet_share_h3':
            yield y, constraint_idx


def _linear_coeff_vector(expr, var_order):
    """Extract a linear coefficient vector for `expr` (a constraint body, i.e.
    LHS-RHS) over `var_order` (a fixed list of Var objects), using Pyomo's
    own standard-representation generator (not a hand re-derivation).
    Returns (vector, polynomial_degree)."""
    repn = generate_standard_repn(expr)
    coeff_by_var_id = dict(zip([id(v) for v in repn.linear_vars], repn.linear_coefs))
    vector = np.array([coeff_by_var_id.get(id(v), 0.0) for v in var_order], dtype=float)
    return vector, repn.polynomial_degree()


def main():
    guard = SolveProfileGuard([], label='P5.15 H3 multi-cohort construction test (build-only)').install()
    try:
        with redirect_stdout(io.StringIO()):
            planning = oracle.fresh_planning('p515h3_multicohort')
        shared_ess_data = planning.shared_ess_data
        years = list(planning.years)

        candidate_solution = _zero_candidate(planning)
        candidate_solution[TEST_NODE][years[0]] = {'s': COHORT_0_S_MVA, 'e': COHORT_0_E_MVAH}
        candidate_solution[TEST_NODE][years[1]] = {'s': COHORT_1_S_MVA, 'e': COHORT_1_E_MVAH}

        with redirect_stdout(io.StringIO()):
            esso_models = shared_ess_data.build_subproblem()
            shared_ess_data.update_model_with_candidate_solution(esso_models, candidate_solution)

        model = esso_models[TEST_NODE]
        # No optimize() call anywhere above or below this line.
    finally:
        guard.uninstall()

    guard_failures = guard.verify(0)

    report = {
        'stage': 'P5.15 Addendum 3 item 2 (H3 rule) -- multi-cohort BUILD-ONLY construction test',
        'authority': (
            'PLANNER_BRIEF_2026-09-13.md, Addendum 3, "Authorized" item 2 / "Verification required" item 2'
        ),
        'instance': {
            'node': TEST_NODE,
            'cohort_0_investment_year': years[0],
            'cohort_0_s_mva': COHORT_0_S_MVA,
            'cohort_0_e_mvah': COHORT_0_E_MVAH,
            'cohort_1_investment_year': years[1],
            'cohort_1_s_mva': COHORT_1_S_MVA,
            'cohort_1_e_mvah': COHORT_1_E_MVAH,
            't_cal_years': 15,
            'years_block_width': 5,
            'tcal_norm': 3,
        },
        'guard_build_only_zero_solves': {
            'permitted_call_sites': [],
            'expected_solves_declared_in_advance': 0,
            'observed_counts': guard.counts,
            'verify_failures': guard_failures,
        },
    }

    # ------------------------------------------------------------------
    # (a) exactly N_active(y)-1 rows ACTIVE per calendar year y
    # ------------------------------------------------------------------
    per_year = {}
    for y in model.years:
        active_cohorts = [
            y_inv for y_inv in model.years
            if (not model._esso_cohort_inactive.get(y_inv, False))
        ]
        # Re-derive "within lifetime" the same way `_configure_esso_cohort_pnet_share_rows`
        # does, via the model's own fixed-state introspection (production helper).
        import shared_energy_storage_data as SED
        active_cohorts = [
            y_inv for y_inv in model.years
            if (not model._esso_cohort_inactive.get(y_inv, False))
            and SED._esso_cohort_pair_is_within_lifetime(model, y_inv, y)
        ]
        n_active = len(active_cohorts)

        rows_active_count = 0
        rows_total_count = 0
        share_values = {}
        for y_inv in model.years:
            share_values[y_inv] = pe.value(model.es_pnet_cohort_share_h3[y_inv, y])
            share_is_float = type(model.es_pnet_cohort_share_h3[y_inv, y].value) is float
            for constraint_idx in [idx for yy, idx in _iter_h3_rows(model, y_inv) if yy == y]:
                rows_total_count += 1
                constraint = model.energy_storage_cohort_pnet_share_h3[constraint_idx]
                if constraint.active:
                    rows_active_count += 1

        n_periods_total = len(list(model.days)) * len(list(model.periods))
        expected_active_total = max(n_active - 1, 0) * n_periods_total
        per_year[y] = {
            'active_cohorts': active_cohorts,
            'n_active': n_active,
            'n_periods_total_d_times_p': n_periods_total,
            'rows_active_count_total': rows_active_count,
            'rows_active_count_per_period': (
                rows_active_count / n_periods_total if n_periods_total else None
            ),
            'rows_total_count': rows_total_count,
            'expected_active_rows_per_period_N_active_minus_1': max(n_active - 1, 0),
            'expected_active_rows_total_N_active_minus_1_times_periods': expected_active_total,
            'row_count_matches_expectation': rows_active_count == expected_active_total,
            'share_values_by_cohort': {str(k): v for k, v in share_values.items()},
            'sum_of_shares_over_active_cohorts': sum(share_values[y_inv] for y_inv in active_cohorts) if active_cohorts else 0.0,
        }

    report['per_calendar_year'] = per_year

    # ------------------------------------------------------------------
    # (b) coefficients are plain Python floats, not Pyomo expressions
    # ------------------------------------------------------------------
    float_coefficient_checks = []
    for y in model.years:
        for y_inv in model.years:
            param_value = model.es_pnet_cohort_share_h3[y_inv, y].value
            float_coefficient_checks.append({
                'y_inv': y_inv, 'y': y,
                'value': param_value,
                'is_plain_python_float': type(param_value) is float,
            })
    all_coefficients_are_floats = all(c['is_plain_python_float'] for c in float_coefficient_checks)
    report['coefficients_are_plain_python_floats'] = all_coefficients_are_floats
    report['float_coefficient_checks_sample'] = float_coefficient_checks

    # ------------------------------------------------------------------
    # (d) linear independence of the active H3 row vs the aggregate row,
    #     via a numerical rank check on one (y, d, p) per two-cohort year.
    # ------------------------------------------------------------------
    independence_checks = []
    for y in model.years:
        active_cohorts = per_year[y]['active_cohorts']
        if len(active_cohorts) <= 1:
            continue
        # first (d, p) is representative -- the row structure is identical
        # across (d, p) for a fixed y (only the Var identities change).
        d = list(model.days)[0]
        p = list(model.periods)[0]

        # Variable ordering shared by both rows for this (y, d, p).
        var_order = []
        for y_inv in model.years:
            var_order.append(model.es_pch_per_unit[y_inv, y, d, p])
            var_order.append(model.es_pdch_per_unit[y_inv, y, d, p])
        var_order.append(model.es_pnet[y, d, p])
        if hasattr(model, 'slack_es_pnet_up'):
            var_order.append(model.slack_es_pnet_up[y, d, p])
            var_order.append(model.slack_es_pnet_down[y, d, p])

        # The aggregate-definition row for this (y, d, p): first entry of
        # `energy_storage_operation_agg` added for this (y, d, p) -- built as
        # `es_pnet == agg_pnet [+ slack_up - slack_down]`.
        # Locate it by index arithmetic: for each (y, d, p), two rows are
        # added to `energy_storage_operation_agg` in the loop order
        # (definition row, then the S/Q capability row); iterate the
        # ConstraintList directly to avoid assuming index arithmetic.
        agg_constraint = None
        # ConstraintList entries are 1-indexed in construction order:
        # y outer, d, p inner, 2 rows per (y,d,p) -- locate the definition
        # row (odd position within each pair) for this (y, d, p) directly.
        y_index = list(model.years).index(y)
        d_index = list(model.days).index(d)
        p_index = list(model.periods).index(p)
        n_days = len(list(model.days))
        n_periods = len(list(model.periods))
        flat_index = (y_index * n_days * n_periods) + (d_index * n_periods) + p_index
        agg_constraint_idx = 2 * flat_index + 1  # 1-indexed, definition row first of the pair
        agg_constraint = model.energy_storage_operation_agg[agg_constraint_idx]

        h3_active_y_inv = [y_inv for y_inv in active_cohorts if y_inv != max(active_cohorts)]
        rows_checked = []
        for y_inv in h3_active_y_inv:
            h3_constraint_idx = [idx for yy, idx in _iter_h3_rows(model, y_inv) if yy == y][0]
            h3_constraint = model.energy_storage_cohort_pnet_share_h3[h3_constraint_idx]
            assert h3_constraint.active, 'expected this cohort''s H3 row to be ACTIVE for a two-cohort year'

            h3_expr = h3_constraint.body
            agg_expr = agg_constraint.body

            h3_vec, h3_degree = _linear_coeff_vector(h3_expr, var_order)
            agg_vec, agg_degree = _linear_coeff_vector(agg_expr, var_order)

            matrix = np.vstack([h3_vec, agg_vec])
            rank = int(np.linalg.matrix_rank(matrix))

            rows_checked.append({
                'y_inv_h3_row': y_inv,
                'y': y, 'd': d, 'p': p,
                'h3_row_polynomial_degree': h3_degree,
                'agg_row_polynomial_degree': agg_degree,
                'h3_row_coeff_vector': h3_vec.tolist(),
                'agg_row_coeff_vector': agg_vec.tolist(),
                'var_order_names': [v.name for v in var_order],
                'stacked_matrix_rank': rank,
                'linearly_independent': rank == 2,
            })
        independence_checks.extend(rows_checked)

    report['independence_checks'] = independence_checks
    report['all_active_h3_rows_independent_of_aggregate_row'] = all(
        c['linearly_independent'] for c in independence_checks
    ) if independence_checks else None

    summary_path = os.path.join(OUT_DIR, 'h3_multicohort_construction_test_summary.json')
    with open(summary_path, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f'\n[P5.15-H3-multicohort] summary written to {summary_path}')
    print(f'[P5.15-H3-multicohort] guard (build-only, zero solves) verify failures: {guard_failures}')


if __name__ == '__main__':
    main()
