"""
P5.15 Addendum 24 item 1 -- zero-solve regression check for the
`p56a_oracle._interface_expression` fix.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 24, frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json`,
`item1_helper_fix.regression_check`:

    "zero-solve: at the certified snapshots available without a new run (and on
    a freshly built TSO block with nonzero interface_delta values), the helper
    equals the model's own interface expression for every (adn node, period,
    p/q), to 0 or round-off; also for the pre-Addendum-12 structure if still
    constructible, state which"

Reuses `p515_s32_zero_solve_checks._build_admm_ready_state` (BY IMPORT,
unchanged) -- the existing zero-solve, production-model-construction,
`.optimize`-stubbed pattern this repository already uses for exactly this
purpose (`p515_s41_hull_polish_checks.py` does the same). No pickled/preserved
snapshot is old enough to postdate the Addendum-12 reparametrization on its
own construction path in a way that would let this check run against a
"certified" fixture without rebuilding the model -- see the `NOTE` at the
bottom of `main()` for exactly what was searched and why the freshly-built
state is used instead.

A `SolveProfileGuard(permitted=(), ...)` is armed for the ENTIRE script;
`verify(0)` is checked before writing output. Write-once output under
`data/SRP1/Results/P515S42/interface_helper_checks/`.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s42_interface_helper_checks.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402 -- BY IMPORT, unchanged
from p515_s40_clone_capture_preflight import _sha256_file  # noqa: E402 -- BY IMPORT, unchanged

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S42', 'interface_helper_checks')
OUT_PATH = os.path.join(OUT_DIR, 'results.json')
EVAL_ID = 'p515s42_interface_helper_checks_probe'

ROUND_OFF_TOL = 1e-9


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _var_ids(expr, include_fixed=True):
    return {id(v) for v in identify_variables(expr, include_fixed=include_fixed)}


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to reuse a non-fresh output dir: {OUT_DIR}')

    guard = SolveProfileGuard(permitted=(), label='P5.15-S42 interface helper checks').install()
    checks = {}
    try:
        planning, tso_model, dso_models, esso_model, consensus_vars, _dual_vars = (
            _build_admm_ready_state(EVAL_ID))
        models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}

        tso = planning.transmission_network
        node_ids = sorted(planning.distribution_networks)

        # ------------------------------------------------------------------
        # Set NONZERO interface_delta_p/q (distinct per node/period, well
        # inside each entry's own +/-rating bound) and NONZERO DSO-side
        # reference-generator dispatch, on EVERY (node, year, day, period) --
        # "nonzero interface_delta values ... for every (adn node, period,
        # p/q)" per the spec, not a single hand-picked coordinate.
        # ------------------------------------------------------------------
        entries = []
        k = 0
        for node in node_ids:
            d_net = planning.distribution_networks[node].network[
                next(iter(tso.years))][next(iter(tso.days))]
            ref_gen = d_net.get_reference_gen_idx()
            for year in tso.years:
                for day in tso.days:
                    t_net = tso.network[year][day]
                    d_net_yd = planning.distribution_networks[node].network[year][day]
                    d_model = dso_models[node][year][day]
                    t_model = tso_model[year][day]
                    dn = list(t_net.active_distribution_network_nodes).index(node)
                    adn_load_idx = t_net.get_adn_load_idx(node)
                    for p in t_model.periods:
                        dvar_p = t_model.interface_delta_p[dn, 0, 0, p]
                        dvar_q = t_model.interface_delta_q[dn, 0, 0, p]
                        rating = dvar_p.ub  # same magnitude as dvar_p.lb (negated) and dvar_q's
                        # 10% of the rating, sign alternating by k, distinct per entry,
                        # comfortably inside [-rating, +rating].
                        sign = 1.0 if (k % 2 == 0) else -1.0
                        delta_p_synth = sign * 0.10 * rating * (1.0 + (k % 7) / 100.0)
                        delta_q_synth = -sign * 0.07 * rating * (1.0 + (k % 5) / 100.0)
                        dvar_p.set_value(delta_p_synth)
                        dvar_q.set_value(delta_q_synth)
                        d_model.pg[ref_gen, 0, 0, p].set_value(0.05 + 0.001 * (k % 11))
                        d_model.qg[ref_gen, 0, 0, p].set_value(0.02 - 0.001 * (k % 9))
                        entries.append((node, year, day, p, dn, adn_load_idx, ref_gen))
                        k += 1
        n_entries = len(entries)

        # ------------------------------------------------------------------
        # Check 1: the FIXED helper equals the model's OWN interface
        # expression (`pc_adn`/`qc_adn`), for EVERY (adn node, year, day,
        # period, p/q).
        # ------------------------------------------------------------------
        new_matches, new_max_abs_diff = 0, 0.0
        for (node, year, day, p, dn, adn_load_idx, ref_gen) in entries:
            t_model = tso_model[year][day]
            for kind in ('p', 'q'):
                helper_val = float(pe.value(O._interface_expression(t_model, dn, p, kind)))
                model_expr = t_model.pc_adn[dn, 0, 0, p] if kind == 'p' else t_model.qc_adn[dn, 0, 0, p]
                model_val = float(pe.value(model_expr))
                diff = abs(helper_val - model_val)
                new_max_abs_diff = max(new_max_abs_diff, diff)
                if diff <= ROUND_OFF_TOL:
                    new_matches += 1
        checks['fixed_helper_equals_model_expression'] = {
            'n_checked': n_entries * 2, 'n_matches': new_matches,
            'max_abs_diff': new_max_abs_diff,
            'exact': new_max_abs_diff == 0.0,
            'note': ('the fixed helper LITERALLY returns t_model.pc_adn/qc_adn '
                     '(same Pyomo object), so equality is exact (0.0), not merely '
                     'round-off -- confirmed numerically above, not merely by '
                     'construction.'),
            'pass': new_matches == n_entries * 2,
        }

        # ------------------------------------------------------------------
        # Check 2: the OLD (pre-Addendum-12) helper differs from the model's
        # own expression by EXACTLY the interface_delta_p/q value (the
        # defect: it omits interface_delta entirely). "for the
        # pre-Addendum-12 structure if still constructible, state which" --
        # it is: `_interface_expression_legacy_pre_addendum12`, unwired, kept
        # callable per CLAUDE.md's deactivate-and-unwire rule.
        # ------------------------------------------------------------------
        old_matches, old_max_abs_diff = 0, 0.0
        old_defect_rows = []
        for (node, year, day, p, dn, adn_load_idx, ref_gen) in entries:
            t_model = tso_model[year][day]
            for kind, delta_attr in (('p', 'interface_delta_p'), ('q', 'interface_delta_q')):
                model_expr = t_model.pc_adn[dn, 0, 0, p] if kind == 'p' else t_model.qc_adn[dn, 0, 0, p]
                model_val = float(pe.value(model_expr))
                old_val = float(pe.value(O._interface_expression_legacy_pre_addendum12(
                    t_model, adn_load_idx, p, kind)))
                delta_val = float(pe.value(getattr(t_model, delta_attr)[dn, 0, 0, p]))
                defect = model_val - old_val
                diff = abs(defect - delta_val)
                old_max_abs_diff = max(old_max_abs_diff, diff)
                if diff <= ROUND_OFF_TOL:
                    old_matches += 1
                else:
                    old_defect_rows.append({'node': node, 'year': year, 'day': day, 'p': p,
                                            'kind': kind, 'defect': defect, 'delta': delta_val})
        checks['old_helper_defect_equals_interface_delta'] = {
            'n_checked': n_entries * 2, 'n_matches': old_matches,
            'max_abs_diff_from_predicted_defect': old_max_abs_diff,
            'exact_or_round_off': ('exact' if old_max_abs_diff == 0.0
                                   else f'round-off (<= {old_max_abs_diff:.3e})'),
            'sample_mismatches': old_defect_rows[:5],
            'pass': old_matches == n_entries * 2,
        }

        # ------------------------------------------------------------------
        # Check 3: free-variable content of the two helpers' expressions, on
        # the FIRST entry (representative) -- the fixed helper's expression
        # depends on EXACTLY interface_delta_p (pc is fixed, legs fixed at 0,
        # per production's ADMM-path construction); the legacy helper's
        # expression depends on NO free variable at all (pc fixed, legs
        # fixed at 0) -- it is a CONSTANT, hence the "zero-gradient,
        # unconditionally infeasible row" WORKER_REPORT_S41_HULL_POLISH_PREP.md
        # section 2 describes.
        # ------------------------------------------------------------------
        node0, year0, day0, p0, dn0, adn_load0, ref_gen0 = entries[0]
        t_model0 = tso_model[year0][day0]
        new_expr0 = O._interface_expression(t_model0, dn0, p0, 'p')
        old_expr0 = O._interface_expression_legacy_pre_addendum12(t_model0, adn_load0, p0, 'p')
        pc_var0 = t_model0.pc[adn_load0, 0, 0, p0]
        delta_var0 = t_model0.interface_delta_p[dn0, 0, 0, p0]
        leg_up0 = t_model0.flex_p_up[adn_load0, 0, 0, p0]
        leg_down0 = t_model0.flex_p_down[adn_load0, 0, 0, p0]
        new_free = _var_ids(new_expr0, include_fixed=False)
        old_free = _var_ids(old_expr0, include_fixed=False)
        checks['free_variable_content'] = {
            'node': node0, 'year': year0, 'day': day0, 'p': p0,
            'pc_fixed': pc_var0.fixed, 'delta_fixed': delta_var0.fixed,
            'leg_up_fixed': leg_up0.fixed, 'leg_down_fixed': leg_down0.fixed,
            'fixed_helper_free_var_ids_count': len(new_free),
            'fixed_helper_free_is_delta_only': new_free == {id(delta_var0)},
            'legacy_helper_free_var_ids_count': len(old_free),
            'legacy_helper_is_constant': len(old_free) == 0,
            'pass': (new_free == {id(delta_var0)}) and (len(old_free) == 0),
        }

        # ------------------------------------------------------------------
        # Check 4: `apply_common_values`, with the FIXED helper, produces a
        # row that is FEASIBLE (satisfiable within `interface_delta_p/q`'s
        # own bounds) at these certified-like synthetic values -- not merely
        # "has a free variable" (check 3) but that the free variable's
        # REQUIRED value to satisfy the row exactly lies inside its bounds.
        # ------------------------------------------------------------------
        common = O.common_coordinated_values(planning, models, consensus_vars,
                                             interface_anchor='midpoint')
        O.apply_common_values(planning, models, common)

        # ConstraintList is 1-indexed; apply_common_values iterates
        # `sorted(distribution_networks)` outer, then (year, day), then
        # `t_model.periods` inner, adding P then Q per period -- so for the
        # FIRST (year, day) pair's t_model, row 1 is (first sorted node,
        # first period, P) and row 2 is the same (node, period)'s Q --
        # exactly `p515_s41_hull_polish_checks.py`'s own documented
        # convention for its sibling ConstraintList.
        first_node = node_ids[0]
        first_year = next(iter(tso.years))
        first_day = next(iter(tso.days))
        t_model_f = tso_model[first_year][first_day]
        first_p = next(iter(t_model_f.periods))
        row_p = t_model_f.p56a_interface_rows[1]
        row_q = t_model_f.p56a_interface_rows[2]

        adn_load_f = tso.network[first_year][first_day].get_adn_load_idx(first_node)
        dn_f = list(tso.network[first_year][first_day].active_distribution_network_nodes).index(first_node)
        pc_fixed_val = float(pe.value(t_model_f.pc[adn_load_f, 0, 0, first_p]))
        qc_fixed_val = float(pe.value(t_model_f.qc[adn_load_f, 0, 0, first_p]))
        common_p_f = common[(first_node, first_year, first_day, first_p)]['common_p']
        common_q_f = common[(first_node, first_year, first_day, first_p)]['common_q']
        required_delta_p = common_p_f - pc_fixed_val
        required_delta_q = common_q_f - qc_fixed_val
        delta_p_var = t_model_f.interface_delta_p[dn_f, 0, 0, first_p]
        delta_q_var = t_model_f.interface_delta_q[dn_f, 0, 0, first_p]

        row_p_free = _var_ids(row_p.body, include_fixed=False)
        row_q_free = _var_ids(row_q.body, include_fixed=False)
        row_p_bounds_ok = (delta_p_var.lb - ROUND_OFF_TOL <= required_delta_p
                           <= delta_p_var.ub + ROUND_OFF_TOL)
        row_q_bounds_ok = (delta_q_var.lb - ROUND_OFF_TOL <= required_delta_q
                           <= delta_q_var.ub + ROUND_OFF_TOL)

        # The row this SAME construction would have produced with the OLD
        # helper: `pc(FIXED) == common_p` -- zero free variables, so
        # "feasible" only by numerical coincidence (pc == common_p exactly),
        # which does not hold here (both sides nonzero and, by construction
        # of the synthetic state, distinct).
        legacy_row_would_be_feasible = abs(pc_fixed_val - common_p_f) <= ROUND_OFF_TOL

        checks['apply_common_values_row_feasible'] = {
            'node': first_node, 'year': first_year, 'day': first_day, 'p': first_p,
            'pc_fixed_val': pc_fixed_val, 'common_p': common_p_f,
            'required_interface_delta_p': required_delta_p,
            'delta_p_bounds': [delta_p_var.lb, delta_p_var.ub],
            'row_p_free_var_is_delta_only': row_p_free == {id(delta_p_var)},
            'row_p_required_value_within_bounds': row_p_bounds_ok,
            'qc_fixed_val': qc_fixed_val, 'common_q': common_q_f,
            'required_interface_delta_q': required_delta_q,
            'delta_q_bounds': [delta_q_var.lb, delta_q_var.ub],
            'row_q_free_var_is_delta_only': row_q_free == {id(delta_q_var)},
            'row_q_required_value_within_bounds': row_q_bounds_ok,
            'legacy_row_would_have_been_feasible_by_construction': legacy_row_would_be_feasible,
            'pass': (row_p_free == {id(delta_p_var)} and row_p_bounds_ok
                     and row_q_free == {id(delta_q_var)} and row_q_bounds_ok
                     and not legacy_row_would_be_feasible),
        }

    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(failures)

    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)
    out = {
        'stage': 'P5.15 Addendum 24 item 1 -- interface-expression helper fix, zero-solve regression check',
        'authority': [
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
            'PLANNER_BRIEF_2026-09-13.md Addendum 24',
        ],
        'note_on_preexisting_fixtures': (
            'the spec asks the check to run "at the certified snapshots available '
            'without a new run" AND "on a freshly built TSO block with nonzero '
            'interface_delta values". This script uses the latter exclusively: no '
            'committed pickle carries a LIVE (unpickled, still-attached-to-Pyomo) TSO '
            'model built on the Addendum-12 ADMM path with a materially nonzero '
            'interface_delta AND a still-solvable state (the preserved P5.12-vintage '
            'fixtures in p515_s31c_zero_solve_checks.FIXTURES predate the interface_delta '
            'reparametrization on the branch that built them, per that file\'s own docstring, '
            'and the certified D run\'s live models are not preserved -- only its JSON report '
            'is). A freshly built, production-constructed, never-solved TSO block '
            '(`p515_s32_zero_solve_checks._build_admm_ready_state`, the same pattern '
            '`p515_s41_hull_polish_checks.py` uses) is therefore the check\'s evidence '
            'base, exactly as the spec\'s parenthetical permits.'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'n_entries_checked': n_entries,
        'checks': checks,
        'solve_profile_guard': {'counts': dict(guard.counts), 'verify_failures': failures},
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(f'[S42-HELPER-CHECKS] wrote {OUT_PATH}')

    manifest = {os.path.relpath(OUT_PATH, REPO): _sha256_file(OUT_PATH)}
    manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-HELPER-CHECKS] wrote {manifest_path}')

    all_pass = (
        checks['fixed_helper_equals_model_expression']['pass']
        and checks['old_helper_defect_equals_interface_delta']['pass']
        and checks['free_variable_content']['pass']
        and checks['apply_common_values_row_feasible']['pass']
    )
    print(f'[S42-HELPER-CHECKS] ALL PASS = {all_pass}')
    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
