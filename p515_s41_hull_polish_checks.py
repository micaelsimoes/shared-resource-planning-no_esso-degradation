"""
P5.15 Addendum 23 item (2) -- zero-solve checks for `p515_s41_hull_polish.py`.

Authority: frozen spec v12
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json`, item
`item2_hull_polish` ("Checks and smoke test" of the Planner task): *"on an
in-memory built block with synthetic achieved values, the interval bounds
are set on exactly the intended variables at the intended values (including
degenerate intervals and the three-agent ESS case); the certified point lies
inside every hull by construction; objective switching leaves everything
else unchanged; no multiplier suffix is exported."*

A REAL, production-constructed ADMM-ready state (TSO model, DSO models,
ESSO subproblem dict) is built via `p515_s32_zero_solve_checks._build_admm_
ready_state` (BY IMPORT, unchanged -- the existing zero-solve, `.optimize`-
stubbed, production-model-construction pattern this repository already uses
for exactly this purpose). SYNTHETIC "achieved" values are then written
directly onto the relevant Vars (never through a solve) to exercise every
branch of `p515_s41_hull_polish.apply_hull_bounds` by hand: a non-degenerate
2-agent case (V, PF), a degenerate 2-agent case, a non-degenerate 3-agent ESS
case, and a degenerate 3-agent ESS case.

A `SolveProfileGuard(permitted=(), ...)` is armed for the ENTIRE script;
`verify(0)` is checked before writing output. Write-once output under
`data/SRP1/Results/P515S41/hull_polish_checks/`.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s41_hull_polish_checks.py
"""

import json
import os
import sys
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import network as NET  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402 -- BY IMPORT, unchanged
import p515_s41_hull_polish as H  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S41', 'hull_polish_checks')
OUT_PATH = os.path.join(OUT_DIR, 'results.json')
EVAL_ID = 'p515s41_hull_polish_checks_probe'


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _snapshot_vars(model):
    """Every Var's (value, lb, ub, fixed) -- used to prove objective switching
    touches nothing else."""
    out = {}
    for comp in model.component_objects(pe.Var, active=None):
        out[comp.name] = {str(index): (data.value, data.lb, data.ub, data.fixed)
                          for index, data in comp.items()}
    return out


def _snapshot_constraint_active(model):
    out = {}
    for comp in model.component_objects(pe.Constraint, active=None):
        out[comp.name] = {str(index): data.active for index, data in comp.items()}
    return out


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to reuse a non-fresh output dir: {OUT_DIR}')

    guard = SolveProfileGuard(permitted=(), label='P5.15-S41 hull polish checks').install()
    checks = {}
    try:
        planning, tso_model, dso_models, esso_model, consensus_vars, _dual_vars = (
            _build_admm_ready_state(EVAL_ID))
        models = {'tso': tso_model, 'dso': dso_models, 'esso': esso_model}

        tso = planning.transmission_network
        node_ids = sorted(planning.distribution_networks)
        year = next(iter(tso.years))
        day = next(iter(tso.days))
        t_net = tso.network[year][day]

        # ------------------------------------------------------------------
        # Check A: non-degenerate 2-agent (V, PF) + non-degenerate 3-agent
        # (ESS) hull, on the FIRST active node/period.
        # ------------------------------------------------------------------
        node_a = node_ids[0]
        d_net_a = planning.distribution_networks[node_a].network[year][day]
        d_model_a = dso_models[node_a][year][day]
        p0 = next(iter(tso_model[year][day].periods))
        # ESSO subproblems index es_pnet/es_qnet by POSITION into
        # shared_ess_data.years/.days, not by the (year, day) label -- see
        # p515_s41_hull_polish.hull_entries_with_esso's own docstring.
        esso_y_idx = list(planning.shared_ess_data.years).index(year)
        esso_d_idx = list(planning.shared_ess_data.days).index(day)
        dn_a = list(t_net.active_distribution_network_nodes).index(node_a)
        adn_idx_a = t_net.get_node_idx(node_a)
        ref_id_a = d_net_a.get_reference_node_id()
        ref_idx_a = d_net_a.get_node_idx(ref_id_a)
        ref_gen_a = d_net_a.get_reference_gen_idx()
        t_sess_a = [e for e, s in enumerate(t_net.shared_energy_storages) if s.bus == node_a]
        d_sess_a = [e for e, s in enumerate(d_net_a.shared_energy_storages) if s.bus == ref_id_a]

        # Voltage: distinct synthetic values.
        tso_v_synth, dso_v_synth = 1.021, 1.043
        tso_model[year][day].vmag_sqr[adn_idx_a, 0, 0, p0].set_value(tso_v_synth ** 2)
        d_model_a.vmag_sqr[ref_idx_a, 0, 0, p0].set_value(dso_v_synth ** 2)

        # Interface P/Q: distinct synthetic values via the FREE underlying
        # Vars (interface_delta_p/q on the TSO side, pg/shared_es_pnet on the
        # DSO side), so `pc_adn`/`pg_adn` (Expressions) evaluate to distinct
        # numbers -- exactly the achieved-value convention
        # `common_coordinated_values` reads.
        tso_model[year][day].interface_delta_p[dn_a, 0, 0, p0].set_value(0.031)
        tso_model[year][day].interface_delta_q[dn_a, 0, 0, p0].set_value(-0.017)
        adn_load_idx_a = t_net.get_adn_load_idx(node_a)
        tso_pc_before = float(pe.value(tso_model[year][day].pc[adn_load_idx_a, 0, 0, p0]))

        d_model_a.pg[ref_gen_a, 0, 0, p0].set_value(0.5)
        d_model_a.qg[ref_gen_a, 0, 0, p0].set_value(0.1)

        # Shared-ESS P/Q: three distinct synthetic values (TSO copy, DSO copy,
        # ESSO's own achieved dispatch). Set BEFORE reading `pc_adn`/`pg_adn`
        # below, since `pg_adn = pg[ref_gen] - sum(shared_es_pnet)` DEPENDS on
        # the DSO's shared_es_pnet -- reading it earlier would capture a
        # stale value (found while running this check).
        tso_ess_p_synth, dso_ess_p_synth, esso_ess_p_synth = 0.012, 0.031, -0.004
        tso_ess_q_synth, dso_ess_q_synth, esso_ess_q_synth = -0.002, 0.006, 0.009
        for e in t_sess_a:
            tso_model[year][day].shared_es_pnet[e, 0, 0, p0].set_value(tso_ess_p_synth)
            tso_model[year][day].shared_es_qnet[e, 0, 0, p0].set_value(tso_ess_q_synth)
        for e in d_sess_a:
            d_model_a.shared_es_pnet[e, 0, 0, p0].set_value(dso_ess_p_synth)
            d_model_a.shared_es_qnet[e, 0, 0, p0].set_value(dso_ess_q_synth)
        esso_model[node_a].es_pnet[esso_y_idx, esso_d_idx, p0].set_value(esso_ess_p_synth)
        esso_model[node_a].es_qnet[esso_y_idx, esso_d_idx, p0].set_value(esso_ess_q_synth)

        # NOW read the achieved P/Q (pc_adn/pg_adn depend on the ESS values
        # just set), matching exactly what hull_entries_with_esso will read.
        tso_p_achieved = float(pe.value(tso_model[year][day].pc_adn[dn_a, 0, 0, p0]))
        tso_q_achieved = float(pe.value(tso_model[year][day].qc_adn[dn_a, 0, 0, p0]))
        dso_p_achieved = float(pe.value(d_model_a.pg_adn[0, 0, p0]))
        dso_q_achieved = float(pe.value(d_model_a.qg_adn[0, 0, p0]))

        checks['distinct_pf_values'] = {
            'tso_pc_before': tso_pc_before, 'tso_p_achieved': tso_p_achieved,
            'dso_p_achieved': dso_p_achieved, 'tso_q_achieved': tso_q_achieved,
            'dso_q_achieved': dso_q_achieved,
            'p_distinct': tso_p_achieved != dso_p_achieved,
            'q_distinct': tso_q_achieved != dso_q_achieved,
        }

        # ------------------------------------------------------------------
        # Check B: degenerate 2-agent V and a degenerate 3-agent ESS on the
        # SECOND period of the SAME node (independent coordinate).
        # ------------------------------------------------------------------
        periods = list(tso_model[year][day].periods)
        p1 = periods[1] if len(periods) > 1 else periods[0]
        degenerate_v = 1.015
        tso_model[year][day].vmag_sqr[adn_idx_a, 0, 0, p1].set_value(degenerate_v ** 2)
        d_model_a.vmag_sqr[ref_idx_a, 0, 0, p1].set_value(degenerate_v ** 2)
        # 0.25 is an exact binary fraction, so the MW<->pu round-trip below
        # (x100 then /100, baseMVA=100) is bit-exact -- 0.007 was tried first
        # and failed (0.007*100/100 != 0.007 in IEEE-754 double), which is a
        # property of the round-trip THIS TEST introduces, not of
        # apply_hull_bounds's exact-equality degenerate test itself.
        degenerate_ess_p = 0.25
        for e in t_sess_a:
            tso_model[year][day].shared_es_pnet[e, 0, 0, p1].set_value(degenerate_ess_p)
        for e in d_sess_a:
            d_model_a.shared_es_pnet[e, 0, 0, p1].set_value(degenerate_ess_p)
        # es_pnet is in MW (shared_es_pnet above is per-unit); multiply by
        # baseMVA so hull_entries_with_esso's own MW->pu division reproduces
        # EXACTLY degenerate_ess_p, making the interval genuinely degenerate
        # across all three agents, not just the TSO/DSO pair.
        esso_model[node_a].es_pnet[esso_y_idx, esso_d_idx, p1].set_value(
            degenerate_ess_p * t_net.baseMVA)
        # interface_delta_p at 0 (default) on both sides for p1 -> also exercises
        # a degenerate interface-P interval (pc == pg_adn's build default), NOT
        # separately re-set here (left at construction default deliberately,
        # so the degenerate branch is reached through a genuinely unperturbed
        # coordinate, not a contrived equal-by-construction one).

        # ------------------------------------------------------------------
        # `hull_entries_with_esso` + `apply_hull_bounds`
        # ------------------------------------------------------------------
        hull = H.hull_entries_with_esso(planning, models, consensus_vars)
        entry_a_p0 = hull[(node_a, year, day, p0)]
        checks['hull_entry_esso_endpoint'] = {
            'esso_sess_p_pu': entry_a_p0['esso_sess_p'], 'esso_sess_q_pu': entry_a_p0['esso_sess_q'],
            'expected_p_pu': esso_ess_p_synth / t_net.baseMVA,
            'expected_q_pu': esso_ess_q_synth / t_net.baseMVA,
            'matches': (abs(entry_a_p0['esso_sess_p'] - esso_ess_p_synth / t_net.baseMVA) < 1e-12
                       and abs(entry_a_p0['esso_sess_q'] - esso_ess_q_synth / t_net.baseMVA) < 1e-12),
        }

        # Snapshot constraint-list row counts BEFORE, to prove exactly ONE row
        # is added per (TSO, DSO) x (P, Q) x period -- "exactly the intended
        # variables", not more, not fewer.
        n_rows_before = {
            'tso_p': len(getattr(tso_model[year][day], 'p515s41_hull_pf_p_rows', [])),
            'tso_q': len(getattr(tso_model[year][day], 'p515s41_hull_pf_q_rows', [])),
            'dso_p': len(getattr(d_model_a, 'p515s41_hull_pf_p_rows', [])),
            'dso_q': len(getattr(d_model_a, 'p515s41_hull_pf_q_rows', [])),
        }

        descriptors = H.apply_hull_bounds(planning, models, hull)

        n_rows_after = {
            'tso_p': len(tso_model[year][day].p515s41_hull_pf_p_rows),
            'tso_q': len(tso_model[year][day].p515s41_hull_pf_q_rows),
            'dso_p': len(d_model_a.p515s41_hull_pf_p_rows),
            'dso_q': len(d_model_a.p515s41_hull_pf_q_rows),
        }
        n_periods_total = len(tso_model[year][day].periods)
        n_nodes_total = len(planning.distribution_networks)
        # `apply_hull_bounds` loops over EVERY DSO node for a given (year, day)
        # and adds interface rows onto the SHARED t_model each time -- so the
        # TSO's row lists grow by n_nodes * n_periods over the whole call,
        # while a given DSO's own model (specific to ONE node) only grows by
        # n_periods (the pass that processes exactly that node).
        expected_added = {
            'tso_p': n_nodes_total * n_periods_total, 'tso_q': n_nodes_total * n_periods_total,
            'dso_p': n_periods_total, 'dso_q': n_periods_total,
        }
        checks['row_counts'] = {
            'before': n_rows_before, 'after': n_rows_after,
            'n_periods_total': n_periods_total, 'n_nodes_total': n_nodes_total,
            'expected_added': expected_added,
            'matches': all(n_rows_after[k] - n_rows_before[k] == expected_added[k]
                          for k in n_rows_before),
        }

        # --- Voltage bound check (non-degenerate, p0) ---
        lo_v_expected, hi_v_expected = (min(tso_v_synth, dso_v_synth) ** 2,
                                        max(tso_v_synth, dso_v_synth) ** 2)
        tv = tso_model[year][day].vmag_sqr[adn_idx_a, 0, 0, p0]
        dv = d_model_a.vmag_sqr[ref_idx_a, 0, 0, p0]
        checks['voltage_nondegenerate'] = {
            'lo_expected': lo_v_expected, 'hi_expected': hi_v_expected,
            'tso_lb': tv.lb, 'tso_ub': tv.ub, 'tso_fixed': tv.fixed,
            'dso_lb': dv.lb, 'dso_ub': dv.ub, 'dso_fixed': dv.fixed,
            'bounds_match': (abs(tv.lb - lo_v_expected) < 1e-12 and abs(tv.ub - hi_v_expected) < 1e-12
                             and abs(dv.lb - lo_v_expected) < 1e-12 and abs(dv.ub - hi_v_expected) < 1e-12),
            'not_fixed': (not tv.fixed) and (not dv.fixed),
            'certified_point_inside': (lo_v_expected <= tso_v_synth ** 2 <= hi_v_expected
                                       and lo_v_expected <= dso_v_synth ** 2 <= hi_v_expected),
        }

        # --- Voltage bound check (degenerate, p1) ---
        tv1 = tso_model[year][day].vmag_sqr[adn_idx_a, 0, 0, p1]
        dv1 = d_model_a.vmag_sqr[ref_idx_a, 0, 0, p1]
        checks['voltage_degenerate'] = {
            'expected_value': degenerate_v ** 2,
            'tso_fixed': tv1.fixed, 'tso_value': tv1.value,
            'dso_fixed': dv1.fixed, 'dso_value': dv1.value,
            'matches': (tv1.fixed and dv1.fixed
                       and abs(tv1.value - degenerate_v ** 2) < 1e-12
                       and abs(dv1.value - degenerate_v ** 2) < 1e-12),
        }

        # --- Interface P/Q bound check (non-degenerate, p0) — read the row
        # bodies back via the constraint's .lower/.upper/.body, and confirm
        # the CURRENT expression value (unaffected by adding a range row) is
        # still inside [lo, hi]. ---
        lo_p_expected, hi_p_expected = (min(tso_p_achieved, dso_p_achieved),
                                        max(tso_p_achieved, dso_p_achieved))
        # node_a is the FIRST node in sorted(planning.distribution_networks)
        # (the same order apply_hull_bounds iterates) and p0 is the FIRST
        # period -- so node_a/p0's row is the FIRST one added to the TSO's
        # (shared, multi-node) row list (`ConstraintList` is 1-indexed).
        first_tso_p_row = tso_model[year][day].p515s41_hull_pf_p_rows[1]
        checks['interface_p_row'] = {
            'lo_expected': lo_p_expected, 'hi_expected': hi_p_expected,
            'row_lower': float(pe.value(first_tso_p_row.lower)) if first_tso_p_row.lower is not None else None,
            'row_upper': float(pe.value(first_tso_p_row.upper)) if first_tso_p_row.upper is not None else None,
            'current_pc_adn_value': float(pe.value(tso_model[year][day].pc_adn[dn_a, 0, 0, p0])),
            'current_pg_adn_value': float(pe.value(d_model_a.pg_adn[0, 0, p0])),
            'row_matches': (abs(float(pe.value(first_tso_p_row.lower)) - lo_p_expected) < 1e-12
                            and abs(float(pe.value(first_tso_p_row.upper)) - hi_p_expected) < 1e-12),
            'certified_point_inside': (lo_p_expected <= tso_p_achieved <= hi_p_expected
                                       and lo_p_expected <= dso_p_achieved <= hi_p_expected),
        }

        # --- Shared-ESS P (3-agent, non-degenerate, p0). `esso_ess_p_synth`
        # is set in MW on the ESSO's own `es_pnet`; converted to per-unit
        # (dividing by baseMVA, the SAME convention hull_entries_with_esso
        # uses) before comparing against the TSO/DSO copies, which are set
        # directly in per-unit on `shared_es_pnet`. ---
        esso_ess_p_synth_pu = esso_ess_p_synth / t_net.baseMVA
        lo_ep_expected, hi_ep_expected = (min(tso_ess_p_synth, dso_ess_p_synth, esso_ess_p_synth_pu),
                                          max(tso_ess_p_synth, dso_ess_p_synth, esso_ess_p_synth_pu))
        t_ess_var0 = tso_model[year][day].shared_es_pnet[t_sess_a[0], 0, 0, p0]
        d_ess_var0 = d_model_a.shared_es_pnet[d_sess_a[0], 0, 0, p0]
        checks['ess_p_three_agent_nondegenerate'] = {
            'lo_expected': lo_ep_expected, 'hi_expected': hi_ep_expected,
            'esso_ess_p_synth_mw': esso_ess_p_synth, 'esso_ess_p_synth_pu': esso_ess_p_synth_pu,
            'tso_lb': t_ess_var0.lb, 'tso_ub': t_ess_var0.ub, 'tso_fixed': t_ess_var0.fixed,
            'dso_lb': d_ess_var0.lb, 'dso_ub': d_ess_var0.ub, 'dso_fixed': d_ess_var0.fixed,
            'bounds_match': (abs(t_ess_var0.lb - lo_ep_expected) < 1e-12
                             and abs(t_ess_var0.ub - hi_ep_expected) < 1e-12
                             and abs(d_ess_var0.lb - lo_ep_expected) < 1e-12
                             and abs(d_ess_var0.ub - hi_ep_expected) < 1e-12),
            'not_fixed': (not t_ess_var0.fixed) and (not d_ess_var0.fixed),
            'esso_var_untouched': (float(pe.value(esso_model[node_a].es_pnet[esso_y_idx, esso_d_idx, p0]))
                                   == esso_ess_p_synth),
            'certified_point_inside': (lo_ep_expected <= tso_ess_p_synth <= hi_ep_expected
                                       and lo_ep_expected <= dso_ess_p_synth <= hi_ep_expected
                                       and lo_ep_expected <= esso_ess_p_synth_pu <= hi_ep_expected),
        }

        # --- Shared-ESS P (3-agent, degenerate, p1: all three equal) ---
        t_ess_var1 = tso_model[year][day].shared_es_pnet[t_sess_a[0], 0, 0, p1]
        d_ess_var1 = d_model_a.shared_es_pnet[d_sess_a[0], 0, 0, p1]
        checks['ess_p_three_agent_degenerate'] = {
            'expected_value': degenerate_ess_p,
            'tso_fixed': t_ess_var1.fixed, 'tso_value': t_ess_var1.value,
            'dso_fixed': d_ess_var1.fixed, 'dso_value': d_ess_var1.value,
            'matches': (t_ess_var1.fixed and d_ess_var1.fixed
                       and abs(t_ess_var1.value - degenerate_ess_p) < 1e-12
                       and abs(d_ess_var1.value - degenerate_ess_p) < 1e-12),
        }

        # ------------------------------------------------------------------
        # "the certified point lies inside every hull by construction" --
        # check across EVERY descriptor produced (not just the hand-picked
        # ones above), by re-reading each quantity's CURRENT value (the
        # achieved value, unperturbed by adding a bound/constraint) and
        # confirming lo <= value <= hi for all of them.
        # ------------------------------------------------------------------
        containment_failures = []
        for d in descriptors:
            val = d['get_value']()
            if not (d['lo'] - 1e-9 <= val <= d['hi'] + 1e-9):
                containment_failures.append(d)
        checks['all_descriptors_contain_achieved_point'] = {
            'n_descriptors': len(descriptors),
            'n_failures': len(containment_failures),
            'pass': len(containment_failures) == 0,
        }

        # ------------------------------------------------------------------
        # Objective switching leaves everything else unchanged.
        # ------------------------------------------------------------------
        model_for_switch = tso_model[year][day]
        active_objectives_before = sorted(
            o.local_name for o in model_for_switch.component_objects(pe.Objective, active=True,
                                                                      descend_into=False))
        vars_before = _snapshot_vars(model_for_switch)
        constraints_before = _snapshot_constraint_active(model_for_switch)
        H._switch_to_base_objective(model_for_switch)
        vars_after = _snapshot_vars(model_for_switch)
        constraints_after = _snapshot_constraint_active(model_for_switch)
        active_objectives_after = sorted(
            o.local_name for o in model_for_switch.component_objects(pe.Objective, active=True,
                                                                      descend_into=False))
        checks['objective_switch_isolated'] = {
            'active_objectives_before': active_objectives_before,
            'active_objectives_after': active_objectives_after,
            'objective_now_active': active_objectives_after == ['objective'],
            'vars_unchanged': vars_before == vars_after,
            'constraints_unchanged': constraints_before == constraints_after,
        }

        # ------------------------------------------------------------------
        # No multiplier suffix exported. `_clear_multiplier_suffixes` is what
        # `_polish_solve_one_block` calls before every polish solve (that
        # function itself is NOT called here -- it solves); the check
        # exercises the exact same call, zero solves.
        # ------------------------------------------------------------------
        for suffix_name in ('ipopt_zL_in', 'ipopt_zU_in', 'dual'):
            getattr(model_for_switch, suffix_name)[
                model_for_switch.vmag_sqr[adn_idx_a, 0, 0, p0]] = 1.2345
        pre_clear_counts = {name: len(getattr(model_for_switch, name))
                            for name in ('ipopt_zL_in', 'ipopt_zU_in', 'dual')}
        NET._clear_multiplier_suffixes(model_for_switch)
        post_clear_counts = {name: len(getattr(model_for_switch, name))
                             for name in ('ipopt_zL_in', 'ipopt_zU_in', 'dual')}
        checks['no_multiplier_import'] = {
            'pre_clear_counts': pre_clear_counts, 'post_clear_counts': post_clear_counts,
            'all_cleared': all(v == 0 for v in post_clear_counts.values()),
        }

        # ------------------------------------------------------------------
        # `from_warm_start=False` (what `network.run_smopf` is called with in
        # `_polish_solve_one_block`) skips the warm-start block entirely --
        # confirmed by reading `network.py:580` (`if from_warm_start and ...`),
        # zero solves, static check.
        # ------------------------------------------------------------------
        import inspect as _inspect
        source = _inspect.getsource(NET._create_smopf_solver)
        checks['from_warm_start_gate_present'] = {
            'source_contains_gate': 'if from_warm_start and solver_params.solver.lower()' in source,
        }

    finally:
        guard.uninstall()

    failures = guard.verify(expected_solves=0)
    if failures:
        raise RuntimeError(failures)

    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)
    out = {
        'stage': 'P5.15 Addendum 23 item 2 -- hull polish zero-solve checks',
        'authority': ['data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'solve_profile_guard': {'counts': dict(guard.counts), 'verify_failures': failures},
    }
    with open(OUT_PATH, 'w') as handle:
        json.dump(out, handle, indent=1, default=str)
    print(f'[S41-HULL-CHECKS] wrote {OUT_PATH}')

    all_pass = (
        checks['distinct_pf_values']['p_distinct'] and checks['distinct_pf_values']['q_distinct']
        and checks['row_counts']['matches']
        and checks['voltage_nondegenerate']['bounds_match']
        and checks['voltage_nondegenerate']['not_fixed']
        and checks['voltage_nondegenerate']['certified_point_inside']
        and checks['voltage_degenerate']['matches']
        and checks['interface_p_row']['certified_point_inside']
        and checks['interface_p_row']['row_matches']
        and checks['ess_p_three_agent_nondegenerate']['bounds_match']
        and checks['ess_p_three_agent_nondegenerate']['not_fixed']
        and checks['ess_p_three_agent_nondegenerate']['esso_var_untouched']
        and checks['ess_p_three_agent_nondegenerate']['certified_point_inside']
        and checks['ess_p_three_agent_degenerate']['matches']
        and checks['all_descriptors_contain_achieved_point']['pass']
        and checks['objective_switch_isolated']['objective_now_active']
        and checks['objective_switch_isolated']['vars_unchanged']
        and checks['objective_switch_isolated']['constraints_unchanged']
        and checks['no_multiplier_import']['all_cleared']
        and checks['from_warm_start_gate_present']['source_contains_gate']
        and checks['hull_entry_esso_endpoint']['matches']
    )
    print(f'[S41-HULL-CHECKS] ALL PASS = {all_pass}')
    if not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
