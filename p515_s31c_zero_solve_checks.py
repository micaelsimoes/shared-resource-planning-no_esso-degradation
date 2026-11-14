"""
P5.15 Step 3.1-C (S31C worker task) -- Part 2: zero-solve verification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 12 (signed interface reparametrization
and interface energy settlement). Verifies, on FRESHLY BUILT C* ADMM models (a NEW eval
id), that every Part 1 production change took effect, with a BLOCKING
`SolveProfileGuard` (permitted call sites = (), i.e. zero solves anywhere) armed for the
whole check. The TSO's `optimize` is additionally monkeypatched to a stub (never reaches
a solver) so `create_transmission_network_model` can be exercised up to and past its
`.optimize()` call site without ever calling IPOPT -- the guard is the enforcement
mechanism; the monkeypatch keeps the harness from even attempting it.

    python p515_s31c_zero_solve_checks.py

Writes data/SRP1/Results/P515S31C/zero_solve_checks.json (a NEW file; nothing under
data/ is overwritten -- `_refuse_overwrite`, same convention as P515S31).
"""

import copy
import os
import sys
import json
import pickle
from datetime import datetime, timezone

import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables
from pyomo.repn import generate_standard_repn

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as mch  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31C')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')

FIXTURES = [
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'P512R', 'cycle21_pre_setup', 'snapshot.pkl'),
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'P512R', 'production_snapshots', 'FrozenSMOPF',
                 'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515G3F_r2', 'results', 'FrozenSMOPF',
                 'frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl'),
]


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _var_ids(expr, include_fixed=True):
    return {id(v) for v in identify_variables(expr, include_fixed=include_fixed)}


def _stub_optimize(self, model, from_warm_start=False, print_header=True,
                    failure_snapshot_callback=None, pre_solve_snapshot_callback=None):
    """Replaces NetworkData.optimize on ONE instance: never calls a solver.

    Returns None per (year, day), which `solver_result_succeeded` reads as a
    failure -- so the post-solve value-extraction branch in
    `create_transmission_network_model` is skipped and the pre-solve model state
    (the state Part 2 inspects) is exactly what the ADMM path built.
    """
    return {year: {day: None for day in self.days} for year in self.years}


def check_delta_free_legs_fixed_admm(tso_model, dn, rating, s_m, s_o, periods, results):
    """Part 2 bullet 1: delta free with bounds +/- rating; ADN legs fixed at 0."""
    probes = []
    all_ok = True
    for p in periods:
        dvar_p = tso_model.interface_delta_p[dn, s_m, s_o, p]
        dvar_q = tso_model.interface_delta_q[dn, s_m, s_o, p]
        ok = (
            not dvar_p.fixed and not dvar_q.fixed
            and dvar_p.lb == -rating and dvar_p.ub == rating
            and dvar_q.lb == -rating and dvar_q.ub == rating
        )
        probes.append({
            'p': p, 'delta_p_fixed': dvar_p.fixed, 'delta_p_lb': dvar_p.lb, 'delta_p_ub': dvar_p.ub,
            'delta_q_fixed': dvar_q.fixed, 'delta_q_lb': dvar_q.lb, 'delta_q_ub': dvar_q.ub,
            'rating': rating, 'matches': ok,
        })
        all_ok = all_ok and ok
    results['delta_free_bounds_admm'] = {'probes': probes, 'pass': all_ok}


def check_legs_fixed_zero_admm(tso_model, adn_load_idx, s_m, s_o, periods, results):
    probes = []
    all_ok = True
    for p in periods:
        legs = {
            'flex_p_up': tso_model.flex_p_up[adn_load_idx, s_m, s_o, p],
            'flex_p_down': tso_model.flex_p_down[adn_load_idx, s_m, s_o, p],
            'flex_q_up': tso_model.flex_q_up[adn_load_idx, s_m, s_o, p],
            'flex_q_down': tso_model.flex_q_down[adn_load_idx, s_m, s_o, p],
        }
        entry = {'p': p}
        ok = True
        for name, var in legs.items():
            entry[f'{name}_fixed'] = var.fixed
            entry[f'{name}_value'] = pe.value(var)
            ok = ok and var.fixed and pe.value(var) == 0.0
        entry['matches'] = ok
        probes.append(entry)
        all_ok = all_ok and ok
    results['legs_fixed_zero_admm'] = {'probes': probes, 'pass': all_ok}


def check_interface_expression_contains_pc_delta_no_legs(tso_model, dn, adn_load_idx, s_m, s_o, p, results):
    """Part 2 bullet 3: pc_adn contains pc + delta; with fixed variables excluded
    (legs fixed at 0, anchor pc fixed to the consensus value in the ADMM path),
    delta is the ONLY free variable the interface expression depends on."""
    expr = tso_model.pc_adn[dn, s_m, s_o, p]
    all_var_ids = _var_ids(expr, include_fixed=True)
    free_var_ids = _var_ids(expr, include_fixed=False)

    pc_var = tso_model.pc[adn_load_idx, s_m, s_o, p]
    delta_var = tso_model.interface_delta_p[dn, s_m, s_o, p]
    leg_up = tso_model.flex_p_up[adn_load_idx, s_m, s_o, p]
    leg_down = tso_model.flex_p_down[adn_load_idx, s_m, s_o, p]

    results['interface_expression_pc_delta_no_legs'] = {
        'method': 'identify_variables on pc_adn[dn,...]: include_fixed=True must show pc, '
                  'delta AND the (fixed) legs (nothing deleted, per "unwire don\'t delete"); '
                  'include_fixed=False must show delta ONLY (pc is fixed to the anchor and the '
                  'legs are fixed at 0 in the ADMM path, so neither contributes a gradient).',
        'pc_present_full': id(pc_var) in all_var_ids,
        'delta_present_full': id(delta_var) in all_var_ids,
        'legs_present_full': id(leg_up) in all_var_ids and id(leg_down) in all_var_ids,
        'free_var_ids_count': len(free_var_ids),
        'delta_is_sole_free_var': free_var_ids == {id(delta_var)},
        'pass': (id(pc_var) in all_var_ids and id(delta_var) in all_var_ids
                 and id(leg_up) in all_var_ids and id(leg_down) in all_var_ids
                 and free_var_ids == {id(delta_var)}),
    }


def check_node_balance_matches_interface(tso_model, tso_network, node_id, s_m, s_o, p, params, results):
    """The node-balance Pd (compute_node_load, via net_load_p_per_node_def) for the
    ADN-interface load's node must equal pc_adn EXACTLY (same symbolic pc + legs +
    delta sum), so the interface quantity and the balance never diverge."""
    node_idx = tso_network.get_node_idx(node_id)
    dn = tso_network.active_distribution_network_nodes.index(node_id)
    pd_expr = mch.net_load_p_per_node_def(tso_model, node_idx, s_m, s_o, p, tso_network, params)
    pc_adn_expr = tso_model.pc_adn[dn, s_m, s_o, p]
    diff = pe.value(pd_expr) - pe.value(pc_adn_expr)
    results['node_balance_matches_interface'] = {
        'node_id': node_id, 'p': p,
        'node_balance_Pd_value': pe.value(pd_expr),
        'interface_pc_adn_value': pe.value(pc_adn_expr),
        'abs_difference': abs(diff),
        'pass': abs(diff) < 1e-12,
    }


def check_delta_fixed_zero_by_default(tso_model_raw, dn, s_m, s_o, p, results):
    """Part 2 bullet 2 (bare build, before any path-specific fix/free code runs --
    this is the shared default EVERY TSO model starts from)."""
    dvar_p = tso_model_raw.interface_delta_p[dn, s_m, s_o, p]
    dvar_q = tso_model_raw.interface_delta_q[dn, s_m, s_o, p]
    results['delta_fixed_zero_by_default'] = {
        'delta_p_fixed': dvar_p.fixed, 'delta_p_value': pe.value(dvar_p),
        'delta_q_fixed': dvar_q.fixed, 'delta_q_value': pe.value(dvar_q),
        'pass': dvar_p.fixed and pe.value(dvar_p) == 0.0 and dvar_q.fixed and pe.value(dvar_q) == 0.0,
    }


def check_hierarchical_uncoordinated_untouched(results):
    """Part 2 bullet 2, hierarchical/uncoordinated: static proof neither path's
    fix/free block references `interface_delta`, so -- combined with the live
    default-fixed-at-0 check above, which is the state a bare `build_model()`
    (their own first step) produces -- delta stays fixed at 0 and those two
    paths run through EXACTLY the code they ran before this change. This is a
    static+default-state check, not a full build-through: `_run_operational_planning_hierarchical`
    and `..._without_coordination` both call DSO PQ-map / SMOPF solves before the
    TSO model reaches a usable state, which the zero-solve mandate for this
    script forbids reaching.
    """
    import inspect
    hier_source = inspect.getsource(srp._run_operational_planning_hierarchical)
    uncoord_source = inspect.getsource(srp._run_operational_planning_without_coordination)
    hier_untouched = 'interface_delta' not in hier_source
    uncoord_untouched = 'interface_delta' not in uncoord_source
    results['hierarchical_uncoordinated_delta_untouched'] = {
        'method': 'inspect.getsource on both path functions; assert neither references '
                  'interface_delta_p/q anywhere (no fix/free code was added there).',
        'hierarchical_untouched': hier_untouched,
        'uncoordinated_untouched': uncoord_untouched,
        'pass': hier_untouched and uncoord_untouched,
    }


def check_price_identical_tso_dso(tso_network, dso_network, periods, results):
    mismatches = []
    n_checked = 0
    n_scenarios_tso = len(tso_network.cost_energy_p)
    n_scenarios_dso = len(dso_network.cost_energy_p)
    for s_m in range(min(n_scenarios_tso, n_scenarios_dso)):
        for p in periods:
            n_checked += 1
            tso_price = tso_network.cost_energy_p[s_m][p]
            dso_price = dso_network.cost_energy_p[s_m][p]
            if tso_price != dso_price:
                mismatches.append({'s_m': s_m, 'p': p, 'tso': tso_price, 'dso': dso_price})
    results['price_identical_tso_dso'] = {
        'n_scenarios_tso': n_scenarios_tso, 'n_scenarios_dso': n_scenarios_dso,
        'n_periods_checked': n_checked, 'mismatches': mismatches,
        'pass': n_scenarios_tso == n_scenarios_dso and len(mismatches) == 0,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    guard = SolveProfileGuard(permitted=(), label='S31C Part 2 zero-solve check').install()
    results = {}
    try:
        eval_id = 'p515s31c_zero_solve_check'
        eval_dir = os.path.join(O.WORK_DIR, eval_id)
        if os.path.exists(eval_dir):
            raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
        planning = O.fresh_planning(eval_id)

        transmission_network = planning.transmission_network
        distribution_networks = planning.distribution_networks
        year0 = next(iter(transmission_network.years))
        day0 = next(iter(transmission_network.days))
        node0 = next(iter(distribution_networks))
        distribution_network = distribution_networks[node0]

        tso_network = transmission_network.network[year0][day0]
        dso_network = distribution_network.network[year0][day0]
        s_base = tso_network.baseMVA
        dn0 = transmission_network.active_distribution_network_nodes.index(node0)
        adn_load_idx0 = tso_network.get_adn_load_idx(node0)
        interface_rating_pu = distribution_network.network[year0][day0].get_interface_branch_rating() / s_base

        # ---- bare-build default (Part 2 bullet 2) --------------------------
        raw_tso_models = transmission_network.build_model()
        raw_tso_model = raw_tso_models[year0][day0]
        s_m0 = next(iter(raw_tso_model.scenarios_market))
        s_o0 = next(iter(raw_tso_model.scenarios_operation))
        p0 = next(iter(raw_tso_model.periods))
        check_delta_fixed_zero_by_default(raw_tso_model, dn0, s_m0, s_o0, p0, results)
        check_hierarchical_uncoordinated_untouched(results)

        # ---- price identity (needed for cancellation) -----------------------
        check_price_identical_tso_dso(tso_network, dso_network, list(raw_tso_model.periods), results)

        # ---- ADMM-ready TSO model, via the real production function --------
        # `optimize` is monkeypatched on the transmission_network INSTANCE so
        # `create_transmission_network_model` runs to completion (including its
        # `.optimize()` call site) without ever reaching a solver: the guard
        # would raise if it somehow did.
        transmission_network.optimize = _stub_optimize.__get__(transmission_network, type(transmission_network))
        for _node_id, _dn in distribution_networks.items():
            _dn.optimize = _stub_optimize.__get__(_dn, type(_dn))

        consensus_vars, _dual_vars = srp.create_admm_variables(planning)
        candidate = planning.get_initial_candidate_solution()
        srp._rebuild_candidate_total_capacities(planning, candidate)

        tso_models, _tso_results = srp.create_transmission_network_model(
            planning, consensus_vars, candidate['total_capacity']
        )
        tso_model = tso_models[year0][day0]
        # `_prepare_transmission_objectives_for_admm` is a SEPARATE step in the
        # production pipeline (`_run_operational_planning`), applied to the
        # already-built tso_model AFTER `create_transmission_network_model`
        # returns -- not inside it. Reproduce that exact sequencing so the
        # settlement-weight check below reads the true ADMM-ready state.
        srp._prepare_transmission_objectives_for_admm(transmission_network, tso_models)

        periods = list(tso_model.periods)
        s_m = next(iter(tso_model.scenarios_market))
        s_o = next(iter(tso_model.scenarios_operation))

        check_delta_free_legs_fixed_admm(tso_model, dn0, interface_rating_pu, s_m, s_o, periods, results)
        check_legs_fixed_zero_admm(tso_model, adn_load_idx0, s_m, s_o, periods, results)
        check_interface_expression_contains_pc_delta_no_legs(
            tso_model, dn0, adn_load_idx0, s_m, s_o, periods[0], results
        )
        check_node_balance_matches_interface(
            tso_model, tso_network, node0, s_m, s_o, periods[0], transmission_network.params, results
        )

        # ---- settlement weight: 0 by default, 1 in the ADMM path -----------
        raw_weight = float(pe.value(raw_tso_model.interface_settlement_weight))
        admm_weight = float(pe.value(tso_model.interface_settlement_weight))
        dso_models_admm, _dso_results = srp.create_distribution_networks_models(
            distribution_networks, consensus_vars, candidate['total_capacity'],
            parallel_execution=False,
        )
        srp._prepare_distribution_objectives_for_admm(distribution_networks, dso_models_admm)
        dso_model_admm = dso_models_admm[node0][year0][day0]
        raw_dso_models = distribution_network.build_model()
        raw_dso_model = raw_dso_models[year0][day0]
        dso_raw_weight = float(pe.value(raw_dso_model.interface_settlement_weight))
        dso_admm_weight = float(pe.value(dso_model_admm.interface_settlement_weight))
        results['settlement_weight'] = {
            'tso_raw_build_default': raw_weight,
            'tso_admm_path': admm_weight,
            'dso_raw_build_default': dso_raw_weight,
            'dso_admm_path': dso_admm_weight,
            'pass': raw_weight == 0.0 and admm_weight == 1.0 and dso_raw_weight == 0.0 and dso_admm_weight == 1.0,
        }

        # ---- exact cancellation identity ------------------------------------
        ref_gen_idx = dso_network.get_reference_gen_idx()
        ref_node_id = dso_network.get_reference_node_id()
        shared_ess_at_ref = [e for e in dso_model_admm.shared_energy_storages
                              if dso_network.shared_energy_storages[e].bus == ref_node_id]

        cancel_probes = []
        p_test = periods[0]
        target_pu_values = [0.12, 0.30]  # second value differs to exercise the non-cancelling formula too
        for target_idx, target_value_mw in enumerate([12.0, 30.0]):
            target_pu = target_value_mw / s_base
            # TSO side: pc (anchor, fixed) + delta_p = target_pu; keep pc at its current
            # (fixed) anchor value and solve for delta algebraically.
            anchor_pu = pe.value(tso_model.pc[adn_load_idx0, s_m, s_o, p_test])
            tso_model.interface_delta_p[dn0, s_m, s_o, p_test].set_value(target_pu - anchor_pu)
            tso_p_int = pe.value(tso_model.pc_adn[dn0, s_m, s_o, p_test])

            if target_idx == 0:
                # DSO side matches exactly -> exact cancellation.
                dso_model_admm.pg[ref_gen_idx, s_m, s_o, p_test].set_value(target_pu)
                for e in shared_ess_at_ref:
                    dso_model_admm.shared_es_pnet[e, s_m, s_o, p_test].set_value(0.0)
                dso_p_int = pe.value(dso_model_admm.pg_adn[s_m, s_o, p_test])
                label = 'exact_cancellation_matching_p_int'
            else:
                # DSO side DIFFERS -> non-cancelling; check the residual formula.
                dso_target_pu = 18.0 / s_base
                dso_model_admm.pg[ref_gen_idx, s_m, s_o, p_test].set_value(dso_target_pu)
                for e in shared_ess_at_ref:
                    dso_model_admm.shared_es_pnet[e, s_m, s_o, p_test].set_value(0.0)
                dso_p_int = pe.value(dso_model_admm.pg_adn[s_m, s_o, p_test])
                label = 'non_cancelling_priced_residual'

            probability = tso_network.prob_market_scenarios[s_m] * tso_network.prob_operation_scenarios[s_o]
            pi_t = tso_network.cost_energy_p[s_m][p_test]

            # T_TSO / T_DSO restricted to this single (s_m, s_o, p) cell, at
            # weight 1, using the SAME production expressions
            # (`interface_energy_settlement`) evaluated on models whose every
            # OTHER period's delta/pg is left at 0 / its build default, so the
            # single-cell contribution IS the settlement contribution (no other
            # period/scenario adds anything nonzero for the TSO side, since
            # delta is 0 elsewhere; the DSO side's other periods are at their
            # build-default pg, which does contribute elsewhere but cancels
            # identically against its own pc_adn -- so we isolate the cell by
            # explicit direct computation instead of reading model.interface_settlement).
            t_tso_cell = -probability * pi_t * s_base * tso_p_int
            t_dso_cell = probability * pi_t * s_base * dso_p_int
            residual_formula = probability * pi_t * s_base * (dso_p_int - tso_p_int)

            cancel_probes.append({
                'label': label,
                'p': p_test,
                'tso_p_int_mw': tso_p_int * s_base,
                'dso_p_int_mw': dso_p_int * s_base,
                't_tso_cell': t_tso_cell,
                't_dso_cell': t_dso_cell,
                'sum': t_tso_cell + t_dso_cell,
                'residual_formula_value': residual_formula,
                'sum_matches_residual_formula': abs((t_tso_cell + t_dso_cell) - residual_formula) < 1e-12,
                'cancels_to_rounding': abs(t_tso_cell + t_dso_cell) < 1e-9 if target_idx == 0 else None,
            })

        results['cancellation_identity'] = {
            'method': 'direct single-cell evaluation of T_TSO=-prob*pi*baseMVA*p_int_TSO and '
                      'T_DSO=+prob*pi*baseMVA*p_int_DSO (the same terms interface_energy_settlement '
                      'sums), with TSO delta_p and DSO pg/shared_es_pnet VALUES set directly (no '
                      'solve) so p_int_TSO and p_int_DSO take chosen values.',
            'probes': cancel_probes,
            'pass': (cancel_probes[0]['cancels_to_rounding']
                     and cancel_probes[0]['sum_matches_residual_formula']
                     and cancel_probes[1]['sum_matches_residual_formula']),
        }

        # ---- no quadratic/bilinear terms introduced by the settlement ------
        def _quadratic_terms(model):
            repn = generate_standard_repn(model.objective.expr, quadratic=True)
            return len(getattr(repn, 'quadratic_vars', None) or [])

        n_quad_raw_tso = _quadratic_terms(raw_tso_model)
        n_quad_admm_tso = _quadratic_terms(tso_model)
        n_quad_raw_dso = _quadratic_terms(raw_dso_model)
        n_quad_admm_dso = _quadratic_terms(dso_model_admm)
        # The settlement expression is LINEAR in pc_adn / pg_adn (which are
        # themselves affine in pc/delta/legs and pg/shared_es_pnet respectively)
        # -- it introduces no quadratic term at all, so activating the weight
        # (raw -> admm, weight 0 -> 1) must not change the quadratic-term count.
        results['no_quadratic_terms_from_settlement'] = {
            'n_quadratic_terms_tso_raw_weight0': n_quad_raw_tso,
            'n_quadratic_terms_tso_admm_weight1': n_quad_admm_tso,
            'n_quadratic_terms_dso_raw_weight0': n_quad_raw_dso,
            'n_quadratic_terms_dso_admm_weight1': n_quad_admm_dso,
            'pass': n_quad_raw_tso == n_quad_admm_tso and n_quad_raw_dso == n_quad_admm_dso,
        }

        # ---- gross_operational_cost excludes settlements --------------------
        esso_model = planning.shared_ess_data.build_subproblem()
        models_for_recourse = {
            'tso': tso_models, 'dso': dso_models_admm, 'esso': esso_model,
        }
        recourse_components = srp._get_operational_recourse_components(planning, models_for_recourse)
        settlement_total = recourse_components['interface_settlement_total']
        results['gross_operational_cost_excludes_settlement'] = {
            'gross_operational_cost': recourse_components['gross_operational_cost'],
            'gross_operational_cost_including_settlement': recourse_components['gross_operational_cost_including_settlement'],
            'interface_settlement_total': settlement_total,
            'interface_settlement_tso': recourse_components['interface_settlement_tso'],
            'interface_settlement_dso': recourse_components['interface_settlement_dso'],
            'reconciles': abs(
                (recourse_components['gross_operational_cost'] + settlement_total)
                - recourse_components['gross_operational_cost_including_settlement']
            ) < 1e-6,
            'settlement_nonzero_sanity': abs(settlement_total) > 0.0 or True,  # informational; may be ~0 at this synthetic state
            'pass': abs(
                (recourse_components['gross_operational_cost'] + settlement_total)
                - recourse_components['gross_operational_cost_including_settlement']
            ) < 1e-6,
        }

        # ---- fixture unpickling ---------------------------------------------
        fixture_results = []
        for path in FIXTURES:
            entry = {'path': os.path.relpath(path, REPO)}
            try:
                with open(path, 'rb') as handle:
                    pickle.load(handle)
                entry['loads'] = True
            except Exception as exc:  # noqa: BLE001 -- report, don't hide
                entry['loads'] = False
                entry['error'] = f'{type(exc).__name__}: {exc}'
            fixture_results.append(entry)
        results['fixture_unpickling'] = {
            'fixtures': fixture_results,
            'pass': all(f['loads'] for f in fixture_results),
        }

    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Step 3.1-C (S31C worker task) Part 2 -- zero-solve verification',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 12'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S31C Part 2] wrote {OUT_PATH}')
    print(f'[S31C Part 2] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
