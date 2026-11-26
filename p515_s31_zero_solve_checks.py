"""
P5.15 Step 3.1 (S31 worker task) -- Part 2: zero-solve verification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 10 ("Sequence after signature") and
P5_15_S31_PENALTY_TABLE_DRAFT.md. Verifies, on FRESHLY BUILT C* models (via
`p56a_oracle.fresh_planning` into a NEW eval id), that every Part 1 production change
took effect, with a BLOCKING `SolveProfileGuard` (permitted call sites = (), i.e. zero
solves anywhere) armed for the whole check.

    python p515_s31_zero_solve_checks.py

Writes data/SRP1/Results/P515S31/zero_solve_checks.json (a NEW file; nothing under
data/ is overwritten -- `_refuse_overwrite`, same convention as the G-gate harness).
"""

import inspect
import json
import os
import sys
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
import shared_energy_storage_data as SED  # noqa: E402
from definitions import PENALTY_ESS_USAGE, EQUALITY_TOLERANCE  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S31')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _var_ids(expr):
    return {id(v) for v in identify_variables(expr)}


def check_flexibility_cost_excludes_adn(tso_model, tso_network, params, results):
    """Row 3 / D1: ADN interface loads must be absent from `flexibility_cost`."""
    s_m = next(iter(tso_model.scenarios_market))
    s_o = next(iter(tso_model.scenarios_operation))
    expr = mch.flexibility_cost(tso_model, tso_network, s_m, s_o, params)
    present_var_ids = _var_ids(expr)

    adn_load_indices = [c for c in tso_model.loads if mch.load_is_tso_adn_interface(tso_network, tso_network.loads[c])]
    non_adn_fl_reg_indices = [
        c for c in tso_model.loads
        if tso_network.loads[c].fl_reg and not mch.load_is_tso_adn_interface(tso_network, tso_network.loads[c])
    ]

    adn_vars_absent = True
    adn_checked = 0
    for c in adn_load_indices:
        for p in tso_model.periods:
            for var in (tso_model.flex_p_down[c, s_m, s_o, p], tso_model.flex_q_down[c, s_m, s_o, p]):
                adn_checked += 1
                if id(var) in present_var_ids:
                    adn_vars_absent = False

    non_adn_present = False
    for c in non_adn_fl_reg_indices:
        for p in tso_model.periods:
            if id(tso_model.flex_p_down[c, s_m, s_o, p]) in present_var_ids or id(tso_model.flex_q_down[c, s_m, s_o, p]) in present_var_ids:
                non_adn_present = True

    results['flexibility_cost_excludes_adn'] = {
        'method': 'identify_variables on the flexibility_cost(...) expression tree; '
                  'assert no flex_p_down/flex_q_down VarData of an ADN-interface load '
                  '(mch.load_is_tso_adn_interface) is among the identified variables, '
                  'and that a non-ADN fl_reg load IS present (expression not trivially empty).',
        'n_adn_loads': len(adn_load_indices),
        'n_adn_vars_checked': adn_checked,
        'adn_vars_absent_from_expression': adn_vars_absent,
        'n_non_adn_fl_reg_loads': len(non_adn_fl_reg_indices),
        'non_adn_present_in_expression': non_adn_present,
        'pass': adn_vars_absent and (len(non_adn_fl_reg_indices) == 0 or non_adn_present),
    }


def check_res_curtailment_weight_zero(tso_model, dso_model, results):
    """Row 5: DSO and TSO penalty_gen_curtailment both 0 after _prepare_*."""
    tso_value = float(pe.value(tso_model.penalty_gen_curtailment))
    dso_value = float(pe.value(dso_model.penalty_gen_curtailment))
    results['res_curtailment_weight_zero'] = {
        'tso_penalty_gen_curtailment': tso_value,
        'dso_penalty_gen_curtailment': dso_value,
        'pass': tso_value == 0.0 and dso_value == 0.0,
    }


def check_ess_usage_split(tso_model, dso_model, results):
    """Row 8: shared-ESS usage Param 0 (after _prepare_*), local-ESS Param PENALTY_ESS_USAGE."""
    tso_shared = float(pe.value(tso_model.penalty_shared_ess_usage))
    tso_local = float(pe.value(tso_model.penalty_ess_usage))
    dso_shared = float(pe.value(dso_model.penalty_shared_ess_usage))
    dso_local = float(pe.value(dso_model.penalty_ess_usage))
    results['ess_usage_penalty_split'] = {
        'tso_penalty_shared_ess_usage': tso_shared,
        'tso_penalty_ess_usage_local': tso_local,
        'dso_penalty_shared_ess_usage': dso_shared,
        'dso_penalty_ess_usage_local': dso_local,
        'expected_local': PENALTY_ESS_USAGE,
        'pass': (tso_shared == 0.0 and dso_shared == 0.0
                 and tso_local == PENALTY_ESS_USAGE and dso_local == PENALTY_ESS_USAGE),
    }


def check_no_bilinear_complementarity(model, network, params, label, results):
    """Row 9: no pch*pdch bilinear term anywhere in the (deactivated) base
    objective -- `admm_objective` is `copy(model.objective.expr)/scale + AL
    terms` (shared_resources_planning.py `update_*_models_to_admm`), so a clean
    `model.objective.expr` implies a clean `admm_objective` too (the AL terms
    added afterwards are quadratic in CONSENSUS residuals, not in pch/pdch
    products -- inspected separately below via `local_ess_day_balance_slack_penalty`
    / `shared_ess_day_balance_slack_penalty`, which are the only pch/pdch-adjacent
    survivors of Step 3 row 9).

    Method: `generate_standard_repn(expr, quadratic=True)` on `model.objective.expr`;
    inspect `repn.quadratic_vars` (list of (var, var) tuples) for any pair whose two
    members are, for the SAME storage index, the pch and pdch variables (local or
    shared ESS).
    """
    repn = generate_standard_repn(model.objective.expr, quadratic=True)
    quadratic_pairs = getattr(repn, 'quadratic_vars', None) or []

    forbidden_pairs = set()
    for e in model.shared_energy_storages:
        for s_m in model.scenarios_market:
            for s_o in model.scenarios_operation:
                for p in model.periods:
                    forbidden_pairs.add(frozenset((id(model.shared_es_pch[e, s_m, s_o, p]), id(model.shared_es_pdch[e, s_m, s_o, p]))))
    if params.es_reg:
        for e in model.energy_storages:
            for s_m in model.scenarios_market:
                for s_o in model.scenarios_operation:
                    for p in model.periods:
                        forbidden_pairs.add(frozenset((id(model.es_pch[e, s_m, s_o, p]), id(model.es_pdch[e, s_m, s_o, p]))))

    found = []
    for var_a, var_b in quadratic_pairs:
        pair = frozenset((id(var_a), id(var_b)))
        if pair in forbidden_pairs:
            found.append((var_a.name, var_b.name))

    results.setdefault('no_bilinear_complementarity', {})[label] = {
        'method': 'generate_standard_repn(model.objective.expr, quadratic=True); '
                  'repn.quadratic_vars inspected for any (pch, pdch) pair of the '
                  'same storage unit (local or shared).',
        'n_quadratic_terms_in_objective': len(quadratic_pairs),
        'n_forbidden_pch_pdch_pairs_checked': len(forbidden_pairs),
        'bilinear_complementarity_pairs_found': found,
        'pass': len(found) == 0,
    }


def check_orphan_slacks(tso_model, tso_network, dso_model, dso_network, results):
    """Row 14 / D2: no Q-balance or ADN-load-P flex slack in the objective, and
    those variables fixed (or absent)."""

    def _check(model, network, label):
        out = {}
        if not hasattr(model, 'slack_flex_q_balance_up'):
            out['variables_present'] = False
            out['pass'] = True
            return out
        out['variables_present'] = True
        s_m = next(iter(model.scenarios_market))
        s_o = next(iter(model.scenarios_operation))
        obj_var_ids = _var_ids(model.objective.expr)

        q_absent_from_objective = True
        q_all_fixed_at_zero = True
        adn_p_absent_from_objective = True
        adn_p_all_fixed_at_zero = True
        non_adn_p_present_in_objective = False
        non_adn_p_not_fixed = False
        checked_q = 0
        checked_adn_p = 0
        checked_non_adn_p = 0

        for c in model.loads:
            is_adn = mch.load_is_tso_adn_interface(network, network.loads[c])
            for s_m_ in model.scenarios_market:
                for s_o_ in model.scenarios_operation:
                    checked_q += 1
                    up_q = model.slack_flex_q_balance_up[c, s_m_, s_o_]
                    down_q = model.slack_flex_q_balance_down[c, s_m_, s_o_]
                    if id(up_q) in obj_var_ids or id(down_q) in obj_var_ids:
                        q_absent_from_objective = False
                    if not (up_q.fixed and down_q.fixed and pe.value(up_q) == 0.0 and pe.value(down_q) == 0.0):
                        q_all_fixed_at_zero = False

                    up_p = model.slack_flex_p_balance_up[c, s_m_, s_o_]
                    down_p = model.slack_flex_p_balance_down[c, s_m_, s_o_]
                    if is_adn:
                        checked_adn_p += 1
                        if id(up_p) in obj_var_ids or id(down_p) in obj_var_ids:
                            adn_p_absent_from_objective = False
                        if not (up_p.fixed and down_p.fixed and pe.value(up_p) == 0.0 and pe.value(down_p) == 0.0):
                            adn_p_all_fixed_at_zero = False
                    elif network.loads[c].fl_reg:
                        checked_non_adn_p += 1
                        if id(up_p) in obj_var_ids or id(down_p) in obj_var_ids:
                            non_adn_p_present_in_objective = True
                        if not up_p.fixed:
                            non_adn_p_not_fixed = True

        out.update({
            'n_checked_q_pairs': checked_q,
            'q_absent_from_objective': q_absent_from_objective,
            'q_all_fixed_at_zero': q_all_fixed_at_zero,
            'n_checked_adn_p_pairs': checked_adn_p,
            'adn_p_absent_from_objective': adn_p_absent_from_objective,
            'adn_p_all_fixed_at_zero': adn_p_all_fixed_at_zero,
            'n_checked_non_adn_p_pairs': checked_non_adn_p,
            'non_adn_p_present_in_objective_sanity': non_adn_p_present_in_objective,
            'non_adn_p_not_fixed_sanity': non_adn_p_not_fixed,
        })
        out['pass'] = (q_absent_from_objective and q_all_fixed_at_zero
                        and adn_p_absent_from_objective and adn_p_all_fixed_at_zero
                        and (checked_non_adn_p == 0 or (non_adn_p_present_in_objective and non_adn_p_not_fixed)))
        return out

    results['orphan_slacks'] = {
        'tso': _check(tso_model, tso_network, 'tso'),
        'dso': _check(dso_model, dso_network, 'dso'),
    }


def check_shared_ess_day_balance_bound(tso_model, results):
    """D6: shared-ESS day-balance slack bounded like the local one, tracking
    the CURRENT candidate capacity."""
    if not hasattr(tso_model, 'slack_shared_es_soc_final_up'):
        results['shared_ess_day_balance_bound'] = {'variables_present': False, 'pass': True}
        return

    e_idx = next(iter(tso_model.shared_energy_storages))
    s_m = next(iter(tso_model.scenarios_market))
    s_o = next(iter(tso_model.scenarios_operation))

    probes = []
    for s_val, e_val in ((1.0, 2.0), (0.0, 0.0), (1.62, 3.24)):
        mch.configure_shared_ess_operational_state(tso_model, e_idx, s_val, e_val)
        var_up = tso_model.slack_shared_es_soc_final_up[e_idx, s_m, s_o]
        var_down = tso_model.slack_shared_es_soc_final_down[e_idx, s_m, s_o]
        expected_ub = 0.0 if (abs(s_val) <= 1e-6 or abs(e_val) <= 1e-6) else e_val * mch.ESS_DAY_BALANCE_SLACK_FRACTION + EQUALITY_TOLERANCE
        probes.append({
            's_capacity': s_val, 'e_capacity': e_val,
            'up_lb': var_up.lb, 'up_ub': var_up.ub,
            'down_lb': var_down.lb, 'down_ub': var_down.ub,
            'expected_ub': expected_ub,
            'matches': (var_up.lb == 0.0 and var_down.lb == 0.0
                        and abs(var_up.ub - expected_ub) < 1e-9 and abs(var_down.ub - expected_ub) < 1e-9),
        })
    # restore to zero (inactive) so this probe doesn't leave the model in a
    # positive-capacity state for any later check in this script.
    mch.configure_shared_ess_operational_state(tso_model, e_idx, 0.0, 0.0)

    results['shared_ess_day_balance_bound'] = {
        'method': 'configure_shared_ess_operational_state at three capacities '
                  '(nonzero, zero/inactive, a different nonzero value), reading '
                  'the Var bounds after each call.',
        'probes': probes,
        'pass': all(p['matches'] for p in probes),
    }


def check_get_feasibility_violation_reads_slacks(planning, results):
    """D3: get_feasibility_violation reads slack_es_pnet_up/down directly."""
    source = inspect.getsource(SED.SharedEnergyStorageData.get_feasibility_violation)
    # Not a bare 'PENALTY_ESSO_SLACK' absence check: the function's own comment
    # legitimately NAMES the old, contaminated divisor to explain why it is no
    # longer used. The code-level check is that the divide-by-PENALTY_ESSO_SLACK
    # EXPRESSION and the old get_feasibility_penalty() indirection are both gone.
    static_ok = ('slack_es_pnet_up' in source and 'slack_es_pnet_down' in source
                 and '/ PENALTY_ESSO_SLACK' not in source
                 and 'get_feasibility_penalty(' not in source)

    sed = planning.shared_ess_data
    esso_models = sed.build_subproblem()
    manual_sum = 0.0
    for node_id in sed.active_distribution_network_nodes:
        model = esso_models[node_id]
        for y_inv in model.years:
            for d in model.days:
                for p in model.periods:
                    manual_sum += pe.value(model.slack_es_pnet_up[y_inv, d, p] + model.slack_es_pnet_down[y_inv, d, p])
    reported = sed.get_feasibility_violation(esso_models)

    results['get_feasibility_violation_reads_slacks'] = {
        'method': 'inspect.getsource static check (no PENALTY_ESSO_SLACK division, '
                  'reads slack_es_pnet_up/down) PLUS a live zero-solve evaluation on '
                  'freshly-built (unsolved, all slacks at their 0.0 initial value) '
                  'ESSO subproblem models, compared with a manual direct sum.',
        'static_source_check_pass': static_ok,
        'manual_sum': manual_sum,
        'reported_by_get_feasibility_violation': reported,
        'live_match': abs(manual_sum - reported) < 1e-12,
        'pass': static_ok and abs(manual_sum - reported) < 1e-12,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    guard = SolveProfileGuard(permitted=(), label='S31 Part 2 zero-solve check').install()
    results = {}
    try:
        eval_id = 'p515s31_zero_solve_check'
        eval_dir = os.path.join(O.WORK_DIR, eval_id)
        if os.path.exists(eval_dir):
            raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
        planning = O.fresh_planning(eval_id)

        transmission_network = planning.transmission_network
        year0 = next(iter(transmission_network.years))
        day0 = next(iter(transmission_network.days))

        node0 = next(iter(planning.distribution_networks))
        distribution_network = planning.distribution_networks[node0]

        tso_models = transmission_network.build_model()
        dso_models = {node0: distribution_network.build_model()}

        # Reach the ADMM-ready state (weight zeroing) with the SAME production
        # functions the ADMM driver calls -- no re-implementation.
        srp._prepare_transmission_objectives_for_admm(transmission_network, tso_models)
        srp._prepare_distribution_objectives_for_admm({node0: distribution_network}, dso_models)

        tso_model = tso_models[year0][day0]
        tso_network = transmission_network.network[year0][day0]
        dso_model = dso_models[node0][year0][day0]
        dso_network = distribution_network.network[year0][day0]

        results['eval_id'] = eval_id
        results['tso_network'] = tso_network.name
        results['dso_network_node0'] = {'node_id': node0, 'name': dso_network.name}

        check_flexibility_cost_excludes_adn(tso_model, tso_network, transmission_network.params, results)
        check_res_curtailment_weight_zero(tso_model, dso_model, results)
        check_ess_usage_split(tso_model, dso_model, results)
        check_no_bilinear_complementarity(tso_model, tso_network, transmission_network.params, 'tso', results)
        check_no_bilinear_complementarity(dso_model, dso_network, distribution_network.params, 'dso', results)
        check_orphan_slacks(tso_model, tso_network, dso_model, dso_network, results)
        check_shared_ess_day_balance_bound(tso_model, results)
        check_get_feasibility_violation_reads_slacks(planning, results)

        # es_reg / local ESS device-count report (Part 1 item 3 obligation:
        # "report whether any SRP1 network has es_reg true").
        es_reg_report = {}
        for name, network_data in (
            [('TSO:' + transmission_network.name, transmission_network)]
            + [(f'DSO:{nid}:{dn.name}', dn) for nid, dn in planning.distribution_networks.items()]
        ):
            sample_year = next(iter(network_data.years))
            sample_day = next(iter(network_data.days))
            sample_network = network_data.network[sample_year][sample_day]
            es_reg_report[name] = {
                'es_reg': network_data.params.es_reg,
                'n_local_energy_storages': len(sample_network.energy_storages),
            }
        results['es_reg_and_local_ess_device_count'] = es_reg_report
    finally:
        guard.uninstall()

    all_pass = all(
        (v.get('pass') if isinstance(v, dict) and 'pass' in v else
         all(vv.get('pass') for vv in v.values()) if isinstance(v, dict) else True)
        for k, v in results.items()
        if k not in ('eval_id', 'tso_network', 'dso_network_node0', 'es_reg_and_local_ess_device_count')
    )

    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Step 3.1 (S31 worker task) Part 2 -- zero-solve verification',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 10', 'P5_15_S31_PENALTY_TABLE_DRAFT.md'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S31 Part 2] wrote {OUT_PATH}')
    print(f'[S31 Part 2] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
