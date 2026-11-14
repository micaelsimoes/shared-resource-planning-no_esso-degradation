"""
P5.15 Addendum 49 (Planner task W93) -- CHECKS (tests) for the uncoordinated benchmark: `uncoordinated_benchmark.py`
and the harness `p515_s53_w93_uncoordinated_benchmark.py`.

WRITTEN DURING THE 3x3 PAIR; NOT RUN (Addendum 49 "Timing": "runs nothing -- not even tests that build a model").
Static checks only at W93 and at W94 (W94 added the coupling-check voltage-pin test below). `pytest` is not installed in the canonical environment, so these follow the repository's
self-running `*_checks.py` convention: each check returns {'name', 'passed', 'detail'}; the group passes iff every
check passes; results are written once to <BENCH.OUT_ROOT_REL>/checks_<group>/checks.json (W106: w106_uncoordinated_settled).

GROUPS (one per process; each arms its own SolveProfileGuard before any production import)
  unit         ZERO solves, no SRP1 data. Synthetic Pyomo blocks: the structural check passes on a correct arm and
               FAILS on an unremoved consensus row, an unremoved consensus Param, an undeclared fixed Var, a missing
               Var component and a wrong fixed-row count (negative controls); the common-Q gate comparison is bitwise
               (one ulp FAILS); the perturbation is reproducible from (seed, delta, label), leaves fixed Vars and
               zeros alone and respects bounds; the warm start skips fixed Vars; the consistency detector finds a
               PLANTED hard voltage violation, a planted thermal violation, and a soft violation only when it exceeds
               the arm's own slack; the IPOPT final-summary parser; the single-scenario refusal; the declared solve
               count.
  srp1-zero    ZERO solves (guard permitted=()). SRP1 + the coordinated cell's persisted x = 0 models (W106: the settled
               cycle-181 models, BENCH.COORDINATED; W93/W94: the W86 models, unwired): passive / price-taker DSO arms
               and fixed / tracking-penalty TSO arms pass the structural check against the certified blocks with the
               declared arithmetic; (W94) the 24-solve check's FIXED side carries the interface-voltage pin on every
               block (72 rows, 216 fixed rows in all) at exactly the penalty's V target -- the penalty side's voltage
               term is exactly 0 at the pinned voltages -- while the penalty side and the TSO ARM carry none (144
               fixed rows), with negative controls (a pin planted on the arm FAILS the structural check; a pin under
               the tracking coupling is refused; moving targets on a pinned model is refused); negative controls: production's own `update_distribution_models_to_admm` applied
               to an arm (the real unremoved consensus) and a planted consensus row both FAIL it; the common-Q
               evaluation reproduces the certified gross BITWISE; a wrong configuration (evaluation tie-breaker 1) is
               refused and, repriced, FAILS the gate by exactly the curtailment it prices; the consistency detector
               flags a PLANTED voltage violation on a re-evaluation clone of a certified DSO block (values only, no
               solve) and does not flag the unplanted clone; the lambda helpers return 864 finite rows whose
               Param-derived lambda agrees with the nodal duals.
  srp1-solves  EXACTLY 2 solves (guard bounded to `uncoordinated_benchmark._solve_block`): (i) the consistency
               re-evaluation of ONE certified DSO block (node 7, 2030 Spring) at its own interface voltage reproduces
               its interface flow and triggers nothing; (ii) ONE fixed-interface TSO block (2030 Spring) at the
               certified DSO schedule solves with an interface residual <= 1e-6 MW.

EXACT COMMANDS (W106: output root data/SRP1/Results/P515S53/w106_uncoordinated_settled, BENCH.OUT_ROOT_REL; repo root;
attached, alone, both streams; the W93 root w93_uncoordinated was never created and is unwired):
    mkdir -p data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark_checks.py --group unit > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/checks_unit.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark_checks.py --group srp1-zero > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/checks_srp1_zero.log 2>&1
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w93_uncoordinated_benchmark_checks.py --group srp1-solves > data/SRP1/Results/P515S53/w106_uncoordinated_settled/launch_logs/checks_srp1_solves.log 2>&1
Exit 0 = every check passed; 1 = a check failed; 2 = refused (precondition).
"""

import argparse
import inspect
import os
import sys
import tempfile
import traceback
from copy import deepcopy
from types import SimpleNamespace

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s53_w93_uncoordinated_benchmark as BENCH  # noqa: E402 -- standard library + the campaign harness only

GROUP_SOLVES = {'unit': 0, 'srp1-zero': 0, 'srp1-solves': 2}
DETAIL_NODE, DETAIL_YEAR, DETAIL_DAY = 7, '2030', 'Spring'


def _check(name, fn):
    try:
        passed, detail = fn()
    except Exception as error:  # noqa: BLE001 -- a check that raises unexpectedly FAILS, with the traceback
        passed, detail = False, {'unexpected_exception': f'{type(error).__name__}: {error}',
                                 'traceback': traceback.format_exc()}
    BENCH._log(f"check {name}: {'PASS' if passed else 'FAIL'}")
    return {'name': name, 'passed': bool(passed), 'detail': detail}


def _expect_raises(exc_type, fn, *must_contain):
    try:
        fn()
    except exc_type as error:
        text = str(error)
        return all(s in text for s in must_contain), {'raised': type(error).__name__, 'message': text[:800],
                                                      'must_contain': list(must_contain)}
    return False, {'raised': None, 'expected': exc_type.__name__}


# ======================================================================================================================
#  unit group -- synthetic blocks, zero solves
# ======================================================================================================================
def _row_rule(m, p):
    return m.x[p] + m.y[p] >= 0.0


def _synthetic_block(pe, consensus):
    m = pe.ConcreteModel()
    m.periods = range(3)
    m.x = pe.Var(m.periods, initialize=1.0)
    m.y = pe.Var(m.periods, initialize=0.0)
    m.y[0].fix(0.0)
    m.row = pe.Constraint(m.periods, rule=_row_rule)
    m.objective = pe.Objective(expr=sum(m.x[p] for p in m.periods))
    if consensus:
        m.rho_pf = pe.Param(mutable=True, initialize=1.0)
        m.dual_pf_p_req = pe.Param(m.periods, mutable=True, initialize=0.0)
        m.p_pf_req = pe.Param(m.periods, mutable=True, initialize=0.0)
        m.admm_objective = pe.Objective(expr=sum(m.x[p] + m.dual_pf_p_req[p] * (m.x[p] - m.p_pf_req[p])
                                                 + m.rho_pf / 2 * (m.x[p] - m.p_pf_req[p]) ** 2
                                                 for p in m.periods))
        m.objective.deactivate()
    return m


def _planted_row_rule(m, p):
    return m.x[p] == 0.0


def _target_row_rule(m, p):
    return m.x[p] == m.uncoord_interface_p_target[p]


def unit_checks(UB, pe):
    results = []
    ref = _synthetic_block(pe, consensus=True)
    reference = UB.block_structure(ref)

    def structural_pass():
        arm = _synthetic_block(pe, consensus=False)
        rec = UB.check_arm_block_structure(arm, reference, label='synthetic')
        return (rec['passed'] and rec['removed_consensus_terms'] == ['admm_objective', 'dual_pf_p_req', 'p_pf_req',
                                                                      'rho_pf'], rec)
    results.append(_check('unit_structure_passes_on_consensus_removal', structural_pass))

    def planted_consensus_row():
        arm = _synthetic_block(pe, consensus=False)
        arm.planted_consensus_row = pe.Constraint(arm.periods, rule=_planted_row_rule)
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_block_structure(arm, reference, label='planted row'),
                              "undeclared Constraint component 'planted_consensus_row'", 'active rows 6 != ')
    results.append(_check('unit_NEGATIVE_unremoved_consensus_row_fails', planted_consensus_row))

    def unremoved_consensus_param():
        arm = _synthetic_block(pe, consensus=False)
        arm.dual_pf_p_req = pe.Param(arm.periods, mutable=True, initialize=0.0)
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_block_structure(arm, reference, label='unremoved Param'),
                              "unremoved consensus term Param 'dual_pf_p_req'")
    results.append(_check('unit_NEGATIVE_unremoved_consensus_param_fails', unremoved_consensus_param))

    def unremoved_consensus_objective():
        arm = _synthetic_block(pe, consensus=True)      # the coordinated block itself, offered as an arm
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_block_structure(arm, reference, label='coordinated as arm'),
                              "unremoved consensus term Objective 'admm_objective'", "active objectives ['admm_objective']")
    results.append(_check('unit_NEGATIVE_coordinated_block_offered_as_arm_fails', unremoved_consensus_objective))

    def undeclared_fixed_var():
        arm = _synthetic_block(pe, consensus=False)
        arm.x[1].fix(1.0)
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_block_structure(arm, reference, label='undeclared fix'),
                              'newly fixed Vars differ from the declared set: 1 unexpected')
    results.append(_check('unit_NEGATIVE_undeclared_fixed_var_fails', undeclared_fixed_var))

    def declared_fixed_var_passes():
        arm = _synthetic_block(pe, consensus=False)
        arm.x[1].fix(1.0)
        rec = UB.check_arm_block_structure(arm, reference, label='declared fix', expected_newly_fixed=['x[1]'])
        return rec['passed'] and rec['newly_fixed_vars'] == 1, rec
    results.append(_check('unit_declared_fixed_var_passes', declared_fixed_var_passes))

    def missing_var_component():
        arm = _synthetic_block(pe, consensus=False)
        arm.del_component(arm.row)
        arm.del_component(arm.y)
        arm.row = pe.Constraint(arm.periods, rule=lambda m, p: m.x[p] >= 0.0)
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_block_structure(arm, reference, label='missing Var'),
                              "Var component 'y' of the coordinated block is absent from the arm")
    results.append(_check('unit_NEGATIVE_missing_var_component_fails', missing_var_component))

    def fixed_rows():
        arm = _synthetic_block(pe, consensus=False)
        arm.uncoord_interface_p_target = pe.Param(arm.periods, mutable=True, initialize=0.5)
        arm.uncoord_interface_p_fixed = pe.Constraint(arm.periods, rule=_target_row_rule)
        comps = ('uncoord_interface_p_target', 'uncoord_interface_p_fixed')
        rec = UB.check_arm_block_structure(arm, reference, label='fixed rows', fixed_row_components=comps,
                                           expected_fixed_rows=3)
        ok_wrong, det_wrong = _expect_raises(
            UB.StructuralCheckError,
            lambda: UB.check_arm_block_structure(arm, reference, label='fixed rows wrong', fixed_row_components=comps,
                                                 expected_fixed_rows=2),
            'fixed rows counted 3 != declared 2')
        return rec['passed'] and ok_wrong, {'pass_record': rec['arithmetic'], 'wrong_count': det_wrong}
    results.append(_check('unit_fixed_rows_counted_and_NEGATIVE_wrong_count_fails', fixed_rows))

    def trivial_rows():
        arm = _synthetic_block(pe, consensus=False)
        arm.row[2].deactivate()
        rec = UB.check_arm_block_structure(arm, reference, label='trivial', trivial_rows_deactivated=['row[2]'],
                                           trivial_row_families=('row',))
        ok_wrong, det_wrong = _expect_raises(
            UB.StructuralCheckError,
            lambda: UB.check_arm_block_structure(arm, reference, label='trivial wrong family',
                                                 trivial_rows_deactivated=['row[2]'],
                                                 trivial_row_families=('flex_energy_balance_p',)),
            "deactivated row 'row[2]' is not in the declared trivial-row families")
        return rec['passed'] and ok_wrong, {'pass_record': rec['arithmetic'], 'wrong_family': det_wrong}
    results.append(_check('unit_trivial_rows_declared_and_NEGATIVE_undeclared_family_fails', trivial_rows))

    def gate_bitwise():
        import math
        q_ref = BENCH.COORDINATED['certified_gross']      # W106: the settled Q181 (W93: the W86 Q132 literal)
        certified = {'gross_operational_cost': q_ref, 'interface_settlement_dso': {'5': 1.0, '7': 2.0}}
        same = {'gross_operational_cost': q_ref,
                'recourse_components': {'gross_operational_cost': q_ref,
                                        'interface_settlement_dso': {5: 1.0, 7: 2.0}}}
        one_ulp = math.nextafter(q_ref, math.inf)
        off = {'gross_operational_cost': one_ulp,
               'recourse_components': {'gross_operational_cost': one_ulp, 'interface_settlement_dso': {5: 1.0, 7: 2.0}}}
        g_same = UB.common_q_gate(same, certified)
        g_off = UB.common_q_gate(off, certified)
        passed = (g_same['passed'] and g_same['n_fields_bitwise_equal'] == 2 and not g_off['passed']
                  and g_off['status'] == 'DIFFERS_EXPLANATION_REQUIRED' and g_off['difference'] > 0.0)
        return passed, {'same': g_same['status'], 'one_ulp': g_off['status'], 'one_ulp_difference': g_off['difference']}
    results.append(_check('unit_common_q_gate_bitwise_and_NEGATIVE_one_ulp_fails', gate_bitwise))

    def perturbation():
        def block():
            m = pe.ConcreteModel()
            m.v = pe.Var(range(6), initialize=1.0, bounds=(0.0, 1.02))
            m.v[1].set_value(0.0)
            m.v[2].set_value(None)
            m.v[3].fix(0.7)
            m.v[4].set_value(1.0)
            m.w = pe.Var(initialize=-2.0)
            return m
        a, b, c = block(), block(), block()
        ra = UB.apply_perturbation(a, seed=20260926, delta=0.05, label='DSO|7|2030|Spring')
        rb = UB.apply_perturbation(b, seed=20260926, delta=0.05, label='DSO|7|2030|Spring')
        rc = UB.apply_perturbation(c, seed=20260926, delta=0.05, label='DSO|9|2030|Spring')
        va = [v.value for v in a.component_data_objects(pe.Var, sort=True)]
        vb = [v.value for v in b.component_data_objects(pe.Var, sort=True)]
        vc = [v.value for v in c.component_data_objects(pe.Var, sort=True)]
        in_bounds = all(v.value is None or (v.lb is None or v.value >= v.lb) and (v.ub is None or v.value <= v.ub)
                        for v in a.component_data_objects(pe.Var))
        bad_seed, _d1 = _expect_raises(ValueError, lambda: UB.apply_perturbation(block(), seed=-1, delta=0.05,
                                                                                  label='x'))
        bad_delta, _d2 = _expect_raises(ValueError, lambda: UB.apply_perturbation(block(), seed=1, delta=5,
                                                                                   label='x'))
        passed = (va == vb and va != vc and a.v[1].value == 0.0 and a.v[2].value is None and a.v[3].value == 0.7
                  and a.v[3].fixed and in_bounds and ra['n_draws'] == 6 and ra == rb and bad_seed and bad_delta)
        return passed, {'a': va, 'c': vc, 'record': ra, 'other_label_record': rc}
    results.append(_check('unit_perturbation_reproducible_bounded_fixed_untouched', perturbation))

    def warm_start():
        m = pe.ConcreteModel()
        m.v = pe.Var(range(3), initialize=0.0, bounds=(0.0, 1.0))
        m.v[0].fix(0.25)
        rec = UB.apply_warm_values(m, {'v[0]': 0.9, 'v[1]': 0.5, 'v[3]': 7.0})
        passed = (m.v[0].value == 0.25 and m.v[1].value == 0.5 and m.v[2].value == 0.0 and rec['n_set'] == 1
                  and rec['n_skipped_fixed'] == 1 and rec['n_missing'] == 1)
        return passed, rec
    results.append(_check('unit_warm_start_skips_fixed_counts_missing', warm_start))

    def detector():
        idx = [(i, 0, 0, p) for i in range(2) for p in range(2)]

        def build():
            m = pe.ConcreteModel()
            m.vmag_sqr = pe.Var(idx, initialize=1.0)
            m.slack_v_sqr_down = pe.Var(idx, initialize=0.0)
            m.slack_v_sqr_up = pe.Var(idx, initialize=0.0)
            for v in list(m.slack_v_sqr_down.values()) + list(m.slack_v_sqr_up.values()):
                v.fix(0.0)
            m.voltage_magnitude_lower_cons = pe.Constraint(
                idx, rule=lambda m, i, a, b, p: m.vmag_sqr[i, a, b, p] + m.slack_v_sqr_down[i, a, b, p] >= 0.9 ** 2)
            m.voltage_magnitude_upper_cons = pe.Constraint(
                idx, rule=lambda m, i, a, b, p: m.vmag_sqr[i, a, b, p] - m.slack_v_sqr_up[i, a, b, p] <= 1.1 ** 2)
            m.flow = pe.Var(initialize=1.0)
            m.branch_flow_limit = pe.Constraint(range(1), rule=lambda m, b: m.flow <= 4.0)
            for c in m.component_data_objects(pe.Constraint):
                c.deactivate()
            record = {'original_vmag_sqr_bounds': {v.name: (0.85 ** 2, 1.15 ** 2) for v in m.vmag_sqr.values()},
                      'original_ref_gen_bounds': {}}
            arm = pe.ConcreteModel()
            arm.slack_v_sqr_down = pe.Var(idx, initialize=0.0)
            arm.slack_v_sqr_up = pe.Var(idx, initialize=0.0)
            return m, record, arm
        tol = dict(hard_tol=1e-6, soft_excess_tol=1e-6, thermal_tol=1e-6)
        m, record, arm = build()
        clean = UB.consistency_violations(m, record, arm, **tol)
        m.vmag_sqr[1, 0, 0, 1].set_value(1.2 ** 2)
        planted = UB.consistency_violations(m, record, arm, **tol)
        m2, record2, arm2 = build()
        m2.vmag_sqr[0, 0, 0, 0].set_value(1.12 ** 2)
        arm2.slack_v_sqr_up[0, 0, 0, 0].set_value(1.12 ** 2 - 1.1 ** 2)
        own = UB.consistency_violations(m2, record2, arm2, **tol)
        arm2.slack_v_sqr_up[0, 0, 0, 0].set_value(0.04)
        deeper = UB.consistency_violations(m2, record2, arm2, **tol)
        m3, record3, arm3 = build()
        m3.flow.set_value(4.5)
        thermal = UB.consistency_violations(m3, record3, arm3, **tol)
        passed = (not clean['trigger_sequential_pass']
                  and planted['trigger_sequential_pass'] and len(planted['hard']) == 1
                  and abs(planted['max_hard_excess_pu2'] - (1.44 - 1.15 ** 2)) < 1e-12
                  and not own['trigger_sequential_pass'] and own['max_soft_excess_pu2'] < 1e-12
                  and deeper['trigger_sequential_pass'] and deeper['max_soft_excess_pu2'] > 1e-3
                  and thermal['trigger_sequential_pass'] and abs(thermal['max_thermal_violation_pu2'] - 0.5) < 1e-12
                  and not thermal['hard'])
        return passed, {'clean': clean['trigger_sequential_pass'], 'planted_hard': planted['max_hard_excess_pu2'],
                        'own_slack_soft_excess': own['max_soft_excess_pu2'],
                        'deeper_soft_excess': deeper['max_soft_excess_pu2'],
                        'thermal': thermal['max_thermal_violation_pu2']}
    results.append(_check('unit_consistency_detector_finds_PLANTED_violations', detector))

    def ipopt_summary():
        text = ('This is Ipopt version 3.14.18, running with linear solver ma97.\n\n'
                'Number of Iterations....: 23\n\n'
                '                                   (scaled)                 (unscaled)\n'
                'Objective...............:   1.2345678901234567e+02    1.2345678901234567e+05\n'
                'Dual infeasibility......:   3.4567890123456789e-09    3.4567890123456789e-06\n'
                'Constraint violation....:   1.0000000000000000e-10    1.0000000000000000e-10\n'
                'Variable bound violation:   0.0000000000000000e+00    0.0000000000000000e+00\n'
                'Complementarity.........:   2.5059035596800626e-09    2.5059035596800626e-06\n'
                'Overall NLP error.......:   3.4567890123456789e-09    3.4567890123456789e-06\n\n'
                'EXIT: Optimal Solution Found.\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'attempt.log')
            with open(path, 'w') as handle:
                handle.write('PREVIOUS ATTEMPT\nDual infeasibility......:   9.0e+00    9.0e+00\n')
                start = handle.tell()
                handle.write(text)
                end = handle.tell()
            got = UB.read_ipopt_attempt_summary(path, [start, end])
            empty = UB.read_ipopt_attempt_summary(path, [0, 10])
        passed = (got['unscaled'].get('dual_infeasibility') == 3.4567890123456789e-06
                  and got['scaled'].get('overall_nlp_error') == 3.4567890123456789e-09
                  and got['unscaled'].get('variable_bound_violation') == 0.0 and got['parse_reason'] is None
                  and empty['parse_reason'] is not None)
        return passed, {'parsed': got, 'empty_segment': empty['parse_reason']}
    results.append(_check('unit_ipopt_final_summary_parsed_from_the_attempt_segment_only', ipopt_summary))

    def single_scenario_and_counts():
        def holder(n_m, n_o, name):
            net = SimpleNamespace(prob_market_scenarios=[1.0 / n_m] * n_m, prob_operation_scenarios=[1.0 / n_o] * n_o)
            years = {'2025': 5, '2030': 5, '2035': 5}
            days = {'Spring': 92, 'Summer': 91, 'Autumn': 91, 'Winter': 91}
            return SimpleNamespace(name=name, years=years, days=days,
                                   network={y: {d: net for d in days} for y in years})
        ok_planning = SimpleNamespace(transmission_network=holder(1, 1, 'tn'),
                                      distribution_networks={n: holder(1, 1, f'dn{n}') for n in (5, 7, 9)})
        bad_planning = SimpleNamespace(transmission_network=holder(1, 1, 'tn'),
                                       distribution_networks={5: holder(2, 1, 'dn5')})
        UB.require_single_scenario(ok_planning)
        refused, detail = _expect_raises(NotImplementedError, lambda: UB.require_single_scenario(bad_planning),
                                         'single-scenario')
        counts = UB.declared_solve_count(ok_planning)
        return refused and counts == {'dso': 36, 'tso': 12, 'total': 48}, {'refusal': detail, 'counts': counts}
    results.append(_check('unit_single_scenario_refusal_and_declared_solve_count', single_scenario_and_counts))
    return results


# ======================================================================================================================
#  srp1-zero group -- SRP1 + the coordinated (W106: settled cycle-181) persisted models, zero solves
# ======================================================================================================================
def srp1_zero_checks(srp, UB, O, pe):
    results = []
    planning = deepcopy(O.load_baseline()['planning'])     # builders mutate network data: a private copy
    candidate = BENCH._x0_candidate(srp, planning)
    record = BENCH._load_json(BENCH.COORDINATED['evaluation_record'])     # W106 (W93: BENCH.W86, unwired)
    models = BENCH._load_coordinated_models()                           # W106 (W93: _load_w86_models, unwired)
    reference = UB.coordinated_reference_structure(planning, models)
    targets = UB.get_dso_interface_schedule(planning, models['dso'])
    cand_tc = candidate['total_capacity']

    def passive_structure():
        dso, build = UB.build_dso_arm_models(planning, cand_tc, arm=UB.ARM_PASSIVE, curtailment_penalty=1.0)
        recs = UB.check_arm_structures(planning, {'dso': dso}, reference, dso_build_record=build)
        details = {}
        ok = len(recs) == 36
        for label, rec in recs.items():
            node_id, year, day = int(label.split('|')[1]), label.split('|')[2], label.split('|')[3]
            block = dso[node_id][year][day]
            n_flex = 4 * len(block.loads) * len(block.scenarios_market) * len(block.scenarios_operation) * len(block.periods)
            n_trivial = len(block.flex_energy_balance_p) if hasattr(block, 'flex_energy_balance_p') else 0
            ok = ok and rec['newly_fixed_vars'] == n_flex and rec['trivial_rows_deactivated'] == n_trivial
            ok = ok and pe.value(block.interface_settlement_weight) == 0.0 and pe.value(block.penalty_gen_curtailment) == 1.0
            ok = ok and rec['interface_voltage_pin']['pinned']
            details[label] = rec['arithmetic']
        return ok, details
    results.append(_check('srp1_passive_dso_structure_passes_with_declared_arithmetic', passive_structure))

    def price_taker_structure():
        dso, build = UB.build_dso_arm_models(planning, cand_tc, arm=UB.ARM_PRICE_TAKER, curtailment_penalty=0.0)
        recs = UB.check_arm_structures(planning, {'dso': dso}, reference, dso_build_record=build)
        ok = len(recs) == 36 and all(r['newly_fixed_vars'] == 0 and r['trivial_rows_deactivated'] == 0
                                     and r['interface_voltage_pin']['pinned'] for r in recs.values())
        for node_id in dso:
            for year in dso[node_id]:
                for day in dso[node_id][year]:
                    block = dso[node_id][year][day]
                    ok = ok and pe.value(block.interface_settlement_weight) == 1.0
                    ok = ok and pe.value(block.penalty_gen_curtailment) == 0.0
        certified_pin = UB.dso_interface_voltage_pin(
            models['dso'][DETAIL_NODE][DETAIL_YEAR][DETAIL_DAY],
            planning.distribution_networks[DETAIL_NODE].network[DETAIL_YEAR][DETAIL_DAY])
        ok = ok and not certified_pin['pinned']      # the coordinated block frees it: the check discriminates
        return ok, {'certified_block_pin': certified_pin,
                    'first': next(iter(recs.values()))['arithmetic'], 'n_blocks': len(recs)}
    results.append(_check('srp1_price_taker_dso_structure_passes_and_pin_discriminates', price_taker_structure))

    def tso_structures():
        out = {}
        ok = True
        for coupling in (UB.TSO_COUPLING_FIXED, UB.TSO_COUPLING_TRACKING_PENALTY):
            # the ARM's TSO build (fixed: TSO_ARM_PIN_INTERFACE_VOLTAGE = False) and the unpinned tracking build
            tso, build = UB.build_tso_arm_model(planning, cand_tc, targets, curtailment_penalty=0.0, coupling=coupling,
                                                pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
            recs = UB.check_arm_structures(planning, {'tso': tso}, reference, tso_build_record=build,
                                           tso_coupling=coupling)
            expected_rows = (BENCH.SRP1_DECLARED['tso_arm_fixed_rows_per_block'] if coupling == UB.TSO_COUPLING_FIXED
                             else 0)
            ok = ok and len(recs) == 12 and all(r['fixed_rows'] == expected_rows for r in recs.values())
            if coupling == UB.TSO_COUPLING_TRACKING_PENALTY:
                block = tso[DETAIL_YEAR][DETAIL_DAY]
                ok = ok and pe.value(block.scenario_tracking_weight) == 9e10
            out[coupling] = next(iter(recs.values()))['arithmetic']
        return ok, out
    results.append(_check('srp1_tso_fixed_and_tracking_structures_pass', tso_structures))

    def coupling_check_voltage_pin():
        # W94: the 24-solve check's fixed side must carry the interface-voltage pin; the TSO arm must not.
        declared = BENCH.SRP1_DECLARED
        detail = {}
        ok = (UB.COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE is True and UB.TSO_ARM_PIN_INTERFACE_VOLTAGE is False)
        # (a) the check's two sides, built by the check's own builder (zero solves)
        check = UB.build_tso_coupling_check_models(planning, candidate, targets, tso_curtailment_penalty=0.0,
                                                   reference_structure=reference)
        fixed, penalty = check[UB.TSO_COUPLING_FIXED], check[UB.TSO_COUPLING_TRACKING_PENALTY]
        ok = ok and fixed['build']['interface_voltage_pinned'] is True
        ok = ok and penalty['build']['interface_voltage_pinned'] is False
        ok = ok and len(fixed['voltage_pin']) == 12 and all(
            e['present'] and e['n_active_rows'] == 3 * 24 and e['max_abs_target_minus_tracked_v_pu'] == 0.0
            for e in fixed['voltage_pin'].values())
        ok = ok and all(r['fixed_rows'] == declared['coupling_check_fixed_side_fixed_rows_per_block']
                        and r['interface_voltage_pinned'] is True for r in fixed['structure'].values())
        ok = ok and all(not e['present'] and e['n_active_rows'] == 0 for e in penalty['voltage_pin'].values())
        ok = ok and all(r['fixed_rows'] == declared['coupling_check_penalty_side_fixed_rows_per_block']
                        for r in penalty['structure'].values())
        detail['fixed_side_arithmetic'] = next(iter(fixed['structure'].values()))['arithmetic']
        # (b) SAME target as the penalty: at the pinned voltages the penalty side's voltage term is exactly 0
        tn = planning.transmission_network
        network = tn.network[DETAIL_YEAR][DETAIL_DAY]
        f_block = fixed['model'][DETAIL_YEAR][DETAIL_DAY]
        p_block = penalty['model'][DETAIL_YEAR][DETAIL_DAY]
        v_param = getattr(f_block, UB.FIXED_INTERFACE_V_TARGET)
        for dn, node_id in enumerate(tn.active_distribution_network_nodes):
            idx = network.get_node_idx(node_id)
            for s_m in p_block.scenarios_market:
                for s_o in p_block.scenarios_operation:
                    for p in p_block.periods:
                        p_block.vmag[idx, s_m, s_o, p].value = pe.value(v_param[dn, p])
        tracking_v = pe.value(p_block.scenario_tracking_voltage)
        ok = ok and tracking_v == 0.0
        detail['penalty_voltage_term_at_pinned_voltages'] = tracking_v
        # (c) the TSO ARM: its own declared build carries no pin, and the arm function passes the arm constant
        arm_tso, arm_build = UB.build_tso_arm_model(planning, cand_tc, targets, curtailment_penalty=0.0,
                                                    coupling=UB.TSO_COUPLING_FIXED,
                                                    pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
        arm_recs = UB.check_arm_structures(planning, {'tso': arm_tso}, reference, tso_build_record=arm_build,
                                           tso_coupling=UB.TSO_COUPLING_FIXED)
        arm_pin = UB.coupling_check_voltage_pin(planning, arm_tso, targets)
        ok = ok and arm_build['interface_voltage_pinned'] is False
        ok = ok and all(not e['present'] and e['n_active_rows'] == 0 for e in arm_pin.values())
        ok = ok and all(r['fixed_rows'] == declared['tso_arm_fixed_rows_per_block'] for r in arm_recs.values())
        arm_src = inspect.getsource(UB.run_operational_planning_uncoordinated)
        ok = ok and 'pin_interface_voltage=TSO_ARM_PIN_INTERFACE_VOLTAGE' in arm_src
        # (d) negative controls
        refused_tracking, det_tracking = _expect_raises(
            ValueError, lambda: UB.build_tso_arm_model(planning, cand_tc, targets, curtailment_penalty=0.0,
                                                       coupling=UB.TSO_COUPLING_TRACKING_PENALTY,
                                                       pin_interface_voltage=True),
            'interface-voltage pin is defined only for')
        refused_move, det_move = _expect_raises(
            ValueError, lambda: UB.set_tso_interface_targets(planning, fixed['model'], targets),
            'does not move the interface-voltage pin')
        a_block = arm_tso[DETAIL_YEAR][DETAIL_DAY]
        a_block.add_component(UB.FIXED_INTERFACE_V_TARGET, pe.Param(
            a_block.active_distribution_networks, a_block.periods, mutable=True, domain=pe.Reals, initialize=1.0))
        a_block.add_component(UB.FIXED_INTERFACE_V_ROW, pe.Constraint(
            a_block.active_distribution_networks, a_block.periods, rule=UB._fixed_interface_v_rule))
        planted_fails, det_planted = _expect_raises(
            UB.StructuralCheckError,
            lambda: UB.check_arm_structures(planning, {'tso': arm_tso}, reference, tso_build_record=arm_build,
                                            tso_coupling=UB.TSO_COUPLING_FIXED),
            f"undeclared Constraint component '{UB.FIXED_INTERFACE_V_ROW}'")
        ok = ok and refused_tracking and refused_move and planted_fails
        detail.update({'arm_first_arithmetic': next(iter(arm_recs.values()))['arithmetic'],
                       'NEGATIVE_pin_under_tracking': det_tracking, 'NEGATIVE_move_targets_on_pinned': det_move,
                       'NEGATIVE_pin_planted_on_arm': det_planted})
        return ok, detail
    results.append(_check('srp1_coupling_check_fixed_side_pins_voltage_and_tso_arm_does_not', coupling_check_voltage_pin))

    def negative_real_consensus():
        dso, build = UB.build_dso_arm_models(planning, cand_tc, arm=UB.ARM_PRICE_TAKER, curtailment_penalty=0.0)
        certified_block = models['dso'][DETAIL_NODE][DETAIL_YEAR][DETAIL_DAY]
        scale = float(pe.value(certified_block.admm_common_objective_scale))
        srp.update_distribution_models_to_admm(planning, dso, planning.params.admm, scale)
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_structures(planning, {'dso': dso}, reference,
                                                              dso_build_record=build),
                              'unremoved consensus term')
    results.append(_check('srp1_NEGATIVE_production_consensus_terms_left_in_fails', negative_real_consensus))

    def negative_planted_row():
        dso, build = UB.build_dso_arm_models(planning, cand_tc, arm=UB.ARM_PRICE_TAKER, curtailment_penalty=0.0)
        block = dso[DETAIL_NODE][DETAIL_YEAR][DETAIL_DAY]
        block.planted_consensus_row = pe.Constraint(
            block.periods, rule=lambda m, p: m.expected_interface_pf_p[p] == 0.1)
        return _expect_raises(UB.StructuralCheckError,
                              lambda: UB.check_arm_structures(planning, {'dso': dso}, reference,
                                                              dso_build_record=build),
                              "undeclared Constraint component 'planted_consensus_row'")
    results.append(_check('srp1_NEGATIVE_planted_consensus_row_fails', negative_planted_row))

    def common_q_reproduces():
        ev = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=BENCH.TIE_BREAKER['evaluation'],
                                  require_unchanged=True)
        gate = UB.common_q_gate(ev, record['recourse_components'])
        return gate['passed'], {k: gate[k] for k in ('status', 'gross_evaluated', 'gross_certified', 'difference',
                                                     'n_fields_bitwise_equal', 'n_fields')}
    results.append(_check('srp1_common_q_reproduces_certified_gross_bitwise', common_q_reproduces))

    def common_q_negative():
        refused, detail = _expect_raises(UB.CommonQConfigurationError,
                                         lambda: UB.evaluate_common_q(planning, models,
                                                                      evaluation_curtailment_penalty=1.0,
                                                                      require_unchanged=True),
                                         'do not carry the common evaluation pricing')
        ev0 = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=0.0, require_unchanged=True)
        ev1 = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=1.0, require_unchanged=False)
        gate1 = UB.common_q_gate(ev1, record['recourse_components'])
        priced = sum(ev0['curtailment']['totals'][k]['eur_at_1_block_weighted'] for k in ('TSO', 'DSO'))
        residual = (ev1['gross_operational_cost'] - ev0['gross_operational_cost']) - priced
        restored = UB.evaluate_common_q(planning, models, evaluation_curtailment_penalty=0.0, require_unchanged=True)
        discriminates = priced > 0.0
        passed = (refused and (not discriminates or (not gate1['passed']
                                                     and abs(residual) <= BENCH.NEGATIVE_CONTROL_EXPLAINED_ABS_TOL_EUR))
                  and restored['gross_operational_cost_hex'] == ev0['gross_operational_cost_hex'])
        return passed, {'refusal': detail, 'repriced_gate': gate1['status'], 'priced_curtailment_eur': priced,
                        'explained_residual_eur': residual, 'discriminates': discriminates,
                        'pricing_restored_bitwise': restored['gross_operational_cost_hex'] == ev0['gross_operational_cost_hex']}
    results.append(_check('srp1_NEGATIVE_common_q_wrong_configuration_fails_and_is_explained', common_q_negative))

    def detector_planted_on_certified_clone():
        dn = planning.distribution_networks[DETAIL_NODE]
        network = dn.network[DETAIL_YEAR][DETAIL_DAY]
        block = models['dso'][DETAIL_NODE][DETAIL_YEAR][DETAIL_DAY]
        v_own = [float(pe.value(block.expected_interface_vmag[p])) for p in block.periods]
        clone, build = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v_own)
        tol = dict(hard_tol=BENCH.CONSISTENCY_TOL['hard_tol_pu2'],
                   soft_excess_tol=BENCH.CONSISTENCY_TOL['soft_excess_tol_pu2'],
                   thermal_tol=BENCH.CONSISTENCY_TOL['thermal_tol_pu2'])
        unplanted = UB.consistency_violations(clone, build, block, **tol)
        ref_idx = network.get_node_idx(network.get_reference_node_id())
        target = next(v for v in clone.vmag_sqr.values() if v.index()[0] != ref_idx and v.index()[3] == 5)
        lb, ub = build['original_vmag_sqr_bounds'][target.name]
        target.set_value((ub ** 0.5 + 0.05) ** 2)
        planted = UB.consistency_violations(clone, build, block, **tol)
        flagged = [h['var'] for h in planted['hard']]
        passed = (not unplanted['trigger_sequential_pass'] and planted['trigger_sequential_pass']
                  and target.name in flagged)
        return passed, {'unplanted': {k: unplanted[k] for k in ('max_hard_excess_pu2', 'max_soft_excess_pu2',
                                                                 'max_thermal_violation_pu2')},
                        'planted_var': target.name, 'planted_flagged': flagged[:5],
                        'build': {k: v for k, v in build.items() if not k.startswith('original_')}}
    results.append(_check('srp1_consistency_detects_PLANTED_voltage_violation_no_solve',
                          detector_planted_on_certified_clone))

    def lambda_rows():
        rows = UB.interface_price_terms(planning, models)
        finite = all(isinstance(r[k], float) and r[k] == r[k] and abs(r[k]) < 1e6
                     for r in rows for k in ('pi', 'c_flex', 'lambda_dso_full', 'lambda_tso_full')
                     if r[k] is not None)
        identities = BENCH.units_check_identities_only(rows)
        mismatch = UB.interface_voltage_mismatch(planning, models['tso'], models['dso'])
        return (len(rows) == 864 and finite and identities['passed'] and len(mismatch['entries']) == 864,
                {'identities': identities, 'certified_max_abs_dv_dn_pu': mismatch['max_abs_dv_dn_pu']})
    results.append(_check('srp1_lambda_rows_864_finite_and_param_vs_dual_identities_hold', lambda_rows))
    return results


# ======================================================================================================================
#  srp1-solves group -- exactly 2 solves
# ======================================================================================================================
def srp1_solve_checks(srp, UB, O, pe, run_dir):
    results = []
    planning, logs_dir = BENCH._new_planning(O, 'checks_srp1_solves')
    solver_options = BENCH.apply_arm_solver_options(srp, planning)
    candidate = BENCH._x0_candidate(srp, planning)
    models = BENCH._load_coordinated_models()                           # W106 (W93: _load_w86_models, unwired)
    reference = UB.coordinated_reference_structure(planning, models)
    targets = UB.get_dso_interface_schedule(planning, models['dso'])
    sink = BENCH.SolveSink(run_dir)

    def reevaluation_at_own_voltage():
        dn = planning.distribution_networks[DETAIL_NODE]
        network = dn.network[DETAIL_YEAR][DETAIL_DAY]
        block = models['dso'][DETAIL_NODE][DETAIL_YEAR][DETAIL_DAY]
        v_own = [float(pe.value(block.expected_interface_vmag[p])) for p in block.periods]
        clone, build = UB.build_consistency_reevaluation_block(block, network, v_actual_dn_pu=v_own)
        _result, rec = UB._solve_block(planning, dn, network, clone, kind='DSO', node_id=DETAIL_NODE,
                                       year=DETAIL_YEAR, day=DETAIL_DAY, phase='checks:reevaluation',
                                       record_callback=sink)
        violations = UB.consistency_violations(clone, build, block, hard_tol=1e-6, soft_excess_tol=1e-6,
                                               thermal_tol=1e-6)
        dp = max(abs(pe.value(clone.expected_interface_pf_p[p]) - pe.value(block.expected_interface_pf_p[p]))
                 for p in block.periods) * network.baseMVA
        return (rec['succeeded'] and not violations['trigger_sequential_pass'] and dp <= 1e-3,
                {'interface_p_change_mw_max': dp, 'violations': {k: violations[k] for k in (
                    'max_hard_excess_pu2', 'max_soft_excess_pu2', 'max_thermal_violation_pu2')}})
    results.append(_check('srp1_solve_reevaluation_at_own_voltage_reproduces_and_triggers_nothing',
                          reevaluation_at_own_voltage))

    def one_fixed_tso_block():
        tso, build = UB.build_tso_arm_model(planning, candidate['total_capacity'], targets, curtailment_penalty=0.0,
                                            coupling=UB.TSO_COUPLING_FIXED,
                                            pin_interface_voltage=UB.TSO_ARM_PIN_INTERFACE_VOLTAGE)
        UB.check_arm_structures(planning, {'tso': tso}, reference, tso_build_record=build,
                                tso_coupling=UB.TSO_COUPLING_FIXED)
        tn = planning.transmission_network
        _result, rec = UB._solve_block(planning, tn, tn.network[DETAIL_YEAR][DETAIL_DAY], tso[DETAIL_YEAR][DETAIL_DAY],
                                       kind='TSO', node_id=None, year=DETAIL_YEAR, day=DETAIL_DAY,
                                       phase='checks:tso_fixed', record_callback=sink)
        label = UB.block_label('TSO', None, DETAIL_YEAR, DETAIL_DAY)
        metrics = UB.tso_block_metrics(planning, tso, targets, only={label})[label]
        certified = UB.tso_block_metrics(planning, models['tso'], targets, only={label})[label]
        return (rec['succeeded'] and metrics['interface_residual_p_mw_max'] <= 1e-6
                and metrics['interface_residual_q_mvar_max'] <= 1e-6,
                {'arm': metrics, 'certified_coordinated_same_block': certified})
    results.append(_check('srp1_solve_one_fixed_interface_tso_block', one_fixed_tso_block))
    return results, {'solver_options': solver_options, 'ipopt_logs_dir': logs_dir,
                     'solve_summary': sink.summary()}


# ======================================================================================================================
#  main
# ======================================================================================================================
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=tuple(GROUP_SOLVES))
    args = parser.parse_args(argv)
    run_id = 'checks_' + args.group.replace('-', '_')
    run_dir = os.path.join(BENCH._abs(BENCH.OUT_ROOT_REL), run_id)
    failures = BENCH.check_preconditions(run_dir)
    if failures:
        print(f'REFUSING TO RUN {run_id}:', *failures, sep='\n  ', flush=True)
        return 2
    BENCH._install_guard(run_id, bounded=GROUP_SOLVES[args.group] > 0)
    BENCH.acquire_lock(run_id)
    try:
        os.makedirs(run_dir, exist_ok=False)
        srp, UB, O, _R = BENCH._production()
        import pyomo.environ as pe
        extra = {}
        try:
            if args.group == 'unit':
                results = unit_checks(UB, pe)
            elif args.group == 'srp1-zero':
                results = srp1_zero_checks(srp, UB, O, pe)
            else:
                results, extra = srp1_solve_checks(srp, UB, O, pe, run_dir)
            guard = BENCH._check_guard(GROUP_SOLVES[args.group], f'{run_id} end')
            guard_ok = True
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            results = [{'name': 'group_raised', 'passed': False,
                        'detail': {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}}]
            guard, guard_ok = {'counts': dict(BENCH._GUARD.counts)}, False
        passed = guard_ok and all(r['passed'] for r in results)
        BENCH._write_json_once(os.path.join(run_dir, 'checks.json'), BENCH.provenance({
            'run': run_id, 'group': args.group, 'declared_solves': GROUP_SOLVES[args.group],
            'solve_profile_guard': guard, 'passed': passed, 'n_checks': len(results),
            'n_failed': sum(1 for r in results if not r['passed']), 'checks': results, **extra}))
        BENCH._write_manifest(run_dir)
        BENCH._log(f'{run_id}: {"ALL PASS" if passed else "FAIL"} ({len(results)} checks)')
        return 0 if passed else 1
    finally:
        BENCH._GUARD.uninstall()
        BENCH.release_lock()


if __name__ == '__main__':
    sys.exit(main())
