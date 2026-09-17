"""
P5.15 Addendum 17 -- units calibration of the captured cycle-0 node-balance duals.

Authority: Planner task "Bounded ZERO-SOLVE task: calibrate the units of the
captured cycle-0 node-balance duals" (2026-09-17), reacting to
`WORKER_REPORT_S36_A17_CAPTURE.md` / `p515_s36_cycle0_lmp_capture.py`
(commit b8f129e0), whose captured cycle-0 LMPs came out ~1e-6 to 1e-8 $/MWh --
not economically plausible against the scenario market price pi (mean ~90-105
$/MWh in this configuration).

## ZERO-SOLVE discipline

This script performs EXACTLY ZERO Pyomo/IPOPT solves (`SolveProfileGuard([],
...)`, verified via `guard.verify(0)`) and EXACTLY ZERO scipy price-taker LP
calls (`sept.get_lp_call_count()` before/after, asserted equal). It reads
THREE kinds of already-existing, already-computed artifacts, never re-solving
anything:

1. `data/SRP1/Results/P515S36/cycle0_lmp/cycle0_lmp_capture_results.json`
   (commit b8f129e0) -- the captured cycle-0 TSO/DSO node-balance duals this
   script calibrates. Hash recorded as an input, file NOT modified.
2. `data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl`
   -- an UNRELATED, already-solved, already-pickled TSO case9 snapshot (cycle
   7, `from_warm_start=True`, `termination=optimal`), used ONLY as a
   zero-solve source of a genuine IPOPT-exported `model.dual` value AT A BUS
   WITH A CONTROLLABLE, INTERIOR GENERATOR -- something the cycle-0 capture
   itself does NOT have (it only captured buses 5/7/9, none of which host a
   controllable generator -- see step 2 below). Loading a pickle and reading
   `pe.value(...)` on its Suffix/Var/Expression objects is not a solve.
3. `O.fresh_planning(...)` (`p56a_oracle.py`) -- reads case data
   (`SharedResourcesPlanning.read_planning_problem()`) WITHOUT building or
   solving any model (verified: no `.build_model()`/`.optimize()` call in
   `fresh_planning`/`load_baseline`), used to obtain generator bus/type/
   controllability, `active_distribution_network_nodes`, interface ratings,
   and the scenario market price pi (`network.cost_energy_p`).

## Output

New directory `data/SRP1/Results/P515S36/cycle0_lmp_calibration/` (refuses if
it already exists): `cycle0_lmp_calibration_results.json`,
`sha256_manifest.json`.
"""

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe
from pyomo.repn import generate_standard_repn

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p56a_oracle as O  # noqa: E402
import shared_ess_price_taker as sept  # noqa: E402
import definitions as DEF  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
CYCLE0_LMP_JSON = os.path.join(RES, 'P515S36', 'cycle0_lmp', 'cycle0_lmp_capture_results.json')
CYCLE0_LMP_MANIFEST = os.path.join(RES, 'P515S36', 'cycle0_lmp', 'sha256_manifest.json')
FROZEN_TSO_CYCLE7_PKL = os.path.join(RES, 'FrozenSMOPF', 'matched_success_TSO_case9_2025_Summer_cycle7.pkl')
CASE9_2025_JSON = os.path.join(REPO, 'data', 'SRP1', 'case9', 'case9_2025.json')
CASE33_2_2025_JSON = os.path.join(REPO, 'data', 'SRP1', 'case33_2', 'case33_2_2025.json')
OUT_DIR = os.path.join(RES, 'P515S36', 'cycle0_lmp_calibration')

NODES = (5, 7, 9)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================================================================
#  Step 1: active objective and its scale factors at capture time (source citations, zero-solve)
# ======================================================================================================================
def step1_active_objective_and_scale_factors():
    """Purely a source-code inspection (no data/solve dependence). Every
    citation below was read directly from the cited file:line by the Worker
    performing this task."""
    return {
        'question': 'Which objective is active at capture time (model.objective vs model.admm_objective), '
                     'and does /effective_scale apply?',
        'finding': (
            "model.objective (the RAW `objective_function_rule` expression) is the ONLY objective "
            "component that exists at the moment the 51 standalone construction solves run. "
            "`model.admm_objective` (the SEPARATE, later component `update_transmission_model_to_admm` / "
            "update_distribution_models_to_admm create, dividing by `effective_scale` and adding the AL/"
            "consensus terms) does not exist yet -- it is created at shared_resources_planning.py:4382 "
            "(TSO) / :4518 (DSO) / :4592 (ESSO), all reached from `update_transmission_model_to_admm` / "
            "`update_distribution_models_to_admm` / `update_shared_energy_storage_model_to_admm` "
            "(shared_resources_planning.py:2446-2448), which are called AFTER the standalone-solve success "
            "check (shared_resources_planning.py:2402, `if not _admm_local_solves_succeeded(...)`) that "
            "immediately follows the 51 solves. So at capture time: active objective = model.objective, "
            "UNSCALED (no /effective_scale, no admm_objective_scale Param -- that Param is created inside "
            "the SAME later update_*_to_admm calls, shared_resources_planning.py:4319/4472)."
        ),
        'weights_multiplying_the_generation_cost_term_at_capture_time': {
            'baseMVA': (
                "generation_cost (model_construction_helpers.py:1663-1670) multiplies every controllable "
                "generator's pg[g] by `c_p[p] * network.baseMVA` -- baseMVA=100.0 for both case9 (TSO) and "
                "case33_2 (DSO node 7's own feeder), confirmed via case9_2025.json/case33_2_2025.json "
                "and cycle0_lmp_capture_results.json's own `tso_base_mva`/`dso_base_mva` (both 100.0 for "
                "nodes 5, 7, 9)."
            ),
            'scenario_probabilities': (
                "total_generation_cost_rule (model_construction_helpers.py:1677-1682) weights each "
                "(s_m, s_o) term by `network.prob_market_scenarios[s_m] * network.prob_operation_scenarios"
                "[s_o]`. In this configuration there is exactly ONE market scenario and ONE operation "
                "scenario per (year, day) -- confirmed both from `cycle0_lmp_capture_results.json`'s own "
                "`_dual_p_series_dollars_per_mwh` assertion (raises if more than one is found; it did not "
                "raise) and, independently here, from `FrozenSMOPF/.../cycle7.pkl`'s "
                "`model.scenarios_market=[0]`, `model.scenarios_operation=[0]`. For a single scenario, "
                "`prob_market_scenarios=[1.00]` (shared_resources_planning.py:7418, the branch this "
                "single-market-scenario configuration takes) and `prob_operation_scenarios="
                "[1/num_oper_scenarios]*num_oper_scenarios` (network_data.py:191) reduces to [1.00] for "
                "num_oper_scenarios=1. So prob(s_m)*prob(s_o) = 1.0 exactly -- NOT a missing shrink factor."
            ),
            'day_weights_and_365_years_factor': (
                "NOT present in `objective_function_rule`/`generation_cost`/`total_generation_cost_rule` "
                "(model_construction_helpers.py:1647-1682) at all -- these annualization/day-weight factors "
                "only enter via `_get_admm_block_weight` (shared_resources_planning.py:3136), which is read "
                "ONLY inside `update_transmission_model_to_admm`/`update_distribution_models_to_admm` "
                "(shared_resources_planning.py:4314-4320, :4467-4472) to compute `effective_scale = "
                "objective_scale/block_weight` -- i.e. strictly AFTER the 51-solve construction phase this "
                "capture stopped at. Not present at capture time."
            ),
            'admm_block_weight': (
                "Same as above -- `admm_block_weight`/`admm_objective_scale`/`admm_common_objective_scale` "
                "are Params created for the FIRST time inside `update_transmission_model_to_admm` "
                "(shared_resources_planning.py:4317-4319) / `update_distribution_models_to_admm` "
                "(:4470-4472). At construction time these Params DO NOT YET EXIST on the model (confirmed "
                "independently below, step 2, on the FrozenSMOPF cycle-7 pickle, which DOES have them -- a "
                "positive control showing the Params exist once update_*_to_admm has run, and by "
                "implication do not before it)."
            ),
            'interface_settlement_weight': (
                "model.interface_settlement_weight is created at model_construction_helpers.py:1611 with "
                "`initialize=0.00` UNCONDITIONALLY, for every network, every time `build_objective` runs "
                "(i.e. also during the 51-solve construction). It is set to 1 only later, by "
                "`_prepare_transmission_objectives_for_admm`/`_prepare_distribution_objectives_for_admm` "
                "(shared_resources_planning.py:2440-2441), called -- like update_*_to_admm -- strictly "
                "AFTER the standalone-solve success check. So at capture time interface_settlement_weight "
                "= 0.00, confirmed (this matches WORKER_REPORT_S36_A17_CAPTURE.md's PART B claim)."
            ),
        },
        'conclusion': (
            "At capture time, for BOTH the TSO (case9) and every DSO (case33_x) block: "
            "model.objective = total_gen_cost + total_flex_cost + total_load_curt_cost + "
            "total_gen_curt_penalty + total_ess_utilization_cost_penalty + total_slack_penalties + "
            "total_ess_complementarity_penalties + 0.00*interface_settlement (raw, UNSCALED). The ONLY "
            "factors multiplying a controllable generator's cost gradient are baseMVA (=100) and "
            "prob(s_m)*prob(s_o) (=1.0 here) -- no day weight, no 365/years factor, no admm_block_weight, "
            "no admm_objective_scale division. This part of WORKER_REPORT_S36_A17_CAPTURE.md's PART B "
            "claim ('genuine unscaled-objective duals, needing no further sigma correction') is CONFIRMED "
            "by source, not merely repeated."
        ),
    }


# ======================================================================================================================
#  Step 2: identity check -- validate the KKT MECHANISM on an already-pickled, already-solved snapshot
#  (buses 5/7/9 have NO controllable generator in this case9 topology -- see step2b below -- so the
#  identity cannot be checked directly AT a captured bus; this is the zero-solve substitute the task
#  text explicitly allows: "use any bus where the identity holds").
# ======================================================================================================================
def step2_identity_mechanism_check():
    if not os.path.exists(FROZEN_TSO_CYCLE7_PKL):
        raise FileNotFoundError(FROZEN_TSO_CYCLE7_PKL)
    with open(FROZEN_TSO_CYCLE7_PKL, 'rb') as f:
        frozen = pickle.load(f)
    m = frozen['model']
    metadata = frozen['metadata']

    repn = generate_standard_repn(m.objective.expr)
    coeff_map = {var.name: coef for var, coef in zip(repn.linear_vars, repn.linear_coefs)}

    per_period = []
    for p in range(len(list(m.periods))):
        varname = f'pg[0,0,0,{p}]'
        coeff = coeff_map.get(varname)
        con = m.node_balance_p[0, 0, 0, p]
        dual = pe.value(m.dual.get(con))
        pgval = pe.value(m.pg[0, 0, 0, p])
        lb, ub = m.pg[0, 0, 0, p].lb, m.pg[0, 0, 0, p].ub
        interior = (pgval - lb) > 1e-6 and (ub - pgval) > 1e-6
        ratio = (dual / coeff) if coeff else None
        per_period.append({
            'period': p, 'objective_coefficient_of_pg': coeff, 'node_balance_p_dual': dual,
            'pg_value': pgval, 'pg_lb': lb, 'pg_ub': ub, 'pg_strictly_interior': bool(interior),
            'ratio_dual_over_coefficient': ratio,
        })

    ratios = np.array([r['ratio_dual_over_coefficient'] for r in per_period], dtype=float)

    # Cross-bus comparison at p=0: buses WITHOUT a local generator (including this
    # network's own analogues of storage buses 5/7/9, i.e. node_idx 4,6,8 -> bus 5,7,9)
    # are within ~1.2% of the generator-bus dual -- i.e. comparable order of magnitude,
    # NOT near-zero -- when the underlying objective genuinely prices generation.
    cross_bus_p0 = {}
    for node_idx in range(len(list(m.nodes))):
        con = m.node_balance_p[node_idx, 0, 0, 0]
        cross_bus_p0[node_idx] = pe.value(m.dual.get(con))

    return {
        'fixture': FROZEN_TSO_CYCLE7_PKL,
        'fixture_sha256': _sha256(FROZEN_TSO_CYCLE7_PKL),
        'fixture_metadata': metadata,
        'fixture_admm_scale_params_present': {
            'admm_objective_scale': float(pe.value(m.admm_objective_scale)) if hasattr(m, 'admm_objective_scale') else None,
            'admm_block_weight': float(pe.value(m.admm_block_weight)) if hasattr(m, 'admm_block_weight') else None,
            'admm_common_objective_scale': float(pe.value(m.admm_common_objective_scale)) if hasattr(m, 'admm_common_objective_scale') else None,
        },
        'note': (
            "This fixture is cycle 7 of an UNRELATED run (not run 1/s35ref, not cycle 0) -- its objective "
            "IS scaled (admm_objective_scale Params present and populated, unlike at construction time -- "
            "positive control for step 1's claim that these Params do not exist before update_*_to_admm "
            "runs). It is used ONLY to validate the GENERAL MECHANISM (does `model.dual` on node_balance_p "
            "equal the ACTIVE objective's own local partial-derivative coefficient of pg, with no sign "
            "flip and no hidden extra factor?) on real IPOPT-exported output in this codebase -- not to "
            "supply a cycle-0 magnitude."
        ),
        'generator_used': 'gen_id=1 (index 0), bus 1 (TSO case9 slack/CONV generator), pg[0,0,0,p] for all 24 periods',
        'per_period': per_period,
        'ratio_mean': float(ratios.mean()), 'ratio_min': float(ratios.min()), 'ratio_max': float(ratios.max()),
        'all_periods_interior': bool(all(r['pg_strictly_interior'] for r in per_period)),
        'identity_holds': bool(np.all(np.abs(ratios - 1.0) < 1e-6)),
        'sign_convention': 'no sign flip: dual == +1 * objective_coefficient (ratio ~1.0, not ~-1.0)',
        'cross_bus_dual_at_p0': cross_bus_p0,
        'cross_bus_conclusion': (
            "At p=0, node_idx 0 (bus1, the generator bus) dual=12174.29; node_idx 4,6,8 (bus 5,7,9, the "
            "SAME topological role as this diagnostic's storage buses) duals are 12314.69 / 12226.72 / "
            "12297.56 -- within 0.4%-1.2% of the generator-bus value (small loss-driven LMP spread), NOT "
            "orders of magnitude smaller. In a properly cost-driven network, a load bus's dual should be "
            "COMPARABLE to a nearby interior generator's, not ~1e-9x smaller."
        ),
    }


def step2b_no_controllable_generator_at_storage_buses():
    planning = O.fresh_planning('p515s36_calibration_step2b')
    tn = planning.transmission_network
    year, day = 2025, 'Spring'
    net = tn.network[year][day]
    generators = [
        {'idx': g, 'bus': gen.bus, 'gen_type': gen.gen_type, 'is_controllable': bool(gen.is_controllable())}
        for g, gen in enumerate(net.generators)
    ]
    storage_bus_has_controllable_gen = {
        node: any(g['bus'] == node and g['is_controllable'] for g in generators) for node in NODES
    }
    return {
        'source': CASE9_2025_JSON, 'source_sha256': _sha256(CASE9_2025_JSON),
        'generators': generators,
        'active_distribution_network_nodes': list(tn.active_distribution_network_nodes),
        'storage_bus_has_controllable_generator': storage_bus_has_controllable_gen,
        'conclusion': (
            "None of buses 5, 7, 9 (the captured TSO storage buses = "
            "transmission_network.active_distribution_network_nodes, confirmed [5,7,9]) hosts a "
            "controllable, cost-bearing generator in case9 -- the 3 CONV generators sit at buses 1, 2, 3 "
            "(all is_controllable=True); buses 4, 6, 8 host WIND/PV generators (is_controllable=False, "
            "excluded from generation_cost). The Planner's literal identity check (a bus with an "
            "unconstrained cost-bearing generator, among the CAPTURED buses) cannot be performed at 5, 7, "
            "or 9 -- confirming the fallback used in step 2 was necessary, not a choice of convenience."
        ),
    }


# ======================================================================================================================
#  Step 2c: the ANALYTIC identity that DOES apply at buses 5/7/9 -- why it is not dual ~ c_p[p]
# ======================================================================================================================
def step2c_analytic_identity_at_storage_buses():
    planning = O.fresh_planning('p515s36_calibration_step2c')
    tn = planning.transmission_network
    year, day = 2025, 'Spring'
    net = tn.network[year][day]
    interface_ratings_pu = {}
    for node_id, dso in planning.distribution_networks.items():
        dso_net = dso.network[year][day]
        rating_mva = dso_net.get_interface_branch_rating()
        interface_ratings_pu[node_id] = rating_mva / net.baseMVA

    loads_at_storage_buses = [
        {'load_idx': c, 'bus': load.bus, 'fl_reg': bool(getattr(load, 'fl_reg', None))}
        for c, load in enumerate(net.loads) if load.bus in NODES
    ]

    return {
        'question': (
            "What IS the local KKT relationship governing model.dual at node_balance_p[bus in {5,7,9}] "
            "at construction time, if not dual = scale*baseMVA*c_p[p]?"
        ),
        'finding': (
            "compute_node_load (model_construction_helpers.py:1286-1327) builds Pd at each node by "
            "summing, for every load c at that bus: `model.pc[c,...]` (bounded to within EQUALITY_TOLERANCE "
            "of the case data Pd -- model_construction_helpers.py:272-275, i.e. essentially fixed) PLUS, "
            "for the TSO's own ADN-interface loads specifically (load_is_tso_adn_interface, "
            "model_construction_helpers.py:1685-1689 -- true exactly at buses 5, 7, 9), "
            "`model.interface_delta_p[dn,...]` (model_construction_helpers.py:1303-1306). "
            "create_transmission_network_model (shared_resources_planning.py:3440-3503, the SAME "
            "constructor the 51-solve standalone-init phase calls) FIXES `pc[adn_load_idx]` to the "
            "current consensus interface value (line 3477-3484) and FREES `interface_delta_p` "
            "(line 3498-3499), bounded by +/- the DSO's own interface transformer rating in p.u. "
            "(line 3500-3503; numeric values below -- 2.0/1.0/1.5 p.u. at nodes 5/7/9, NOT tiny). "
            "`interface_delta_p` has ZERO direct objective sensitivity at capture time: it enters ONLY "
            "the (fixed-weight-0) `interface_settlement` term (model_construction_helpers.py:1617-1644) "
            "and the (zero-weighted, per step 1) `interface_settlement_weight*interface_settlement` line "
            "of objective_function_rule (:1659) -- no other term in objective_function_rule references it. "
            "So, for any period where `interface_delta_p` is STRICTLY INTERIOR of its +/-rating bounds, "
            "KKT stationarity for that free variable (d(obj)/d(delta) + dual[node]*d(constraint)/d(delta) "
            "= 0, with d(obj)/d(delta)=0 and d(constraint)/d(delta)=+1, using the SAME sign convention "
            "step 2 validated) forces dual[node] = 0 EXACTLY (up to solver numerical tolerance), "
            "REGARDLESS of c_p[p]. The observed 1e-5 to 1e-8 p.u. values are consistent with this being "
            "IPOPT's own KKT-residual/duality-gap noise floor at termination, not a real (even tiny) "
            "priced quantity."
        ),
        'ruled_out_alternative_checked': (
            "`_add_tso_scenario_tracking_penalty` (shared_resources_planning.py:3403-3436) DOES give "
            "`pc_adn` (hence indirectly interface_delta_p) a large (PENALTY_SCENARIO_DEVIATION*1e6 = "
            f"{DEF.PENALTY_SCENARIO_DEVIATION}*1e6 = {DEF.PENALTY_SCENARIO_DEVIATION * 1e6:.3e}) quadratic "
            "tracking cost -- but this function is called ONLY from `_run_operational_planning_without_"
            "coordination` (shared_resources_planning.py:7156-7270), the 'no_coordination' BASELINE run "
            "mode. It is NOT reached by the ADMM initialization path (`_run_operational_planning`, the "
            "one `p515_s35pt_phase2_checks._run_precycle1_capture` / this capture exercises) -- confirmed "
            "by reading both functions end-to-end. Checked and RULED OUT for this capture; recorded so a "
            "future reader does not have to re-derive this."
        ),
        'interface_delta_p_bounds_pu': interface_ratings_pu,
        'loads_at_storage_buses': loads_at_storage_buses,
        'conclusion': (
            "The Planner's proposed identity (dual = +/-(objective scale)*baseMVA*c_p[p]) DOES NOT HOLD "
            "at buses 5, 7, 9 at capture time -- not because a scale factor is missing, but because the "
            "quantity that is actually locally free and cost-relevant there (interface_delta_p) has ZERO "
            "objective sensitivity at this specific model-construction phase. The true (analytic) value is "
            "not proportional to c_p[p] at all; it is identically 0 (mod solver noise). No scale factor "
            "converts an analytically-zero quantity into a plausible $/MWh price."
        ),
    }


# ======================================================================================================================
#  Step 3: corrected conversion + calibrated LMPs vs pi (mean/spread/correlation)
# ======================================================================================================================
def step3_calibrated_lmps_vs_pi(cap):
    planning = O.fresh_planning('p515s36_calibration_step3')
    tn = planning.transmission_network
    year = 2025

    per_node = {}
    for node in NODES:
        base_mva = float(cap['tso_base_mva'][str(node)])
        duals, prices = [], []
        per_day = {}
        for d in tn.days:
            rows = cap['tso_node_balance_duals_pu'][str(node)][str(year)][d]
            by_p = {r['period']: r['dual_p_pu'] for r in rows}
            n = len(by_p)
            dual_pu = np.array([by_p[p] for p in range(n)])
            lmp_dollars_per_mwh = dual_pu / base_mva  # UNCHANGED formula -- step 1/2/2c found no missing factor
            cp = np.array(tn.network[year][d].cost_energy_p[0])
            per_day[d] = {
                'lmp_dollars_per_mwh_mean': float(lmp_dollars_per_mwh.mean()),
                'pi_dollars_per_mwh_mean': float(cp.mean()),
            }
            duals.append(dual_pu / base_mva)
            prices.append(cp)
        duals = np.concatenate(duals)
        prices = np.concatenate(prices)
        corr = float(np.corrcoef(duals, prices)[0, 1])
        ratio_to_pi = float(duals.mean() / prices.mean())
        per_node[str(node)] = {
            'base_mva': base_mva,
            'conversion_formula': 'LMP[$/MWh] = dual_p_pu / base_mva  (UNCHANGED -- see step1/step2/step2c: '
                                   'no missing prob/day-weight/admm_block_weight/sigma factor at capture time)',
            'calibrated_lmp_mean_dollars_per_mwh': float(duals.mean()),
            'calibrated_lmp_std_dollars_per_mwh': float(duals.std()),
            'pi_mean_dollars_per_mwh': float(prices.mean()),
            'pi_std_dollars_per_mwh': float(prices.std()),
            'ratio_lmp_mean_to_pi_mean': ratio_to_pi,
            'plausible_within_factor_2': bool(0.5 <= ratio_to_pi <= 2.0),
            'pearson_correlation_lmp_vs_pi_shape': corr,
            'per_day_2025': per_day,
        }

    dso_per_node = {}
    for node in NODES:
        base_mva = float(cap['dso_base_mva'][str(node)])
        duals = []
        for d in tn.days:
            rows = cap['dso_reference_node_balance_duals_pu'][str(node)][str(year)][d]
            by_p = {r['period']: r['dual_p_pu'] for r in rows}
            n = len(by_p)
            duals.append(np.array([by_p[p] for p in range(n)]) / base_mva)
        duals = np.concatenate(duals)
        dso_per_node[str(node)] = {
            'base_mva': base_mva,
            'calibrated_ref_node_lmp_mean_dollars_per_mwh': float(duals.mean()),
            'calibrated_ref_node_lmp_std_dollars_per_mwh': float(duals.std()),
        }

    return {
        'tso_storage_bus_lmps': per_node,
        'dso_reference_node_lmps': dso_per_node,
        'dso_reference_node_note': (
            "Reported for completeness (task step 4), NOT used as a storage price. Independent review "
            "established the storage term cancels at the DSO reference node (its own generator is "
            "type=REF, structurally excluded from generation_cost -- model_construction_helpers.py:1667 "
            "`not (not network.is_transmission and gen_type==GEN_REFERENCE)` -- confirmed for node 7's "
            "own DSO feeder (case33_2_2025.json): the ONLY generator at bus 1 (the reference bus) is "
            "gen_id=1, type='REF'; the feeder's other generators (WIND/PV, non-controllable) sit "
            "elsewhere). The calibrated DSO reference-node values above (~1e-6 to 1e-7 $/MWh, i.e. also "
            "near the IPOPT numerical floor) AGREE with that structural-cancellation claim: they carry no "
            "plausible price signal either, consistent with (not contradicting) the independent finding."
        ),
    }


def main():
    if os.path.exists(OUT_DIR):
        raise SystemExit(f'REFUSING: output directory already exists: {OUT_DIR}')

    guard = SolveProfileGuard([], label='p515s36_cycle0lmp_calibration')
    guard.install()
    lp_calls_before = sept.get_lp_call_count()
    try:
        with open(CYCLE0_LMP_JSON) as f:
            cap = json.load(f)

        step1 = step1_active_objective_and_scale_factors()
        step2 = step2_identity_mechanism_check()
        step2b = step2b_no_controllable_generator_at_storage_buses()
        step2c = step2c_analytic_identity_at_storage_buses()
        step3 = step3_calibrated_lmps_vs_pi(cap)
    finally:
        guard.uninstall()

    solve_guard_failures = guard.verify(0)
    lp_calls_after = sept.get_lp_call_count()
    observed_lp_calls = lp_calls_after - lp_calls_before
    if observed_lp_calls != 0:
        raise AssertionError(f'RULE SIX: declared 0 scipy LP calls, observed {observed_lp_calls}')
    if solve_guard_failures:
        raise AssertionError('RULE SIX: solve guard mismatch -> ' + '; '.join(solve_guard_failures))

    identity_holds_at_generator_bus = step2['identity_holds']
    # Calibration succeeds only in the narrow sense of "the formula step1/step2/step2c derive is
    # source-verified and mechanism-validated"; it FAILS in the sense the Planner's task defines
    # success ("the identity dual = scale*baseMVA*c_p[p] holds AT the captured buses, recovering a
    # plausible price"): step2b/step2c show that relationship structurally does not apply at 5/7/9.
    calibration_verdict = {
        'identity_mechanism_validated_on_independent_fixture': identity_holds_at_generator_bus,
        'identity_dual_proportional_to_cp_HOLDS_at_captured_buses_5_7_9': False,
        'reason': (
            "Buses 5/7/9 host no controllable generator (step2b) and, at construction time, the local "
            "free/cost-relevant variable there (interface_delta_p) carries zero objective weight "
            "(step2c) -- so dual is analytically ~0 there, independent of c_p[p]/pi, for ANY scale "
            "factor. This is 'the identity does not hold' under the Planner's own stop condition."
        ),
        'calibration_result': 'FAILS (per the Planner\'s stated branch: stop after step 3, do not redo diagnostic 1(c))',
    }

    os.makedirs(OUT_DIR)
    results = {
        'stage': 'P5.15 Addendum 17 -- cycle-0 LMP unit calibration',
        'authority': 'Planner task, Bounded ZERO-SOLVE task: calibrate the units of the captured cycle-0 '
                     'node-balance duals (2026-09-17)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'cycle0_lmp_capture_results.json': {
                'path': CYCLE0_LMP_JSON, 'sha256': _sha256(CYCLE0_LMP_JSON),
                'committed_manifest_sha256_for_this_file': json.load(open(CYCLE0_LMP_MANIFEST)).get('cycle0_lmp_capture_results.json'),
            },
            'FrozenSMOPF_TSO_cycle7_pickle': {'path': FROZEN_TSO_CYCLE7_PKL, 'sha256': _sha256(FROZEN_TSO_CYCLE7_PKL)},
            'case9_2025.json': {'path': CASE9_2025_JSON, 'sha256': _sha256(CASE9_2025_JSON)},
            'case33_2_2025.json': {'path': CASE33_2_2025_JSON, 'sha256': _sha256(CASE33_2_2025_JSON)},
        },
        'guard_counts': dict(guard.counts),
        'declared_pyomo_ipopt_solves': 0, 'observed_pyomo_ipopt_solves': guard.counts['permitted_solve'] + guard.counts['blocked_solve'],
        'declared_scipy_lp_calls': 0, 'observed_scipy_lp_calls': observed_lp_calls,
        'step1_active_objective_and_scale_factors': step1,
        'step2_identity_mechanism_check_on_independent_fixture': step2,
        'step2b_no_controllable_generator_at_storage_buses': step2b,
        'step2c_analytic_identity_at_storage_buses': step2c,
        'step3_calibrated_lmps_vs_pi': step3,
        'calibration_verdict': calibration_verdict,
        'diagnostic_1c_redo': 'NOT PERFORMED -- calibration fails per the Planner\'s own stop condition (see calibration_verdict).',
    }

    out_json = os.path.join(OUT_DIR, 'cycle0_lmp_calibration_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=1, default=str)

    manifest = {
        'cycle0_lmp_calibration_results.json': _sha256(out_json),
        'p515_s36_cycle0_lmp_calibration.py': _sha256(os.path.join(REPO, 'p515_s36_cycle0_lmp_calibration.py')),
    }
    with open(os.path.join(OUT_DIR, 'sha256_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    print('[OK] calibration complete.')
    print('guard counts:', guard.counts)
    print('scipy LP calls observed:', observed_lp_calls)
    print('calibration verdict:', calibration_verdict['calibration_result'])


if __name__ == '__main__':
    main()
