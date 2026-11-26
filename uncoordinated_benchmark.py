"""
P5.15 Addendum 49 -- the UNCOORDINATED BENCHMARK: production subproblems with the TSO-DSO coupling removed.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 49 (rulings 1-3, the consistency convention, the resolution rule and
the 2026-09-26 clarification on the tie-breaker and on lambda_t recovery); Planner task W93.

WHY A NEW MODULE, BESIDE THE OLD PATH. Ruling 3: "new production function beside the old ... the existing path stays
unwired and untouched (SRP1 bitwise gate)". `shared_resources_planning._run_operational_planning_without_coordination`
is NOT edited, NOT called and NOT imported by name here; its source sha256 stays pinned by
`p515_s53_srp1_bitwise_gate.py` (`PIN_BENCHMARK_SOURCE_SHA256`). This file is a separate module so that nothing in
`shared_resources_planning.py` or `model_construction_helpers.py` changes (W93 was written while the 3x3 pair was
running, and that campaign's children import those files from disk).

WHAT EACH ARM IS (x = 0 on SRP1; every arm is built from the production constructors, never re-implemented):
  * DSO blocks -- `NetworkData.build_model` + `update_model_with_candidate_solution`, then exactly the component set
    `create_distribution_networks_models_sequential` adds (the expected-interface Vars and their defining rows with
    production's own rule functions, `configure_shared_ess_operational_state`, `add_scenario_commitment_terms` with
    `premium_alpha = 0`, `_set_row18_inactive_for_initialisation`), WITHOUT its initialisation solve and WITHOUT
    `update_distribution_models_to_admm` -- so no consensus term (lambda, rho, proximal) is ever added and the
    interface voltage stays where the build puts it: `e[ref]` in `e_bounds` = vg +/- SMALL_TOLERANCE, the SETPOINT.
      - PASSIVE: build-default pricing (interface settlement weight 0 -- no price), every flexibility Var
        (flex_p_up/down, flex_q_up/down) FIXED AT 0.0; the rows that fixing leaves with no free variable
        (`flex_energy_balance_p`, constant 0 inside +/- SMALL_TOLERANCE) are deactivated and counted; the
        curtailment tie-breaker `penalty_gen_curtailment` is the declared decision value (ruled 1 EUR/MWh: the
        minimum-curtailment selection rule in an otherwise empty objective).
      - PRICE-TAKER: production's own local pricing, `_prepare_distribution_objectives_for_admm` (settlement weight
        1: the DSO pays pi_t for its interface energy; shared-ESS usage weight 0; curtailment cost), and nothing else
        -- lambda = 0, rho = 0 because the augmented-Lagrangian terms are never built; the tie-breaker is the
        declared decision value (ruled 0).
  * TSO block -- the same production TSO construction `create_transmission_network_model` performs (interface
    voltage slacks fixed at 0, `pc`/`qc` of the ADN load fixed, flexibility legs fixed at 0, the scenario-free
    `interface_delta_p/q` freed within +/- the interface rating, expected-interface Vars and rows, shared-ESS state,
    `add_scenario_commitment_terms`), WITHOUT its initialisation solve and WITHOUT `update_transmission_model_to_admm`;
    priced by `_prepare_transmission_objectives_for_admm`; tie-breaker at the declared decision value (ruled 0).
    Coupling (ruling 2):
      - 'fixed_interface' (THE ARM): interface P and Q fixed to the DSO arm's schedule by two families of ROWS,
        `expected_interface_pf_p[dn,t] == uncoord_interface_p_target[dn,t]` (and q), whose right-hand sides are
        MUTABLE Params; `pc`/`qc` are fixed at the same targets, so `interface_delta` is 0 at the solution.
        Interface voltage bounded by the normal bounds, never fixed. NO tracking penalty.
      - 'tracking_penalty' (the 24-solve check only): the same block, no fixed rows, with production's own
        `_add_tso_scenario_tracking_penalty` (9e10 weight; V, P and Q tracked) at the same targets.
    THE 24-SOLVE CHECK'S FIXED SIDE (W94, Planner ruling on W93 item 3) is 'fixed_interface' PLUS a voltage pin:
    `expected_interface_vmag[dn,t] == uncoord_interface_v_target[dn,t]` (a third family of rows on a mutable Param),
    the target being EXACTLY the one the penalty tracks (`_tso_interface_vmag_targets_pu`: targets['v_kv'] / the TN
    node base kV -- the same helper builds the penalty's `interface_vmag`). Reason: the penalty tracks P, Q AND V, so
    a fixed side pinning P and Q only would leave V free and the two sides would be DIFFERENT problems; a TN-cost
    difference could then come from the voltage freedom rather than from the scaling hazard the check exists to
    measure (Addendum 49 ruling 2). With the pin, the penalty's limit (P = P_req, Q = Q_req, V = V_req; single
    scenario, so E[vmag_adn] = vmag_adn) IS the fixed side, and any difference in TN cost, interface residual,
    iteration count or dual infeasibility is numerical. Declared as `COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE`.
    THE TSO ARM ITSELF IS NOT PINNED (`TSO_ARM_PIN_INTERFACE_VOLTAGE = False`): ruling 2's arm has interface P/Q fixed
    and the voltage within its normal bounds.
  * The coordinated arm is the certified x = 0 cell, not re-run.

STRUCTURAL CHECK (ruling 3), run at build time on every block and raising `StructuralCheckError`: against the
certified coordinated block of the same (agent, year, day) -- same Var components and Var count; the SAME set of
fixed Vars except the declared newly-fixed ones (passive: the flexibility Vars); active rows = coordinated rows
- removed consensus rows (0: in production the consensus enters only as augmented-Lagrangian OBJECTIVE terms,
`update_*_model_to_admm`, never as rows -- verified by requiring every reference Var/Constraint component to exist in
the arm) - declared trivially-constant rows deactivated + the declared fixed rows; every component the reference
has and the arm lacks must be a declared consensus term (`CONSENSUS_TERM_COMPONENTS`); the arm must carry NO
consensus term and no undeclared component; the arm's only active objective is `objective`.

COMMON Q (ruling 3 + clarification): `evaluate_common_q` reprices every TSO/DSO block with production's own
ADMM-subproblem pricing (`_prepare_transmission_objectives_for_admm` / `_prepare_distribution_objectives_for_admm`)
and the declared EVALUATION tie-breaker (ruled 0), evaluates production's `_get_operational_recourse_components`, and
restores every Param it touched. Objective convention: `OBJECTIVE_CONVENTION`.

CONSISTENCY (the SRP1 convention): `interface_voltage_mismatch`, `build_consistency_reevaluation_block`,
`consistency_violations`, `pin_dso_interface_voltage`.

EVERY solve goes through `_solve_block` (the one call site a harness's `SolveProfileGuard` permits) and produces a
per-solve record (status, termination, wall time, IPOPT attempt records with the parsed final error summary)
handed to the caller's `record_callback` BEFORE any failure is raised.

Single-scenario only (SRP1). Above one scenario the fixed-interface rows would pin an expectation the per-scenario
DSO flows do not follow; `require_single_scenario` raises rather than generalise untested.

W116 (PLANNER_BRIEF_2026-09-13.md Addendum 57 Decision 1(b); benchmark spec v3): the NO-REVERSE-FLOW arms. With
`no_reverse_flow=True` (`build_dso_arm_models`, `run_operational_planning_uncoordinated`) every DSO arm block carries
the row family NO_REVERSE_FLOW_ROW, pg_adn[s_m, s_o, p] >= 0 (import only; sign and source lines at the constant),
declared per block in the build record and counted by the structural check; the consistency re-evaluation deactivates
it with the other DN limit rows and evaluates it as a hard limit. Default False = the spec-v2 arm, unchanged. The TSO
arm (`build_tso_arm_model`) is not touched.
"""

import math
import re
import time
import zlib
from functools import partial

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.visitor import identify_variables

import shared_resources_planning as srp
from definitions import SMALL_TOLERANCE
from helper_functions import fix_or_set, solver_result_summary
from model_construction_helpers import (
    add_scenario_commitment_terms,
    configure_shared_ess_operational_state,
    dn_interface_expected_pf_p_rule,
    dn_interface_expected_pf_q_rule,
    dn_interface_expected_sess_p_rule,
    dn_interface_expected_sess_q_rule,
    dn_interface_expected_vmag_rule,
    expected_market_price,
    gen_curtailment_definitional_value,
    sess_na_scenario,
    tn_interface_expected_pf_p_rule,
    tn_interface_expected_pf_q_rule,
    tn_interface_expected_sess_p_rule,
    tn_interface_expected_sess_q_rule,
    tn_interface_expected_vmag_rule,
)

# ======================================================================================================================
#  declarations
# ======================================================================================================================
ARM_PASSIVE = 'passive'
ARM_PRICE_TAKER = 'price_taker'
DSO_ARMS = (ARM_PASSIVE, ARM_PRICE_TAKER)

TSO_COUPLING_FIXED = 'fixed_interface'
TSO_COUPLING_TRACKING_PENALTY = 'tracking_penalty'
TSO_COUPLINGS = (TSO_COUPLING_FIXED, TSO_COUPLING_TRACKING_PENALTY)

START_COLD = 'cold'
START_WARM = 'warm_from_certified'
START_PERTURBED = 'perturbed'
STARTS = (START_COLD, START_WARM, START_PERTURBED)

OBJECTIVE_CONVENTION = (
    'gross_operational_cost = production _get_operational_recourse_components: sum of every TSO/DSO block\'s '
    'objective_function_rule value x block weight (years x days x discount annualization), MINUS the contracted '
    'interface settlement (a transfer) and the solver-only voltage pin -- i.e. SETTLEMENT-EXCLUDED; salvage NOT '
    'subtracted (net_operational_recourse = gross - salvage, reported separately). Every block is priced for the '
    'evaluation with production\'s own ADMM-subproblem pricing (_prepare_transmission_objectives_for_admm / '
    '_prepare_distribution_objectives_for_admm) and the curtailment tie-breaker penalty_gen_curtailment at the '
    'declared EVALUATION value (recorded as evaluation_curtailment_penalty); the decision objectives the arms were '
    'solved with are recorded separately.')

# Components the coordination path adds on top of the build (`update_transmission_model_to_admm`,
# `update_distribution_models_to_admm`, and the campaign's `p58_rescale.patched_admm_objectives`, which adds
# RESCALED_OBJECTIVE 'p58_rescaled_admm_objective'). Every one is a Param or an Objective: the consensus lives in the
# objective only. A harness asserts, from the production source, that this set covers every component those two
# functions add.
CONSENSUS_TERM_COMPONENTS = frozenset({
    'rho_v', 'vmag_req', 'dual_vmag_req',
    'rho_pf', 'p_pf_req', 'q_pf_req', 'dual_pf_p_req', 'dual_pf_q_req',
    'rho_ess', 'p_ess_req', 'q_ess_req', 'dual_ess_p_req', 'dual_ess_q_req',
    'rho_ess_prev', 'p_ess_prev', 'q_ess_prev', 'dual_ess_p_prev', 'dual_ess_q_prev',
    'prox_gamma_v', 'prox_gamma_pf', 'prox_gamma_ess',
    'prox_v_prev', 'prox_pf_p_prev', 'prox_pf_q_prev', 'prox_ess_p_prev', 'prox_ess_q_prev',
    'admm_common_objective_scale', 'admm_block_weight', 'admm_objective_scale',
    'admm_objective', 'p58_rescaled_admm_objective',
})
# Removed consensus ROWS: none (see the module docstring); kept as a named constant so the arithmetic of the
# structural check is written out exactly as ruling 3 states it.
REMOVED_CONSENSUS_ROWS = 0

FIXED_INTERFACE_P_TARGET = 'uncoord_interface_p_target'
FIXED_INTERFACE_Q_TARGET = 'uncoord_interface_q_target'
FIXED_INTERFACE_P_ROW = 'uncoord_interface_p_fixed'
FIXED_INTERFACE_Q_ROW = 'uncoord_interface_q_fixed'
# W94: the interface-voltage pin -- used ONLY by the fixed side of the 24-solve coupling check (see the module
# docstring). The arm never carries it.
FIXED_INTERFACE_V_TARGET = 'uncoord_interface_v_target'
FIXED_INTERFACE_V_ROW = 'uncoord_interface_v_fixed'
TSO_ARM_PIN_INTERFACE_VOLTAGE = False                    # ruling 2: the arm's interface voltage is within its bounds
COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE = True   # W94: the check's fixed side = the penalty's limit problem
COUPLING_CHECK_FIXED_SIDE_VOLTAGE_PIN = {
    'pin_interface_voltage': COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE,
    'applies_to': "the 'fixed_interface' side of the 24-solve TSO coupling check ONLY; the TSO arm is unpinned "
                  f'(TSO_ARM_PIN_INTERFACE_VOLTAGE = {TSO_ARM_PIN_INTERFACE_VOLTAGE})',
    'row': f'{FIXED_INTERFACE_V_ROW}: expected_interface_vmag[dn,t] == {FIXED_INTERFACE_V_TARGET}[dn,t] (mutable Param)',
    'target': "targets['v_kv'][t] / TN node base kV (_tso_interface_vmag_targets_pu) -- the SAME values production's "
              '_add_tso_scenario_tracking_penalty receives as interface_vmag on the tracking-penalty side',
    'why': "production's _add_tso_scenario_tracking_penalty tracks P, Q AND V; pinning P and Q only would leave V "
           'free on the fixed side, so a TN-cost difference could come from the voltage freedom rather than from the '
           'scaling hazard the check measures (Addendum 49 ruling 2). With the pin both sides are the same '
           'mathematical problem (the penalty\'s limit P = P_req, Q = Q_req, V = V_req); any difference is numerical',
    'authority': 'Planner task W94 (ruling on W93 item 3)',
}
TRACKING_PENALTY_COMPONENTS = ('scenario_tracking_weight', 'scenario_tracking_voltage',
                               'scenario_tracking_interface_power', 'scenario_tracking_penalty')

FLEXIBILITY_VAR_FAMILIES = ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down')
PASSIVE_TRIVIAL_ROW_FAMILIES = ('flex_energy_balance_p', 'flex_energy_balance_q', 'flex_energy_balance_s')

# W116 (Addendum 57 Decision 1(b)): the NO-REVERSE-FLOW rule of an uncoordinated DSO arm -- the connection-agreement
# rule "import only" at the interface. One ROW per (market scenario, operation scenario, period) of every DSO arm
# block, on production's own per-scenario DSO interface expression:
#     uncoord_no_reverse_flow[s_m, s_o, p]:   pg_adn[s_m, s_o, p] >= 0
# pg_adn (network.py L444, `model.pg_adn = pe.Expression(...)`, DSO blocks only) is
# `model_construction_helpers.interface_pf_p_distribution_def` (L1287-1297): pg[ref_gen, s_m, s_o, p] minus the
# scenario-free shared-ESS net power at the reference bus -- the DN reference generator is the upstream grid, and a
# generator INJECTS into its bus (compute_node_gen, L1425-1433; the load convention of compute_node_load L1402-1419), so
# pg_adn > 0 is power flowing from the TN INTO the DN (IMPORT) and pg_adn < 0 is REVERSE flow (export to the TN). The
# same sign is the DSO's settlement convention (`interface_energy_settlement`, L1729-1756: the DSO block pays
# +pi_t * baseMVA * pg_adn, the cost of importing) and the TSO copy's (`interface_pf_p_transmission_def`, L1241-1262:
# pc_adn = the TN's LOAD at the ADN bus; the consensus pairs E[pg_adn] with E[pc_adn] through expected_interface_pf_p,
# `dn_interface_expected_pf_p_def` L2416-2424 / `tn_interface_expected_pf_p_def` L2473-2475). The expected-interface
# Var the TSO arm is fixed to, expected_interface_pf_p[p] = E_s[pg_adn[s, p]], therefore inherits the bound.
# A ROW (not a Var bound) because pg_adn is an Expression. Only `build_dso_arm_models(no_reverse_flow=True)` adds it;
# the coordinated path, production and the TSO arm never carry it. Declared rows per block = |S_m| x |S_o| x |P|,
# counted by the structural check. In the consistency re-evaluation the row is deactivated like the voltage and
# thermal limit rows (the reference generator is freed there) and EVALUATED afterwards as a hard DN limit
# (`consistency_violations`, kind 'no_reverse_flow', excess in p.u. under the same key as the reference-generator
# bounds), so a violation at the TN's actual voltage triggers the declared sequential pass.
NO_REVERSE_FLOW_ROW = 'uncoord_no_reverse_flow'

# Pricing Params `_prepare_*_objectives_for_admm` may set, plus the tie-breaker: the set `evaluate_common_q` snapshots
# and restores.
PRICING_PARAMS = ('penalty_ess_usage', 'penalty_shared_ess_usage', 'cost_load_curtailment', 'penalty_gen_curtailment',
                  'penalty_load_curtailment', 'penalty_flex_usage', 'interface_settlement_weight')

# Consistency re-evaluation (DSO): decisions held fixed at the arm's solution; the rest is the power-flow state.
DSO_DECISION_VAR_FAMILIES = (
    'flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down',
    'slack_flex_p_balance_up', 'slack_flex_p_balance_down', 'slack_flex_q_balance_up', 'slack_flex_q_balance_down',
    'pc_curt_down', 'pc_curt_up', 'qc_curt_down', 'qc_curt_up',
    'r',
    'es_pch', 'es_pdch', 'es_pch_hat', 'es_pdch_hat', 'es_pnet', 'es_qnet', 'es_soc',
    'slack_es_soc_final_up', 'slack_es_soc_final_down',
    'shared_es_pch', 'shared_es_pdch', 'shared_es_pch_hat', 'shared_es_pdch_hat', 'shared_es_pnet',
    'shared_es_qnet', 'shared_es_soc', 'slack_shared_es_soc_final_up', 'slack_shared_es_soc_final_down',
    'expected_shared_ess_p', 'expected_shared_ess_q',
)
# Slacks held at zero in the re-evaluation (the limit rows they relax are deactivated and evaluated instead).
DSO_REEVALUATION_ZERO_SLACKS = ('slack_v_sqr_down', 'slack_v_sqr_up', 'slack_node_balance_p_up',
                                'slack_node_balance_p_down', 'slack_node_balance_q_up', 'slack_node_balance_q_down',
                                'slack_flow_ij_sqr', 'slack_flow_ji_sqr')
VOLTAGE_LIMIT_ROWS = ('voltage_magnitude_lower_cons', 'voltage_magnitude_upper_cons')
THERMAL_LIMIT_ROWS = ('branch_flow_limit', 'branch_flow_limit_ji')

_IPOPT_SUMMARY_ROW_RE = re.compile(
    r'^(Objective|Dual infeasibility|Constraint violation|Variable bound violation|Complementarity|'
    r'Overall NLP error)\.*:\s*([-+\d.eEinfINFnaN]+)\s+([-+\d.eEinfINFnaN]+)\s*$')


class UncoordinatedBenchmarkError(RuntimeError):
    pass


class StructuralCheckError(UncoordinatedBenchmarkError):
    pass


class CommonQConfigurationError(UncoordinatedBenchmarkError):
    pass


class ArmSolveFailure(UncoordinatedBenchmarkError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


# ======================================================================================================================
#  small helpers
# ======================================================================================================================
def require_single_scenario(planning_problem):
    """Raises unless every TN/DN network block has exactly one market and one operation scenario (SRP1)."""
    holders = [planning_problem.transmission_network] + [
        planning_problem.distribution_networks[n] for n in sorted(planning_problem.distribution_networks)]
    for holder in holders:
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                n_m = len(network.prob_market_scenarios)
                n_o = len(network.prob_operation_scenarios)
                if n_m != 1 or n_o != 1:
                    raise NotImplementedError(
                        f'uncoordinated benchmark: {holder.name} {year} {day} has {n_m} market x {n_o} operation '
                        'scenarios. The fixed-interface rows pin the TSO to the DSO\'s EXPECTED schedule, which the '
                        'per-scenario DSO flows need not follow above one scenario; this module is single-scenario '
                        '(SRP1) by ruling and refuses rather than generalise untested.')


def iter_network_blocks(planning_problem, models):
    """(kind, node_id, year, day, network_data, network, block) for the TSO and every DSO, in a fixed order; an
    agent whose entry in `models` is None is skipped."""
    tn = planning_problem.transmission_network
    if models.get('tso') is not None:
        for year in tn.years:
            for day in tn.days:
                yield 'TSO', None, year, day, tn, tn.network[year][day], models['tso'][year][day]
    if models.get('dso') is not None:
        for node_id in sorted(planning_problem.distribution_networks):
            dn = planning_problem.distribution_networks[node_id]
            for year in dn.years:
                for day in dn.days:
                    yield 'DSO', node_id, year, day, dn, dn.network[year][day], models['dso'][node_id][year][day]


def block_label(kind, node_id, year, day):
    return f'{kind}|{"-" if node_id is None else node_id}|{year}|{day}'


def declared_solve_count(planning_problem):
    """Solves of ONE arm evaluation: every DSO block once, then every TSO block once."""
    n_blocks = len(planning_problem.transmission_network.years) * len(planning_problem.transmission_network.days)
    return {'dso': len(planning_problem.distribution_networks) * n_blocks, 'tso': n_blocks,
            'total': (len(planning_problem.distribution_networks) + 1) * n_blocks}


def _active_objective_names(block):
    return sorted(o.name for o in block.component_data_objects(pe.Objective, active=True, descend_into=True))


def constraint_violation(con_data):
    """max(lower - body, body - upper, 0) at the current point (None if the body cannot be evaluated)."""
    try:
        body = float(pe.value(con_data.body))
    except Exception:  # noqa: BLE001
        return None
    worst = 0.0
    if con_data.has_lb():
        worst = max(worst, float(pe.value(con_data.lower)) - body)
    if con_data.has_ub():
        worst = max(worst, body - float(pe.value(con_data.upper)))
    return worst


def _has_free_variable(expr):
    for _v in identify_variables(expr, include_fixed=False):
        return True
    return False


# ======================================================================================================================
#  builders (no solve anywhere in this section)
# ======================================================================================================================
def _no_reverse_flow_rule(m, s_m, s_o, p):
    """W116 (Addendum 57): import only -- production's DSO interface expression pg_adn >= 0 (see NO_REVERSE_FLOW_ROW)."""
    return m.pg_adn[s_m, s_o, p] >= 0.0


def build_dso_arm_models(planning_problem, candidate_total_capacity, *, arm, curtailment_penalty,
                         no_reverse_flow=False):
    """Every DSO block of one arm, built and configured, NOT solved. `curtailment_penalty` (EUR/MWh) is the declared
    DECISION value of the tie-breaker `penalty_gen_curtailment`; it has no default -- the caller states it.
    `no_reverse_flow` (W116, Addendum 57): True adds the NO_REVERSE_FLOW_ROW family (pg_adn >= 0 in every scenario and
    period) to every block and declares its row count in the build record; the default False keeps the build of every
    earlier caller (spec v2 arms, W114/W115) unchanged.

    Returns (dso_models, build_record)."""
    if arm not in DSO_ARMS:
        raise ValueError(f'unknown DSO arm {arm!r}; expected one of {DSO_ARMS}')
    if curtailment_penalty is None or not math.isfinite(float(curtailment_penalty)) or curtailment_penalty < 0.0:
        raise ValueError(f'curtailment_penalty must be a declared finite value >= 0; got {curtailment_penalty!r}')
    if not isinstance(no_reverse_flow, bool):
        raise ValueError(f'no_reverse_flow must be a declared bool; got {no_reverse_flow!r}')
    require_single_scenario(planning_problem)
    distribution_networks = planning_problem.distribution_networks
    dso_models = {}
    record = {'arm': arm, 'decision_curtailment_penalty': float(curtailment_penalty),
              'no_reverse_flow': no_reverse_flow, 'blocks': {}}
    for node_id in sorted(distribution_networks):
        distribution_network = distribution_networks[node_id]
        distribution_network.update_data_with_candidate_solution(candidate_total_capacity)
        dso_model = distribution_network.build_model()
        distribution_network.update_model_with_candidate_solution(dso_model, candidate_total_capacity)
        for year in distribution_network.years:
            for day in distribution_network.days:
                network = distribution_network.network[year][day]
                block = dso_model[year][day]
                ref_node_id = network.get_reference_node_id()
                shared_ess_idx = network.get_shared_energy_storage_idx(ref_node_id)
                # The component set `create_distribution_networks_models_sequential` adds, with the same names and
                # production's own rule functions (the structural check compares against the certified block).
                block.expected_interface_vmag = pe.Var(block.periods, domain=pe.NonNegativeReals, initialize=1.00)
                block.expected_interface_pf_p = pe.Var(block.periods, domain=pe.Reals, initialize=0.0)
                block.expected_interface_pf_q = pe.Var(block.periods, domain=pe.Reals, initialize=0.0)
                block.expected_shared_ess_p = pe.Var(block.periods, domain=pe.Reals, initialize=0.0)
                block.expected_shared_ess_q = pe.Var(block.periods, domain=pe.Reals, initialize=0.0)
                block.expected_interface_vmag_def = pe.Constraint(block.periods, rule=partial(dn_interface_expected_vmag_rule, network=network))
                block.expected_interface_pf_p_def = pe.Constraint(block.periods, rule=partial(dn_interface_expected_pf_p_rule, network=network))
                block.expected_interface_pf_q_def = pe.Constraint(block.periods, rule=partial(dn_interface_expected_pf_q_rule, network=network))
                block.expected_shared_ess_p_def = pe.Constraint(block.periods, rule=partial(dn_interface_expected_sess_p_rule, network=network, shared_ess_idx=shared_ess_idx))
                block.expected_shared_ess_q_def = pe.Constraint(block.periods, rule=partial(dn_interface_expected_sess_q_rule, network=network, shared_ess_idx=shared_ess_idx))
                configure_shared_ess_operational_state(
                    block, shared_ess_idx, pe.value(block.shared_es_s_rated_fixed[shared_ess_idx]),
                    pe.value(block.shared_es_e_rated_fixed[shared_ess_idx]))
                # No day-ahead commitment and no settlement premium in an uncoordinated arrangement (as the retired
                # path, Addendum 38): alpha = 0. At one scenario nothing is wired.
                add_scenario_commitment_terms(block, network, distribution_network.params, premium_alpha=0.0)
                srp._set_row18_inactive_for_initialisation(block)
        dso_models[node_id] = dso_model

    if arm == ARM_PRICE_TAKER:
        # Production's own local pricing of the DSO subproblem (settlement weight 1 etc.), nothing more.
        srp._prepare_distribution_objectives_for_admm(distribution_networks, dso_models)

    for node_id in sorted(distribution_networks):
        distribution_network = distribution_networks[node_id]
        for year in distribution_network.years:
            for day in distribution_network.days:
                block = dso_models[node_id][year][day]
                block.penalty_gen_curtailment.set_value(float(curtailment_penalty))
                block_record = {'newly_fixed_flexibility_vars': [], 'trivial_rows_deactivated': [],
                                'trivial_row_max_violation': 0.0}
                if no_reverse_flow:
                    block.add_component(NO_REVERSE_FLOW_ROW, pe.Constraint(
                        block.scenarios_market, block.scenarios_operation, block.periods, rule=_no_reverse_flow_rule))
                    block_record['no_reverse_flow_rows_expected'] = (
                        len(block.scenarios_market) * len(block.scenarios_operation) * len(block.periods))
                if arm == ARM_PASSIVE:
                    fixed_names = []
                    for family in FLEXIBILITY_VAR_FAMILIES:
                        component = getattr(block, family, None)
                        if component is None:
                            continue
                        for var_data in component.values():
                            if not var_data.fixed:
                                var_data.fix(0.0)
                                fixed_names.append(var_data.name)
                    block_record['newly_fixed_flexibility_vars'] = fixed_names
                    deactivated, worst = [], 0.0
                    for family in PASSIVE_TRIVIAL_ROW_FAMILIES:
                        component = getattr(block, family, None)
                        if component is None:
                            continue
                        for con_data in component.values():
                            if con_data.active and not _has_free_variable(con_data.body):
                                violation = constraint_violation(con_data)
                                worst = max(worst, violation if violation is not None else math.inf)
                                con_data.deactivate()
                                deactivated.append(con_data.name)
                    if worst > SMALL_TOLERANCE:
                        raise StructuralCheckError(
                            f'passive arm DSO {node_id} {year} {day}: a row made constant by fixing the flexibility '
                            f'at 0 is violated by {worst!r} (> SMALL_TOLERANCE); refusing to deactivate it.')
                    block_record['trivial_rows_deactivated'] = deactivated
                    block_record['trivial_row_max_violation'] = worst
                record['blocks'][block_label('DSO', node_id, year, day)] = block_record
    return dso_models, record


def _fixed_interface_p_rule(m, dn, p):
    return m.expected_interface_pf_p[dn, p] == getattr(m, FIXED_INTERFACE_P_TARGET)[dn, p]


def _fixed_interface_q_rule(m, dn, p):
    return m.expected_interface_pf_q[dn, p] == getattr(m, FIXED_INTERFACE_Q_TARGET)[dn, p]


def _fixed_interface_v_rule(m, dn, p):
    return m.expected_interface_vmag[dn, p] == getattr(m, FIXED_INTERFACE_V_TARGET)[dn, p]


def _tso_interface_vmag_targets_pu(network, adn_nodes, interface_targets, year, day):
    """{ADN node id: [V target, TN p.u.] per period} = targets['v_kv'] / the TN node base kV. The ONE source of the
    voltage target for both the tracking penalty's `interface_vmag` and the check's fixed-side voltage pin (W94)."""
    out = {}
    for node_id in adn_nodes:
        v_base_tn = network.get_node_base_kv(node_id)
        out[node_id] = [float(v) / v_base_tn for v in interface_targets[node_id][year][day]['v_kv']]
    return out


def build_tso_arm_model(planning_problem, candidate_total_capacity, interface_targets, *, curtailment_penalty,
                        coupling, pin_interface_voltage):
    """The TSO blocks, built and configured, NOT solved. `interface_targets[node][year][day]` holds 'p_mw', 'q_mvar'
    (lists over periods) and 'v_kv' (used by the tracking-penalty coupling and by the voltage pin). `curtailment_penalty`
    is the declared DECISION tie-breaker; no default. `pin_interface_voltage` (no default; W94): True adds the
    interface-voltage pin rows -- permitted ONLY with 'fixed_interface' and used ONLY by the 24-solve check's fixed
    side; the arm passes TSO_ARM_PIN_INTERFACE_VOLTAGE (False). Returns (tso_model, build_record)."""
    if coupling not in TSO_COUPLINGS:
        raise ValueError(f'unknown TSO coupling {coupling!r}; expected one of {TSO_COUPLINGS}')
    if not isinstance(pin_interface_voltage, bool):
        raise ValueError(f'pin_interface_voltage must be a declared bool; got {pin_interface_voltage!r}')
    if pin_interface_voltage and coupling != TSO_COUPLING_FIXED:
        raise ValueError(f'the interface-voltage pin is defined only for {TSO_COUPLING_FIXED!r}; got {coupling!r}')
    if curtailment_penalty is None or not math.isfinite(float(curtailment_penalty)) or curtailment_penalty < 0.0:
        raise ValueError(f'curtailment_penalty must be a declared finite value >= 0; got {curtailment_penalty!r}')
    require_single_scenario(planning_problem)
    transmission_network = planning_problem.transmission_network
    distribution_networks = planning_problem.distribution_networks
    transmission_network.update_data_with_candidate_solution(candidate_total_capacity)
    tso_model = transmission_network.build_model()
    transmission_network.update_model_with_candidate_solution(tso_model, candidate_total_capacity)
    record = {'coupling': coupling, 'decision_curtailment_penalty': float(curtailment_penalty),
              'interface_voltage_pinned': pin_interface_voltage, 'blocks': {}}
    adn_nodes = list(transmission_network.active_distribution_network_nodes)

    for year in transmission_network.years:
        for day in transmission_network.days:
            network = transmission_network.network[year][day]
            block = tso_model[year][day]
            s_base = network.baseMVA
            block.active_distribution_networks = range(len(adn_nodes))
            p_target, q_target = {}, {}
            for dn in block.active_distribution_networks:
                adn_node_id = adn_nodes[dn]
                adn_node_idx = network.get_node_idx(adn_node_id)
                adn_load_idx = network.get_adn_load_idx(adn_node_id)
                interface_transf_rating = (distribution_networks[adn_node_id].network[year][day]
                                           .get_interface_branch_rating() / s_base)
                targets = interface_targets[adn_node_id][year][day]
                for p in block.periods:
                    p_target[dn, p] = float(targets['p_mw'][p]) / s_base
                    q_target[dn, p] = float(targets['q_mvar'][p]) / s_base
                for s_m in block.scenarios_market:
                    for s_o in block.scenarios_operation:
                        for p in block.periods:
                            # As `create_transmission_network_model`, line by line, with the DSO arm's schedule where
                            # production puts the DSO's initialisation consensus value.
                            if transmission_network.params.slacks.grid_operation.voltage:
                                block.slack_v_sqr_down[adn_node_idx, s_m, s_o, p].fix(0.00)
                                block.slack_v_sqr_up[adn_node_idx, s_m, s_o, p].fix(0.00)
                            block.pc[adn_load_idx, s_m, s_o, p].setub(None)
                            block.pc[adn_load_idx, s_m, s_o, p].setlb(None)
                            block.qc[adn_load_idx, s_m, s_o, p].setub(None)
                            block.qc[adn_load_idx, s_m, s_o, p].setlb(None)
                            fix_or_set(block.pc[adn_load_idx, s_m, s_o, p], p_target[dn, p])
                            fix_or_set(block.qc[adn_load_idx, s_m, s_o, p], q_target[dn, p])
                            block.flex_p_up[adn_load_idx, s_m, s_o, p].fix(0.00)
                            block.flex_p_down[adn_load_idx, s_m, s_o, p].fix(0.00)
                            block.flex_q_up[adn_load_idx, s_m, s_o, p].fix(0.00)
                            block.flex_q_down[adn_load_idx, s_m, s_o, p].fix(0.00)
                            if (s_m, s_o) == sess_na_scenario(block):
                                block.interface_delta_p[dn, s_m, s_o, p].fixed = False
                                block.interface_delta_q[dn, s_m, s_o, p].fixed = False
                                block.interface_delta_p[dn, s_m, s_o, p].setlb(-interface_transf_rating)
                                block.interface_delta_p[dn, s_m, s_o, p].setub(interface_transf_rating)
                                block.interface_delta_q[dn, s_m, s_o, p].setlb(-interface_transf_rating)
                                block.interface_delta_q[dn, s_m, s_o, p].setub(interface_transf_rating)

            block.expected_interface_vmag = pe.Var(block.active_distribution_networks, block.periods, domain=pe.NonNegativeReals, initialize=1.0)
            block.expected_interface_pf_p = pe.Var(block.active_distribution_networks, block.periods, domain=pe.Reals, initialize=0.0)
            block.expected_interface_pf_q = pe.Var(block.active_distribution_networks, block.periods, domain=pe.Reals, initialize=0.0)
            block.expected_shared_ess_p = pe.Var(block.shared_energy_storages, block.periods, domain=pe.Reals, initialize=0.0)
            block.expected_shared_ess_q = pe.Var(block.shared_energy_storages, block.periods, domain=pe.Reals, initialize=0.0)
            block.expected_interface_vmag_def = pe.Constraint(block.active_distribution_networks, block.periods, rule=partial(tn_interface_expected_vmag_rule, network=network))
            block.expected_interface_pf_p_def = pe.Constraint(block.active_distribution_networks, block.periods, rule=partial(tn_interface_expected_pf_p_rule, network=network))
            block.expected_interface_pf_q_def = pe.Constraint(block.active_distribution_networks, block.periods, rule=partial(tn_interface_expected_pf_q_rule, network=network))
            block.expected_shared_ess_p_def = pe.Constraint(block.shared_energy_storages, block.periods, rule=partial(tn_interface_expected_sess_p_rule, network=network))
            block.expected_shared_ess_q_def = pe.Constraint(block.shared_energy_storages, block.periods, rule=partial(tn_interface_expected_sess_q_rule, network=network))
            for e in block.shared_energy_storages:
                configure_shared_ess_operational_state(block, e, pe.value(block.shared_es_s_rated_fixed[e]),
                                                       pe.value(block.shared_es_e_rated_fixed[e]))
            add_scenario_commitment_terms(block, network, transmission_network.params, premium_alpha=0.0)

            if coupling == TSO_COUPLING_FIXED:
                block.add_component(FIXED_INTERFACE_P_TARGET, pe.Param(
                    block.active_distribution_networks, block.periods, mutable=True, domain=pe.Reals,
                    initialize=p_target))
                block.add_component(FIXED_INTERFACE_Q_TARGET, pe.Param(
                    block.active_distribution_networks, block.periods, mutable=True, domain=pe.Reals,
                    initialize=q_target))
                block.add_component(FIXED_INTERFACE_P_ROW, pe.Constraint(
                    block.active_distribution_networks, block.periods, rule=_fixed_interface_p_rule))
                block.add_component(FIXED_INTERFACE_Q_ROW, pe.Constraint(
                    block.active_distribution_networks, block.periods, rule=_fixed_interface_q_rule))
            n_row_families = 0
            if coupling == TSO_COUPLING_FIXED:
                n_row_families = 3 if pin_interface_voltage else 2
            if pin_interface_voltage:
                v_targets = _tso_interface_vmag_targets_pu(network, adn_nodes, interface_targets, year, day)
                v_target = {(dn, p): v_targets[adn_nodes[dn]][p]
                            for dn in block.active_distribution_networks for p in block.periods}
                block.add_component(FIXED_INTERFACE_V_TARGET, pe.Param(
                    block.active_distribution_networks, block.periods, mutable=True, domain=pe.Reals,
                    initialize=v_target))
                block.add_component(FIXED_INTERFACE_V_ROW, pe.Constraint(
                    block.active_distribution_networks, block.periods, rule=_fixed_interface_v_rule))
            record['blocks'][block_label('TSO', None, year, day)] = {
                'n_fixed_rows_expected': (n_row_families * len(block.active_distribution_networks)
                                          * len(block.periods)),
                'interface_voltage_pinned': pin_interface_voltage}

    srp._prepare_transmission_objectives_for_admm(transmission_network, tso_model)
    for year in transmission_network.years:
        for day in transmission_network.days:
            tso_model[year][day].penalty_gen_curtailment.set_value(float(curtailment_penalty))

    if coupling == TSO_COUPLING_TRACKING_PENALTY:
        for year in transmission_network.years:
            for day in transmission_network.days:
                network = transmission_network.network[year][day]
                year_day_vmag = _tso_interface_vmag_targets_pu(network, adn_nodes, interface_targets, year, day)
                year_day_pf = {}
                for node_id in adn_nodes:
                    targets = interface_targets[node_id][year][day]
                    year_day_pf[node_id] = {'p': [float(v) for v in targets['p_mw']],
                                            'q': [float(v) for v in targets['q_mvar']]}
                srp._add_tso_scenario_tracking_penalty(tso_model[year][day], network, adn_nodes, year_day_vmag,
                                                       year_day_pf)
    return tso_model, record


def set_tso_interface_targets(planning_problem, tso_model, interface_targets):
    """Moves the fixed-interface targets (mutable Params) and the fixed `pc`/`qc` to a new DSO schedule, with no
    rebuild. Returns the largest absolute target change in MW / MVAr. Refuses a voltage-pinned model (the coupling
    check's fixed side): it would move P/Q and silently leave the voltage pin behind."""
    transmission_network = planning_problem.transmission_network
    adn_nodes = list(transmission_network.active_distribution_network_nodes)
    largest = 0.0
    for year in transmission_network.years:
        for day in transmission_network.days:
            network = transmission_network.network[year][day]
            block = tso_model[year][day]
            if hasattr(block, FIXED_INTERFACE_V_TARGET):
                raise ValueError(f'{block_label("TSO", None, year, day)}: set_tso_interface_targets does not move the '
                                 'interface-voltage pin (coupling-check fixed side only)')
            s_base = network.baseMVA
            p_param = getattr(block, FIXED_INTERFACE_P_TARGET)
            q_param = getattr(block, FIXED_INTERFACE_Q_TARGET)
            for dn, node_id in enumerate(adn_nodes):
                adn_load_idx = network.get_adn_load_idx(node_id)
                targets = interface_targets[node_id][year][day]
                for p in block.periods:
                    p_new = float(targets['p_mw'][p]) / s_base
                    q_new = float(targets['q_mvar'][p]) / s_base
                    largest = max(largest, abs(p_new - pe.value(p_param[dn, p])) * s_base,
                                  abs(q_new - pe.value(q_param[dn, p])) * s_base)
                    p_param[dn, p].set_value(p_new)
                    q_param[dn, p].set_value(q_new)
                    for s_m in block.scenarios_market:
                        for s_o in block.scenarios_operation:
                            fix_or_set(block.pc[adn_load_idx, s_m, s_o, p], p_new)
                            fix_or_set(block.qc[adn_load_idx, s_m, s_o, p], q_new)
    return largest


def get_dso_interface_schedule(planning_problem, dso_models):
    """The DSOs' own expected interface schedule: P [MW], Q [MVAr] (DN baseMVA), V [kV] and V [DN p.u.]."""
    schedule = {}
    for node_id in sorted(planning_problem.distribution_networks):
        distribution_network = planning_problem.distribution_networks[node_id]
        schedule[node_id] = {}
        for year in distribution_network.years:
            schedule[node_id][year] = {}
            for day in distribution_network.days:
                network = distribution_network.network[year][day]
                block = dso_models[node_id][year][day]
                s_base = network.baseMVA
                v_base = network.get_node_base_kv(network.get_reference_node_id())
                periods = list(block.periods)
                v_pu = [float(pe.value(block.expected_interface_vmag[p])) for p in periods]
                schedule[node_id][year][day] = {
                    'p_mw': [float(pe.value(block.expected_interface_pf_p[p])) * s_base for p in periods],
                    'q_mvar': [float(pe.value(block.expected_interface_pf_q[p])) * s_base for p in periods],
                    'v_pu_dn': v_pu,
                    'v_kv': [v * v_base for v in v_pu],
                }
    return schedule


def get_tso_interface_schedule(planning_problem, tso_model):
    """The TSO's own expected interface values per ADN node: P [MW], Q [MVAr], V [kV], V [TN p.u.]."""
    transmission_network = planning_problem.transmission_network
    adn_nodes = list(transmission_network.active_distribution_network_nodes)
    schedule = {}
    for dn, node_id in enumerate(adn_nodes):
        schedule[node_id] = {}
        for year in transmission_network.years:
            schedule[node_id][year] = {}
            for day in transmission_network.days:
                network = transmission_network.network[year][day]
                block = tso_model[year][day]
                s_base = network.baseMVA
                v_base = network.get_node_base_kv(node_id)
                periods = list(block.periods)
                v_pu = [float(pe.value(block.expected_interface_vmag[dn, p])) for p in periods]
                schedule[node_id][year][day] = {
                    'p_mw': [float(pe.value(block.expected_interface_pf_p[dn, p])) * s_base for p in periods],
                    'q_mvar': [float(pe.value(block.expected_interface_pf_q[dn, p])) * s_base for p in periods],
                    'v_pu_tn': v_pu,
                    'v_kv': [v * v_base for v in v_pu],
                }
    return schedule


# ======================================================================================================================
#  structural check (ruling 3)
# ======================================================================================================================
_STRUCTURE_CTYPES = ((pe.Var, 'Var'), (pe.Constraint, 'Constraint'), (pe.Param, 'Param'),
                     (pe.Objective, 'Objective'), (pe.Expression, 'Expression'))


def block_structure(block):
    """Component map and counts of one block (Sets and Suffixes excluded)."""
    components = {}
    for ctype, label in _STRUCTURE_CTYPES:
        for component in block.component_objects(ctype, descend_into=True):
            components[component.name] = label
    n_var = 0
    fixed_names = set()
    for var_data in block.component_data_objects(pe.Var, descend_into=True):
        n_var += 1
        if var_data.fixed:
            fixed_names.add(var_data.name)
    n_con = 0
    n_con_active = 0
    for con_data in block.component_data_objects(pe.Constraint, active=None, descend_into=True):
        n_con += 1
        if con_data.active:
            n_con_active += 1
    return {'components': components, 'n_var': n_var, 'n_var_fixed': len(fixed_names),
            'n_var_free': n_var - len(fixed_names), 'fixed_var_names': fixed_names, 'n_con': n_con,
            'n_con_active': n_con_active, 'active_objectives': _active_objective_names(block)}


def structure_summary_for_record(structure):
    """JSON-safe copy (the fixed-name set is replaced by its size)."""
    out = {k: v for k, v in structure.items() if k != 'fixed_var_names'}
    out['components'] = dict(sorted(structure['components'].items()))
    return out


def coordinated_reference_structure(planning_problem, certified_models):
    """The structure of every certified coordinated TSO/DSO block, keyed by `block_label`."""
    return {block_label(kind, node_id, year, day): block_structure(block)
            for kind, node_id, year, day, _nd, _net, block in iter_network_blocks(planning_problem, certified_models)}


def check_arm_block_structure(arm_block, reference, *, label, expected_newly_fixed=(), trivial_rows_deactivated=(),
                              trivial_row_families=(), fixed_row_components=(), expected_fixed_rows=0,
                              extra_components=()):
    """Ruling 3's check on ONE block. Raises StructuralCheckError listing every failure; returns the record."""
    arm = block_structure(arm_block)
    failures = []
    ref_components = reference['components']
    arm_components = arm['components']
    missing = {n: t for n, t in ref_components.items() if n not in arm_components}
    extra = {n: t for n, t in arm_components.items() if n not in ref_components}
    for name, ctype in sorted(missing.items()):
        if ctype in ('Var', 'Constraint'):
            failures.append(f'{ctype} component {name!r} of the coordinated block is absent from the arm (declared '
                            f'removed consensus rows: {REMOVED_CONSENSUS_ROWS})')
        elif name not in CONSENSUS_TERM_COMPONENTS:
            failures.append(f'{ctype} component {name!r} of the coordinated block is absent from the arm and is not a '
                            'declared consensus term')
    for name, ctype in sorted(arm_components.items()):
        if name in CONSENSUS_TERM_COMPONENTS:
            failures.append(f'unremoved consensus term {ctype} {name!r} in the arm')
    allowed_extra = set(fixed_row_components) | set(extra_components)
    for name, ctype in sorted(extra.items()):
        if name not in allowed_extra:
            failures.append(f'undeclared {ctype} component {name!r} in the arm')
    for name in sorted(allowed_extra):
        if name not in extra:
            failures.append(f'declared component {name!r} is absent from the arm')
    if arm['n_var'] != reference['n_var']:
        failures.append(f"Var count {arm['n_var']} != coordinated {reference['n_var']}")
    newly_fixed = arm['fixed_var_names'] - reference['fixed_var_names']
    newly_freed = reference['fixed_var_names'] - arm['fixed_var_names']
    expected_newly_fixed = set(expected_newly_fixed)
    if newly_fixed != expected_newly_fixed:
        unexpected = sorted(newly_fixed - expected_newly_fixed)
        not_fixed = sorted(expected_newly_fixed - newly_fixed)
        failures.append(f'newly fixed Vars differ from the declared set: {len(unexpected)} unexpected '
                        f'(first {unexpected[:5]}), {len(not_fixed)} declared but not fixed (first {not_fixed[:5]})')
    if newly_freed:
        failures.append(f'{len(newly_freed)} Vars fixed in the coordinated block are free in the arm '
                        f'(first {sorted(newly_freed)[:5]})')
    for name in trivial_rows_deactivated:
        family = name.split('[', 1)[0]
        if family not in trivial_row_families:
            failures.append(f'deactivated row {name!r} is not in the declared trivial-row families {trivial_row_families}')
    fixed_rows_counted = 0
    for component_name in fixed_row_components:
        component = arm_block.component(component_name)
        if component is not None and component.ctype is pe.Constraint:
            fixed_rows_counted += sum(1 for c in component.values() if c.active)
    if fixed_rows_counted != int(expected_fixed_rows):
        failures.append(f'fixed rows counted {fixed_rows_counted} != declared {expected_fixed_rows}')
    expected_active = (reference['n_con_active'] - REMOVED_CONSENSUS_ROWS - len(trivial_rows_deactivated)
                       + int(expected_fixed_rows))
    if arm['n_con_active'] != expected_active:
        failures.append(f"active rows {arm['n_con_active']} != coordinated {reference['n_con_active']} - removed "
                        f'consensus rows {REMOVED_CONSENSUS_ROWS} - trivial rows deactivated '
                        f'{len(trivial_rows_deactivated)} + fixed rows {expected_fixed_rows} = {expected_active}')
    if arm['active_objectives'] != ['objective']:
        failures.append(f"active objectives {arm['active_objectives']} != ['objective']")
    record = {
        'label': label, 'passed': not failures, 'failures': failures,
        'arm': structure_summary_for_record(arm),
        'coordinated': {k: reference[k] for k in ('n_var', 'n_var_fixed', 'n_var_free', 'n_con', 'n_con_active',
                                                  'active_objectives')},
        'removed_consensus_terms': sorted(missing),
        'removed_consensus_rows': REMOVED_CONSENSUS_ROWS,
        'fixed_rows': int(expected_fixed_rows),
        'trivial_rows_deactivated': len(trivial_rows_deactivated),
        'newly_fixed_vars': len(newly_fixed),
        'arithmetic': (f"rows {arm['n_con_active']} = {reference['n_con_active']} - {REMOVED_CONSENSUS_ROWS} - "
                       f'{len(trivial_rows_deactivated)} + {expected_fixed_rows}; vars {arm["n_var"]} = '
                       f"{reference['n_var']}; free {arm['n_var_free']} = {reference['n_var_free']} - "
                       f'{len(expected_newly_fixed)}'),
    }
    if failures:
        raise StructuralCheckError(f'structural check FAILED for {label}: ' + ' | '.join(failures))
    return record


def dso_interface_voltage_pin(block, network):
    """The DSO reference-bus `e` bounds: pinned at the setpoint when its width is <= 2 x SMALL_TOLERANCE."""
    ref_idx = network.get_node_idx(network.get_reference_node_id())
    widths, centres = [], []
    for s_m in block.scenarios_market:
        for s_o in block.scenarios_operation:
            for p in block.periods:
                var_data = block.e[ref_idx, s_m, s_o, p]
                if var_data.lb is None or var_data.ub is None:
                    widths.append(math.inf)
                    centres.append(None)
                else:
                    widths.append(var_data.ub - var_data.lb)
                    centres.append(0.5 * (var_data.ub + var_data.lb))
    return {'max_width_pu': max(widths), 'pinned': max(widths) <= 2.0 * SMALL_TOLERANCE + 1e-12,
            'setpoint_pu': sorted({c for c in centres if c is not None})}


def check_arm_structures(planning_problem, arm_models, reference_structure, *, dso_build_record=None,
                         tso_build_record=None, tso_coupling=None):
    """Runs `check_arm_block_structure` on every DSO block in `arm_models['dso']` and every TSO block in
    `arm_models['tso']` (either may be absent). Raises on the first failing block."""
    records = {}
    if arm_models.get('dso') is not None:
        for node_id in sorted(planning_problem.distribution_networks):
            dn = planning_problem.distribution_networks[node_id]
            for year in dn.years:
                for day in dn.days:
                    label = block_label('DSO', node_id, year, day)
                    block = arm_models['dso'][node_id][year][day]
                    built = (dso_build_record or {}).get('blocks', {}).get(label, {})
                    # W116: the no-reverse-flow rows are the DSO arm's declared added rows (0 when not built)
                    nrf_rows = int(built.get('no_reverse_flow_rows_expected', 0))
                    rec = check_arm_block_structure(
                        block, reference_structure[label], label=label,
                        expected_newly_fixed=built.get('newly_fixed_flexibility_vars', ()),
                        trivial_rows_deactivated=built.get('trivial_rows_deactivated', ()),
                        trivial_row_families=PASSIVE_TRIVIAL_ROW_FAMILIES,
                        fixed_row_components=(NO_REVERSE_FLOW_ROW,) if nrf_rows else (),
                        expected_fixed_rows=nrf_rows)
                    rec['no_reverse_flow_rows'] = nrf_rows
                    pin = dso_interface_voltage_pin(block, dn.network[year][day])
                    if not pin['pinned']:
                        raise StructuralCheckError(f'{label}: the DSO interface voltage is not pinned at the setpoint '
                                                   f'(e[ref] bound width {pin["max_width_pu"]!r})')
                    rec['interface_voltage_pin'] = pin
                    records[label] = rec
    if arm_models.get('tso') is not None:
        tn = planning_problem.transmission_network
        pinned = (tso_build_record or {}).get('interface_voltage_pinned')
        if not isinstance(pinned, bool):
            raise StructuralCheckError(f'no declared interface_voltage_pinned in the TSO build record (got {pinned!r})')
        if pinned and tso_coupling != TSO_COUPLING_FIXED:
            raise StructuralCheckError(f'interface voltage pinned under coupling {tso_coupling!r}')
        if tso_coupling == TSO_COUPLING_FIXED:
            fixed_rows = (FIXED_INTERFACE_P_TARGET, FIXED_INTERFACE_Q_TARGET, FIXED_INTERFACE_P_ROW,
                          FIXED_INTERFACE_Q_ROW)
            if pinned:
                fixed_rows += (FIXED_INTERFACE_V_TARGET, FIXED_INTERFACE_V_ROW)
            extra = ()
        elif tso_coupling == TSO_COUPLING_TRACKING_PENALTY:
            fixed_rows = ()
            extra = TRACKING_PENALTY_COMPONENTS
        else:
            raise ValueError(f'tso_coupling must be declared ({TSO_COUPLINGS}); got {tso_coupling!r}')
        for year in tn.years:
            for day in tn.days:
                label = block_label('TSO', None, year, day)
                block = arm_models['tso'][year][day]
                expected = (tso_build_record or {}).get('blocks', {}).get(label, {}).get('n_fixed_rows_expected')
                if expected is None:
                    raise StructuralCheckError(f'{label}: no declared fixed-row count in the TSO build record')
                records[label] = check_arm_block_structure(
                    block, reference_structure[label], label=label, fixed_row_components=fixed_rows,
                    expected_fixed_rows=expected, extra_components=extra)
                records[label]['interface_voltage_pinned'] = pinned
    return records


# ======================================================================================================================
#  starts
# ======================================================================================================================
def extract_block_values(block):
    """{Var data name: value} for every Var with a value (the warm-start source)."""
    return {v.name: v.value for v in block.component_data_objects(pe.Var, descend_into=True) if v.value is not None}


def extract_model_values(planning_problem, models):
    return {block_label(kind, node_id, year, day): extract_block_values(block)
            for kind, node_id, year, day, _nd, _net, block in iter_network_blocks(planning_problem, models)}


def apply_warm_values(block, values):
    """Primal warm start: every FREE Var present in `values` takes that value (fixed Vars are never touched). IPOPT
    is still called without multiplier warm start (the arm is a different problem from the coordinated one)."""
    n_set = n_skipped_fixed = n_missing = n_outside_bounds = 0
    for var_data in block.component_data_objects(pe.Var, descend_into=True):
        if var_data.fixed:
            n_skipped_fixed += 1
            continue
        value = values.get(var_data.name)
        if value is None:
            n_missing += 1
            continue
        var_data.set_value(value, skip_validation=True)
        n_set += 1
        if (var_data.lb is not None and value < var_data.lb) or (var_data.ub is not None and value > var_data.ub):
            n_outside_bounds += 1
    return {'n_set': n_set, 'n_skipped_fixed': n_skipped_fixed, 'n_missing': n_missing,
            'n_outside_arm_bounds': n_outside_bounds}


def apply_perturbation(block, *, seed, delta, label):
    """Declared perturbation of the current point of ONE block (applied after `apply_warm_values`):
        for every FREE Var, in component_data_objects(pe.Var, sort=True) order, draw u ~ Uniform[-1, 1) from
        numpy Generator(PCG64(SeedSequence([seed, crc32(label)]))) -- one draw per free Var, valued or not --
        and set v <- clip(v * (1 + delta * u), lb, ub); a Var with value None or exactly 0.0 keeps its value.
    Reproducible from (seed, delta, label) alone; `label` = block_label(kind, node_id, year, day)."""
    if not (isinstance(seed, int) and seed >= 0):
        raise ValueError(f'perturbation seed must be a declared non-negative int; got {seed!r}')
    if not (isinstance(delta, float) and 0.0 < delta < 1.0):
        raise ValueError(f'perturbation delta must be a declared float in (0, 1); got {delta!r}')
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(label.encode('utf-8'))]))
    n_draws = n_changed = n_clipped = 0
    max_rel = 0.0
    for var_data in block.component_data_objects(pe.Var, sort=True, descend_into=True):
        if var_data.fixed:
            continue
        u = float(rng.uniform(-1.0, 1.0))
        n_draws += 1
        value = var_data.value
        if value is None or value == 0.0:
            continue
        new = value * (1.0 + delta * u)
        if var_data.lb is not None and new < var_data.lb:
            new = var_data.lb
            n_clipped += 1
        if var_data.ub is not None and new > var_data.ub:
            new = var_data.ub
            n_clipped += 1
        if new != value:
            n_changed += 1
            max_rel = max(max_rel, abs(new - value) / abs(value))
        var_data.set_value(new, skip_validation=True)
    return {'label': label, 'seed': seed, 'delta': delta, 'crc32_label': zlib.crc32(label.encode('utf-8')),
            'n_draws': n_draws, 'n_changed': n_changed, 'n_clipped': n_clipped, 'max_relative_change': max_rel}


def apply_start(planning_problem, arm_models, *, start, warm_values=None, perturbation=None, agents=('DSO', 'TSO')):
    """cold: nothing (the build's own initial point); warm_from_certified: `apply_warm_values`; perturbed: warm, then
    `apply_perturbation(seed, delta)`. Returns the per-block record."""
    if start not in STARTS:
        raise ValueError(f'unknown start {start!r}; expected one of {STARTS}')
    if start in (START_WARM, START_PERTURBED) and warm_values is None:
        raise ValueError(f'start {start!r} needs the certified warm values')
    if start == START_PERTURBED and not (isinstance(perturbation, dict) and 'seed' in perturbation
                                         and 'delta' in perturbation):
        raise ValueError('start perturbed needs a declared perturbation {"seed": int, "delta": float}')
    records = {}
    for kind, node_id, year, day, _nd, _net, block in iter_network_blocks(planning_problem, arm_models):
        if kind not in agents:
            continue
        label = block_label(kind, node_id, year, day)
        rec = {'start': start}
        if start in (START_WARM, START_PERTURBED):
            rec['warm'] = apply_warm_values(block, warm_values[label])
        if start == START_PERTURBED:
            rec['perturbation'] = apply_perturbation(block, seed=perturbation['seed'], delta=perturbation['delta'],
                                                     label=label)
        records[label] = rec
    return records


def _partial_models(planning_problem, dso_models=None, tso_model=None):
    """A models dict with only the requested agents (for iter_network_blocks-based helpers)."""
    return {'tso': tso_model, 'dso': dso_models}


# ======================================================================================================================
#  solves -- the ONE permitted call site
# ======================================================================================================================
def read_ipopt_attempt_summary(log_path, log_bytes):
    """The final error summary IPOPT printed in THIS attempt's byte range of its (appended) output file:
    {'scaled': {...}, 'unscaled': {...}} with keys objective, dual_infeasibility, constraint_violation,
    variable_bound_violation, complementarity, overall_nlp_error. Parse problems are recorded, not raised."""
    out = {'scaled': {}, 'unscaled': {}, 'parse_reason': None}
    if not log_path or not log_bytes or log_bytes[0] is None or log_bytes[1] is None:
        out['parse_reason'] = 'no log byte range recorded'
        return out
    try:
        with open(log_path, 'r', errors='replace') as handle:
            handle.seek(log_bytes[0])
            text = handle.read(max(0, log_bytes[1] - log_bytes[0]))
    except OSError as error:
        out['parse_reason'] = f'log unreadable: {error}'
        return out
    for line in text.splitlines():
        match = _IPOPT_SUMMARY_ROW_RE.match(line.strip())
        if not match:
            continue
        key = match.group(1).lower().replace(' ', '_')
        try:
            out['scaled'][key] = float(match.group(2))
            out['unscaled'][key] = float(match.group(3))
        except ValueError:
            out['parse_reason'] = f'unparseable summary row {line.strip()!r}'
    if 'dual_infeasibility' not in out['unscaled']:
        out['parse_reason'] = (out['parse_reason'] or '') + 'no final "Dual infeasibility" row in the attempt segment'
    return out


def _solve_block(planning_problem, network_data, network, block, *, kind, node_id, year, day, phase,
                 record_callback=None):
    """Solve ONE block through production (`Network.run_smopf`: primary attempt and production's own retry tiers),
    record it, hand the record to `record_callback`, and raise ArmSolveFailure if it did not succeed. No IPOPT
    multiplier warm start (`from_warm_start=False`); the primal start is whatever the block holds."""
    started = time.time()
    result = network.run_smopf(block, network_data.params, from_warm_start=False, print_header=True)
    wall = time.time() - started
    attempts = srp._drain_network_ipopt_solve_records(planning_problem, phase)
    for attempt in attempts:
        attempt['final_summary'] = read_ipopt_attempt_summary(attempt.get('log_path'), attempt.get('log_bytes'))
    succeeded = bool(srp._solver_result_succeeded(result))
    objective_value = None
    if succeeded:
        active = list(block.component_data_objects(pe.Objective, active=True, descend_into=True))
        objective_value = float(pe.value(active[0])) if len(active) == 1 else None
    record = {
        'block': block_label(kind, node_id, year, day), 'kind': kind, 'node_id': node_id, 'year': year, 'day': day,
        'phase': phase, 'succeeded': succeeded, 'summary': solver_result_summary(result),
        'solver_status': str(getattr(getattr(result, 'solver', None), 'status', None)),
        'termination_condition': str(getattr(getattr(result, 'solver', None), 'termination_condition', None)),
        'wall_s': wall, 'n_attempts': len(attempts), 'retried': len(attempts) != 1,
        'attempts': attempts, 'active_objectives': _active_objective_names(block),
        'active_objective_value': objective_value, 'utc_end': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    if record_callback is not None:
        record_callback(record)
    if not succeeded:
        raise ArmSolveFailure(f"solve failed: {record['block']} phase {phase}: {record['summary']}", record)
    return result, record


def solve_dso_models(planning_problem, dso_models, *, phase, record_callback=None):
    results = {}
    for node_id in sorted(planning_problem.distribution_networks):
        dn = planning_problem.distribution_networks[node_id]
        results[node_id] = {}
        for year in dn.years:
            results[node_id][year] = {}
            for day in dn.days:
                results[node_id][year][day], _rec = _solve_block(
                    planning_problem, dn, dn.network[year][day], dso_models[node_id][year][day], kind='DSO',
                    node_id=node_id, year=year, day=day, phase=phase, record_callback=record_callback)
    return results


def solve_tso_model(planning_problem, tso_model, *, phase, record_callback=None):
    tn = planning_problem.transmission_network
    results = {}
    for year in tn.years:
        results[year] = {}
        for day in tn.days:
            results[year][day], _rec = _solve_block(
                planning_problem, tn, tn.network[year][day], tso_model[year][day], kind='TSO', node_id=None,
                year=year, day=day, phase=phase, record_callback=record_callback)
    return results


# ======================================================================================================================
#  THE NEW PRODUCTION FUNCTION (ruling 3)
# ======================================================================================================================
def run_operational_planning_uncoordinated(planning_problem, candidate_solution, *, arm, dso_curtailment_penalty,
                                           tso_curtailment_penalty, reference_structure, start, warm_values=None,
                                           perturbation=None, record_callback=None, no_reverse_flow=False):
    """One uncoordinated arm: the DSOs solve their production subproblem with the coupling removed (at the substation
    setpoint voltage), then the TSO solves its production subproblem with the interface P/Q hard-fixed (mutable
    Params) to the DSOs' schedule. Exactly `declared_solve_count(planning_problem)['total']` solves when every block
    succeeds at its primary attempt.

    Every keyword without a default must be stated by the caller:
      arm                      'passive' | 'price_taker'
      dso_curtailment_penalty  DECISION tie-breaker of the DSO blocks, EUR/MWh (ruled: passive 1, price-taker 0)
      tso_curtailment_penalty  DECISION tie-breaker of the TSO blocks, EUR/MWh (ruled 0)
      reference_structure      `coordinated_reference_structure(planning, certified_models)` -- the structural check
                               runs on every block at build time and raises StructuralCheckError
      start                    'cold' | 'warm_from_certified' | 'perturbed' (the latter two need `warm_values`
                               = `extract_model_values(planning, certified_models)`; 'perturbed' also needs
                               `perturbation` = {'seed': int, 'delta': float})
      no_reverse_flow          W116 (Addendum 57 Decision 1(b)): True = the DSO arm carries NO_REVERSE_FLOW_ROW
                               (import only); the default False is the spec-v2 arm, unchanged. The TSO arm is the same
                               either way.
    Returns {'models', 'results', 'dso_interface_schedule', 'tso_interface_schedule', 'build', 'structure',
    'start_records', 'declared_solves'}; `record_callback(record)` receives every per-solve record as it happens."""
    require_single_scenario(planning_problem)
    candidate_total_capacity = candidate_solution['total_capacity']
    declared = declared_solve_count(planning_problem)

    dso_models, dso_build = build_dso_arm_models(planning_problem, candidate_total_capacity, arm=arm,
                                                 curtailment_penalty=dso_curtailment_penalty,
                                                 no_reverse_flow=no_reverse_flow)
    structure = check_arm_structures(planning_problem, {'dso': dso_models}, reference_structure,
                                     dso_build_record=dso_build)
    start_records = apply_start(planning_problem, _partial_models(planning_problem, dso_models=dso_models),
                                start=start, warm_values=warm_values, perturbation=perturbation, agents=('DSO',))
    dso_results = solve_dso_models(planning_problem, dso_models, phase=f'{arm}:{start}:dso',
                                   record_callback=record_callback)
    dso_schedule = get_dso_interface_schedule(planning_problem, dso_models)

    tso_model, tso_build = build_tso_arm_model(planning_problem, candidate_total_capacity, dso_schedule,
                                               curtailment_penalty=tso_curtailment_penalty,
                                               coupling=TSO_COUPLING_FIXED,
                                               pin_interface_voltage=TSO_ARM_PIN_INTERFACE_VOLTAGE)
    structure.update(check_arm_structures(planning_problem, {'tso': tso_model}, reference_structure,
                                          tso_build_record=tso_build, tso_coupling=TSO_COUPLING_FIXED))
    start_records.update(apply_start(planning_problem, _partial_models(planning_problem, tso_model=tso_model),
                                     start=start, warm_values=warm_values, perturbation=perturbation,
                                     agents=('TSO',)))
    tso_results = solve_tso_model(planning_problem, tso_model, phase=f'{arm}:{start}:tso',
                                  record_callback=record_callback)
    return {
        'models': {'tso': tso_model, 'dso': dso_models},
        'results': {'tso': tso_results, 'dso': dso_results},
        'dso_interface_schedule': dso_schedule,
        'tso_interface_schedule': get_tso_interface_schedule(planning_problem, tso_model),
        'build': {'dso': dso_build, 'tso': tso_build},
        'structure': structure,
        'start_records': start_records,
        'declared_solves': declared,
    }


def coupling_check_voltage_pin(planning_problem, tso_model, interface_targets):
    """Zero solves. Per TSO block: whether the interface-voltage pin rows exist, how many are active, and the largest
    |pin target - the tracking penalty's V target| (both from `_tso_interface_vmag_targets_pu`, so 0.0 exactly when
    present)."""
    tn = planning_problem.transmission_network
    adn_nodes = list(tn.active_distribution_network_nodes)
    out = {}
    for year in tn.years:
        for day in tn.days:
            block = tso_model[year][day]
            row = block.component(FIXED_INTERFACE_V_ROW)
            param = block.component(FIXED_INTERFACE_V_TARGET)
            entry = {'present': row is not None and param is not None,
                     'n_active_rows': 0 if row is None else sum(1 for c in row.values() if c.active),
                     'n_rows_expected_if_pinned': len(adn_nodes) * len(block.periods),
                     'max_abs_target_minus_tracked_v_pu': None}
            if param is not None:
                tracked = _tso_interface_vmag_targets_pu(tn.network[year][day], adn_nodes, interface_targets,
                                                         year, day)
                entry['max_abs_target_minus_tracked_v_pu'] = max(
                    abs(pe.value(param[dn, p]) - tracked[node_id][p])
                    for dn, node_id in enumerate(adn_nodes) for p in block.periods)
            out[block_label('TSO', None, year, day)] = entry
    return out


def build_tso_coupling_check_models(planning_problem, candidate_solution, interface_targets, *,
                                    tso_curtailment_penalty, reference_structure):
    """Zero solves. The two sides of ruling 2's check, built, structurally checked, NOT solved: 'fixed_interface'
    with the interface-voltage pin (COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE -- the check's fixed side only;
    see the module docstring) and 'tracking_penalty' (unpinned). Raises StructuralCheckError unless the fixed side
    carries the pin on every block, at exactly the penalty's V target, and the penalty side carries none.
    Returns {coupling: {'model', 'build', 'structure', 'voltage_pin'}}."""
    require_single_scenario(planning_problem)
    pin_by_coupling = {TSO_COUPLING_FIXED: COUPLING_CHECK_FIXED_SIDE_PIN_INTERFACE_VOLTAGE,
                       TSO_COUPLING_TRACKING_PENALTY: False}
    out = {}
    for coupling in (TSO_COUPLING_FIXED, TSO_COUPLING_TRACKING_PENALTY):
        tso_model, build = build_tso_arm_model(planning_problem, candidate_solution['total_capacity'],
                                               interface_targets, curtailment_penalty=tso_curtailment_penalty,
                                               coupling=coupling, pin_interface_voltage=pin_by_coupling[coupling])
        structure = check_arm_structures(planning_problem, {'tso': tso_model}, reference_structure,
                                         tso_build_record=build, tso_coupling=coupling)
        pin = coupling_check_voltage_pin(planning_problem, tso_model, interface_targets)
        for label, entry in pin.items():
            if pin_by_coupling[coupling]:
                ok = (entry['present'] and entry['n_active_rows'] == entry['n_rows_expected_if_pinned']
                      and entry['max_abs_target_minus_tracked_v_pu'] == 0.0)
            else:
                ok = not entry['present'] and entry['n_active_rows'] == 0
            if not ok:
                raise StructuralCheckError(f'coupling check {coupling!r} {label}: interface-voltage pin '
                                           f'{"missing or wrong" if pin_by_coupling[coupling] else "unexpected"} '
                                           f'({entry})')
        out[coupling] = {'model': tso_model, 'build': build, 'structure': structure, 'voltage_pin': pin}
    return out


def run_tso_interface_coupling_check(planning_problem, candidate_solution, interface_targets, *,
                                     tso_curtailment_penalty, reference_structure, record_callback=None):
    """Ruling 2's 24-solve check: 'fixed_interface' WITH the interface-voltage pin (W94; the check's fixed side
    only) against 'tracking_penalty' at the SAME targets (2 x |years| x |days| solves). Both sides are built and
    checked (`build_tso_coupling_check_models`) BEFORE the first solve. Returns both models, per-block metrics, the
    structural records and the voltage-pin records."""
    out = build_tso_coupling_check_models(planning_problem, candidate_solution, interface_targets,
                                          tso_curtailment_penalty=tso_curtailment_penalty,
                                          reference_structure=reference_structure)
    for coupling in (TSO_COUPLING_FIXED, TSO_COUPLING_TRACKING_PENALTY):
        tso_model = out[coupling]['model']
        out[coupling]['results'] = solve_tso_model(planning_problem, tso_model, phase=f'coupling_check:{coupling}',
                                                   record_callback=record_callback)
        out[coupling]['metrics'] = tso_block_metrics(planning_problem, tso_model, interface_targets)
    return out


def tso_block_metrics(planning_problem, tso_model, interface_targets, only=None):
    """Per TSO block: the block's own gross cost (objective_function_rule minus the contracted settlement and the
    voltage pin -- the block term of gross_operational_cost), weighted and unweighted; the interface residual
    |expected TSO interface - target| in MW / MVAr; the tracking-penalty terms where they exist. `only`: an optional
    collection of block labels to evaluate (e.g. the one block a check solved); None = every block."""
    tn = planning_problem.transmission_network
    adn_nodes = list(tn.active_distribution_network_nodes)
    out = {}
    for year in tn.years:
        for day in tn.days:
            if only is not None and block_label('TSO', None, year, day) not in only:
                continue
            network = tn.network[year][day]
            block = tso_model[year][day]
            s_base = network.baseMVA
            local = (network.get_primal_value(block, tn.params)
                     - srp._get_local_interface_settlement(block, part='contracted')
                     - srp._get_local_voltage_pin(block))
            weight = srp._get_admm_block_weight(tn, year, day)
            residual_p = residual_q = residual_v_pu = 0.0
            for dn, node_id in enumerate(adn_nodes):
                targets = interface_targets[node_id][year][day]
                v_base = network.get_node_base_kv(node_id)
                for p in block.periods:
                    residual_p = max(residual_p, abs(pe.value(block.expected_interface_pf_p[dn, p]) * s_base
                                                     - float(targets['p_mw'][p])))
                    residual_q = max(residual_q, abs(pe.value(block.expected_interface_pf_q[dn, p]) * s_base
                                                     - float(targets['q_mvar'][p])))
                    residual_v_pu = max(residual_v_pu, abs(pe.value(block.expected_interface_vmag[dn, p])
                                                           - float(targets['v_kv'][p]) / v_base))
            entry = {'block_gross_cost_unweighted': float(local), 'block_weight': weight,
                     'block_gross_cost_weighted': float(weight * local),
                     'interface_residual_p_mw_max': residual_p, 'interface_residual_q_mvar_max': residual_q,
                     'interface_voltage_minus_target_pu_max': residual_v_pu}
            if hasattr(block, 'scenario_tracking_penalty'):
                entry['tracking_weight'] = float(pe.value(block.scenario_tracking_weight))
                entry['tracking_voltage_term'] = float(pe.value(block.scenario_tracking_voltage))
                entry['tracking_interface_power_term'] = float(pe.value(block.scenario_tracking_interface_power))
                entry['tracking_penalty_value'] = float(pe.value(block.scenario_tracking_penalty))
            out[block_label('TSO', None, year, day)] = entry
    return out


# ======================================================================================================================
#  common Q (ruling 3 + clarification)
# ======================================================================================================================
def _pricing_snapshot(planning_problem, models):
    snapshot = {}
    for kind, node_id, year, day, _nd, _net, block in iter_network_blocks(planning_problem, models):
        snapshot[block_label(kind, node_id, year, day)] = {
            name: float(pe.value(getattr(block, name))) for name in PRICING_PARAMS if hasattr(block, name)}
    return snapshot


def _restore_pricing(planning_problem, models, snapshot):
    for kind, node_id, year, day, _nd, _net, block in iter_network_blocks(planning_problem, models):
        for name, value in snapshot[block_label(kind, node_id, year, day)].items():
            getattr(block, name).set_value(value)


def curtailment_report(planning_problem, models):
    """RES curtailment per block from production's own definitional helper at weight 1 (MWh per representative day,
    scenario expectation), plus its day-weighted (years x days, undiscounted) and block-weighted totals per agent."""
    per_block = {}
    totals = {'TSO': {'mwh_rep_day_sum': 0.0, 'mwh_day_weighted': 0.0, 'eur_at_1_block_weighted': 0.0},
              'DSO': {'mwh_rep_day_sum': 0.0, 'mwh_day_weighted': 0.0, 'eur_at_1_block_weighted': 0.0}}
    for kind, node_id, year, day, network_data, network, block in iter_network_blocks(planning_problem, models):
        value = 0.0
        for s_m in block.scenarios_market:
            for s_o in block.scenarios_operation:
                probability = network.prob_market_scenarios[s_m] * network.prob_operation_scenarios[s_o]
                value += probability * float(pe.value(gen_curtailment_definitional_value(
                    block, network, s_m, s_o, network_data.params, 1.0)))
        day_weight = float(network_data.years[year]) * float(network_data.days[day])
        block_weight = srp._get_admm_block_weight(network_data, year, day)
        per_block[block_label(kind, node_id, year, day)] = {'mwh_rep_day': value, 'day_weight': day_weight,
                                                            'block_weight': block_weight}
        totals[kind]['mwh_rep_day_sum'] += value
        totals[kind]['mwh_day_weighted'] += day_weight * value
        totals[kind]['eur_at_1_block_weighted'] += block_weight * value
    return {'per_block': per_block, 'totals': totals,
            'definition': 'model_construction_helpers.gen_curtailment_definitional_value(weight=1.0) = '
                          'sum over curtaillable generators and periods of baseMVA x (pg_avail - pg) [MWh per '
                          'representative day]; expectation over scenarios'}


def evaluate_common_q(planning_problem, models, *, evaluation_curtailment_penalty, require_unchanged):
    """Q of a set of TSO/DSO models under the COMMON evaluation pricing, then every touched Param restored.

    `models` = {'tso', 'dso', 'esso'} (the ESSO dict only provides the salvage value). Pricing: production's
    `_prepare_transmission_objectives_for_admm` and `_prepare_distribution_objectives_for_admm` on every block, then
    `penalty_gen_curtailment := evaluation_curtailment_penalty` (declared, no default; ruled 0). With
    `require_unchanged=True` any Param the repricing would change raises CommonQConfigurationError BEFORE Q is
    evaluated (the gate's form: the certified models must already carry the common pricing). Returns the production
    recourse components, the per-block decomposition, the pricing audit and the curtailment report."""
    require_single_scenario(planning_problem)
    if evaluation_curtailment_penalty is None or not math.isfinite(float(evaluation_curtailment_penalty)):
        raise ValueError(f'evaluation_curtailment_penalty must be declared; got {evaluation_curtailment_penalty!r}')
    before = _pricing_snapshot(planning_problem, models)
    try:
        srp._prepare_transmission_objectives_for_admm(planning_problem.transmission_network, models['tso'])
        srp._prepare_distribution_objectives_for_admm(planning_problem.distribution_networks, models['dso'])
        for _kind, _node, _year, _day, _nd, _net, block in iter_network_blocks(planning_problem, models):
            block.penalty_gen_curtailment.set_value(float(evaluation_curtailment_penalty))
        after = _pricing_snapshot(planning_problem, models)
        changed = {}
        for label, params in before.items():
            diffs = {name: {'before': value, 'after': after[label][name]} for name, value in params.items()
                     if after[label][name] != value}
            if diffs:
                changed[label] = diffs
        if require_unchanged and changed:
            first = next(iter(changed.items()))
            raise CommonQConfigurationError(
                f'common-Q evaluation: {len(changed)} blocks do not carry the common evaluation pricing '
                f'(require_unchanged=True); first {first}')
        components = srp._get_operational_recourse_components(planning_problem, models)
        block_components = srp._get_operational_recourse_block_components(planning_problem, models)
        curtailment = curtailment_report(planning_problem, models)
    finally:
        _restore_pricing(planning_problem, models, before)
    return {
        'objective_convention': OBJECTIVE_CONVENTION,
        'evaluation_curtailment_penalty': float(evaluation_curtailment_penalty),
        'gross_operational_cost': components['gross_operational_cost'],
        'gross_operational_cost_hex': float(components['gross_operational_cost']).hex(),
        'recourse_components': components,
        'block_components': {block_label(k, n, y, d): v for (k, n, y, d), v in block_components.items()
                             if k != 'SALVAGE'},
        'salvage_block': block_components.get(('SALVAGE', None, None, None)),
        'pricing_before': before, 'pricing_changed_by_evaluation': changed,
        'curtailment': curtailment,
    }


def common_q_gate(evaluation, certified_recourse_components):
    """Bitwise comparison of the evaluation's recourse components with the certified cell's. PASS iff
    gross_operational_cost is bitwise equal; every other field is compared and reported."""
    fields = {}
    for name, certified in certified_recourse_components.items():
        mine = evaluation['recourse_components'].get(name)
        if isinstance(certified, dict):
            equal = isinstance(mine, dict) and {str(k): v for k, v in mine.items()} == {str(k): v for k, v in certified.items()}
        else:
            equal = (mine is not None and certified is not None and float(mine) == float(certified)
                     and float(mine).hex() == float(certified).hex())
        fields[name] = {'equal_bitwise': bool(equal), 'evaluated': mine, 'certified': certified}
    gross = float(evaluation['gross_operational_cost'])
    certified_gross = float(certified_recourse_components['gross_operational_cost'])
    bitwise = gross == certified_gross and gross.hex() == certified_gross.hex()
    return {
        'status': 'PASS_BITWISE' if bitwise else 'DIFFERS_EXPLANATION_REQUIRED',
        'passed': bool(bitwise),
        'gross_evaluated': gross, 'gross_evaluated_hex': gross.hex(),
        'gross_certified': certified_gross, 'gross_certified_hex': certified_gross.hex(),
        'difference': gross - certified_gross,
        'relative_difference': (gross - certified_gross) / certified_gross if certified_gross else None,
        'fields': fields, 'n_fields_bitwise_equal': sum(1 for f in fields.values() if f['equal_bitwise']),
        'n_fields': len(fields),
    }


# ======================================================================================================================
#  consistency (the SRP1 convention)
# ======================================================================================================================
def interface_voltage_mismatch(planning_problem, tso_model, dso_models):
    """Per ADN node and (year, day, hour): the TN's actual interface voltage (TSO expected_interface_vmag) against
    the voltage the DSO assumed (DSO expected_interface_vmag), through kV. Max |dV| per node and hour is over the
    (year, day) blocks."""
    tn = planning_problem.transmission_network
    adn_nodes = list(tn.active_distribution_network_nodes)
    entries = []
    per_node_hour = {}
    v_actual_dn_pu = {}
    for dn, node_id in enumerate(adn_nodes):
        distribution_network = planning_problem.distribution_networks[node_id]
        v_actual_dn_pu[node_id] = {}
        for year in tn.years:
            v_actual_dn_pu[node_id][year] = {}
            for day in tn.days:
                tnet = tn.network[year][day]
                dnet = distribution_network.network[year][day]
                v_base_tn = tnet.get_node_base_kv(node_id)
                v_base_dn = dnet.get_node_base_kv(dnet.get_reference_node_id())
                series = []
                for p in tso_model[year][day].periods:
                    v_tn_kv = float(pe.value(tso_model[year][day].expected_interface_vmag[dn, p])) * v_base_tn
                    v_dn_kv = float(pe.value(dso_models[node_id][year][day].expected_interface_vmag[p])) * v_base_dn
                    dv_dn_pu = (v_tn_kv - v_dn_kv) / v_base_dn
                    series.append(v_tn_kv / v_base_dn)
                    entries.append({'node_id': node_id, 'year': year, 'day': day, 'hour': p + 1,
                                    'v_tn_kv': v_tn_kv, 'v_dn_assumed_kv': v_dn_kv, 'dv_kv': v_tn_kv - v_dn_kv,
                                    'dv_dn_pu': dv_dn_pu})
                    key = (node_id, p + 1)
                    per_node_hour[key] = max(per_node_hour.get(key, 0.0), abs(dv_dn_pu))
                v_actual_dn_pu[node_id][year][day] = series
    return {
        'entries': entries,
        'max_abs_dv_dn_pu_per_node_hour': {f'{n}|{h}': v for (n, h), v in sorted(per_node_hour.items())},
        'max_abs_dv_dn_pu': max((abs(e['dv_dn_pu']) for e in entries), default=0.0),
        'v_actual_dn_pu': v_actual_dn_pu,
    }


def build_consistency_reevaluation_block(dso_block, network, *, v_actual_dn_pu):
    """A CLONE of a solved DSO arm block for the consistency re-evaluation: every DSO decision
    (DSO_DECISION_VAR_FAMILIES and the P/Q of every generator except the reference) FIXED at the arm's solution,
    the slacks in DSO_REEVALUATION_ZERO_SLACKS fixed at 0, the reference-bus voltage FIXED at the TN's actual value
    (e = V, f = 0), the voltage and thermal limit rows DEACTIVATED (evaluated afterwards by
    `consistency_violations`), node-voltage and reference-generator bounds RELAXED (their original bounds recorded),
    and every row left without a free variable deactivated with its residual recorded. The result is the DN power
    flow at the actual voltage. Returns (clone, record).
    W121 (benchmark spec v5; defect found by W120): the fixed reference-bus e and f are made ADMISSIBLE at the fixed
    value -- their bounds are cleared (and recorded under 'reference_voltage_setpoint_bounds_cleared') before they are
    fixed, as the loop below does for the unfixed e/f. Those bounds are the DSO's voltage SETPOINT, not a physical DN
    limit: model_construction_helpers.e_bounds gives the DN reference bus vg +/- SMALL_TOLERANCE (vg = 1.0 on SRP1:
    [0.9999, 1.0001]) and f_bounds +/- EQUALITY_TOLERANCE. Kept, the fixed e at a TN voltage outside that band made
    Pyomo's NL writer raise InfeasibleConstraintException before any IPOPT launch. The reference bus's PHYSICAL limits
    (vmag_sqr_bounds: node.v_min^2, node.v_max^2, hard at a BUS_REF node) are unchanged: recorded in
    'original_vmag_sqr_bounds' like every bus and checked afterwards by `consistency_violations` (hard)."""
    m = dso_block.clone()
    ref_idx = network.get_node_idx(network.get_reference_node_id())
    ref_gen = network.get_reference_gen_idx()
    record = {'n_decisions_fixed': 0, 'n_slacks_zeroed': 0, 'original_vmag_sqr_bounds': {},
              'original_ref_gen_bounds': {}, 'relaxed_row_families': list(VOLTAGE_LIMIT_ROWS + THERMAL_LIMIT_ROWS),
              'trivial_rows_deactivated': 0, 'trivial_row_max_violation': 0.0,
              'reference_voltage_setpoint_bounds_cleared': {}}
    for family in DSO_DECISION_VAR_FAMILIES:
        component = getattr(m, family, None)
        if component is None:
            continue
        for var_data in component.values():
            if not var_data.fixed:
                if var_data.value is None:
                    raise UncoordinatedBenchmarkError(f'{var_data.name} has no value to fix for the re-evaluation')
                var_data.fix(var_data.value, skip_validation=True)
                record['n_decisions_fixed'] += 1
    for g in m.generators:
        if g == ref_gen:
            continue
        for s_m in m.scenarios_market:
            for s_o in m.scenarios_operation:
                for p in m.periods:
                    for var_data in (m.pg[g, s_m, s_o, p], m.qg[g, s_m, s_o, p]):
                        if not var_data.fixed:
                            var_data.fix(var_data.value, skip_validation=True)
                            record['n_decisions_fixed'] += 1
    for family in DSO_REEVALUATION_ZERO_SLACKS:
        component = getattr(m, family, None)
        if component is None:
            continue
        for var_data in component.values():
            var_data.fix(0.0)
            record['n_slacks_zeroed'] += 1
    for s_m in m.scenarios_market:
        for s_o in m.scenarios_operation:
            for p in m.periods:
                # W121: the setpoint bounds cleared (recorded) so the fixed value is admissible; value unchanged
                for var_data, value in ((m.e[ref_idx, s_m, s_o, p], float(v_actual_dn_pu[p])),
                                        (m.f[ref_idx, s_m, s_o, p], 0.0)):
                    record['reference_voltage_setpoint_bounds_cleared'][var_data.name] = (var_data.lb, var_data.ub)
                    var_data.setlb(None)
                    var_data.setub(None)
                    var_data.fix(value, skip_validation=True)
                for var_data in (m.pg[ref_gen, s_m, s_o, p], m.qg[ref_gen, s_m, s_o, p]):
                    record['original_ref_gen_bounds'][var_data.name] = (var_data.lb, var_data.ub)
                    var_data.setlb(None)
                    var_data.setub(None)
    for var_data in m.vmag_sqr.values():
        record['original_vmag_sqr_bounds'][var_data.name] = (var_data.lb, var_data.ub)
        var_data.setlb(0.0)
        var_data.setub(None)
    for var_data in m.vmag.values():
        var_data.setlb(0.0)
        var_data.setub(None)
    for component_name in ('e', 'f'):
        for var_data in getattr(m, component_name).values():
            if not var_data.fixed:
                var_data.setlb(None)
                var_data.setub(None)
    for family in VOLTAGE_LIMIT_ROWS + THERMAL_LIMIT_ROWS:
        component = getattr(m, family, None)
        if component is None:
            continue
        for con_data in component.values():
            con_data.deactivate()
    # W116: a no-reverse-flow arm's rows are a DN limit on the (freed) reference generator -- deactivated here and
    # evaluated by `consistency_violations` (absent on every other arm: nothing changes there)
    nrf_component = getattr(m, NO_REVERSE_FLOW_ROW, None)
    if nrf_component is not None:
        for con_data in nrf_component.values():
            con_data.deactivate()
        record['relaxed_row_families'].append(NO_REVERSE_FLOW_ROW)
    worst = 0.0
    for con_data in list(m.component_data_objects(pe.Constraint, active=True, descend_into=True)):
        if not _has_free_variable(con_data.body):
            violation = constraint_violation(con_data)
            worst = max(worst, violation if violation is not None else math.inf)
            con_data.deactivate()
            record['trivial_rows_deactivated'] += 1
    record['trivial_row_max_violation'] = worst
    return m, record


def consistency_violations(reeval_block, record, arm_block, *, hard_tol, soft_excess_tol, thermal_tol):
    """Limit violations of the re-evaluated DN point, from production's own (deactivated) limit rows and the
    recorded original bounds:
      hard     vmag_sqr outside its ORIGINAL Var bounds (the band the DSO model admits at all) [p.u.^2], or the
               reference generator's P/Q outside its original bounds [p.u.; recorded under the same key]
      soft     violation of voltage_magnitude_lower/upper_cons with the voltage slack at 0 (beyond v_min/v_max)
               [p.u.^2]; `soft_excess` = soft - the arm's OWN voltage slack at the same index (a new or deeper
               violation the DSO did not choose)
      thermal  violation of branch_flow_limit / branch_flow_limit_ji (flow^2 against rating^2) [p.u.^2]
    Trigger for the one sequential pass (declared): any hard > hard_tol, any thermal > thermal_tol, or any
    soft_excess > soft_excess_tol."""
    hard, soft, thermal = [], [], []
    for var_data in reeval_block.vmag_sqr.values():
        lb, ub = record['original_vmag_sqr_bounds'][var_data.name]
        value = float(var_data.value)
        excess = max((lb - value) if lb is not None else 0.0, (value - ub) if ub is not None else 0.0, 0.0)
        if excess > 0.0:
            hard.append({'var': var_data.name, 'kind': 'vmag_sqr_band', 'vmag_sqr': value, 'bounds': [lb, ub],
                         'excess_pu2': excess})
    for family in VOLTAGE_LIMIT_ROWS:
        component = getattr(reeval_block, family, None)
        if component is None:
            continue
        slack_name = 'slack_v_sqr_down' if family.endswith('lower_cons') else 'slack_v_sqr_up'
        arm_slack = getattr(arm_block, slack_name, None)
        for index, con_data in component.items():
            violation = constraint_violation(con_data)
            if violation is None or violation <= 0.0:
                continue
            own = float(pe.value(arm_slack[index])) if arm_slack is not None and index in arm_slack else 0.0
            soft.append({'row': con_data.name, 'violation_pu2': violation, 'arm_own_slack_pu2': own,
                         'soft_excess_pu2': violation - own})
    for family in THERMAL_LIMIT_ROWS:
        component = getattr(reeval_block, family, None)
        if component is None:
            continue
        for con_data in component.values():
            violation = constraint_violation(con_data)
            if violation is not None and violation > 0.0:
                thermal.append({'row': con_data.name, 'violation_pu2': violation,
                                'flow_sqr': float(pe.value(con_data.body)),
                                'upper': float(pe.value(con_data.upper)) if con_data.has_ub() else None})
    for name, (lb, ub) in record.get('original_ref_gen_bounds', {}).items():
        value = float(pe.value(reeval_block.find_component(name)))
        excess = max((lb - value) if lb is not None else 0.0, (value - ub) if ub is not None else 0.0, 0.0)
        if excess > 0.0:
            hard.append({'var': name, 'kind': 'reference_generator_bound', 'value_pu': value, 'bounds': [lb, ub],
                         'excess_pu2': excess})
    # W116: the no-reverse-flow rows (present only on a no-reverse-flow arm), a hard DN limit [p.u.; recorded under the
    # same key as the reference-generator bounds]
    nrf_component = getattr(reeval_block, NO_REVERSE_FLOW_ROW, None)
    if nrf_component is not None:
        for con_data in nrf_component.values():
            violation = constraint_violation(con_data)
            if violation is not None and violation > 0.0:
                hard.append({'row': con_data.name, 'kind': 'no_reverse_flow', 'pg_adn_pu': float(pe.value(con_data.body)),
                             'excess_pu2': violation})
    max_hard = max((h['excess_pu2'] for h in hard), default=0.0)
    max_soft_excess = max((s['soft_excess_pu2'] for s in soft), default=0.0)
    max_thermal = max((t['violation_pu2'] for t in thermal), default=0.0)
    trigger = max_hard > hard_tol or max_thermal > thermal_tol or max_soft_excess > soft_excess_tol
    return {'hard': hard, 'soft': soft, 'thermal': thermal, 'max_hard_excess_pu2': max_hard,
            'max_soft_violation_pu2': max((s['violation_pu2'] for s in soft), default=0.0),
            'max_soft_excess_pu2': max_soft_excess, 'max_thermal_violation_pu2': max_thermal,
            'tolerances': {'hard_tol_pu2': hard_tol, 'soft_excess_tol_pu2': soft_excess_tol,
                           'thermal_tol_pu2': thermal_tol},
            'trigger_sequential_pass': bool(trigger)}


def reevaluate_dso_at_actual_voltage(planning_problem, dso_models, v_actual_dn_pu, *, tolerances,
                                     record_callback=None, phase='consistency:reevaluation'):
    """The consistency re-evaluation of every DSO block (one solve per block). Returns per-block violations, the
    change of the DSO interface P/Q against the arm's schedule, and the aggregate trigger."""
    blocks = {}
    trigger = False
    for node_id in sorted(planning_problem.distribution_networks):
        dn = planning_problem.distribution_networks[node_id]
        for year in dn.years:
            for day in dn.days:
                network = dn.network[year][day]
                arm_block = dso_models[node_id][year][day]
                clone, build = build_consistency_reevaluation_block(
                    arm_block, network, v_actual_dn_pu=v_actual_dn_pu[node_id][year][day])
                _result, solve_record = _solve_block(planning_problem, dn, network, clone, kind='DSO',
                                                     node_id=node_id, year=year, day=day, phase=phase,
                                                     record_callback=record_callback)
                violations = consistency_violations(clone, build, arm_block, hard_tol=tolerances['hard_tol_pu2'],
                                                    soft_excess_tol=tolerances['soft_excess_tol_pu2'],
                                                    thermal_tol=tolerances['thermal_tol_pu2'])
                s_base = network.baseMVA
                dp = max(abs(pe.value(clone.expected_interface_pf_p[p]) - pe.value(arm_block.expected_interface_pf_p[p]))
                         for p in clone.periods) * s_base
                dq = max(abs(pe.value(clone.expected_interface_pf_q[p]) - pe.value(arm_block.expected_interface_pf_q[p]))
                         for p in clone.periods) * s_base
                trigger = trigger or violations['trigger_sequential_pass']
                blocks[block_label('DSO', node_id, year, day)] = {
                    'build': {k: v for k, v in build.items() if k not in ('original_vmag_sqr_bounds',
                                                                          'original_ref_gen_bounds')},
                    'violations': violations, 'interface_p_change_mw_max': dp, 'interface_q_change_mvar_max': dq,
                    'solve': {k: solve_record[k] for k in ('succeeded', 'termination_condition', 'n_attempts')}}
                del clone
    return {'blocks': blocks, 'trigger_sequential_pass': bool(trigger)}


def pin_dso_interface_voltage(planning_problem, dso_models, v_target_dn_pu):
    """Moves every DSO block's reference-bus `e` bounds to the given per-period voltage +/- SMALL_TOLERANCE -- the
    same form `e_bounds` gives the setpoint -- for the sequential pass (DSO re-solved at the TN's actual voltage)."""
    largest_shift = 0.0
    for node_id in sorted(planning_problem.distribution_networks):
        dn = planning_problem.distribution_networks[node_id]
        for year in dn.years:
            for day in dn.days:
                network = dn.network[year][day]
                block = dso_models[node_id][year][day]
                ref_idx = network.get_node_idx(network.get_reference_node_id())
                for s_m in block.scenarios_market:
                    for s_o in block.scenarios_operation:
                        for p in block.periods:
                            var_data = block.e[ref_idx, s_m, s_o, p]
                            v = float(v_target_dn_pu[node_id][year][day][p])
                            old_centre = 0.5 * (var_data.lb + var_data.ub)
                            largest_shift = max(largest_shift, abs(v - old_centre))
                            var_data.setlb(v - SMALL_TOLERANCE)
                            var_data.setub(v + SMALL_TOLERANCE)
    return largest_shift


# ======================================================================================================================
#  lambda_t helpers (zero solves; read off persisted coordinated models)
# ======================================================================================================================
def interface_price_terms(planning_problem, certified_models):
    """Per ADN node, (year, day) and hour, from the persisted coordinated models:
      pi                  expected market price of the DN settlement (model_construction_helpers.expected_market_price)
      c_flex              DN flexibility price cost_flex (market-scenario expectation)
      lambda_dso_linear   pi + eff_dso * dual_pf_p_req / (r_pu * B_dn)
      lambda_dso_full     pi + eff_dso * [dual_pf_p_req + rho_pf (E - z)/r_pu] / (r_pu * B_dn)
      lambda_tso_linear   pi - eff_tso * dual_pf_p_req / (r_pu * B_tn)
      lambda_tso_full     pi - eff_tso * [dual_pf_p_req + rho_pf (E - z)/r_pu + gamma_pf (E - E_prev)/r_pu] / (r_pu * B_tn)
      lmp_tn_bus          dual(node_balance_p[TN bus]) * m_tso / B_tn    (W28's LMP)
      y0_dn_ref           dual(node_balance_p[DN ref bus]) * m_dso / B_dn (W31's y0)
    with eff = admm_objective_scale (= sigma / block weight), r_pu = interface rating / B of that side (the rating the
    AL term is normalised by), m = 1 when the active objective is `p58_rescaled_admm_objective` (already in EUR) and
    eff when it is `admm_objective` (divided by eff). The interface-P channel is scaled by sigma (through eff) and the
    interface rating only; S_ref (shared_ess_reference_rating_mva) and D5 (the ESSO AL scale) enter the ESS channel
    only (update_*_model_to_admm) and are therefore not undone here. This CORRECTS the wording of Addendum 49's
    clarification ("undo sigma, S_ref, D5"; accepted by the Planner in W94). Source, shared_resources_planning.py at
    a8c58da0: update_transmission_model_to_admm -- effective_scale = objective_scale / block_weight and the objective
    divided by it (L5137-5142), interface P/Q AL terms divided by interface_transf_rating (L5156-5157), S_ref only in
    the shared-ESS loop's shared_ess_rating (L5179); update_distribution_models_to_admm -- the same (L5402-5407,
    L5428-5429), S_ref only in shared_ess_rating (L5410-5415); D5 = al_scale_esso, stored as admm_esso_al_scale and
    multiplying ONLY the ESSO's AL terms in update_shared_energy_storage_model_to_admm (L5486, L5515-5518), never
    passed to the TSO/DSO updates. Signs are derived (stationarity of the
    expected-interface Vars and of interface_delta_p) and are CHECKED against the two independent nodal duals by the
    harness's units check; this function asserts nothing."""
    tn = planning_problem.transmission_network
    adn_nodes = list(tn.active_distribution_network_nodes)
    rows = []
    for dn_idx, node_id in enumerate(adn_nodes):
        distribution_network = planning_problem.distribution_networks[node_id]
        for year in tn.years:
            for day in tn.days:
                tnet = tn.network[year][day]
                dnet = distribution_network.network[year][day]
                t_block = certified_models['tso'][year][day]
                d_block = certified_models['dso'][node_id][year][day]
                b_tn, b_dn = tnet.baseMVA, dnet.baseMVA
                rating_mva = dnet.get_interface_branch_rating()
                r_pu_tn, r_pu_dn = rating_mva / b_tn, rating_mva / b_dn
                eff_tn = float(pe.value(t_block.admm_objective_scale))
                eff_dn = float(pe.value(d_block.admm_objective_scale))
                t_active = _active_objective_names(t_block)
                d_active = _active_objective_names(d_block)
                m_tn = 1.0 if t_active == ['p58_rescaled_admm_objective'] else (eff_tn if t_active == ['admm_objective'] else None)
                m_dn = 1.0 if d_active == ['p58_rescaled_admm_objective'] else (eff_dn if d_active == ['admm_objective'] else None)
                tn_bus_idx = tnet.get_node_idx(node_id)
                dn_ref_idx = dnet.get_node_idx(dnet.get_reference_node_id())
                s_m0, s_o0 = 0, 0
                gamma = float(pe.value(t_block.prox_gamma_pf)) if hasattr(t_block, 'prox_gamma_pf') else 0.0
                for p in t_block.periods:
                    pi = float(expected_market_price(dnet, p))
                    pi_tn = float(expected_market_price(tnet, p))
                    c_flex = sum(dnet.prob_market_scenarios[s] * dnet.cost_flex[s][p]
                                 for s in range(len(dnet.prob_market_scenarios)))
                    d_dual = float(pe.value(d_block.dual_pf_p_req[p]))
                    d_rho = float(pe.value(d_block.rho_pf))
                    d_e = float(pe.value(d_block.expected_interface_pf_p[p]))
                    d_z = float(pe.value(d_block.p_pf_req[p]))
                    t_dual = float(pe.value(t_block.dual_pf_p_req[dn_idx, p]))
                    t_rho = float(pe.value(t_block.rho_pf))
                    t_e = float(pe.value(t_block.expected_interface_pf_p[dn_idx, p]))
                    t_z = float(pe.value(t_block.p_pf_req[dn_idx, p]))
                    t_prev = (float(pe.value(t_block.prox_pf_p_prev[dn_idx, p]))
                              if hasattr(t_block, 'prox_pf_p_prev') else t_e)
                    d_grad_lin = d_dual / r_pu_dn
                    d_grad_full = (d_dual + d_rho * (d_e - d_z) / r_pu_dn) / r_pu_dn
                    t_grad_lin = t_dual / r_pu_tn
                    t_grad_full = (t_dual + t_rho * (t_e - t_z) / r_pu_tn + gamma * (t_e - t_prev) / r_pu_tn) / r_pu_tn
                    bal_t = t_block.node_balance_p[tn_bus_idx, s_m0, s_o0, p]
                    bal_d = d_block.node_balance_p[dn_ref_idx, s_m0, s_o0, p]
                    lmp = (float(t_block.dual[bal_t]) * m_tn / b_tn) if (m_tn is not None and bal_t in t_block.dual) else None
                    y0 = (float(d_block.dual[bal_d]) * m_dn / b_dn) if (m_dn is not None and bal_d in d_block.dual) else None
                    delta = t_block.interface_delta_p[dn_idx, s_m0, s_o0, p]
                    delta_interior = (delta.value is not None and delta.lb is not None and delta.ub is not None
                                      and delta.lb + 1e-6 < delta.value < delta.ub - 1e-6)
                    rows.append({
                        'node_id': node_id, 'year': year, 'day': day, 'hour': p + 1, 'period': p,
                        'pi': pi, 'pi_tn': pi_tn, 'c_flex': c_flex,
                        'lambda_dso_linear': pi + eff_dn * d_grad_lin / b_dn,
                        'lambda_dso_full': pi + eff_dn * d_grad_full / b_dn,
                        'lambda_tso_linear': pi - eff_tn * t_grad_lin / b_tn,
                        'lambda_tso_full': pi - eff_tn * t_grad_full / b_tn,
                        'lmp_tn_bus': lmp, 'y0_dn_ref': y0, 'tso_interface_delta_interior': bool(delta_interior),
                        'raw': {'dual_pf_p_req_dso': d_dual, 'dual_pf_p_req_tso': t_dual, 'rho_pf_dso': d_rho,
                                'rho_pf_tso': t_rho, 'gamma_pf_tso': gamma, 'E_dso_pu': d_e, 'z_dso_pu': d_z,
                                'E_tso_pu': t_e, 'z_tso_pu': t_z, 'eff_dso': eff_dn, 'eff_tso': eff_tn,
                                'rating_mva': rating_mva, 'B_dn': b_dn, 'B_tn': b_tn,
                                'active_objective_tso': t_active, 'active_objective_dso': d_active},
                    })
    return rows
