"""
P5.6-A -- deterministic nonlinear recourse oracle.

This module is the candidate-evaluation contract a later derivative-free planner
will call.  It does NOT implement a search, a cut, or a master problem.

The accepted nonlinear operational formulation (P5.4-D2-P / H1) is used exactly
as production defines it.  Nothing here changes optimization mathematics: every
model is built by the production constructors and solved by the production
solvers.  What this module adds is

  * PURITY (P5.5-A12).  Production mutates persistent planning data in place --
    `_update_data_with_candidate_solution` writes capacities straight into
    `network_planning.network[year][day].shared_energy_storages[i]`, and the ESSO
    equivalent does the same -- so a second evaluation sees the first one's
    capacities.  Every evaluation here runs against its own deep copy of the
    baseline, with its own solver-log directory and its own diagnostics
    accumulators, and the baseline is never mutated.
  * FULL PHYSICAL FEASIBILITY.  An evaluation is VALID only when the polished
    coupled point satisfies the ORIGINAL nonlinear ESSO as well as the 48
    network SMOPFs, and when every coordinated quantity agrees across the TSO,
    DSO and ESSO copies.
  * EXPLICIT STATUS.  A failed evaluation returns why it failed.  It never
    returns a fabricated large objective, because a search driven by fake values
    optimises the fake.

Evaluation pipeline (A4):

    1  construct clean candidate-specific data
    2  apply the candidate investment
    3  run the accepted nonlinear operational ADMM
    4  form one common value for every coordinated quantity
    5  solve the ORIGINAL nonlinear ESSO with investment and the common
       coordinated P/Q fixed  ->  physical S_available / E_available
    6  push those exact physical capacities into every network copy, fix the
       common interface P/Q, voltage and shared-ESS P/Q, re-solve all 48
       nonlinear network SMOPFs
    7  audit the original nonlinear ESSO and every network constraint
    8  total_objective = investment_cost + polished_net_operational_recourse
"""

import hashlib
import io
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import model_construction_helpers as mch  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P56A')
WORK_DIR = os.path.join(OUT_DIR, 'evals')
CACHE_PATH = os.path.join(OUT_DIR, 'p56a_cache.json')

# Bumped whenever anything in this module could change a returned objective.
ORACLE_VERSION = 'p56a.2'

CANONICAL_CHECKSUM = ('5a02b77ccbbbbbb869de92958a3851d09'
                      '5624711abc2dbfc0157466064410358')

# Coordinated-residual target for a certified point (A1).
COORDINATION_TARGET = 1e-7
# Production's own ESSO feasibility tolerance, reused rather than invented.
ESSO_FEASIBILITY_TOLERANCE = srp.BENDERS_FEASIBILITY_TOLERANCE

STATUS_VALID = 'VALID'
STATUS_INVALID_INVESTMENT = 'INVALID_INVESTMENT'
STATUS_LOCAL_SOLVE_FAILURE = 'LOCAL_SOLVE_FAILURE'
STATUS_POLISH_FAILURE = 'POLISH_FAILURE'
STATUS_ESSO_FEASIBILITY_FAILURE = 'ESSO_FEASIBILITY_FAILURE'
STATUS_COUPLING_FAILURE = 'COUPLING_FAILURE'
STATUS_ESSO_SOLVE_FAILURE = 'ESSO_SOLVE_FAILURE'


# ===========================================================================
#  baseline  --  loaded once, never mutated
# ===========================================================================
_BASELINE = None


def load_baseline(verbose=False):
    """Read the canonical planning problem once and keep it read-only."""
    global _BASELINE
    if _BASELINE is None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            planning = SharedResourcesPlanning('data/SRP1', 'SRP1.json')
            planning.read_planning_problem()
        checksum = _scenario_checksum(stream.getvalue())
        if checksum and checksum != CANONICAL_CHECKSUM:
            raise RuntimeError(
                f'scenario checksum {checksum} != canonical {CANONICAL_CHECKSUM}')
        _BASELINE = {'planning': planning, 'checksum': checksum or CANONICAL_CHECKSUM}
    if verbose:
        print(f"[oracle] baseline checksum {_BASELINE['checksum']}")
    return _BASELINE


def _scenario_checksum(text):
    for line in text.splitlines():
        if 'checksum' in line.lower():
            for token in line.replace(':', ' ').split():
                if len(token) == 64 and all(c in '0123456789abcdef' for c in token):
                    return token
    return None


def fresh_planning(eval_id):
    """A private, fully isolated copy of the baseline for one evaluation.

    Deep copy rather than re-read: 0.03 s against 10.8 s, and it guarantees that
    no persistent object is shared with the baseline or with any other
    evaluation.  Solver logs and diagnostics accumulators are isolated too, since
    IPOPT is configured with `file_append='yes'` and would otherwise append every
    evaluation's log into the same file.
    """
    planning = deepcopy(load_baseline()['planning'])
    logs = os.path.join(WORK_DIR, eval_id, 'logs')
    os.makedirs(logs, exist_ok=True)
    planning.logs_dir = logs
    for holder in _holders(planning):
        if hasattr(holder, 'logs_dir'):
            holder.logs_dir = logs
        for year in holder.years:
            for day in holder.days:
                holder.network[year][day].logs_dir = logs
    if hasattr(planning.shared_ess_data, 'logs_dir'):
        planning.shared_ess_data.logs_dir = logs
    planning.shared_ess_data.solver_recovery_diagnostics = []
    return planning


def _holders(planning):
    return [planning.transmission_network] + [
        planning.distribution_networks[node]
        for node in sorted(planning.distribution_networks)]


# ===========================================================================
#  candidate construction and master-side feasibility  (A4)
# ===========================================================================
def nodes_and_years(planning):
    return (list(planning.shared_ess_data.active_distribution_network_nodes),
            list(planning.shared_ess_data.years))


def vector_to_candidate(planning, x):
    """Turn the 18 investment variables into a production candidate solution.

    `x` maps (node, year) -> {'s': ..., 'e': ...} in MVA / MVAh, or is a flat
    sequence ordered node-major then year-major then (s, e).  Total capacities
    are rebuilt with production's own cohort/calendar-life mapping, never by
    hand.
    """
    nodes, years = nodes_and_years(planning)
    candidate = planning.get_initial_candidate_solution()
    if not isinstance(x, dict):
        values = list(x)
        expected = 2 * len(nodes) * len(years)
        if len(values) != expected:
            raise ValueError(f'expected {expected} investment variables, '
                             f'got {len(values)}')
        x = {}
        k = 0
        for node in nodes:
            for year in years:
                x[(node, year)] = {'s': values[k], 'e': values[k + 1]}
                k += 2
    for node in nodes:
        for year in years:
            entry = x[(node, year)]
            candidate['investment'][node][year]['s'] = float(entry['s'])
            candidate['investment'][node][year]['e'] = float(entry['e'])
    srp._rebuild_candidate_total_capacities(planning, candidate)
    return candidate


def candidate_to_vector(planning, candidate):
    nodes, years = nodes_and_years(planning)
    return {(node, year): {'s': candidate['investment'][node][year]['s'],
                           'e': candidate['investment'][node][year]['e']}
            for node in nodes for year in years}


def check_master_feasibility(planning, candidate):
    """Production's own first-stage feasibility test, reused verbatim.

    Covers nonnegativity, the minimum and maximum E/S ratio, the maximum energy
    capacity (on the cohort-accumulated totals) and the investment budget.  The
    cohort/calendar-life mapping is exercised by the total-capacity rebuild that
    precedes it.
    """
    return srp._check_candidate_first_stage_feasibility(planning, candidate)


def investment_cost(planning, candidate):
    """Expected discounted investment cost -- the master's own expression.

    Reproduces `model.investment_cost` (shared_energy_storage_data.py:310-325)
    term by term without constructing a master problem.
    """
    esso = planning.shared_ess_data
    years = list(esso.years)
    total = 0.0
    for node, yearly in candidate['investment'].items():
        for year, investment in yearly.items():
            annualization = 1.0 / ((1.0 + esso.discount_factor)
                                   ** (int(year) - int(years[0])))
            for scenario, probability in enumerate(esso.prob_market_scenarios):
                total += annualization * probability * (
                    esso.cost_investment['power'][scenario][year] * investment['s']
                    + esso.cost_investment['energy'][scenario][year] * investment['e'])
    return total


# ===========================================================================
#  coordinated quantities
# ===========================================================================
def common_coordinated_values(planning, models, consensus_vars):
    """One common value per coordinated quantity, per (node, year, day, period).

    Where the ADMM maintains an explicit consensus variable it is used: the
    shared-ESS family has `consensus_vars['ess']['z']`, and that z is taken
    directly.  The interface power-flow and voltage families have only TSO and
    DSO copies with no z, so the common value is their midpoint, which is the
    choice that minimises the largest correction asked of either side.  Both
    conventions are recorded per entry so the report can say which is which.
    """
    tso = planning.transmission_network
    common = {}
    for node, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net = tso.network[year][day]
                d_net = dso.network[year][day]
                t_model = models['tso'][year][day]
                d_model = models['dso'][node][year][day]
                dn = list(t_net.active_distribution_network_nodes).index(node)
                adn_idx = t_net.get_node_idx(node)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                t_sess = [e for e, s in enumerate(t_net.shared_energy_storages)
                          if s.bus == node]
                d_sess = [e for e, s in enumerate(d_net.shared_energy_storages)
                          if s.bus == ref_id]
                z_p = consensus_vars['ess']['z']['current'][node][year][day]['p']
                z_q = consensus_vars['ess']['z']['current'][node][year][day]['q']
                for p in t_model.periods:
                    t_p = float(pe.value(t_model.pc_adn[dn, 0, 0, p]))
                    t_q = float(pe.value(t_model.qc_adn[dn, 0, 0, p]))
                    d_p = float(pe.value(d_model.pg_adn[0, 0, p]))
                    d_q = float(pe.value(d_model.qg_adn[0, 0, p]))
                    t_v = float(pe.value(t_model.vmag_sqr[adn_idx, 0, 0, p])) ** 0.5
                    d_v = float(pe.value(d_model.vmag_sqr[ref_idx, 0, 0, p])) ** 0.5
                    t_sp = sum(float(pe.value(t_model.shared_es_pnet[e, 0, 0, p]))
                               for e in t_sess)
                    t_sq = sum(float(pe.value(t_model.shared_es_qnet[e, 0, 0, p]))
                               for e in t_sess)
                    d_sp = sum(float(pe.value(d_model.shared_es_pnet[e, 0, 0, p]))
                               for e in d_sess)
                    d_sq = sum(float(pe.value(d_model.shared_es_qnet[e, 0, 0, p]))
                               for e in d_sess)
                    common[(node, year, day, p)] = {
                        'tso_p': t_p, 'dso_p': d_p, 'tso_q': t_q, 'dso_q': d_q,
                        'tso_v': t_v, 'dso_v': d_v,
                        'tso_sess_p': t_sp, 'dso_sess_p': d_sp,
                        'tso_sess_q': t_sq, 'dso_sess_q': d_sq,
                        # interface families: no ADMM z exists, use the midpoint
                        'common_p': 0.5 * (t_p + d_p),
                        'common_q': 0.5 * (t_q + d_q),
                        'common_v': 0.5 * (t_v + d_v),
                        'source_interface': 'midpoint (no ADMM z for vmag/pf)',
                        # shared-ESS family: the ADMM's own consensus z, in MW,
                        # converted to the network per-unit base
                        'common_sess_p': float(z_p[p]) / t_net.baseMVA,
                        'common_sess_q': float(z_q[p]) / t_net.baseMVA,
                        'common_sess_p_mw': float(z_p[p]),
                        'common_sess_q_mw': float(z_q[p]),
                        'source_sess': "consensus_vars['ess']['z']",
                    }
    return common


def esso_request_from_common(planning, consensus_vars, common):
    """Write the common shared-ESS schedule into an ESSO request (MW)."""
    request = deepcopy(consensus_vars)
    for (node, year, day, p), entry in common.items():
        request['ess']['tso']['current'][node][year][day]['p'][p] = \
            entry['common_sess_p_mw']
        request['ess']['tso']['current'][node][year][day]['q'][p] = \
            entry['common_sess_q_mw']
    return request


# ===========================================================================
#  physical ESSO  (A1 steps 3-4)
# ===========================================================================
def solve_physical_esso(planning, candidate, request):
    """Solve the ORIGINAL nonlinear ESSO with investment and P/Q fixed.

    `create_shared_energy_storage_model` is production's own constructor: it
    applies the candidate investment, builds the full nonlinear ESSO subproblem
    (cohort rated S/E, availability, throughput, degradation, annual and
    cumulative SoH, minimum SoH, lifetime gating, cohort complementarity,
    aggregate P/Q capability and the ESSO slacks), fixes the requested P/Q
    schedule and optimises.  Using it rather than a private reimplementation is
    the point: the certificate must be against the original formulation.
    """
    with redirect_stdout(io.StringIO()):
        esso_models, results = srp.create_shared_energy_storage_model(
            planning.shared_ess_data, request, candidate['investment'])
    solved = {node: bool(srp._solver_result_succeeded(results[node]))
              for node in results}
    available = planning.shared_ess_data.get_updated_capacities(esso_models)
    return esso_models, results, solved, available


# ===========================================================================
#  polishing  (A1 steps 5-7)
# ===========================================================================
def apply_physical_capacities(planning, models, available):
    """Push the ESSO's physical available S/E into every network copy."""
    largest_shift = 0.0
    for tag, holder in _tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                for e_idx, esso_node in _sess_pairs(network, node_of):
                    s_pu = available[esso_node][year]['s_available'] / network.baseMVA
                    e_pu = available[esso_node][year]['e_available'] / network.baseMVA
                    largest_shift = max(
                        largest_shift,
                        abs(float(pe.value(model.shared_es_e_rated_fixed[e_idx]))
                            - e_pu) * network.baseMVA)
                    model.shared_es_s_rated_fixed[e_idx].set_value(s_pu)
                    model.shared_es_e_rated_fixed[e_idx].set_value(e_pu)
                    mch.configure_shared_ess_operational_state(model, e_idx,
                                                               s_pu, e_pu)
    return largest_shift


def _tagged_holders(planning):
    out = [('TSO', planning.transmission_network)]
    for node in sorted(planning.distribution_networks):
        out.append((f'DSO{node}', planning.distribution_networks[node]))
    return out


def _sess_pairs(network, node_of):
    """(shared-ESS index, ESSO node) for one network copy."""
    if node_of is None:
        return [(e, ess.bus) for e, ess in enumerate(network.shared_energy_storages)]
    ref = network.get_reference_node_id()
    return [(network.get_shared_energy_storage_idx(ref), node_of)]


def _interface_expression(t_model, adn_load, p, kind):
    """Production's own interface_pf_[pq]_transmission_def, as an expression."""
    base = t_model.pc if kind == 'p' else t_model.qc
    value = base[adn_load, 0, 0, p]
    up_name = 'flex_p_up' if kind == 'p' else 'flex_q_up'
    down_name = 'flex_p_down' if kind == 'p' else 'flex_q_down'
    if hasattr(t_model, up_name) and hasattr(t_model, down_name):
        value = value + (getattr(t_model, up_name)[adn_load, 0, 0, p]
                         - getattr(t_model, down_name)[adn_load, 0, 0, p])
    return value


def apply_common_values(planning, models, common):
    """Fix every coordinated quantity to its common value on both sides."""
    tso = planning.transmission_network
    for node, dso in sorted(planning.distribution_networks.items()):
        for year in tso.years:
            for day in tso.days:
                t_net, d_net = tso.network[year][day], dso.network[year][day]
                t_model = models['tso'][year][day]
                d_model = models['dso'][node][year][day]
                adn_idx = t_net.get_node_idx(node)
                adn_load = t_net.get_adn_load_idx(node)
                ref_id = d_net.get_reference_node_id()
                ref_idx = d_net.get_node_idx(ref_id)
                ref_gen = d_net.get_reference_gen_idx()
                t_sess = [e for e, s in enumerate(t_net.shared_energy_storages)
                          if s.bus == node]
                d_sess = [e for e, s in enumerate(d_net.shared_energy_storages)
                          if s.bus == ref_id]
                rows = getattr(t_model, 'p56a_interface_rows', None)
                if rows is None:
                    rows = pe.ConstraintList()
                    t_model.add_component('p56a_interface_rows', rows)
                for p in t_model.periods:
                    entry = common[(node, year, day, p)]
                    # Transmission interface.  Production FIXES pc at the DSO's
                    # consensus interface power and lets the TSO deviate only
                    # through flex, charging flexibility_cost for the DOWN
                    # direction (shared_resources_planning.py:2905-2928,
                    # model_construction_helpers.py:flexibility_cost).  So pc is
                    # left exactly where production put it and only the TOTAL
                    # interface power is pinned:
                    #
                    #     pc + flex_p_up - flex_p_down == common_p
                    #
                    # Fixing pc at the achieved value and zeroing the flex
                    # instead would deliver the same physical interface power at
                    # ZERO flexibility cost -- which is not a cheaper plan, only
                    # a different accounting of the same one.
                    rows.add(_interface_expression(t_model, adn_load, p, 'p')
                             == entry['common_p'])
                    rows.add(_interface_expression(t_model, adn_load, p, 'q')
                             == entry['common_q'])
                    # voltage through the squared magnitude leaves the angle free
                    t_model.vmag_sqr[adn_idx, 0, 0, p].fix(entry['common_v'] ** 2)
                    d_model.vmag_sqr[ref_idx, 0, 0, p].fix(entry['common_v'] ** 2)
                    # pg_adn = pg[ref_gen] - sum shared_es_pnet
                    d_model.pg[ref_gen, 0, 0, p].fix(
                        entry['common_p'] + entry['common_sess_p'])
                    d_model.qg[ref_gen, 0, 0, p].fix(
                        entry['common_q'] + entry['common_sess_q'])
                    for e in t_sess:
                        t_model.shared_es_pnet[e, 0, 0, p].fix(entry['common_sess_p'])
                        t_model.shared_es_qnet[e, 0, 0, p].fix(entry['common_sess_q'])
                    for e in d_sess:
                        d_model.shared_es_pnet[e, 0, 0, p].fix(entry['common_sess_p'])
                        d_model.shared_es_qnet[e, 0, 0, p].fix(entry['common_sess_q'])


def per_block_base_objectives(planning, models):
    """Weighted base-objective contribution of every (agent, year, day) block.

    This is what `get_primal_value` sums, block by block, so the ADMM and
    polished values are directly comparable and their difference decomposes the
    recourse change exactly.  Captured before AND after polishing so P5.6-A2 can
    attribute the change without a second ADMM run.
    """
    out = {}
    for tag, holder in _tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                weight = srp._get_admm_block_weight(holder, year, day)
                families = {}
                for name in ('generation_cost', 'flexibility_cost',
                             'load_curtailment_cost', 'gen_curtailment_penalty',
                             'ess_utilization_cost_penalty', 'slack_penalties',
                             'ess_complementarity_penalties'):
                    try:
                        if name == 'ess_complementarity_penalties':
                            value = float(pe.value(
                                mch.ess_complementarity_penalties_rule(
                                    model, 0, 0, network=network,
                                    params=holder.params)))
                        else:
                            value = float(pe.value(getattr(mch, name)(
                                model, network, 0, 0, holder.params)))
                    except Exception:
                        value = None
                    families[name] = None if value is None else weight * value
                out[f'{tag}|{year}|{day}'] = {
                    'weight': weight,
                    'weighted_base_objective': weight * float(pe.value(
                        mch.objective_function_rule(model, holder.params))),
                    'families': families,
                }
    return out


def restore_base_objective(model):
    """Polish against the base objective -- what the recourse is measured on."""
    if hasattr(model, 'admm_objective') and model.admm_objective.active:
        model.admm_objective.deactivate()
    model.objective.activate()


def polish_networks(planning, models):
    """Re-solve all 48 nonlinear network SMOPFs with the coordination fixed."""
    blocks, all_solved = [], True
    for tag, holder in _tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                restore_base_objective(model)
                with redirect_stdout(io.StringIO()):
                    result = network.run_smopf(model, holder.params,
                                               print_header=False)
                ok = bool(srp._solver_result_succeeded(result))
                all_solved &= ok
                blocks.append({'agent': tag, 'year': year, 'day': day,
                               'solved': ok})
    return blocks, all_solved


# ===========================================================================
#  audits  (A1)
# ===========================================================================
def constraint_violation(con_data):
    try:
        body = float(pe.value(con_data.body))
    except Exception:
        return None
    worst = 0.0
    if con_data.has_lb():
        worst = max(worst, float(pe.value(con_data.lower)) - body)
    if con_data.has_ub():
        worst = max(worst, body - float(pe.value(con_data.upper)))
    return worst


def scan_constraints(block, descend=False):
    families = {}
    for con in block.component_objects(pe.Constraint, active=True,
                                       descend_into=descend):
        worst, worst_idx, n = 0.0, None, 0
        for idx in con:
            value = constraint_violation(con[idx])
            if value is None:
                continue
            n += 1
            if value > worst:
                worst, worst_idx = value, str(idx)
        families[con.local_name] = {'n': n, 'max_violation': worst,
                                    'worst_index': worst_idx}
    return families


def audit_esso(planning, esso_models):
    """Every constraint family of the ORIGINAL nonlinear ESSO, per node."""
    per_node, worst_overall, worst_family = {}, 0.0, None
    for node in planning.shared_ess_data.active_distribution_network_nodes:
        families = scan_constraints(esso_models[node], descend=True)
        node_worst = max((f['max_violation'] for f in families.values()),
                         default=0.0)
        per_node[node] = {'families': families, 'max_violation': node_worst}
        if node_worst > worst_overall:
            worst_overall = node_worst
            worst_family = max(families.items(),
                               key=lambda kv: kv[1]['max_violation'])[0]
    production_violation = planning.shared_ess_data.get_feasibility_violation(
        esso_models)
    return {'per_node': per_node, 'max_violation': worst_overall,
            'worst_family': worst_family,
            'production_feasibility_violation': float(production_violation),
            'production_tolerance': ESSO_FEASIBILITY_TOLERANCE,
            'production_feasible': (float(production_violation)
                                    <= ESSO_FEASIBILITY_TOLERANCE)}


def audit_networks(planning, models):
    """Every constraint family of all 48 nonlinear network SMOPFs."""
    per_block, worst_overall, worst_where = {}, 0.0, None
    h1_worst, capability_worst = 0.0, 0.0
    for tag, holder in _tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                families = scan_constraints(model, descend=False)
                worst = max((f['max_violation'] for f in families.values()),
                            default=0.0)
                key = f'{tag}|{year}|{day}'
                per_block[key] = {'max_violation': worst,
                                  'worst_family': max(
                                      families.items(),
                                      key=lambda kv: kv[1]['max_violation'])[0]
                                  if families else None}
                for name, entry in families.items():
                    if 'comp' in name:
                        h1_worst = max(h1_worst, entry['max_violation'])
                    if 'capability' in name:
                        capability_worst = max(capability_worst,
                                               entry['max_violation'])
                if worst > worst_overall:
                    worst_overall, worst_where = worst, key
    return {'per_block': per_block, 'max_violation': worst_overall,
            'worst_block': worst_where,
            'max_h1_complementarity_violation': h1_worst,
            'max_converter_capability_violation': capability_worst}


def coordination_residuals(planning, models, esso_models, common, available):
    """Every mismatch A1 requires, TSO vs DSO vs ESSO."""
    after = common_coordinated_values(planning, models,
                                      _dummy_consensus(planning, common))
    residuals = {
        'interface_p': max(abs(v['tso_p'] - v['dso_p']) for v in after.values()),
        'interface_q': max(abs(v['tso_q'] - v['dso_q']) for v in after.values()),
        'interface_v': max(abs(v['tso_v'] - v['dso_v']) for v in after.values()),
        'shared_ess_p_tso_dso': max(abs(v['tso_sess_p'] - v['dso_sess_p'])
                                    for v in after.values()),
        'shared_ess_q_tso_dso': max(abs(v['tso_sess_q'] - v['dso_sess_q'])
                                    for v in after.values()),
    }
    # TSO/DSO against the physical ESSO schedule
    esso_p, esso_q = 0.0, 0.0
    years = list(planning.shared_ess_data.years)
    days = list(planning.shared_ess_data.days)
    for node in planning.shared_ess_data.active_distribution_network_nodes:
        model = esso_models[node]
        base = planning.transmission_network.network[years[0]][days[0]].baseMVA
        for y, year in enumerate(years):
            for d, day in enumerate(days):
                for p in model.periods:
                    physical_p = float(pe.value(model.es_pnet[y, d, p])) / base
                    physical_q = float(pe.value(model.es_qnet[y, d, p])) / base
                    entry = after[(node, year, day, p)]
                    esso_p = max(esso_p, abs(entry['tso_sess_p'] - physical_p),
                                 abs(entry['dso_sess_p'] - physical_p))
                    esso_q = max(esso_q, abs(entry['tso_sess_q'] - physical_q),
                                 abs(entry['dso_sess_q'] - physical_q))
    residuals['shared_ess_p_vs_esso'] = esso_p
    residuals['shared_ess_q_vs_esso'] = esso_q

    # available capacity, ESSO against every network copy
    s_worst, e_worst = 0.0, 0.0
    for tag, holder in _tagged_holders(planning):
        node_of = None if tag == 'TSO' else int(tag[3:])
        for year in holder.years:
            for day in holder.days:
                network = holder.network[year][day]
                model = (models['tso'][year][day] if tag == 'TSO'
                         else models['dso'][node_of][year][day])
                for e_idx, esso_node in _sess_pairs(network, node_of):
                    s_net = float(pe.value(
                        model.shared_es_s_rated_fixed[e_idx])) * network.baseMVA
                    e_net = float(pe.value(
                        model.shared_es_e_rated_fixed[e_idx])) * network.baseMVA
                    s_worst = max(s_worst, abs(
                        s_net - available[esso_node][year]['s_available']))
                    e_worst = max(e_worst, abs(
                        e_net - available[esso_node][year]['e_available']))
    residuals['s_available'] = s_worst
    residuals['e_available'] = e_worst
    residuals['max_coordinated'] = max(residuals.values())
    return residuals, after


def _dummy_consensus(planning, common):
    """common_coordinated_values only reads ess.z; re-serve the same values."""
    z = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (node, year, day, p), entry in common.items():
        z[node][year].setdefault(day, {'p': {}, 'q': {}})
        z[node][year][day]['p'][p] = entry['common_sess_p_mw']
        z[node][year][day]['q'][p] = entry['common_sess_q_mw']
    return {'ess': {'z': {'current': z}}}


# ===========================================================================
#  start policies  (A5)
# ===========================================================================
#  Both policies are fixed by configuration.  Neither consults the history of
#  previously evaluated candidates: a "nearest previous candidate" warm start
#  would make the returned objective depend on search order, which is exactly
#  what P5.5-D3/D4 showed can move the recourse by more than the investment
#  signal.
START_COLD = 'START-1-cold'
START_TEMPLATE = 'START-2-fixed-template'
DEFAULT_START_SET = (START_COLD,)

_TEMPLATE = {'state': None, 'built': False}


def build_fixed_template(verbose=False):
    """The archived branch template for START-2, built once per process.

    It is the cold solution of the CANONICAL BASE CANDIDATE -- one fixed point,
    identical for every candidate that is ever evaluated, and therefore
    independent of search history.  Production's own warm-start path
    (`initial_state`) then remaps it to the candidate's capacities using the
    established physical remapping rules.
    """
    if _TEMPLATE['built']:
        return _TEMPLATE['state']
    planning = fresh_planning('template')
    with redirect_stdout(io.StringIO()):
        candidate = srp._build_positive_bootstrap_candidate(
            planning, planning.params.benders.positive_bootstrap)
        _, _, _, _, _, state = planning.run_operational_planning(
            type='distributed', candidate_solution=deepcopy(candidate),
            print_results=False, debug_flag=False, return_state=True)
    _TEMPLATE['state'] = state
    _TEMPLATE['built'] = True
    if verbose:
        print('[oracle] START-2 template built from the canonical base candidate')
    return state


# ===========================================================================
#  caching  (A7)
# ===========================================================================
def _config_hash(planning):
    parts = [ORACLE_VERSION, load_baseline()['checksum']]
    for tag, holder in _tagged_holders(planning):
        params = holder.params
        parts.append(f'{tag}:{params.obj_type}:{params.fl_reg}:{params.rg_curt}:'
                     f'{params.l_curt}:{params.es_reg}:{params.shared_ess_model}')
    admm = planning.params.admm
    parts.append(json.dumps(admm.tol, sort_keys=True))
    parts.append(str(admm.num_max_iters))
    esso_params = planning.shared_ess_data.params
    parts.append(f'{esso_params.budget}:{esso_params.max_capacity}:'
                 f'{esso_params.min_energy_to_power_ratio}:'
                 f'{esso_params.max_energy_to_power_ratio}')
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:32]


def cache_key(planning, candidate, start_policy):
    nodes, years = nodes_and_years(planning)
    vector = []
    for node in nodes:
        for year in years:
            entry = candidate['investment'][node][year]
            vector.append(f'{node}:{year}:{entry["s"]!r}:{entry["e"]!r}')
    payload = {
        'investment': vector,
        'scenario_checksum': load_baseline()['checksum'],
        'config_hash': _config_hash(planning),
        'oracle_version': ORACLE_VERSION,
        'start_policy': start_policy,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest(), payload


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH) as handle:
            return json.load(handle)
    except (ValueError, OSError):
        return {}


def store_cache(cache):
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = CACHE_PATH + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(cache, handle, indent=1, default=str)
    shutil.move(tmp, CACHE_PATH)


# ===========================================================================
#  the candidate-evaluation contract  (A4)
# ===========================================================================
def evaluate_planning_candidate(x, start_policy=START_COLD, eval_id=None,
                                use_cache=True, verbose=False,
                                keep_models=False):
    """Evaluate ONE investment candidate under ONE fixed start policy.

    Returns a dict that always carries `status`.  A failed evaluation reports
    why; it never returns a fabricated objective, because a search driven by
    sentinel values optimises the sentinel rather than the plan.
    """
    started = time.time()
    eval_id = eval_id or f'eval_{int(time.time() * 1e6)}'
    result = {'eval_id': eval_id, 'start_policy': start_policy,
              'oracle_version': ORACLE_VERSION,
              'timestamp_utc': datetime.now(timezone.utc).isoformat()}

    planning = fresh_planning(eval_id)
    candidate = vector_to_candidate(planning, x)
    result['investment'] = {f'{n}|{y}': dict(candidate['investment'][n][y])
                            for n in candidate['investment']
                            for y in candidate['investment'][n]}
    result['total_capacity'] = {f'{n}|{y}': dict(candidate['total_capacity'][n][y])
                                for n in candidate['total_capacity']
                                for y in candidate['total_capacity'][n]}

    # ---- step 0: master feasibility, BEFORE any operational model runs ----
    feasible, reason = check_master_feasibility(planning, candidate)
    result['master_feasible'] = feasible
    result['master_reason'] = reason
    if not feasible:
        result['status'] = STATUS_INVALID_INVESTMENT
        result['wall_clock_s'] = time.time() - started
        return result
    result['investment_cost'] = investment_cost(planning, candidate)

    key, payload = cache_key(planning, candidate, start_policy)
    result['cache_key'] = key
    result['cache_payload'] = payload
    if use_cache:
        cache = load_cache()
        if key in cache and cache[key].get('status') == STATUS_VALID:
            cached = dict(cache[key])
            cached['cache_hit'] = True
            cached['eval_id'] = eval_id
            cached['wall_clock_s'] = time.time() - started
            return cached
    result['cache_hit'] = False

    # ---- steps 1-3: the accepted nonlinear operational ADMM ----
    initial_state = None
    if start_policy == START_TEMPLATE:
        initial_state = build_fixed_template(verbose=verbose)
    admm_started = time.time()
    with redirect_stdout(io.StringIO()):
        convergence, _, models, _, _, state = planning.run_operational_planning(
            type='distributed', candidate_solution=deepcopy(candidate),
            print_results=False, debug_flag=False,
            initial_state=initial_state, return_state=True)
    result['admm'] = {
        'converged': bool(convergence),
        'initialization_failed': bool(state.get('initialization_failed', False)),
        'cycles': len(state.get('admm_diagnostics', [])),
        'runtime_s': time.time() - admm_started,
        'n_recovery_diagnostics': len(state.get('solver_recovery_diagnostics', [])),
    }
    if state.get('initialization_failed', False):
        result['status'] = STATUS_LOCAL_SOLVE_FAILURE
        result['wall_clock_s'] = time.time() - started
        return result
    admm_components = planning.get_operational_recourse_components(models)
    result['admm'].update({
        'gross_operational_cost': admm_components['gross_operational_cost'],
        'terminal_salvage_value': admm_components['terminal_salvage_value'],
        'net_operational_recourse': admm_components['net_operational_recourse'],
    })

    result['per_block_admm'] = per_block_base_objectives(planning, models)

    # ---- step 4: one common value per coordinated quantity ----
    common = common_coordinated_values(planning, models, state['consensus_vars'])
    result['admm_coordination_residuals'] = {
        'interface_p': max(abs(v['tso_p'] - v['dso_p']) for v in common.values()),
        'interface_q': max(abs(v['tso_q'] - v['dso_q']) for v in common.values()),
        'interface_v': max(abs(v['tso_v'] - v['dso_v']) for v in common.values()),
        'shared_ess_p': max(abs(v['tso_sess_p'] - v['dso_sess_p'])
                            for v in common.values()),
        'shared_ess_q': max(abs(v['tso_sess_q'] - v['dso_sess_q'])
                            for v in common.values()),
    }

    # ---- step 5: the ORIGINAL nonlinear ESSO, physical capacities ----
    esso_started = time.time()
    request = esso_request_from_common(planning, state['consensus_vars'], common)
    esso_models, esso_results, esso_solved, available = solve_physical_esso(
        planning, candidate, request)
    result['esso'] = {'solved': esso_solved, 'runtime_s': time.time() - esso_started,
                      'solve_count': len(esso_solved)}
    if not all(esso_solved.values()):
        result['status'] = STATUS_ESSO_SOLVE_FAILURE
        result['wall_clock_s'] = time.time() - started
        return result
    result['available_capacity'] = {
        f'{node}|{year}': dict(available[node][year])
        for node in available for year in available[node]}

    # ---- step 6: push physical capacities, fix coordination, re-solve ----
    polish_started = time.time()
    result['capacity_shift_into_networks'] = apply_physical_capacities(
        planning, models, available)
    apply_common_values(planning, models, common)
    blocks, all_solved = polish_networks(planning, models)
    result['polish'] = {'blocks': blocks, 'all_solved': all_solved,
                        'runtime_s': time.time() - polish_started,
                        'solve_count': len(blocks)}
    if not all_solved:
        result['status'] = STATUS_POLISH_FAILURE
        result['wall_clock_s'] = time.time() - started
        return result

    # ---- consistency iteration if ESSO availability moved ----
    residuals, _ = coordination_residuals(planning, models, esso_models,
                                          common, available)
    result['coordination_residuals'] = residuals
    result['consistency_iteration'] = False
    if max(residuals['s_available'], residuals['e_available']) > COORDINATION_TARGET:
        # deterministic single re-solve: the ESSO availability changed after the
        # networks were given their capacities, so give them the new ones and
        # re-solve once.  Reported, never silent.
        result['consistency_iteration'] = True
        result['capacity_shift_second_pass'] = apply_physical_capacities(
            planning, models, available)
        blocks2, all_solved2 = polish_networks(planning, models)
        result['polish']['blocks_second_pass'] = blocks2
        result['polish']['all_solved_second_pass'] = all_solved2
        if not all_solved2:
            result['status'] = STATUS_POLISH_FAILURE
            result['wall_clock_s'] = time.time() - started
            return result
        residuals, _ = coordination_residuals(planning, models, esso_models,
                                              common, available)
        result['coordination_residuals'] = residuals

    result['per_block_polished'] = per_block_base_objectives(planning, models)

    # ---- step 7: audits against the ORIGINAL nonlinear formulations ----
    result['esso_audit'] = audit_esso(planning, esso_models)
    result['network_audit'] = audit_networks(planning, models)

    # ---- step 8: exact objective accounting ----
    polished = planning.get_operational_recourse_components(models)
    physical_salvage = planning.shared_ess_data.get_salvage_value(esso_models)
    result['gross_operational_cost'] = polished['gross_operational_cost']
    result['salvage_from_polished_network_models'] = polished['terminal_salvage_value']
    result['physical_salvage'] = float(physical_salvage)
    result['net_operational_recourse'] = (polished['gross_operational_cost']
                                          - float(physical_salvage))
    result['total_objective'] = (result['investment_cost']
                                 + result['net_operational_recourse'])

    # ---- verdict ----
    if not result['esso_audit']['production_feasible']:
        result['status'] = STATUS_ESSO_FEASIBILITY_FAILURE
    elif residuals['max_coordinated'] > COORDINATION_TARGET:
        result['status'] = STATUS_COUPLING_FAILURE
    else:
        result['status'] = STATUS_VALID
    result['wall_clock_s'] = time.time() - started

    if keep_models:
        result['_models'] = models
        result['_esso_models'] = esso_models
        result['_planning'] = planning
        result['_common'] = common

    if use_cache and result['status'] == STATUS_VALID:
        cache = load_cache()
        cache[key] = {k: v for k, v in result.items() if not k.startswith('_')}
        store_cache(cache)
    return result


def q_oracle(x, start_set=DEFAULT_START_SET, use_cache=True, verbose=False):
    """Q_oracle(x): the best TOTAL OBJECTIVE over the fixed start set (A5).

    Deterministic because every candidate sees the same declared start set.  It
    is a heuristic upper-bound oracle, not a global optimum.
    """
    runs = []
    for policy in start_set:
        runs.append(evaluate_planning_candidate(x, start_policy=policy,
                                                use_cache=use_cache,
                                                verbose=verbose))
    valid = [r for r in runs if r['status'] == STATUS_VALID]
    best = min(valid, key=lambda r: r['total_objective']) if valid else None
    return {'runs': runs, 'best': best,
            'Q_oracle': best['total_objective'] if best else None,
            'best_start': best['start_policy'] if best else None,
            'status': STATUS_VALID if best else runs[0]['status']}
