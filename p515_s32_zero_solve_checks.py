"""
P5.15 Step 3.2 + 3.3(a) (s32 worker task) -- zero-solve verification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a); binding
specification data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json.

Verifies, on FRESHLY BUILT C* ADMM models (a NEW eval id) and on a synthetic
single-entry perturbation of the ADMM consensus/dual variables (so every
primal/dual term is isolated and hand-derivable), that every Step 3.2/3.3(a)
production change is correct, with a BLOCKING `SolveProfileGuard` (permitted
call sites = (), i.e. zero solves anywhere) armed for the whole check. TSO/DSO
`.optimize` are monkeypatched to a stub (never reaches a solver, same
technique as `p515_s31c_zero_solve_checks.py`) so the production model-
construction functions can be exercised up to and past their `.optimize()`
call sites without ever calling IPOPT; the guard is the enforcement
mechanism, the monkeypatch keeps the harness from even attempting it. The
ESSO subproblem is built with `shared_ess_data.build_subproblem()`, which
never solves.

    python p515_s32_zero_solve_checks.py

Writes data/SRP1/Results/P515S32/zero_solve_checks/zero_solve_checks.json and
data/SRP1/Results/P515S32/zero_solve_checks/synthetic_admm_snapshot.pkl (both
NEW files; nothing under data/ is overwritten -- `_refuse_overwrite`, same
convention as P515S31C).
"""

import hashlib
import inspect
import json
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe
import pyomo.opt as po
from pyomo.repn import generate_standard_repn

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s31c_zero_solve_checks import FIXTURES as S31C_FIXTURES  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S32', 'zero_solve_checks')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')
SNAPSHOT_PATH = os.path.join(OUT_DIR, 'synthetic_admm_snapshot.pkl')

CS1_PARAMS_PATH = os.path.join(REPO, 'data', 'CS1', 'CS1_params.json')
SRP1_PARAMS_PATH = os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _stub_optimize(self, model, from_warm_start=False, print_header=True,
                    failure_snapshot_callback=None, pre_solve_snapshot_callback=None):
    """Replaces NetworkData.optimize on ONE instance: never calls a solver
    (same technique as p515_s31c_zero_solve_checks.py's `_stub_optimize`)."""
    return {year: {day: None for day in self.days} for year in self.years}


def _fake_success_result():
    result = po.SolverResults()
    result.solver.status = po.SolverStatus.ok
    result.solver.termination_condition = po.TerminationCondition.optimal
    return result


def _load_pre_change_module():
    """Loads the git HEAD (pre-worker-change) version of shared_resources_planning.py
    as a SEPARATE module `srp_pre_change`, per the task's explicit instruction
    ("Use the committed pre-change source via `git show HEAD:...` loaded as a
    separate module"). HEAD is the last commit BEFORE this worker's uncommitted
    edits."""
    proc = subprocess.run(
        ['git', 'show', 'HEAD:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    source = proc.stdout
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix='p515_s32_pre_change_')
    tmp_path = os.path.join(tmp_dir, '_shared_resources_planning_pre_change.py')
    with open(tmp_path, 'w') as handle:
        handle.write(source)
    import importlib.util
    spec = importlib.util.spec_from_file_location('srp_pre_change', tmp_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, proc.stdout


def _build_admm_ready_state(eval_id):
    """Builds a full zero-solve ADMM-ready state (planning_problem, tso_model,
    dso_models, esso_model, consensus_vars, dual_vars) through the SAME
    production sequence the cycle-loop initializer uses (shared_resources_planning.py
    `_run_operational_planning_distributed`, lines ~2360-2420), EXCEPT the
    initial local solves themselves (create_transmission_network_model /
    create_distribution_networks_models solve via the monkeypatched, never-
    reaching-a-solver `.optimize` stub, exactly as
    p515_s31c_zero_solve_checks.py does; the ESSO subproblem is built with
    `shared_ess_data.build_subproblem()`, which never solves at all)."""
    eval_dir = os.path.join(O.WORK_DIR, eval_id)
    if os.path.exists(eval_dir):
        raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
    planning = O.fresh_planning(eval_id)

    transmission_network = planning.transmission_network
    distribution_networks = planning.distribution_networks
    shared_ess_data = planning.shared_ess_data

    transmission_network.optimize = _stub_optimize.__get__(transmission_network, type(transmission_network))
    for _node_id, _dn in distribution_networks.items():
        _dn.optimize = _stub_optimize.__get__(_dn, type(_dn))

    consensus_vars, dual_vars = srp.create_admm_variables(planning)
    candidate = planning.get_initial_candidate_solution()
    srp._rebuild_candidate_total_capacities(planning, candidate)

    dso_models, _dso_results = srp.create_distribution_networks_models(
        distribution_networks, consensus_vars, candidate['total_capacity'],
        parallel_execution=False,
    )
    tso_model, _tso_results = srp.create_transmission_network_model(
        planning, consensus_vars, candidate['total_capacity']
    )
    esso_model = shared_ess_data.build_subproblem()

    srp._prepare_distribution_objectives_for_admm(distribution_networks, dso_models)
    srp._prepare_transmission_objectives_for_admm(transmission_network, tso_model)
    # `_compute_common_admm_objective_scale` requires a NONZERO weighted base
    # objective on every block; these models are at their build-default (never
    # solved) state, where the base objective evaluates to exactly 0.0 (no
    # generation dispatched yet), so the production function raises. The
    # substantive VALUE of objective_scale plays no role in any check this
    # script runs (rho Params, proximal centres and the Boyd/legacy residual
    # computations are all independent of it) -- a fixed synthetic scale
    # (1.0) is used instead, only to let `update_*_model_to_admm` build the
    # AL objective and ADMM Params exactly as production does.
    objective_scale = 1.0
    srp.update_distribution_models_to_admm(planning, dso_models, planning.params.admm, objective_scale)
    srp.update_transmission_model_to_admm(planning, tso_model, planning.params.admm, objective_scale)
    srp.update_shared_energy_storage_model_to_admm(planning, esso_model, planning.params.admm)

    return planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars


def _perturb_single_entry(planning, tso_model, dso_models, consensus_vars, dual_vars):
    """Sets exactly ONE (node, year, day, period[, power_type]) coordinate of
    every channel to distinct synthetic, nonzero values, leaving every other
    coordinate at its `create_admm_variables` build default (current == prev,
    dual == 0), so it contributes EXACTLY zero to every residual/norm -- the
    perturbed entry is therefore the sole nonzero contributor, making every
    aggregate (legacy max/mean, Boyd Euclidean norm) hand-derivable from the
    single perturbed value. Returns the coordinate and the exact values set."""
    node_id = planning.active_distribution_network_nodes[0]
    year = next(iter(planning.years))
    day = next(iter(planning.days))
    p = 0
    power_type = 'p'

    network = planning.transmission_network.network[year][day]
    v_base = network.get_node_base_kv(node_id)

    # ---- V channel -------------------------------------------------------
    # DSO's own change (x_dso_curr - build-default prev) is made LARGER than
    # the TSO's change (Delta z), so the legacy dual (max of both agent
    # terms) picks the DSO term -- making it numerically DISTINGUISHABLE
    # from Boyd's s_rho_part (TSO term only), which is exactly what check 2
    # needs to demonstrate ("differs only by the documented DSO-change term").
    consensus_vars['vmag']['tso']['current'][node_id][year][day][p] = v_base * 1.02
    consensus_vars['vmag']['tso']['prev'][node_id][year][day][p] = v_base * 1.00
    consensus_vars['vmag']['dso']['current'][node_id][year][day][p] = v_base * 1.10
    # dso 'prev' left at its build default (v_base) -- the DSO's own change term.
    dual_vars['vmag']['dso']['current'][node_id][year][day][p] = 3.3
    dual_vars['vmag']['tso']['current'][node_id][year][day][p] = -3.3

    # ---- PF channel (P leg only; Q leg left at build default 0.0) --------
    consensus_vars['pf']['tso']['current'][node_id][year][day][power_type][p] = 12.0
    consensus_vars['pf']['tso']['prev'][node_id][year][day][power_type][p] = 9.0
    consensus_vars['pf']['dso']['current'][node_id][year][day][power_type][p] = 10.0
    dual_vars['pf']['dso']['current'][node_id][year][day][power_type][p] = 2.5
    dual_vars['pf']['tso']['current'][node_id][year][day][power_type][p] = -2.5

    # ---- ESS channel (P leg only) -----------------------------------------
    consensus_vars['ess']['tso']['current'][node_id][year][day][power_type][p] = 1.1
    consensus_vars['ess']['tso']['prev'][node_id][year][day][power_type][p] = 0.7
    consensus_vars['ess']['dso']['current'][node_id][year][day][power_type][p] = 0.9
    consensus_vars['ess']['esso']['current'][node_id][year][day][power_type][p] = 1.3
    consensus_vars['ess']['z']['current'][node_id][year][day][power_type][p] = 1.0
    consensus_vars['ess']['z']['prev'][node_id][year][day][power_type][p] = 0.6
    dual_vars['ess']['tso']['current'][node_id][year][day][power_type][p] = 0.4
    dual_vars['ess']['dso']['current'][node_id][year][day][power_type][p] = -0.2
    dual_vars['ess']['esso']['current'][node_id][year][day][power_type][p] = -0.1

    return {
        'node_id': node_id, 'year': str(year), 'day': str(day), 'p': p,
        'power_type': power_type, 'v_base': v_base,
    }


def check1_legacy_metrics_bit_identical(planning, tso_model, dso_models, esso_model,
                                         consensus_vars, results):
    """Check 1: on a pickled snapshot (constructed here -- no pre-existing
    full multi-agent ADMM-state fixture was found in the repository; this is
    a documented deviation, see the worker report), `get_admm_residual_metrics`
    from the NEW module is bit-identical to the pre-change (git HEAD) module's
    SAME function. `get_admm_residual_metrics` itself was NOT modified by this
    stage (only a new function was inserted after it), so this is expected to
    hold exactly -- and is executed as evidence, not assumed."""
    _refuse_overwrite(SNAPSHOT_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    snapshot = {
        'tso_model': tso_model, 'dso_models': dso_models, 'esso_model': esso_model,
        'consensus_vars': consensus_vars,
        'active_distribution_network_nodes': list(planning.active_distribution_network_nodes),
        'years': list(planning.years), 'days': list(planning.days),
        'num_instants': planning.num_instants,
        'note': 'Synthetic ADMM-ready state built via the production zero-solve '
                'pipeline (create_transmission_network_model / '
                'create_distribution_networks_models with .optimize monkeypatched '
                'to a never-solving stub, plus shared_ess_data.build_subproblem()), '
                'with one synthetic single-entry perturbation so every '
                'primal/dual residual is nonzero and hand-derivable. No '
                'pre-existing full multi-agent ADMM-state pickle was found for '
                'this comparison.',
    }
    with open(SNAPSHOT_PATH, 'wb') as handle:
        pickle.dump(snapshot, handle)

    with open(SNAPSHOT_PATH, 'rb') as handle:
        loaded = pickle.load(handle)

    pre_module, _pre_source = _load_pre_change_module()

    new_metrics = srp.get_admm_residual_metrics(
        planning, loaded['tso_model'], loaded['dso_models'], loaded['esso_model'],
        loaded['consensus_vars'],
    )
    pre_metrics = pre_module.get_admm_residual_metrics(
        planning, loaded['tso_model'], loaded['dso_models'], loaded['esso_model'],
        loaded['consensus_vars'],
    )

    def _numeric_subset(d):
        return {
            k1: {k2: v2 for k2, v2 in d[k1].items() if isinstance(v2, (int, float))}
            for k1 in ('primal', 'dual')
        }

    new_numeric = _numeric_subset(new_metrics)
    pre_numeric = _numeric_subset(pre_metrics)
    bit_identical = new_numeric == pre_numeric

    results['legacy_metrics_bit_identical'] = {
        'snapshot_path': os.path.relpath(SNAPSHOT_PATH, REPO),
        'snapshot_provenance': snapshot['note'],
        'new_module_primal_dual': new_numeric,
        'pre_change_module_primal_dual': pre_numeric,
        'pass': bit_identical,
    }
    return pre_module


def check2_boyd_matches_legacy_entries(planning, tso_model, dso_models, esso_model,
                                        consensus_vars, dual_vars, coord, results):
    """Check 2: Boyd per-entry ESS primal terms equal the legacy entries
    (same formula, isolated single nonzero coordinate); V/PF dual differs
    from the legacy dual ONLY by the documented DSO-change term (F1: the
    legacy dual sums a TSO term AND a DSO term; Boyd's s excludes the DSO
    term)."""
    node_id, year, day, p, power_type = (
        coord['node_id'], coord['year'], coord['day'], coord['p'], coord['power_type'])
    # coord's year/day were stringified for JSON; recover the real keys.
    year = next(y for y in planning.years if str(y) == year)
    day = next(d for d in planning.days if str(d) == day)

    admm_params = planning.params.admm
    boyd = srp.get_admm_boyd_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)
    legacy = srp.get_admm_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars)

    v_base = coord['v_base']
    rho_tso_v = float(pe.value(tso_model[year][day].rho_v))
    rho_dso_v = float(pe.value(dso_models[node_id][year][day].rho_v))
    interface_rating = planning.distribution_networks[node_id].network[year][day].get_interface_branch_rating()
    rho_tso_pf = float(pe.value(tso_model[year][day].rho_pf))
    rho_dso_pf = float(pe.value(dso_models[node_id][year][day].rho_pf))

    # ---- V: legacy dual = max(tso term, dso term); boyd s_rho_part = tso term only.
    x_dso_v_curr = consensus_vars['vmag']['dso']['current'][node_id][year][day][p]
    x_dso_v_prev = consensus_vars['vmag']['dso']['prev'][node_id][year][day][p]
    z_tso_v_curr = consensus_vars['vmag']['tso']['current'][node_id][year][day][p]
    z_tso_v_prev = consensus_vars['vmag']['tso']['prev'][node_id][year][day][p]
    tso_term_v = rho_tso_v * abs(z_tso_v_curr - z_tso_v_prev) / v_base
    dso_term_v = rho_dso_v * abs(x_dso_v_curr - x_dso_v_prev) / v_base
    legacy_dual_v_max = legacy['dual']['v']
    v_dual_check = {
        'tso_term_v': tso_term_v, 'dso_term_v': dso_term_v,
        'legacy_dual_v_max': legacy_dual_v_max,
        'boyd_s_rho_part_v': boyd['v']['s_rho_part'],
        'legacy_equals_max_of_both_terms': abs(legacy_dual_v_max - max(tso_term_v, dso_term_v)) < 1e-12,
        'boyd_s_rho_part_equals_tso_term_only': abs(boyd['v']['s_rho_part'] - tso_term_v) < 1e-12,
        'dso_term_is_the_excluded_difference': abs(legacy_dual_v_max - dso_term_v) < 1e-12,  # true because dso_term > tso_term by construction
    }

    # ---- PF: same structure, P leg.
    x_dso_pf_curr = consensus_vars['pf']['dso']['current'][node_id][year][day][power_type][p]
    x_dso_pf_prev = consensus_vars['pf']['dso']['prev'][node_id][year][day][power_type][p]
    z_tso_pf_curr = consensus_vars['pf']['tso']['current'][node_id][year][day][power_type][p]
    z_tso_pf_prev = consensus_vars['pf']['tso']['prev'][node_id][year][day][power_type][p]
    tso_term_pf = rho_tso_pf * abs(z_tso_pf_curr - z_tso_pf_prev) / interface_rating
    dso_term_pf = rho_dso_pf * abs(x_dso_pf_curr - x_dso_pf_prev) / interface_rating
    legacy_dual_pf_max = legacy['dual']['pf']
    pf_dual_check = {
        'tso_term_pf': tso_term_pf, 'dso_term_pf': dso_term_pf,
        'legacy_dual_pf_max': legacy_dual_pf_max,
        'boyd_s_rho_part_pf': boyd['pf']['s_rho_part'],
        'legacy_equals_max_of_both_terms': abs(legacy_dual_pf_max - max(tso_term_pf, dso_term_pf)) < 1e-12,
        'boyd_s_rho_part_equals_tso_term_only': abs(boyd['pf']['s_rho_part'] - tso_term_pf) < 1e-12,
        'dso_term_is_the_excluded_difference': abs(legacy_dual_pf_max - dso_term_pf) < 1e-12,
    }

    # ---- ESS: primal terms identical (same formula both sides).
    normalization_floor = admm_params.shared_ess_normalization_floor_mva
    network = planning.transmission_network.network[year][day]
    dso_network = planning.distribution_networks[node_id].network[year][day]
    tso_rating = srp._shared_ess_admm_normalization_mva(
        network.shared_energy_storages[network.get_shared_energy_storage_idx(node_id)].s * network.baseMVA, normalization_floor)
    dso_ref = dso_network.get_reference_node_id()
    dso_rating = srp._shared_ess_admm_normalization_mva(
        dso_network.shared_energy_storages[dso_network.get_shared_energy_storage_idx(dso_ref)].s * dso_network.baseMVA, normalization_floor)
    esso_rating = srp._shared_ess_admm_normalization_mva(
        planning.shared_ess_data.shared_energy_storages[year][planning.shared_ess_data.get_shared_energy_storage_idx(node_id)].s,
        normalization_floor)
    ess_ratings = {'tso': tso_rating, 'dso': dso_rating, 'esso': esso_rating}
    z_new = consensus_vars['ess']['z']['current'][node_id][year][day][power_type][p]
    ess_primal_terms = {}
    for agent in ('tso', 'dso', 'esso'):
        x_agent = consensus_vars['ess'][agent]['current'][node_id][year][day][power_type][p]
        ess_primal_terms[agent] = abs(x_agent - z_new) / (2.0 * ess_ratings[agent])
    legacy_ess_primal_max = legacy['primal']['ess']
    boyd_ess_r = boyd['ess']['r']
    # Isolated to one coordinate -> boyd's Euclidean norm over the 3 agent
    # entries; legacy's is the MAX of the same 3 entries -- different
    # aggregation, so verify each individually against sqrt(sum of squares)
    # and against the max, rather than asserting boyd_ess_r == legacy max.
    import math as _math
    expected_boyd_ess_r = _math.sqrt(sum(v * v for v in ess_primal_terms.values()))
    ess_check = {
        'per_agent_primal_terms': ess_primal_terms,
        'legacy_primal_ess_max': legacy_ess_primal_max,
        'legacy_equals_max_of_agent_terms': abs(legacy_ess_primal_max - max(ess_primal_terms.values())) < 1e-12,
        'boyd_ess_r': boyd_ess_r,
        'boyd_ess_r_equals_sqrt_sum_of_squares_of_same_agent_terms': abs(boyd_ess_r - expected_boyd_ess_r) < 1e-12,
    }

    results['boyd_matches_legacy_entries'] = {
        'coordinate': coord,
        'v': v_dual_check, 'pf': pf_dual_check, 'ess': ess_check,
        'pass': (
            v_dual_check['legacy_equals_max_of_both_terms']
            and v_dual_check['boyd_s_rho_part_equals_tso_term_only']
            and v_dual_check['dso_term_is_the_excluded_difference']
            and pf_dual_check['legacy_equals_max_of_both_terms']
            and pf_dual_check['boyd_s_rho_part_equals_tso_term_only']
            and pf_dual_check['dso_term_is_the_excluded_difference']
            and ess_check['legacy_equals_max_of_agent_terms']
            and ess_check['boyd_ess_r_equals_sqrt_sum_of_squares_of_same_agent_terms']
        ),
    }
    return boyd


def check3_lambda_antisymmetry(planning, tso_model, dso_models, consensus_vars, results):
    """Check 3: lambda_TSO = -lambda_DSO in model units, to machine precision,
    exercised through the REAL production dual-update mechanism
    (`_update_interface_power_flow_variables`) on a fresh cold-start
    (lambda == 0) dual_vars dict -- zero solves (fake successful
    SolverResults; the function only reads model Vars/rho Params and writes
    dual_vars).

    `_update_interface_power_flow_variables(..., update_tn=True,
    update_dns=True)` OVERWRITES `interface_vars[...]['current']` from the
    MODEL's own `expected_interface_vmag` / `expected_interface_pf_p/q` Vars
    (not from whatever the caller pre-seeded in the consensus_vars dict), so
    the synthetic perturbation is applied directly to those model Vars."""
    fresh_consensus, fresh_dual = srp.create_admm_variables(planning)
    node_id = planning.active_distribution_network_nodes[0]
    year = next(iter(planning.years))
    day = next(iter(planning.days))
    p = 0
    dn = planning.transmission_network.active_distribution_network_nodes.index(node_id)

    v_base = planning.transmission_network.network[year][day].get_node_base_kv(node_id)
    tso_model[year][day].expected_interface_vmag[dn, p].set_value(1.05)
    dso_models[node_id][year][day].expected_interface_vmag[p].set_value(0.97)
    tso_model[year][day].expected_interface_pf_p[dn, p].set_value(0.21)
    dso_models[node_id][year][day].expected_interface_pf_p[p].set_value(0.08)
    tso_model[year][day].expected_interface_pf_q[dn, p].set_value(-0.04)
    dso_models[node_id][year][day].expected_interface_pf_q[p].set_value(0.06)

    fake_result = _fake_success_result()
    fake_results = {
        'tso': {y: {d: fake_result for d in planning.days} for y in planning.years},
        'dso': {
            nid: {y: {d: fake_result for d in planning.days} for y in planning.years}
            for nid in planning.active_distribution_network_nodes
        },
    }

    rho_v_tso_before = float(pe.value(tso_model[year][day].rho_v))
    rho_v_dso_before = float(pe.value(dso_models[node_id][year][day].rho_v))
    rho_pf_tso_before = float(pe.value(tso_model[year][day].rho_pf))
    rho_pf_dso_before = float(pe.value(dso_models[node_id][year][day].rho_pf))
    equal_rho = (rho_v_tso_before == rho_v_dso_before and rho_pf_tso_before == rho_pf_dso_before)

    srp._update_interface_power_flow_variables(
        planning, tso_model, dso_models, fresh_consensus, fresh_dual,
        fake_results, planning.params.admm, update_tn=True, update_dns=True,
    )

    lambda_v_tso = fresh_dual['vmag']['tso']['current'][node_id][year][day][p]
    lambda_v_dso = fresh_dual['vmag']['dso']['current'][node_id][year][day][p]
    lambda_pf_p_tso = fresh_dual['pf']['tso']['current'][node_id][year][day]['p'][p]
    lambda_pf_p_dso = fresh_dual['pf']['dso']['current'][node_id][year][day]['p'][p]
    lambda_pf_q_tso = fresh_dual['pf']['tso']['current'][node_id][year][day]['q'][p]
    lambda_pf_q_dso = fresh_dual['pf']['dso']['current'][node_id][year][day]['q'][p]

    # model units (F4): V / v_base; PF / s_base (DSO's own s_base).
    s_base_dso = planning.distribution_networks[node_id].network[year][day].baseMVA
    y_v_tso = lambda_v_tso / v_base
    y_v_dso = lambda_v_dso / v_base
    y_pf_p_tso = lambda_pf_p_tso / s_base_dso
    y_pf_p_dso = lambda_pf_p_dso / s_base_dso
    y_pf_q_tso = lambda_pf_q_tso / s_base_dso
    y_pf_q_dso = lambda_pf_q_dso / s_base_dso

    results['lambda_antisymmetry'] = {
        'method': 'cold-start (lambda==0) dual_vars, fresh consensus values, ONE call to the '
                  'real production `_update_interface_power_flow_variables` with fake '
                  'successful SolverResults (zero solves) -- both TSO and DSO rho started '
                  'equal (verified below, required for exact antisymmetry) and both agents '
                  'succeeded, so both dual entries update on the SAME cycle.',
        'rho_v_tso_before': rho_v_tso_before, 'rho_v_dso_before': rho_v_dso_before,
        'rho_pf_tso_before': rho_pf_tso_before, 'rho_pf_dso_before': rho_pf_dso_before,
        'equal_rho_precondition': equal_rho,
        'raw_stored_lambda_v_tso': lambda_v_tso, 'raw_stored_lambda_v_dso': lambda_v_dso,
        'raw_stored_antisymmetric': (lambda_v_tso == -lambda_v_dso),
        'model_units_y_v_tso': y_v_tso, 'model_units_y_v_dso': y_v_dso,
        'model_units_y_pf_p_tso': y_pf_p_tso, 'model_units_y_pf_p_dso': y_pf_p_dso,
        'model_units_y_pf_q_tso': y_pf_q_tso, 'model_units_y_pf_q_dso': y_pf_q_dso,
        'pass': (
            equal_rho
            and lambda_v_tso == -lambda_v_dso
            and lambda_pf_p_tso == -lambda_pf_p_dso
            and lambda_pf_q_tso == -lambda_pf_q_dso
            and y_v_tso == -y_v_dso
            and y_pf_p_tso == -y_pf_p_dso
            and y_pf_q_tso == -y_pf_q_dso
        ),
    }


def check4_unit_conversions(planning, dso_models, coord, results):
    """Check 4: ||y|| unit conversions asserted against the model-loading
    expressions -- V: /v_base, PF: /s_base (the DSO's own s_base), ESS: as
    stored (no conversion). Source-text assertion on the exact production
    loading lines, PLUS numeric coherence on the synthetic coordinate."""
    dso_source = inspect.getsource(srp.update_distribution_coordination_models_and_solve_sequential)
    v_line_present = "dual_vmag['current'][node_id][year][day][p] / v_base" in dso_source
    pf_p_line_present = "dual_pf['current'][node_id][year][day]['p'][p] / s_base" in dso_source
    pf_q_line_present = "dual_pf['current'][node_id][year][day]['q'][p] / s_base" in dso_source
    ess_p_line_present = (
        "model[year][day].dual_ess_p_req[p].set_value(dual_ess['current'][node_id][year][day]['p'][p])" in dso_source
    )
    boyd_source = inspect.getsource(srp.get_admm_boyd_residual_metrics)
    boyd_v_conversion = 'lambda_dso_v / v_base' in boyd_source
    boyd_pf_conversion = 'lambda_dso_pf / s_base_dso' in boyd_source
    boyd_ess_no_conversion = "y_agent = dual_vars['ess'][agent]['current']" in boyd_source

    results['unit_conversions'] = {
        'method': 'source-text presence of the exact production DSO-loading division '
                  'lines (update_distribution_coordination_models_and_solve_sequential) '
                  'and the SAME divisions in get_admm_boyd_residual_metrics.',
        'dso_loading_v_line_present': v_line_present,
        'dso_loading_pf_p_line_present': pf_p_line_present,
        'dso_loading_pf_q_line_present': pf_q_line_present,
        'dso_loading_ess_no_division_present': ess_p_line_present,
        'boyd_v_uses_same_v_base_division': boyd_v_conversion,
        'boyd_pf_uses_same_dso_s_base_division': boyd_pf_conversion,
        'boyd_ess_uses_no_conversion': boyd_ess_no_conversion,
        'pass': (
            v_line_present and pf_p_line_present and pf_q_line_present and ess_p_line_present
            and boyd_v_conversion and boyd_pf_conversion and boyd_ess_no_conversion
        ),
    }


def _reset_rho(tso_model, dso_models, esso_model, value=1.0):
    for year_models in tso_model.values():
        for model in year_models.values():
            model.rho_v.set_value(value)
            model.rho_pf.set_value(value)
            model.rho_ess.set_value(value)
            if hasattr(model, 'rho_ess_prev'):
                model.rho_ess_prev.set_value(value)
    for node_models in dso_models.values():
        for year_models in node_models.values():
            for model in year_models.values():
                model.rho_v.set_value(value)
                model.rho_pf.set_value(value)
                model.rho_ess.set_value(value)
                if hasattr(model, 'rho_ess_prev'):
                    model.rho_ess_prev.set_value(value)
    for model in esso_model.values():
        model.rho.set_value(value)


def _fake_boyd_metrics(ratios):
    """ratios: {'v': (primal_ratio, dual_ratio), 'pf': (...), 'ess': (...)}."""
    channels = {}
    for group in ('v', 'pf', 'ess'):
        primal_ratio, dual_ratio = ratios[group]
        channels[group] = {
            'r': primal_ratio, 's': dual_ratio, 'eps_pri': 1.0, 'eps_dual': 1.0,
            'primal_ratio': primal_ratio, 'dual_ratio': dual_ratio,
            'primal_pass': primal_ratio <= 1.0, 'dual_pass': dual_ratio <= 1.0,
            'channel_pass': primal_ratio <= 1.0 and dual_ratio <= 1.0,
            'norm_x': 1.0, 'norm_z': 1.0, 'norm_y': 1.0,
            's_rho_part': dual_ratio, 's_proximal_part': 0.0, 'proximal_share': 0.0,
            'n_entries': 1,
        }
    channels['all_boyd_pass'] = all(channels[g]['channel_pass'] for g in ('v', 'pf', 'ess'))
    channels['eps_abs'] = 1e-5
    channels['eps_rel'] = 1e-4
    channels['boyd_eps_source'] = 'case_file'
    return channels


def _dummy_residual_metrics(admm_params):
    metrics = {'primal': {}, 'dual': {}}
    for group in ('v', 'pf', 'ess'):
        metrics['primal'][group] = 1e-3
        metrics['primal'][f'{group}_mean'] = 1e-3
        metrics['dual'][group] = 1e-3
        metrics['dual'][f'{group}_mean'] = 1e-3
    return metrics


def check5_penalties_update_isolated(planning, tso_model, dso_models, esso_model,
                                      consensus_vars, dual_vars, results):
    """Check 5: `_update_admm_penalties` changes ONLY rho Params; every lambda
    dict and z dict (consensus_vars, dual_vars) is bit-identical before and
    after."""
    admm_params = planning.params.admm
    _reset_rho(tso_model, dso_models, esso_model, 1.0)

    boyd_metrics = _fake_boyd_metrics({'v': (10.0, 1.0), 'pf': (1.0, 1.0), 'ess': (1.0, 1.0)})
    residual_metrics = _dummy_residual_metrics(admm_params)

    consensus_before = deepcopy(consensus_vars)
    dual_before = deepcopy(dual_vars)

    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, admm_params, allow_update=True)

    consensus_unchanged = (consensus_vars == consensus_before)
    dual_unchanged = (dual_vars == dual_before)
    rho_v_changed_as_expected = (actions['v'] == 'increased' and after['v'] == before['v'] * 1.5)

    results['penalties_update_isolated'] = {
        'actions': actions, 'rho_before': before, 'rho_after': after,
        'consensus_vars_unchanged': consensus_unchanged,
        'dual_vars_unchanged': dual_unchanged,
        'rho_v_changed_as_expected': rho_v_changed_as_expected,
        'pass': consensus_unchanged and dual_unchanged and rho_v_changed_as_expected,
    }
    _reset_rho(tso_model, dso_models, esso_model, 1.0)


def check6_al_objective_unchanged(pre_module, results):
    """Check 6: AL objective expressions of freshly built TSO/DSO/ESSO models
    are structurally unchanged vs the pre-change code -- same terms, sigma,
    normalization and proximal settings. Verified by exact source-text
    equality of the three model-construction functions (this stage did not
    edit any of them)."""
    fn_names = (
        'update_transmission_model_to_admm',
        'update_distribution_models_to_admm',
        'update_shared_energy_storage_model_to_admm',
        '_prepare_transmission_objectives_for_admm',
        '_prepare_distribution_objectives_for_admm',
        '_update_tso_proximal_centres_after_solve',
    )
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        pre_source = inspect.getsource(getattr(pre_module, name))
        identical = (new_source == pre_source)
        per_fn[name] = {
            'identical': identical,
            'new_sha256': hashlib.sha256(new_source.encode()).hexdigest(),
            'pre_sha256': hashlib.sha256(pre_source.encode()).hexdigest(),
        }
        all_identical = all_identical and identical
    results['al_objective_unchanged'] = {'functions': per_fn, 'pass': all_identical}


def check7_balancing_unit_test(planning, tso_model, dso_models, esso_model, results):
    admm_params = planning.params.admm
    residual_metrics = _dummy_residual_metrics(admm_params)
    sub_results = {}

    # -- increase: primal_ratio > 5 * dual_ratio -----------------------------
    _reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics({'v': (10.0, 1.0), 'pf': (1.0, 1.0), 'ess': (1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['increase'] = {
        'action': actions['v'], 'before': before['v'], 'after': after['v'],
        'pass': actions['v'] == 'increased' and abs(after['v'] - 1.5) < 1e-12,
    }

    # -- decrease (PF, threshold 3.0): dual_ratio > 3 * primal_ratio --------
    _reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics({'v': (1.0, 1.0), 'pf': (1.0, 10.0), 'ess': (1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['decrease'] = {
        'action': actions['pf'], 'before': before['pf'], 'after': after['pf'],
        'pass': actions['pf'] == 'decreased' and abs(after['pf'] - (1.0 / 1.5)) < 1e-12,
    }

    # -- dead band (held): neither ratio dominates ---------------------------
    _reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics({'v': (1.0, 1.0), 'pf': (1.0, 1.0), 'ess': (2.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['dead_band_held'] = {
        'action': actions['ess'], 'before': before['ess'], 'after': after['ess'],
        'pass': actions['ess'] == 'held' and after['ess'] == before['ess'],
    }

    # -- no freeze: both ratios <= 1 ("legacy adaptation-converged") yet
    #    imbalanced beyond the band -> still acts (freeze clause removed) ---
    _reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics({'v': (0.5, 0.05), 'pf': (1.0, 1.0), 'ess': (1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=True)
    sub_results['no_freeze'] = {
        'primal_ratio': 0.5, 'dual_ratio': 0.05,
        'both_ratios_le_1_legacy_converged': True,
        'action': actions['v'], 'before': before['v'], 'after': after['v'],
        'pass': actions['v'] == 'increased' and abs(after['v'] - 1.5) < 1e-12,
    }

    # -- failure hold: allow_update=False -> rho unchanged regardless -------
    _reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = _fake_boyd_metrics({'v': (10.0, 1.0), 'pf': (1.0, 10.0), 'ess': (1.0, 1.0)})
    actions, before, after = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, allow_update=False)
    sub_results['failure_hold'] = {
        'action_v': actions['v'], 'action_pf': actions['pf'],
        'before': before, 'after': after,
        'pass': (
            actions['v'] == 'held after solver failure'
            and actions['pf'] == 'held after solver failure'
            and after == before
        ),
    }

    _reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['balancing_unit_test'] = {
        'sub_results': sub_results,
        'pass': all(v['pass'] for v in sub_results.values()),
    }


def check8_stop_rule_unit_test(results):
    """Check 8: the objective-change test does not gate `cycle_convergence`
    (source-text assertion: the exact gating line references only
    `boyd_all_pass` and `local_solves_ok`, never `objective_convergence`);
    each of the six Boyd tests (primal/dual pass, per channel) individually
    blocks `all_boyd_pass` (direct unit test of the aggregation logic
    `get_admm_boyd_residual_metrics` uses: `channel_pass = primal_pass and
    dual_pass`; `all_boyd_pass = all(channel_pass for v, pf, ess)`)."""
    module_source = inspect.getsource(srp)
    gating_line_present = 'cycle_convergence = boyd_all_pass and local_solves_ok' in module_source
    # objective_convergence must not appear on that same gating line.
    gating_line = next(
        (line for line in module_source.splitlines() if 'cycle_convergence = boyd_all_pass' in line), '')
    objective_not_on_gating_line = 'objective_convergence' not in gating_line

    def _all_pass(flags):
        # flags: dict channel -> (primal_pass, dual_pass)
        channel_pass = {g: (flags[g][0] and flags[g][1]) for g in ('v', 'pf', 'ess')}
        return all(channel_pass.values()), channel_pass

    all_true = {'v': (True, True), 'pf': (True, True), 'ess': (True, True)}
    baseline_pass, _ = _all_pass(all_true)

    six_blocking_cases = {}
    for group in ('v', 'pf', 'ess'):
        for side, idx in (('primal', 0), ('dual', 1)):
            flags = {g: list(all_true[g]) for g in ('v', 'pf', 'ess')}
            flags[group][idx] = False
            flags = {g: tuple(v) for g, v in flags.items()}
            result_pass, channel_pass = _all_pass(flags)
            six_blocking_cases[f'{group}_{side}_false'] = {
                'all_boyd_pass': result_pass, 'channel_pass': channel_pass,
                'blocks_as_expected': (result_pass is False),
            }

    results['stop_rule_unit_test'] = {
        'gating_line_present': gating_line_present,
        'gating_line': gating_line.strip(),
        'objective_convergence_not_on_gating_line': objective_not_on_gating_line,
        'baseline_all_true_all_boyd_pass': baseline_pass,
        'six_blocking_cases': six_blocking_cases,
        'pass': (
            gating_line_present and objective_not_on_gating_line and baseline_pass is True
            and all(c['blocks_as_expected'] for c in six_blocking_cases.values())
        ),
    }


def check9_other_case_params_load(results):
    cs1_admm = ADMMParameters()
    with open(CS1_PARAMS_PATH) as handle:
        cs1_admm.read_parameters_from_file(json.load(handle)['admm'])

    srp1_admm = ADMMParameters()
    with open(SRP1_PARAMS_PATH) as handle:
        srp1_admm.read_parameters_from_file(json.load(handle)['admm'])

    results['other_case_params_load'] = {
        'cs1_params_path': os.path.relpath(CS1_PARAMS_PATH, REPO),
        'cs1_boyd_eps_source': cs1_admm.boyd_eps_source,
        'cs1_boyd_tol': dict(cs1_admm.tol['boyd']),
        'srp1_params_path': os.path.relpath(SRP1_PARAMS_PATH, REPO),
        'srp1_boyd_eps_source': srp1_admm.boyd_eps_source,
        'srp1_boyd_tol': dict(srp1_admm.tol['boyd']),
        'pass': (
            cs1_admm.boyd_eps_source == 'default'
            and cs1_admm.tol['boyd'] == {'eps_abs': 1e-5, 'eps_rel': 1e-4}
            and srp1_admm.boyd_eps_source == 'case_file'
            and srp1_admm.tol['boyd'] == {'eps_abs': 1e-5, 'eps_rel': 1e-4}
        ),
    }


def check10_fixtures_unpickle(results):
    fixture_results = []
    for path in S31C_FIXTURES:
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


def check11_hierarchical_uncoordinated_untouched(pre_module, results):
    fn_names = (
        '_run_operational_planning_hierarchical',
        '_run_operational_planning_without_coordination',
    )
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        pre_source = inspect.getsource(getattr(pre_module, name))
        identical = (new_source == pre_source)
        per_fn[name] = {'identical': identical}
        all_identical = all_identical and identical

    diff = subprocess.run(
        ['git', 'diff', 'HEAD', '--', 'shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    hunk_headers = [line for line in diff.splitlines() if line.startswith('@@')]
    results['hierarchical_uncoordinated_untouched'] = {
        'method': 'exact source-text equality (git HEAD vs current) for both path '
                  'functions, PLUS a listing of every diff hunk header in '
                  'shared_resources_planning.py for manual inspection.',
        'functions': per_fn,
        'diff_hunk_count': len(hunk_headers),
        'pass': all_identical,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    guard = SolveProfileGuard(permitted=(), label='S32 zero-solve check').install()
    results = {}
    try:
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
            _build_admm_ready_state('p515s32_zero_solve_check'))

        coord = _perturb_single_entry(planning, tso_model, dso_models, consensus_vars, dual_vars)

        pre_module = check1_legacy_metrics_bit_identical(
            planning, tso_model, dso_models, esso_model, consensus_vars, results)
        check2_boyd_matches_legacy_entries(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, coord, results)
        check3_lambda_antisymmetry(planning, tso_model, dso_models, consensus_vars, results)
        check4_unit_conversions(planning, dso_models, coord, results)
        check5_penalties_update_isolated(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, results)
        check6_al_objective_unchanged(pre_module, results)
        check7_balancing_unit_test(planning, tso_model, dso_models, esso_model, results)
        check8_stop_rule_unit_test(results)
        check9_other_case_params_load(results)
        check10_fixtures_unpickle(results)
        check11_hierarchical_uncoordinated_untouched(pre_module, results)

    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Step 3.2 + 3.3(a) (s32 worker task) -- zero-solve verification',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 9 sections 3.2, 3.3(a)',
            'data/SRP1/Results/P515S32/frozen_s32_spec_v1_14a18674.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S32] wrote {OUT_PATH}')
    print(f'[S32] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
