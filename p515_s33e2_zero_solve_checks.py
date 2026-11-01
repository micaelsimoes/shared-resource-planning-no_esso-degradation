"""
P5.15 Step 3.2 E2 (s33e2 worker task) -- zero-solve verification.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 14; binding specification
data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json (supersedes v2
data/SRP1/Results/P515S32/frozen_s32_spec_v2_516bd749.json). The changes
under test:

  * TSO prox_gamma_v/pf/ess become MUTABLE Params (were immutable).
  * `admm.proximal_regularization.tso.gamma_policy` ('fixed' | 'tied_to_rho',
    default 'fixed') and `.tau` (default 1.0): under 'tied_to_rho', gamma_c
    is initialized to tau*(initial rho_c) at model construction, and
    `_update_admm_penalties` resets gamma_c = tau*rho_c whenever it scales
    rho (i.e. whenever it actually changes rho, guarded the same way).
  * `get_admm_boyd_residual_metrics` reads gamma from the TSO model's
    `prox_gamma_*` Params IN FORCE, not the case-file constant.
  * `admm.penalty_update.freeze_after_cycle` (default None): when set and
    `iter > freeze_after_cycle`, EVERY channel is held
    (`'held (frozen after cycle N)'`), with no rho or gamma change,
    REGARDLESS of `allow_update` (the failure hold's own label/precedence
    applies only when NOT frozen).
  * `admm.minimum_consecutive_converged_cycles` (case file: 3, unchanged
    code) gates `convergence` via the pre-existing
    `consecutive_converged_cycles`/`cycle_convergence` counter.
  * SRP1's case file now sets `gamma_policy='tied_to_rho'`, `tau=1.0`,
    `freeze_after_cycle=30`, `minimum_consecutive_converged_cycles=3` --
    every OTHER case study keeps loading with the unchanged defaults
    ('fixed', None, 1.0).

Verifies, on FRESHLY BUILT C* ADMM models (NEW eval ids) and on synthetic
single-entry / hand-derived states, checks (a)-(j) below, with a BLOCKING
`SolveProfileGuard` (permitted call sites = (), i.e. zero solves anywhere)
armed for the whole check:

  (a) tied gamma: freshly built TSO models (case-file default,
      gamma_policy='tied_to_rho') have gamma_c == tau*rho_c immediately
      after construction, on every channel and every TSO model; after a
      synthetic balancing decrease AND a synthetic increase through the
      REAL `_update_admm_penalties`, gamma tracks the new rho on every
      channel and TSO model;
  (b) fixed gamma: under 'fixed' (explicit override -- the case file
      default is now tied), gamma stays at the configured value across a
      synthetic decrease/increase, and every legacy metric, every Boyd
      field and the update's actions/rho are numerically identical to
      commit d06f3636 (the pre-v3 Boyd/balancing implementation) on a
      snapshot;
  (c) Boyd reads the model's gamma: on a synthetic state where the TSO
      model's prox_gamma_v is set to a value DIFFERENT from the case-file
      gamma constant, `s`/`s_proximal_part` follow the model value, not the
      constant;
  (d) freeze: at iter=30 (not frozen: 30 > 30 is False) an otherwise-
      provocative ratio acts normally; at iter=31 (frozen) the SAME ratio
      is held with label 'held (frozen after cycle 30)' and rho/gamma are
      byte-unchanged; freeze takes precedence over the failure hold
      (iter=31, allow_update=False -> still the frozen label), while the
      failure hold keeps its own label when NOT frozen (iter=29,
      allow_update=False -> 'held after solver failure');
  (e) stop rule: source-text confirms the exact two-line
      increment/reset formula and the `minimum_consecutive_converged_cycles`
      gate (`shared_resources_planning.py` lines quoted below), and that
      `cycle_convergence` still only reads `boyd_all_pass`/`local_solves_ok`
      (objective test not gating, unchanged from v1/v2); a mechanical
      simulation of that EXACT quoted formula over a
      pass/pass/fail/pass/pass/pass sequence with
      minimum_consecutive_converged_cycles=3 resets after the fail and
      stops on the 3rd consecutive pass thereafter;
  (f) update isolation: `_update_admm_penalties` (tied policy) changes ONLY
      rho and gamma Params on the models; consensus_vars/dual_vars
      (lambda/z) are bit-identical before/after;
  (g) objective structure: `update_transmission_model_to_admm`'s diff vs
      commit d06f3636 is confined to gamma-Param mutability/policy lines
      (no term, coefficient, sigma or proximal-centring change); with
      gamma=1 (case-file default under EITHER policy) the built objective
      evaluates to the SAME number under the current code ('fixed'
      override) and under commit d06f3636, on a hand-completed (zero
      request/dual Params set to 0.0) snapshot;
  (h) other cases: CS1 (and, for completeness, CS7/HR1/OP1/OP2) still load
      with gamma_policy='fixed', tau=1.0, freeze_after_cycle=None;
  (i) fixtures: every S31C-stage preserved fixture unpickles, PLUS the s32
      FrozenSMOPF pickles under data/SRP1/Results/FrozenSMOPF/ (built with
      IMMUTABLE prox_gamma_v, pre-v3) still unpickle under the current,
      mutable-gamma code;
  (j) untouched paths: `_run_operational_planning_hierarchical` and
      `_run_operational_planning_without_coordination` have no hunks
      relative to HEAD.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s33e2_zero_solve_checks.py

Writes data/SRP1/Results/P515S33/zero_solve_checks_e2/zero_solve_checks_e2.json
(a NEW directory; refuses to overwrite).
"""

import hashlib
import inspect
import json
import math
import os
import pickle
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s32_zero_solve_checks as V1  # noqa: E402
import p515_s32v2_zero_solve_checks as V2  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33', 'zero_solve_checks_e2')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks_e2.json')

SPEC_V3_PATH = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33', 'frozen_s33_e2_spec_v3_825f1f02.json')
SPEC_V3_SHA256 = '825f1f02d5319137e0248fc529b9aefdca0251272529ee140e09af6e73a04780'
PRE_CHANGE_COMMIT = 'd06f3636'  # P5.15 Step 3.2+3.3(a): s32 residual balancing, frozen spec v2 (production, pre-v3)

CS_PARAMS_PATHS = {
    'CS1': os.path.join(REPO, 'data', 'CS1', 'CS1_params.json'),
    'CS7': os.path.join(REPO, 'data', 'CS7', 'CS7_params.json'),
    'HR1': os.path.join(REPO, 'data', 'HR1', 'HR1_params.json'),
    'OP1': os.path.join(REPO, 'data', 'OP1', 'OP1_params.json'),
    'OP2': os.path.join(REPO, 'data', 'OP2', 'OP2_params.json'),
}
SRP1_PARAMS_PATH = os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')

# s32 FrozenSMOPF pickles built with IMMUTABLE prox_gamma_v (pre-v3;
# confirmed by inspection: prox_gamma_v.mutable == False, value == 1.0).
S32_FROZEN_SMOPF_FIXTURES = [
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF',
                 'matched_success_TSO_case9_2025_Summer_cycle7.pkl'),
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF',
                 'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
]


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _spec_v3_hash():
    with open(SPEC_V3_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _tso_gamma_snapshot(tso_model):
    """{'v': [...], 'pf': [...], 'ess': [...]} -- one entry per TSO
    (year, day) model, mirroring srp._get_admm_gamma_summary's iteration
    order but WITHOUT averaging (per-model detail, for the "every TSO
    model" checks)."""
    snap = {'v': [], 'pf': [], 'ess': []}
    for year_models in tso_model.values():
        for model in year_models.values():
            snap['v'].append(pe.value(model.prox_gamma_v))
            snap['pf'].append(pe.value(model.prox_gamma_pf))
            snap['ess'].append(pe.value(model.prox_gamma_ess))
    return snap


def _tso_rho_snapshot(tso_model):
    snap = {'v': [], 'pf': [], 'ess': []}
    for year_models in tso_model.values():
        for model in year_models.values():
            snap['v'].append(pe.value(model.rho_v))
            snap['pf'].append(pe.value(model.rho_pf))
            snap['ess'].append(pe.value(model.rho_ess))
    return snap


def _build_admm_ready_state_fixed_gamma(eval_id):
    """Same production sequence as `V1._build_admm_ready_state`, EXCEPT
    `admm_params.proximal_regularization['tso']['gamma_policy']` is forced
    to 'fixed' BEFORE `update_transmission_model_to_admm` builds the AL
    objective/gamma Params (the case-file default is now 'tied_to_rho') --
    exercises the CURRENT code's 'fixed' branch (bit-identical to pre-v3)
    on a fresh eval id. Duplicates `V1._build_admm_ready_state`'s steps
    (that function is a frozen artifact of a prior stage and takes no
    policy override) rather than modifying it."""
    eval_dir = os.path.join(O.WORK_DIR, eval_id)
    if os.path.exists(eval_dir):
        raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
    planning = O.fresh_planning(eval_id)
    planning.params.admm.proximal_regularization['tso']['gamma_policy'] = 'fixed'

    transmission_network = planning.transmission_network
    distribution_networks = planning.distribution_networks
    shared_ess_data = planning.shared_ess_data

    transmission_network.optimize = V1._stub_optimize.__get__(transmission_network, type(transmission_network))
    for _node_id, _dn in distribution_networks.items():
        _dn.optimize = V1._stub_optimize.__get__(_dn, type(_dn))

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
    objective_scale = 1.0
    srp.update_distribution_models_to_admm(planning, dso_models, planning.params.admm, objective_scale)
    srp.update_transmission_model_to_admm(planning, tso_model, planning.params.admm, objective_scale)
    srp.update_shared_energy_storage_model_to_admm(planning, esso_model, planning.params.admm)

    return planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars


def check_a_tied_gamma_tracks_rho(results):
    """Check (a): freshly built TSO models (case-file default,
    gamma_policy='tied_to_rho') have gamma_c == tau*rho_c immediately after
    construction; a synthetic decrease then a synthetic increase (through
    the REAL `_update_admm_penalties`) keep gamma_c == tau*rho_c on every
    channel and every TSO model."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s33e2_check_a_tied_gamma'))
    admm_params = planning.params.admm
    tau = admm_params.proximal_regularization['tso']['tau']
    policy = admm_params.proximal_regularization['tso']['gamma_policy']

    gamma0 = _tso_gamma_snapshot(tso_model)
    rho0 = _tso_rho_snapshot(tso_model)
    init_matches = all(
        abs(g - tau * r) < 1e-12
        for group in ('v', 'pf', 'ess')
        for g, r in zip(gamma0[group], rho0[group])
    )

    residual_metrics = V1._dummy_residual_metrics(admm_params)

    # -- synthetic decrease (all channels) ---------------------------------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    for year_models in tso_model.values():
        for model in year_models.values():
            model.prox_gamma_v.set_value(tau * 1.0)
            model.prox_gamma_pf.set_value(tau * 1.0)
            model.prox_gamma_ess.set_value(tau * 1.0)
    boyd = V2._fake_boyd_metrics_v2({'v': (1.0, 30.0, 10.0), 'pf': (1.0, 30.0, 10.0), 'ess': (1.0, 30.0, 10.0)})
    actions_dec, before_dec, after_dec, gbefore_dec, gafter_dec, frozen_dec = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, iter=1, allow_update=True)
    rho_after_dec = _tso_rho_snapshot(tso_model)
    gamma_after_dec = _tso_gamma_snapshot(tso_model)
    decrease_tracks = (
        all(a == 'decreased' for a in actions_dec.values())
        and all(
            abs(g - tau * r) < 1e-9
            for group in ('v', 'pf', 'ess')
            for g, r in zip(gamma_after_dec[group], rho_after_dec[group])
        )
        and all(abs(gafter_dec[g] - tau * after_dec[g]) < 1e-9 for g in ('v', 'pf', 'ess'))
    )

    # -- synthetic increase (all channels) ---------------------------------
    boyd = V2._fake_boyd_metrics_v2({'v': (30.0, 1.0, 1.0), 'pf': (30.0, 1.0, 1.0), 'ess': (30.0, 1.0, 1.0)})
    actions_inc, before_inc, after_inc, gbefore_inc, gafter_inc, frozen_inc = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, iter=2, allow_update=True)
    rho_after_inc = _tso_rho_snapshot(tso_model)
    gamma_after_inc = _tso_gamma_snapshot(tso_model)
    increase_tracks = (
        all(a == 'increased' for a in actions_inc.values())
        and all(
            abs(g - tau * r) < 1e-9
            for group in ('v', 'pf', 'ess')
            for g, r in zip(gamma_after_inc[group], rho_after_inc[group])
        )
        and all(abs(gafter_inc[g] - tau * after_inc[g]) < 1e-9 for g in ('v', 'pf', 'ess'))
    )

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['tied_gamma_tracks_rho'] = {
        'gamma_policy': policy, 'tau': tau,
        'initial_gamma_equals_tau_times_rho': init_matches,
        'gamma0': gamma0, 'rho0': rho0,
        'decrease': {
            'actions': actions_dec, 'rho_after': rho_after_dec, 'gamma_after': gamma_after_dec,
            'gamma_summary_after': gafter_dec, 'frozen': frozen_dec, 'tracks': decrease_tracks,
        },
        'increase': {
            'actions': actions_inc, 'rho_after': rho_after_inc, 'gamma_after': gamma_after_inc,
            'gamma_summary_after': gafter_inc, 'frozen': frozen_inc, 'tracks': increase_tracks,
        },
        'pass': (policy == 'tied_to_rho' and tau == 1.0 and init_matches and decrease_tracks and increase_tracks),
    }


def check_b_fixed_gamma(results):
    """Check (b): under 'fixed' (explicit override), gamma stays at the
    configured value across a synthetic decrease/increase, and every
    legacy metric, Boyd field, action and rho is numerically identical to
    commit d06f3636 on a snapshot."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        _build_admm_ready_state_fixed_gamma('p515s33e2_check_b_fixed_gamma'))
    admm_params = planning.params.admm
    configured_gamma = dict(admm_params.proximal_regularization['tso']['gamma'])

    gamma0 = _tso_gamma_snapshot(tso_model)
    init_matches_configured = all(
        abs(g - configured_gamma[group]) < 1e-12
        for group in ('v', 'pf', 'ess') for g in gamma0[group]
    )

    residual_metrics = V1._dummy_residual_metrics(admm_params)
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    boyd = V2._fake_boyd_metrics_v2({'v': (1.0, 30.0, 10.0), 'pf': (30.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})
    actions, before, after, gbefore, gafter, frozen = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, iter=1, allow_update=True)
    gamma_after = _tso_gamma_snapshot(tso_model)
    gamma_unchanged = all(
        abs(g - configured_gamma[group]) < 1e-12
        for group in ('v', 'pf', 'ess') for g in gamma_after[group]
    )

    # -- numeric comparison vs commit d06f3636 on a fresh snapshot ---------
    pre_module, _pre_source = V2._load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_d06f3636_b')
    planning2, tso_model2, dso_models2, esso_model2, consensus_vars2, dual_vars2 = (
        V1._build_admm_ready_state('p515s33e2_check_b_pre_change_snapshot'))
    coord2 = V1._perturb_single_entry(planning2, tso_model2, dso_models2, consensus_vars2, dual_vars2)
    admm_params2 = planning2.params.admm

    new_legacy = srp.get_admm_residual_metrics(planning2, tso_model2, dso_models2, esso_model2, consensus_vars2)
    pre_legacy = pre_module.get_admm_residual_metrics(planning2, tso_model2, dso_models2, esso_model2, consensus_vars2)

    def _numeric_subset(d):
        return {
            k1: {k2: v2 for k2, v2 in d[k1].items() if isinstance(v2, (int, float))}
            for k1 in ('primal', 'dual')
        }
    legacy_identical = (_numeric_subset(new_legacy) == _numeric_subset(pre_legacy))

    new_boyd = srp.get_admm_boyd_residual_metrics(
        planning2, tso_model2, dso_models2, esso_model2, consensus_vars2, dual_vars2, admm_params2)
    pre_boyd = pre_module.get_admm_boyd_residual_metrics(
        planning2, tso_model2, dso_models2, esso_model2, consensus_vars2, dual_vars2, admm_params2)

    boyd_ok = True
    boyd_diff = {}
    for group in ('v', 'pf', 'ess'):
        common = set(new_boyd[group]) & set(pre_boyd[group])
        mismatches = {
            k: {'new': new_boyd[group][k], 'pre': pre_boyd[group][k]}
            for k in sorted(common) if new_boyd[group][k] != pre_boyd[group][k]
        }
        boyd_diff[group] = mismatches
        boyd_ok = boyd_ok and (len(mismatches) == 0)

    residual_metrics2 = V1._dummy_residual_metrics(admm_params2)
    V1._reset_rho(tso_model2, dso_models2, esso_model2, 1.0)
    boyd_fake = V2._fake_boyd_metrics_v2({'v': (10.0, 1.0, 1.0), 'pf': (1.0, 30.0, 10.0), 'ess': (1.0, 1.0, 1.0)})
    new_actions, new_before, new_after, _gb, _ga, _fr = srp._update_admm_penalties(
        tso_model2, dso_models2, esso_model2, residual_metrics2, boyd_fake, admm_params2, iter=None, allow_update=True)
    V1._reset_rho(tso_model2, dso_models2, esso_model2, 1.0)
    pre_actions, pre_before, pre_after = pre_module._update_admm_penalties(
        tso_model2, dso_models2, esso_model2, residual_metrics2, boyd_fake, admm_params2, allow_update=True)
    V1._reset_rho(tso_model2, dso_models2, esso_model2, 1.0)
    update_identical = (new_actions == pre_actions and new_before == pre_before and new_after == pre_after)

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['fixed_gamma'] = {
        'configured_gamma': configured_gamma,
        'initial_gamma_equals_configured': init_matches_configured,
        'gamma_unchanged_after_update': gamma_unchanged,
        'update_actions': actions, 'rho_before': before, 'rho_after': after,
        'gamma_before_summary': gbefore, 'gamma_after_summary': gafter, 'frozen': frozen,
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'snapshot_coordinate': coord2,
        'legacy_metrics_identical': legacy_identical,
        'boyd_mismatches_per_channel': boyd_diff,
        'boyd_fields_identical': boyd_ok,
        'penalty_update_actions_rho_identical': update_identical,
        'pass': (
            init_matches_configured and gamma_unchanged and legacy_identical
            and boyd_ok and update_identical
        ),
    }


def check_c_boyd_uses_model_gamma(results):
    """Check (c): on a synthetic state where the TSO model's `prox_gamma_v`
    is set to a value DIFFERENT from the case-file gamma constant,
    `s`/`s_proximal_part` follow the MODEL value."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s33e2_check_c_boyd_model_gamma'))
    admm_params = planning.params.admm
    case_file_gamma_v = admm_params.proximal_regularization['tso']['gamma']['v']

    node_id = planning.active_distribution_network_nodes[0]
    year = next(iter(planning.years))
    day = next(iter(planning.days))
    p = 0
    v_base = planning.transmission_network.network[year][day].get_node_base_kv(node_id)

    # -- set the MODEL gamma to something the case-file constant is NOT ----
    model_gamma_v = case_file_gamma_v + 7.0
    tso_model[year][day].prox_gamma_v.set_value(model_gamma_v)
    tso_model[year][day].rho_v.set_value(1.0)

    dz_norm = 0.03
    consensus_vars['vmag']['tso']['prev'][node_id][year][day][p] = v_base * 1.0
    consensus_vars['vmag']['tso']['current'][node_id][year][day][p] = v_base * (1.0 + dz_norm)
    consensus_vars['vmag']['dso']['current'][node_id][year][day][p] = v_base * (1.0 + dz_norm)  # r_v = 0

    boyd = srp.get_admm_boyd_residual_metrics(
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, admm_params)
    v_entry = boyd['v']

    rho_v = 1.0
    s_proximal_expected_model = model_gamma_v * dz_norm
    s_proximal_expected_case_file = case_file_gamma_v * dz_norm
    s_full_expected_model = math.sqrt(rho_v ** 2 + model_gamma_v ** 2) * dz_norm

    follows_model = (
        abs(v_entry['s_proximal_part'] - s_proximal_expected_model) < 1e-9
        and abs(v_entry['s'] - s_full_expected_model) < 1e-9
    )
    not_case_file = abs(v_entry['s_proximal_part'] - s_proximal_expected_case_file) > 1e-6

    results['boyd_uses_model_gamma'] = {
        'case_file_gamma_v': case_file_gamma_v, 'model_gamma_v': model_gamma_v,
        'dz_norm': dz_norm, 'rho_v': rho_v,
        's_proximal_expected_model': s_proximal_expected_model,
        's_proximal_expected_case_file_WRONG_if_matched': s_proximal_expected_case_file,
        'observed_s_proximal_part': v_entry['s_proximal_part'], 'observed_s': v_entry['s'],
        'follows_model_value': follows_model,
        'does_not_follow_case_file_constant': not_case_file,
        'pass': follows_model and not_case_file,
    }


def check_d_freeze(results):
    """Check (d): freeze precedence and behaviour at the boundary cycle."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s33e2_check_d_freeze'))
    admm_params = planning.params.admm
    freeze_after_cycle = admm_params.penalty_update['freeze_after_cycle']
    residual_metrics = V1._dummy_residual_metrics(admm_params)
    provocative = V2._fake_boyd_metrics_v2({'v': (30.0, 1.0, 1.0), 'pf': (30.0, 1.0, 1.0), 'ess': (30.0, 1.0, 1.0)})

    # -- iter == freeze_after_cycle (30): NOT frozen (30 > 30 is False), acts
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    actions_30, before_30, after_30, gb_30, ga_30, frozen_30 = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, provocative, admm_params,
        iter=freeze_after_cycle, allow_update=True)

    # -- iter == freeze_after_cycle + 1 (31): frozen, held, no change ------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    actions_31, before_31, after_31, gb_31, ga_31, frozen_31 = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, provocative, admm_params,
        iter=freeze_after_cycle + 1, allow_update=True)
    expected_label = f'held (frozen after cycle {freeze_after_cycle})'
    frozen_holds = (
        frozen_31 is True
        and all(a == expected_label for a in actions_31.values())
        and after_31 == before_31
        and all(ga_31[g] == gb_31[g] for g in ('v', 'pf', 'ess') if gb_31[g] is not None)
    )

    # -- frozen takes precedence over the failure hold (iter=31, allow_update=False)
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    actions_31_fail, before_31_fail, after_31_fail, _gb, _ga, frozen_31_fail = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, provocative, admm_params,
        iter=freeze_after_cycle + 1, allow_update=False)
    frozen_precedence_over_failure = (
        frozen_31_fail is True
        and all(a == expected_label for a in actions_31_fail.values())
        and after_31_fail == before_31_fail
    )

    # -- failure hold keeps its own label when NOT frozen (iter=29) --------
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    actions_29_fail, before_29_fail, after_29_fail, _gb2, _ga2, frozen_29_fail = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, provocative, admm_params,
        iter=freeze_after_cycle - 1, allow_update=False)
    failure_hold_when_not_frozen = (
        frozen_29_fail is False
        and all(a == 'held after solver failure' for a in actions_29_fail.values())
        and after_29_fail == before_29_fail
    )

    # -- iter=30 case actually adapted (sanity: the provocative ratio bites)
    acted_at_30 = (frozen_30 is False and all(a == 'increased' for a in actions_30.values()))

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['freeze'] = {
        'freeze_after_cycle': freeze_after_cycle,
        'iter_30_not_frozen_acts': {
            'frozen': frozen_30, 'actions': actions_30, 'before': before_30, 'after': after_30,
            'pass': acted_at_30,
        },
        'iter_31_frozen_holds': {
            'frozen': frozen_31, 'actions': actions_31, 'before': before_31, 'after': after_31,
            'expected_label': expected_label, 'pass': frozen_holds,
        },
        'iter_31_frozen_precedence_over_failure_hold': {
            'frozen': frozen_31_fail, 'actions': actions_31_fail, 'pass': frozen_precedence_over_failure,
        },
        'iter_29_failure_hold_own_label_when_not_frozen': {
            'frozen': frozen_29_fail, 'actions': actions_29_fail, 'pass': failure_hold_when_not_frozen,
        },
        'precedence_implemented': (
            'frozen (highest) > not-adaptive (\'fixed\') > failure hold '
            '(\'held after solver failure\', own label, ONLY when not frozen) > '
            'ordinary increase/decrease/dead-band'
        ),
        'pass': (
            acted_at_30 and frozen_holds and frozen_precedence_over_failure and failure_hold_when_not_frozen
        ),
    }


def check_e_stop_rule(results):
    """Check (e): source-text confirms the exact increment/reset formula and
    that the objective test does not gate `cycle_convergence`; a mechanical
    simulation of that EXACT quoted formula requires 3 consecutive all-pass
    cycles (2 then a fail resets the count; 3 stops)."""
    module_source = inspect.getsource(srp)
    lines_present = {
        'increment': 'consecutive_converged_cycles += 1' in module_source,
        'reset': 'consecutive_converged_cycles = 0' in module_source,
        'gate': 'convergence = (consecutive_converged_cycles >= admm_parameters.minimum_consecutive_converged_cycles)' in module_source,
        'cycle_convergence_gating_line': 'cycle_convergence = boyd_all_pass and local_solves_ok' in module_source,
    }
    gating_line = next(
        (line for line in module_source.splitlines() if 'cycle_convergence = boyd_all_pass' in line), '')
    objective_not_gating = 'objective_convergence' not in gating_line

    def _simulate(pass_sequence, minimum):
        """Mechanical reproduction of the two quoted lines
        (`consecutive_converged_cycles += 1` / `= 0`) plus the quoted gate
        -- NOT a re-implementation of any algorithmic decision (the
        increment/reset/gate are a fixed three-line arithmetic recipe,
        confirmed present verbatim in `module_source` above)."""
        count = 0
        history = []
        stopped_at = None
        for i, cycle_pass in enumerate(pass_sequence, start=1):
            count = count + 1 if cycle_pass else 0
            convergence = count >= minimum
            history.append({'cycle': i, 'cycle_convergence': cycle_pass,
                             'consecutive_converged_cycles': count, 'convergence': convergence})
            if convergence and stopped_at is None:
                stopped_at = i
        return history, stopped_at

    sequence = [True, True, False, True, True, True]
    history, stopped_at = _simulate(sequence, minimum=3)
    reset_after_fail = (history[2]['consecutive_converged_cycles'] == 0)  # cycle 3 (index 2) is the False
    stops_on_third_consecutive_after_reset = (stopped_at == 6)

    results['stop_rule'] = {
        'source_lines_present': lines_present,
        'gating_line': gating_line.strip(),
        'objective_test_not_gating': objective_not_gating,
        'minimum_consecutive_converged_cycles_case_file': 3,
        'simulated_sequence': sequence,
        'simulated_history': history,
        'reset_after_fail': reset_after_fail,
        'stopped_at_cycle': stopped_at,
        'stops_on_third_consecutive_after_reset': stops_on_third_consecutive_after_reset,
        'pass': (
            all(lines_present.values()) and objective_not_gating
            and reset_after_fail and stops_on_third_consecutive_after_reset
        ),
    }


def check_f_update_isolation(results):
    """Check (f): `_update_admm_penalties` (tied policy) changes ONLY rho
    and gamma Params on the models; consensus_vars/dual_vars are
    bit-identical before/after."""
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s33e2_check_f_isolation'))
    admm_params = planning.params.admm
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)

    residual_metrics = V1._dummy_residual_metrics(admm_params)
    boyd = V2._fake_boyd_metrics_v2({'v': (30.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)})

    consensus_before = deepcopy(consensus_vars)
    dual_before = deepcopy(dual_vars)

    actions, before, after, gbefore, gafter, frozen = srp._update_admm_penalties(
        tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params, iter=1, allow_update=True)

    consensus_unchanged = (consensus_vars == consensus_before)
    dual_unchanged = (dual_vars == dual_before)
    rho_v_changed = (actions['v'] == 'increased' and after['v'] == before['v'] * 1.5)
    gamma_v_tracked = all(
        abs(g - 1.0 * r) < 1e-9
        for g, r in zip(_tso_gamma_snapshot(tso_model)['v'], _tso_rho_snapshot(tso_model)['v'])
    )

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['update_isolation'] = {
        'actions': actions, 'rho_before': before, 'rho_after': after,
        'gamma_before': gbefore, 'gamma_after': gafter, 'frozen': frozen,
        'consensus_vars_unchanged': consensus_unchanged, 'dual_vars_unchanged': dual_unchanged,
        'rho_v_changed_as_expected': rho_v_changed, 'gamma_v_tracked_rho': gamma_v_tracked,
        'pass': consensus_unchanged and dual_unchanged and rho_v_changed and gamma_v_tracked,
    }


def _zero_out_admm_requests(model):
    """Sets every otherwise-uninitialized `*_req`/`dual_*` Param on a
    freshly-built TSO model to 0.0, so `pe.value(model.admm_objective)` is
    well-defined at a zero-solve, build-default state. Zero solves; only
    mutates Pyomo Param VALUES (their declared defaults are `None`)."""
    for dn in model.active_distribution_networks:
        for p in model.periods:
            model.vmag_req[dn, p].set_value(0.0)
            model.dual_vmag_req[dn, p].set_value(0.0)
            model.p_pf_req[dn, p].set_value(0.0)
            model.q_pf_req[dn, p].set_value(0.0)
            model.dual_pf_p_req[dn, p].set_value(0.0)
            model.dual_pf_q_req[dn, p].set_value(0.0)
    for e in model.shared_energy_storages:
        for p in model.periods:
            model.p_ess_req[e, p].set_value(0.0)
            model.q_ess_req[e, p].set_value(0.0)
            model.dual_ess_p_req[e, p].set_value(0.0)
            model.dual_ess_q_req[e, p].set_value(0.0)


def _build_tso_model_only(module, eval_id, gamma_policy=None):
    """Builds only the TSO model (through `module`'s own
    `update_transmission_model_to_admm`), zero solves. Used to compare the
    CURRENT module (with an explicit 'fixed' override) against a
    historical module (pinned to a commit) on the SAME production
    sequence."""
    eval_dir = os.path.join(O.WORK_DIR, eval_id)
    if os.path.exists(eval_dir):
        raise RuntimeError(f'refusing to start: eval dir already exists (network logs append): {eval_dir}')
    planning = O.fresh_planning(eval_id)
    if gamma_policy is not None:
        planning.params.admm.proximal_regularization['tso']['gamma_policy'] = gamma_policy

    transmission_network = planning.transmission_network
    distribution_networks = planning.distribution_networks

    transmission_network.optimize = V1._stub_optimize.__get__(transmission_network, type(transmission_network))
    for _node_id, _dn in distribution_networks.items():
        _dn.optimize = V1._stub_optimize.__get__(_dn, type(_dn))

    consensus_vars, dual_vars = module.create_admm_variables(planning)
    candidate = planning.get_initial_candidate_solution()
    module._rebuild_candidate_total_capacities(planning, candidate)

    module.create_distribution_networks_models(
        distribution_networks, consensus_vars, candidate['total_capacity'], parallel_execution=False)
    tso_model, _tso_results = module.create_transmission_network_model(
        planning, consensus_vars, candidate['total_capacity'])

    module._prepare_transmission_objectives_for_admm(transmission_network, tso_model)
    objective_scale = 1.0
    module.update_transmission_model_to_admm(planning, tso_model, planning.params.admm, objective_scale)

    return planning, tso_model


def check_g_objective_structure(results):
    """Check (g): (g1) the diff of `update_transmission_model_to_admm`
    against commit d06f3636 is confined to gamma-Param mutability/policy
    lines; (g2) with gamma=1 (case-file default under EITHER policy) the
    built objective evaluates to the SAME number under the current code
    ('fixed' override) and under commit d06f3636."""
    proc = subprocess.run(
        ['git', 'show', f'{PRE_CHANGE_COMMIT}:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    pre_module, _ = V2._load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_d06f3636_g')
    new_source = inspect.getsource(srp.update_transmission_model_to_admm)
    pre_source = inspect.getsource(pre_module.update_transmission_model_to_admm)

    import difflib
    diff_lines = list(difflib.unified_diff(
        pre_source.splitlines(), new_source.splitlines(), lineterm=''))
    changed_lines = [
        line for line in diff_lines
        if (line.startswith('+') or line.startswith('-'))
        and not line.startswith('+++') and not line.startswith('---')
    ]
    allowed_keywords = ('gamma', 'tau', 'policy', 'mutable', '#', 'p5.15', 'spec v3', 'addendum')
    # Bare control-flow lines (e.g. the `else:` introducing the 'fixed'
    # branch) carry no gamma-related keyword themselves but are structurally
    # part of the gamma if/else this task adds; allow them explicitly rather
    # than widening the keyword list to something that could hide an
    # unrelated change.
    allowed_bare_lines = ('else:',)
    offending_lines = [
        line for line in changed_lines
        if not any(kw in line.lower() for kw in allowed_keywords)
        and line[1:].strip() not in allowed_bare_lines
    ]
    structural_confined = (len(offending_lines) == 0)

    # -- g2: numeric objective-value equality, gamma=1 both sides ----------
    planning_new, tso_new = _build_tso_model_only(
        srp, 'p515s33e2_check_g_current_fixed', gamma_policy='fixed')
    planning_pre, tso_pre = _build_tso_model_only(
        pre_module, 'p515s33e2_check_g_pre_change', gamma_policy=None)

    year = next(iter(planning_new.years))
    day = next(iter(planning_new.days))
    m_new = tso_new[year][day]
    m_pre = tso_pre[year][day]
    _zero_out_admm_requests(m_new)
    _zero_out_admm_requests(m_pre)

    gamma_new = pe.value(m_new.prox_gamma_v)
    gamma_pre = pe.value(m_pre.prox_gamma_v)
    obj_new = pe.value(m_new.admm_objective)
    obj_pre = pe.value(m_pre.admm_objective)
    value_matches = (abs(obj_new - obj_pre) < 1e-6) and abs(gamma_new - 1.0) < 1e-12 and abs(gamma_pre - 1.0) < 1e-12

    results['objective_structure'] = {
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'diff_hunk_count': len([l for l in diff_lines if l.startswith('@@')]),
        'n_changed_lines': len(changed_lines),
        'offending_lines': offending_lines,
        'structural_diff_confined_to_gamma': structural_confined,
        'gamma_new_fixed_override': gamma_new, 'gamma_pre_change': gamma_pre,
        'objective_value_current_fixed': obj_new, 'objective_value_pre_change': obj_pre,
        'objective_values_match': value_matches,
        'pass': structural_confined and value_matches,
    }


def check_h_other_case_params_load(results):
    per_case = {}
    for name, path in CS_PARAMS_PATHS.items():
        admm = ADMMParameters()
        with open(path) as handle:
            admm.read_parameters_from_file(json.load(handle)['admm'])
        per_case[name] = {
            'path': os.path.relpath(path, REPO),
            'gamma_policy': admm.proximal_regularization['tso']['gamma_policy'],
            'tau': admm.proximal_regularization['tso']['tau'],
            'freeze_after_cycle': admm.penalty_update['freeze_after_cycle'],
        }
    all_defaults = all(
        v['gamma_policy'] == 'fixed' and v['tau'] == 1.0 and v['freeze_after_cycle'] is None
        for v in per_case.values()
    )

    srp1_admm = ADMMParameters()
    with open(SRP1_PARAMS_PATH) as handle:
        srp1_admm.read_parameters_from_file(json.load(handle)['admm'])
    srp1_e2 = {
        'gamma_policy': srp1_admm.proximal_regularization['tso']['gamma_policy'],
        'tau': srp1_admm.proximal_regularization['tso']['tau'],
        'freeze_after_cycle': srp1_admm.penalty_update['freeze_after_cycle'],
        'minimum_consecutive_converged_cycles': srp1_admm.minimum_consecutive_converged_cycles,
    }
    srp1_ok = (
        srp1_e2['gamma_policy'] == 'tied_to_rho' and srp1_e2['tau'] == 1.0
        and srp1_e2['freeze_after_cycle'] == 30 and srp1_e2['minimum_consecutive_converged_cycles'] == 3
    )

    results['other_case_params_load'] = {
        'other_cases': per_case, 'other_cases_all_defaults': all_defaults,
        'srp1': srp1_e2, 'srp1_matches_spec_v3': srp1_ok,
        'pass': all_defaults and srp1_ok,
    }


def check_i_fixtures_unpickle(results):
    fixture_results = []
    for path in list(V1.S31C_FIXTURES) + S32_FROZEN_SMOPF_FIXTURES:
        entry = {'path': os.path.relpath(path, REPO)}
        try:
            with open(path, 'rb') as handle:
                obj = pickle.load(handle)
            entry['loads'] = True
            if path in S32_FROZEN_SMOPF_FIXTURES and isinstance(obj, dict) and 'model' in obj:
                model = obj['model']
                if hasattr(model, 'prox_gamma_v'):
                    entry['prox_gamma_v_mutable'] = bool(model.prox_gamma_v.mutable)
                    entry['prox_gamma_v_value'] = pe.value(model.prox_gamma_v)
        except Exception as exc:  # noqa: BLE001 -- report, don't hide
            entry['loads'] = False
            entry['error'] = f'{type(exc).__name__}: {exc}'
        fixture_results.append(entry)
    results['fixture_unpickling'] = {
        'fixtures': fixture_results,
        's32_frozen_smopf_fixtures_have_immutable_gamma': all(
            f.get('prox_gamma_v_mutable') is False
            for f in fixture_results if f['path'] in [os.path.relpath(p, REPO) for p in S32_FROZEN_SMOPF_FIXTURES]
        ),
        'pass': all(f['loads'] for f in fixture_results),
    }


def check_j_untouched_paths(results):
    proc = subprocess.run(
        ['git', 'show', 'HEAD:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    import tempfile
    import importlib.util
    tmp_dir = tempfile.mkdtemp(prefix='p515_s33e2_head_')
    tmp_path = os.path.join(tmp_dir, '_shared_resources_planning_head.py')
    with open(tmp_path, 'w') as handle:
        handle.write(proc.stdout)
    spec = importlib.util.spec_from_file_location('srp_head_e2', tmp_path)
    head_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head_module)

    fn_names = ('_run_operational_planning_hierarchical', '_run_operational_planning_without_coordination')
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        head_source = inspect.getsource(getattr(head_module, name))
        identical = (new_source == head_source)
        per_fn[name] = {'identical': identical}
        all_identical = all_identical and identical

    diff = subprocess.run(
        ['git', 'diff', 'HEAD', '--', 'shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    hunk_headers = [line for line in diff.splitlines() if line.startswith('@@')]

    results['hierarchical_uncoordinated_untouched'] = {
        'method': 'exact source-text equality (git HEAD vs current tree) for both path '
                  'functions, plus every diff hunk header in shared_resources_planning.py.',
        'functions': per_fn, 'diff_hunk_count': len(hunk_headers), 'diff_hunk_headers': hunk_headers,
        'pass': all_identical,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    observed_hash = _spec_v3_hash()
    if observed_hash != SPEC_V3_SHA256:
        raise RuntimeError(f'frozen s33e2 spec v3 hash mismatch: file={observed_hash} expected={SPEC_V3_SHA256}')

    guard = SolveProfileGuard(permitted=(), label='S33E2 zero-solve check').install()
    results = {}
    try:
        check_a_tied_gamma_tracks_rho(results)
        check_b_fixed_gamma(results)
        check_c_boyd_uses_model_gamma(results)
        check_d_freeze(results)
        check_e_stop_rule(results)
        check_f_update_isolation(results)
        check_g_objective_structure(results)
        check_h_other_case_params_load(results)
        check_i_fixtures_unpickle(results)
        check_j_untouched_paths(results)
    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Step 3.2 E2 (s33e2 worker task) -- zero-solve verification',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 14',
            'data/SRP1/Results/P515S33/frozen_s33_e2_spec_v3_825f1f02.json',
        ],
        'spec_file': os.path.relpath(SPEC_V3_PATH, REPO),
        'spec_file_sha256': SPEC_V3_SHA256,
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S33E2] wrote {OUT_PATH}')
    print(f'[S33E2] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
