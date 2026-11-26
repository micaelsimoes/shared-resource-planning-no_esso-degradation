"""
P5.15 Addendum 20 (s38, the PF-pace experiment) preparation worker task --
zero-solve verification of the production tau-validation relaxation and the
new harness-level PF per-entry capture / freeze-policy mechanisms.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 20; binding specification
data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json.

The ONE production change under test (`admm_parameters.py`,
`_read_parameters_from_file`): `proximal_regularization.tso.tau` is now
validated `tau < 0.0 -> ValueError` (was `tau <= 0.0`), so `tau == 0.0`
(arm A/C) loads. `tau < 0.0` still raises. No other production behaviour is
changed -- `tau` continues to only MULTIPLY (`gamma_c = tau * rho_c`, and
`(gamma_c / 2) * proximal_c ** 2` in the TSO objective); no code path
divides by tau or by any `prox_gamma_*` Param (verified by repo-wide grep,
see WORKER_REPORT_S38_PREP.md).

Verifies, with a BLOCKING `SolveProfileGuard` (permitted=(), i.e. zero
solves anywhere) armed for the whole script:

  (i)   `ADMMParameters.read_parameters_from_file` now loads tau=0.0 (was
        rejected) and still rejects tau=-0.1 (ValueError);
  (ii)  a FRESHLY BUILT (zero-solve) TSO ADMM model
        (`p515_g_g1_g4_admm_gates._s38_build_probe_tso_model`) under arm
        A's configuration (tau=0.0) has `prox_gamma_v/pf/ess` Params ==
        exactly 0.0 on every channel; under arm B's configuration (tau=1.0)
        the same Params == the model's own rho_v/pf/ess Params; the TSO
        proximal objective TERM (`(gamma_c / 2) * (z - z_prev) ** 2`),
        evaluated on the ACTUAL built model at arbitrary NON-ZERO Var/Param
        values (z != z_prev, both nonzero), is exactly 0.0 under A on
        every leg (v, pf_p, pf_q, ess_p, ess_q);
  (iii) `_update_admm_penalties`'s freeze-policy replay (reusing
        `p515_s37_zero_solve_checks._run_trajectory`/`_synthetic_boyd_
        sequence`/`_reset_gamma_tied` UNMODIFIED, on a FRESH `V1._build_
        admm_ready_state` model set, with `freeze_backstop_cycle`
        overridden to 200 -- the Addendum 20 value): no channel is frozen
        by the (removed, for this stage) cycle-60 backstop; EVERY
        non-exempt channel freezes at cycle 200 (reason='backstop'); a
        channel that never acts (dead-band the whole run) never freezes
        via the 'streak' reason, only via the cycle-200 backstop; a
        channel that acts once then holds freezes via 'streak' once its
        post-action unchanged-streak reaches 10; a channel exempt under
        arm A's own list (['ess', 'pf']) never changes rho/gamma across
        the whole synthetic run, even under a strongly imbalanced,
        'increased'-forcing ratio pattern;
  (iv)  the PF per-entry capture reconstruction identity
        (`p515_g_g1_g4_admm_gates.s38_pf_capture_hooks`), exercised for
        REAL (not deferred to the preflight) on a zero-solve `V1._build_
        admm_ready_state` model set with a MULTI-ENTRY synthetic PF
        perturbation (12 nonzero entries across 2 active nodes x 2 power
        types x 3 periods -- not a single-entry case), confirming the
        wrapper's reconstructed r/s reproduce production's own
        `result['pf']['r']`/`['s']` to relative 1e-9, and that the
        function correctly restores `srp.get_admm_boyd_residual_metrics`
        on exit.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s38_zero_solve_checks.py

Writes data/SRP1/Results/P515S38/zero_solve_checks/zero_solve_checks.json
(a NEW directory; refuses to overwrite).
"""

import hashlib
import json
import os
import random
import sys
import tempfile
from copy import deepcopy
from datetime import datetime, timezone

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s32_zero_solve_checks as V1  # noqa: E402
import p515_s37_zero_solve_checks as S37Z  # noqa: E402 -- reused: _run_trajectory,
                                            # _synthetic_boyd_sequence, _reset_gamma_tied
import p515_g_g1_g4_admm_gates as G  # noqa: E402 -- _s38_build_probe_tso_model, s38_pf_capture_hooks

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38', 'zero_solve_checks')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')

SPEC_V9_PATH = G.S38_SPEC_PATH
SRP1_PARAMS_PATH = os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')

INCREASE_RATIOS = (30.0, 1.0, 1.0)   # primal >> dual_balance -> 'increased'
DECREASE_RATIOS = (1.0, 30.0, 30.0)  # dual_balance >> primal -> 'decreased'
HOLD_RATIOS = (1.0, 1.0, 1.0)        # dead band -> 'held'


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _spec_v9_hash():
    with open(SPEC_V9_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


# ===========================================================================
# (i) tau validation
# ===========================================================================
def check_i_tau_validation(results):
    with open(SRP1_PARAMS_PATH) as handle:
        base = json.load(handle)['admm']

    def _try(tau):
        admm = ADMMParameters()
        data = deepcopy(base)
        data['proximal_regularization'] = dict(data.get('proximal_regularization', {}))
        data['proximal_regularization']['tso'] = dict(data['proximal_regularization'].get('tso', {}))
        data['proximal_regularization']['tso']['tau'] = tau
        try:
            admm.read_parameters_from_file(data)
            return True, None, admm
        except ValueError as exc:
            return False, str(exc), None

    tau0_ok, tau0_err, tau0_admm = _try(0.0)
    tau_neg_ok, tau_neg_err, _ = _try(-0.1)
    tau_pos_ok, tau_pos_err, tau_pos_admm = _try(0.25)

    results['tau_validation'] = {
        'tau_0_loads': tau0_ok, 'tau_0_error': tau0_err,
        'tau_0_value_stored': (tau0_admm.proximal_regularization['tso']['tau'] if tau0_admm else None),
        'tau_negative_rejected': (not tau_neg_ok), 'tau_negative_error': tau_neg_err,
        'tau_0p25_loads': tau_pos_ok,
        'tau_0p25_value_stored': (tau_pos_admm.proximal_regularization['tso']['tau'] if tau_pos_admm else None),
        'pass': (
            tau0_ok and tau0_admm.proximal_regularization['tso']['tau'] == 0.0
            and (not tau_neg_ok) and tau_neg_err is not None and 'non-negative' in tau_neg_err
            and tau_pos_ok and tau_pos_admm.proximal_regularization['tso']['tau'] == 0.25
        ),
    }


# ===========================================================================
# (ii) built-TSO-model gamma Params + proximal objective term
# ===========================================================================
def check_ii_tso_gamma_and_proximal_term(results):
    planning_a, tso_a = G._s38_build_probe_tso_model('p515s38zsc_probe_a', 0.0)
    planning_b, tso_b = G._s38_build_probe_tso_model('p515s38zsc_probe_b', 1.0)

    year0_a = next(iter(planning_a.transmission_network.years))
    day0_a = next(iter(planning_a.transmission_network.days))
    block_a = tso_a[year0_a][day0_a]
    year0_b = next(iter(planning_b.transmission_network.years))
    day0_b = next(iter(planning_b.transmission_network.days))
    block_b = tso_b[year0_b][day0_b]

    gamma_a = {'v': pe.value(block_a.prox_gamma_v), 'pf': pe.value(block_a.prox_gamma_pf),
               'ess': pe.value(block_a.prox_gamma_ess)}
    gamma_b = {'v': pe.value(block_b.prox_gamma_v), 'pf': pe.value(block_b.prox_gamma_pf),
               'ess': pe.value(block_b.prox_gamma_ess)}
    rho_b = {'v': pe.value(block_b.rho_v), 'pf': pe.value(block_b.rho_pf), 'ess': pe.value(block_b.rho_ess)}

    gamma_a_all_zero = all(v == 0.0 for v in gamma_a.values())
    gamma_b_equals_rho = all(gamma_b[g] == rho_b[g] for g in ('v', 'pf', 'ess'))

    # -- proximal TERM evaluated at ARBITRARY NONZERO z vs z_prev, arm A ----
    dn = next(iter(block_a.active_distribution_networks))
    p0 = next(iter(block_a.periods))
    e0 = next(iter(block_a.shared_energy_storages))

    block_a.expected_interface_vmag[dn, p0].set_value(7.3)
    block_a.prox_v_prev[dn, p0].set_value(5.1)
    term_v = pe.value((block_a.prox_gamma_v / 2) * (block_a.expected_interface_vmag[dn, p0]
                                                     - block_a.prox_v_prev[dn, p0]) ** 2)

    block_a.expected_interface_pf_p[dn, p0].set_value(0.42)
    block_a.prox_pf_p_prev[dn, p0].set_value(-0.17)
    term_pf_p = pe.value((block_a.prox_gamma_pf / 2) * (block_a.expected_interface_pf_p[dn, p0]
                                                         - block_a.prox_pf_p_prev[dn, p0]) ** 2)

    block_a.expected_interface_pf_q[dn, p0].set_value(0.09)
    block_a.prox_pf_q_prev[dn, p0].set_value(0.55)
    term_pf_q = pe.value((block_a.prox_gamma_pf / 2) * (block_a.expected_interface_pf_q[dn, p0]
                                                         - block_a.prox_pf_q_prev[dn, p0]) ** 2)

    block_a.expected_shared_ess_p[e0, p0].set_value(1.23)
    block_a.prox_ess_p_prev[e0, p0].set_value(-0.87)
    term_ess_p = pe.value((block_a.prox_gamma_ess / 2) * (block_a.expected_shared_ess_p[e0, p0]
                                                           - block_a.prox_ess_p_prev[e0, p0]) ** 2)

    block_a.expected_shared_ess_q[e0, p0].set_value(-0.31)
    block_a.prox_ess_q_prev[e0, p0].set_value(0.64)
    term_ess_q = pe.value((block_a.prox_gamma_ess / 2) * (block_a.expected_shared_ess_q[e0, p0]
                                                           - block_a.prox_ess_q_prev[e0, p0]) ** 2)

    terms = {'v': term_v, 'pf_p': term_pf_p, 'pf_q': term_pf_q, 'ess_p': term_ess_p, 'ess_q': term_ess_q}
    all_terms_exactly_zero = all(v == 0.0 for v in terms.values())

    results['tso_gamma_and_proximal_term'] = {
        'note_on_pf_term_computation': (
            'the pf_p/pf_q terms are computed WITHOUT the production interface_transf_rating '
            'scaling factor (a positive constant not stored as a model attribute) -- dividing by '
            'a positive constant before squaring does not change whether a zero-gamma term '
            'evaluates to exactly zero, so this is a faithful, if simplified, reproduction'),
        'gamma_arm_a_tau_0': gamma_a, 'gamma_arm_a_all_exactly_zero': gamma_a_all_zero,
        'gamma_arm_b_tau_1': gamma_b, 'rho_arm_b': rho_b, 'gamma_arm_b_equals_rho': gamma_b_equals_rho,
        'proximal_terms_arm_a_at_arbitrary_nonzero_z_vs_z_prev': terms,
        'all_terms_exactly_zero': all_terms_exactly_zero,
        'pass': bool(gamma_a_all_zero and gamma_b_equals_rho and all_terms_exactly_zero),
    }


# ===========================================================================
# (iii) freeze-policy replay
# ===========================================================================
def _pattern(n, sequence_ratios):
    return [dict(sequence_ratios) for _ in range(n)]


def check_iii_freeze_policy_replay(results):
    planning, tso_model, dso_models, esso_model, _cv, _dv = (
        V1._build_admm_ready_state('p515s38zsc_check_iii'))
    admm_params = planning.params.admm
    admm_params.penalty_update['freeze_backstop_cycle'] = 200
    admm_params.penalty_update['freeze_after_unchanged_cycles'] = 10

    # ---- sub-check 1: no freeze at cycle 60, every channel always acting --
    admm_params.penalty_update['balancing_exempt_channels'] = []
    seq_always_act = []
    for c in range(65):
        pattern = {'v': INCREASE_RATIOS if c % 2 == 0 else DECREASE_RATIOS,
                   'pf': INCREASE_RATIOS if c % 2 == 0 else DECREASE_RATIOS,
                   'ess': INCREASE_RATIOS if c % 2 == 0 else DECREASE_RATIOS}
        seq_always_act.append(pattern)
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj_no60, _fs = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq_always_act)
    row60 = next(r for r in traj_no60 if r['cycle'] == 60)
    no_freeze_at_60 = all(not row60['freeze_state'][g]['frozen'] for g in ('v', 'pf', 'ess'))
    no_channel_reason_backstop_at_60 = all(row60['freeze_state'][g]['reason'] != 'backstop' for g in ('v', 'pf', 'ess'))

    # ---- sub-check 2: every non-exempt channel freezes at cycle 200 via
    #      backstop -- keep acting (increase/decrease alternation, never a
    #      hold) all the way to cycle 199 so ONLY the backstop can trigger
    #      the freeze (a streak requires >=10 CONSECUTIVE non-acting cycles,
    #      which this pattern never produces) --------------------------------
    admm_params.penalty_update['balancing_exempt_channels'] = []
    seq_to_200 = []
    for c in range(200):
        pattern = {'v': INCREASE_RATIOS if c % 2 == 0 else DECREASE_RATIOS,
                   'pf': INCREASE_RATIOS if c % 2 == 0 else DECREASE_RATIOS,
                   'ess': INCREASE_RATIOS if c % 2 == 0 else DECREASE_RATIOS}
        seq_to_200.append(pattern)
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj_200, _fs2 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq_to_200)
    row200 = traj_200[-1]
    assert row200['cycle'] == 200
    all_frozen_at_200 = all(row200['freeze_state'][g]['frozen'] for g in ('v', 'pf', 'ess'))
    all_reason_backstop_at_200 = all(row200['freeze_state'][g]['reason'] == 'backstop' for g in ('v', 'pf', 'ess'))
    all_action_backstop_label_at_200 = all(
        row200['actions'][g] == 'held (frozen backstop cycle 200)' for g in ('v', 'pf', 'ess'))
    not_frozen_before_200 = all(
        not traj_200[i]['freeze_state'][g]['frozen'] for i in range(199) for g in ('v', 'pf', 'ess'))

    # ---- sub-check 3a: a channel that NEVER acts (dead band the whole run)
    #      never freezes via 'streak', only (eventually) via backstop -------
    admm_params.penalty_update['balancing_exempt_channels'] = []
    seq_never_acts_v = [{'v': HOLD_RATIOS, 'pf': HOLD_RATIOS, 'ess': HOLD_RATIOS} for _ in range(210)]
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj_never_acts, _fs3 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq_never_acts_v)
    v_never_ever_acted = all(not r['freeze_state']['v']['ever_acted'] for r in traj_never_acts)
    v_never_frozen_via_streak = all(r['freeze_state']['v']['reason'] != 'streak' for r in traj_never_acts)
    v_frozen_via_backstop_at_200 = (traj_never_acts[199]['freeze_state']['v']['frozen']
                                    and traj_never_acts[199]['freeze_state']['v']['reason'] == 'backstop')

    # ---- sub-check 3b: a channel that acts ONCE then holds freezes via
    #      'streak' once its post-action unchanged-streak reaches 10 -------
    seq_act_once_then_hold = [{'v': INCREASE_RATIOS, 'pf': HOLD_RATIOS, 'ess': HOLD_RATIOS}]
    seq_act_once_then_hold += [{'v': HOLD_RATIOS, 'pf': HOLD_RATIOS, 'ess': HOLD_RATIOS} for _ in range(30)]
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj_streak, _fs4 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq_act_once_then_hold)
    streak_freeze_cycle = next(
        (r['cycle'] for r in traj_streak if r['freeze_state']['v']['reason'] == 'streak'), None)
    # cycle 1: 'increased' (unchanged_streak reset to 0, ever_acted=True); cycles 2-11: held
    # (10 held cycles -> streak counts 1..10 at the END of cycle 11's own call). The
    # streak >= freeze_after_unchanged_cycles check runs INSIDE the SAME
    # `_update_admm_penalties` call that just incremented the streak (production
    # code, `shared_resources_planning.py` `_update_admm_penalties`, the
    # `if not channel_frozen_this_cycle:` block runs every cycle, unconditionally,
    # right after that cycle's action is decided) -- so `frozen` is already True in
    # the row RECORDED for cycle 11 itself, not cycle 12 (the docstring's "takes
    # effect starting the NEXT cycle" describes when the ALREADY-frozen state next
    # SUPPRESSES the ordinary increase/decrease/hold branch -- cycle 12 onward --
    # not when the freeze_state dict itself flips to frozen=True, which happens at
    # cycle 11).
    streak_freeze_at_expected_cycle = (streak_freeze_cycle == 11)
    streak_freeze_before_backstop = (streak_freeze_cycle is not None and streak_freeze_cycle < 200)

    # ---- sub-check 4: PF exempt under arm A's own list (['ess', 'pf'])
    #      never changes rho/gamma, even under a strongly imbalanced,
    #      'increased'-forcing ratio pattern -------------------------------
    admm_params.penalty_update['balancing_exempt_channels'] = ['ess', 'pf']
    strongly_imbalanced_pf = (1.0e6, 1.0e-6, 1.0e-6)
    seq_pf_exempt = []
    for c in range(210):
        seq_pf_exempt.append({'v': HOLD_RATIOS, 'pf': strongly_imbalanced_pf, 'ess': strongly_imbalanced_pf})
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj_pf_exempt, _fs5 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq_pf_exempt)
    pf_rho_constant = all(
        r['rho_before']['pf'] == traj_pf_exempt[0]['rho_before']['pf']
        and r['rho_after']['pf'] == traj_pf_exempt[0]['rho_before']['pf'] for r in traj_pf_exempt)
    pf_gamma_constant = all(
        r['gamma_before']['pf'] == traj_pf_exempt[0]['gamma_before']['pf']
        and r['gamma_after']['pf'] == traj_pf_exempt[0]['gamma_before']['pf'] for r in traj_pf_exempt)
    pf_action_always_exempt = all(r['actions']['pf'] == 'exempt (fixed)' for r in traj_pf_exempt)
    pf_never_at_clamp = all(r['freeze_state']['pf']['at_clamp'] is False for r in traj_pf_exempt)
    pf_survives_backstop_at_200 = (traj_pf_exempt[199]['actions']['pf'] == 'exempt (fixed)')

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)

    results['freeze_policy_replay'] = {
        'no_freeze_at_cycle_60': {
            'no_channel_frozen_at_60': no_freeze_at_60,
            'no_channel_reason_backstop_at_60': no_channel_reason_backstop_at_60,
            'row_60_freeze_state': row60['freeze_state'],
            'pass': bool(no_freeze_at_60 and no_channel_reason_backstop_at_60),
        },
        'freeze_at_cycle_200_every_non_exempt_channel': {
            'all_frozen_at_200': all_frozen_at_200,
            'all_reason_backstop_at_200': all_reason_backstop_at_200,
            'all_action_label_correct_at_200': all_action_backstop_label_at_200,
            'not_frozen_before_200': not_frozen_before_200,
            'pass': bool(all_frozen_at_200 and all_reason_backstop_at_200
                        and all_action_backstop_label_at_200 and not_frozen_before_200),
        },
        'streak_freeze_only_after_prior_action': {
            'never_acts_case': {
                'v_never_ever_acted': v_never_ever_acted,
                'v_never_frozen_via_streak': v_never_frozen_via_streak,
                'v_frozen_via_backstop_at_200': v_frozen_via_backstop_at_200,
            },
            'act_once_then_hold_case': {
                'streak_freeze_cycle': streak_freeze_cycle,
                'streak_freeze_at_expected_cycle_11': streak_freeze_at_expected_cycle,
                'streak_freeze_before_backstop': streak_freeze_before_backstop,
            },
            'pass': bool(v_never_ever_acted and v_never_frozen_via_streak and v_frozen_via_backstop_at_200
                        and streak_freeze_at_expected_cycle and streak_freeze_before_backstop),
        },
        'pf_exempt_in_a_never_changes': {
            'pf_rho_constant': pf_rho_constant, 'pf_gamma_constant': pf_gamma_constant,
            'pf_action_always_exempt': pf_action_always_exempt, 'pf_never_at_clamp': pf_never_at_clamp,
            'pf_survives_backstop_at_cycle_200': pf_survives_backstop_at_200,
            'ratios_used': strongly_imbalanced_pf,
            'pass': bool(pf_rho_constant and pf_gamma_constant and pf_action_always_exempt
                        and pf_never_at_clamp and pf_survives_backstop_at_200),
        },
    }
    results['freeze_policy_replay']['pass'] = all(
        v['pass'] for k, v in results['freeze_policy_replay'].items() if isinstance(v, dict) and 'pass' in v)


# ===========================================================================
# (iv) PF capture reconstruction identity -- exercised for real, zero solves
# ===========================================================================
def check_iv_pf_capture_identity(results):
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s38zsc_check_iv'))

    random.seed(20260917)
    active_nodes = list(planning.active_distribution_network_nodes)
    n_perturbed = 0
    for node_id in active_nodes:
        for year in planning.years:
            for day in planning.days:
                for power_type in ('p', 'q'):
                    for p in range(min(3, planning.num_instants)):
                        consensus_vars['pf']['tso']['current'][node_id][year][day][power_type][p] = random.uniform(-5, 5)
                        consensus_vars['pf']['tso']['prev'][node_id][year][day][power_type][p] = random.uniform(-5, 5)
                        consensus_vars['pf']['dso']['current'][node_id][year][day][power_type][p] = random.uniform(-5, 5)
                        dual_vars['pf']['dso']['current'][node_id][year][day][power_type][p] = random.uniform(-2, 2)
                        n_perturbed += 1

    # A genuine system temp dir (not under the committed OUT_DIR): the
    # sidecar paths below are a required argument of `s38_pf_capture_hooks`
    # (it writes a real file), but their CONTENT is not evidence this check
    # commits -- only the identity numbers extracted from `pf_stride_path`
    # below are recorded in the written `zero_solve_checks.json`.
    tmp_dir = tempfile.mkdtemp(prefix='p515_s38_zsc_check_iv_')
    recourse_jump_path = os.path.join(tmp_dir, 'recourse_jump.jsonl')
    ess_stride_path = os.path.join(tmp_dir, 'ess_stride.jsonl')
    floor_sidecar_path = os.path.join(tmp_dir, 'floor.jsonl')
    pf_stride_path = os.path.join(tmp_dir, 'pf_stride.jsonl')
    for path in (recourse_jump_path, ess_stride_path, floor_sidecar_path, pf_stride_path):
        if os.path.exists(path):
            os.remove(path)

    real_fn_before = srp.get_admm_boyd_residual_metrics
    with G.s38_pf_capture_hooks(recourse_jump_path, ess_stride_path, floor_sidecar_path,
                                pf_stride_path, {}, stride=1):
        result = srp.get_admm_boyd_residual_metrics(
            planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars, planning.params.admm)
    real_fn_restored = (srp.get_admm_boyd_residual_metrics is real_fn_before)

    with open(pf_stride_path) as handle:
        line = json.loads(handle.readline())

    rel_err_r, rel_err_s = line['rel_err_r'], line['rel_err_s']
    identity_holds = line['identity_holds']

    # `n_entries_captured` is the FULL PF entry count the wrapper iterates
    # (every node x year x day x power_type x `num_instants` period, per
    # production's own loop) -- NOT limited to the `n_perturbed` subset this
    # check randomized (only 3 periods per node/year/day/power_type, to keep
    # the perturbation loop cheap); every non-perturbed entry contributes its
    # BUILD-DEFAULT (current==prev, dual==0) values, which is why the
    # captured count is larger than the perturbed count and is NOT itself a
    # pass/fail criterion here -- only the r/s IDENTITY is.
    results['pf_capture_identity'] = {
        'n_entries_perturbed_inputs': n_perturbed, 'n_entries_captured': len(line['entries']),
        'production_boyd_pf_r': result['pf']['r'], 'production_boyd_pf_s': result['pf']['s'],
        'reconstructed_r': line['reconstructed_r'], 'reconstructed_s': line['reconstructed_s'],
        'rel_err_r': rel_err_r, 'rel_err_s': rel_err_s, 'identity_holds': identity_holds,
        'wrapped_function_restored_after_exit': real_fn_restored,
        'pass': bool(identity_holds and rel_err_r <= 1e-9 and rel_err_s <= 1e-9 and real_fn_restored
                    and len(line['entries']) >= n_perturbed),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    observed_hash = _spec_v9_hash()
    print(f'[S38 zero-solve checks] spec v9 sha256={observed_hash}')

    guard = SolveProfileGuard(permitted=(), label='S38 zero-solve check').install()
    results = {}
    try:
        check_i_tau_validation(results)
        check_ii_tso_gamma_and_proximal_term(results)
        check_iii_freeze_policy_replay(results)
        check_iv_pf_capture_identity(results)
    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Addendum 20 (s38 preparation worker task) -- zero-solve verification '
                 'of the tau relaxation, the PF per-entry capture hook and the Addendum 20 '
                 'freeze policy (10-unchanged + absolute freeze at 200)',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 20',
            'data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json',
        ],
        'spec_file': os.path.relpath(SPEC_V9_PATH, REPO),
        'spec_file_sha256_observed': observed_hash,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S38] wrote {OUT_PATH}')
    print(f'[S38] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
