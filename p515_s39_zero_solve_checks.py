"""
P5.15 Addendum 21 (s39 preparation worker task) -- zero-solve verification of
the new production conditional balancing exemption
(`admm.penalty_update.balancing_exempt_until`) and the s39 harness's
structural working-dir-id fix.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 21; binding specification
data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json.

The production change under test (`admm_parameters.py`, `_read_parameters_
from_file`; `shared_resources_planning.py`, `_init_admm_freeze_state` and
`_update_admm_penalties` -- see that function's own docstring, "Addendum 21
addition", for the full rule): an optional per-channel CONDITIONAL balancing
exemption, `admm.penalty_update.balancing_exempt_until` (default {}, e.g.
`{'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}`). A channel
listed here is held `'exempt (fixed)'` until its own Boyd `dual_ratio` (full
s/eps_dual) has been strictly below the configured threshold for the
configured number of CONSECUTIVE calls, at which point it lifts ONE-WAY (in
that SAME call, falling through to the standard balancing rule) and is
thereafter subject to the standard unchanged-streak freeze and the absolute
backstop, exactly like any other non-exempt channel. A channel may not be in
BOTH `balancing_exempt_channels` and `balancing_exempt_until`.

Verifies, with a BLOCKING `SolveProfileGuard` (permitted=(), i.e. zero
solves anywhere) armed for the whole script:

  (i)   DEFAULT-OFF IDENTITY: re-runs the balancing-replay validation
        methodology of `p515_s38_balancing_replay.py` (same technique --
        lightweight stand-in Pyomo models, the REAL `shared_resources_
        planning._update_admm_penalties`/`_init_admm_freeze_state`, per-cycle
        re-seeding from each trajectory's OWN recorded `rho_*_before`/
        `gamma_*_before` ground truth) against FIVE already-completed
        trajectories -- run 1 (s35ref), both s37 arms (rho_ess 0.01 /
        0.001) and, NEWLY, both s38 arms (A: tau=0, PF+ESS exempt; B:
        tau=1, ESS exempt) -- none of which ever set `balancing_exempt_
        until` (absent key, default {}), confirming 0 mismatches between
        the CURRENT (post-Addendum-21) `_update_admm_penalties` and every
        run's own recorded action/rho/gamma/freeze-state trajectory. A
        per-run config (exempt channels, freeze backstop/unchanged-cycles,
        tau) is INFERRED from that trajectory's own first recorded cycle
        (`balancing_exempt_{v,pf,ess}`, `freeze_backstop_cycle`, `freeze_
        after_unchanged_cycles`, `gamma_tau`), not hand-declared, so this
        check cannot silently validate the wrong configuration;
  (ii)  SYNTHETIC SEQUENCES on a fresh zero-solve model set (`p515_s32_
        zero_solve_checks._build_admm_ready_state`), replayed through
        `p515_s37_zero_solve_checks._run_trajectory` (REUSED unmodified):
        a 4-then-break-then-5 sequence lifts at exactly the cycle the 5th
        CONSECUTIVE below-threshold call completes (not the 9th); the lift
        is ONE-WAY (forcing the ratio back above threshold after the lift
        never re-exempts); the standard rule actually BALANCES the channel
        in the lift cycle's own update when the ratio test says so (a
        forced 'increased' at the lift cycle, not merely 'held'); the
        unchanged-streak freeze (after 10) fires only counting from a
        POST-LIFT action (never from cycles spent pending); the absolute
        backstop (cycle 200) freezes a channel that lifted early and then
        holds, but does NOT interrupt a channel still pending (never
        lifted) past cycle 200; `admm_parameters.ADMMParameters.read_
        parameters_from_file` raises `ValueError` for every documented bad
        configuration (unknown channel, non-positive threshold, non-
        positive consecutive_cycles, and a channel listed in BOTH
        `balancing_exempt_channels` and `balancing_exempt_until`);
  (iii) `proximal_regularization.tso.tau = 0.0` and `minimum_consecutive_
        converged_cycles = 10` load via `read_parameters_from_file` and are
        BOTH in force on a freshly built (UNSOLVED) planning object's
        `params.admm`, mirroring exactly what `_s39_configure_hook` applies
        to the real arm's planning object;
  (iv)  the s39 harness's STRUCTURAL working-dir-id fix (`p515_g_g1_g4_
        admm_gates._s39_working_dir_ids`): for EACH arm (C, D), the set of
        three ids used in 'real' mode is disjoint from the set used in
        'preflight' mode, and, across both arms and both modes, all twelve
        derived ids are pairwise distinct (no accidental collision anywhere
        in the id space this stage introduces).

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s39_zero_solve_checks.py

Writes data/SRP1/Results/P515S39/zero_solve_checks/zero_solve_checks.json
(a NEW directory; refuses to overwrite).
"""

import hashlib
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone
from math import isclose

import pyomo.environ as pe

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s32_zero_solve_checks as V1  # noqa: E402
import p515_s37_zero_solve_checks as S37Z  # noqa: E402 -- reused: _run_trajectory
import p515_s38_balancing_replay as B38  # noqa: E402 -- reused: model builders, boyd/
                                          # residual metric adapters (NOT its own
                                          # `run_validation`/`build_params`, which hard-code
                                          # backstop=60 and never set tau -- see check (i))
import p515_g_g1_g4_admm_gates as G  # noqa: E402 -- _s39_working_dir_ids

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S39', 'zero_solve_checks')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')

SPEC_V10_PATH = G.S39_SPEC_PATH
CASE_FILE = os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')

CHANNELS = ('v', 'pf', 'ess')

# -- check (i): the five already-completed trajectories, config INFERRED
#    from each trajectory's own first cycle (see docstring) --------------
CHECK_I_TRAJECTORIES = {
    'run1_s35ref': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S35_REF_run', 'g_baseline.json'),
    's37_rho0p01': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S37_RHO0P01_run', 'g_s37_rho0p01.json'),
    's37_rho0p001': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S37_RHO0P001_run', 'g_s37_rho0p001.json'),
    's38_A_tau0': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38_A_TAU0_run', 'g_s38_A_tau0.json'),
    's38_B_pfbal': os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S38_B_PFBAL_run', 'g_s38_B_pfbal.json'),
}


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _spec_v10_hash():
    with open(SPEC_V10_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load_case_admm_block():
    with open(CASE_FILE) as handle:
        return json.load(handle)['admm']


def _rho_close(a, b):
    return isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


# ===========================================================================
# (i) default-off identity -- extends p515_s38_balancing_replay's OWN
#     methodology (reused model builders / metric adapters) to 5 runs, with
#     backstop/freeze-unchanged/tau/exempt-channels INFERRED per run from
#     the trajectory itself (never hand-declared, never mismatched).
# ===========================================================================
def _infer_run_config(traj):
    """Config inferred from the trajectory's own first cycle. `balancing_
    exempt_{v,pf,ess}` predates Addendum 19 (run 1 / s35ref has neither
    field -- `.get(..., False)` correctly infers no exempt channels, its
    own actual configuration, not a guess)."""
    first = traj[0]
    return {
        'balancing_exempt_channels': [g for g in CHANNELS if first.get(f'balancing_exempt_{g}', False)],
        'freeze_backstop_cycle': first['freeze_backstop_cycle'],
        'freeze_after_unchanged_cycles': first['freeze_after_unchanged_cycles'],
        'tau': first['gamma_tau'],
    }


def _build_params_for_validation(admm_block_template, inferred_cfg):
    """Real `ADMMParameters`, loaded via the real `read_parameters_from_
    file`, from a deep copy of the case file's `admm` block with exactly
    the fields `_infer_run_config` read off the trajectory varied --
    `balancing_exempt_until` is DELIBERATELY left at its case-file default
    ({}, absent) for every one of these five runs (none of them ever used
    it), which is exactly what this check calls "default-off identity"."""
    admm_block = deepcopy(admm_block_template)
    admm_block['penalty_update'] = deepcopy(admm_block['penalty_update'])
    admm_block['penalty_update']['balancing_exempt_channels'] = list(inferred_cfg['balancing_exempt_channels'])
    admm_block['penalty_update']['freeze_backstop_cycle'] = inferred_cfg['freeze_backstop_cycle']
    admm_block['penalty_update']['freeze_after_unchanged_cycles'] = inferred_cfg['freeze_after_unchanged_cycles']
    admm_block['proximal_regularization'] = deepcopy(admm_block.get('proximal_regularization', {}))
    admm_block['proximal_regularization']['tso'] = dict(admm_block['proximal_regularization'].get('tso', {}))
    admm_block['proximal_regularization']['tso']['tau'] = inferred_cfg['tau']
    params = ADMMParameters()
    params.read_parameters_from_file(admm_block)
    return params


def _run_validation(run_name, traj_path, admm_block_template):
    with open(traj_path) as handle:
        traj = json.load(handle)['cycle_trajectory']

    inferred_cfg = _infer_run_config(traj)
    params = _build_params_for_validation(admm_block_template, inferred_cfg)
    # `balancing_exempt_until` must be the empty-default -- otherwise this
    # is not a "default-off" identity check at all.
    assert params.penalty_update['balancing_exempt_until'] == {}, (
        f'{run_name}: expected default-off balancing_exempt_until, got '
        f"{params.penalty_update['balancing_exempt_until']}")

    freeze_state = srp._init_admm_freeze_state()
    tso = B38.make_tso_model(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dso = B38.make_dso_model(0.0, 0.0, 0.0)
    esso = B38.make_esso_model(0.0)
    tso_model = {0: {0: tso}}
    dso_models = {0: {0: {0: dso}}}
    esso_model = {0: esso}

    mismatches = []
    for cyc in traj:
        cycle = cyc['cycle']
        B38.set_tso_rho(tso, cyc['rho_v_before'], cyc['rho_pf_before'], cyc['rho_ess_before'])
        B38.set_tso_gamma(tso, cyc['gamma_v_before'], cyc['gamma_pf_before'], cyc['gamma_ess_before'])
        B38.set_dso_rho(dso, cyc['rho_v_before'], cyc['rho_pf_before'], cyc['rho_ess_before'])
        B38.set_esso_rho(esso, cyc['rho_ess_before'])

        boyd_metrics = B38.boyd_metrics_from_cycle(cyc)
        residual_metrics = B38.residual_metrics_from_cycle(cyc)

        actions, before, after, before_gamma, after_gamma, rho_freeze_active, freeze_state = (
            srp._update_admm_penalties(
                tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, params,
                iter=cycle, allow_update=bool(cyc['local_solves_ok']), freeze_state=freeze_state))

        for g in CHANNELS:
            channel_mismatches = []
            if actions[g] != cyc[f'rho_{g}_action']:
                channel_mismatches.append(f"action: got '{actions[g]}' expected '{cyc[f'rho_{g}_action']}'")
            if not _rho_close(after[g], cyc[f'rho_{g}_after']):
                channel_mismatches.append(f"rho_after: got {after[g]!r} expected {cyc[f'rho_{g}_after']!r}")
            if after_gamma[g] is not None and not _rho_close(after_gamma[g], cyc[f'gamma_{g}_after']):
                channel_mismatches.append(
                    f"gamma_after: got {after_gamma[g]!r} expected {cyc[f'gamma_{g}_after']!r}")
            if freeze_state[g]['frozen'] != bool(cyc[f'rho_frozen_{g}']):
                channel_mismatches.append(
                    f"frozen: got {freeze_state[g]['frozen']} expected {bool(cyc[f'rho_frozen_{g}'])}")
            if freeze_state[g]['unchanged_streak'] != cyc[f'rho_unchanged_streak_{g}']:
                channel_mismatches.append(
                    f"unchanged_streak: got {freeze_state[g]['unchanged_streak']} "
                    f"expected {cyc[f'rho_unchanged_streak_{g}']}")
            if freeze_state[g]['at_clamp'] != bool(cyc[f'rho_at_clamp_{g}']):
                channel_mismatches.append(
                    f"at_clamp: got {freeze_state[g]['at_clamp']} expected {bool(cyc[f'rho_at_clamp_{g}'])}")
            if freeze_state[g]['exempt_until_lifted']:
                channel_mismatches.append('exempt_until_lifted unexpectedly True on a default-off run')
            if channel_mismatches:
                mismatches.append({'run': run_name, 'cycle': cycle, 'channel': g, 'mismatches': channel_mismatches})

    return {
        'run': run_name, 'trajectory_path': os.path.relpath(traj_path, REPO),
        'n_cycles': len(traj), 'inferred_config': inferred_cfg,
        'passed': len(mismatches) == 0, 'n_mismatches': len(mismatches),
        'mismatches': mismatches[:50],
    }


def check_i_default_off_identity(results):
    admm_block_template = _load_case_admm_block()
    runs = {}
    overall_passed = True
    for run_name, traj_path in CHECK_I_TRAJECTORIES.items():
        result = _run_validation(run_name, traj_path, admm_block_template)
        runs[run_name] = result
        overall_passed = overall_passed and result['passed']
    results['default_off_identity'] = {
        'method': ("p515_s38_balancing_replay's own methodology (stand-in Pyomo models, the "
                   "REAL srp._update_admm_penalties/_init_admm_freeze_state, per-cycle "
                   "re-seeding from ground truth), extended to 5 trajectories with per-run "
                   "config INFERRED from each trajectory's own first cycle"),
        'runs': runs, 'pass': overall_passed,
    }


# ===========================================================================
# (ii) synthetic sequences
# ===========================================================================
def _seq_pattern(v=(1.0, 1.0, 1.0), pf=(1.0, 1.0, 1.0), ess=(1.0, 1.0, 1.0)):
    return {'v': v, 'pf': pf, 'ess': ess}


HOLD = (1.0, 1.0, 1.0)
FORCE_INCREASE = (30.0, 1.0, 1.0)  # primal >> dual_balance -> 'increased'


def check_ii_synthetic_sequences(results):
    planning, tso_model, dso_models, esso_model, _cv, _dv = (
        V1._build_admm_ready_state('p515s39zsc_check_ii'))
    admm_params = planning.params.admm
    admm_params.penalty_update['freeze_after_unchanged_cycles'] = 10
    admm_params.penalty_update['freeze_backstop_cycle'] = 200
    admm_params.penalty_update['balancing_exempt_channels'] = []

    def ess_dual(ratio):
        return (1.0, ratio, 1.0)  # (primal_ratio, dual_ratio=FULL s/eps_dual, dual_ratio_balance)

    # ---- sub-check 1: 4-then-break-then-5 lifts at the cycle the 5th
    #      CONSECUTIVE below-threshold call completes (cycle 10, not 9) ----
    admm_params.penalty_update['balancing_exempt_until'] = {
        'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}
    seq = [_seq_pattern(ess=ess_dual(0.5)) for _ in range(4)]
    seq.append(_seq_pattern(ess=ess_dual(2.0)))  # break the streak
    seq += [_seq_pattern(ess=ess_dual(0.5)) for _ in range(5)]  # 5 consecutive -> lift at cycle 10
    seq += [_seq_pattern(ess=ess_dual(0.5)) for _ in range(3)]
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj, _fs = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq)
    lift_row = next((r for r in traj if r['freeze_state']['ess']['exempt_until_lift_cycle'] is not None), None)
    lift_cycle = lift_row['cycle'] if lift_row else None
    pending_before_lift = all(
        r['actions']['ess'] == 'exempt (fixed)' for r in traj if lift_cycle is None or r['cycle'] < lift_cycle)
    streak_at_break_resets = (traj[4]['freeze_state']['ess']['exempt_until_streak'] == 0)  # cycle 5 = the break
    sub1 = {
        'lift_cycle': lift_cycle, 'expected_lift_cycle': 10,
        'lift_cycle_correct': (lift_cycle == 10),
        'pending_exempt_fixed_before_lift': pending_before_lift,
        'streak_reset_at_break_cycle_5': streak_at_break_resets,
        'streak_trajectory': [r['freeze_state']['ess']['exempt_until_streak'] for r in traj],
        'pass': bool(lift_cycle == 10 and pending_before_lift and streak_at_break_resets),
    }

    # ---- sub-check 2: one-way -- forcing the ratio back above threshold
    #      after the lift never re-exempts ------------------------------
    seq2 = [_seq_pattern(ess=ess_dual(0.5)) for _ in range(5)]  # lifts at cycle 5
    seq2 += [_seq_pattern(ess=ess_dual(50.0)) for _ in range(10)]  # well above threshold, post-lift
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj2, _fs2 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq2)
    lifted_at_5 = traj2[4]['freeze_state']['ess']['exempt_until_lifted']
    never_re_exempted = all(r['actions']['ess'] != 'exempt (fixed)' for r in traj2[5:])
    sub2 = {
        'lifted_at_cycle_5': lifted_at_5, 'never_re_exempted_after': never_re_exempted,
        'actions_after_lift': [r['actions']['ess'] for r in traj2[5:]],
        'pass': bool(lifted_at_5 and never_re_exempted),
    }

    # ---- sub-check 3: the standard rule actually BALANCES the channel in
    #      the lift cycle's own update when the ratio test says so (forced
    #      'increased', not merely 'held') ------------------------------
    seq3 = [_seq_pattern(ess=ess_dual(0.5)) for _ in range(4)]
    # 5th below-threshold call ALSO has a strongly-imbalanced primal/dual_
    # ratio_balance pair -- primal_ratio=30 forces 'increased' once the
    # standard rule is reached (dual_ratio itself, index 1, stays 0.5 <
    # threshold so the streak still completes).
    seq3.append(_seq_pattern(ess=(30.0, 0.5, 1.0)))
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj3, _fs3 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq3)
    lift_row3 = traj3[4]
    sub3 = {
        'lift_cycle': lift_row3['cycle'], 'lifted': lift_row3['freeze_state']['ess']['exempt_until_lifted'],
        'action_at_lift': lift_row3['actions']['ess'],
        'rho_before': lift_row3['rho_before']['ess'], 'rho_after': lift_row3['rho_after']['ess'],
        'rho_actually_changed': not _rho_close(lift_row3['rho_before']['ess'], lift_row3['rho_after']['ess']),
        'pass': bool(lift_row3['freeze_state']['ess']['exempt_until_lifted']
                    and lift_row3['actions']['ess'] == 'increased'
                    and not _rho_close(lift_row3['rho_before']['ess'], lift_row3['rho_after']['ess'])),
    }

    # ---- sub-check 4: unchanged-streak freeze counts ONLY from a POST-LIFT
    #      action (never from cycles spent pending) -- lift at cycle 5 with
    #      a real 'increased' action (as sub-check 3), then HOLD forever;
    #      freeze via 'streak' at lift_cycle + 10 = cycle 15, not earlier --
    seq4 = [_seq_pattern(ess=ess_dual(0.5)) for _ in range(4)]
    seq4.append(_seq_pattern(ess=(30.0, 0.5, 1.0)))  # cycle 5: lifts AND increases
    seq4 += [_seq_pattern(ess=HOLD) for _ in range(15)]  # cycles 6-20: hold
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj4, _fs4 = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq4)
    streak_freeze_cycle = next(
        (r['cycle'] for r in traj4 if r['freeze_state']['ess']['reason'] == 'streak'), None)
    not_frozen_before_15 = all(
        not r['freeze_state']['ess']['frozen'] for r in traj4 if r['cycle'] < 15)
    sub4 = {
        'lift_cycle': 5, 'streak_freeze_cycle': streak_freeze_cycle, 'expected_streak_freeze_cycle': 15,
        'not_frozen_before_expected_cycle': not_frozen_before_15,
        'pass': bool(streak_freeze_cycle == 15 and not_frozen_before_15),
    }

    # ---- sub-check 5a: absolute backstop (200) freezes a channel that
    #      lifted early (cycle 5, real action) and then keeps ACTING
    #      (alternating increase/decrease, never a long-enough hold streak)
    #      all the way to cycle 199 -- so ONLY the backstop, not 'streak',
    #      can trigger the freeze (isolates the backstop mechanism) --------
    FORCE_DECREASE = (1.0, 30.0, 30.0)
    seq5a = [_seq_pattern(ess=ess_dual(0.5)) for _ in range(4)]
    seq5a.append(_seq_pattern(ess=(30.0, 0.5, 1.0)))  # cycle 5: lifts AND increases
    for c in range(195):  # cycles 6-200
        seq5a.append(_seq_pattern(ess=FORCE_INCREASE if c % 2 == 0 else FORCE_DECREASE))
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj5a, _fs5a = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq5a)
    row200a = next(r for r in traj5a if r['cycle'] == 200)
    frozen_by_200_post_lift = row200a['freeze_state']['ess']['frozen']
    frozen_via_backstop_not_streak = (row200a['freeze_state']['ess']['reason'] == 'backstop')
    not_frozen_before_200 = all(
        not r['freeze_state']['ess']['frozen'] for r in traj5a if r['cycle'] < 200)

    # ---- sub-check 5b: the absolute backstop does NOT interrupt a channel
    #      still PENDING (never lifted) past cycle 200 -- dual_ratio never
    #      dips below threshold ------------------------------------------
    seq5b = [_seq_pattern(ess=ess_dual(50.0)) for _ in range(250)]
    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    traj5b, _fs5b = S37Z._run_trajectory(tso_model, dso_models, esso_model, admm_params, seq5b)
    row250b = next(r for r in traj5b if r['cycle'] == 250)
    still_pending_at_250 = (row250b['actions']['ess'] == 'exempt (fixed)'
                            and not row250b['freeze_state']['ess']['exempt_until_lifted'])
    sub5 = {
        'post_lift_frozen_by_cycle_200': frozen_by_200_post_lift,
        'post_lift_frozen_via_backstop_not_streak': frozen_via_backstop_not_streak,
        'post_lift_not_frozen_before_cycle_200': not_frozen_before_200,
        'pending_channel_action_at_cycle_250': row250b['actions']['ess'],
        'pending_channel_never_lifted_at_250': not row250b['freeze_state']['ess']['exempt_until_lifted'],
        'pending_survives_backstop_at_250': still_pending_at_250,
        'pass': bool(frozen_by_200_post_lift and frozen_via_backstop_not_streak and not_frozen_before_200
                    and still_pending_at_250),
    }

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    S37Z._reset_gamma_tied(tso_model, 1.0)
    admm_params.penalty_update['balancing_exempt_until'] = {}

    # ---- sub-check 6: read_parameters_from_file rejects every documented
    #      bad configuration -----------------------------------------------
    base = _load_case_admm_block()

    def _try(exempt_until_value, exempt_channels_value=None):
        admm = ADMMParameters()
        data = deepcopy(base)
        data['penalty_update'] = dict(data.get('penalty_update', {}))
        data['penalty_update']['balancing_exempt_until'] = exempt_until_value
        if exempt_channels_value is not None:
            data['penalty_update']['balancing_exempt_channels'] = exempt_channels_value
        try:
            admm.read_parameters_from_file(data)
            return True, None
        except ValueError as exc:
            return False, str(exc)

    ok_valid, err_valid = _try({'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}})
    ok_bad_channel, err_bad_channel = _try({'xyz': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}})
    ok_bad_threshold, err_bad_threshold = _try({'ess': {'dual_ratio_below': 0.0, 'consecutive_cycles': 5}})
    ok_bad_threshold_neg, err_bad_threshold_neg = _try({'ess': {'dual_ratio_below': -1.0, 'consecutive_cycles': 5}})
    ok_bad_consecutive, err_bad_consecutive = _try({'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 0}})
    ok_overlap, err_overlap = _try(
        {'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}, exempt_channels_value=['ess'])

    sub6 = {
        'valid_config_loads': ok_valid, 'valid_config_error': err_valid,
        'unknown_channel_rejected': (not ok_bad_channel), 'unknown_channel_error': err_bad_channel,
        'zero_threshold_rejected': (not ok_bad_threshold), 'zero_threshold_error': err_bad_threshold,
        'negative_threshold_rejected': (not ok_bad_threshold_neg), 'negative_threshold_error': err_bad_threshold_neg,
        'zero_consecutive_rejected': (not ok_bad_consecutive), 'zero_consecutive_error': err_bad_consecutive,
        'overlap_with_exempt_channels_rejected': (not ok_overlap), 'overlap_error': err_overlap,
        'pass': bool(
            ok_valid and (not ok_bad_channel) and (not ok_bad_threshold) and (not ok_bad_threshold_neg)
            and (not ok_bad_consecutive) and (not ok_overlap)
            and err_bad_channel and 'balancing_exempt_until' in err_bad_channel
            and err_bad_threshold and 'dual_ratio_below' in err_bad_threshold
            and err_bad_consecutive and 'consecutive_cycles' in err_bad_consecutive
            and err_overlap and 'BOTH' in err_overlap
        ),
    }

    results['synthetic_sequences'] = {
        'sub1_lift_after_4_break_5_consecutive': sub1,
        'sub2_one_way_never_re_exempted': sub2,
        'sub3_balances_at_lift_when_ratio_test_says_so': sub3,
        'sub4_streak_freeze_counts_from_lift': sub4,
        'sub5_absolute_backstop_post_lift_vs_pending': sub5,
        'sub6_validation_errors': sub6,
    }
    results['synthetic_sequences']['pass'] = all(
        v['pass'] for v in results['synthetic_sequences'].values() if isinstance(v, dict) and 'pass' in v)


# ===========================================================================
# (iii) tau=0 and minimum_consecutive_converged_cycles=10 load AND are in
#       force on a freshly built (unsolved) planning object
# ===========================================================================
def check_iii_tau0_and_min_consecutive_10(results):
    base = _load_case_admm_block()
    data = deepcopy(base)
    data['proximal_regularization'] = dict(data.get('proximal_regularization', {}))
    data['proximal_regularization']['tso'] = dict(data['proximal_regularization'].get('tso', {}))
    data['proximal_regularization']['tso']['tau'] = 0.0
    data['minimum_consecutive_converged_cycles'] = 10

    admm = ADMMParameters()
    admm.read_parameters_from_file(data)
    tau_loads_as_0 = (admm.proximal_regularization['tso']['tau'] == 0.0)
    min_consecutive_loads_as_10 = (admm.minimum_consecutive_converged_cycles == 10)

    planning, tso_model, dso_models, esso_model, _cv, _dv = (
        V1._build_admm_ready_state('p515s39zsc_check_iii'))
    planning.params.admm.proximal_regularization['tso']['tau'] = 0.0
    planning.params.admm.minimum_consecutive_converged_cycles = 10
    tau_in_force_on_planning = (planning.params.admm.proximal_regularization['tso']['tau'] == 0.0)
    min_consecutive_in_force_on_planning = (planning.params.admm.minimum_consecutive_converged_cycles == 10)

    results['tau0_and_min_consecutive_10'] = {
        'loaded_via_read_parameters_from_file': {
            'tau_loads_as_0': tau_loads_as_0, 'min_consecutive_loads_as_10': min_consecutive_loads_as_10,
        },
        'in_force_on_built_unsolved_planning_object': {
            'tau_in_force': tau_in_force_on_planning,
            'min_consecutive_in_force': min_consecutive_in_force_on_planning,
        },
        'pass': bool(tau_loads_as_0 and min_consecutive_loads_as_10
                    and tau_in_force_on_planning and min_consecutive_in_force_on_planning),
    }


# ===========================================================================
# (iv) structural fix: working-dir ids never collide, per arm and across
#      arms/modes
# ===========================================================================
def check_iv_working_dir_ids_disjoint(results):
    per_arm = {}
    all_ids = []
    all_pass = True
    for arm_key in ('s39_C', 's39_D'):
        ids_by_mode = G._s39_working_dir_ids(arm_key)
        real_ids = set(ids_by_mode['real'].values())
        preflight_ids = set(ids_by_mode['preflight'].values())
        disjoint = real_ids.isdisjoint(preflight_ids)
        all_pass = all_pass and disjoint
        per_arm[arm_key] = {'ids_by_mode': ids_by_mode, 'real_disjoint_from_preflight': disjoint}
        all_ids.extend(real_ids)
        all_ids.extend(preflight_ids)

    all_twelve_distinct = (len(all_ids) == len(set(all_ids)) == 12)
    all_pass = all_pass and all_twelve_distinct

    results['working_dir_ids_disjoint'] = {
        'per_arm': per_arm, 'all_ids_flat': sorted(all_ids),
        'n_ids_total': len(all_ids), 'n_ids_distinct': len(set(all_ids)),
        'all_twelve_ids_pairwise_distinct': all_twelve_distinct,
        'pass': bool(all_pass),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    observed_hash = _spec_v10_hash()
    print(f'[S39 zero-solve checks] spec v10 sha256={observed_hash}')
    if observed_hash != G.S39_SPEC_SHA256:
        raise RuntimeError(
            f'frozen s39 spec hash mismatch: file={observed_hash} expected={G.S39_SPEC_SHA256}')

    guard = SolveProfileGuard(permitted=(), label='S39 zero-solve check').install()
    results = {}
    try:
        check_i_default_off_identity(results)
        check_ii_synthetic_sequences(results)
        check_iii_tau0_and_min_consecutive_10(results)
        check_iv_working_dir_ids_disjoint(results)
    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Addendum 21 (s39 preparation worker task) -- zero-solve verification '
                 'of the conditional balancing exemption (balancing_exempt_until) and the s39 '
                 'harness working-dir-id structural fix',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 21',
            'data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json',
        ],
        'spec_file': os.path.relpath(SPEC_V10_PATH, REPO),
        'spec_file_sha256_observed': observed_hash,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S39] wrote {OUT_PATH}')
    print(f'[S39] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
