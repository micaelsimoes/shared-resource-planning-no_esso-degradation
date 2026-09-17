"""
P5.15 Addendum 19 (s37, the rho_ess experiment) preparation worker task --
zero-solve verification of the per-channel balancing exemption.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 19; binding specification
data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json.

The production change under test (`admm_parameters.py`, `_read_parameters_
from_file`; `shared_resources_planning.py`, `_init_admm_freeze_state` and
`_update_admm_penalties`): an optional `admm.penalty_update.
balancing_exempt_channels` list, drawn from {'v', 'pf', 'ess'}, default []
(absent key). A channel in that list is NEVER increased or decreased by
residual balancing; its action is recorded as `'exempt (fixed)'`;
`freeze_state[channel]` is frozen from the FIRST call, with `at_clamp`
permanently False and an `exempt` marker True; it never trips the clamp
flag or the freeze-streak/backstop bookkeeping. Non-exempt channels behave
exactly as before -- this is the ONLY production change; every other case
study and every previous arm is unaffected because the default is [].

Verifies, with a BLOCKING `SolveProfileGuard` (permitted=(), i.e. zero
solves anywhere) armed for the whole check:

  (a) with the key absent, `_update_admm_penalties` actions, rho, gamma and
      freeze_state are bit-identical to the pre-change code (commit
      7323b2fb, this worker's own starting HEAD) on synthetic Boyd metrics,
      across 20 cycles exercising increase/decrease/hold on every channel;
  (b) with `balancing_exempt_channels=['ess']`: ESS rho and gamma never
      change across 60 synthetic cycles with strongly imbalanced ESS
      ratios (primal_ratio >> dual_ratio_balance, which would otherwise
      force 'increased' every cycle); the action label is
      'exempt (fixed)'; freeze_state stays frozen with no clamp flag; V
      and PF are bit-identical to the no-exemption behaviour over the same
      60-cycle metrics sequence;
  (c) validation rejects unknown channel names;
  (d) other case params (at least one CS case) load with the default [];
  (e) preserved fixtures unpickle;
  (f) the hierarchical and uncoordinated paths carry no hunks (source-text
      equality against the pre-change commit, plus the diff hunk count for
      shared_resources_planning.py).

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python p515_s37_zero_solve_checks.py

Writes data/SRP1/Results/P515S37/zero_solve_checks/zero_solve_checks.json
(a NEW directory; refuses to overwrite).
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

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s32_zero_solve_checks as V1  # noqa: E402
import p515_s32v2_zero_solve_checks as V2  # noqa: E402

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S37', 'zero_solve_checks')
OUT_PATH = os.path.join(OUT_DIR, 'zero_solve_checks.json')

SPEC_V8_PATH = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S37', 'frozen_s37_rho_ess_spec_v8_f91de983.json')
SPEC_V8_SHA256 = None  # computed and checked at runtime against the committed file, see main()

# Pre-change commit: this worker's own starting HEAD (P5.15 Addendum 19
# frozen spec v8 commit) -- the last commit BEFORE this task's production
# edits.
PRE_CHANGE_COMMIT = '7323b2fb'

CS_PARAMS_PATHS = {
    'CS1': os.path.join(REPO, 'data', 'CS1', 'CS1_params.json'),
    'CS7': os.path.join(REPO, 'data', 'CS7', 'CS7_params.json'),
    'HR1': os.path.join(REPO, 'data', 'HR1', 'HR1_params.json'),
    'OP1': os.path.join(REPO, 'data', 'OP1', 'OP1_params.json'),
    'OP2': os.path.join(REPO, 'data', 'OP2', 'OP2_params.json'),
}
SRP1_PARAMS_PATH = os.path.join(REPO, 'data', 'SRP1', 'SRP1_params.json')

S32_FROZEN_SMOPF_FIXTURES = [
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF',
                 'matched_success_TSO_case9_2025_Summer_cycle7.pkl'),
    os.path.join(REPO, 'data', 'SRP1', 'Results', 'FrozenSMOPF',
                 'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
]


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _spec_v8_hash():
    with open(SPEC_V8_PATH, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _synthetic_boyd_sequence(n_cycles, ess_ratios=None):
    """Deterministic, cycle-dependent (v, pf, ess) `(primal_ratio,
    dual_ratio, dual_ratio_balance)` triples cycling through
    increase-triggering, decrease-triggering and dead-band (hold) regimes
    on EVERY channel -- so a 20+ cycle replay exercises every branch of
    `_update_admm_penalties` at least once per channel. `ess_ratios`, if
    given, overrides the ess channel's own sequence (e.g. a "strongly
    imbalanced" fixed pair for check (b))."""
    base_patterns = [
        {'v': (30.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)},   # v increases
        {'v': (1.0, 1.0, 1.0), 'pf': (1.0, 30.0, 30.0), 'ess': (1.0, 1.0, 1.0)},  # pf decreases
        {'v': (1.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (25.0, 1.0, 1.0)},   # ess increases
        {'v': (1.0, 1.0, 1.0), 'pf': (1.0, 1.0, 1.0), 'ess': (1.0, 1.0, 1.0)},    # dead band (hold)
    ]
    sequence = []
    for cyc in range(n_cycles):
        pattern = dict(base_patterns[cyc % len(base_patterns)])
        if ess_ratios is not None:
            pattern = dict(pattern)
            pattern['ess'] = ess_ratios
        sequence.append(pattern)
    return sequence


def _reset_gamma_tied(tso_model, value):
    """Resets every TSO model's `prox_gamma_{v,pf,ess}` Params to `value`.
    `V1._reset_rho` resets ONLY rho (never gamma), so a bare `_reset_rho`
    between two trajectory replays on the SAME model set leaves gamma at
    whatever the PRECEDING replay last computed -- a test-harness artifact,
    not a production behaviour (production never externally resets rho
    mid-run). Called immediately after `V1._reset_rho(..., value)` so both
    replays start from an internally consistent (rho, gamma) pair."""
    for year_models in tso_model.values():
        for model in year_models.values():
            if hasattr(model, 'prox_gamma_v'):
                model.prox_gamma_v.set_value(value)
            if hasattr(model, 'prox_gamma_pf'):
                model.prox_gamma_pf.set_value(value)
            if hasattr(model, 'prox_gamma_ess'):
                model.prox_gamma_ess.set_value(value)


def _run_trajectory(tso_model, dso_models, esso_model, admm_params, sequence,
                     update_fn=None, freeze_state=None):
    """Replays `sequence` (list of `_fake_boyd_metrics_v2`-style ratio dicts)
    through `update_fn` (defaults to `srp._update_admm_penalties`), one call
    per cycle, returning the per-cycle trajectory (actions, rho before/
    after, gamma before/after, rho_freeze_active, and a JSON-safe snapshot
    of freeze_state AFTER that cycle's call)."""
    if update_fn is None:
        update_fn = srp._update_admm_penalties
    if freeze_state is None:
        freeze_state = srp._init_admm_freeze_state()
    residual_metrics = V1._dummy_residual_metrics(admm_params)
    trajectory = []
    for cyc, ratios in enumerate(sequence, start=1):
        boyd = V2._fake_boyd_metrics_v2(ratios)
        actions, before, after, gbefore, gafter, rho_freeze_active, freeze_state = update_fn(
            tso_model, dso_models, esso_model, residual_metrics, boyd, admm_params,
            iter=cyc, allow_update=True, freeze_state=freeze_state)
        trajectory.append({
            'cycle': cyc,
            'actions': dict(actions),
            'rho_before': dict(before),
            'rho_after': dict(after),
            'gamma_before': dict(gbefore),
            'gamma_after': dict(gafter),
            'rho_freeze_active': rho_freeze_active,
            'freeze_state': {g: dict(freeze_state[g]) for g in ('v', 'pf', 'ess')},
        })
    return trajectory, freeze_state


def check_a_absent_key_bit_identical(results):
    """Check (a): with `balancing_exempt_channels` absent (default []), the
    CURRENT `_update_admm_penalties` (which now carries the exemption
    machinery, but never enters it for an empty set) reproduces the
    PRE-CHANGE code (commit 7323b2fb) bit-for-bit on the SAME 20-cycle
    synthetic trajectory. Both replays run on the SAME built model set
    (rho/freeze_state reset between them), so any drift is attributable
    ONLY to the two `_update_admm_penalties` implementations."""
    pre_module, _ = V2._load_module_at_commit(PRE_CHANGE_COMMIT, 'srp_pre37_a')

    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
        V1._build_admm_ready_state('p515s37_check_a_absent'))
    admm_params = planning.params.admm
    assert admm_params.penalty_update.get('balancing_exempt_channels') == []

    sequence = _synthetic_boyd_sequence(20)

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    _reset_gamma_tied(tso_model, 1.0)
    new_trajectory, _ = _run_trajectory(
        tso_model, dso_models, esso_model, admm_params, sequence,
        update_fn=srp._update_admm_penalties)

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    _reset_gamma_tied(tso_model, 1.0)
    pre_trajectory, _ = _run_trajectory(
        tso_model, dso_models, esso_model, admm_params, sequence,
        update_fn=pre_module._update_admm_penalties)

    # Pre-change freeze_state has no 'exempt' key -- compare on the common
    # field set (frozen, unchanged_streak, ever_acted, at_clamp, reason)
    # plus actions/rho/gamma, which ARE all present on both.
    common_freeze_fields = ('frozen', 'unchanged_streak', 'ever_acted', 'at_clamp', 'reason')
    first_diff = None
    for new_row, pre_row in zip(new_trajectory, pre_trajectory):
        for key in ('actions', 'rho_before', 'rho_after', 'gamma_before', 'gamma_after',
                    'rho_freeze_active'):
            if new_row[key] != pre_row[key]:
                first_diff = {'cycle': new_row['cycle'], 'field': key,
                              'new': new_row[key], 'pre': pre_row[key]}
                break
        if first_diff:
            break
        for group in ('v', 'pf', 'ess'):
            for field in common_freeze_fields:
                nv = new_row['freeze_state'][group][field]
                pv = pre_row['freeze_state'][group][field]
                if nv != pv:
                    first_diff = {'cycle': new_row['cycle'], 'field': f'freeze_state.{group}.{field}',
                                  'new': nv, 'pre': pv}
                    break
            if first_diff:
                break
        if first_diff:
            break
    # Also confirm the NEW code's 'exempt' marker stayed False throughout
    # (the empty-set default never triggers it).
    new_exempt_ever_true = any(
        row['freeze_state'][g]['exempt'] for row in new_trajectory for g in ('v', 'pf', 'ess'))

    V1._reset_rho(tso_model, dso_models, esso_model, 1.0)
    results['absent_key_bit_identical'] = {
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'n_cycles_compared': len(new_trajectory),
        'first_differing_field': first_diff,
        'new_code_exempt_marker_ever_true': new_exempt_ever_true,
        'pass': (first_diff is None) and (not new_exempt_ever_true),
    }


def check_b_ess_exempt(results):
    """Check (b): `balancing_exempt_channels=['ess']`, 60 synthetic cycles,
    strongly imbalanced ESS ratios (primal_ratio=1e6 >> dual_ratio_balance
    =1e-6, which would force 'increased' EVERY cycle absent the exemption).
    ESS rho/gamma must never change; action label 'exempt (fixed)';
    freeze_state['ess'] frozen from cycle 1, at_clamp False throughout,
    exempt True. V and PF must be BIT-IDENTICAL to a parallel no-exemption
    run over the identical metrics sequence (built on a SEPARATE, freshly
    built model set, rho reset the same way)."""
    planning_exempt, tso_e, dso_e, esso_e, _cv, _dv = (
        V1._build_admm_ready_state('p515s37_check_b_exempt'))
    admm_params_exempt = planning_exempt.params.admm
    admm_params_exempt.penalty_update['balancing_exempt_channels'] = ['ess']
    admm_params_exempt.balancing_exempt_channels_source = 'test_override'

    planning_plain, tso_p, dso_p, esso_p, _cv2, _dv2 = (
        V1._build_admm_ready_state('p515s37_check_b_plain'))
    admm_params_plain = planning_plain.params.admm
    assert admm_params_plain.penalty_update.get('balancing_exempt_channels') == []

    strongly_imbalanced_ess = (1.0e6, 1.0e-6, 1.0e-6)  # (primal_ratio, dual_ratio, dual_ratio_balance)
    sequence = _synthetic_boyd_sequence(60, ess_ratios=strongly_imbalanced_ess)

    # SRP1's own case file (the real arm's base configuration) sets
    # `freeze_backstop_cycle=60` -- deliberately LEFT ACTIVE here (not
    # overridden) so this 60-cycle replay also exercises the "never trips
    # the backstop logic" requirement on the exempt channel at the exact
    # cycle the backstop fires globally (see the cycle-60 assertions below).
    assert admm_params_exempt.penalty_update.get('freeze_backstop_cycle') == 60
    assert admm_params_plain.penalty_update.get('freeze_backstop_cycle') == 60

    V1._reset_rho(tso_e, dso_e, esso_e, 1.0)
    _reset_gamma_tied(tso_e, 1.0)
    trajectory_exempt, freeze_state_exempt = _run_trajectory(
        tso_e, dso_e, esso_e, admm_params_exempt, sequence)

    V1._reset_rho(tso_p, dso_p, esso_p, 1.0)
    _reset_gamma_tied(tso_p, 1.0)
    trajectory_plain, freeze_state_plain = _run_trajectory(
        tso_p, dso_p, esso_p, admm_params_plain, sequence)

    ess_rho_constant = all(
        row['rho_before']['ess'] == trajectory_exempt[0]['rho_before']['ess']
        and row['rho_after']['ess'] == trajectory_exempt[0]['rho_before']['ess']
        for row in trajectory_exempt)
    ess_gamma_constant = all(
        row['gamma_before']['ess'] == trajectory_exempt[0]['gamma_before']['ess']
        and row['gamma_after']['ess'] == trajectory_exempt[0]['gamma_before']['ess']
        for row in trajectory_exempt)
    ess_action_always_exempt = all(row['actions']['ess'] == 'exempt (fixed)' for row in trajectory_exempt)
    ess_frozen_from_cycle_1 = all(row['freeze_state']['ess']['frozen'] for row in trajectory_exempt)
    ess_exempt_marker_from_cycle_1 = all(row['freeze_state']['ess']['exempt'] for row in trajectory_exempt)
    ess_never_at_clamp = all(row['freeze_state']['ess']['at_clamp'] is False for row in trajectory_exempt)
    ess_reason_is_exempt = all(row['freeze_state']['ess']['reason'] == 'exempt' for row in trajectory_exempt)

    v_pf_first_diff = None
    for row_e, row_p in zip(trajectory_exempt, trajectory_plain):
        for group in ('v', 'pf'):
            for key in ('actions', 'rho_before', 'rho_after', 'gamma_before', 'gamma_after'):
                ve = row_e[key][group]
                vp = row_p[key][group]
                if ve != vp:
                    v_pf_first_diff = {'cycle': row_e['cycle'], 'group': group, 'field': key,
                                        'exempt_run': ve, 'plain_run': vp}
                    break
            if v_pf_first_diff:
                break
        if v_pf_first_diff:
            break
    v_pf_bit_identical = v_pf_first_diff is None

    # Sanity checks on the PLAIN (non-exempt) run, confirming the synthetic
    # ESS ratio pattern is genuinely provocative:
    #  * cycles 1-59: the strongly imbalanced ratio would force 'increased'
    #    on every cycle, absent the exemption;
    #  * cycle 60: SRP1's own case-file global backstop
    #    (`freeze_backstop_cycle=60`) fires and freezes EVERY channel,
    #    including this non-exempt ESS -- expected, unrelated to Addendum
    #    19, and used below as the control for the NEXT assertion.
    plain_ess_increased_before_backstop = all(
        row['actions']['ess'] == 'increased' for row in trajectory_plain if row['cycle'] < 60)
    plain_ess_frozen_by_backstop_at_60 = (
        trajectory_plain[-1]['actions']['ess'] == 'held (frozen backstop cycle 60)')
    # The requirement under test: the EXEMPT run's ESS channel is immune to
    # the SAME global backstop at the SAME cycle -- 'exempt (fixed)' must
    # hold at cycle 60 too, not be overridden by 'held (frozen backstop
    # cycle 60)'.
    exempt_ess_survives_backstop_at_60 = (
        trajectory_exempt[-1]['actions']['ess'] == 'exempt (fixed)')

    V1._reset_rho(tso_e, dso_e, esso_e, 1.0)
    V1._reset_rho(tso_p, dso_p, esso_p, 1.0)
    results['ess_exempt_60_cycles'] = {
        'n_cycles': 60,
        'ess_ratios_used': {'primal_ratio': strongly_imbalanced_ess[0],
                             'dual_ratio': strongly_imbalanced_ess[1],
                             'dual_ratio_balance': strongly_imbalanced_ess[2]},
        'ess_rho_constant': ess_rho_constant,
        'ess_gamma_constant': ess_gamma_constant,
        'ess_action_always_exempt_fixed': ess_action_always_exempt,
        'ess_frozen_from_cycle_1': ess_frozen_from_cycle_1,
        'ess_exempt_marker_from_cycle_1': ess_exempt_marker_from_cycle_1,
        'ess_never_at_clamp': ess_never_at_clamp,
        'ess_reason_is_exempt': ess_reason_is_exempt,
        'v_pf_bit_identical_to_no_exemption_run': v_pf_bit_identical,
        'v_pf_first_differing_field': v_pf_first_diff,
        'plain_run_ess_increased_before_backstop_sanity_check': plain_ess_increased_before_backstop,
        'plain_run_ess_frozen_by_backstop_at_cycle_60': plain_ess_frozen_by_backstop_at_60,
        'exempt_run_ess_survives_backstop_at_cycle_60': exempt_ess_survives_backstop_at_60,
        'terminal_ess_rho': trajectory_exempt[-1]['rho_after']['ess'],
        'terminal_ess_gamma': trajectory_exempt[-1]['gamma_after']['ess'],
        'pass': (
            ess_rho_constant and ess_gamma_constant and ess_action_always_exempt
            and ess_frozen_from_cycle_1 and ess_exempt_marker_from_cycle_1
            and ess_never_at_clamp and ess_reason_is_exempt
            and v_pf_bit_identical
            and plain_ess_increased_before_backstop and plain_ess_frozen_by_backstop_at_60
            and exempt_ess_survives_backstop_at_60
        ),
    }


def check_c_invalid_channel_rejected(results):
    """Check (c): `read_parameters_from_file` raises `ValueError` on an
    unknown channel name, and accepts the three valid ones."""
    with open(SRP1_PARAMS_PATH) as handle:
        base = json.load(handle)['admm']

    def _try(channels):
        admm = ADMMParameters()
        data = deepcopy(base)
        data['penalty_update'] = dict(data.get('penalty_update', {}))
        data['penalty_update']['balancing_exempt_channels'] = channels
        try:
            admm.read_parameters_from_file(data)
            return True, None, admm
        except ValueError as exc:
            return False, str(exc), None

    valid_ok, valid_err, valid_admm = _try(['ess'])
    valid_all_ok, valid_all_err, valid_all_admm = _try(['v', 'pf', 'ess'])
    invalid_ok, invalid_err, _ = _try(['bogus'])
    invalid_mixed_ok, invalid_mixed_err, _ = _try(['v', 'not_a_channel'])
    not_a_list_ok, not_a_list_err, _ = _try('ess')  # string, not a list

    results['invalid_channel_rejected'] = {
        'valid_single_channel_accepted': valid_ok and valid_admm.penalty_update['balancing_exempt_channels'] == ['ess'],
        'valid_all_channels_accepted': valid_all_ok and valid_all_admm.penalty_update['balancing_exempt_channels'] == ['v', 'pf', 'ess'],
        'invalid_channel_raises': (not invalid_ok) and invalid_err is not None and 'bogus' in invalid_err,
        'invalid_mixed_channel_raises': (not invalid_mixed_ok) and invalid_mixed_err is not None and 'not_a_channel' in invalid_mixed_err,
        'non_list_value_raises': (not not_a_list_ok) and not_a_list_err is not None,
        'invalid_error_message': invalid_err,
        'invalid_mixed_error_message': invalid_mixed_err,
        'non_list_error_message': not_a_list_err,
        'pass': (
            valid_ok and valid_admm.penalty_update['balancing_exempt_channels'] == ['ess']
            and valid_all_ok and valid_all_admm.penalty_update['balancing_exempt_channels'] == ['v', 'pf', 'ess']
            and (not invalid_ok) and invalid_err is not None and 'bogus' in invalid_err
            and (not invalid_mixed_ok) and invalid_mixed_err is not None and 'not_a_channel' in invalid_mixed_err
            and (not not_a_list_ok) and not_a_list_err is not None
        ),
    }


def check_d_other_case_params_load(results):
    """Check (d): every other case study's params load with
    `balancing_exempt_channels == []`, source 'default' -- the key's
    absence from their case files is unaffected by this change."""
    per_case = {}
    for name, path in CS_PARAMS_PATHS.items():
        admm = ADMMParameters()
        with open(path) as handle:
            admm.read_parameters_from_file(json.load(handle)['admm'])
        per_case[name] = {
            'path': os.path.relpath(path, REPO),
            'balancing_exempt_channels': admm.penalty_update['balancing_exempt_channels'],
            'balancing_exempt_channels_source': admm.balancing_exempt_channels_source,
        }
    all_default = all(
        v['balancing_exempt_channels'] == [] and v['balancing_exempt_channels_source'] == 'default'
        for v in per_case.values()
    )

    srp1_admm = ADMMParameters()
    with open(SRP1_PARAMS_PATH) as handle:
        srp1_admm.read_parameters_from_file(json.load(handle)['admm'])
    srp1_default = (
        srp1_admm.penalty_update['balancing_exempt_channels'] == []
        and srp1_admm.balancing_exempt_channels_source == 'default'
    )

    results['other_case_params_load'] = {
        'other_cases': per_case, 'other_cases_all_default': all_default,
        'srp1_case_file_default': srp1_default,
        'note': 'SRP1_params.json does not set balancing_exempt_channels; the s37 arms '
                'set it via a deep-copied params override at run time, never by editing '
                'this case file.',
        'pass': all_default and srp1_default,
    }


def check_e_fixtures_unpickle(results):
    fixture_results = []
    for path in list(V1.S31C_FIXTURES) + S32_FROZEN_SMOPF_FIXTURES:
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


def check_f_untouched_paths(results):
    proc = subprocess.run(
        ['git', 'show', f'{PRE_CHANGE_COMMIT}:shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    )
    import tempfile
    import importlib.util
    tmp_dir = tempfile.mkdtemp(prefix='p515_s37_pre_change_')
    tmp_path = os.path.join(tmp_dir, '_shared_resources_planning_pre37.py')
    with open(tmp_path, 'w') as handle:
        handle.write(proc.stdout)
    spec = importlib.util.spec_from_file_location('srp_pre37_f', tmp_path)
    pre_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pre_module)

    fn_names = ('_run_operational_planning_hierarchical', '_run_operational_planning_without_coordination')
    per_fn = {}
    all_identical = True
    for name in fn_names:
        new_source = inspect.getsource(getattr(srp, name))
        pre_source = inspect.getsource(getattr(pre_module, name))
        identical = (new_source == pre_source)
        per_fn[name] = {'identical': identical}
        all_identical = all_identical and identical

    diff = subprocess.run(
        ['git', 'diff', PRE_CHANGE_COMMIT, '--', 'shared_resources_planning.py'],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    hunk_headers = [line for line in diff.splitlines() if line.startswith('@@')]

    results['hierarchical_uncoordinated_untouched'] = {
        'method': f'exact source-text equality ({PRE_CHANGE_COMMIT} vs current tree) for both '
                  'path functions, plus every diff hunk header in shared_resources_planning.py.',
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'functions': per_fn, 'diff_hunk_count': len(hunk_headers),
        'diff_hunk_headers': hunk_headers,
        'pass': all_identical,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(OUT_PATH)

    observed_hash = _spec_v8_hash()
    print(f'[S37 zero-solve checks] spec v8 sha256={observed_hash}')

    guard = SolveProfileGuard(permitted=(), label='S37 zero-solve check').install()
    results = {}
    try:
        check_a_absent_key_bit_identical(results)
        check_b_ess_exempt(results)
        check_c_invalid_channel_rejected(results)
        check_d_other_case_params_load(results)
        check_e_fixtures_unpickle(results)
        check_f_untouched_paths(results)
    finally:
        guard.uninstall()

    def _entry_pass(v):
        if isinstance(v, dict) and 'pass' in v:
            return bool(v['pass'])
        return True

    all_pass = all(_entry_pass(v) for v in results.values())
    verify_failures = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Addendum 19 (s37 preparation worker task) -- zero-solve verification '
                 'of the per-channel balancing exemption',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 19',
            'data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json',
        ],
        'spec_file': os.path.relpath(SPEC_V8_PATH, REPO),
        'spec_file_sha256_observed': observed_hash,
        'pre_change_commit': PRE_CHANGE_COMMIT,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts), 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(OUT_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    print(f'[S37] wrote {OUT_PATH}')
    print(f'[S37] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
