"""
P5.15 Addendum 21 (s39 preparation worker task, W1 item 3) -- zero-solve,
OPEN-LOOP replay of arm D's two-phase ESS policy on an already-completed
trajectory, using the REAL production `shared_resources_planning._update_
admm_penalties`.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 21; binding specification
data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json
(`pre_launch_evidence_for_D`: "before D launches, a zero-solve open-loop
replay of the two-phase ESS policy (real _update_admm_penalties) on arm C's
completed trajectory (and on v9 arm A's) records the predicted lift cycle
and first ESS balancing action; recorded as a note, not a prediction
change").

## Method (extends `p515_s38_balancing_replay.py`'s OWN methodology, reused
## by import -- never reimplemented)

Same technique as that script's `run_counterfactual`: lightweight stand-in
Pyomo `ConcreteModel` objects (`p515_s38_balancing_replay.make_tso_model` /
`make_dso_model` / `make_esso_model`), driving the REAL, unmodified
`shared_resources_planning._update_admm_penalties` with `boyd_metrics`/
`residual_metrics` rebuilt, per cycle, directly from the trajectory's own
recorded fields (`p515_s38_balancing_replay.boyd_metrics_from_cycle` /
`residual_metrics_from_cycle`) -- OPEN-LOOP: the residual ratios are the
OBSERVED values from the real (donor) trajectory, never updated in response
to this replay's own counterfactual rho change (there is no re-solve to
produce them). rho/gamma are carried forward cycle-to-cycle from THIS
script's own prior output, starting from the donor trajectory's cycle-1
`rho_*_before` (the real run's initial condition). `freeze_state` (in/out)
is carried the same way, via `shared_resources_planning._init_admm_freeze_
state()`.

## Arm D's configuration (applied here, NOT the donor trajectory's own)

Loaded via the REAL `ADMMParameters.read_parameters_from_file`, from a deep
copy of the case file's `admm` block, with: `balancing_exempt_channels = []`
(ESS is NOT unconditionally exempt in D), `balancing_exempt_until =
{'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}` (the new
production conditional exemption, spec v10 `arms.s39_D`),
`freeze_after_unchanged_cycles = 10`, `freeze_backstop_cycle = 200`
(Addendum 20's rule, unchanged), `proximal_regularization.tso.tau = 0.0`
(spec v10 `decisions_in_force.tau`). This is D's policy replayed on the
DONOR trajectory's observed ratios -- the donor's own actual configuration
(e.g. s38 arm A's PF exemption, or arm B's tau=1) is irrelevant here; only
its recorded boyd ratios and cycle count are read.

## No production, case-file or harness file is edited. No solve is ever
## attempted: `SolveProfileGuard([])` is installed for the whole script and
## `verify(expected_solves=0)` is asserted before writing any output.

## Reported per run

  - the predicted ESS lift cycle (`freeze_state['ess']['exempt_until_lift_
    cycle']`, first cycle it becomes non-None) and the streak trajectory
    leading to it;
  - the first ESS balancing action strictly AFTER the lift (cycle,
    direction -- 'increased'/'decreased' -- and rho before/after); if the
    lift cycle's own action already balances (per `_update_admm_penalties`'s
    "balanced in that same update" rule), that IS the first ESS balancing
    action and is reported as such, not skipped;
  - every SUBSEQUENT open-loop firing after that first action, explicitly
    labelled non-predictive: the real trajectory would itself change once
    rho differs from the donor's observed path, and this open-loop replay
    cannot follow that (same caveat `p515_s38_balancing_replay.py`'s own
    counterfactual section states).

## Usage (write-once; refuses to overwrite an existing output file)

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python \\
        p515_s39_d_policy_replay.py <trajectory_g_json_path> <output_dir>

Writes `<output_dir>/d_policy_replay_on_<label>.json`, `<label>` derived
from the trajectory file's own basename (e.g. `g_s38_A_tau0.json` ->
`s38_A_tau0`) -- so repeated calls with the SAME `output_dir` but DIFFERENT
trajectory paths (e.g. arm A's and arm B's) never collide, each individual
output file is still write-once.

This worker task's own two invocations (arm A's and arm B's completed s38
trajectories), per the W1 task list:

    .../bin/python p515_s39_d_policy_replay.py \\
        data/SRP1/Results/P515S38_A_TAU0_run/g_s38_A_tau0.json \\
        data/SRP1/Results/P515S39/d_policy_replay_on_s38 \\
        > data/SRP1/Results/P515S39/d_policy_replay_on_s38/replay_A_run.log 2>&1

    .../bin/python p515_s39_d_policy_replay.py \\
        data/SRP1/Results/P515S38_B_PFBAL_run/g_s38_B_pfbal.json \\
        data/SRP1/Results/P515S39/d_policy_replay_on_s38 \\
        > data/SRP1/Results/P515S39/d_policy_replay_on_s38/replay_B_run.log 2>&1
"""

import hashlib
import json
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p513_solve_profile_guard import SolveProfileGuard
from admm_parameters import ADMMParameters
from shared_resources_planning import _update_admm_penalties, _init_admm_freeze_state
import p515_s38_balancing_replay as B38  # noqa: E402 -- reused model builders / metric adapters

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CASE_FILE = os.path.join(REPO_ROOT, 'data', 'SRP1', 'SRP1_params.json')

CHANNELS = ('v', 'pf', 'ess')

# -- arm D's own configuration (spec v10 `arms.s39_D`), applied REGARDLESS
#    of the donor trajectory's own configuration --------------------------
D_EXEMPT_CHANNELS = []
D_EXEMPT_UNTIL = {'ess': {'dual_ratio_below': 1.0, 'consecutive_cycles': 5}}
D_FREEZE_AFTER_UNCHANGED_CYCLES = 10
D_FREEZE_BACKSTOP_CYCLE = 200
D_TAU = 0.0

MATCHED_CYCLES = [1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 131, 140, 145, 150, 175, 200, 250, 300]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _label_from_trajectory_path(traj_path):
    base = os.path.basename(traj_path)
    if base.startswith('g_') and base.endswith('.json'):
        return base[len('g_'):-len('.json')]
    return os.path.splitext(base)[0]


def build_d_policy_params(admm_block_template):
    admm_block = deepcopy(admm_block_template)
    admm_block['penalty_update'] = deepcopy(admm_block['penalty_update'])
    admm_block['penalty_update']['balancing_exempt_channels'] = list(D_EXEMPT_CHANNELS)
    admm_block['penalty_update']['balancing_exempt_until'] = deepcopy(D_EXEMPT_UNTIL)
    admm_block['penalty_update']['freeze_after_unchanged_cycles'] = D_FREEZE_AFTER_UNCHANGED_CYCLES
    admm_block['penalty_update']['freeze_backstop_cycle'] = D_FREEZE_BACKSTOP_CYCLE
    admm_block['proximal_regularization'] = deepcopy(admm_block.get('proximal_regularization', {}))
    admm_block['proximal_regularization']['tso'] = dict(admm_block['proximal_regularization'].get('tso', {}))
    admm_block['proximal_regularization']['tso']['tau'] = D_TAU
    params = ADMMParameters()
    params.read_parameters_from_file(admm_block)
    assert params.penalty_update['balancing_exempt_until'] == D_EXEMPT_UNTIL
    assert params.penalty_update['balancing_exempt_channels'] == []
    return params


def replay_d_policy(traj_path, admm_block_template):
    with open(traj_path) as handle:
        traj = json.load(handle)['cycle_trajectory']

    params = build_d_policy_params(admm_block_template)
    freeze_state = _init_admm_freeze_state()

    first_cycle = traj[0]
    rho_v, rho_pf, rho_ess = first_cycle['rho_v_before'], first_cycle['rho_pf_before'], first_cycle['rho_ess_before']
    gamma_v, gamma_pf, gamma_ess = (
        first_cycle['gamma_v_before'], first_cycle['gamma_pf_before'], first_cycle['gamma_ess_before'])

    tso = B38.make_tso_model(rho_v, rho_pf, rho_ess, gamma_v, gamma_pf, gamma_ess)
    dso = B38.make_dso_model(rho_v, rho_pf, rho_ess)
    esso = B38.make_esso_model(rho_ess)
    tso_model = {0: {0: tso}}
    dso_models = {0: {0: {0: dso}}}
    esso_model = {0: esso}

    per_cycle_records = []
    predicted_lift_cycle = None
    ess_streak_trajectory = []
    first_ess_action_after_lift = None
    subsequent_ess_firings_after_lift = []

    for cyc in traj:
        cycle = cyc['cycle']
        boyd_metrics = B38.boyd_metrics_from_cycle(cyc)
        residual_metrics = B38.residual_metrics_from_cycle(cyc)

        actions, before, after, before_gamma, after_gamma, rho_freeze_active, freeze_state = (
            _update_admm_penalties(
                tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, params,
                iter=cycle, allow_update=bool(cyc['local_solves_ok']), freeze_state=freeze_state))

        ess_state = freeze_state['ess']
        ess_streak_trajectory.append({
            'cycle': cycle, 'streak': ess_state['exempt_until_streak'],
            'lifted': ess_state['exempt_until_lifted'], 'action': actions['ess'],
        })
        if predicted_lift_cycle is None and ess_state['exempt_until_lift_cycle'] is not None:
            predicted_lift_cycle = ess_state['exempt_until_lift_cycle']

        record = {
            'cycle': cycle, 'action_ess': actions['ess'],
            'rho_ess_before': before['ess'], 'rho_ess_after': after['ess'],
            'donor_boyd_ess_dual_ratio': boyd_metrics['ess']['dual_ratio'],
            'donor_boyd_ess_primal_ratio': boyd_metrics['ess']['primal_ratio'],
            'donor_boyd_ess_dual_ratio_balance': boyd_metrics['ess']['dual_ratio_balance'],
            'ess_exempt_until_streak': ess_state['exempt_until_streak'],
            'ess_exempt_until_lifted': ess_state['exempt_until_lifted'],
            'ess_exempt_until_lift_cycle': ess_state['exempt_until_lift_cycle'],
        }
        per_cycle_records.append(record)

        if (predicted_lift_cycle is not None and cycle >= predicted_lift_cycle
                and actions['ess'] in ('increased', 'decreased')):
            firing = {
                'cycle': cycle, 'direction': actions['ess'],
                'rho_before': before['ess'], 'rho_after': after['ess'],
                'is_lift_cycle': (cycle == predicted_lift_cycle),
            }
            if first_ess_action_after_lift is None:
                first_ess_action_after_lift = firing
            else:
                subsequent_ess_firings_after_lift.append(firing)

    n_cycles = len(traj)
    matched = [c for c in MATCHED_CYCLES if c <= n_cycles]
    matched_series = [r for r in per_cycle_records if r['cycle'] in matched or r['cycle'] == n_cycles]

    return {
        'trajectory_path': os.path.relpath(traj_path, REPO_ROOT),
        'trajectory_sha256': sha256_of(traj_path),
        'n_cycles': n_cycles,
        'd_policy_config': {
            'balancing_exempt_channels': params.penalty_update['balancing_exempt_channels'],
            'balancing_exempt_until': params.penalty_update['balancing_exempt_until'],
            'freeze_after_unchanged_cycles': params.penalty_update['freeze_after_unchanged_cycles'],
            'freeze_backstop_cycle': params.penalty_update['freeze_backstop_cycle'],
            'tau': params.proximal_regularization['tso']['tau'],
        },
        'PREDICTED_ESS_LIFT_CYCLE': predicted_lift_cycle,
        'FIRST_ESS_BALANCING_ACTION_AFTER_LIFT': first_ess_action_after_lift,
        'subsequent_open_loop_firings_after_lift_NOT_PREDICTIVE': subsequent_ess_firings_after_lift,
        'n_subsequent_firings': len(subsequent_ess_firings_after_lift),
        'ess_streak_trajectory_matched_and_around_lift': (
            ess_streak_trajectory if predicted_lift_cycle is None else
            [r for r in ess_streak_trajectory
             if r['cycle'] in matched or abs(r['cycle'] - predicted_lift_cycle) <= 3]),
        'matched_cycle_series': matched_series,
        'terminal_cycle_state': per_cycle_records[-1],
        'caveat': ("Open-loop replay: boyd_metrics at every cycle are the OBSERVED ratios from "
                   "the DONOR (real) trajectory, never updated in response to this replay's own "
                   "counterfactual rho change -- there is no re-solve to produce them. Everything "
                   "reported after FIRST_ESS_BALANCING_ACTION_AFTER_LIFT (the subsequent firings) "
                   "is NOT predictive of what a real re-run of arm D would produce -- the real "
                   "trajectory would itself change once rho differs from the donor's observed "
                   "path, and this replay cannot follow that."),
    }


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    traj_path = os.path.abspath(argv[0])
    output_dir = os.path.abspath(argv[1])
    if not os.path.exists(traj_path):
        raise FileNotFoundError(traj_path)

    label = _label_from_trajectory_path(traj_path)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'd_policy_replay_on_{label}.json')
    if os.path.exists(out_path):
        raise FileExistsError(
            f'{out_path} already exists -- this script refuses to overwrite a committed '
            f'output (write-once rule, CLAUDE.md).')

    guard = SolveProfileGuard([], label='p515_s39_d_policy_replay').install()
    try:
        with open(CASE_FILE) as handle:
            admm_block_template = json.load(handle)['admm']

        print(f'=== D-POLICY OPEN-LOOP REPLAY on {os.path.relpath(traj_path, REPO_ROOT)} ===')
        result = replay_d_policy(traj_path, admm_block_template)
        print(f"predicted ESS lift cycle: {result['PREDICTED_ESS_LIFT_CYCLE']}")
        first_action = result['FIRST_ESS_BALANCING_ACTION_AFTER_LIFT']
        if first_action is not None:
            print(f"first ESS balancing action after lift: cycle {first_action['cycle']} "
                  f"({first_action['direction']}, rho {first_action['rho_before']:.6g} -> "
                  f"{first_action['rho_after']:.6g}, is_lift_cycle={first_action['is_lift_cycle']})")
        else:
            print(f"no ESS balancing action within {result['n_cycles']} cycles "
                  f"(lift_cycle={result['PREDICTED_ESS_LIFT_CYCLE']})")
        print(f"n subsequent (non-predictive) firings: {result['n_subsequent_firings']}")

        output = {
            'stage': ("P5.15 Addendum 21 (s39 preparation worker task) -- zero-solve open-loop "
                     "replay of arm D's two-phase ESS policy on a donor trajectory"),
            'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 21',
                          'data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json'],
            'label': label, 'output_path': os.path.relpath(out_path, REPO_ROOT),
            'result': result,
        }

        failures = guard.verify(expected_solves=0)
        output['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
        if failures:
            raise RuntimeError(f'SolveProfileGuard verification failed: {failures}')

        with open(out_path, 'w') as handle:
            json.dump(output, handle, indent=1, default=str)
        print(f'Wrote {out_path}')
        print('SolveProfileGuard: 0 solves, 0 execs -- verified.')
    finally:
        guard.uninstall()
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
