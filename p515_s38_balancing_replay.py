"""
P5.15 Addendum 20, item (a) -- zero-solve replay of the residual-balancing rule
on the saved trajectories: would rho_pf have been lowered without the cycle-60
backstop, and when?

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 20 ("(a) zero-solve replay of
the balancing rule on the saved trajectories (would rho_pf have been lowered,
and when)") and frozen spec v9
data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json (read-only;
never staged by this script).

## Method: the REAL production function, not a reimplementation

This script drives the actual production decision function
`shared_resources_planning._update_admm_penalties` (see that function's own
docstring, ~line 6581 onward, for the full precedence rules: exempt > frozen
(legacy-cycle / streak / backstop) > not-adaptive > failure-hold > increase /
decrease / dead-band; the residual-balance thresholds used are
`boyd_metrics[group]['primal_ratio']` and `boyd_metrics[group]['dual_ratio_balance']`,
NOT the legacy tolerance-normalized ratios, which are diagnostic-only).

It is fed with:
  - lightweight stand-in Pyomo `ConcreteModel` objects exposing exactly the
    attributes `_update_admm_penalties` / `_get_admm_penalty_summary` /
    `_get_admm_gamma_summary` read or write: `rho_v`, `rho_pf`, `rho_ess`
    (mutable `Param`s on one TSO model, one DSO model, one ESSO model --
    `_get_admm_penalty_summary` only ever averages across models, so one of
    each is sufficient and produces bit-identical output to N of each, given
    identical rho); `prox_gamma_v/pf/ess` (mutable `Param`s, TSO model only,
    read/written under `gamma_policy == 'tied_to_rho'`).
  - `boyd_metrics[group]` and `residual_metrics` dicts rebuilt, per cycle,
    directly from the committed `cycle_trajectory` entries of each run's
    `g_*.json` (fields `boyd_{group}_r/s/eps_pri/eps_dual/primal_ratio/
    dual_ratio/dual_ratio_balance` and `primal_{group}[_mean]`,
    `dual_{group}_mean` respectively -- these are the exact same dicts the
    production ADMM loop builds every cycle and that got serialized into the
    trajectory).
  - `params`, a real `admm_parameters.ADMMParameters` instance loaded via its
    own `read_parameters_from_file` from a deep copy of
    `data/SRP1/SRP1_params.json`'s `admm` block, with two keys varied by this
    script only: `penalty_update.balancing_exempt_channels` (must be `['ess']`
    to reproduce the s37 arms, `[]` for run 1 -- verified against each
    trajectory's own `balancing_exempt_{v,pf,ess}` fields) and, for the
    COUNTERFACTUAL only, `penalty_update.freeze_backstop_cycle` (60 -> 200,
    the Addendum 20 rule; `freeze_after_unchanged_cycles` is left at the case
    file's 10, already the Addendum 20 value).
  - `freeze_state`, the real `_init_admm_freeze_state()` dict, carried
    cycle-to-cycle exactly as the production ADMM loop carries it (this is
    where the streak/ever-acted/frozen/exempt bookkeeping lives; it is NOT
    reset every cycle).

No production, case-file, or harness file is edited. No Pyomo/IPOPT solve is
ever attempted: `SolveProfileGuard([])` is installed for the whole script
(permitted call-site list empty) and `verify(expected_solves=0)` is asserted
before writing any output.

## VALIDATION (mandatory gate, run before any counterfactual)

Cycle by cycle, in trajectory order, for each of the three runs: the stand-in
models' rho_v/rho_pf/rho_ess/gamma_* Params are set to that cycle's OWN
recorded `rho_{c}_before` / `gamma_{c}_before` (ground truth, not carried from
the previous cycle's computed output -- this isolates any mismatch to the
single cycle that produced it), `_update_admm_penalties` is called with the
run's ACTUAL configuration (`iter=cycle`, `allow_update=local_solves_ok`,
`freeze_backstop_cycle=60`, the run's own `balancing_exempt_channels`), and
the returned action / rho_after / gamma_after / `freeze_state[group]['frozen']`
/ `['unchanged_streak']` / `['at_clamp']` are compared against the trajectory's
own `rho_{c}_action`, `rho_{c}_after`, `gamma_{c}_after`, `rho_frozen_{c}`,
`rho_unchanged_streak_{c}`, `rho_at_clamp_{c}` for every cycle and every
channel (rho/gamma compared with `math.isclose(rel_tol=1e-9, abs_tol=1e-12)`;
everything else by exact equality). ANY mismatch is recorded; if there is even
one, this script prints and writes a FAILED validation verdict and the
counterfactual section is still computed (so the evidence is not lost) but is
explicitly labelled NOT TRUSTED.

## COUNTERFACTUAL

Same three trajectories. Same `boyd_metrics`/`residual_metrics` inputs, read
from the REAL (observed) trajectory at every cycle (open-loop: the residual
ratios are never updated in response to a counterfactual rho change -- there
is no re-solve to produce them). `freeze_backstop_cycle` is changed from 60 to
200; `freeze_after_unchanged_cycles` stays 10 (already Addendum 20's value in
the case file). Unlike validation, rho/gamma ARE carried forward cycle-to-cycle
from this script's OWN prior counterfactual output (not from the trajectory),
starting from cycle 1's actual `rho_{c}_before` (the real run's initial
condition, identical for both the real and counterfactual paths). This
reproduces the rho path the balancing rule would have produced under the
Addendum 20 freeze rule, given the ratios actually observed.

Per run and channel this script reports: the first cycle at which the
counterfactual rule takes a real action (increase or decrease) that differs
from the actual trajectory's action at that cycle; for PF specifically, the
first DECREASE cycle and the `primal_ratio` / `dual_ratio_balance` values that
triggered it; and whether the channel is frozen by cycle 200 (open-loop). All
actions after the FIRST divergence from the real trajectory are reported
separately as "subsequent open-loop firings" and explicitly flagged as not
predictive of what a real re-run would do (the real trajectory would change
once rho differs, and this replay cannot follow that).

## PF ratio series

For all three runs, `dual_ratio_balance / primal_ratio` for the PF channel, at
the matched cycles 1, 2, 5, 10, 20, 30, 50, 60, 75, 100, 125, 150, 200, 226,
300, 400, 477 (only those <= the run's cycle count), plus the first cycle (over
the FULL trajectory, not just the matched list) at which this ratio exceeds
3.0 (the PF decrease threshold, `residual_balance_ratio_pf_decrease`) and 5.0
(the general `residual_balance_ratio`, PF's increase side reciprocal).

## Outputs (write-once; this script refuses to overwrite an existing output)

  data/SRP1/Results/P515S38/balancing_replay/validation.json
  data/SRP1/Results/P515S38/balancing_replay/counterfactual.json
  data/SRP1/Results/P515S38/balancing_replay/evidence_manifest_sha256.json

Run as:
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python \\
      p515_s38_balancing_replay.py \\
      > data/SRP1/Results/P515S38/balancing_replay/run.log 2>&1
"""

import hashlib
import json
import os
import sys
from copy import deepcopy
from math import isclose

import pyomo.environ as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p513_solve_profile_guard import SolveProfileGuard
from admm_parameters import ADMMParameters
from shared_resources_planning import _update_admm_penalties, _init_admm_freeze_state


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CASE_FILE = os.path.join(REPO_ROOT, 'data', 'SRP1', 'SRP1_params.json')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S38', 'balancing_replay')

RUNS = {
    'run1_s35ref': {
        'trajectory_path': os.path.join(
            REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S35_REF_run', 'g_baseline.json'),
        'balancing_exempt_channels': [],
        'label': 'run 1 (s35ref) -- ESS not exempt, 477 cycles, backstop 60 (actual)',
    },
    's37_rho0p01': {
        'trajectory_path': os.path.join(
            REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S37_RHO0P01_run', 'g_s37_rho0p01.json'),
        'balancing_exempt_channels': ['ess'],
        'label': 's37 arm rho_ess=0.01 -- ESS exempt, 150 cycles, backstop 60 (actual)',
    },
    's37_rho0p001': {
        'trajectory_path': os.path.join(
            REPO_ROOT, 'data', 'SRP1', 'Results', 'P515S37_RHO0P001_run', 'g_s37_rho0p001.json'),
        'balancing_exempt_channels': ['ess'],
        'label': 's37 arm rho_ess=0.001 -- ESS exempt, 150 cycles, backstop 60 (actual)',
    },
}

CHANNELS = ('v', 'pf', 'ess')
MATCHED_CYCLES = [1, 2, 5, 10, 20, 30, 50, 60, 75, 100, 125, 150, 200, 226, 300, 400, 477]
ACTUAL_BACKSTOP = 60
COUNTERFACTUAL_BACKSTOP = 200
PF_DECREASE_THRESHOLD = 3.0
PF_INCREASE_THRESHOLD = 5.0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_case_admm_block():
    with open(CASE_FILE, 'r') as f:
        case_data = json.load(f)
    return case_data['admm']


def build_params(admm_block_template, balancing_exempt_channels, freeze_backstop_cycle):
    """Real `ADMMParameters`, loaded via the real `read_parameters_from_file`,
    from a deep copy of the case file's `admm` block with exactly the two
    keys documented in the module docstring varied."""
    admm_block = deepcopy(admm_block_template)
    admm_block['penalty_update'] = deepcopy(admm_block['penalty_update'])
    admm_block['penalty_update']['balancing_exempt_channels'] = list(balancing_exempt_channels)
    admm_block['penalty_update']['freeze_backstop_cycle'] = freeze_backstop_cycle
    params = ADMMParameters()
    params.read_parameters_from_file(admm_block)
    return params


def make_tso_model(rho_v, rho_pf, rho_ess, gamma_v, gamma_pf, gamma_ess):
    m = pe.ConcreteModel()
    m.rho_v = pe.Param(initialize=rho_v, mutable=True)
    m.rho_pf = pe.Param(initialize=rho_pf, mutable=True)
    m.rho_ess = pe.Param(initialize=rho_ess, mutable=True)
    m.prox_gamma_v = pe.Param(initialize=gamma_v, mutable=True)
    m.prox_gamma_pf = pe.Param(initialize=gamma_pf, mutable=True)
    m.prox_gamma_ess = pe.Param(initialize=gamma_ess, mutable=True)
    return m


def make_dso_model(rho_v, rho_pf, rho_ess):
    m = pe.ConcreteModel()
    m.rho_v = pe.Param(initialize=rho_v, mutable=True)
    m.rho_pf = pe.Param(initialize=rho_pf, mutable=True)
    m.rho_ess = pe.Param(initialize=rho_ess, mutable=True)
    return m


def make_esso_model(rho_ess):
    m = pe.ConcreteModel()
    m.rho = pe.Param(initialize=rho_ess, mutable=True)
    return m


def set_tso_rho(model, rho_v, rho_pf, rho_ess):
    model.rho_v.set_value(rho_v)
    model.rho_pf.set_value(rho_pf)
    model.rho_ess.set_value(rho_ess)


def set_tso_gamma(model, gamma_v, gamma_pf, gamma_ess):
    model.prox_gamma_v.set_value(gamma_v)
    model.prox_gamma_pf.set_value(gamma_pf)
    model.prox_gamma_ess.set_value(gamma_ess)


def set_dso_rho(model, rho_v, rho_pf, rho_ess):
    model.rho_v.set_value(rho_v)
    model.rho_pf.set_value(rho_pf)
    model.rho_ess.set_value(rho_ess)


def set_esso_rho(model, rho_ess):
    model.rho.set_value(rho_ess)


def boyd_metrics_from_cycle(cyc):
    metrics = {}
    for g in CHANNELS:
        metrics[g] = {
            'r': cyc[f'boyd_{g}_r'],
            's': cyc[f'boyd_{g}_s'],
            'eps_pri': cyc[f'boyd_{g}_eps_pri'],
            'eps_dual': cyc[f'boyd_{g}_eps_dual'],
            'primal_ratio': cyc[f'boyd_{g}_primal_ratio'],
            'dual_ratio': cyc[f'boyd_{g}_dual_ratio'],
            'dual_ratio_balance': cyc[f'boyd_{g}_dual_ratio_balance'],
        }
    return metrics


def residual_metrics_from_cycle(cyc):
    residual_metrics = {'primal': {}, 'dual': {}}
    for g in CHANNELS:
        residual_metrics['primal'][g] = cyc[f'primal_{g}']
        residual_metrics['primal'][f'{g}_mean'] = cyc[f'primal_{g}_mean']
        residual_metrics['dual'][f'{g}_mean'] = cyc[f'dual_{g}_mean']
    return residual_metrics


def rho_close(a, b):
    return isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def run_validation(run_name, run_cfg, admm_block_template):
    with open(run_cfg['trajectory_path'], 'r') as f:
        traj = json.load(f)['cycle_trajectory']

    # Cross-check the run's own recorded exemption flags against the
    # configuration this script is about to assert for it.
    exempt_declared = set(run_cfg['balancing_exempt_channels'])
    for g in CHANNELS:
        key = f'balancing_exempt_{g}'
        if key in traj[0]:
            recorded_exempt = bool(traj[0][key])
            declared_exempt = g in exempt_declared
            if recorded_exempt != declared_exempt:
                raise AssertionError(
                    f'{run_name}: trajectory records balancing_exempt_{g}='
                    f'{recorded_exempt} but this script declared {declared_exempt}')

    params = build_params(admm_block_template, run_cfg['balancing_exempt_channels'], ACTUAL_BACKSTOP)
    freeze_state = _init_admm_freeze_state()

    tso = make_tso_model(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    dso = make_dso_model(0.0, 0.0, 0.0)
    esso = make_esso_model(0.0)
    tso_model = {0: {0: tso}}
    dso_models = {0: {0: {0: dso}}}
    esso_model = {0: esso}

    per_cycle_records = []
    mismatches = []

    for cyc in traj:
        cycle = cyc['cycle']

        rho_v_b = cyc['rho_v_before']
        rho_pf_b = cyc['rho_pf_before']
        rho_ess_b = cyc['rho_ess_before']
        gamma_v_b = cyc['gamma_v_before']
        gamma_pf_b = cyc['gamma_pf_before']
        gamma_ess_b = cyc['gamma_ess_before']

        set_tso_rho(tso, rho_v_b, rho_pf_b, rho_ess_b)
        set_tso_gamma(tso, gamma_v_b, gamma_pf_b, gamma_ess_b)
        set_dso_rho(dso, rho_v_b, rho_pf_b, rho_ess_b)
        set_esso_rho(esso, rho_ess_b)

        boyd_metrics = boyd_metrics_from_cycle(cyc)
        residual_metrics = residual_metrics_from_cycle(cyc)

        actions, before, after, before_gamma, after_gamma, rho_freeze_active, freeze_state = (
            _update_admm_penalties(
                tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, params,
                iter=cycle, allow_update=bool(cyc['local_solves_ok']), freeze_state=freeze_state))

        cycle_record = {'cycle': cycle, 'channels': {}}
        for g in CHANNELS:
            expected_action = cyc[f'rho_{g}_action']
            expected_after = cyc[f'rho_{g}_after']
            expected_gamma_after = cyc[f'gamma_{g}_after']
            expected_frozen = bool(cyc[f'rho_frozen_{g}'])
            expected_streak = cyc[f'rho_unchanged_streak_{g}']
            expected_at_clamp = bool(cyc[f'rho_at_clamp_{g}'])

            got_action = actions[g]
            got_after = after[g]
            got_gamma_after = after_gamma[g]
            got_frozen = freeze_state[g]['frozen']
            got_streak = freeze_state[g]['unchanged_streak']
            got_at_clamp = freeze_state[g]['at_clamp']

            channel_mismatches = []
            if got_action != expected_action:
                channel_mismatches.append(
                    f"action: got '{got_action}' expected '{expected_action}'")
            if not rho_close(got_after, expected_after):
                channel_mismatches.append(
                    f'rho_after: got {got_after!r} expected {expected_after!r}')
            if got_gamma_after is not None and not rho_close(got_gamma_after, expected_gamma_after):
                channel_mismatches.append(
                    f'gamma_after: got {got_gamma_after!r} expected {expected_gamma_after!r}')
            if got_frozen != expected_frozen:
                channel_mismatches.append(
                    f'frozen: got {got_frozen} expected {expected_frozen}')
            if got_streak != expected_streak:
                channel_mismatches.append(
                    f'unchanged_streak: got {got_streak} expected {expected_streak}')
            if got_at_clamp != expected_at_clamp:
                channel_mismatches.append(
                    f'at_clamp: got {got_at_clamp} expected {expected_at_clamp}')

            cycle_record['channels'][g] = {
                'action': got_action,
                'rho_after': got_after,
                'gamma_after': got_gamma_after,
                'frozen': got_frozen,
                'unchanged_streak': got_streak,
                'at_clamp': got_at_clamp,
                'mismatches': channel_mismatches,
            }
            if channel_mismatches:
                mismatches.append({'run': run_name, 'cycle': cycle, 'channel': g,
                                    'mismatches': channel_mismatches})

        per_cycle_records.append(cycle_record)

    return {
        'run': run_name,
        'label': run_cfg['label'],
        'n_cycles': len(traj),
        'passed': len(mismatches) == 0,
        'n_mismatches': len(mismatches),
        'mismatches': mismatches,
        'note': ('per-cycle re-seeding from the trajectory\'s own rho_before/gamma_before '
                 '(ground truth), so a mismatch is attributable to the single cycle that '
                 'produced it, not to accumulated drift'),
    }


def run_counterfactual(run_name, run_cfg, admm_block_template):
    with open(run_cfg['trajectory_path'], 'r') as f:
        traj = json.load(f)['cycle_trajectory']

    params = build_params(admm_block_template, run_cfg['balancing_exempt_channels'],
                           COUNTERFACTUAL_BACKSTOP)
    freeze_state = _init_admm_freeze_state()

    first_cycle = traj[0]
    rho_v = first_cycle['rho_v_before']
    rho_pf = first_cycle['rho_pf_before']
    rho_ess = first_cycle['rho_ess_before']
    gamma_v = first_cycle['gamma_v_before']
    gamma_pf = first_cycle['gamma_pf_before']
    gamma_ess = first_cycle['gamma_ess_before']

    tso = make_tso_model(rho_v, rho_pf, rho_ess, gamma_v, gamma_pf, gamma_ess)
    dso = make_dso_model(rho_v, rho_pf, rho_ess)
    esso = make_esso_model(rho_ess)
    tso_model = {0: {0: tso}}
    dso_models = {0: {0: {0: dso}}}
    esso_model = {0: esso}

    per_cycle_records = []
    # First cycle, per channel, at which the counterfactual action DIFFERS
    # from the real trajectory's recorded action at that cycle.
    first_divergence = {g: None for g in CHANNELS}
    # First cycle, per channel, at which the counterfactual rule itself takes
    # a real 'increased'/'decreased' action (regardless of what the real
    # trajectory did).
    first_real_action = {g: None for g in CHANNELS}
    first_pf_decrease = None
    subsequent_firings = {g: [] for g in CHANNELS}

    for cyc in traj:
        cycle = cyc['cycle']

        boyd_metrics = boyd_metrics_from_cycle(cyc)
        residual_metrics = residual_metrics_from_cycle(cyc)

        actions, before, after, before_gamma, after_gamma, rho_freeze_active, freeze_state = (
            _update_admm_penalties(
                tso_model, dso_models, esso_model, residual_metrics, boyd_metrics, params,
                iter=cycle, allow_update=bool(cyc['local_solves_ok']), freeze_state=freeze_state))

        cycle_record = {'cycle': cycle, 'channels': {}}
        for g in CHANNELS:
            actual_action = cyc[f'rho_{g}_action']
            got_action = actions[g]

            cycle_record['channels'][g] = {
                'action': got_action,
                'rho_before': before[g],
                'rho_after': after[g],
                'gamma_after': after_gamma[g],
                'frozen': freeze_state[g]['frozen'],
                'unchanged_streak': freeze_state[g]['unchanged_streak'],
                'at_clamp': freeze_state[g]['at_clamp'],
                'actual_action_at_this_cycle': actual_action,
                'boyd_primal_ratio': boyd_metrics[g]['primal_ratio'],
                'boyd_dual_ratio_balance': boyd_metrics[g]['dual_ratio_balance'],
            }

            if got_action in ('increased', 'decreased'):
                if first_real_action[g] is None:
                    first_real_action[g] = {
                        'cycle': cycle, 'action': got_action,
                        'rho_before': before[g], 'rho_after': after[g],
                        'primal_ratio': boyd_metrics[g]['primal_ratio'],
                        'dual_ratio_balance': boyd_metrics[g]['dual_ratio_balance'],
                    }
                else:
                    subsequent_firings[g].append({
                        'cycle': cycle, 'action': got_action,
                        'rho_before': before[g], 'rho_after': after[g],
                        'primal_ratio': boyd_metrics[g]['primal_ratio'],
                        'dual_ratio_balance': boyd_metrics[g]['dual_ratio_balance'],
                    })
                if g == 'pf' and got_action == 'decreased' and first_pf_decrease is None:
                    first_pf_decrease = {
                        'cycle': cycle,
                        'rho_before': before[g], 'rho_after': after[g],
                        'primal_ratio': boyd_metrics[g]['primal_ratio'],
                        'dual_ratio_balance': boyd_metrics[g]['dual_ratio_balance'],
                        'dual_over_primal_ratio': (
                            boyd_metrics[g]['dual_ratio_balance'] / boyd_metrics[g]['primal_ratio']
                            if boyd_metrics[g]['primal_ratio'] else None),
                    }

            if first_divergence[g] is None and got_action != actual_action:
                first_divergence[g] = {
                    'cycle': cycle,
                    'counterfactual_action': got_action,
                    'actual_action': actual_action,
                    'rho_before': before[g], 'rho_after': after[g],
                }

        per_cycle_records.append(cycle_record)

    channel_summary = {}
    for g in CHANNELS:
        channel_summary[g] = {
            'first_real_action': first_real_action[g],
            'first_divergence_from_actual_trajectory': first_divergence[g],
            'subsequent_open_loop_firings_not_predictive': subsequent_firings[g],
            'n_subsequent_firings': len(subsequent_firings[g]),
            'frozen_by_200_open_loop': per_cycle_records[min(199, len(per_cycle_records) - 1)]['channels'][g]['frozen']
                if len(per_cycle_records) >= 1 else None,
            'terminal_state': per_cycle_records[-1]['channels'][g],
        }

    return {
        'run': run_name,
        'label': run_cfg['label'],
        'n_cycles': len(traj),
        'freeze_rule': {
            'freeze_after_unchanged_cycles': params.penalty_update['freeze_after_unchanged_cycles'],
            'freeze_backstop_cycle': params.penalty_update['freeze_backstop_cycle'],
        },
        'channel_summary': channel_summary,
        'pf_first_decrease': first_pf_decrease,
        'per_cycle': per_cycle_records,
        'caveat': ("Open-loop replay: boyd_metrics at every cycle are the OBSERVED ratios "
                   "from the real (backstop-60) trajectory, never updated in response to a "
                   "counterfactual rho change. Everything reported after "
                   "'first_divergence_from_actual_trajectory' is not predictive of what a real "
                   "re-run would produce -- the real trajectory would itself change once rho "
                   "differs from the observed path, and this replay cannot follow that."),
    }


def pf_ratio_series(run_name, run_cfg):
    with open(run_cfg['trajectory_path'], 'r') as f:
        traj = json.load(f)['cycle_trajectory']

    by_cycle = {cyc['cycle']: cyc for cyc in traj}
    n_cycles = len(traj)

    series = {}
    for c in MATCHED_CYCLES:
        if c in by_cycle:
            cyc = by_cycle[c]
            primal = cyc['boyd_pf_primal_ratio']
            dual_bal = cyc['boyd_pf_dual_ratio_balance']
            ratio = dual_bal / primal if primal else None
            series[c] = {
                'boyd_pf_primal_ratio': primal,
                'boyd_pf_dual_ratio_balance': dual_bal,
                'dual_ratio_balance_over_primal_ratio': ratio,
            }

    first_exceeds_3 = None
    first_exceeds_5 = None
    for cyc in traj:
        primal = cyc['boyd_pf_primal_ratio']
        dual_bal = cyc['boyd_pf_dual_ratio_balance']
        if not primal:
            continue
        ratio = dual_bal / primal
        if first_exceeds_3 is None and ratio > PF_DECREASE_THRESHOLD:
            first_exceeds_3 = {'cycle': cyc['cycle'], 'ratio': ratio}
        if first_exceeds_5 is None and ratio > PF_INCREASE_THRESHOLD:
            first_exceeds_5 = {'cycle': cyc['cycle'], 'ratio': ratio}
        if first_exceeds_3 is not None and first_exceeds_5 is not None:
            break

    return {
        'run': run_name,
        'n_cycles': n_cycles,
        'matched_cycle_series': series,
        'first_cycle_ratio_exceeds_3p0_pf_decrease_threshold': first_exceeds_3,
        'first_cycle_ratio_exceeds_5p0_general_threshold': first_exceeds_5,
    }


def main():
    guard = SolveProfileGuard([], label='p515_s38_balancing_replay').install()
    try:
        os.makedirs(OUT_DIR, exist_ok=True)

        validation_path = os.path.join(OUT_DIR, 'validation.json')
        counterfactual_path = os.path.join(OUT_DIR, 'counterfactual.json')
        for p in (validation_path, counterfactual_path):
            if os.path.exists(p):
                raise FileExistsError(
                    f'{p} already exists -- this script refuses to overwrite a committed output '
                    f'(write-once rule, CLAUDE.md).')

        admm_block_template = load_case_admm_block()

        print('=== VALIDATION (actual configuration, backstop=60) ===')
        validation_results = {}
        overall_passed = True
        for run_name, run_cfg in RUNS.items():
            print(f'-- {run_name}: {run_cfg["label"]}')
            result = run_validation(run_name, run_cfg, admm_block_template)
            validation_results[run_name] = result
            overall_passed = overall_passed and result['passed']
            print(f'   n_cycles={result["n_cycles"]} passed={result["passed"]} '
                  f'n_mismatches={result["n_mismatches"]}')
            if not result['passed']:
                for m in result['mismatches'][:20]:
                    print(f'   MISMATCH cycle={m["cycle"]} channel={m["channel"]}: '
                          f'{m["mismatches"]}')

        validation_output = {
            'stage': 'P5.15 Addendum 20 item (a) -- balancing replay VALIDATION',
            'overall_passed': overall_passed,
            'runs': validation_results,
        }
        with open(validation_path, 'w') as f:
            json.dump(validation_output, f, indent=1)
        print(f'Wrote {validation_path}')

        print()
        print('=== COUNTERFACTUAL (freeze_after_unchanged_cycles=10, freeze_backstop_cycle=200) ===')
        if not overall_passed:
            print('VALIDATION DID NOT PASS -- counterfactual is computed for completeness but '
                  'is NOT TRUSTED. See validation.json for the mismatches.')

        counterfactual_results = {}
        pf_ratio_results = {}
        for run_name, run_cfg in RUNS.items():
            print(f'-- {run_name}: {run_cfg["label"]}')
            cf = run_counterfactual(run_name, run_cfg, admm_block_template)
            counterfactual_results[run_name] = cf
            for g in CHANNELS:
                fa = cf['channel_summary'][g]['first_real_action']
                if fa is not None:
                    print(f'   {g}: first real action at cycle {fa["cycle"]} '
                          f'({fa["action"]}, rho {fa["rho_before"]:.6g} -> {fa["rho_after"]:.6g})')
                else:
                    print(f'   {g}: no real action within {cf["n_cycles"]} cycles')
            pf_ratio_results[run_name] = pf_ratio_series(run_name, run_cfg)

        counterfactual_output = {
            'stage': 'P5.15 Addendum 20 item (a) -- balancing replay COUNTERFACTUAL',
            'validation_passed': overall_passed,
            'trusted': overall_passed,
            'freeze_rule_counterfactual': {
                'freeze_after_unchanged_cycles': 10,
                'freeze_backstop_cycle': COUNTERFACTUAL_BACKSTOP,
            },
            'runs': counterfactual_results,
            'pf_ratio_series': pf_ratio_results,
        }
        with open(counterfactual_path, 'w') as f:
            json.dump(counterfactual_output, f, indent=1)
        print(f'Wrote {counterfactual_path}')

        failures = guard.verify(expected_solves=0)
        if failures:
            raise RuntimeError(f'SolveProfileGuard verification failed: {failures}')
        print()
        print('SolveProfileGuard: 0 solves, 0 execs -- verified.')
        print(f'Overall validation passed: {overall_passed}')

    finally:
        guard.uninstall()


if __name__ == '__main__':
    main()
