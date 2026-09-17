"""P5.15 Addendum 21 - the oracle arms under the new certification bar
(s39_C, s39_D) - INDEPENDENT evaluation.

Frozen spec v10, data/SRP1/Results/P515S39/frozen_s39_oracle_spec_v10_f1b2b999.json.

Zero solves, SolveProfileGuard armed. Recomputes EVERYTHING from the arm's own
per-cycle trajectory (g_<arm>.json), its ess_entry_stride sidecar, its
pf_entry_stride sidecar and (arm D only) its NEW ess_exempt_until_state
sidecar -- never trusts boyd_terminal.json's own `stopped_by`/`converged_at_
cycle` fields (the v1/v2 defect s35ref/s35pt/s37/s38's evaluators found).
Reuses, BY IMPORT, the generic (non arm-specific) helpers already written for
s37/s38: `p515_s37_evaluate` (`_load_ess_stride_p_vectors`, `_step_series`,
`_oscillation_series`, `_oscillation_flag`, `_rho_gamma_trajectory`,
`_channel_ratio_trajectory`, `_efc_trajectory`, `_failures_by_tier`) and
`p515_s38_evaluate` (`_pf_stride_path`, `_load_pf_stride`, `_pf_decomposition_
at_cycle`, `_tso_instability_trigger` -- none of these read any s38-arm-
specific global; they operate purely on `run_dir`/`rows`/`entries`).

Every formula used below:

CERTIFICATION (spec v10 `certification_bar.certified_iff`): all three
channels (V, PF, ESS) inside their Boyd tolerances (primal_ratio <= 1 AND
dual_ratio <= 1) AND every local solve successful, for 10 CONSECUTIVE
cycles, within the cap of 300 -- derived STRUCTURALLY from the trajectory's
own per-row bookkeeping fields (`cycle_convergence` = `boyd_all_pass AND
local_solves_ok`, `consecutive_converged_cycles` = the running streak of
that field, BOTH computed by production itself,
`shared_resources_planning.py` ~2735-2741 -- NOT re-derived by scanning,
and NOT read from `boyd_terminal.json`'s own summary fields, which the
s35ref/s35pt/s37/s38 evaluators already found buggy for `required > 1`):
certified iff the LAST row has `cycle_convergence` True, `consecutive_
converged_cycles >= 10`, and `cycles_run <= 300`. Certification cycle = that
last row's own cycle (spec v10 `certification_bar.stopping_rule_
consequence`: with `minimum_consecutive_converged_cycles = 10` in force,
production's OWN Boyd stop coincides with the certification bar, so a
certified run's last row IS the certification cycle).

REPORTED, NOT GATED (spec v10 `certification_bar.reported_not_gated`):
  * per-channel terminal ratio: max(boyd_<c>_primal_ratio, boyd_<c>_dual_
    ratio) at the arm's last cycle, for c in {v, pf, ess};
  * rule-ten objective ratio: the last cycle's own `objective_change_ratio`
    (= objective_change_abs / objective_tolerance);
  * V/PF/ESS clamp flags: `rho_at_clamp_<c>` ever True over the whole run,
    per channel;
  * cost vs run 1 (spec v10 `certification_bar.cost_bar`): `bar = arm_max_
    objective_step_over_its_own_last_10_cycles + run1_max_step_last10`
    (the second term, 256.2581009864807, and `run1_cost`, 651039166.0347285,
    are READ from the frozen spec v10 JSON itself BY PATH, sha256-recorded
    -- never re-typed as literals here); `abs_diff = |Q_arm - Q_run1|`;
    `inside_bar = abs_diff <= bar`, reported (never gates CERTIFIED).

PF/V/ESS FIRST-PASS CYCLE: the first cycle at which `boyd_<channel>_
channel_pass` (= primal_pass AND dual_pass, same cycle) is True.

PREDICTIONS (spec v10 `predictions_recorded_in_advance`), scored ONLY from
that SAME arm's own evaluation (never cross-arm -- s38's evaluator wrongly
marked a prediction SCORED when only one of the two arms it needed was
present; s39's two predictions are per-arm by construction, so this
category of bug cannot recur):
  * s39_C: "PF first channel pass <= 131" AND "certification (10th
    consecutive all-pass cycle) by ~145" -- operationalized as
    certification cycle <= 150 (spec v10's own operationalization). SCORED
    iff arm C's evaluation is present; else `status: PENDING`.
  * s39_D: "certifies under the bar" AND "storage channel terminal ratio
    max(primal, dual) < 0.5 at certification" -- the LATTER is evaluated AT
    the certification cycle if certified, else at the terminal cycle (not
    certified -> both predictions score False, reported honestly). SCORED
    iff arm D's evaluation is present; else `status: PENDING`.

ADOPTION (spec v10 `predictions_recorded_in_advance.adoption`): "if D
certifies with storage terminal ratio < 0.5, D is the production oracle;
else C (if C certifies); if neither certifies, stop for review" -- marked
PENDING unless BOTH C and D evaluations exist (spec v10 `report`: "adoption
... PENDING unless both C and D evaluations exist").

TSO-INSTABILITY TRIGGER (v9 definitions, spec v10 `tso_instability_
monitoring`: "v9 trigger definitions... evaluated and REPORTED for C and D;
no fallback arm is authorized in this stage"), REUSED UNMODIFIED via
`p515_s38_evaluate._tso_instability_trigger` (fully generic: unrecovered TSO
failure; TSO failure rate > 2x run 1's 58/477; PF or ESS oscillation flag).

PER-ENTRY PF DECOMPOSITION: identical formula to s38's own (REUSED via
`p515_s38_evaluate._pf_decomposition_at_cycle`) at the s39 matched cycles
(1,2,5,10,20,30,50,75,100,125,131,140,145,150,175,200,250,300, capped at the
arm's own cap). LATE-PHASE PER-GROUP DECAY RATE (FIXED here -- s38's own
formula was a per-PAIR ratio treated as if matched cycles were evenly
spaced by 1, which they are not, e.g. ...,131,140,145,150,...): for each
grouping dimension and group value, and each CONSECUTIVE PAIR of matched
cycles (c0, c1) with gap = c1 - c0 falling inside the window [100, 200] (or
the latest available pair, if fewer than two matched cycles fall inside the
window -- `window_used_note` records the fallback), the TRUE per-cycle
geometric rate is `rate = (group_s(c1) / group_s(c0)) ** (1 / gap)` --
group_s(cycle) = sqrt(sum of entry s**2 within that group AT that matched
cycle) -- and the reported decay rate for that group is the geometric mean
of `rate` over every such pair. A rate < 1 means that group's PF residual
contribution is decaying PER CYCLE in the late phase; > 1 means it is
growing per cycle; this is now comparable across window pairs of different
spacing, which s38's own formula was not.

ESS EXEMPT-UNTIL STATE (arm D only, spec v10 `arms.s39_D`): read from the
harness's `ess_exempt_until_state_s39_D.jsonl` sidecar (one JSON line per
cycle, written by `p515_g_g1_g4_admm_gates.s39_exempt_until_capture_hooks`,
straight from the REAL `freeze_state` `_update_admm_penalties` returned that
cycle -- zero re-derivation here): the LIFT CYCLE (first line where
`channels.ess.lifted` is True) and EVERY ESS action from the trajectory's
own `rho_ess_action` field from the lift cycle on (RE-DERIVED independently
from the g_<arm>.json trajectory, not merely copied from the sidecar, as a
cross-check -- both are reported, and a mismatch between them is flagged,
never silently reconciled).

MONITORING (reported, not gated): local-solve/network failures by tier;
oscillation flags on PF and ESS; EFC/day, rho/gamma-per-channel-with-action-
label trajectories, channel ratio trajectories, all sampled at the s39
matched cycles; per-entry PF capture identity (re-verified here from the
sidecar's own per-cycle `identity_holds`, not merely trusted).

Usage: python p515_s39_evaluate.py [RUN_DIR] [--dry-run]
  RUN_DIR defaults to the s39_C arm's output root. Writes
  RUN_DIR/s39_evaluation.json (write-once; refuses to overwrite). The OTHER
  s39 arm, if its own output root currently exists, is read (never
  re-computed by mutating its directory) for the cross-arm ADOPTION section
  only; predictions are always scored per-arm (see above).
"""
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p515_s37_evaluate as E37  # noqa: E402 -- reused generic helpers, see module docstring
import p515_s38_evaluate as E38  # noqa: E402 -- reused generic helpers, see module docstring

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')  # run 1
CH = ('v', 'pf', 'ess')
CAP = 300
REQUIRED_CONSECUTIVE = 10
SPEC_V10_PATH = os.path.join(RES, 'P515S39', 'frozen_s39_oracle_spec_v10_f1b2b999.json')
MATCH = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 131, 140, 145, 150, 175, 200, 250, 300)
DECAY_WINDOW = (100, 200)
PF_GROUP_DIMENSIONS = ('node_id', 'year', 'day', 'power_type', 'period')

C_PF_FIRST_PASS_BOUND = 131
C_CERTIFICATION_CYCLE_BOUND = 150
D_STORAGE_TERMINAL_RATIO_BOUND = 0.5

ARM_ROOTS = {
    's39_C': os.path.join(RES, 'P515S39_C_run'),
    's39_D': os.path.join(RES, 'P515S39_D_run'),
}


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _identify_arm(run_dir):
    run_dir = os.path.abspath(run_dir)
    for arm, root in ARM_ROOTS.items():
        if os.path.abspath(root) == run_dir:
            return arm
    raise RuntimeError(f'{run_dir} does not match either s39 arm output root: {ARM_ROOTS}')


def _rows(run_dir):
    return E37._rows(run_dir)


def _first_pass_cycle(rows, channel):
    for r in rows:
        if r.get(f'boyd_{channel}_channel_pass'):
            return r['cycle']
    return None


def _certification_from_trajectory(rows, cap, required):
    """Spec v10 `certification_bar`, derived STRUCTURALLY from the
    trajectory's own per-row `cycle_convergence`/`consecutive_converged_
    cycles` bookkeeping (production's own fields, `shared_resources_
    planning.py` ~2735-2741) -- see module docstring."""
    last = rows[-1]
    cycles_run = len(rows)
    certified = bool(
        last.get('cycle_convergence') and (last.get('consecutive_converged_cycles') or 0) >= required
        and cycles_run <= cap)
    return {
        'certified': certified,
        'certification_cycle': (last['cycle'] if certified else None),
        'cycles_run': cycles_run, 'cap': cap, 'required_consecutive_cycles': required,
        'terminal_cycle_convergence': last.get('cycle_convergence'),
        'terminal_consecutive_converged_cycles': last.get('consecutive_converged_cycles'),
        'local_solve_failures_total': sum(1 for r in rows if r.get('local_solves_ok') is False),
    }


def _terminal_ratios(last):
    return {c: max(last.get(f'boyd_{c}_primal_ratio') or 0.0, last.get(f'boyd_{c}_dual_ratio') or 0.0)
            for c in CH}


def _max_objective_step_last_n(rows, n=10):
    tail = rows[-n:] if len(rows) >= n else rows
    steps = [r.get('objective_change_abs') for r in tail if r.get('objective_change_abs') is not None]
    return {'max_step': (max(steps) if steps else None), 'n_cycles_used': len(tail),
            'n_steps_available': len(steps)}


def _late_phase_decay_rates(matched_entries, window):
    """FIXED formula (see module docstring): the TRUE per-cycle geometric
    rate between each consecutive PAIR of matched cycles (c0, c1), gap =
    c1 - c0: `rate = (group_s(c1) / group_s(c0)) ** (1 / gap)`. The
    reported decay rate per group is the geometric mean of `rate` over
    every such pair inside `window` (falls back to the latest available
    pair if fewer than two matched cycles fall inside it)."""
    all_cycles = sorted(matched_entries)
    cycles_in_window = [c for c in all_cycles if window[0] <= c <= window[1]]
    fell_back = len(cycles_in_window) < 2
    cycles_used = cycles_in_window if not fell_back else all_cycles[-2:]
    per_dimension = {}
    for dim in PF_GROUP_DIMENSIONS:
        per_cycle_group_s = {}
        for c in cycles_used:
            group_sumsq = {}
            for e in matched_entries[c]:
                key = str(e[dim])
                group_sumsq[key] = group_sumsq.get(key, 0.0) + e['s'] ** 2
            per_cycle_group_s[c] = {k: math.sqrt(v) for k, v in group_sumsq.items()}
        rates_by_group = {}
        for i in range(len(cycles_used) - 1):
            c0, c1 = cycles_used[i], cycles_used[i + 1]
            gap = c1 - c0
            if gap <= 0:
                continue
            s0_map, s1_map = per_cycle_group_s.get(c0, {}), per_cycle_group_s.get(c1, {})
            for g, s0 in s0_map.items():
                s1 = s1_map.get(g)
                if s1 is not None and s0 > 0 and s1 > 0:
                    per_cycle_rate = (s1 / s0) ** (1.0 / gap)
                    rates_by_group.setdefault(g, []).append(per_cycle_rate)
        decay = {}
        for g, rates in rates_by_group.items():
            decay[g] = math.exp(sum(math.log(r) for r in rates) / len(rates))
        per_dimension[dim] = decay
    return {
        'window_requested': list(window), 'cycles_used': cycles_used,
        'window_used_note': (
            'requested window' if not fell_back else
            'fewer than two matched cycles fell inside the requested window -- '
            'fell back to the latest available pair of matched cycles'),
        'formula': 'rate = (group_s(c1) / group_s(c0)) ** (1 / (c1 - c0)); reported value = '
                   'geometric mean of rate over consecutive matched-cycle pairs in the window '
                   '(FIXED from s38: true per-cycle rate, not a raw pairwise ratio)',
        'geometric_mean_per_cycle_rate_per_group_per_dimension': per_dimension,
    }


def _ess_exempt_until_state(run_dir):
    hits = glob.glob(os.path.join(run_dir, 'ess_exempt_until_state_*.jsonl'))
    if not hits:
        return {'available': False}
    path = hits[0]
    lines = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    lift_line = next((rec for rec in lines if rec.get('channels', {}).get('ess', {}).get('lifted')), None)
    return {
        'available': True, 'path': os.path.relpath(path, REPO), 'n_lines': len(lines),
        'sidecar_lift_cycle': (lift_line['cycle'] if lift_line else None),
        'sidecar_lift_record': (lift_line['channels']['ess'] if lift_line else None),
    }


def _ess_actions_from_lift(rows, lift_cycle):
    if lift_cycle is None:
        return []
    return [{'cycle': r['cycle'], 'action': r.get('rho_ess_action'), 'rho_ess_after': r.get('rho_ess_after')}
            for r in rows if r['cycle'] >= lift_cycle]


def _evaluate_arm(arm, run_dir):
    g, g_path = _rows(run_dir)
    rows = g['cycle_trajectory']
    if not rows:
        raise RuntimeError(f'{run_dir}: empty cycle_trajectory')
    last = rows[-1]

    cert = _certification_from_trajectory(rows, CAP, REQUIRED_CONSECUTIVE)
    terminal_ratios = _terminal_ratios(last)
    clamp_flags = {c: any(r.get(f'rho_at_clamp_{c}') for r in rows) for c in CH}

    # -- cost bar (reported, never gated) ------------------------------------
    spec_data = json.load(open(SPEC_V10_PATH))
    cost_bar_spec = spec_data['certification_bar']['cost_bar']
    run1_cost = cost_bar_spec['run1_cost']
    run1_max_step_last10 = cost_bar_spec['run1_max_step_last10']
    arm_step = _max_objective_step_last_n(rows, 10)
    arm_cost = last.get('gross_operational_cost')
    bar = ((arm_step['max_step'] + run1_max_step_last10)
          if arm_step['max_step'] is not None else None)
    abs_diff = (abs(arm_cost - run1_cost) if arm_cost is not None else None)
    inside_bar = (abs_diff <= bar) if (abs_diff is not None and bar is not None) else None

    pf_first_pass = _first_pass_cycle(rows, 'pf')
    v_first_pass = _first_pass_cycle(rows, 'v')
    ess_first_pass = _first_pass_cycle(rows, 'ess')

    match_cycles = tuple(c for c in MATCH if c <= min(CAP, len(rows)) or c == rows[-1]['cycle'])
    cycles_to_keep_full = set(c for c in MATCH if c <= len(rows)) | {last['cycle']}
    pf_path, pf_z, pf_matched_entries, pf_identity = E38._load_pf_stride(run_dir, cycles_to_keep_full)

    identity_all_hold = (bool(pf_identity) and all(v.get('identity_holds') for v in pf_identity.values()))
    max_rel_err_r = max((v.get('rel_err_r') or 0.0) for v in pf_identity.values()) if pf_identity else None
    max_rel_err_s = max((v.get('rel_err_s') or 0.0) for v in pf_identity.values()) if pf_identity else None

    pf_decomposition_at_matched = {
        c: E38._pf_decomposition_at_cycle(entries) for c, entries in sorted(pf_matched_entries.items())
    }
    decay_rates = _late_phase_decay_rates(pf_matched_entries, DECAY_WINDOW) if pf_matched_entries else None

    pf_step_series = E37._step_series(pf_z)
    pf_osc_series = E37._oscillation_series(pf_step_series)
    pf_osc_flag = E37._oscillation_flag(pf_osc_series)

    ess_vectors, ess_stride_path = E37._load_ess_stride_p_vectors(run_dir)
    ess_step_series = E37._step_series(ess_vectors)
    ess_osc_series = E37._oscillation_series(ess_step_series)
    ess_osc_flag = E37._oscillation_flag(ess_osc_series)

    tso_trigger = E38._tso_instability_trigger(run_dir, rows)

    efc_traj = E37._efc_trajectory(rows, match_cycles)

    ess_exempt_until_sidecar = _ess_exempt_until_state(run_dir)
    trajectory_lift_cycle = next(
        (r['cycle'] for r in rows if r.get('rho_ess_action') not in (None, 'exempt (fixed)')), None
    ) if arm == 's39_D' else None
    ess_lift_and_actions = None
    if arm == 's39_D':
        sidecar_lift = ess_exempt_until_sidecar.get('sidecar_lift_cycle')
        lift_cycle_mismatch = (
            sidecar_lift is not None and trajectory_lift_cycle is not None
            and sidecar_lift != trajectory_lift_cycle)
        ess_lift_and_actions = {
            'sidecar': ess_exempt_until_sidecar,
            'trajectory_derived_lift_cycle': trajectory_lift_cycle,
            'lift_cycle_mismatch_sidecar_vs_trajectory': lift_cycle_mismatch,
            'ess_actions_from_lift_cycle': _ess_actions_from_lift(rows, trajectory_lift_cycle),
        }

    monitoring = {
        'local_solve_failures': g.get('local_solve_failures'),
        'network_failures_by_tier': E37._failures_by_tier(run_dir, rows),
        'pf_oscillation': {'flag': pf_osc_flag,
                           'per_cycle_sampled': [{'cycle': c, **pf_osc_series[c]} for c in sorted(pf_osc_series)
                                                  if c in match_cycles or c == last['cycle']]},
        'ess_oscillation': {'flag': ess_osc_flag,
                            'per_cycle_sampled': [{'cycle': c, **ess_osc_series[c]} for c in sorted(ess_osc_series)
                                                   if c in match_cycles or c == last['cycle']]},
        'rho_gamma_trajectory_sampled': E37._rho_gamma_trajectory(rows, match_cycles),
        'channel_ratios_sampled': E37._channel_ratio_trajectory(rows, match_cycles),
        'efc_trajectory_sampled': efc_traj,
        'pf_capture_identity': {
            'path': os.path.relpath(pf_path, REPO) if pf_path else None,
            'all_cycles_identity_holds': identity_all_hold,
            'max_rel_err_r': max_rel_err_r, 'max_rel_err_s': max_rel_err_s,
            'n_cycles_checked': len(pf_identity),
        },
    }

    return {
        'arm': arm, 'run_dir': os.path.relpath(run_dir, REPO),
        'g_baseline_path': os.path.relpath(g_path, REPO), 'g_baseline_sha256': _sha256(g_path),
        'instance': g.get('instance'), 'cycles_run': len(rows),
        'CERTIFICATION': cert,
        'REPORTED_NOT_GATED': {
            'terminal_ratios_per_channel': terminal_ratios,
            'rule_ten_objective_ratio': last.get('objective_change_ratio'),
            'clamp_flags_ever_true_per_channel': clamp_flags,
            'cost_vs_run1': {
                'arm_cost': arm_cost, 'run1_cost': run1_cost, 'abs_diff': abs_diff,
                'arm_max_objective_step_last_10_cycles': arm_step,
                'run1_max_step_last10': run1_max_step_last10,
                'bar': bar, 'inside_bar': inside_bar,
                'spec_v10_path': os.path.relpath(SPEC_V10_PATH, REPO),
                'spec_v10_sha256': _sha256(SPEC_V10_PATH),
                'reading': cost_bar_spec.get('reading'),
            },
        },
        'FIRST_PASS_CYCLES': {'pf': pf_first_pass, 'v': v_first_pass, 'ess': ess_first_pass},
        'TSO_INSTABILITY_TRIGGER': tso_trigger,
        'ESS_LIFT_AND_ACTIONS': ess_lift_and_actions,
        'PF_DECOMPOSITION': {
            'at_matched_cycles': pf_decomposition_at_matched,
            'late_phase_decay_rate': decay_rates,
        },
        'wall_clock_s': g.get('wall_clock_s'), 'solves': g.get('solve_profile'),
        'monitoring': monitoring,
    }


def _score_predictions_c(c_eval):
    if c_eval is None:
        return {'status': 'PENDING', 'reason': 'arm C evaluation not present'}
    pf_first_pass = c_eval['FIRST_PASS_CYCLES']['pf']
    cert_cycle = c_eval['CERTIFICATION']['certification_cycle']
    pred1 = bool(pf_first_pass is not None and pf_first_pass <= C_PF_FIRST_PASS_BOUND)
    pred2 = bool(cert_cycle is not None and cert_cycle <= C_CERTIFICATION_CYCLE_BOUND)
    return {
        'status': 'SCORED',
        'prediction_1_pf_first_pass_le_131': {'pf_first_pass_cycle': pf_first_pass,
                                               'bound': C_PF_FIRST_PASS_BOUND, 'pass': pred1},
        'prediction_2_certification_cycle_le_150': {'certification_cycle': cert_cycle,
                                                     'bound': C_CERTIFICATION_CYCLE_BOUND, 'pass': pred2},
        'both_pass': bool(pred1 and pred2),
    }


def _score_predictions_d(d_eval):
    if d_eval is None:
        return {'status': 'PENDING', 'reason': 'arm D evaluation not present'}
    certified = d_eval['CERTIFICATION']['certified']
    cert_cycle = d_eval['CERTIFICATION']['certification_cycle']
    last_row_ratios = d_eval['REPORTED_NOT_GATED']['terminal_ratios_per_channel']
    storage_terminal_ratio_at_cert_or_terminal = last_row_ratios['ess']
    pred1 = bool(certified)
    pred2 = bool(certified and storage_terminal_ratio_at_cert_or_terminal < D_STORAGE_TERMINAL_RATIO_BOUND)
    return {
        'status': 'SCORED',
        'prediction_1_certifies_under_the_bar': {'certified': certified, 'certification_cycle': cert_cycle,
                                                  'pass': pred1},
        'prediction_2_storage_terminal_ratio_lt_0p5_at_certification': {
            'storage_terminal_ratio': storage_terminal_ratio_at_cert_or_terminal,
            'bound': D_STORAGE_TERMINAL_RATIO_BOUND,
            'evaluated_at': ('certification cycle (== terminal cycle when certified)'
                             if certified else 'terminal cycle (NOT certified -- reported honestly, scores False)'),
            'pass': pred2,
        },
        'both_pass': bool(pred1 and pred2),
    }


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run_dir = os.path.abspath(args[0]) if args else ARM_ROOTS['s39_C']
    arm = _identify_arm(run_dir)
    out_path = os.path.join(run_dir, 's39_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    if not os.path.exists(os.path.join(REF_RUN, 'g_baseline.json')):
        raise RuntimeError(f'run 1 (s35ref) reference not found at {REF_RUN}/g_baseline.json')
    if not os.path.exists(SPEC_V10_PATH):
        raise RuntimeError(f'frozen spec v10 not found at {SPEC_V10_PATH}')

    guard = SolveProfileGuard(permitted=(), label='P5.15 s39 independent evaluation').install()
    try:
        this_eval = _evaluate_arm(arm, run_dir)

        other_arm = 's39_D' if arm == 's39_C' else 's39_C'
        other_root = ARM_ROOTS[other_arm]
        other_eval = None
        if glob.glob(os.path.join(other_root, 'g_*.json')):
            try:
                other_eval = _evaluate_arm(other_arm, other_root)
            except Exception as error:  # noqa: BLE001 -- report, don't hide
                other_eval = {'error': f'{type(error).__name__}: {error}'}

        c_eval = this_eval if arm == 's39_C' else (other_eval if other_eval and 'error' not in other_eval else None)
        d_eval = this_eval if arm == 's39_D' else (other_eval if other_eval and 'error' not in other_eval else None)

        predictions = {'s39_C': _score_predictions_c(c_eval), 's39_D': _score_predictions_d(d_eval)}

        if c_eval is None or d_eval is None:
            adoption = {
                'status': 'PENDING',
                'reason': 'both C and D evaluations must exist to determine the spec v10 adoption rule',
                'c_available': c_eval is not None, 'd_available': d_eval is not None,
            }
        else:
            d_certified = d_eval['CERTIFICATION']['certified']
            d_storage_ratio = d_eval['REPORTED_NOT_GATED']['terminal_ratios_per_channel']['ess']
            c_certified = c_eval['CERTIFICATION']['certified']
            if d_certified and d_storage_ratio < D_STORAGE_TERMINAL_RATIO_BOUND:
                verdict, oracle = ('D certifies with storage terminal ratio < 0.5 -- D is the '
                                   'production oracle.', 's39_D')
            elif c_certified:
                verdict, oracle = ("D does not meet its storage-ratio condition -- C certifies, "
                                   "so C is the production oracle (spec v10 fallback).", 's39_C')
            else:
                verdict, oracle = ('Neither arm certifies under the new bar -- stop for review.', None)
            adoption = {
                'status': 'DECIDED', 'c_certified': c_certified, 'd_certified': d_certified,
                'd_storage_terminal_ratio': d_storage_ratio, 'oracle': oracle, 'verdict': verdict,
            }

        out = {
            'stage': 'P5.15 Addendum 21 - oracle arms under the new certification bar - independent evaluation',
            'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 21', 'data/SRP1/Results/P515S39/'
                          'frozen_s39_oracle_spec_v10_f1b2b999.json'],
            'LEAD': {
                'arm': arm,
                'certified': this_eval['CERTIFICATION']['certified'],
                'certification_cycle': this_eval['CERTIFICATION']['certification_cycle'],
                'cycle_count': this_eval['cycles_run'],
                'pf_first_pass_cycle': this_eval['FIRST_PASS_CYCLES']['pf'],
                'storage_terminal_ratio': this_eval['REPORTED_NOT_GATED']['terminal_ratios_per_channel']['ess'],
                'cost_inside_bar': this_eval['REPORTED_NOT_GATED']['cost_vs_run1']['inside_bar'],
                'tso_instability_triggered': (this_eval.get('TSO_INSTABILITY_TRIGGER') or {}).get('TRIGGERED'),
            },
            'this_arm': this_eval,
            'other_arm_present': other_eval,
            'cross_arm': {'PREDICTIONS': predictions, 'ADOPTION': adoption},
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)

    print(json.dumps({'LEAD': out['LEAD'], 'CERTIFICATION': out['this_arm']['CERTIFICATION'],
                      'REPORTED_NOT_GATED': out['this_arm']['REPORTED_NOT_GATED'],
                      'TSO_INSTABILITY_TRIGGER': out['this_arm']['TSO_INSTABILITY_TRIGGER'],
                      'cross_arm': out['cross_arm']},
                     indent=1, default=str))
    if not dry:
        with open(out_path, 'w') as handle:
            json.dump(out, handle, indent=1, default=str)
        print(f'[S39 evaluate] wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
