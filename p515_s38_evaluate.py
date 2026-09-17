"""P5.15 Addendum 20 - the PF-pace experiment (arms s38_A_tau0,
s38_A_tau0p25_fallback, s38_B_pfbal, s38_C_combined) - INDEPENDENT
evaluation.

Frozen spec v9, data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json.

Zero solves, SolveProfileGuard armed. Recomputes EVERYTHING from the arm's own
per-cycle trajectory (g_<arm>.json), its ess_entry_stride sidecar and its NEW
pf_entry_stride sidecar -- never trusts boyd_terminal.json's own `stopped_by`
field (the same v1/v2 defect s35ref/s35pt/s37's evaluators found; see
`_stop_from_trajectory`, reused BY IMPORT from `p515_s37_evaluate`).

Every formula used below:

CERTIFICATION (spec v9 `certification_per_arm.certified_iff`), all four required:
  (a) Boyd stop on all three channels, 3 consecutive converged cycles, within
      300, derived from the trajectory (`p515_s37_evaluate._stop_from_trajectory`,
      reused unmodified with CAP=300);
  (b) no V or PF channel EVER frozen at a rho clamp over the whole run
      (`rho_at_clamp_v`/`rho_at_clamp_pf` -- the ESS channel is intentionally
      exempt/frozen from cycle 1 in every s38 arm, and PF is ALSO exempt in A
      and its fallback; an exempt channel's `at_clamp` is set False by
      production and never re-enters the legacy/backstop/streak clamp logic,
      so this criterion is expected to pass trivially for an exempt channel
      -- reported per channel, not silently assumed);
  (c) storage-channel terminal ratio max(boyd_ess_primal_ratio,
      boyd_ess_dual_ratio) < 0.9;
  (d) terminal system cost (gross_operational_cost) within the rule-nine bar
      of run 1 (s35ref): |Q_arm - Q_run1| <= terminal_step(arm) +
      terminal_step(run1); run 1's terminal cost/step are READ from its
      committed g_baseline.json BY PATH, sha256-recorded.

PF/V/ESS FIRST-PASS CYCLE: the first cycle at which `boyd_<channel>_channel_pass`
(= primal_pass AND dual_pass, same cycle) is True.

HELPS (spec v9 `certification_per_arm.helps`): an arm HELPS iff its PF
first-pass cycle exists and is <= 180 (>= 20% earlier than run 1's 226),
whether or not it certifies.

TSO-INSTABILITY TRIGGER (spec v9 `tso_instability_trigger_for_A_fallback`,
evaluated ONLY on s38_A_tau0's completed run, before B launches). TRIGGERED
iff ANY of:
  (i) any unrecovered TSO (case9) network failure (`network_failures_*.jsonl`,
      agent=='TSO', class=='unrecovered');
  (ii) TSO (case9) network-failure rate over the arm's own cycles
      (n_tso_events / cycles_run) > 2x run 1's rate (58/477 =
      0.12159329140461216 per cycle), i.e. > 0.24318658280922433 per cycle;
  (iii) OSCILLATION FLAG on the per-entry TSO PF consensus step (z_TSO PF,
      ALL nodes/years/days/p/q/periods -- unlike the ESS definition below,
      which is P-only): sign-change fraction > 0.2 for 10 consecutive
      cycles, or cosine between consecutive step vectors < 0 for 5
      consecutive cycles (entries with |step| < 1e-9 MW ignored from the
      sign-change fraction only, per the v8 ESS definition applied here to
      PF -- `p515_s37_evaluate._oscillation_series`/`_oscillation_flag`,
      REUSED unmodified on the PF z-vector series this module builds);
  (iv) the SAME oscillation flag on the ESS stride (v8/s37 definition,
      P-only, `p515_s37_evaluate` helpers reused unmodified).
`not_a_trigger`: failure to certify, slow PF convergence, or DSO failures
alone (per spec v9).

PREDICTIONS (spec v9 `predictions_recorded_in_advance`), scored once the
relevant arm(s) are present:
  * "PF first pass well before 226 in at least one arm" -- operationalized as
    PF first-pass cycle <= 180 in at least one of {A (or its fallback if A
    triggered), B};
  * "A is the stronger candidate" -- operationalized as A's (or the
    fallback's, if A triggered) PF first-pass cycle < B's, OR A/fallback
    passes PF within 300 and B does not.

PER-ENTRY PF DECOMPOSITION (spec v9 `report`), at matched cycles
(1,2,5,10,20,30,50,75,100,125,150,175,200,226,250,275,300, capped at the
arm's own cap): for each grouping dimension in {node_id, year, day,
power_type, period}, the SHARE of total sum(s**2) and sum(r**2) contributed
by each group value at that cycle (share = group_sumsq / total_sumsq), plus
the top-10 individual entries by s**2. LATE-PHASE PER-GROUP DECAY RATE
(defined here, before computing, per the CLAUDE.md evidence rule): for each
grouping dimension, and each group value within it, the GEOMETRIC MEAN of
the ratio group_s(cycle_{i+1}) / group_s(cycle_i) -- where group_s(cycle) =
sqrt(sum of entry s**2 within that group AT THAT MATCHED CYCLE) -- taken over
consecutive PAIRS of MATCHED cycles falling inside the window [100, 200]; if
fewer than two matched cycles fall inside that window (e.g. the arm stopped
before cycle 100), the LATEST available pair of matched cycles is used
instead and the fallback is recorded explicitly (`window_used_note`). A
ratio < 1 means that group's PF residual contribution is decaying in the
late phase; > 1 means it is growing.

MONITORING (reported, not gated): local-solve/network failures by tier;
oscillation flags on PF and ESS; EFC/day, rho/gamma-per-channel-with-action-
label trajectories, channel ratio trajectories, all sampled at matched
cycles; per-entry PF capture identity (`rel_err_r`/`rel_err_s`/
`identity_holds`, read back from the sidecar the harness itself asserted
per cycle -- re-verified here, not merely trusted, by checking every line's
own recorded `identity_holds` is True).

Usage: python p515_s38_evaluate.py [RUN_DIR] [--dry-run]
  RUN_DIR defaults to the s38_A_tau0 arm's output root. Writes
  RUN_DIR/s38_evaluation.json (write-once; refuses to overwrite). Every
  OTHER s38 arm whose own output root currently exists is read (NEVER
  re-computed by mutating its directory) for the cross-arm sections
  (fallback-trigger-implied "A used for the verdict", predictions, helps
  summary, adoption reasoning per spec v9 `certification_per_arm.adoption`).
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
import p515_s37_evaluate as E37  # noqa: E402 -- reused helpers, see module docstring

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')  # run 1
CH = ('v', 'pf', 'ess')
CAP = 300
REQUIRED_CONSECUTIVE = 3
STORAGE_TERMINAL_RATIO_BOUND = 0.9
PF_HELPS_CYCLE_BOUND = 180
RUN1_PF_FIRST_PASS_CYCLE = 226  # spec v9 `reference`
RUN1_BOYD_STOP_CYCLE = 477  # spec v9 `reference`
RUN1_TSO_FAILURE_RATE = 58.0 / 477.0  # spec v9 `reference.tso_case9_failure_rate_per_cycle`
TSO_FAILURE_RATE_TRIGGER_BOUND = 2.0 * RUN1_TSO_FAILURE_RATE
OSCILLATION_SIGN_CHANGE_THRESHOLD = 0.2
OSCILLATION_SIGN_CHANGE_STREAK = 10
OSCILLATION_COSINE_STREAK = 5
MATCH = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 226, 250, 275, 300)
DECAY_WINDOW = (100, 200)
PF_GROUP_DIMENSIONS = ('node_id', 'year', 'day', 'power_type', 'period')

ARM_ROOTS = {
    's38_A_tau0': os.path.join(RES, 'P515S38_A_TAU0_run'),
    's38_A_tau0p25_fallback': os.path.join(RES, 'P515S38_A_TAU0P25_run'),
    's38_B_pfbal': os.path.join(RES, 'P515S38_B_PFBAL_run'),
    's38_C_combined': os.path.join(RES, 'P515S38_C_COMBINED_run'),
}
ARM_TAU = {'s38_A_tau0': 0.0, 's38_A_tau0p25_fallback': 0.25, 's38_B_pfbal': 1.0, 's38_C_combined': 0.0}
ARM_EXEMPT = {
    's38_A_tau0': ['ess', 'pf'], 's38_A_tau0p25_fallback': ['ess', 'pf'],
    's38_B_pfbal': ['ess'], 's38_C_combined': ['ess'],
}


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _identify_arm(run_dir):
    run_dir = os.path.abspath(run_dir)
    for arm, root in ARM_ROOTS.items():
        if os.path.abspath(root) == run_dir:
            return arm
    raise RuntimeError(f'{run_dir} does not match any s38 arm output root: {ARM_ROOTS}')


def _rows(run_dir):
    return E37._rows(run_dir)


def _first_pass_cycle(rows, channel):
    for r in rows:
        if r.get(f'boyd_{channel}_channel_pass'):
            return r['cycle']
    return None


def _pf_stride_path(run_dir):
    hits = glob.glob(os.path.join(run_dir, 'pf_entry_stride_*.jsonl'))
    return hits[0] if hits else None


def _load_pf_stride(run_dir, cycles_to_keep_full):
    """Single streaming pass over the (possibly large) PF per-entry sidecar:
    returns (path, z_vectors, matched_entries, identity_by_cycle).
    `z_vectors[cycle]` is a flat numpy array of `z_tso_current` over EVERY
    entry, in the sidecar's own per-line order (stable across cycles: the
    harness's `s38_pf_capture_hooks` iterates the SAME node/year/day/
    power_type/period collections in the SAME order every cycle).
    `matched_entries[cycle]` is the RAW entries list, kept ONLY for cycles in
    `cycles_to_keep_full` (the decomposition is computed at matched cycles
    only -- reading and retaining every cycle's full entry list for a
    300-cycle run would be needlessly expensive). `identity_by_cycle[cycle]`
    is the harness's OWN per-cycle identity-check result
    (`rel_err_r`/`rel_err_s`/`identity_holds`), re-verified here, not merely
    trusted."""
    path = _pf_stride_path(run_dir)
    z_vectors, matched_entries, identity_by_cycle = {}, {}, {}
    if not path:
        return None, z_vectors, matched_entries, identity_by_cycle
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cycle = rec.get('cycle')
            entries = rec.get('entries', [])
            z_vectors[cycle] = np.asarray([e['z_tso_current'] for e in entries], dtype=float)
            identity_by_cycle[cycle] = {
                'rel_err_r': rec.get('rel_err_r'), 'rel_err_s': rec.get('rel_err_s'),
                'identity_holds': rec.get('identity_holds'),
            }
            if cycle in cycles_to_keep_full:
                matched_entries[cycle] = entries
    return path, z_vectors, matched_entries, identity_by_cycle


def _pf_decomposition_at_cycle(entries):
    total_s2 = sum(e['s'] ** 2 for e in entries)
    total_r2 = sum(e['r'] ** 2 for e in entries)
    by_dimension = {}
    for dim in PF_GROUP_DIMENSIONS:
        group_s2, group_r2 = {}, {}
        for e in entries:
            key = str(e[dim])
            group_s2[key] = group_s2.get(key, 0.0) + e['s'] ** 2
            group_r2[key] = group_r2.get(key, 0.0) + e['r'] ** 2
        by_dimension[dim] = {
            'share_of_total_s2': {k: (v / total_s2 if total_s2 else None) for k, v in group_s2.items()},
            'share_of_total_r2': {k: (v / total_r2 if total_r2 else None) for k, v in group_r2.items()},
        }
    top10 = sorted(entries, key=lambda e: e['s'] ** 2, reverse=True)[:10]
    top10_reported = [{'node_id': e['node_id'], 'year': e['year'], 'day': e['day'],
                       'power_type': e['power_type'], 'period': e['period'],
                       'r': e['r'], 's': e['s'], 's2': e['s'] ** 2} for e in top10]
    return {'total_s2': total_s2, 'total_r2': total_r2, 'by_dimension': by_dimension,
            'top10_entries_by_s2': top10_reported, 'n_entries': len(entries)}


def _late_phase_decay_rates(matched_entries, window):
    """Geometric-mean ratio of group s (= sqrt(sum entry s**2 within that
    group at that matched cycle)) between CONSECUTIVE matched cycles inside
    `window`; falls back to the latest available pair of matched cycles if
    fewer than two fall inside the window. See module docstring for the
    formula."""
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
        ratios_by_group = {}
        for i in range(len(cycles_used) - 1):
            c0, c1 = cycles_used[i], cycles_used[i + 1]
            s0_map, s1_map = per_cycle_group_s.get(c0, {}), per_cycle_group_s.get(c1, {})
            for g, s0 in s0_map.items():
                s1 = s1_map.get(g)
                if s1 is not None and s0 > 0:
                    ratios_by_group.setdefault(g, []).append(s1 / s0)
        decay = {}
        for g, ratios in ratios_by_group.items():
            positive = [r for r in ratios if r > 0]
            if positive:
                decay[g] = math.exp(sum(math.log(r) for r in positive) / len(positive))
        per_dimension[dim] = decay
    return {
        'window_requested': list(window), 'cycles_used': cycles_used,
        'window_used_note': (
            'requested window' if not fell_back else
            'fewer than two matched cycles fell inside the requested window -- '
            'fell back to the latest available pair of matched cycles'),
        'geometric_mean_ratio_per_group_per_dimension': per_dimension,
    }


def _tso_failures(run_dir):
    hits = glob.glob(os.path.join(run_dir, 'network_failures_*.jsonl'))
    if not hits:
        return {'available': False}
    events = [json.loads(l) for l in open(hits[0]) if l.strip()]
    tso_events = [e for e in events if e.get('agent') == 'TSO']
    unrecovered = [e for e in tso_events if e.get('class') == 'unrecovered']
    return {'available': True, 'path': os.path.relpath(hits[0], REPO),
            'n_tso_events': len(tso_events), 'n_tso_unrecovered': len(unrecovered),
            'tso_unrecovered_events': unrecovered}


def _tso_instability_trigger(run_dir, rows):
    """Spec v9 `tso_instability_trigger_for_A_fallback`, all four criteria."""
    cycles_run = len(rows)
    tso_fail = _tso_failures(run_dir)
    crit_i = bool(tso_fail.get('available') and tso_fail.get('n_tso_unrecovered', 0) > 0)
    tso_rate = (tso_fail.get('n_tso_events', 0) / cycles_run) if (tso_fail.get('available') and cycles_run) else None
    crit_ii = bool(tso_rate is not None and tso_rate > TSO_FAILURE_RATE_TRIGGER_BOUND)

    pf_path, pf_z, _pf_matched, pf_identity = _load_pf_stride(run_dir, cycles_to_keep_full=set())
    pf_step_series = E37._step_series(pf_z)
    pf_osc_series = E37._oscillation_series(pf_step_series)
    pf_osc_flag = E37._oscillation_flag(pf_osc_series)
    crit_iii = bool(pf_osc_flag['flag'])

    ess_vectors, ess_stride_path = E37._load_ess_stride_p_vectors(run_dir)
    ess_step_series = E37._step_series(ess_vectors)
    ess_osc_series = E37._oscillation_series(ess_step_series)
    ess_osc_flag = E37._oscillation_flag(ess_osc_series)
    crit_iv = bool(ess_osc_flag['flag'])

    triggered = bool(crit_i or crit_ii or crit_iii or crit_iv)
    return {
        'criterion_i_unrecovered_tso_failure': {'pass_meaning_triggered': crit_i, **tso_fail},
        'criterion_ii_tso_failure_rate_gt_2x_run1': {
            'triggered': crit_ii, 'observed_rate_per_cycle': tso_rate,
            'bound_2x_run1': TSO_FAILURE_RATE_TRIGGER_BOUND, 'run1_rate': RUN1_TSO_FAILURE_RATE,
        },
        'criterion_iii_pf_oscillation_flag': {'triggered': crit_iii, 'detail': pf_osc_flag,
                                              'pf_stride_path': os.path.relpath(pf_path, REPO) if pf_path else None},
        'criterion_iv_ess_oscillation_flag': {'triggered': crit_iv, 'detail': ess_osc_flag,
                                              'ess_stride_path': os.path.relpath(ess_stride_path, REPO) if ess_stride_path else None},
        'TRIGGERED': triggered,
        'note': 'not_a_trigger: failure to certify, slow PF convergence, or DSO failures alone (spec v9).',
    }


def _evaluate_arm(arm, run_dir):
    g, g_path = _rows(run_dir)
    rows = g['cycle_trajectory']
    if not rows:
        raise RuntimeError(f'{run_dir}: empty cycle_trajectory')
    last = rows[-1]
    required = last.get('required_consecutive_cycles') or REQUIRED_CONSECUTIVE
    stop = E37._stop_from_trajectory(rows, CAP, required)

    clamp_v_pf = any(r.get(f'rho_at_clamp_{c}') for r in rows for c in ('v', 'pf'))
    clamp_ess = any(r.get('rho_at_clamp_ess') for r in rows)  # reported, never gates certification

    storage_terminal_ratio = max(last.get('boyd_ess_primal_ratio') or 0.0, last.get('boyd_ess_dual_ratio') or 0.0)

    ref_g_path = os.path.join(REF_RUN, 'g_baseline.json')
    ref_g = json.load(open(ref_g_path))
    ref_rows = ref_g['cycle_trajectory']
    ref_last = ref_rows[-1]
    ref_cost, ref_step = ref_last['gross_operational_cost'], ref_last.get('objective_change_abs')
    cost, step = last['gross_operational_cost'], last.get('objective_change_abs')
    bar = (step + ref_step) if (step is not None and ref_step is not None) else None
    abs_diff = abs(cost - ref_cost)
    crit_cost = (abs_diff <= bar) if bar is not None else None

    crit_a = (stop['stopped_by'] == 'boyd')
    crit_b = (not clamp_v_pf)
    crit_c = (storage_terminal_ratio < STORAGE_TERMINAL_RATIO_BOUND)
    crit_d = bool(crit_cost)
    certified = bool(crit_a and crit_b and crit_c and crit_d)

    pf_first_pass = _first_pass_cycle(rows, 'pf')
    v_first_pass = _first_pass_cycle(rows, 'v')
    ess_first_pass = _first_pass_cycle(rows, 'ess')
    helps = bool(pf_first_pass is not None and pf_first_pass <= PF_HELPS_CYCLE_BOUND)

    match_cycles = tuple(c for c in MATCH if c <= CAP)
    cycles_to_keep_full = set(match_cycles) | {last['cycle']}
    pf_path, pf_z, pf_matched_entries, pf_identity = _load_pf_stride(run_dir, cycles_to_keep_full)

    identity_all_hold = (
        bool(pf_identity) and all(v.get('identity_holds') for v in pf_identity.values()))
    max_rel_err_r = max((v.get('rel_err_r') or 0.0) for v in pf_identity.values()) if pf_identity else None
    max_rel_err_s = max((v.get('rel_err_s') or 0.0) for v in pf_identity.values()) if pf_identity else None

    pf_decomposition_at_matched = {
        c: _pf_decomposition_at_cycle(entries) for c, entries in sorted(pf_matched_entries.items())
    }
    decay_rates = _late_phase_decay_rates(pf_matched_entries, DECAY_WINDOW) if pf_matched_entries else None

    pf_step_series = E37._step_series(pf_z)
    pf_osc_series = E37._oscillation_series(pf_step_series)
    pf_osc_flag = E37._oscillation_flag(pf_osc_series)

    ess_vectors, ess_stride_path = E37._load_ess_stride_p_vectors(run_dir)
    ess_step_series = E37._step_series(ess_vectors)
    ess_osc_series = E37._oscillation_series(ess_step_series)
    ess_osc_flag = E37._oscillation_flag(ess_osc_series)

    tso_trigger = _tso_instability_trigger(run_dir, rows) if arm == 's38_A_tau0' else None

    efc_traj = E37._efc_trajectory(rows, match_cycles)
    settling = {c: max(last.get(f'boyd_{c}_primal_ratio') or 0.0, last.get(f'boyd_{c}_dual_ratio') or 0.0)
                for c in CH}

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
        'arm': arm, 'tau': ARM_TAU[arm], 'balancing_exempt_channels': ARM_EXEMPT[arm],
        'run_dir': os.path.relpath(run_dir, REPO),
        'g_baseline_path': os.path.relpath(g_path, REPO), 'g_baseline_sha256': _sha256(g_path),
        'instance': g.get('instance'), 'cycles_run': len(rows),
        'CERTIFICATION': {
            'criterion_a_boyd_stop_within_300': {**stop, 'pass': crit_a},
            'criterion_b_no_v_pf_clamp': {'clamp_v_pf_any': clamp_v_pf, 'clamp_ess_any_reported_only': clamp_ess,
                                          'pass': crit_b},
            'criterion_c_storage_terminal_ratio_below_0p9': {
                'storage_terminal_ratio': storage_terminal_ratio, 'bound': STORAGE_TERMINAL_RATIO_BOUND,
                'boyd_ess_primal_ratio': last.get('boyd_ess_primal_ratio'),
                'boyd_ess_dual_ratio': last.get('boyd_ess_dual_ratio'), 'pass': crit_c},
            'criterion_d_cost_within_rule_nine_bar_of_run1': {
                'arm_cost': cost, 'run1_cost': ref_cost, 'abs_diff': abs_diff,
                'arm_terminal_step': step, 'run1_terminal_step': ref_step, 'bar': bar,
                'run1_g_baseline_path': os.path.relpath(ref_g_path, REPO),
                'run1_g_baseline_sha256': _sha256(ref_g_path),
                'valid_note': 'valid because both runs are then settled (criterion a already requires '
                              'the arm itself to have stopped under Boyd)',
                'pass': crit_d},
            'CERTIFIED': certified,
        },
        'FIRST_PASS_CYCLES': {'pf': pf_first_pass, 'v': v_first_pass, 'ess': ess_first_pass},
        'HELPS': {'value': helps, 'pf_first_pass_cycle': pf_first_pass, 'bound': PF_HELPS_CYCLE_BOUND,
                  'run1_pf_first_pass_cycle': RUN1_PF_FIRST_PASS_CYCLE},
        'TSO_INSTABILITY_TRIGGER': tso_trigger,
        'PF_DECOMPOSITION': {
            'at_matched_cycles': pf_decomposition_at_matched,
            'late_phase_decay_rate': decay_rates,
        },
        'settling_quality': {**settling, 'objective_rule_ten': last.get('objective_change_ratio'),
                             'terminal_step_over_tolerance': (
                                 (last.get('objective_change_abs') / last.get('objective_tolerance'))
                                 if last.get('objective_change_abs') and last.get('objective_tolerance') else None)},
        'wall_clock_s': g.get('wall_clock_s'), 'solves': g.get('solve_profile'),
        'monitoring': monitoring,
    }


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run_dir = os.path.abspath(args[0]) if args else ARM_ROOTS['s38_A_tau0']
    arm = _identify_arm(run_dir)
    out_path = os.path.join(run_dir, 's38_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    if not os.path.exists(os.path.join(REF_RUN, 'g_baseline.json')):
        raise RuntimeError(f'run 1 (s35ref) reference not found at {REF_RUN}/g_baseline.json')

    guard = SolveProfileGuard(permitted=(), label='P5.15 s38 independent evaluation').install()
    try:
        this_eval = _evaluate_arm(arm, run_dir)

        other_evals = {}
        for other_arm, other_root in ARM_ROOTS.items():
            if other_arm == arm:
                continue
            if glob.glob(os.path.join(other_root, 'g_*.json')):
                try:
                    other_evals[other_arm] = _evaluate_arm(other_arm, other_root)
                except Exception as error:  # noqa: BLE001 -- report, don't hide
                    other_evals[other_arm] = {'error': f'{type(error).__name__}: {error}'}

        all_evals = dict(other_evals)
        all_evals[arm] = this_eval

        # -- which "A" is used for the verdict (spec v9: the fallback
        #    REPLACES A in the verdict only if A's own trigger fired; A's
        #    own run/evaluation is always kept and reported separately) ----
        a_eval = all_evals.get('s38_A_tau0')
        fallback_eval = all_evals.get('s38_A_tau0p25_fallback')
        a_used_for_verdict = None
        a_used_arm_key = None
        if a_eval is not None and 'error' not in a_eval:
            triggered = bool((a_eval.get('TSO_INSTABILITY_TRIGGER') or {}).get('TRIGGERED'))
            if not triggered:
                a_used_for_verdict, a_used_arm_key = a_eval, 's38_A_tau0'
            elif fallback_eval is not None and 'error' not in fallback_eval:
                a_used_for_verdict, a_used_arm_key = fallback_eval, 's38_A_tau0p25_fallback'
            else:
                a_used_arm_key = 's38_A_tau0p25_fallback (triggered, not yet run)'

        b_eval = all_evals.get('s38_B_pfbal')

        predictions = {'status': 'PENDING', 'reason': 'neither an A-class nor B evaluation is available yet'}
        helps_summary = {arm_key: (ev.get('HELPS') if isinstance(ev, dict) else None)
                         for arm_key, ev in all_evals.items()}
        if a_used_for_verdict is not None or b_eval is not None:
            a_pf = (a_used_for_verdict or {}).get('FIRST_PASS_CYCLES', {}).get('pf')
            b_pf = (b_eval or {}).get('FIRST_PASS_CYCLES', {}).get('pf')
            pred1_pass = bool((a_pf is not None and a_pf <= PF_HELPS_CYCLE_BOUND) or
                              (b_pf is not None and b_pf <= PF_HELPS_CYCLE_BOUND))
            pred2_pass = None
            if a_pf is not None or b_pf is not None:
                if a_pf is not None and b_pf is not None:
                    pred2_pass = bool(a_pf < b_pf)
                elif a_pf is not None and b_pf is None:
                    pred2_pass = True
                elif a_pf is None and b_pf is not None:
                    pred2_pass = False
            predictions = {
                'status': 'SCORED',
                'a_used_for_verdict_arm': a_used_arm_key,
                'prediction_1_pf_first_pass_well_before_226': {
                    'a_pf_first_pass_cycle': a_pf, 'b_pf_first_pass_cycle': b_pf,
                    'bound': PF_HELPS_CYCLE_BOUND, 'run1_reference': RUN1_PF_FIRST_PASS_CYCLE,
                    'pass': pred1_pass,
                },
                'prediction_2_a_is_stronger_candidate': {
                    'a_pf_first_pass_cycle': a_pf, 'b_pf_first_pass_cycle': b_pf, 'pass': pred2_pass,
                },
            }

        certified_map = {k: (v.get('CERTIFICATION', {}).get('CERTIFIED') if isinstance(v, dict) and 'CERTIFICATION' in v else None)
                         for k, v in all_evals.items()}
        a_certified = (a_used_for_verdict or {}).get('CERTIFICATION', {}).get('CERTIFIED')
        b_certified = (b_eval or {}).get('CERTIFICATION', {}).get('CERTIFIED')
        a_helps = (a_used_for_verdict or {}).get('HELPS', {}).get('value')
        b_helps = (b_eval or {}).get('HELPS', {}).get('value')

        if a_used_for_verdict is None or b_eval is None:
            adoption = {
                'status': 'PENDING',
                'reason': 'both an A-class run (s38_A_tau0 or, if triggered, its fallback) AND '
                          's38_B_pfbal must be present to determine the spec v9 adoption rule',
                'a_available': a_used_for_verdict is not None, 'b_available': b_eval is not None,
            }
        else:
            both_help = bool(a_helps and b_helps)
            if both_help:
                verdict = ("Addendum 20 (c): both A (or its fallback) and B HELP -- one combined "
                          "confirmation run (s38_C_combined) is authorized, launched only after "
                          "the Planner's stop-for-review.")
            elif a_helps or b_helps:
                which = a_used_arm_key if a_helps else 's38_B_pfbal'
                verdict = (f"exactly one arm HELPS ({which}) -- candidate oracle configuration, "
                          "pending review.")
            else:
                verdict = 'NEITHER helps -- stop for review.'
            adoption = {
                'status': 'DECIDED', 'a_helps': a_helps, 'b_helps': b_helps, 'both_help': both_help,
                'a_certified': a_certified, 'b_certified': b_certified, 'verdict': verdict,
            }

        out = {
            'stage': 'P5.15 Addendum 20 - the PF-pace experiment - independent evaluation',
            'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 20',
                          'data/SRP1/Results/P515S38/frozen_s38_pf_pace_spec_v9_7a2b4ab7.json'],
            'LEAD': {
                'arm': arm, 'tau': ARM_TAU[arm],
                'certification_verdict': this_eval['CERTIFICATION']['CERTIFIED'],
                'helps': this_eval['HELPS']['value'],
                'pf_first_pass_cycle': this_eval['FIRST_PASS_CYCLES']['pf'],
                'cycle_count': this_eval['cycles_run'],
                'tso_instability_triggered': (this_eval.get('TSO_INSTABILITY_TRIGGER') or {}).get('TRIGGERED'),
            },
            'this_arm': this_eval,
            'other_arms_present': other_evals,
            'cross_arm': {
                'a_used_for_verdict_arm': a_used_arm_key,
                'certified_by_arm': certified_map, 'helps_by_arm': helps_summary,
                'PREDICTIONS': predictions, 'ADOPTION': adoption,
            },
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)

    print(json.dumps({'LEAD': out['LEAD'], 'CERTIFICATION': out['this_arm']['CERTIFICATION'],
                      'HELPS': out['this_arm']['HELPS'],
                      'TSO_INSTABILITY_TRIGGER': out['this_arm']['TSO_INSTABILITY_TRIGGER'],
                      'cross_arm': {k: v for k, v in out['cross_arm'].items()
                                    if k not in ('certified_by_arm', 'helps_by_arm')}},
                     indent=1, default=str))
    if not dry:
        with open(out_path, 'w') as handle:
            json.dump(out, handle, indent=1, default=str)
        print(f'[S38 evaluate] wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
