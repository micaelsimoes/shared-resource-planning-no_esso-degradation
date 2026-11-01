"""P5.15 Addendum 19 - the rho_ess experiment (arms s37_rho0p01, s37_rho0p001) - INDEPENDENT evaluation.

Frozen spec v8, data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json.

Zero solves, SolveProfileGuard armed. Recomputes EVERYTHING from the arm's own per-cycle trajectory
(g_baseline.json) and its ess_entry_stride sidecar (per-entry storage x/z at stride 1) -- never trusts
boyd_terminal.json's own `stopped_by` field (the SAME v1/v2 defect s35ref/s35pt's evaluators found:
`write_boyd_terminal_s35ref` compares `converged_at_cycle` -- the FIRST cycle of the terminal
consecutive-converged run -- against the last cycle, which is wrong whenever more than one consecutive
cycle is required).

CERTIFICATION (spec v8 `certification_per_arm.certified_iff`), all four required:
  (a) Boyd stop on all three channels, 3 consecutive converged cycles, within 150, derived from the
      trajectory;
  (b) no V or PF channel frozen at a rho clamp (the ESS channel is INTENTIONALLY exempt/frozen from
      cycle 1 by this stage's own production change -- excluded from this criterion by the spec's own
      wording);
  (c) storage-channel terminal ratio max(boyd_ess_primal_ratio, boyd_ess_dual_ratio) < 0.9;
  (d) terminal system cost (gross_operational_cost) within the rule-nine bar of run 1 (s35ref):
      |Q_arm - Q_run1| <= terminal_step(arm) + terminal_step(run1); run 1's terminal cost/step are READ
      from its committed g_baseline.json BY PATH, sha256-recorded, never typed in as literals.

ADOPTION (spec v8 `certification_per_arm.adoption`): if both arms certify, adopt the LARGER rho_ess
(0.01); if exactly one certifies, that one; if neither, stop for review. With only one arm's run
present, that arm is reported and adoption is marked PENDING.

PREDICTIONS (spec v8 `predictions_recorded_in_advance`), each evaluated:
  * storage step per cycle scales ~ 1/rho_ess: per-entry RMS storage consensus (P only) step over
    cycles 2-20 -- the arm's own value against run 1's SAME cycles, reporting the ratio against the
    predicted ~11x; and, once both arms exist, 0.001's RMS against 0.01's, against the predicted ~10x;
  * EFC/day reaches >= 1.06 by cycle 50 in the 0.01 arm;
  * at least one arm certifies.

MONITORING (spec v8 `monitoring_reported_not_gated`, exact definitions), reported not gated:
  * local-solve failures and network failures by tier;
  * per-cycle ESS sign-change fraction (storage consensus P entries, all nodes/years/days/periods,
    step z_c - z_(c-1) changing sign relative to z_(c-1) - z_(c-2), entries with |step| < 1e-9 MW
    ignored) and the cosine between consecutive step vectors; OSCILLATION FLAG if the sign-change
    fraction exceeds 0.2 for 10 consecutive cycles or the cosine is negative for 5 consecutive cycles;
  * EFC/day trajectory, per-cycle RMS/max storage consensus step, rho/gamma trajectories, and all
    channel (v/pf/ess) primal/dual ratios, sampled at matched cycles.

Usage: python p515_s37_evaluate.py [RUN_DIR] [--dry-run]
  RUN_DIR defaults to the s37_rho0p01 arm's output root. Writes RUN_DIR/s37_evaluation.json
  (write-once; refuses to overwrite). The sibling arm (the other rho_ess value), if its own run
  directory exists, is read (never re-computed) for the cross-arm predictions and the adoption rule.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
REF_RUN = os.path.join(RES, 'P515S35_REF_run')
CH = ('v', 'pf', 'ess')
CAP = 150
REQUIRED_CONSECUTIVE = 3
EFC_TARGET_BY_CYCLE_50 = 1.06
PREDICTED_STEP_RATIO_ARM_VS_REF = 11.0  # 0.01 arm vs run 1 (rho_ess 0.1125), spec v8 prediction
PREDICTED_STEP_RATIO_0P001_VS_0P01 = 10.0  # spec v8 prediction
STORAGE_TERMINAL_RATIO_BOUND = 0.9
OSCILLATION_SIGN_CHANGE_THRESHOLD = 0.2
OSCILLATION_SIGN_CHANGE_STREAK = 10
OSCILLATION_COSINE_STREAK = 5
RMS_WINDOW = (2, 20)  # cycles 2-20 inclusive, spec v8 prediction
MATCH = (1, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150)

ARM_ROOTS = {
    's37_rho0p01': os.path.join(RES, 'P515S37_RHO0P01_run'),
    's37_rho0p001': os.path.join(RES, 'P515S37_RHO0P001_run'),
}
ARM_RHO_ESS = {'s37_rho0p01': 0.01, 's37_rho0p001': 0.001}


def _sha256(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _identify_arm(run_dir):
    run_dir = os.path.abspath(run_dir)
    for arm, root in ARM_ROOTS.items():
        if os.path.abspath(root) == run_dir:
            return arm
    raise RuntimeError(f'{run_dir} does not match either s37 arm output root: {ARM_ROOTS}')


def _rows(run_dir):
    hits = glob.glob(os.path.join(run_dir, 'g_*.json'))
    if len(hits) != 1:
        raise RuntimeError(f'expected one g_*.json in {run_dir}, found {hits}')
    g = json.load(open(hits[0]))
    return g, hits[0]


def _stop_from_trajectory(rows, cap, required):
    """v2 method (s35ref/s35pt evaluators' own fix): stop cause derived from
    the trajectory itself, never from boyd_terminal.json's `stopped_by`
    field, which is wrong whenever more than one consecutive converged
    cycle is required (production records the FIRST cycle of the
    consecutive run, not the last)."""
    allpass = {r['cycle'] for r in rows if r.get('boyd_all_pass')}
    tail = [r['cycle'] for r in rows[-required:]] if len(rows) >= required else []
    ok = (len(tail) == required and all(c in allpass for c in tail)
          and all(tail[i] + 1 == tail[i + 1] for i in range(len(tail) - 1))
          and len(rows) < cap)
    return {'stopped_by': 'boyd' if ok else 'cap', 'stop_run_cycles': tail if ok else None,
            'cycles': len(rows), 'cap': cap, 'required_consecutive_cycles': required}


def _load_ess_stride_p_vectors(run_dir):
    """{cycle: numpy array} -- storage consensus z (P power_type only),
    flattened over (node_id, year, day, period) in the sidecar's own
    per-cycle entry order (stable across cycles: `s34_capture_hooks`
    iterates the SAME node/year/day collections in the SAME order every
    cycle)."""
    hits = glob.glob(os.path.join(run_dir, 'ess_entry_stride_*.jsonl'))
    vectors = {}
    if not hits:
        return vectors, None
    path = hits[0]
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cycle = rec.get('cycle')
            flat = []
            for entry in rec.get('entries', []):
                if entry.get('power_type') != 'p':
                    continue
                flat.extend(entry.get('z', []))
            vectors[cycle] = np.asarray(flat, dtype=float)
    return vectors, path


def _step_series(vectors):
    """{cycle: {'step': ndarray, 'rms': float, 'max_abs': float}} for every
    cycle c with both c and c-1 captured."""
    out = {}
    for c in sorted(vectors):
        if (c - 1) not in vectors:
            continue
        cur, prev = vectors[c], vectors[c - 1]
        if cur.shape != prev.shape or cur.size == 0:
            continue
        step = cur - prev
        out[c] = {'step': step, 'rms': float(np.sqrt(np.mean(step ** 2))),
                  'max_abs': float(np.max(np.abs(step)))}
    return out


def _rms_over_window(vectors, lo, hi):
    parts = []
    for c in range(lo, hi + 1):
        if c in vectors and (c - 1) in vectors and vectors[c].shape == vectors[c - 1].shape:
            parts.append(vectors[c] - vectors[c - 1])
    if not parts:
        return None
    all_steps = np.concatenate(parts)
    return float(np.sqrt(np.mean(all_steps ** 2))) if all_steps.size else None


def _oscillation_series(step_series):
    """Per spec v8 `monitoring_reported_not_gated.ess_step_direction_sign_
    changes`: at cycle c, compares step_c (= z_c - z_(c-1)) against
    step_(c-1) (= z_(c-1) - z_(c-2)) -- sign-change fraction over entries
    with |step_c| >= 1e-9 MW, plus the cosine between the two step
    vectors (over ALL entries, unfiltered)."""
    out = {}
    for c in sorted(step_series):
        prev = c - 1
        if prev not in step_series:
            continue
        cur_step = step_series[c]['step']
        prev_step = step_series[prev]['step']
        if cur_step.shape != prev_step.shape or cur_step.size == 0:
            continue
        mask = np.abs(cur_step) >= 1e-9
        n_considered = int(mask.sum())
        sign_change_fraction = (
            float(np.mean(np.sign(cur_step[mask]) != np.sign(prev_step[mask])))
            if n_considered else None)
        denom = float(np.linalg.norm(cur_step) * np.linalg.norm(prev_step))
        cosine = float(np.dot(cur_step, prev_step) / denom) if denom > 0 else None
        out[c] = {'sign_change_fraction': sign_change_fraction, 'cosine_vs_previous_step': cosine,
                  'n_entries_considered': n_considered, 'n_entries_total': int(cur_step.size)}
    return out


def _oscillation_flag(osc_series):
    cycles = sorted(osc_series)
    streak_frac, streak_cos = 0, 0
    flagged_cycle, flagged_reason = None, None
    for c in cycles:
        e = osc_series[c]
        frac, cos = e['sign_change_fraction'], e['cosine_vs_previous_step']
        streak_frac = streak_frac + 1 if (frac is not None and frac > OSCILLATION_SIGN_CHANGE_THRESHOLD) else 0
        streak_cos = streak_cos + 1 if (cos is not None and cos < 0) else 0
        if flagged_cycle is None and streak_frac >= OSCILLATION_SIGN_CHANGE_STREAK:
            flagged_cycle, flagged_reason = c, f'sign_change_fraction > {OSCILLATION_SIGN_CHANGE_THRESHOLD} for {OSCILLATION_SIGN_CHANGE_STREAK} consecutive cycles'
        if flagged_cycle is None and streak_cos >= OSCILLATION_COSINE_STREAK:
            flagged_cycle, flagged_reason = c, f'cosine < 0 for {OSCILLATION_COSINE_STREAK} consecutive cycles'
    return {'flag': flagged_cycle is not None, 'first_flagged_cycle': flagged_cycle, 'reason': flagged_reason}


def _channel_ratio_trajectory(rows, match_cycles):
    out = []
    for r in rows:
        if r['cycle'] not in match_cycles and r['cycle'] != rows[-1]['cycle']:
            continue
        entry = {'cycle': r['cycle']}
        for c in CH:
            entry[f'{c}_primal_ratio'] = r.get(f'boyd_{c}_primal_ratio')
            entry[f'{c}_dual_ratio'] = r.get(f'boyd_{c}_dual_ratio')
            entry[f'{c}_dual_ratio_balance'] = r.get(f'boyd_{c}_dual_ratio_balance')
            entry[f'{c}_channel_pass'] = r.get(f'boyd_{c}_channel_pass')
        out.append(entry)
    return out


def _rho_gamma_trajectory(rows, match_cycles):
    out = []
    for r in rows:
        if r['cycle'] not in match_cycles and r['cycle'] != rows[-1]['cycle']:
            continue
        entry = {'cycle': r['cycle']}
        for c in CH:
            entry[f'rho_{c}'] = r.get(f'rho_{c}_after')
            entry[f'rho_{c}_action'] = r.get(f'rho_{c}_action')
            entry[f'gamma_{c}'] = r.get(f'gamma_{c}_after')
            entry[f'balancing_exempt_{c}'] = r.get(f'balancing_exempt_{c}')
        out.append(entry)
    return out


def _efc_trajectory(rows, match_cycles):
    return [{'cycle': r['cycle'], 'efc_per_day_max': r.get('efc_per_day_max')}
            for r in rows if r['cycle'] in match_cycles or r['cycle'] == rows[-1]['cycle']]


def _failures_by_tier(run_dir, rows):
    hits = glob.glob(os.path.join(run_dir, 'network_failures_*.jsonl'))
    if not hits:
        return {'available': False}
    events = [json.loads(l) for l in open(hits[0]) if l.strip()]
    classes = ('recovered_tier1', 'recovered_tier2', 'unrecovered', 'not_attempted', 'indeterminate')
    counts = {c: sum(1 for e in events if e.get('class') == c) for c in classes}
    counts['recovered'] = counts['recovered_tier1'] + counts['recovered_tier2']
    return {'available': True, 'n_events': len(events), 'classes': counts,
            'per_cycle': (len(events) / len(rows)) if rows else None}


def _evaluate_arm(arm, run_dir):
    g, g_path = _rows(run_dir)
    rows = g['cycle_trajectory']
    if not rows:
        raise RuntimeError(f'{run_dir}: empty cycle_trajectory')
    last = rows[-1]
    required = last.get('required_consecutive_cycles') or REQUIRED_CONSECUTIVE
    stop = _stop_from_trajectory(rows, CAP, required)
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

    vectors, stride_path = _load_ess_stride_p_vectors(run_dir)
    step_series = _step_series(vectors)
    osc_series = _oscillation_series(step_series)
    osc_flag = _oscillation_flag(osc_series)
    rms_2_20 = _rms_over_window(vectors, *RMS_WINDOW)

    ref_vectors, ref_stride_path = _load_ess_stride_p_vectors(REF_RUN)
    ref_rms_2_20 = _rms_over_window(ref_vectors, *RMS_WINDOW)
    step_ratio_vs_ref = (rms_2_20 / ref_rms_2_20) if (rms_2_20 is not None and ref_rms_2_20) else None

    efc_traj = _efc_trajectory(rows, MATCH)
    efc_by_cycle = {r['cycle']: r.get('efc_per_day_max') for r in rows}
    efc_at_or_before_50 = [v for c, v in efc_by_cycle.items() if c <= 50 and v is not None]
    efc_prediction = None
    if arm == 's37_rho0p01':
        efc_at_50 = efc_by_cycle.get(50)
        max_by_50 = max(efc_at_or_before_50) if efc_at_or_before_50 else None
        efc_prediction = {
            'target': EFC_TARGET_BY_CYCLE_50, 'efc_at_cycle_50': efc_at_50,
            'max_efc_by_cycle_50': max_by_50,
            'pass': bool(max_by_50 is not None and max_by_50 >= EFC_TARGET_BY_CYCLE_50),
        }

    settling = {c: max(last.get(f'boyd_{c}_primal_ratio') or 0.0, last.get(f'boyd_{c}_dual_ratio') or 0.0)
                for c in CH}

    monitoring = {
        'local_solve_failures': g.get('local_solve_failures'),
        'network_failures_by_tier': _failures_by_tier(run_dir, rows),
        'oscillation': {
            'per_cycle_sampled': [{'cycle': c, **osc_series[c]} for c in sorted(osc_series)
                                   if c in MATCH or c == rows[-1]['cycle']],
            'oscillation_flag': osc_flag,
            'definition': ('sign-change fraction over storage-consensus P entries (all nodes/years/days/'
                           'periods) with |step_c| >= 1e-9 MW, step_c = z_c - z_(c-1) vs step_(c-1); '
                           'cosine between consecutive step vectors (unfiltered); flag if sign-change '
                           f'fraction > {OSCILLATION_SIGN_CHANGE_THRESHOLD} for {OSCILLATION_SIGN_CHANGE_STREAK} '
                           f'consecutive cycles or cosine < 0 for {OSCILLATION_COSINE_STREAK} consecutive cycles'),
        },
        'step_sizes': {
            'per_cycle_sampled': [{'cycle': c, 'rms': step_series[c]['rms'], 'max_abs': step_series[c]['max_abs']}
                                   for c in sorted(step_series) if c in MATCH or c == rows[-1]['cycle']],
            'rms_over_cycles_2_20': rms_2_20,
            'path': os.path.relpath(stride_path, REPO) if stride_path else None,
        },
        'rho_gamma_trajectory_sampled': _rho_gamma_trajectory(rows, MATCH),
        'channel_ratios_sampled': _channel_ratio_trajectory(rows, MATCH),
        'efc_trajectory_sampled': efc_traj,
    }

    return {
        'arm': arm, 'rho_ess': ARM_RHO_ESS[arm], 'run_dir': os.path.relpath(run_dir, REPO),
        'g_baseline_path': os.path.relpath(g_path, REPO), 'g_baseline_sha256': _sha256(g_path),
        'instance': g.get('instance'), 'cycles_run': len(rows),
        'CERTIFICATION': {
            'criterion_a_boyd_stop_within_150': {**stop, 'pass': crit_a},
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
                'valid_note': 'valid because both runs are then settled (criterion a already requires the arm '
                              'itself to have stopped under Boyd)',
                'pass': crit_d},
            'CERTIFIED': certified,
        },
        'PREDICTIONS': {
            'step_size_scaling_vs_run1': {
                'arm_rms_step_cycles_2_20': rms_2_20, 'run1_rms_step_cycles_2_20': ref_rms_2_20,
                'ratio_arm_over_run1': step_ratio_vs_ref, 'predicted_ratio': PREDICTED_STEP_RATIO_ARM_VS_REF,
                'run1_stride_path': os.path.relpath(ref_stride_path, REPO) if ref_stride_path else None,
            },
            'efc_1p06_by_cycle_50_0p01_arm_only': efc_prediction,
        },
        'settling_quality': {**settling, 'objective_rule_ten': last.get('objective_change_ratio'),
                             'terminal_step_over_tolerance': (
                                 (last.get('objective_change_abs') / last.get('objective_tolerance'))
                                 if last.get('objective_change_abs') and last.get('objective_tolerance') else None)},
        'wall_clock_s': g.get('wall_clock_s'), 'solves': g.get('solve_profile'),
        'monitoring': monitoring,
        '_rms_2_20': rms_2_20,  # internal, consumed by the cross-arm section below; stripped before write
    }


def main(argv):
    dry = '--dry-run' in argv
    args = [a for a in argv if not a.startswith('--')]
    run_dir = os.path.abspath(args[0]) if args else ARM_ROOTS['s37_rho0p01']
    arm = _identify_arm(run_dir)
    sibling_arm = 's37_rho0p001' if arm == 's37_rho0p01' else 's37_rho0p01'
    out_path = os.path.join(run_dir, 's37_evaluation.json')
    if not dry and os.path.exists(out_path):
        raise RuntimeError(f'refusing to overwrite {out_path}')
    if not os.path.exists(os.path.join(REF_RUN, 'g_baseline.json')):
        raise RuntimeError(f'run 1 (s35ref) reference not found at {REF_RUN}/g_baseline.json')

    guard = SolveProfileGuard(permitted=(), label='P5.15 s37 independent evaluation').install()
    try:
        this_eval = _evaluate_arm(arm, run_dir)

        sibling_root = ARM_ROOTS[sibling_arm]
        sibling_present = bool(glob.glob(os.path.join(sibling_root, 'g_*.json')))
        sibling_eval = _evaluate_arm(sibling_arm, sibling_root) if sibling_present else None

        cross_arm = {'sibling_arm': sibling_arm, 'sibling_present': sibling_present}
        if sibling_present:
            this_rms, sib_rms = this_eval['_rms_2_20'], sibling_eval['_rms_2_20']
            # 0.001 / 0.01 ratio, oriented regardless of which arm this evaluation is for.
            rms_001 = this_rms if arm == 's37_rho0p001' else sib_rms
            rms_01 = this_rms if arm == 's37_rho0p01' else sib_rms
            ratio_0p001_over_0p01 = (rms_001 / rms_01) if (rms_001 is not None and rms_01) else None
            cross_arm['step_size_scaling_0p001_vs_0p01'] = {
                'rms_step_0p001_cycles_2_20': rms_001, 'rms_step_0p01_cycles_2_20': rms_01,
                'ratio_0p001_over_0p01': ratio_0p001_over_0p01,
                'predicted_ratio': PREDICTED_STEP_RATIO_0P001_VS_0P01,
            }
            both_certify = this_eval['CERTIFICATION']['CERTIFIED'] and sibling_eval['CERTIFICATION']['CERTIFIED']
            either_certifies = this_eval['CERTIFICATION']['CERTIFIED'] or sibling_eval['CERTIFICATION']['CERTIFIED']
            if both_certify:
                adopted = 's37_rho0p01'  # larger rho_ess
                adopted_rho_ess = ARM_RHO_ESS[adopted]
                verdict = 'ADOPTED (both certify; larger rho_ess per spec v8 adoption rule)'
            elif either_certifies:
                adopted = arm if this_eval['CERTIFICATION']['CERTIFIED'] else sibling_arm
                adopted_rho_ess = ARM_RHO_ESS[adopted]
                verdict = 'ADOPTED (exactly one arm certifies)'
            else:
                adopted, adopted_rho_ess = None, None
                verdict = 'NEITHER CERTIFIES -- stop for review (Option 1, settlement active at initialization, is the parked fallback)'
            cross_arm['ADOPTION'] = {
                'status': 'DECIDED', 'both_certify': both_certify, 'either_certifies': either_certifies,
                'adopted_arm': adopted, 'adopted_rho_ess': adopted_rho_ess, 'verdict': verdict,
            }
        else:
            cross_arm['ADOPTION'] = {
                'status': 'PENDING', 'reason': f'sibling arm {sibling_arm} run directory not found: {sibling_root}',
                'this_arm_certified': this_eval['CERTIFICATION']['CERTIFIED'],
            }
        cross_arm['at_least_one_arm_certifies'] = (
            this_eval['CERTIFICATION']['CERTIFIED'] or (sibling_eval['CERTIFICATION']['CERTIFIED'] if sibling_eval else False))

        out = {
            'stage': 'P5.15 Addendum 19 - the rho_ess experiment - independent evaluation',
            'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 19',
                          'data/SRP1/Results/P515S37/frozen_s37_rho_ess_spec_v8_f91de983.json'],
            'LEAD': {
                'arm': arm, 'rho_ess': ARM_RHO_ESS[arm],
                'certification_verdict': this_eval['CERTIFICATION']['CERTIFIED'],
                'cycle_count': this_eval['cycles_run'],
                'efc_trajectory_sampled': this_eval['monitoring']['efc_trajectory_sampled'],
            },
            'this_arm': {k: v for k, v in this_eval.items() if not k.startswith('_')},
            'cross_arm': cross_arm,
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)

    print(json.dumps({'LEAD': out['LEAD'], 'CERTIFICATION': out['this_arm']['CERTIFICATION'],
                      'PREDICTIONS': out['this_arm']['PREDICTIONS'], 'cross_arm': {
                          k: v for k, v in out['cross_arm'].items() if k != 'sibling_present'}},
                     indent=1, default=str))
    if not dry:
        with open(out_path, 'w') as handle:
            json.dump(out, handle, indent=1, default=str)
        print(f'[S37 evaluate] wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
