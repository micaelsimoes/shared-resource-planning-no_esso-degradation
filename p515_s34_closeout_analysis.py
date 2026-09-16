"""P5.15 Step 3.4 gate (s34) - closeout analysis (zero solves, SolveProfileGuard armed).

Three quantities the gate report's central claims rest on, computed from committed artifacts only:

1. ESS dual-ratio decay. ratio_k = boyd_ess_dual_ratio at cycle k. Decay per 30 cycles over the frozen-rho window is
   d = ratio_150 / ratio_120; cycles to reach 1.0 at that geometric rate = 30 * ln(1/ratio_150) / ln(d). Reported with
   the same computation over 90->150 as a stability check. Extrapolation is descriptive, not a prediction.
2. EFC/day trend. Late slope = (EFC_150 - EFC_120)/30 per cycle; cycles from 150 to the SoH-binding threshold 1.4612
   and to EFC* = 1.8520 at that slope, plus the same from the 90->150 slope. EFC* and the threshold are the frozen
   spec's benchmarks (data/SRP1/Results/P515S34/EFC_benchmark/, commit d1b8cf9c).
3. ESS step persistence. From ess_entry_stride_baseline.jsonl: one record per cycle, holding `entries`, a list of 72
   blocks keyed by (node_id, year, day, power_type), each carrying `z`, the per-period shared-ESS consensus values.
   The flat vector for a cycle is the concatenation of every block's `z` in sorted key order (72 x 24 = 1728 values);
   the key order is asserted identical across cycles. step_k = z_k - z_(k-1); cos(step_k, step_(k-1)) per cycle.
   Spec v4's falsifier is "step rise < 3x AND cos persisting at 1.000"; the rise is 5.14x, so this decides the
   second half.
Write-once output: data/SRP1/Results/P515S34_run/s34_closeout_analysis.json
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S34_run')
OUT = os.path.join(RUN, 's34_closeout_analysis.json')
STRIDE = os.path.join(RUN, 'ess_entry_stride_baseline.jsonl')
SOH, EFC_STAR = 1.4612, 1.8520


def _decay(series, a, b, target=1.0):
    ra, rb = series[a - 1], series[b - 1]
    if not (ra and rb) or rb <= 0 or ra <= 0:
        return None
    d = rb / ra
    span = b - a
    if d >= 1.0 or rb <= target:
        return {'window': [a, b], 'ratio_start': ra, 'ratio_end': rb, 'decay_per_window': d, 'cycles_to_target': None}
    cycles = span * math.log(target / rb) / math.log(d)
    return {'window': [a, b], 'ratio_start': ra, 'ratio_end': rb, 'decay_per_window': d,
            'cycles_to_target_from_end': cycles}


def _cos(u, v):
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    return (sum(x * y for x, y in zip(u, v)) / (nu * nv)) if nu and nv else None


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s34 closeout').install()
    try:
        rows = json.load(open(os.path.join(RUN, 'g_baseline.json')))['cycle_trajectory']
        dual = [r['boyd_ess_dual_ratio'] for r in rows]
        efc = [r.get('efc_per_day_max') for r in rows]
        n = len(rows)

        out = {'stage': 'P5.15 s34 closeout analysis', 'cycles': n,
               'ess_dual_decay': {'terminal_ratio': dual[-1],
                                  'window_120_150': _decay(dual, 120, 150),
                                  'window_90_150': _decay(dual, 90, 150)},
               'efc_trend': {'terminal': efc[-1], 'soh_threshold': SOH, 'efc_star': EFC_STAR}}
        for a, b in ((120, 150), (90, 150)):
            if efc[a - 1] and efc[b - 1]:
                slope = (efc[b - 1] - efc[a - 1]) / (b - a)
                out['efc_trend'][f'slope_{a}_{b}_per_cycle'] = slope
                out['efc_trend'][f'cycles_from_150_to_soh_at_slope_{a}_{b}'] = ((SOH - efc[-1]) / slope) if slope > 0 else None
                out['efc_trend'][f'cycles_from_150_to_efc_star_at_slope_{a}_{b}'] = ((EFC_STAR - efc[-1]) / slope) if slope > 0 else None
        out['efc_trend']['monotone_increasing_fraction'] = sum(
            1 for i in range(1, n) if efc[i] is not None and efc[i - 1] is not None and efc[i] >= efc[i - 1]) / (n - 1)

        # ESS step persistence from the per-entry stride capture
        persistence = {'source': os.path.relpath(STRIDE, REPO)}
        if not os.path.exists(STRIDE):
            persistence.update({'available': False, 'reason': 'stride file absent'})
        else:
            recs, keyorder = {}, None
            with open(STRIDE) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    blocks = r.get('entries')
                    if not isinstance(blocks, list) or not blocks:
                        persistence.update({'available': False, 'reason': f'no entries list; keys={sorted(r)}'})
                        break
                    keyed = sorted(((b['node_id'], str(b['year']), str(b['day']), b['power_type']), b['z'])
                                   for b in blocks)
                    keys = [k for k, _ in keyed]
                    if keyorder is None:
                        keyorder = keys
                    elif keys != keyorder:
                        persistence.update({'available': False, 'reason': 'block ordering differs between cycles'})
                        break
                    vec = []
                    for _, z in keyed:
                        vec.extend(float(x) for x in z)
                    recs[int(r['cycle'])] = vec
            if persistence.get('available') is not False and len(recs) >= 3:
                cycles = sorted(recs)
                lengths = {len(v) for v in recs.values()}
                if len(lengths) != 1:
                    persistence.update({'available': False, 'reason': f'vector lengths differ: {sorted(lengths)}'})
                else:
                    steps = {b: [y - x for x, y in zip(recs[a], recs[b])]
                             for a, b in zip(cycles, cycles[1:])}
                    sc = sorted(steps)
                    cos_series = [(sc[i], _cos(steps[sc[i]], steps[sc[i - 1]])) for i in range(1, len(sc))]
                    cos_series = [(k, c) for k, c in cos_series if c is not None]
                    late = [c for k, c in cos_series if k >= 120]
                    early = [c for k, c in cos_series if k <= 40]
                    persistence.update({
                        'available': True, 'n_cycles_with_vectors': len(recs),
                        'entry_count_per_cycle': lengths.pop(),
                        'cos_sampled': [(k, round(c, 6)) for k, c in cos_series
                                        if k in (10, 30, 60, 90, 120, 140, 150)],
                        'cos_mean_first_40': (sum(early) / len(early)) if early else None,
                        'cos_mean_last_30': (sum(late) / len(late)) if late else None,
                        'cos_min_last_30': min(late) if late else None,
                        'step_norm_sampled': [(k, math.sqrt(sum(x * x for x in steps[k])))
                                              for k in sc if k in (10, 30, 60, 90, 120, 150)],
                    })
        out['ess_step_persistence'] = persistence
        out['falsifier_reading'] = {
            'step_rise_vs_s33e2': 5.144232214072042,
            'threshold': 3.0,
            'cos_criterion': 'spec v4 falsifies the stiffness reading only if the rise is < 3x AND the cosine persists at 1.000',
            'verdict_on_rise': 'not falsified (5.14x > 3x)'}
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
