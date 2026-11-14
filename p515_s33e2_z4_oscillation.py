"""P5.15 Addendum 15 item 4 - Z4: attribute the ~30-cycle oscillation in gate s33e2 (zero solves, guard armed).

Question (Planner, Addendum 15 item 4): `gross_operational_cost` in s33e2 does not settle monotonically over
cycles 101-150; it oscillates with an apparent period ~30 cycles. This script characterises the oscillation and
tests explicit coupling hypotheses. It performs no solves and writes only to a new directory.

All formulas used below are reproduced here (not only in code) so the report is self-contained.

1. Detrending (two methods, both computed; the report states which is treated as primary and why):
   a. Centered moving average, half-window h = window // 2:
        trend_i   = mean( x_{max(0,i-h)} .. x_{min(n,i+h+1)-1} )      (window shrinks at the edges)
        resid_i   = x_i - trend_i
      Default window = 9 cycles (chosen smaller than the ~20-30 cycle period visible by eye in the raw series,
      so the moving average tracks the slow contraction without also averaging away the oscillation itself;
      this is a design choice, not a fitted one, and is stated as such).
   b. Linear (OLS) detrend over the window's own local cycle index i = 0..n-1:
        x_i = a + b*i + resid_i   (a, b by ordinary least squares)
   The moving-average residual is treated as primary because the raw series is visibly non-linear (fast
   contraction before ~cycle 60, near-flat after) over the 31-150 window; the linear residual is reported
   alongside as a robustness check, and periods are reported as robust when both agree within +/-15%.

2. Autocorrelation (biased estimator), on the mean-zero residual r (r = resid - mean(resid)), lags k=0..K:
        rho(k) = [ sum_{t=0}^{n-k-1} r_t * r_{t+k} ] / [ sum_{t=0}^{n-1} r_t^2 ]
   Candidate periods are the lags k>=3 that are local maxima of rho (rho(k) > rho(k-1) and rho(k) > rho(k+1)),
   ranked by rho(k) descending; top 3 reported.

3. Discrete Fourier transform, on the same mean-zero residual r, length n:
        X_k = sum_{t=0}^{n-1} r_t * exp(-2*pi*i*k*t/n),   k = 1 .. floor(n/2)
        power_k  = |X_k|^2
        period_k = n / k
        amplitude_k = 2*|X_k| / n     (peak amplitude of the real sinusoid at that frequency)
   Top 3 k by power_k reported (k=0, the mean, is excluded by construction since r is mean-zero).

4. Amplitude-vs-tolerance: the DFT amplitude of the dominant period and the residual's peak-to-peak
   (max(resid) - min(resid)) are both reported as multiples of `objective_tolerance` read from the run's own
   trajectory (objective_relative_tolerance * |gross_operational_cost|, terminal value; matches the Planner's
   65,383 figure at these cycles).

3b. Extrema/envelope decay analysis (added because sections 2-3 showed amplitude falling by roughly an order of
   magnitude within the 31-150 window, which biases both ACF and DFT period estimates - both assume a
   stationary amplitude): a first pass of naive local-extremum detection on the raw MA-detrended residual picks
   up many single-cycle noise wiggles, not the slow oscillation, so extrema are located on a further-smoothed
   copy s = moving_average_trend(r, window=3) (same centered-moving-average formula as detrend step 1a,
   window=3, applied a second time purely to reject single-cycle noise before locating turning points); the
   reported extremum *value* is still r (the 9-cycle-detrended residual) evaluated at that cycle, not s. On s,
        local maximum at cycle c_i  if  s_{i-1} < s_i > s_{i+1}
        local minimum at cycle c_i  if  s_{i-1} > s_i < s_{i+1}
   giving an alternating sequence of extrema (c_i, r_i). Two things are computed from this sequence:
        period_from_extrema  = mean of (c_{i+2} - c_i) over same-type consecutive extrema (peak-to-peak,
                                trough-to-trough), i.e. one full cycle each
        envelope decay        = OLS fit of ln|r_i| against c_i over the extrema sequence:
                                ln|r_i| = ln(A0) - k*c_i + noise
                                decay_per_cycle = exp(-k) (multiplicative amplitude ratio per cycle)
                                half_life_cycles = ln(2) / k
   Separately, zero-crossings of r (linear-interpolated crossing cycle between i and i+1 where sign(r_i) !=
   sign(r_{i+1})) give a third, amplitude-independent period estimate:
        period_from_zero_crossings = 2 * mean( consecutive same-direction zero-crossing spacing )
   (a full period is two zero-crossings of the same direction, or four crossings total; using same-direction
   crossings avoids the duty-cycle asymmetry of an asymmetric waveform).

5. Cross-correlation / phase, between two mean-zero residual series r_x (reference, cost) and r_y (candidate),
   both restricted to the common window, at integer lag L in [-10, 10]:
        ccf(L) = Pearson_corr( r_x[t], r_y[t+L] )   over the overlapping range of t
   Reported lag* = argmax_L |ccf(L)|, ccf(lag*). Convention: lag* > 0 means r_y's value `lag*` cycles LATER
   lines up with today's r_x, i.e. r_y (the candidate series) LAGS the cost series by lag* cycles. lag* < 0
   means the candidate LEADS the cost series by |lag*| cycles.

6. "Shares the period": a series' own DFT-dominant period (via step 3, same window) is within +/-15% of the
   cost series' DFT-dominant period (same window).

7. Failure-cycle phase lock, given period P (the cost series' dominant period over the relevant window) and the
   set of failure cycles F falling in that window:
        phase_i = 2*pi * (cycle_i mod P) / P,   for cycle_i in F
        R = | (1/|F|) * sum_i exp(1j * phase_i) |         (circular resultant length, in [0, 1])
   Null distribution: M=5000 uniform-without-replacement draws of |F| cycles from the window's full cycle set,
   R computed the same way for each draw; empirical p-value = fraction of draws with R_draw >= R_observed.
   R near 1 (and a small empirical p-value) supports phase-locking; R near 0 (large p-value) argues against it.

8. Cost-step vs failure-cycle magnitude: the signed per-cycle cost step
        step_k = gross_operational_cost_k - gross_operational_cost_{k-1}   (== recourse_change_k, cross-checked)
   is compared at failure cycles vs non-failure cycles in the window (mean, median, and a Mann-Whitney U test,
   scipy.stats.mannwhitneyu, two-sided - a standard non-parametric rank-sum test for whether the two samples
   come from the same distribution, used here rather than a t-test because n(failures) is small and step_k is
   not assumed normal).

9. Channel-coupling lagged correlation (section 4c): for each candidate channel series x and lag L in {0,1,2,3},
        r(L) = Pearson_corr( step_k, x_{k-L} )     over cycles k in the window
   i.e. the channel's raw value L cycles earlier against today's signed cost step. Best (channel, L) is the one
   maximising |r(L)|.

10. ESSO throughput / net-imbalance per cycle, from esso_capture/baseline/node{5,7,9}_cycle{k:03d}.jsonl (one
    row per (year, day, period) entry, pnet = pch - pdch, charging positive):
        throughput_k   = sum over all entries and nodes of (pch + pdch)     ("how much the shared ESS is moving")
        net_abs_pnet_k = sum over all entries and nodes of |pnet|          ("aggregate net activity")
    both computed per node too. class == 'indeterminate' (vs 'barrier-set') in the same capture is used as a
    per-cycle count of complementarity-indeterminate entries, a proxy for active-set change (section 4e); this
    is the finest per-cycle bound/active-set signal available in the committed captures - see the script's
    stdout for what is NOT recoverable (the RECOURSE JUMP block breakdown is only emitted through cycle 62; see
    section 4d in the report).

Sources (all read-only, already committed): data/SRP1/Results/P515S33_E2_run/{g_baseline.json,
network_failures_baseline.jsonl, stdout_baseline.log, esso_capture/baseline/*.jsonl}; and, for contrast,
data/SRP1/Results/P515S32_run/g_baseline.json and data/SRP1/Results/P515S31C_run/g_baseline.json.

Write-once outputs (refuses if the directory already exists): data/SRP1/Results/P515S33/Z4/z4_oscillation.json,
data/SRP1/Results/P515S33/Z4/z4_manifest_sha256.json.
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
from scipy.stats import mannwhitneyu

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RES = os.path.join(REPO, 'data', 'SRP1', 'Results')
RUN = os.path.join(RES, 'P515S33_E2_run')
S32_G = os.path.join(RES, 'P515S32_run', 'g_baseline.json')
S31C_G = os.path.join(RES, 'P515S31C_run', 'g_baseline.json')
OUT_DIR = os.path.join(RES, 'P515S33', 'Z4')
OUT_JSON = os.path.join(OUT_DIR, 'z4_oscillation.json')
OUT_MANIFEST = os.path.join(OUT_DIR, 'z4_manifest_sha256.json')

CH = ('v', 'pf', 'ess')
NODES = (5, 7, 9)
MA_WINDOW = 9
MAX_ACF_LAG = 40
TOP_N = 3
XCORR_MAX_LAG = 10
CHANNEL_LAGS = (0, 1, 2, 3)
N_PERMUTATIONS = 5000
RNG_SEED = 20260916

FAILURE_CYCLES = (3, 7, 13, 17, 34, 36, 37, 40, 60, 73, 79, 109, 111, 116, 118, 119, 126, 138)


# ---------- generic series tools (formulas in module docstring) ----------

def moving_average_trend(x, window=MA_WINDOW):
    x = np.asarray(x, dtype=float)
    n = len(x)
    h = window // 2
    trend = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        trend[i] = x[lo:hi].mean()
    return trend


def linear_trend(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    idx = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(idx, x, 1)  # np.polyfit(idx, x, 1) returns [slope, intercept]
    return intercept + slope * idx


def autocorrelation(r, max_lag):
    r = np.asarray(r, dtype=float)
    n = len(r)
    denom = float(np.dot(r, r))
    max_lag = min(max_lag, n - 2)
    rho = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        rho[k] = (np.dot(r[:n - k], r[k:]) / denom) if denom > 0 else float('nan')
    return rho


def acf_top_periods(r, max_lag=MAX_ACF_LAG, top=TOP_N):
    rho = autocorrelation(r, max_lag)
    peaks = [k for k in range(3, len(rho) - 1) if rho[k] > rho[k - 1] and rho[k] > rho[k + 1]]
    peaks.sort(key=lambda k: rho[k], reverse=True)
    return [{'period_cycles': k, 'acf': float(rho[k])} for k in peaks[:top]]


def dft_top_periods(r, top=TOP_N):
    r = np.asarray(r, dtype=float)
    n = len(r)
    X = np.fft.rfft(r)
    power = np.abs(X) ** 2
    ks = np.arange(1, len(X))  # exclude k=0 (mean, already removed)
    order = ks[np.argsort(power[1:])[::-1]]
    out = []
    for k in order[:top]:
        out.append({'period_cycles': float(n / k), 'k': int(k),
                     'amplitude': float(2 * np.abs(X[k]) / n), 'power': float(power[k])})
    return out


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def cross_correlation_lag(r_x, r_y, max_lag=XCORR_MAX_LAG):
    r_x = np.asarray(r_x, dtype=float)
    r_y = np.asarray(r_y, dtype=float)
    n = len(r_x)
    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs, ys = r_x[: n - lag] if lag > 0 else r_x, r_y[lag:]
        else:
            xs, ys = r_x[-lag:], r_y[: n + lag]
        if len(xs) < 10:
            continue
        c = pearson(xs, ys)
        if c is None:
            continue
        if best is None or abs(c) > abs(best[1]):
            best = (lag, c)
    return {'lag': best[0], 'ccf': best[1]} if best else None


def extrema_envelope(cycles, resid0):
    """Peak/trough sequence, period-from-extrema-spacing, ln|amplitude| decay fit, zero-crossing period.
    Formulas in the module docstring, section 3b."""
    cycles = np.asarray(cycles)
    r = np.asarray(resid0, dtype=float)
    n = len(r)
    s = moving_average_trend(r, window=3)
    peaks = [i for i in range(1, n - 1) if s[i - 1] < s[i] > s[i + 1]]
    troughs = [i for i in range(1, n - 1) if s[i - 1] > s[i] < s[i + 1]]
    extrema = sorted(peaks + troughs)

    def spacing_period(idx_list):
        if len(idx_list) < 2:
            return None
        cs = cycles[idx_list]
        diffs = np.diff(cs)
        return float(diffs.mean()), [int(d) for d in diffs]

    peak_period = spacing_period(peaks)
    trough_period = spacing_period(troughs)

    decay = None
    if len(extrema) >= 3:
        cs = cycles[extrema].astype(float)
        amps = np.abs(r[extrema])
        amps = np.where(amps <= 0, np.nan, amps)
        mask = ~np.isnan(amps)
        if mask.sum() >= 3:
            k_neg, ln_a0 = np.polyfit(cs[mask], np.log(amps[mask]), 1)
            k = -k_neg
            pred = ln_a0 + k_neg * cs[mask]
            resid_fit = np.log(amps[mask]) - pred
            ss_res = float(np.sum(resid_fit ** 2))
            ss_tot = float(np.sum((np.log(amps[mask]) - np.log(amps[mask]).mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
            decay = {
                'n_extrema_used': int(mask.sum()),
                'decay_rate_k_per_cycle': float(k),
                'multiplicative_decay_per_cycle': float(np.exp(-k)),
                'half_life_cycles': float(np.log(2) / k) if k > 0 else None,
                'ln_amplitude_fit_r_squared': r2,
                'A0_at_first_extremum_cycle': float(np.exp(ln_a0 + k_neg * cs[mask][0])),
            }

    zero_cross_cycles = []
    for i in range(n - 1):
        if r[i] == 0:
            continue
        if np.sign(r[i]) != np.sign(r[i + 1]):
            frac = abs(r[i]) / (abs(r[i]) + abs(r[i + 1]))
            zero_cross_cycles.append(float(cycles[i]) + frac * float(cycles[i + 1] - cycles[i]))
    # direction of each crossing: from r[i] sign to r[i+1] sign
    directions = []
    for i in range(n - 1):
        if r[i] == 0:
            continue
        if np.sign(r[i]) != np.sign(r[i + 1]):
            directions.append('up' if r[i] < r[i + 1] else 'down')
    same_dir_period = None
    for direction in ('up', 'down'):
        cs = [c for c, d in zip(zero_cross_cycles, directions) if d == direction]
        if len(cs) >= 2:
            diffs = np.diff(cs)
            same_dir_period = float(diffs.mean())
            break

    return {
        'peak_cycles': [int(cycles[i]) for i in peaks], 'peak_values': [float(r[i]) for i in peaks],
        'trough_cycles': [int(cycles[i]) for i in troughs], 'trough_values': [float(r[i]) for i in troughs],
        'period_from_peak_spacing_mean_cycles': peak_period[0] if peak_period else None,
        'peak_spacings': peak_period[1] if peak_period else None,
        'period_from_trough_spacing_mean_cycles': trough_period[0] if trough_period else None,
        'trough_spacings': trough_period[1] if trough_period else None,
        'envelope_decay_fit': decay,
        'zero_crossing_cycles': [round(c, 2) for c in zero_cross_cycles],
        'period_from_zero_crossings_cycles': same_dir_period,
    }


def detrend_both(cost_like):
    ma = cost_like - moving_average_trend(cost_like)
    lin = cost_like - linear_trend(cost_like)
    return ma, lin


# ---------- data loading ----------

def load_trajectory(path):
    return json.load(open(path))['cycle_trajectory']


def series(rows, field):
    return np.array([r[field] for r in rows], dtype=float)


def window_rows(rows, lo, hi):
    return [r for r in rows if lo <= r['cycle'] <= hi]


def esso_percycle(run_dir, cycles, nodes=NODES):
    cap = os.path.join(run_dir, 'esso_capture', 'baseline')
    throughput_total, net_abs_total = [], []
    net_abs_by_node = {n: [] for n in nodes}
    indeterminate_count = []
    for cyc in cycles:
        tp, na, ic = 0.0, 0.0, 0
        na_node = {n: 0.0 for n in nodes}
        for n in nodes:
            path = os.path.join(cap, f'node{n}_cycle{cyc:03d}.jsonl')
            with open(path) as h:
                for line in h:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    pch, pdch, pnet = row['pch'], row['pdch'], row['pnet']
                    tp += pch + pdch
                    na += abs(pnet)
                    na_node[n] += abs(pnet)
                    if row.get('class') == 'indeterminate':
                        ic += 1
                    if row.get('class_bar') == 'indeterminate':
                        ic += 1
        throughput_total.append(tp)
        net_abs_total.append(na)
        indeterminate_count.append(ic)
        for n in nodes:
            net_abs_by_node[n].append(na_node[n])
    return {
        'throughput_total': np.array(throughput_total),
        'net_abs_pnet_total': np.array(net_abs_total),
        'net_abs_pnet_by_node': {str(n): np.array(v) for n, v in net_abs_by_node.items()},
        'indeterminate_count': np.array(indeterminate_count, dtype=float),
    }


RECOURSE_JUMP_CYCLE_RE = re.compile(r'^\[RECOURSE JUMP\] cycle=(\d+) \| abs_change=([-\d.eE+]+) \| tol=([-\d.eE+]+) \| signed_change=([-\d.eE+]+)$')
AGGREGATE_LINE_RE = re.compile(r'^\s{2}(TSO|DSO node=(\d+)|SALVAGE) \| delta=([-\d.eE+]+)$')


def parse_recourse_jump_blocks(stdout_path):
    """Parse the [RECOURSE JUMP] ... Aggregate signed changes: blocks. Returns {cycle: {agent_label: delta}}."""
    out = {}
    cur_cycle = None
    in_aggregate = False
    with open(stdout_path) as h:
        for line in h:
            line = line.rstrip('\n')
            m = RECOURSE_JUMP_CYCLE_RE.match(line)
            if m:
                cur_cycle = int(m.group(1))
                out[cur_cycle] = {}
                in_aggregate = False
                continue
            if line == '[RECOURSE JUMP] Aggregate signed changes:':
                in_aggregate = True
                continue
            if in_aggregate:
                m2 = AGGREGATE_LINE_RE.match(line)
                if m2:
                    label = m2.group(1)
                    out[cur_cycle][label] = float(m2.group(3))
                    continue
                else:
                    in_aggregate = False
    return out


# ---------- section builders ----------

def characterise_cost(rows, lo, hi, label):
    wrows = window_rows(rows, lo, hi)
    cycles = [r['cycle'] for r in wrows]
    gc = series(wrows, 'gross_operational_cost')
    ma_resid, lin_resid = detrend_both(gc)
    ma_resid0 = ma_resid - ma_resid.mean()
    lin_resid0 = lin_resid - lin_resid.mean()
    acf_ma = acf_top_periods(ma_resid0)
    dft_ma = dft_top_periods(ma_resid0)
    acf_lin = acf_top_periods(lin_resid0)
    dft_lin = dft_top_periods(lin_resid0)
    obj_tol = wrows[-1]['objective_tolerance']
    return {
        'window': [lo, hi], 'label': label, 'n_cycles': len(wrows),
        'cycles_first_last': [cycles[0], cycles[-1]],
        'raw_span': float(gc.max() - gc.min()),
        'ma_detrend_window': MA_WINDOW,
        'ma_residual_peak_to_peak': float(ma_resid0.max() - ma_resid0.min()),
        'lin_residual_peak_to_peak': float(lin_resid0.max() - lin_resid0.min()),
        'objective_tolerance_terminal': float(obj_tol),
        'ma_residual_ptp_over_tolerance': float((ma_resid0.max() - ma_resid0.min()) / obj_tol),
        'acf_top_periods_ma_detrend': acf_ma,
        'dft_top_periods_ma_detrend': dft_ma,
        'dft_dominant_amplitude_over_tolerance_ma_detrend': (
            float(dft_ma[0]['amplitude'] / obj_tol) if dft_ma else None),
        'acf_top_periods_linear_detrend': acf_lin,
        'dft_top_periods_linear_detrend': dft_lin,
        'agreement_ma_vs_linear_dominant_period_pct_diff': (
            float(abs(dft_ma[0]['period_cycles'] - dft_lin[0]['period_cycles']) / dft_ma[0]['period_cycles'] * 100)
            if dft_ma and dft_lin else None),
        '_ma_residual0': ma_resid0, '_cycles': cycles,
    }


def survey_series(rows, lo, hi, cost_dominant_period, extra_series, dominant_tol_frac=0.15):
    wrows = window_rows(rows, lo, hi)
    cycles = [r['cycle'] for r in wrows]
    gc = series(wrows, 'gross_operational_cost')
    cost_resid = gc - moving_average_trend(gc)
    cost_resid0 = cost_resid - cost_resid.mean()

    def one(name, values):
        values = np.asarray(values, dtype=float)
        if np.allclose(values, values[0]):
            return {'series': name, 'constant': True}
        resid = values - moving_average_trend(values)
        resid0 = resid - resid.mean()
        dft = dft_top_periods(resid0, top=1)
        dom_period = dft[0]['period_cycles'] if dft else None
        shares = (dom_period is not None and cost_dominant_period is not None
                  and abs(dom_period - cost_dominant_period) / cost_dominant_period <= dominant_tol_frac)
        xcorr = cross_correlation_lag(cost_resid0, resid0)
        return {
            'series': name, 'constant': False,
            'dft_dominant_period_cycles': dom_period,
            'dft_dominant_amplitude': dft[0]['amplitude'] if dft else None,
            'shares_cost_dominant_period': shares,
            'phase_vs_cost': xcorr,
        }

    fields_by_channel = ('boyd_{c}_r', 'boyd_{c}_s', 'boyd_{c}_norm_x', 'boyd_{c}_norm_z', 'boyd_{c}_norm_y',
                          'boyd_{c}_primal_ratio', 'boyd_{c}_dual_ratio')
    legacy_by_channel = ('primal_{c}', 'dual_{c}', 'primal_{c}_mean', 'dual_{c}_mean')

    results = []
    for c in CH:
        for tmpl in fields_by_channel + legacy_by_channel:
            field = tmpl.format(c=c)
            results.append(one(field, series(wrows, field)))
    results.append(one('recourse', series(wrows, 'recourse')))
    results.append(one('recourse_change', series(wrows, 'recourse_change')))
    for name, values in extra_series.items():
        results.append(one(name, values))
    return {'window': [lo, hi], 'cost_dominant_period_cycles': cost_dominant_period,
            'dominant_period_match_tolerance_frac': dominant_tol_frac, 'series': results,
            'sharing_dominant_period': [r['series'] for r in results if r.get('shares_cost_dominant_period')]}


def freeze_coupling(rows):
    windows = ((3, 30), (31, 60), (61, 90), (91, 120), (121, 150))
    out = []
    for lo, hi in windows:
        wrows = window_rows(rows, lo, hi)
        gc = series(wrows, 'gross_operational_cost')
        resid = gc - moving_average_trend(gc)
        resid0 = resid - resid.mean()
        dft = dft_top_periods(resid0, top=1)
        out.append({'window': [lo, hi], 'n_cycles': len(wrows),
                     'residual_std': float(resid0.std()),
                     'residual_ptp': float(resid0.max() - resid0.min()),
                     'dft_dominant_period_cycles': dft[0]['period_cycles'] if dft else None,
                     'dft_dominant_amplitude': dft[0]['amplitude'] if dft else None})
    return out


def failure_coupling(rows, lo, hi, period, failure_cycles):
    wrows = window_rows(rows, lo, hi)
    cycles = np.array([r['cycle'] for r in wrows])
    steps = np.array([r['recourse_change'] for r in wrows])  # signed, == gc_k - gc_{k-1}
    f_in_window = [c for c in failure_cycles if lo <= c <= hi]
    if not f_in_window or period is None:
        return {'window': [lo, hi], 'period_used': period, 'failure_cycles_in_window': f_in_window,
                'note': 'insufficient data for phase-lock test'}
    rng = np.random.default_rng(RNG_SEED)

    def resultant(sel_cycles):
        phase = 2 * np.pi * (np.asarray(sel_cycles) % period) / period
        return float(np.abs(np.mean(np.exp(1j * phase))))

    R_obs = resultant(f_in_window)
    n_f = len(f_in_window)
    draws = np.array([resultant(rng.choice(cycles, size=n_f, replace=False)) for _ in range(N_PERMUTATIONS)])
    p_value = float(np.mean(draws >= R_obs))

    fail_mask = np.isin(cycles, f_in_window)
    steps_fail = steps[fail_mask]
    steps_other = steps[~fail_mask]
    u_stat, u_p = mannwhitneyu(steps_fail, steps_other, alternative='two-sided')
    return {
        'window': [lo, hi], 'period_used': period, 'failure_cycles_in_window': f_in_window,
        'n_failure_cycles_in_window': n_f,
        'resultant_length_R_observed': R_obs,
        'permutations': N_PERMUTATIONS, 'empirical_p_value_R': p_value,
        'positive_step_frac_failure_cycles': float(np.mean(steps_fail > 0)),
        'positive_step_frac_other_cycles': float(np.mean(steps_other > 0)),
        'mean_step_failure_cycles': float(steps_fail.mean()), 'mean_step_other_cycles': float(steps_other.mean()),
        'median_step_failure_cycles': float(np.median(steps_fail)), 'median_step_other_cycles': float(np.median(steps_other)),
        'mannwhitneyu_statistic': float(u_stat), 'mannwhitneyu_p_value': float(u_p),
    }


def channel_coupling(rows, lo, hi):
    wrows = window_rows(rows, lo, hi)
    steps = series(wrows, 'recourse_change')
    candidates = {}
    for c in CH:
        for fld in ('boyd_{c}_r', 'boyd_{c}_s', 'boyd_{c}_norm_z', 'boyd_{c}_dual_ratio', 'boyd_{c}_primal_ratio'):
            field = fld.format(c=c)
            candidates[field] = series(wrows, field)
    best = None
    ranked = []
    for name, values in candidates.items():
        for lag in CHANNEL_LAGS:
            if lag == 0:
                xs, ys = steps, values
            else:
                xs, ys = steps[lag:], values[:-lag]
            r = pearson(xs, ys)
            if r is None:
                continue
            ranked.append({'series': name, 'lag': lag, 'pearson_r': r})
            if best is None or abs(r) > abs(best['pearson_r']):
                best = {'series': name, 'lag': lag, 'pearson_r': r}
    ranked.sort(key=lambda d: abs(d['pearson_r']), reverse=True)
    return {'window': [lo, hi], 'formula': 'r(L) = pearson(step_k, x_{k-L})', 'best': best, 'top10': ranked[:10]}


def bound_activeset_coupling(rows, lo, hi, cost_dominant_period, esso_stats, dominant_tol_frac=0.15):
    wrows = window_rows(rows, lo, hi)

    def one(name, values):
        values = np.asarray(values, dtype=float)
        if np.allclose(values, values[0]):
            return {'series': name, 'constant': True}
        resid0 = values - moving_average_trend(values)
        resid0 = resid0 - resid0.mean()
        dft = dft_top_periods(resid0, top=1)
        dom_period = dft[0]['period_cycles'] if dft else None
        shares = (dom_period is not None and cost_dominant_period is not None
                  and abs(dom_period - cost_dominant_period) / cost_dominant_period <= dominant_tol_frac)
        return {'series': name, 'constant': False, 'dft_dominant_period_cycles': dom_period,
                'dft_dominant_amplitude': dft[0]['amplitude'] if dft else None,
                'shares_cost_dominant_period': shares}

    results = []
    for c in CH:
        for fld in ('slack_consensus_{c}', 'slack_stationarity_{c}'):
            field = fld.format(c=c)
            results.append(one(field, series(wrows, field)))
    results.append(one('slack_objective', series(wrows, 'slack_objective')))
    results.append(one('esso_indeterminate_class_count', esso_stats['indeterminate_count']))
    return {'window': [lo, hi], 'series': results,
            'note': 'per-cycle voltage/ESS active-set breakdown beyond slack_* aggregates and the esso_capture '
                    'class/class_bar flags is not serialized; a bound-crossing event log keyed by (node, year, '
                    'day, period) would be needed to localize active-set changes below this aggregate level.'}


def block_localisation(stdout_path, lo, hi):
    blocks = parse_recourse_jump_blocks(stdout_path)
    cycles_with_data = sorted(blocks.keys())
    in_window = [c for c in cycles_with_data if lo <= c <= hi]
    return {
        'window': [lo, hi],
        'recourse_jump_diagnostic_cycles_with_data': cycles_with_data,
        'max_cycle_with_data': max(cycles_with_data) if cycles_with_data else None,
        'cycles_in_window_with_data': in_window,
        'n_cycles_in_window_with_data': len(in_window),
        'n_cycles_in_window_total': hi - lo + 1,
        'data_in_window': {str(c): blocks[c] for c in in_window},
        'verdict': ('NOT DETERMINABLE from committed artifacts: the [RECOURSE JUMP] block decomposition in '
                    'stdout_baseline.log is only emitted through cycle 62 (gated by a jump-detection threshold '
                    'in production, not printed unconditionally every cycle), so it does not cover the '
                    f'oscillation window {lo}-{hi}. To localize the oscillation to specific (agent, node, year, '
                    'day) blocks in a future run, the per-cycle block-level recourse decomposition would need to '
                    'be captured unconditionally (or persisted to a structured per-cycle artifact) rather than '
                    'gated behind the jump-detection print.'),
    }


def contrast_run(rows, windows):
    out = {}
    for lo, hi in windows:
        wrows = window_rows(rows, lo, hi)
        if len(wrows) < 10:
            out[f'{lo}_{hi}'] = {'note': f'only {len(wrows)} cycles available in this run, skipped'}
            continue
        gc = series(wrows, 'gross_operational_cost')
        resid0 = gc - moving_average_trend(gc)
        resid0 = resid0 - resid0.mean()
        acf = acf_top_periods(resid0)
        dft = dft_top_periods(resid0)
        obj_tol = wrows[-1]['objective_tolerance']
        out[f'{lo}_{hi}'] = {
            'n_cycles': len(wrows), 'raw_span': float(gc.max() - gc.min()),
            'residual_ptp': float(resid0.max() - resid0.min()),
            'objective_tolerance_terminal': float(obj_tol),
            'residual_ptp_over_tolerance': float((resid0.max() - resid0.min()) / obj_tol),
            'acf_top_periods': acf, 'dft_top_periods': dft,
        }
    out['gamma_policy'] = rows[0].get('gamma_policy')
    out['freeze_after_cycle'] = rows[0].get('freeze_after_cycle')
    out['n_cycles_run'] = len(rows)
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def strip_private(d):
    if isinstance(d, dict):
        return {k: strip_private(v) for k, v in d.items() if not k.startswith('_')}
    if isinstance(d, list):
        return [strip_private(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def main():
    if os.path.exists(OUT_DIR):
        raise RuntimeError(f'refusing to write into existing directory {OUT_DIR}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s33e2 Z4 oscillation attribution').install()
    try:
        rows = load_trajectory(os.path.join(RUN, 'g_baseline.json'))
        s32_rows = load_trajectory(S32_G)
        s31c_rows = load_trajectory(S31C_G)
        stdout_path = os.path.join(RUN, 'stdout_baseline.log')

        cost_31_150 = characterise_cost(rows, 31, 150, 'post-freeze (31-150)')
        cost_61_150 = characterise_cost(rows, 61, 150, 'late window (61-150)')
        cost_101_150 = characterise_cost(rows, 101, 150, 'Planner-quoted window (101-150)')

        dom_period_61_150 = cost_61_150['dft_top_periods_ma_detrend'][0]['period_cycles'] if cost_61_150['dft_top_periods_ma_detrend'] else None
        dom_period_31_150 = cost_31_150['dft_top_periods_ma_detrend'][0]['period_cycles'] if cost_31_150['dft_top_periods_ma_detrend'] else None

        envelope_31_150 = extrema_envelope(cost_31_150['_cycles'], cost_31_150['_ma_residual0'])
        envelope_61_150 = extrema_envelope(cost_61_150['_cycles'], cost_61_150['_ma_residual0'])

        esso_61_150 = esso_percycle(RUN, list(range(61, 151)))
        extra = {
            'esso_throughput_total': esso_61_150['throughput_total'],
            'esso_net_abs_pnet_total': esso_61_150['net_abs_pnet_total'],
            'esso_net_abs_pnet_node5': esso_61_150['net_abs_pnet_by_node']['5'],
            'esso_net_abs_pnet_node7': esso_61_150['net_abs_pnet_by_node']['7'],
            'esso_net_abs_pnet_node9': esso_61_150['net_abs_pnet_by_node']['9'],
        }
        survey_61_150 = survey_series(rows, 61, 150, dom_period_61_150, extra)

        freeze = freeze_coupling(rows)
        fail_31_150 = failure_coupling(rows, 31, 150, dom_period_31_150, FAILURE_CYCLES)
        fail_61_150 = failure_coupling(rows, 61, 150, dom_period_61_150, FAILURE_CYCLES)
        chan_coupling = channel_coupling(rows, 61, 150)
        bound_coupling = bound_activeset_coupling(rows, 61, 150, dom_period_61_150, esso_61_150)
        block_loc = block_localisation(stdout_path, 61, 150)
        block_loc_101_150 = block_localisation(stdout_path, 101, 150)
        s32_contrast = contrast_run(s32_rows, ((31, 150), (61, 150)))
        s31c_contrast = contrast_run(s31c_rows, ((31, 90),))  # s31c capped at 90 cycles

        out = {
            'stage': 'P5.15 Addendum 15 item 4 - Z4 oscillation attribution (s33e2)',
            'question': ('gross_operational_cost oscillates over cycles 101-150 instead of settling; attribute '
                         'the oscillation: quantity, period, amplitude, and coupling to channel / rho-gamma '
                         'freeze / network failures / other.'),
            'section2_cost_characterisation': {
                '31_150': strip_private(cost_31_150), '61_150': strip_private(cost_61_150),
                '101_150': strip_private(cost_101_150),
            },
            'section2b_extrema_envelope_decay': {'31_150': envelope_31_150, '61_150': envelope_61_150},
            'section3_same_period_elsewhere_window_61_150': survey_61_150,
            'section4a_failure_coupling': {'window_31_150': fail_31_150, 'window_61_150': fail_61_150,
                                            'failure_cycles_all': list(FAILURE_CYCLES)},
            'section4b_freeze_coupling': freeze,
            'section4c_channel_coupling': chan_coupling,
            'section4d_block_localisation': {'window_61_150': block_loc, 'window_101_150': block_loc_101_150},
            'section4e_bound_activeset_coupling': bound_coupling,
            'section5_s32_contrast': s32_contrast,
            'section5b_s31c_contrast_bonus': s31c_contrast,
        }
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)

    os.makedirs(OUT_DIR)
    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=1)

    manifest = {}
    for path in (OUT_JSON, os.path.abspath(__file__)):
        manifest[os.path.relpath(path, REPO)] = {'bytes': os.path.getsize(path), 'sha256': sha256_of(path)}
    with open(OUT_MANIFEST, 'w') as f:
        json.dump(manifest, f, indent=1)

    print('cost 31-150 dominant period (MA detrend, DFT):', cost_31_150['dft_top_periods_ma_detrend'][:1])
    print('cost 61-150 dominant period (MA detrend, DFT):', cost_61_150['dft_top_periods_ma_detrend'][:1])
    print('cost 101-150 dominant period (MA detrend, DFT):', cost_101_150['dft_top_periods_ma_detrend'][:1])
    print('envelope 31-150 peak spacings:', envelope_31_150['peak_spacings'], 'decay fit:', envelope_31_150['envelope_decay_fit'])
    print('envelope 31-150 period from zero crossings:', envelope_31_150['period_from_zero_crossings_cycles'])
    print('ma_residual_ptp_over_tolerance 101-150:', cost_101_150['ma_residual_ptp_over_tolerance'])
    print('series sharing dominant period (61-150):', survey_61_150['sharing_dominant_period'])
    print('failure coupling 61-150 R / p-value:', fail_61_150.get('resultant_length_R_observed'), fail_61_150.get('empirical_p_value_R'))
    print('failure coupling 61-150 mannwhitneyu p:', fail_61_150.get('mannwhitneyu_p_value'))
    print('best channel predictor (61-150):', chan_coupling['best'])
    print('block localisation 101-150 cycles with data:', block_loc_101_150['n_cycles_in_window_with_data'])
    print('s32 contrast 61-150 dominant period:', s32_contrast['61_150']['dft_top_periods'][:1])
    print('s31c contrast 31-90 dominant period:', s31c_contrast.get('31_90', {}).get('dft_top_periods', [])[:1])
    print('guard', out['solve_profile_guard'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
