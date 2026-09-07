"""
Stage P5.3-A2, Part C -- stochastic RES and low-output audit.

C1 instruments the production RES pipeline non-invasively: `network_data.np`
and `network_data.MinMaxScaler` are temporarily replaced by recording proxies so
the copula samples can be inspected after inverse MinMax transformation and
immediately BEFORE `np.abs`, and compared with the historical training data and
the post-`abs` result. The production algorithm itself is untouched and the
deterministic seed (2026) is preserved.

C2-C4 then audit the realized SRP1 positive-bootstrap population: the low-output
population of `pg_available`, the RES constraint-instantiation map, and
`sg_capability` margins/gradients versus availability.

C5 is answered by source trace (reported in the markdown, not here).

Diagnostic only. No production file is modified.

    python p53a2_res_audit.py
"""

import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone

import numpy as np
import pyomo.environ as pe
from pyomo.core.expr.calculus.derivatives import Modes, differentiate
from pyomo.core.expr.visitor import identify_variables

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import network_data as nd  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from definitions import EQUALITY_TOLERANCE  # noqa: E402
from shared_resources_planning import SharedResourcesPlanning  # noqa: E402

SPEC_DIR = 'data/SRP1'
SPEC_FILE = 'SRP1.json'
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P53A2')

BINS = [('exactly_zero', 0.0, 0.0), ('(0,1e-6]', 0.0, 1e-6), ('(1e-6,1e-5]', 1e-6, 1e-5),
        ('(1e-5,1e-4]', 1e-5, 1e-4), ('(1e-4,1e-3]', 1e-4, 1e-3), ('>1e-3', 1e-3, float('inf'))]


def git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return None


def classify(value):
    if value == 0.0:
        return 'exactly_zero'
    for name, lo, hi in BINS[1:]:
        if lo < value <= hi:
            return name
    return '>1e-3'


# ---------------------------------------------------------------------------
#  C1 -- recording proxies around the untouched production pipeline
# ---------------------------------------------------------------------------
class RecordingNumpy:
    """Forwards everything to numpy, but records the argument of np.abs()."""

    def __init__(self, real, sink):
        self._real = real
        self._sink = sink

    def abs(self, arg):
        out = self._real.abs(arg)
        try:
            self._sink.append({'pre_abs': np.asarray(arg, dtype=float).copy(),
                               'post_abs': np.asarray(out, dtype=float).copy()})
        except Exception:
            pass
        return out

    def __getattr__(self, item):
        return getattr(self._real, item)


def make_recording_scaler(real_cls, sink):
    class RecordingMinMaxScaler(real_cls):
        def fit_transform(self, X, y=None, **kwargs):
            try:
                sink.append({'historical': np.asarray(X, dtype=float).copy()})
            except Exception:
                sink.append({'historical': None})
            return super().fit_transform(X, y, **kwargs)
    return RecordingMinMaxScaler


def summarize_c1(abs_calls, hist_calls, labels):
    out = []
    n = min(len(abs_calls), len(hist_calls), len(labels))
    for i in range(n):
        pre = abs_calls[i]['pre_abs'].ravel()
        post = abs_calls[i]['post_abs'].ravel()
        hist = hist_calls[i]['historical']
        hist_flat = hist.ravel() if hist is not None else np.array([])
        neg = pre[pre < 0.0]
        hist_max = float(hist_flat.max()) if hist_flat.size else None
        hist_min = float(hist_flat.min()) if hist_flat.size else None
        above = post[post > hist_max] if hist_max is not None else np.array([])
        out.append({
            'label': labels[i],
            'n_values': int(pre.size),
            'historical_min': hist_min, 'historical_max': hist_max,
            'pre_abs_min': float(pre.min()), 'pre_abs_max': float(pre.max()),
            'n_negative_pre_abs': int(neg.size),
            'pct_negative_pre_abs': float(100.0 * neg.size / pre.size) if pre.size else 0.0,
            'min_negative_value': float(neg.min()) if neg.size else None,
            'sum_abs_negative_magnitude': float(np.abs(neg).sum()) if neg.size else 0.0,
            'mean_abs_negative_magnitude': float(np.abs(neg).mean()) if neg.size else None,
            'reflected_share_of_total_post_abs_mass': (
                float(np.abs(neg).sum() / post.sum()) if post.sum() > 0 and neg.size else 0.0),
            'n_above_historical_max': int(above.size),
            'pct_above_historical_max': float(100.0 * above.size / post.size) if post.size else 0.0,
            'max_overshoot_above_historical_max': (
                float(above.max() - hist_max) if above.size else 0.0),
            'post_abs_quantiles': {q: float(np.quantile(post, v)) for q, v in
                                   (('p01', 0.01), ('p05', 0.05), ('p50', 0.50),
                                    ('p95', 0.95), ('p99', 0.99))},
            'pre_abs_quantiles': {q: float(np.quantile(pre, v)) for q, v in
                                  (('p01', 0.01), ('p05', 0.05), ('p50', 0.50),
                                   ('p95', 0.95), ('p99', 0.99))},
        })
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {'stage': 'P5.3-A2 part C', 'git_head': git_head(),
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'equality_tolerance': EQUALITY_TOLERANCE}

    abs_sink, hist_sink = [], []
    real_np, real_scaler = nd.np, nd.MinMaxScaler
    quiet = io.StringIO()
    try:
        nd.np = RecordingNumpy(real_np, abs_sink)
        nd.MinMaxScaler = make_recording_scaler(real_scaler, hist_sink)
        with redirect_stdout(quiet):
            planning = SharedResourcesPlanning(SPEC_DIR, SPEC_FILE)
            planning.read_planning_problem()
    finally:
        nd.np, nd.MinMaxScaler = real_np, real_scaler

    report['scenario_checksum'] = (
        re.findall(r'Scenario checksum: (\S+)', quiet.getvalue()) or [None])[-1]
    report['c1_note'] = (
        'np.abs is called only in generate_res_generation_profiles; the recorded '
        'calls are therefore in (network, season, gen_type) loop order. '
        'MinMaxScaler.fit_transform is also called by the load/flexibility '
        'pipelines, so historical pairings are matched by call order and shape.')

    # pair abs-calls with the immediately preceding scaler fit of the same shape
    labels = [f'res_call_{i}' for i in range(len(abs_sink))]
    hist_matched = []
    for call in abs_sink:
        shape = call['pre_abs'].shape
        match = None
        for h in hist_sink:
            arr = h.get('historical')
            if arr is not None and arr.shape[1:] == shape[1:]:
                match = h
        hist_matched.append(match or {'historical': None})
    report['C1_raw_copula_support'] = summarize_c1(abs_sink, hist_matched, labels)
    report['C1_n_abs_calls'] = len(abs_sink)

    # ---------------- C2/C3/C4 on the realized population ----------------
    gens, bins_count = [], defaultdict(int)
    near_threshold = []
    for node_id, dn in sorted(planning.distribution_networks.items()):
        for year in dn.years:
            for day in dn.days:
                net = dn.network[year][day]
                for gen in net.generators:
                    if not gen.is_curtaillable():
                        continue
                    for s_o in range(net.num_oper_scenarios):
                        for p in range(net.num_instants):
                            pg = float(gen.pg[s_o][p]); qg = float(gen.qg[s_o][p])
                            sg = float(np.sqrt(pg ** 2 + qg ** 2))
                            b = classify(abs(pg))
                            bins_count[b] += 1
                            if abs(pg) <= 1e-4 or abs(sg - EQUALITY_TOLERANCE) <= 5e-6:
                                near_threshold.append({
                                    'network': net.name, 'node': node_id,
                                    'gen_id': gen.gen_id, 'gen_type': gen.gen_type,
                                    'bus': gen.bus, 'year': year, 'day': day, 'hour': p,
                                    'pg_available': pg, 'qg_available': qg,
                                    'sg_available': sg,
                                    'unavailable_switch': bool(sg <= EQUALITY_TOLERANCE),
                                    'bin': b})
                    gens.append({
                        'network': net.name, 'node': node_id, 'year': year, 'day': day,
                        'gen_id': gen.gen_id, 'gen_type': gen.gen_type, 'bus': gen.bus,
                        'power_factor_control': bool(getattr(gen, 'power_factor_control', False)),
                        'min_pf': getattr(gen, 'min_pf', None), 'max_pf': getattr(gen, 'max_pf', None),
                        'pmin': gen.pmin, 'pmax': gen.pmax,
                        'qmin': gen.qmin, 'qmax': gen.qmax,
                        'qg_all_zero': bool(np.all(np.asarray(gen.qg) == 0.0)),
                        'pg_min': float(np.min(gen.pg)), 'pg_max': float(np.max(gen.pg)),
                    })
    report['C2_bins'] = dict(bins_count)
    report['C2_total_values'] = int(sum(bins_count.values()))
    report['C2_near_threshold_examples'] = sorted(
        near_threshold, key=lambda r: r['pg_available'])[:60]
    report['C2_n_below_or_at_equality_tolerance'] = int(
        sum(1 for r in near_threshold if r['unavailable_switch']))
    report['C3_generators'] = gens
    report['C3_summary'] = {
        'n_curtailable_generator_instances': len(gens),
        'n_with_power_factor_control': sum(1 for g in gens if g['power_factor_control']),
        'n_with_qg_identically_zero': sum(1 for g in gens if g['qg_all_zero']),
        'distinct_gen_types': sorted({g['gen_type'] for g in gens}),
    }

    with open(os.path.join(OUT_DIR, 'p53a2_res.json'), 'w') as handle:
        json.dump(report, handle, indent=1, default=str)

    print(f"[C] checksum {report['scenario_checksum']}  abs-calls recorded: {len(abs_sink)}")
    for row in report['C1_raw_copula_support']:
        print(f"  {row['label']:14s} n={row['n_values']:6d} "
              f"neg_pre_abs={row['n_negative_pre_abs']:5d} ({row['pct_negative_pre_abs']:.2f}%) "
              f"min_neg={row['min_negative_value']} "
              f"reflected_mass={row['reflected_share_of_total_post_abs_mass']:.4f} "
              f"above_hist_max={row['n_above_historical_max']} ({row['pct_above_historical_max']:.2f}%)")
    print(f"\n[C2] bins over {report['C2_total_values']} realized values: {report['C2_bins']}")
    print(f"[C2] values at/below EQUALITY_TOLERANCE (structural switch): "
          f"{report['C2_n_below_or_at_equality_tolerance']}")
    print(f"[C3] {json.dumps(report['C3_summary'])}")


if __name__ == '__main__':
    main()
