"""
S1 — criterion validity for `stationarity_pf`. Diagnose only; zero solves, ENFORCED.

Frozen spec: data/SRP1/Results/P514S1/frozen_s1_criterion_spec_v1_6c1d0a81.json

Mechanism under test: the pf dual residual is rho * |z^k - z^{k-1}| / base
(shared_resources_planning.py:4938) and is compared against an ABSOLUTE tolerance,
so slack = tolerance/residual must fall as rho rises. Predeclared:
  P1  g = dual_pf_mean / rho_pf (the increment with rho divided out) is invariant
      if the criterion is rho-dominated
  P2  median slack(1000)/slack(300) ~ 0.300 if increments are unchanged

Nothing is changed by this harness. It reads preserved artifacts only.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

P510 = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P510')
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'SRP1', 'Results', 'P514S1')
SPEC = 'data/SRP1/Results/P514S1/frozen_s1_criterion_spec_v1_6c1d0a81.json'
SETTINGS = {300: 'p510_b_fixedrho_pf300.json', 500: 'p510_b_fixedrho_pf500.json',
            1000: 'p510_b_fixedrho_pf1000.json'}
COMMON_RANGE = 6   # shortest per-repeat trajectory (rho_pf = 300)


def load_cycles(filename):
    """Per-cycle records.

    Each p510_b file holds THREE repeat rows of one rho setting, and the per-cycle
    data lives in each row's `cycle_detail`: 3x6 + 3x9 + 3x16 = 93 cycles in total,
    matching the count preserved in p510_e_criteria.json. Reading the rows themselves
    would yield the nine terminal summaries instead, which is the defect the
    predeclared cross-check caught on the first run of this harness.
    """
    with open(os.path.join(P510, filename)) as handle:
        payload = json.load(handle)
    cycles = []
    for repeat_index, row in enumerate(payload['rows']):
        for cycle in row.get('cycle_detail') or []:
            record = dict(cycle)
            record['rho_pf'] = row.get('rho_pf')
            record['repeat_index'] = repeat_index
            record['candidate'] = row.get('candidate')
            cycles.append(record)
    return cycles


def loglog_exponent(points):
    """Least-squares p in y ~ x^(-p) from (x, y) points."""
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return -num / den if den else None


def trend(series):
    """Least-squares slope of log(slack) against cycle index."""
    points = [(i + 1, v) for i, v in enumerate(series) if v and v > 0]
    if len(points) < 3:
        return None
    xs = [p[0] for p in points]
    ys = [math.log(p[1]) for p in points]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    return (sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den) if den else None


def main():
    guard = SolveProfileGuard(permitted=(), label='S1 (zero solves)').install()
    try:
        report = analyse()
    finally:
        guard.uninstall()
    report['solve_profile'] = {'declared': 0, 'observed': dict(guard.counts),
                               'enforced_by': 'SolveProfileGuard, blocking mode'}
    report['solve_profile']['failures'] = guard.verify(0)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 's1_criterion_scaling.json')
    with open(out, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != 'per_setting_detail'},
                     indent=2, default=str)[:3800])
    print(f'\n[S1] written to {out}')
    return 0


def analyse():
    report = {'stage': 'S1', 'spec': SPEC, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'nothing_changed': 'no tolerance, criterion, rho or parameter was modified',
              'per_setting': {}, 'per_setting_detail': {}}

    for rho, filename in SETTINGS.items():
        rows = load_cycles(filename)
        recs = []
        for row in rows:
            mean_ratio = row.get('dual_pf_mean_ratio')
            mean_residual = row.get('dual_pf_mean')
            if not mean_ratio or mean_residual is None:
                continue
            row_rho = row.get('rho_pf', rho)
            recs.append({
                'cycle': row.get('cycle'),
                'repeat_index': row.get('repeat_index'),
                'rho_pf': row_rho,
                'dual_pf_mean': mean_residual,
                'dual_pf_mean_ratio': mean_ratio,
                'stationarity_pf_slack': 1.0 / mean_ratio,
                'tolerance_in_force': mean_residual / mean_ratio,
                'g_increment': mean_residual / row_rho,
                'dual_pf_max': row.get('dual_pf'),
                'dual_pf_max_over_mean': (row.get('dual_pf') / mean_residual) if mean_residual else None,
            })
        slacks = [r['stationarity_pf_slack'] for r in recs]
        gs = [r['g_increment'] for r in recs]
        report['per_setting_detail'][str(rho)] = recs
        report['per_setting'][str(rho)] = {
            'cycles': len(recs),
            'median_slack': statistics.median(slacks) if slacks else None,
            'median_slack_common_range': statistics.median(
                [r['stationarity_pf_slack'] for r in recs if r['cycle'] and r['cycle'] <= COMMON_RANGE]) if recs else None,
            'terminal_slack': slacks[-1] if slacks else None,
            'max_slack': max(slacks) if slacks else None,
            'median_g_increment': statistics.median(gs) if gs else None,
            'median_g_common_range': statistics.median(
                [r['g_increment'] for r in recs if r['cycle'] and r['cycle'] <= COMMON_RANGE]) if recs else None,
            'tolerance_in_force': round(statistics.median(
                [r['tolerance_in_force'] for r in recs]), 12) if recs else None,
            'log_slack_trend_per_cycle': trend(slacks),
            'slack_ever_at_or_above_1': any(s >= 1.0 for s in slacks),
        }

    base = report['per_setting']['300']
    high = report['per_setting']['1000']
    mid = report['per_setting']['500']

    p2_full = high['median_slack'] / base['median_slack']
    p2_common = high['median_slack_common_range'] / base['median_slack_common_range']
    p1_full = {'g500_over_g300': mid['median_g_increment'] / base['median_g_increment'],
               'g1000_over_g300': high['median_g_increment'] / base['median_g_increment']}
    p1_common = {'g500_over_g300': mid['median_g_common_range'] / base['median_g_common_range'],
                 'g1000_over_g300': high['median_g_common_range'] / base['median_g_common_range']}

    exponent = loglog_exponent([(rho, report['per_setting'][str(rho)]['median_slack'])
                                for rho in SETTINGS])
    exponent_common = loglog_exponent([(rho, report['per_setting'][str(rho)]['median_slack_common_range'])
                                       for rho in SETTINGS])

    def classify(ratio):
        if 0.25 <= ratio <= 0.40:
            return 'rho_dominated — mechanism CONFIRMED'
        if 0.85 <= ratio <= 1.15:
            return 'compensating — mechanism ABSENT'
        return 'partial — quantify the exponent'

    report['P1_increment_invariance'] = {
        'definition': 'g = dual_pf_mean / rho_pf = mean |z^k - z^(k-1)| / interface_rating',
        'full_trajectory': p1_full, 'common_range_1_to_6': p1_common,
        'prediction': 'g approximately invariant if the criterion is rho-dominated',
    }
    report['P2_slack_scaling'] = {
        'predicted_if_increments_unchanged': 0.300,
        'observed_full_trajectory': p2_full,
        'observed_common_range_1_to_6': p2_common,
        'classification_full': classify(p2_full),
        'classification_common_range': classify(p2_common),
        'loglog_exponent_p_full': exponent,
        'loglog_exponent_p_common_range': exponent_common,
        'note': 'slack ~ rho^(-p); p = 1 is exact rho domination, p = 0 is full compensation',
    }

    # cross-check against the preserved P5.10-E summary
    with open(os.path.join(P510, 'p510_e_criteria.json')) as handle:
        e_report = json.load(handle)
    preserved = [c['binding_slack'] for c in e_report['per_cycle']
                 if c['binding_criterion'] == 'stationarity_pf']
    rederived = sorted(r['stationarity_pf_slack']
                       for recs in report['per_setting_detail'].values() for r in recs)
    matched = sum(1 for value in sorted(preserved)
                  if any(abs(value - other) < 1e-12 for other in rederived))
    report['cross_check_against_p510_e'] = {
        'binding_criterion_counts': e_report['binding_criterion_counts_all_cycles'],
        'preserved_stationarity_pf_cycles': len(preserved),
        'rederived_values': len(rederived),
        'matched_to_1e-12': matched,
        'status': 'PASS' if matched == len(preserved) else 'FAIL — re-derivation does not reproduce the preserved values',
    }
    for rho in SETTINGS:
        recs = report['per_setting_detail'][str(rho)]
        ratios = [r['dual_pf_max_over_mean'] for r in recs if r['dual_pf_max_over_mean']]
        report['per_setting'][str(rho)]['median_max_over_mean'] = (
            statistics.median(ratios) if ratios else None)
    report['Q3_aggregation'] = {
        'question': 'consensus uses max AND mean per family; stationarity uses only the mean',
        'dual_pf_max_is_preserved_as': 'cycle_detail["dual_pf"]',
        'median_max_over_mean_per_setting': {
            str(rho): report['per_setting'][str(rho)]['median_max_over_mean'] for rho in SETTINGS},
        'implication': 'a max-based stationarity criterion would bind this much harder than the '
                       'mean-based one actually in force',
    }
    report['scope_limit'] = (
        'All 93 cycles share one binding criterion, so the dataset holds no counterexample. '
        'It supports a characterisation of stationarity_pf, not any claim about what '
        'distinguishes binding from non-binding criteria. P1/P2 are within-criterion '
        'comparisons across settings and are unaffected.')
    return report


if __name__ == '__main__':
    sys.exit(main())
