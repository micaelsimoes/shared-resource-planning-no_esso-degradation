"""P5.15 Step 3.2 E2 - Z2: is the shared-ESS drift price arbitrage? (zero solves, SolveProfileGuard armed).

Hypothesis under test (Advisor): the drift is a constant TSO temporal-arbitrage gradient - shared-ESS charging is
bought and discharging credited at the TSO's hourly energy cost, and shared-ESS usage carries no economic cost
(Step 3.1 row 8), so the iterate walks toward more arbitrage until round-trip losses stop it.

Sources (read-only, already committed):
  * per-cycle ESSO captures data/SRP1/Results/P515S33_E2_run/esso_capture/baseline/node{5,7,9}_cycle{k:03d}.jsonl,
    one row per (year index y, day index d, period p) with pnet = pch - pdch (charging positive);
  * hourly prices pi_t from the run's own interface_settlement_detail_s31c.json,
    interface_reporting_detail[node][year][day]['periods'][p]['price_per_mwh'];
  * year/day index order from data/SRP1/SRP1.json (asserted: 3 years x 4 days x 24 periods = 288 entries).

Per node, over a late window (cycles A..B, default 100..150):
  drift(entry)      = pnet_B - pnet_A
  price_dev(entry)  = pi - mean(pi over the same (year, day) block)      (arbitrage is within-day)
  cos_price_dev     = cos(drift, price_dev)            (< 0 = charge when cheap, discharge when dear)
  pearson_r         = correlation of the same two vectors
  implied_margin    = - sum(drift * price_dev)         (>0 = the drift direction earns money at these prices)
A negative cosine of material size supports arbitrage (H1); |cos| ~ 0 supports a numerically incidental direction (H2).
Write-once output: data/SRP1/Results/P515S33_E2_run/s33e2_z2_price_alignment.json
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

RUN = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S33_E2_run')
CAP = os.path.join(RUN, 'esso_capture', 'baseline')
OUT = os.path.join(RUN, 's33e2_z2_price_alignment.json')
NODES = (5, 7, 9)
WINDOWS = ((100, 150), (31, 150), (140, 150))


def _pnet(node, cycle):
    rows = []
    with open(os.path.join(CAP, f'node{node}_cycle{cycle:03d}.jsonl')) as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                rows.append(((r['y'], r['d'], r['p']), float(r['pnet'])))
    rows.sort(key=lambda kv: kv[0])
    return [k for k, _ in rows], [v for _, v in rows]


def _cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return (sum(x * y for x, y in zip(a, b)) / (na * nb)) if na and nb else None


def _pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    return _cos([x - ma for x in a], [y - mb for y in b])


def main():
    if os.path.exists(OUT):
        raise RuntimeError(f'refusing to overwrite {OUT}')
    guard = SolveProfileGuard(permitted=(), label='P5.15 s33e2 Z2').install()
    try:
        case = json.load(open(os.path.join(REPO, 'data', 'SRP1', 'SRP1.json')))
        years = [str(y) for y in (case.get('Years') or case.get('years'))]
        days = list((case.get('Days') or case.get('days')))
        detail = json.load(open(os.path.join(RUN, 'interface_settlement_detail_s31c.json')))['interface_reporting_detail']
        out = {'stage': 'P5.15 s33e2 Z2 - shared-ESS drift vs hourly price', 'years_order': years, 'days_order': days,
               'sign_convention': 'pnet = pch - pdch (charging positive); cos < 0 means charge when price is below the '
                                  'day mean and discharge when above', 'nodes': {}}
        for node in NODES:
            keys, _ = _pnet(node, 150)
            prices = []
            for (y, d, p) in keys:
                year, day = years[y], days[d]
                prices.append(float(detail[str(node)][year][day]['periods'][str(p)]['price_per_mwh']))
            block_sum, block_n = {}, {}
            for (y, d, _), pi in zip(keys, prices):
                block_sum[(y, d)] = block_sum.get((y, d), 0.0) + pi
                block_n[(y, d)] = block_n.get((y, d), 0) + 1
            dev = [pi - block_sum[(y, d)] / block_n[(y, d)] for (y, d, _), pi in zip(keys, prices)]
            node_out = {'n_entries': len(keys), 'price_mean': sum(prices) / len(prices),
                        'price_dev_abs_mean': sum(abs(x) for x in dev) / len(dev), 'windows': {}}
            for a, b in WINDOWS:
                ka, pa = _pnet(node, a)
                kb, pb = _pnet(node, b)
                if ka != keys or kb != keys:
                    raise RuntimeError('entry ordering differs between cycles')
                drift = [x - y for x, y in zip(pb, pa)]
                node_out['windows'][f'{a}_{b}'] = {
                    'drift_norm': math.sqrt(sum(x * x for x in drift)),
                    'cos_price_dev': _cos(drift, dev),
                    'pearson_r': _pearson(drift, dev),
                    'implied_margin_currency_per_cycle': -sum(x * y for x, y in zip(drift, dev)) / (b - a),
                    'charge_entries_mean_price_dev': (sum(d_ for d_, x in zip(dev, drift) if x > 0)
                                                      / max(1, len([x for x in drift if x > 0]))),
                    'discharge_entries_mean_price_dev': (sum(d_ for d_, x in zip(dev, drift) if x < 0)
                                                        / max(1, len([x for x in drift if x < 0]))),
                    'n_charging': len([x for x in drift if x > 0]), 'n_discharging': len([x for x in drift if x < 0]),
                }
            out['nodes'][str(node)] = node_out
    finally:
        guard.uninstall()
    failures = guard.verify(expected_solves=0)
    out['solve_profile_guard'] = {'counts': dict(guard.counts), 'verify_failures': failures}
    if failures:
        raise RuntimeError(failures)
    json.dump(out, open(OUT, 'w'), indent=1)
    for node, n in out['nodes'].items():
        print(f"node {node}: mean price {n['price_mean']:.2f}/MWh, mean |deviation from day mean| {n['price_dev_abs_mean']:.2f}")
        for w, v in n['windows'].items():
            print(f"   cycles {w}: cos(drift, price_dev) {v['cos_price_dev']:+.3f} | pearson {v['pearson_r']:+.3f} | "
                  f"drift norm {v['drift_norm']:.4g} | margin/cycle {v['implied_margin_currency_per_cycle']:+.4g} | "
                  f"charging {v['n_charging']} entries at mean dev {v['charge_entries_mean_price_dev']:+.2f}, "
                  f"discharging {v['n_discharging']} at {v['discharge_entries_mean_price_dev']:+.2f}")
    print('guard', out['solve_profile_guard'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
