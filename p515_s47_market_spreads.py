"""P5.15 Addendum 31 (task W22) -- paper-scale market-scenario price spreads (spec v17 step Z31).
ZERO SOLVES, no model construction, read-only.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 31; frozen spec v17
`data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json` step Z31.

WHAT THIS SCRIPT DOES
  1. Reads the market prices exactly as production does: `SharedResourcesPlanning.read_planning_problem`
     on (a) the SRP1 case `data/SRP1/SRP1.json` and (b) the committed paper-scale derived case
     `data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json` (5 representative
     years 2025/2028/2031/2034/2037 x 3 years each, 4 days, 5 market x 5 operation scenarios) -- the
     approach of `p515_s45_probability_audit.py` (paper_data_only): results/diagrams/logs redirected into
     this script's output directory, and Network.build_model, NetworkData.build_model,
     SharedEnergyStorageData.build_subproblem / .build_master_problem replaced by blockers that RAISE
     (declared and verified count: 0).
     STOP POINT (declared, count verified exactly 1 per read = 2): the read is ended at
     `SharedEnergyStorageData.read_parameters_from_file`, the first statement after production has
     (i) generated and selected the market prices (`_read_market_data_from_file`), (ii) bound them and the
     market-scenario probabilities to every TSO/DSO block, and (iii) computed and printed the scenario
     checksum. This keeps the run independent of `data/SRP1/SharedESS/SRP1_ESS_Params.json`, which is
     being edited concurrently by another task; nothing read after that point affects prices.
     Checksums: SRP1 == p56a_oracle.CANONICAL_CHECKSUM; paper == 1e8bdd3e... (the scale measurement's
     record, build_child_stdout.log:33). Both are hard requirements.
  2. Spread definition = W19's Z4 (p515_s46_zero_solve_reports.py, commit dd3afa6e): per representative
     (year, day) price vector c (24 hourly EUR/MWh), max_minus_min = max c - min c and
     top4_mean_minus_bottom4_mean = mean of the 4 highest minus mean of the 4 lowest (h = E/P = 4 h).
     Horizon averages weight each (year, day) by num_years[y] * num_days[d] (undiscounted) and,
     separately, by the model block weight num_years[y] * num_days[d] / 1.02^(y - 2025).
     Z4 cannot be imported (its module installs its own guard and imports the campaign harness, which is
     under concurrent edit), so the definition is re-stated here and PROVEN identical by re-applying it
     to Z4's committed per-day prices: every per-day value and the SRP1 figure 97.99086039739976 must be
     reproduced to 1e-9.
  3. Per market scenario s = 1..5 and for the probability-weighted mean price profile
     mean_c[y,d,p] = sum_s pm[y][s] * c[y,d,s,p], pm = the networks' own market-scenario probabilities
     (`network.prob_market_scenarios`, bound at shared_resources_planning.py from
     planning.prob_market_scenarios[year] = [1/NumMarketScenarios] * NumMarketScenarios in
     `_read_market_data_from_file`). Also the probability-weighted average of the scenario spreads.
  4. SRP1 correspondence: same-year comparison (2025, the only year both instances share) of SRP1's
     single price row to each paper scenario row and to the mean profile (max |diff|, least-squares scale
     k = <a,b>/<b,b>, Pearson r); and a growth-normalized comparison over all years (every profile divided
     by production's energy growth factor (1+g)^(y-2025), g read with production's
     `_read_market_base_profiles` and the lookup expression of `_read_market_data_from_file`).
  5. R = (mean-profile horizon-weighted 4 h spread) / 97.99086039739976 (SRP1, Z4, undiscounted) and the
     ratio of the average of the scenario spreads to the same figure -- recorded as a PREDICTION.

Output (write-once, new directory) data/SRP1/Results/P515S47/market_spreads/:
    market_spreads.json, market_spreads.md, launch.log, production_stdout_capture.log,
    manifest_sha256.json, production_read/ (production's scenario plots; hashed in the manifest only).

Launch (attached, alone, both streams captured), then the manifest:
    mkdir -p data/SRP1/Results/P515S47/market_spreads
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_market_spreads.py \\
        > data/SRP1/Results/P515S47/market_spreads/launch.log 2>&1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s47_market_spreads.py --manifest
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')   # production plots scenarios while reading data

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402  (pyomo only)

STAGE = 'P5.15 Addendum 31 W22 -- paper-scale market-scenario spreads (spec v17 Z31; zero solves)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 31',
             'data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json step Z31']
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'market_spreads')
OUT_DIR = os.path.join(REPO, OUT_REL)
RESULTS_NAME = 'market_spreads.json'
MD_NAME = 'market_spreads.md'
MANIFEST_NAME = 'manifest_sha256.json'
STDOUT_CAPTURE = 'production_stdout_capture.log'
DATA_DIR = os.path.join(REPO, 'data', 'SRP1')

SRP1_CASE_REL = 'data/SRP1/SRP1.json'
PAPER_CASE_REL = 'data/SRP1/Results/P515S44/scale_measurement/paper_build/case/SRP1__paper.json'
PAPER_CHECKSUM_RECORDED = '1e8bdd3e5233442a87fbe44b78281c8ae61b09c09700388fed20ab9684d18aef'
PAPER_CHECKSUM_SOURCE = ('data/SRP1/Results/P515S44/scale_measurement/paper_build/'
                         'build_child_stdout.log:33 ("[INFO] Scenario checksum: ...")')
MARKET_REL = 'data/SRP1/MarketData/SRP1_market_data.xlsx'

Z4_JSON_REL = 'data/SRP1/Results/P515S46/zero_solve_reports/zero_solve_reports.json'
Z4_COMMIT = 'dd3afa6e'
SRP1_Z4_SPREAD = 97.99086039739976          # Z4 all_years_day_weighted_undiscounted top4 spread
SRP1_Z4_SPREAD_R2 = 97.01788674187569       # Z4 all_years_model_block_weight_r2 top4 spread
H = 4                                       # Z4 window: E/P = 1.0 / 0.25 = 4 h
PRODUCTION_RATE = 0.02
REPRO_TOL = 1e-9
MATCH_TOL = 1e-9                            # 'identical' = max |diff| <= 1e-9 EUR/MWh (stated before the run)
DECLARED_STOPS = 2                          # one stop-point hit per read (SRP1, paper)
OBJ_CONGESTION_MANAGEMENT = 2               # definitions.py:43 (pm = [1.0] for such networks)

F_MM = 'max_minus_min'
F_TH = f'top{H}_mean_minus_bottom{H}_mean'
SPREAD_FORMULA = (f'per representative (year, day) and price vector c (24 hourly EUR/MWh): {F_MM} = max_p c_p - '
                  f'min_p c_p; {F_TH} = mean of the {H} highest c_p minus mean of the {H} lowest (W19 Z4 '
                  f'definition, p515_s46_zero_solve_reports.py @ {Z4_COMMIT}). Horizon average = sum_(y,d) w * '
                  f'spread / sum_(y,d) w with w = num_years[y] * num_days[d] (undiscounted) or w = num_years[y] * '
                  f'num_days[d] / 1.02^(y - 2025) (model block weight, production DiscountFactor 0.02).')


class StopAfterPricesBound(RuntimeError):
    """Raised at the declared stop point (SharedEnergyStorageData.read_parameters_from_file)."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, check=True, text=True).stdout


def _one_line(rel, needle):
    with open(os.path.join(REPO, rel)) as handle:
        hits = [i for i, line in enumerate(handle, 1) if needle in line]
    if len(hits) != 1:
        raise RuntimeError(f'citation {needle!r} found {len(hits)} times in {rel}')
    return f'{rel}:{hits[0]}'


def _refuse_concurrent():
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout
    me = os.getpid()
    others = [l for l in ps.splitlines() if 'p515_s47_market_spreads.py' in l and '--manifest' not in l
              and int(l.split()[0]) != me and 'python' in l]
    if others:
        raise RuntimeError(f'another copy is running: {others}')


# ======================================================================================
#  spread definition (W19 Z4)
# ======================================================================================
def day_spreads(prices):
    import numpy as np
    arr = np.asarray(prices, dtype=float)
    srt = np.sort(arr)
    return {'min': float(srt[0]), 'max': float(srt[-1]), 'mean': float(arr.mean()),
            F_MM: float(srt[-1] - srt[0]), F_TH: float(srt[-H:].mean() - srt[:H].mean())}


def weights(years, days, year, day, y0):
    wu = years[year] * days[day]
    return wu, wu / (1.0 + PRODUCTION_RATE) ** (int(year) - int(y0))


def horizon(rows, field, wkey, subset=None):
    rs = [r for r in rows if subset is None or subset(r)]
    return sum(r[field] * r[wkey] for r in rs) / sum(r[wkey] for r in rs)


def z4_reproduction():
    """Re-apply the definition to Z4's committed per-day prices; must reproduce Z4 to 1e-9."""
    z = json.load(open(os.path.join(REPO, Z4_JSON_REL)))['Z4']
    max_dev = 0.0
    rows = []
    for r in z['per_representative_day']:
        s = day_spreads(r['prices'])
        max_dev = max(max_dev, abs(s[F_MM] - r[F_MM]), abs(s[F_TH] - r[F_TH]))
        rows.append({**s, 'weight_undiscounted': r['weight_undiscounted'], 'weight_model_r2': r['weight_model_r2']})
    th_u = horizon(rows, F_TH, 'weight_undiscounted')
    th_r = horizon(rows, F_TH, 'weight_model_r2')
    return {'z4_json': Z4_JSON_REL, 'z4_json_sha256': _sha256_file(os.path.join(REPO, Z4_JSON_REL)),
            'z4_commit': Z4_COMMIT, 'max_abs_dev_per_day': max_dev,
            'horizon_top4_undiscounted_recomputed': th_u, 'horizon_top4_r2_recomputed': th_r,
            'committed_undiscounted': z['spread_summary']['all_years_day_weighted_undiscounted'][F_TH],
            'committed_r2': z['spread_summary']['all_years_model_block_weight_r2'][F_TH],
            'ok': (max_dev <= REPRO_TOL and abs(th_u - SRP1_Z4_SPREAD) <= REPRO_TOL
                   and abs(th_r - SRP1_Z4_SPREAD_R2) <= REPRO_TOL
                   and z['spread_summary']['all_years_day_weighted_undiscounted'][F_TH] == SRP1_Z4_SPREAD),
            'z4_prices': {(r['year'], r['day']): r['prices'] for r in z['per_representative_day']}}


# ======================================================================================
#  production read (data only; construction blocked; stop point after prices are bound)
# ======================================================================================
class Blocks:
    def __init__(self):
        import network as network_module
        import network_data as network_data_module
        import shared_energy_storage_data as SED
        self.targets = [(network_module.Network, 'build_model'), (network_data_module.NetworkData, 'build_model'),
                        (SED.SharedEnergyStorageData, 'build_subproblem'),
                        (SED.SharedEnergyStorageData, 'build_master_problem')]
        self.stop_target = (SED.SharedEnergyStorageData, 'read_parameters_from_file')
        self.blocked_calls = []
        self.stop_hits = 0
        self.originals = {}

    def install(self):
        def blocker(name):
            def _raise(*a, **k):
                self.blocked_calls.append(name)
                raise RuntimeError(f'W22: model construction blocked ({name})')
            return _raise

        def stop(*a, **k):
            self.stop_hits += 1
            raise StopAfterPricesBound('declared stop point reached')
        for cls, attr in self.targets:
            self.originals[(cls, attr)] = getattr(cls, attr)
            setattr(cls, attr, blocker(f'{cls.__name__}.{attr}'))
        cls, attr = self.stop_target
        self.originals[(cls, attr)] = getattr(cls, attr)
        setattr(cls, attr, stop)

    def uninstall(self):
        for (cls, attr), fn in self.originals.items():
            setattr(cls, attr, fn)


def production_read(case_rel, label, sink):
    import numpy as np
    from shared_resources_planning import SharedResourcesPlanning
    rel = os.path.relpath(os.path.join(REPO, case_rel), DATA_DIR)
    planning = SharedResourcesPlanning(DATA_DIR, rel)
    planning.name = 'SRP1'
    read_dir = os.path.join(OUT_DIR, 'production_read', label)
    planning.results_dir = os.path.join(read_dir, 'Results')
    planning.diagrams_dir = os.path.join(read_dir, 'Diagrams')
    planning.logs_dir = os.path.join(planning.results_dir, 'Logs')
    stopped = False
    sink.write(f'===== production read: {label} ({case_rel}) =====\n')
    sink.flush()
    try:
        with redirect_stdout(sink):
            planning.read_planning_problem()
    except StopAfterPricesBound:
        stopped = True
    sink.flush()
    if not stopped:
        raise RuntimeError(f'{label}: the read did not reach the declared stop point')
    meta = planning.scenario_metadata
    years = {int(y): n for y, n in planning.years.items()}
    days = dict(planning.days)
    prices, pm = {}, {}
    binding = {'tso_and_dso_price_array_is_planning_object': True, 'pm_is_planning_object': True,
               'obj_types': {}}
    holders = [('TSO', planning.transmission_network)] + [
        (f'DSO_{k}', v) for k, v in planning.distribution_networks.items()]
    for name, holder in holders:
        binding['obj_types'][name] = holder.params.obj_type
    for y in years:
        pm[y] = [float(p) for p in planning.prob_market_scenarios[y]]
        for d in days:
            ref = planning.cost_energy_p[y][d]
            prices[(y, d)] = np.asarray(ref, dtype=float)
            for name, holder in holders:
                net = holder.network[y][d]
                binding['tso_and_dso_price_array_is_planning_object'] &= net.cost_energy_p is ref
                binding['pm_is_planning_object'] &= net.prob_market_scenarios is planning.prob_market_scenarios[y]
    return {'case': case_rel, 'case_sha256': _sha256_file(os.path.join(REPO, case_rel)),
            'years': years, 'days': days, 'num_market_scenarios': planning.num_market_scenarios,
            'random_seed': planning.random_seed, 'market_data_file': planning.market_data_file,
            'discount_factor': planning.discount_factor,
            'scenario_checksum': meta['combined_scenario_checksum'],
            'market_scenario_checksum': meta['market_scenario_checksum'],
            'binding': binding, 'pm': pm, 'prices': prices, 'stopped_at_declared_point': stopped}


def energy_growth_factor():
    """Production's growth factor: `_read_market_base_profiles` + the lookup expression of
    `_read_market_data_from_file` (shared_resources_planning.py)."""
    import shared_resources_planning as srp
    base = srp._read_market_base_profiles(os.path.join(DATA_DIR, 'MarketData', 'SRP1_market_data.xlsx'))
    gf = base['growth_factors']
    return float(gf[gf['Growth factors'] == 'Energy']['Value, [%]'].iloc[0])


# ======================================================================================
#  analysis
# ======================================================================================
def instance_spreads(rd):
    """Per (year, day): each scenario, the pm-weighted mean profile, the pm-weighted average of spreads."""
    import numpy as np
    years, days = rd['years'], rd['days']
    y0 = min(years)
    ns = rd['num_market_scenarios']
    rows = {'mean_profile': [], 'avg_of_scenario_spreads': []}
    for s in range(ns):
        rows[f's{s + 1}'] = []
    for y in sorted(years):
        pm = np.asarray(rd['pm'][y])
        for d in days:
            c = rd['prices'][(y, d)]
            if c.shape[0] != ns:
                raise RuntimeError(f'{y}/{d}: {c.shape[0]} price rows != {ns} scenarios')
            wu, wr = weights(years, days, y, d, y0)
            base = {'year': y, 'day': d, 'num_years': years[y], 'num_days': days[d],
                    'weight_undiscounted': wu, 'weight_model_r2': wr}
            per_s = []
            for s in range(ns):
                sp = day_spreads(c[s])
                per_s.append(sp)
                rows[f's{s + 1}'].append({**base, **sp, 'prices': [float(x) for x in c[s]]})
            mean_c = pm @ c
            rows['mean_profile'].append({**base, **day_spreads(mean_c), 'pm': pm.tolist(),
                                         'prices': [float(x) for x in mean_c]})
            rows['avg_of_scenario_spreads'].append({**base, 'pm': pm.tolist(),
                                                    F_MM: float(sum(pm[s] * per_s[s][F_MM] for s in range(ns))),
                                                    F_TH: float(sum(pm[s] * per_s[s][F_TH] for s in range(ns))),
                                                    'mean': float(sum(pm[s] * per_s[s]['mean'] for s in range(ns)))})
    summary = {}
    for key, rs in rows.items():
        summary[key] = {
            'horizon_undiscounted': {F_TH: horizon(rs, F_TH, 'weight_undiscounted'),
                                     F_MM: horizon(rs, F_MM, 'weight_undiscounted'),
                                     'mean_price': horizon(rs, 'mean', 'weight_undiscounted')},
            'horizon_model_r2': {F_TH: horizon(rs, F_TH, 'weight_model_r2'),
                                 F_MM: horizon(rs, F_MM, 'weight_model_r2'),
                                 'mean_price': horizon(rs, 'mean', 'weight_model_r2')},
            'per_year_day_weighted': {str(y): {F_TH: horizon(rs, F_TH, 'weight_undiscounted', lambda r, y=y: r['year'] == y),
                                               F_MM: horizon(rs, F_MM, 'weight_undiscounted', lambda r, y=y: r['year'] == y),
                                               'mean_price': horizon(rs, 'mean', 'weight_undiscounted',
                                                                     lambda r, y=y: r['year'] == y)}
                                      for y in sorted(years)}}
    return rows, summary


def _cmp(a, b):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    return {'max_abs_diff': float(np.max(np.abs(a - b))), 'ls_scale_k_(a~k*b)': float(a @ b / (b @ b)),
            'pearson_r': float(np.corrcoef(a, b)[0, 1]), 'identical_within_tol': bool(np.max(np.abs(a - b)) <= MATCH_TOL)}


def correspondence(srp, paper, g):
    import numpy as np
    out = {'metric': ('a = SRP1 price row, b = paper profile (same day/season). max_abs_diff = max_p |a_p - b_p| '
                      f'(EUR/MWh; "identical" iff <= {MATCH_TOL}); ls_scale_k = <a,b>/<b,b> (least-squares a ~ k b); '
                      'pearson_r = correlation over the 24 hours. Growth-normalized: every profile divided by '
                      '(1+g)^(y-2025), g = production energy growth factor, i.e. brought to 2025 price level.'),
           'energy_growth_factor_g': g}
    common = sorted(set(srp['years']) & set(paper['years']))
    out['common_years'] = common
    same_year = {}
    for y in common:
        pm = np.asarray(paper['pm'][y])
        for d in srp['days']:
            a = srp['prices'][(y, d)][0]
            c = paper['prices'][(y, d)]
            rec = {f's{s + 1}': _cmp(a, c[s]) for s in range(c.shape[0])}
            rec['mean_profile'] = _cmp(a, pm @ c)
            same_year[f'{y}/{d}'] = rec
    out['same_year'] = same_year
    # growth-normalized: SRP1 (y, d) vs every paper (y', d, s) and each paper year's mean profile
    y0 = 2025
    norm = {}
    for y in sorted(srp['years']):
        for d in srp['days']:
            a = srp['prices'][(y, d)][0] / (1 + g) ** (y - y0)
            best, best_key, matches = None, None, []
            vs_mean = {}
            for y2 in sorted(paper['years']):
                c = paper['prices'][(y2, d)] / (1 + g) ** (y2 - y0)
                for s in range(c.shape[0]):
                    dev = float(np.max(np.abs(a - c[s])))
                    if best is None or dev < best:
                        best, best_key = dev, f'{y2}/s{s + 1}'
                    if dev <= MATCH_TOL:
                        matches.append(f'{y2}/s{s + 1}')
                vs_mean[str(y2)] = _cmp(a, np.asarray(paper['pm'][y2]) @ c)
            allmean = np.mean([np.asarray(paper['pm'][y2]) @ (paper['prices'][(y2, d)] / (1 + g) ** (y2 - y0))
                               for y2 in sorted(paper['years'])], axis=0)
            norm[f'{y}/{d}'] = {'nearest_paper_profile': best_key, 'nearest_max_abs_diff_2025_level': best,
                                'identical_paper_profiles': matches,
                                'vs_paper_mean_profile_each_year': vs_mean,
                                'vs_paper_mean_profile_all_years_equal_weight': _cmp(a, allmean)}
    out['growth_normalized'] = norm
    return out


def sample_prefix_check(srp_rd, paper_rd):
    """Mechanism check for the 2025 finding: pandas DataFrame.sample(n, random_state=k) on the
    100-row synthetic pool -- is the n=1 draw the first row of the n=5 draw for production's 2025
    selection seeds? (derive_random_seed and the seed labels as in _read_market_data_from_file.)"""
    import pandas as pd
    from helper_functions import derive_random_seed
    market_seed = derive_random_seed(paper_rd['random_seed'], 'market')
    pool = pd.DataFrame({'i': range(100)})
    res = {}
    for d in paper_rd['days']:
        k = derive_random_seed(market_seed, 'selection', 'energy', 2025, str(d))
        one = list(pool.sample(n=srp_rd['num_market_scenarios'], random_state=k).index)
        five = list(pool.sample(n=paper_rd['num_market_scenarios'], random_state=k).index)
        res[str(d)] = {'seed': k, 'n1_rows': one, 'n5_rows': five, 'n1_is_prefix_of_n5': five[:len(one)] == one}
    return {'note': ('index-level check only, on a dummy 100-row frame (the synthetic pool size n_samples=100 of '
                     '_generate_market_price_scenarios); no prices are generated here'),
            'same_random_seed_both_cases': srp_rd['random_seed'] == paper_rd['random_seed'],
            'same_market_file_both_cases': srp_rd['market_data_file'] == paper_rd['market_data_file'],
            'per_day': res}


# ======================================================================================
#  markdown
# ======================================================================================
def _f(x, nd=2):
    return f'{x:,.{nd}f}'


def markdown(res):
    L = [f'# {STAGE}', '', f'Started {res["started_utc"]}; HEAD {res["git_HEAD"]}; script sha256 '
         f'{res["script_sha256"]}.', '', 'Authority: ' + '; '.join(AUTHORITY) + '.', '',
         '**Zero solves** (armed SolveProfileGuard(permitted=()), verify(0) failures: '
         f'{res["solve_profile_guard"]["verify_0_failures"]}; counts {res["solve_profile_guard"]["counts"]}). '
         f'**No model construction** (blocked calls: {res["model_construction_blocked_calls"]}; blockers '
         f'{res["model_construction_blockers"]}). Declared stop point hits: {res["stop_point"]["hits"]} '
         f'(declared {DECLARED_STOPS}).', '']
    L += ['## Instances (production reader)', '',
          '| instance | case | case sha256 | scenario checksum | expected | match | years (num_years) | market scen. | pm (network) |',
          '|---|---|---|---|---|---|---|---|---|']
    for k in ('srp1', 'paper'):
        r = res['instances'][k]
        L.append(f'| {k} | {r["case"]} | {r["case_sha256"][:12]} | {r["scenario_checksum"]} | '
                 f'{r["expected_checksum"][:8]}... ({r["expected_checksum_source"]}) | {r["checksum_matches"]} | '
                 f'{r["years"]} | {r["num_market_scenarios"]} | {sorted(set(tuple(v) for v in r["pm"].values()))} |')
    L += ['', 'pm source: ' + res['pm_source'], '',
          'Binding checks: ' + json.dumps({k: res['instances'][k]['binding'] for k in ('srp1', 'paper')}), '',
          f'SRP1 prices from this read vs Z4 committed prices: max |diff| = '
          f'{res["srp1_vs_z4_committed_prices_max_abs_diff"]:.3e}.', '',
          f'## Spread definition', '', SPREAD_FORMULA, '',
          f'Z4 reproduction (definition identity): max per-day |diff| {res["z4_reproduction"]["max_abs_dev_per_day"]:.3e}; '
          f'recomputed horizon (undiscounted) {res["z4_reproduction"]["horizon_top4_undiscounted_recomputed"]!r} vs '
          f'committed {res["z4_reproduction"]["committed_undiscounted"]!r}; ok = {res["z4_reproduction"]["ok"]}.', '']
    P = res['paper']['summary']
    keys = [k for k in P if k.startswith('s')] + ['mean_profile', 'avg_of_scenario_spreads']
    years = sorted(res['instances']['paper']['years'], key=int)
    L += ['## Paper scale: horizon-weighted spreads (EUR/MWh)', '',
          'Rows: scenario s (its row in every (year, day) block), the pm-weighted mean price profile, and the '
          'pm-weighted average of the scenario spreads. Weight = num_years x num_days (undiscounted) and model '
          'block weight (/1.02^(y-2025)).', '',
          f'| profile | 4 h spread undisc. | 4 h spread r2 | max-min undisc. | max-min r2 | mean price undisc. | '
          + ' | '.join(f'4 h {y}' for y in years) + ' | ' + ' | '.join(f'max-min {y}' for y in years) + ' |',
          '|---' * (6 + 2 * len(years)) + '|']
    for k in keys:
        s = P[k]
        L.append(f'| {k} | {_f(s["horizon_undiscounted"][F_TH])} | {_f(s["horizon_model_r2"][F_TH])} | '
                 f'{_f(s["horizon_undiscounted"][F_MM])} | {_f(s["horizon_model_r2"][F_MM])} | '
                 f'{_f(s["horizon_undiscounted"]["mean_price"])} | '
                 + ' | '.join(_f(s['per_year_day_weighted'][y][F_TH]) for y in years) + ' | '
                 + ' | '.join(_f(s['per_year_day_weighted'][y][F_MM]) for y in years) + ' |')
    S = res['srp1']['summary']['s1']
    sy = sorted(S['per_year_day_weighted'], key=int)
    L += ['', f'SRP1 (single scenario, recomputed here): 4 h undiscounted {_f(S["horizon_undiscounted"][F_TH])}, '
          f'r2 {_f(S["horizon_model_r2"][F_TH])}; max-min undiscounted {_f(S["horizon_undiscounted"][F_MM])}; per year 4 h '
          + ', '.join(f'{y}: {_f(S["per_year_day_weighted"][y][F_TH])}' for y in sy) + '.', '']
    L += ['## Paper scale: per (year, day) 4 h spread', '', '| year/day | ' + ' | '.join(keys) + ' |',
          '|---' * (len(keys) + 1) + '|']
    rows = res['paper']['per_representative_day']
    for i, r0 in enumerate(rows['s1']):
        L.append(f'| {r0["year"]}/{r0["day"]} | ' + ' | '.join(_f(rows[k][i][F_TH]) for k in keys) + ' |')
    C = res['correspondence']
    L += ['', '## SRP1 correspondence', '', 'Metric: ' + C['metric'], '',
          f'Same-year comparison ({C["common_years"]}, the only year both instances share):', '',
          '| day | ' + ' | '.join(f'{k} max|diff| / k / r' for k in C['same_year'][next(iter(C['same_year']))]) + ' |',
          '|---' * (1 + len(C['same_year'][next(iter(C['same_year']))])) + '|']
    for key, rec in C['same_year'].items():
        L.append(f'| {key} | ' + ' | '.join(f'{_f(v["max_abs_diff"], 3)} / {v["ls_scale_k_(a~k*b)"]:.4f} / '
                                            f'{v["pearson_r"]:.4f}' for v in rec.values()) + ' |')
    L += ['', 'Growth-normalized (2025 level) comparison over all SRP1 years:', '',
          '| SRP1 year/day | nearest paper profile | max|diff| | identical paper profiles | vs paper all-year mean profile max|diff| / k / r |',
          '|---|---|---|---|---|']
    for key, rec in C['growth_normalized'].items():
        m = rec['vs_paper_mean_profile_all_years_equal_weight']
        L.append(f'| {key} | {rec["nearest_paper_profile"]} | {_f(rec["nearest_max_abs_diff_2025_level"], 3)} | '
                 f'{rec["identical_paper_profiles"]} | {_f(m["max_abs_diff"], 3)} / {m["ls_scale_k_(a~k*b)"]:.4f} / '
                 f'{m["pearson_r"]:.4f} |')
    L += ['', 'Mechanism check (index level): ' + json.dumps(res['sample_prefix_check']), '',
          'Finding: ' + res['correspondence_finding'], '']
    R = res['ratios']
    L += ['## Ratios -- PREDICTION for the paper-scale evaluations', '', R['formula'], '',
          '| quantity | value |', '|---|---|']
    for k, v in R['values'].items():
        L.append(f'| {k} | {v!r} |')
    L += ['', 'PREDICTION: ' + R['prediction'], '',
          f'Expert prediction (spec v17 Z31_expert): {res["expert_prediction_Z31"]} -- check: '
          f'{res["checks"]["mean_profile_spread_le_avg_of_scenario_spreads_every_day"]} (every (year, day)), '
          f'{res["checks"]["mean_profile_spread_le_avg_of_scenario_spreads_horizon"]} (horizon).', '',
          '## Checks', '', '| check | result |', '|---|---|']
    for k, v in res['checks'].items():
        L.append(f'| {k} | {v} |')
    L += ['', f'ok = {res["ok"]}', '']
    return '\n'.join(L)


# ======================================================================================
#  main
# ======================================================================================
def write_manifest():
    entries = {}
    for dirpath, _dirs, files in os.walk(OUT_DIR):
        for f in sorted(files):
            p = os.path.join(dirpath, f)
            rel = os.path.relpath(p, OUT_DIR)
            if rel == MANIFEST_NAME:
                continue
            entries[rel] = _sha256_file(p)
    path = os.path.join(OUT_DIR, MANIFEST_NAME)
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite {path}')
    with open(path, 'w') as handle:
        json.dump({'created_utc': _utc(), 'directory': OUT_REL, 'files': dict(sorted(entries.items())),
                   'script': {'p515_s47_market_spreads.py': _sha256_file(os.path.abspath(__file__))}},
                  handle, indent=2)
    print(f'[manifest] {len(entries)} files -> {path}')


def _jsonable(obj):
    if isinstance(obj, dict):
        return {(k if isinstance(k, str) else '/'.join(map(str, k)) if isinstance(k, tuple) else str(k)):
                _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return obj


def main():
    if '--manifest' in sys.argv:
        write_manifest()
        return 0
    _refuse_concurrent()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_json = os.path.join(OUT_DIR, RESULTS_NAME)
    for name in (RESULTS_NAME, MD_NAME, STDOUT_CAPTURE, MANIFEST_NAME):
        if os.path.exists(os.path.join(OUT_DIR, name)):
            raise RuntimeError(f'write-once: {name} exists in {OUT_REL}')
    started, t0 = _utc(), time.time()
    head = _git(['rev-parse', 'HEAD']).strip()
    print(f'[{STAGE}] start {started}; HEAD {head}', flush=True)
    guard = SolveProfileGuard(permitted=(), label=STAGE).install()
    import numpy as np
    import p56a_oracle as O
    blocks = Blocks()
    blocks.install()
    res = {'stage': STAGE, 'authority': AUTHORITY, 'started_utc': started, 'git_HEAD': head,
           'script_sha256': _sha256_file(os.path.abspath(__file__)), 'spread_formula': SPREAD_FORMULA,
           'expert_prediction_Z31': json.load(open(os.path.join(
               REPO, 'data/SRP1/Results/P515S47/frozen_s47_baseline_spec_v17_ff0056b8.json')))[
               'predictions_recorded_before_run']['Z31_expert']}
    checks = {}
    ok = True
    try:
        with open(os.path.join(OUT_DIR, STDOUT_CAPTURE), 'w') as sink:
            print('[read] SRP1 ...', flush=True)
            srp_rd = production_read(SRP1_CASE_REL, 'srp1', sink)
            print('[read] paper ...', flush=True)
            paper_rd = production_read(PAPER_CASE_REL, 'paper', sink)
        g = energy_growth_factor()
    finally:
        blocks.uninstall()
        guard.uninstall()
    failures = guard.verify(0)
    res['solve_profile_guard'] = {'permitted': [], 'counts': guard.counts, 'verify_0_failures': failures}
    res['model_construction_blockers'] = [f'{c.__name__}.{a}' for c, a in blocks.targets]
    res['model_construction_blocked_calls'] = blocks.blocked_calls
    res['stop_point'] = {'target': f'{blocks.stop_target[0].__name__}.{blocks.stop_target[1]}',
                         'hits': blocks.stop_hits, 'declared': DECLARED_STOPS,
                         'why': ('first statement after market prices are generated, selected, bound to every '
                                 'TSO/DSO block and the scenario checksum is computed; keeps the run independent '
                                 'of the concurrently edited SRP1_ESS_Params.json'),
                         'site': _one_line('shared_resources_planning.py', 'shared_ess_data.read_parameters_from_file()')}
    checks['solve_guard_verify_0'] = not failures
    checks['no_model_construction_call'] = len(blocks.blocked_calls) == 0
    checks['stop_point_hits_exactly_declared'] = blocks.stop_hits == DECLARED_STOPS

    expected = {'srp1': (O.CANONICAL_CHECKSUM, 'p56a_oracle.CANONICAL_CHECKSUM'),
                'paper': (PAPER_CHECKSUM_RECORDED, PAPER_CHECKSUM_SOURCE)}
    inst = {}
    for k, rd in (('srp1', srp_rd), ('paper', paper_rd)):
        inst[k] = {x: rd[x] for x in ('case', 'case_sha256', 'years', 'days', 'num_market_scenarios', 'random_seed',
                                      'market_data_file', 'discount_factor', 'scenario_checksum',
                                      'market_scenario_checksum', 'binding', 'pm')}
        inst[k]['expected_checksum'], inst[k]['expected_checksum_source'] = expected[k]
        inst[k]['checksum_matches'] = rd['scenario_checksum'] == expected[k][0]
        checks[f'{k}_scenario_checksum_matches'] = inst[k]['checksum_matches']
        checks[f'{k}_prices_bound_to_every_block_are_planning_object'] = \
            rd['binding']['tso_and_dso_price_array_is_planning_object']
        checks[f'{k}_network_pm_is_planning_prob_market_scenarios'] = rd['binding']['pm_is_planning_object']
        checks[f'{k}_no_congestion_management_network'] = all(
            v != OBJ_CONGESTION_MANAGEMENT for v in rd['binding']['obj_types'].values())
        checks[f'{k}_discount_factor_0_02'] = rd['discount_factor'] == PRODUCTION_RATE
    checks['paper_pm_all_0_2'] = all(np.allclose(v, [0.2] * 5) for v in paper_rd['pm'].values())
    res['instances'] = inst
    res['pm_source'] = (
        'network.prob_market_scenarios of every TSO/DSO block (' +
        _one_line('shared_resources_planning.py', 'transmission_network.network[year][day].prob_market_scenarios = planning_problem.prob_market_scenarios[year]')
        + ', ' + _one_line('shared_resources_planning.py', 'distribution_network.network[year][day].prob_market_scenarios = planning_problem.prob_market_scenarios[year]')
        + ') = planning.prob_market_scenarios[year] = [1/NumMarketScenarios] x NumMarketScenarios ('
        + _one_line('shared_resources_planning.py', 'planning_problem.prob_market_scenarios[year] = [(1 / planning_problem.num_market_scenarios)] * planning_problem.num_market_scenarios')
        + '); identity checked above. Consistent with the W3 probability audit (P515S45/probability_audit, [0.2]x5).')
    res['price_source'] = {'generation_and_selection': _one_line('shared_resources_planning.py',
                                                                 'def _read_market_data_from_file(planning_problem):'),
                           'price_array': _one_line('shared_resources_planning.py',
                                                    'planning_problem.cost_energy_p[year][day] = np.array(energy_selected_profiles * energy_growth_cumul)'),
                           'market_file': MARKET_REL, 'market_file_sha256': _sha256_file(os.path.join(REPO, MARKET_REL))}

    z4 = z4_reproduction()
    zp = z4.pop('z4_prices')
    res['z4_reproduction'] = z4
    checks['z4_definition_reproduced_to_1e-9'] = z4['ok']
    dev = max(float(np.max(np.abs(srp_rd['prices'][(y, d)][0] - np.asarray(zp[(y, d)]))))
              for y in srp_rd['years'] for d in srp_rd['days'])
    res['srp1_vs_z4_committed_prices_max_abs_diff'] = dev
    checks['srp1_prices_equal_z4_committed_prices'] = dev <= REPRO_TOL

    srp_rows, srp_sum = instance_spreads(srp_rd)
    pap_rows, pap_sum = instance_spreads(paper_rd)
    res['srp1'] = {'summary': srp_sum}
    checks['srp1_recomputed_spread_equals_97.99_to_1e-9'] = abs(
        srp_sum['s1']['horizon_undiscounted'][F_TH] - SRP1_Z4_SPREAD) <= REPRO_TOL
    res['paper'] = {'summary': pap_sum, 'per_representative_day': pap_rows}
    checks['mean_profile_spread_le_avg_of_scenario_spreads_every_day'] = all(
        m[F_TH] <= a[F_TH] + 1e-12 for m, a in zip(pap_rows['mean_profile'], pap_rows['avg_of_scenario_spreads']))
    checks['mean_profile_spread_le_avg_of_scenario_spreads_horizon'] = (
        pap_sum['mean_profile']['horizon_undiscounted'][F_TH] <= pap_sum['avg_of_scenario_spreads']['horizon_undiscounted'][F_TH])

    corr = correspondence(srp_rd, paper_rd, g)
    res['correspondence'] = corr
    res['sample_prefix_check'] = sample_prefix_check(srp_rd, paper_rd)
    sy = corr['same_year']
    ident = {k: [s for s, v in rec.items() if v['identical_within_tol']] for k, rec in sy.items()}
    norm_ident = {k: v['identical_paper_profiles'] for k, v in corr['growth_normalized'].items()}
    res['correspondence_finding'] = (
        f'Same year ({corr["common_years"]}): paper profiles identical (max|diff| <= {MATCH_TOL}) to SRP1 per day: '
        f'{ident}. Growth-normalized, SRP1 (year/day) identical to paper (year/scenario): {norm_ident}. '
        'Scenario labels are per (year, day) block draws, not trajectories across blocks (each block selects '
        'its rows with its own seed).')

    mp_u = pap_sum['mean_profile']['horizon_undiscounted'][F_TH]
    av_u = pap_sum['avg_of_scenario_spreads']['horizon_undiscounted'][F_TH]
    mp_r = pap_sum['mean_profile']['horizon_model_r2'][F_TH]
    av_r = pap_sum['avg_of_scenario_spreads']['horizon_model_r2'][F_TH]
    # like-for-like on the price level: divide each (year, day) spread by (1+g)^(y-2025) before weighting
    def norm_h(rows_, wkey):
        return (sum(r[F_TH] / (1 + g) ** (r['year'] - 2025) * r[wkey] for r in rows_)
                / sum(r[wkey] for r in rows_))
    vals = {
        'SRP1_4h_spread_undiscounted (Z4)': SRP1_Z4_SPREAD,
        'paper_mean_profile_4h_spread_undiscounted': mp_u,
        'paper_avg_of_scenario_4h_spreads_undiscounted': av_u,
        'R = mean_profile / SRP1 (undiscounted)': mp_u / SRP1_Z4_SPREAD,
        'avg_of_scenario_spreads / SRP1 (undiscounted)': av_u / SRP1_Z4_SPREAD,
        'flattening = mean_profile / avg_of_scenario_spreads (undiscounted)': mp_u / av_u,
        'SRP1_4h_spread_r2 (Z4)': SRP1_Z4_SPREAD_R2,
        'R_r2 = mean_profile_r2 / SRP1_r2': mp_r / SRP1_Z4_SPREAD_R2,
        'avg_of_scenario_spreads_r2 / SRP1_r2': av_r / SRP1_Z4_SPREAD_R2,
        'like-for-like 2025 (only common year): paper mean_profile 2025 / SRP1 2025 (day-weighted)':
            pap_sum['mean_profile']['per_year_day_weighted']['2025'][F_TH]
            / srp_sum['s1']['per_year_day_weighted']['2025'][F_TH],
        'like-for-like 2025 (only common year): paper avg_of_scenario_spreads 2025 / SRP1 2025 (day-weighted)':
            pap_sum['avg_of_scenario_spreads']['per_year_day_weighted']['2025'][F_TH]
            / srp_sum['s1']['per_year_day_weighted']['2025'][F_TH],
        'supplementary: growth-normalized (2025 level) R = paper mean_profile / SRP1, undiscounted weights':
            norm_h(pap_rows['mean_profile'], 'weight_undiscounted') / norm_h(srp_rows['s1'], 'weight_undiscounted'),
        'supplementary: growth-normalized avg_of_scenario_spreads / SRP1, undiscounted weights':
            norm_h(pap_rows['avg_of_scenario_spreads'], 'weight_undiscounted') / norm_h(srp_rows['s1'], 'weight_undiscounted'),
    }
    res['ratios'] = {
        'formula': (f'R = [sum_(y,d) w * {F_TH}(mean profile)] / [sum w] (paper, w = num_years x num_days, '
                    f'undiscounted) / {SRP1_Z4_SPREAD!r} (SRP1, Z4, same weighting); the average of scenario spreads '
                    f'replaces the mean-profile spread by sum_s pm_s * {F_TH}(scenario s) per (y, d); the r2 variants '
                    f'use the model block weight and SRP1 r2 = {SRP1_Z4_SPREAD_R2!r}; the growth-normalized '
                    f'supplementary ratios divide every (y, d) spread by (1+g)^(y-2025) before weighting, g = {g!r}, '
                    f'removing the effect of the two instances\' different representative years.'),
        'values': vals,
        'prediction': (f'first-order estimate of the paper-scale arbitrage value per MWh relative to SRP1: R = '
                       f'{mp_u / SRP1_Z4_SPREAD:.4f} (mean-profile 4 h spread {mp_u:.2f} vs SRP1 {SRP1_Z4_SPREAD:.2f} '
                       f'EUR/MWh); the average of the scenario spreads would give {av_u / SRP1_Z4_SPREAD:.4f}, the '
                       f'difference being the flattening of the non-anticipative mean price profile.')}
    res['checks'] = checks
    ok = all(checks.values())
    res['ok'] = ok
    res['elapsed_s'] = time.time() - t0
    post = _git(['status', '--porcelain'])
    res['post_git_status_tracked_changes'] = [l for l in post.splitlines() if not l.startswith('??')]
    res['ended_utc'] = _utc()
    res = _jsonable(res)
    with open(out_json, 'w') as handle:
        json.dump(res, handle, indent=1)
    with open(os.path.join(OUT_DIR, MD_NAME), 'w') as handle:
        handle.write(markdown(res))
    print(json.dumps(vals, indent=1))
    print('[checks] ' + json.dumps(checks, indent=1))
    print(f'[{STAGE}] ok={ok}; guard counts {guard.counts}; blocked {blocks.blocked_calls}; '
          f'stop hits {blocks.stop_hits}; wrote {out_json}', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
