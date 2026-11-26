"""P5.15 S56 -- input preparation for the pilot on the common grid 2025/2030/2035 (ZERO SOLVES).

Planner task "E: input preparation" (TASKS.md, S56 order; S56_DESIGN_NOTE_v1.md D1, D2, additional requirements 1
and 3, pilot step 1).

1. SP INSTANCE "SRP1-3x3" (label s56_srp1grid_3x3). Derived from data/SRP1/SRP1.json by the route the W89 3 x 3
   instance used (p515_s53_w89_3x3_campaign.derive_instance_text: `p515_s44_scale_measurement.derive_case(base,
   overrides)` written with `json.dumps(case, indent='\\t')`), with base 'srp1' (years unchanged) and overrides
   {num_market_scenarios: 3, num_operation_scenarios: 3} -- ONLY the five scenario counts change. The route is
   re-checked here: derive_case('paper', {3, 3}) with the same writer must reproduce the committed W89 case bytes.
   Read through the production reader the 3 x 3 pair used (`p515_s44_scale_measurement.read_planning_from_derived_case`;
   plots / results / logs of each read redirected into reads/<label>/), the combined scenario checksum recorded.
2. BUILD GATES (each a check, expected vs observed): (a) prefix property, bitwise, every year: scenario row 0 of
   every SRP1-3x3 scenario array == the SRP1.json single-scenario array (at the checksum point and after the read);
   (b) the non-scenario object identical to SRP1.json's (audit A's recursive walk, its declared exclusions only); (c)
   12 (year, day) blocks, probabilities 1/3 per dimension and 1/9 per (s_m, s_o) pair on every network block; (d)
   mean energy price (production `model_construction_helpers.expected_market_price`) > 0 in every hour of every
   block (premium floor not needed), minimum reported; (e) admissible domain = 1,704 (the S55 rule, by import).
3. EV INSTANCE (label s56_srp1grid_3x3_ev): probability-weighted means of the SRP1-3x3 arrays (audit A's
   definition and functions, by import: `p515_s56_expected_profiles.compute_means`), loaded by the production reader
   + the injection hook of `p515_s56_ev_injection` at `_compute_scenario_metadata`; A2/A3 plain means; A5: the
   reactive-flexibility arrays keep production's constant (not injected). Verified: read-back bitwise vs the means,
   the object walk vs SRP1-3x3 (declared scenario fields only), the EV checksum; the in-child assertion
   `p515_s56_ev_injection.assert_ev_arrays` must PASS on the EV read and RAISE on a plain SRP1.json read, on the EV
   case read without injection, and on the SRP1-3x3 read.
4. PREDICTION 6 INPUT: ratio of the mean-profile horizon h-hour energy-price spread to the draw-0 (published)
   spread, and the ensemble expectation of per-scenario spreads, h = 4 and h = 2, day-weighted (undiscounted) and
   discount-weighted (model block weight) exactly as W22 / W52 (`p515_s47_market_spreads.weights` / `.horizon`).
5. SCENARIO UNCERTAINTY SUMMARY (audit A Q2 analogue, by import): scenario 0 / 1 / 2 vs mean per array family,
   ensemble spread per family, system aggregates per block, RES / Q-P / flexibility-sign facts.

Guards: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0, plus the
guards installed by the imported audit A, S55 and Phase B modules (each verified at 0).

Output: NEW directory data/SRP1/Results/P515S56/instances_v1/ (the launch command creates it with mkdir, which
fails if it exists, to hold launch.log; the script refuses if it holds anything else, or if another copy runs).

Launch (attached, both streams captured):
    mkdir data/SRP1/Results/P515S56/instances_v1
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_build_instances.py \\
        > data/SRP1/Results/P515S56/instances_v1/launch.log 2>&1
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 build instances (never solves)').install()

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager, redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402
import shared_energy_storage_data as ESSD  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402  (derive_case + the production read of a derived case)
import p515_s44_campaign_harness as H  # noqa: E402  (validate_derived_instance only; parent side, no model import)
import p515_s47_market_spreads as W22  # noqa: E402  (spread definition and weights, by import)
import p515_s56_expected_profiles as AA  # noqa: E402  (audit A functions; installs its own permitted=() guard)
import p515_s56_ev_injection as EVI  # noqa: E402

SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1')
# DRY RUN ONLY: `--scratch-out <absolute dir outside the repository>` redirects every output there (used to test the
# script before the write-once launch). The real launch passes no argument.
if '--scratch-out' in sys.argv:
    OUT_REL = os.path.abspath(sys.argv[sys.argv.index('--scratch-out') + 1])
    if OUT_REL.startswith(REPO + os.sep):
        raise SystemExit('--scratch-out must be outside the repository')
OUT_DIR = os.path.join(REPO, OUT_REL)
ALLOWED_PREEXISTING = {'launch.log'}

SP_LABEL = 's56_srp1grid_3x3'
EV_LABEL = 's56_srp1grid_3x3_ev'
SP_CASE_NAME = 'SRP1__s56_srp1grid_3x3.json'
EV_CASE_NAME = 'SRP1__s56_srp1grid_3x3_ev.json'
EV_PROFILES_NAME = 'SRP1__s56_srp1grid_3x3_ev_profiles.json'
SP_RECORD_NAME = 'instance_record.json'
EV_RECORD_NAME = 'ev_instance_record.json'
RESULTS_NAME = 'build_instances.json'
MANIFEST_NAME = 'manifest_sha256.json'

DERIVE_BASE = 'srp1'
DERIVE_OVERRIDES = {'num_market_scenarios': 3, 'num_operation_scenarios': 3}
N_SCEN = 3
CASE_SRP1_REL = os.path.join('data', 'SRP1', 'SRP1.json')
SRP1_CANONICAL_CHECKSUM = '5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358'
W89_CASE_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
W89_RECORD_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'instance_record.json')
S55_JSON_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'admissible_plan_count', 'admissible_plan_count.json')
S55_CSV_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'admissible_plan_count', 'admissible_plans.csv')
EXPECTED_YEARS = [2025, 2030, 2035]
EXPECTED_DOMAIN = 1704
EXPECTED_DOMAIN_SPLIT = {'2025': 149, '2030': 522, '2035': 1033}
Y0 = 2025
REF_UNIT_Z = (0, 0, 1, 2, 0, 0, 0)   # y2025: n7 0.25 MVA / 1 MWh (the reference unit)
TOL_EUR = 1e-6
REPRO_TOL = 1e-9

SPREAD_FORMULA = {
    'day_spread_h': ('per (year, day) and 24 hourly energy prices c (EUR/MWh, planning.cost_energy_p[year][day] row): '
                     'spread_h(c) = mean of the h highest c_t - mean of the h lowest c_t (sorted; W19 Z4 / W22 '
                     'definition, p515_s47_market_spreads.day_spreads for h = 4; checked bitwise)'),
    'profiles': ('mean_profile: spread_h(cbar), cbar[t] = sum_s pi_m[s] c[s, t] (the EV price profile); draw0: '
                 'spread_h(c[0]) (scenario row 0 = the published SRP1 single scenario, gate (a)); s1, s2: rows 1, 2; '
                 'expected: sum_s pi_m[s] spread_h(c[s]) (ensemble expectation of per-scenario spreads)'),
    'horizon': ('H = sum_b w_b spread_b / sum_b w_b over the 12 (year, day) blocks; w_u = num_years[y] * num_days[d] '
                '("day-weighted", undiscounted); w_r2 = w_u / 1.02^(y - 2025) ("discount-weighted", the model block '
                'weight, production DiscountFactor 0.02) -- p515_s47_market_spreads.weights / .horizon, as W52 '
                '(p515_s53_3x3_selection.py) used them'),
    'ratios': ('R_mean = H(mean_profile) / H(draw0); R_expected = H(expected) / H(draw0); mean_over_expected = '
               'H(mean_profile) / H(expected); each for w_u and w_r2, h = 4 and h = 2'),
}

CHECKS = []
STARTED = time.time()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, expected=None, observed=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'expected': expected, 'observed': observed})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + ('' if ok else f' -- expected {str(expected)[:300]}; observed '
                                                               f'{str(observed)[:600]}'))


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _git(args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def _input_record(path):
    tracked = bool(_git(['ls-files', '--', path]).strip())
    clean = not _git(['status', '--porcelain', '--', path]).strip() if tracked else None
    return {'path': path, 'sha256': _sha(path), 'tracked': tracked, 'clean': clean}


def _rel(path):
    return os.path.relpath(path, REPO)


def _other_copy_running():
    ps = subprocess.run(['ps', '-A', '-o', 'pid=,command='], capture_output=True, text=True, check=True).stdout
    me = os.getpid()
    found = []
    for ln in ps.splitlines():
        tok = ln.split()
        if len(tok) < 3 or int(tok[0]) == me:
            continue
        if os.path.basename(tok[1]).startswith('python') and any(os.path.basename(t) == SCRIPT for t in tok[2:4]):
            found.append(ln.strip())
    return found


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


class _Stages:
    @contextmanager
    def stage(self, _name):
        yield


def production_read(case_path_abs, label, profiles=None):
    """Production read of the case via the 3 x 3 pair's reader, with an EVI.ScenarioPointHook armed (capture only,
    or injection when `profiles` is given). Returns planning, the snapshot at the checksum point, the read record and
    the hook."""
    read_dir = os.path.join(OUT_DIR, 'reads', label)
    if os.path.exists(read_dir):
        raise RuntimeError(f'refusing to overwrite {read_dir}')
    os.makedirs(read_dir)
    snaps, point_checks = [], []

    def on_point(planning):
        snaps.append(AA.snapshot(planning))
        if profiles is not None:   # the in-child assertion at the injection point (checksum not yet computed)
            try:
                point_checks.append(EVI.assert_ev_arrays(planning, profiles, point='checksum'))
            except EVI.EvInjectionMismatch as exc:
                point_checks.append({'ok': False, 'error': str(exc)[:2000], 'summary': exc.summary})
    hook = EVI.ScenarioPointHook(profiles=profiles, on_point=on_point).install()
    stream = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(stream):
            planning = S.read_planning_from_derived_case({'derived_case': {'path': _rel(case_path_abs)}}, read_dir,
                                                         _Stages())
    finally:
        hook.uninstall()
        with open(os.path.join(read_dir, 'production_stdout.log'), 'w') as handle:
            handle.write(stream.getvalue())
    if hook.calls != 1 or len(snaps) != 1:
        raise RuntimeError(f'{label}: checksum hook called {hook.calls} times (expected 1)')
    rec = {'case_path': _rel(case_path_abs), 'case_sha256': _sha(case_path_abs), 'wall_s': time.time() - t0,
           'read_dir': _rel(read_dir), 'scenario_metadata': dict(planning.scenario_metadata),
           'injection': hook.inject_counts, 'assertion_at_checksum_point': point_checks}
    return planning, snaps[0], rec


def strip_counts(case):
    c = json.loads(json.dumps(case))
    c.pop('NumMarketScenarios', None)
    c['TransmissionNetwork'].pop('num_operation_scenarios', None)
    for dn in c['DistributionNetworks']:
        dn.pop('num_operation_scenarios', None)
    return c


# ======================================================================================================================
#  gates
# ======================================================================================================================
def gate_prefix(snap_sp, snap_1, label):
    """(a) row 0 of every SRP1-3x3 array == the SRP1.json single-scenario array, bitwise (float64 bytes)."""
    same_keys = list(snap_sp) == list(snap_1)
    n, n_eq, by_year, by_fam, mism, shapes_ok = 0, 0, Counter(), Counter(), [], True
    for k in snap_sp:
        a3 = snap_sp[k]['A']
        if k not in snap_1:
            mism.append({'key': AA.key_str(k), 'what': 'missing in SRP1'})
            continue
        a1 = snap_1[k]['A']
        shapes_ok = shapes_ok and a3.shape == (N_SCEN, 24) and a1.shape == (1, 24)
        n += 1
        year = k[2] if k[0] == 'market' else k[1]
        if a3.shape[1:] == a1.shape[1:] and a1.shape[0] == 1 and a3[0:1].tobytes() == a1.tobytes():
            n_eq += 1
            by_year[str(year)] += 1
            by_fam[AA.family_of(k)] += 1
        else:
            mism.append({'key': AA.key_str(k), 'shape_sp': list(a3.shape), 'shape_srp1': list(a1.shape)})
    n_by_year = Counter(str(k[2] if k[0] == 'market' else k[1]) for k in snap_sp)
    holds = same_keys and shapes_ok and n_eq == n == len(snap_sp) and not mism
    return {'point': label, 'definition': ('for every scenario array of the SRP1-3x3 read (market energy / flexibility '
                                           'prices per (year, day); per network block, per load pd, qd, flexibility '
                                           'P up / down, Q up / down; per generator pg, qg), row 0 as float64 bytes '
                                           '== the SRP1.json read\'s single row'),
            'same_key_list_in_order': same_keys, 'shapes_3x24_and_1x24': shapes_ok, 'n_arrays': len(snap_sp),
            'n_compared': n, 'n_equal': n_eq, 'n_by_year': dict(sorted(n_by_year.items())),
            'n_equal_by_year': dict(sorted(by_year.items())), 'n_equal_by_family': dict(sorted(by_fam.items())),
            'n_mismatches': len(mism), 'mismatches_first': mism[:20], 'holds': holds}


def gate_blocks_and_probabilities(p):
    out = {'years': {str(y): v for y, v in p.years.items()}, 'days': dict(p.days),
           'n_year_day_blocks': len(p.years) * len(p.days),
           'blocks_per_holder': {h: sum(1 for y in hd.years for d in hd.days if hd.network[y][d] is not None)
                                 for h, hd in EVI.holders(p)},
           'expected_block_counts': S.expected_block_counts(p),
           'solves_per_cycle': S.declared_solve_profile(p, 1)['solves_per_cycle']}
    third = [1.0 / 3.0] * 3
    out['market_probabilities'] = {str(y): list(p.prob_market_scenarios[y]) for y in sorted(p.years)}
    pair_bad, pair_sums, n_pairs, op_bad, mk_bad = [], [], 0, [], []
    for h, hd in EVI.holders(p):
        for y in hd.years:
            for d in hd.days:
                net = hd.network[y][d]
                if list(net.prob_operation_scenarios) != third:
                    op_bad.append(f'{h}|{y}|{d}')
                if list(net.prob_market_scenarios) != third:
                    mk_bad.append(f'{h}|{y}|{d}')
                tot = 0.0
                for s_m in range(len(net.prob_market_scenarios)):
                    for s_o in range(len(net.prob_operation_scenarios)):
                        pr = SRP._scenario_probability(net, s_m, s_o)
                        n_pairs += 1
                        tot += pr
                        if pr != 1.0 / 9.0:
                            pair_bad.append(f'{h}|{y}|{d}|{s_m}|{s_o}:{pr!r}')
                pair_sums.append(tot)
    out.update({'n_network_blocks': len(pair_sums), 'n_pairs_checked': n_pairs,
                'pair_probability_value': 1.0 / 9.0, 'pairs_not_one_ninth': pair_bad[:10],
                'n_pairs_not_one_ninth': len(pair_bad), 'pair_probability_sum_min': min(pair_sums),
                'pair_probability_sum_max': max(pair_sums), 'blocks_operation_prob_not_third': op_bad[:10],
                'blocks_market_prob_not_third': mk_bad[:10],
                'block_weights_tso': {f'{y}|{d}': SRP._get_admm_block_weight(p.transmission_network, y, d)
                                      for y in p.transmission_network.years for d in p.transmission_network.days},
                'median_block_weight_production': SRP._compute_median_admm_block_weight(p)})
    return out


def gate_premium_floor(p, means):
    vals = []
    for h, hd in EVI.holders(p):
        for y in hd.years:
            for d in hd.days:
                net = hd.network[y][d]
                for t in range(p.num_instants):
                    vals.append((float(MCH.expected_market_price(net, t)), f'{h}|{y}|{d}|h{t}'))
    vmin = min(vals)
    from_means = min((float(np.min(means[('market', 'energy', y, d)])), f'{y}|{d}') for y in p.years for d in p.days)
    per_block = {f'{y}|{d}': {'min_hourly_mean_price': float(np.min(means[('market', 'energy', y, d)])),
                              'max_hourly_mean_price': float(np.max(means[('market', 'energy', y, d)]))}
                 for y in sorted(p.years) for d in p.days}
    return {'definition': ('pibar_t = model_construction_helpers.expected_market_price(network, t) = sum_sm '
                           'prob_market_scenarios[s_m] * cost_energy_p[s_m][t], every hour of every network block of '
                           'the TSO and every DSO (Addendum 38: a premium floor only if some pibar_t <= 0)'),
            'n_hours_checked': len(vals), 'min_hourly_mean_price': vmin[0], 'argmin': vmin[1],
            'n_nonpositive': sum(1 for v, _ in vals if v <= 0.0),
            'min_from_audit_A_means': from_means[0], 'argmin_from_means': from_means[1],
            'per_block': per_block, 'premium_floor_needed': bool(vmin[0] <= 0.0)}


# ======================================================================================================================
#  spreads (prediction 6 input)
# ======================================================================================================================
def spread_h(prices, h):
    srt = np.sort(np.asarray(prices, dtype=float))
    return float(srt[-h:].mean() - srt[:h].mean())


def spreads(p, snap_sp, means, snap_1):
    out = {'formula': SPREAD_FORMULA, 'discount_rate_used': W22.PRODUCTION_RATE,
           'planning_discount_factor': p.discount_factor}
    _ck('spreads: W22 discount rate == planning DiscountFactor', W22.PRODUCTION_RATE == p.discount_factor,
        W22.PRODUCTION_RATE, p.discount_factor)
    w4_bitwise, w_eq_model = True, True
    for h in (4, 2):
        rows = []
        for y in sorted(p.years):
            for d in p.days:
                ke = ('market', 'energy', y, d)
                A, w = snap_sp[ke]['A'], snap_sp[ke]['w']
                wu, wr = W22.weights(p.years, p.days, y, d, Y0)
                w_eq_model = w_eq_model and abs(wr - SRP._get_admm_block_weight(p.transmission_network, y, d)) <= 1e-9 * wr
                r = {'year': y, 'day': d, 'weight_undiscounted': wu, 'weight_model_r2': wr,
                     'mean_profile': spread_h(means[ke].ravel(), h),
                     'draw0': spread_h(A[0], h), 's1': spread_h(A[1], h), 's2': spread_h(A[2], h),
                     'expected': float(sum(w[s] * spread_h(A[s], h) for s in range(A.shape[0]))),
                     'srp1_read': spread_h(snap_1[ke]['A'][0], h)}
                if h == 4:
                    w4_bitwise = w4_bitwise and (r['mean_profile'] == W22.day_spreads(means[ke].ravel())[W22.F_TH]
                                                 and r['draw0'] == W22.day_spreads(A[0])[W22.F_TH])
                rows.append(r)
        hz = {wk: {f: W22.horizon(rows, f, wk) for f in ('mean_profile', 'draw0', 's1', 's2', 'expected', 'srp1_read')}
              for wk in ('weight_undiscounted', 'weight_model_r2')}
        ratios = {wk: {'R_mean = H(mean_profile)/H(draw0)': v['mean_profile'] / v['draw0'],
                       'R_expected = H(expected)/H(draw0)': v['expected'] / v['draw0'],
                       'H(mean_profile)/H(expected)': v['mean_profile'] / v['expected']} for wk, v in hz.items()}
        per_year = {str(y): {wk: {f: W22.horizon(rows, f, wk, lambda r, y=y: r['year'] == y)
                                  for f in ('mean_profile', 'draw0', 'expected')}
                             for wk in ('weight_undiscounted', 'weight_model_r2')} for y in sorted(p.years)}
        out[f'h{h}'] = {'horizon': hz, 'ratios': ratios, 'per_year': per_year,
                        'per_block': {f'{r["year"]}|{r["day"]}': {k: r[k] for k in r if k not in ('year', 'day')}
                                      for r in rows}}
        _ck(f'spreads h={h}: draw-0 spread == SRP1.json read spread, every block (bitwise)',
            all(r['draw0'] == r['srp1_read'] for r in rows), True, [r['draw0'] - r['srp1_read'] for r in rows])
    _ck('spreads: h = 4 spread == p515_s47_market_spreads.day_spreads (bitwise; mean profile and draw 0)', w4_bitwise,
        True, w4_bitwise)
    _ck('spreads: W22 discount weight == production _get_admm_block_weight (<= 1e-9 rel)', w_eq_model, True, w_eq_model)
    d0u, d0r = out['h4']['horizon']['weight_undiscounted']['draw0'], out['h4']['horizon']['weight_model_r2']['draw0']
    _ck('spreads: draw-0 4 h horizon spread == W22 SRP1_Z4_SPREAD (undiscounted, <= 1e-9)',
        abs(d0u - W22.SRP1_Z4_SPREAD) <= REPRO_TOL, W22.SRP1_Z4_SPREAD, d0u)
    _ck('spreads: draw-0 4 h horizon spread == W22 SRP1_Z4_SPREAD_R2 (model weight, <= 1e-9)',
        abs(d0r - W22.SRP1_Z4_SPREAD_R2) <= REPRO_TOL, W22.SRP1_Z4_SPREAD_R2, d0r)
    return out


# ======================================================================================================================
#  uncertainty summary (audit A Q2 analogue)
# ======================================================================================================================
def uncertainty_summary(p, snap_sp, means):
    from definitions import GEN_CURTAILLABLE_TYPES
    keys = list(snap_sp)
    q = {}
    for s in range(N_SCEN):
        q[f'scenario{s}_vs_mean_all_years'] = AA.diff_stats(
            [(k, snap_sp[k]['A'][s:s + 1], means[k], snap_sp[k]['baseMVA']) for k in keys])
    for y in sorted(p.years):
        ky = [k for k in keys if (k[2] if k[0] == 'market' else k[1]) == y]
        q[f'scenario0_vs_mean_{y}'] = AA.diff_stats([(k, snap_sp[k]['A'][0:1], means[k], snap_sp[k]['baseMVA'])
                                                     for k in ky])
    q['ensemble_spread'] = AA.ensemble_ranges(snap_sp)
    agg = {}
    for y in sorted(p.years):
        for d in p.days:
            row = {'load_MWh_s0': 0.0, 'load_MWh_mean': 0.0, 'res_MWh_s0': 0.0, 'res_MWh_mean': 0.0}
            for k in keys:
                if k[0] == 'market' or k[1] != y or k[2] != d:
                    continue
                b = snap_sp[k]['baseMVA']
                if k[3] == 'load' and k[-1] == 'pd' and k[0] != 'TSO':   # every TSO load is an ADN load (audit A)
                    row['load_MWh_s0'] += float(np.sum(snap_sp[k]['A'][0]) * b)
                    row['load_MWh_mean'] += float(np.sum(means[k]) * b)
                if k[3] == 'gen' and k[-1] == 'pg' and k[5] in GEN_CURTAILLABLE_TYPES:
                    row['res_MWh_s0'] += float(np.sum(snap_sp[k]['A'][0]) * b)
                    row['res_MWh_mean'] += float(np.sum(means[k]) * b)
            ke = ('market', 'energy', y, d)
            row['mean_price_s0'] = float(np.mean(snap_sp[ke]['A'][0]))
            row['mean_price_mean'] = float(np.mean(means[ke]))
            agg[f'{y}|{d}'] = row
    q['system_aggregates_per_block'] = agg
    q['res_capacity'] = AA.res_capacity_facts(p, snap_sp)
    q['load_q_over_p'] = AA.load_pf_facts(snap_sp)
    q['flex_bounds_sign'] = AA.flex_bound_facts(snap_sp)
    return q


# ======================================================================================================================
#  (e) admissible domain (S55 rule, by import)
# ======================================================================================================================
def domain_gate(p_sp, p_1):
    import pyomo.environ as pe
    import p515_s55_count_admissible_plans as C   # installs its own permitted=() guard and a pickle block at import
    PB = C.PB
    res = {}
    with open(PB.INVESTMENT_COST_RESULTS['path']) as handle:
        w2 = json.load(handle)
    _ck('(e) W2 investment-cost results sha256 == PB pin', _sha(PB.INVESTMENT_COST_RESULTS['path'])
        == PB.INVESTMENT_COST_RESULTS['sha256'], PB.INVESTMENT_COST_RESULTS['sha256'],
        _sha(PB.INVESTMENT_COST_RESULTS['path']))
    _ck('(e) cost workbook sha256 == PB.COST_FILE pin', _sha(PB.COST_FILE['path']) == PB.COST_FILE['sha256'],
        PB.COST_FILE['sha256'], _sha(PB.COST_FILE['path']))
    costs = PB.unit_costs_from_w2(w2)
    years = tuple(sorted(costs))
    _ck('(e) lattice years (S55 rule, W2 costs) == SRP1-3x3 years', list(years) == sorted(p_sp.years),
        sorted(p_sp.years), list(years))
    lattice = PB.Lattice(years, costs)
    # the instance's own production-read cost tables / parameters == SRP1.json's == the standalone production read
    sd, s1 = p_sp.shared_ess_data, p_1.shared_ess_data
    ess = C.build_production_ess_data(years, p_sp.discount_factor)
    same_tables = all(sd.cost_investment[k][c][y] == s1.cost_investment[k][c][y] == ess.cost_investment[k][c][y]
                      for k in ('power', 'energy') for c in range(len(ess.prob_market_scenarios)) for y in years)
    same_probs = list(sd.prob_market_scenarios) == list(s1.prob_market_scenarios) == list(ess.prob_market_scenarios)
    prm = {nm: (getattr(sd.params, nm), getattr(s1.params, nm), getattr(ess.params, nm))
           for nm in ('budget', 'max_capacity', 'min_energy_to_power_ratio', 'max_energy_to_power_ratio')}
    _ck('(e) investment-cost tables: SRP1-3x3 read == SRP1.json read == standalone production read (bitwise)',
        same_tables and same_probs, True, {'tables': same_tables, 'probabilities': same_probs})
    _ck('(e) budget / E cap / E-P ratio bounds: SRP1-3x3 == SRP1.json == standalone == lattice constants',
        all(a == b == c for a, b, c in prm.values()) and prm['budget'][0] == lattice.budget
        and prm['max_capacity'][0] == PB.E_MAX_MWH, {'budget': lattice.budget, 'E_max': PB.E_MAX_MWH,
                                                     'ratio': [2.0, 4.0]}, prm)
    unit = {}
    for y in years:
        disc = 1.0 / (1.0 + sd.discount_factor) ** (y - years[0])
        cp = disc * sum(pr * sd.cost_investment['power'][c][y] for c, pr in enumerate(sd.prob_market_scenarios))
        ce = disc * sum(pr * sd.cost_investment['energy'][c][y] for c, pr in enumerate(sd.prob_market_scenarios))
        unit[str(y)] = {'power_from_instance_read': cp, 'energy_from_instance_read': ce,
                        'power_lattice': lattice.c_p[y], 'energy_lattice': lattice.c_e[y],
                        'abs_diff_power': abs(cp - lattice.c_p[y]), 'abs_diff_energy': abs(ce - lattice.c_e[y])}
    _ck('(e) discounted expected unit costs from the SRP1-3x3 read == lattice unit costs (<= 1e-6)',
        all(v['abs_diff_power'] <= TOL_EUR and v['abs_diff_energy'] <= TOL_EUR for v in unit.values()), '<= 1e-6',
        {y: [v['abs_diff_power'], v['abs_diff_energy']] for y, v in unit.items()})
    domain = lattice.domain()
    adm = [z for z in domain if lattice.has_storage(z)]
    adm_set = set(adm)
    by_year = Counter(str(lattice.years[z[-1]]) for z in adm)
    _ck('(e) admissible plans excluding x = 0 == 1,704', len(adm) == EXPECTED_DOMAIN, EXPECTED_DOMAIN, len(adm))
    _ck('(e) split by year == 149 / 522 / 1,033', dict(sorted(by_year.items())) == EXPECTED_DOMAIN_SPLIT,
        EXPECTED_DOMAIN_SPLIT, dict(sorted(by_year.items())))
    # the committed S55 plan list: same labels, I(x) within 1e-6
    s55_rows = {}
    with open(S55_CSV_REL) as handle:
        hdr = handle.readline().strip().split(',')
        for ln in handle:
            f = dict(zip(hdr, ln.strip().split(',')))
            s55_rows[f['label']] = float(f['I_eur'])
    mine = {lattice.label(z): lattice.investment_cost(z) for z in adm}
    max_d = max(abs(mine[k] - s55_rows[k]) for k in mine if k in s55_rows)
    _ck('(e) label set == committed S55 admissible_plans.csv', set(mine) == set(s55_rows), len(s55_rows),
        {'n_mine': len(mine), 'only_mine': sorted(set(mine) - set(s55_rows))[:5],
         'only_s55': sorted(set(s55_rows) - set(mine))[:5]})
    _ck('(e) I(x) == committed S55 admissible_plans.csv (<= 1e-6)', max_d <= TOL_EUR, '<= 1e-6', max_d)
    # full grid: lattice reasons and the production master check (as S55 (a) + (b))
    check = C.production_check_fn(ess, lattice.nodes, years)
    per_node_grid = [(zp, ze) for zp in range(C.GRID_MAX + 1) for ze in range(C.GRID_MAX + 1)]
    grid_l, grid_m, n_grid, disagree = set(), set(), 0, []
    for yi, year in enumerate(years):
        for combo in product(per_node_grid, repeat=len(lattice.nodes)):
            z = tuple(v for pair in combo for v in pair) + (yi,)
            if not lattice.has_storage(z):
                continue
            n_grid += 1
            why = lattice.reasons(z)
            ok_m, why_m = check([(zp, ze, year) for zp, ze in combo])
            if not why:
                grid_l.add(z)
            if ok_m:
                grid_m.add(z)
            if ok_m != (not why) and len(disagree) < 20:
                disagree.append({'z': list(z), 'lattice': why, 'master': why_m})
    _ck('(e)(a) grid reason-free points == domain() minus x = 0', grid_l == adm_set, 0,
        {'grid_only': len(grid_l - adm_set), 'domain_only': len(adm_set - grid_l)})
    _ck('(e)(b) production master check feasible set == lattice admissible set', grid_m == adm_set, 0,
        {'master_only': len(grid_m - adm_set), 'lattice_only': len(adm_set - grid_m), 'first': disagree})
    _ck('(e) no admissible point at the grid edge', all(max(z[:-1]) < C.GRID_MAX for z in grid_l | grid_m), True, None)
    # production master investment_cost expression on THIS instance (BUILD ONLY): x0, the reference unit, and the
    # cheapest / most expensive admissible plan per year
    master = ESSD._build_master_problem(sd)
    nodes_m = list(sd.active_distribution_network_nodes)
    years_m = list(sd.years)
    targets = [('x0', lattice.x0()), ('ref_unit_n7_0.25_1_2025', REF_UNIT_Z)]
    for yi in range(len(years)):
        cand = sorted((lattice.investment_cost(z), z) for z in adm if z[-1] == yi)
        targets += [(f'cheapest_{years[yi]}', cand[0][1]), (f'dearest_{years[yi]}', cand[-1][1])]
    mchk = []
    for name, z in targets:
        for e in master.energy_storages:
            for y in master.years:
                master.es_s_investment[e, y].set_value(0.0)
                master.es_e_investment[e, y].set_value(0.0)
        if lattice.has_storage(z):
            year = years[z[-1]]
            for n, (zp, ze) in zip(lattice.nodes, lattice.node_pairs(z)):
                e = nodes_m.index(n)
                master.es_s_investment[e, years_m.index(year)].set_value(zp * PB.GRANULE_P_MVA)
                master.es_e_investment[e, years_m.index(year)].set_value(ze * PB.GRANULE_E_MWH)
        im = float(pe.value(master.investment_cost))
        il = lattice.investment_cost(z) if lattice.has_storage(z) else 0.0
        mchk.append({'name': name, 'label': lattice.label(z) if lattice.has_storage(z) else 'x0',
                     'I_lattice': il, 'I_master_expression': im, 'abs_diff': abs(im - il)})
    _ck('(e) production master investment_cost expression (SRP1-3x3, build only) == lattice I(x) (<= 1e-6)',
        all(m['abs_diff'] <= TOL_EUR for m in mchk), '<= 1e-6', [m['abs_diff'] for m in mchk])
    with open(W89_RECORD_REL) as handle:
        i_ref = json.load(handle)['facts']['investment_cost']['n7_4h_e1']['I_x_eur_master_expression']
    _ck('(e) reference unit I(x) == W89 instance record master value (<= 1e-6)',
        abs(mchk[1]['I_master_expression'] - i_ref) <= TOL_EUR, i_ref, mchk[1]['I_master_expression'])
    with open(S55_JSON_REL) as handle:
        s55 = json.load(handle)
    res.update({'rule': ('S55 primary rule by import: p515_s47_phase_b_record.Lattice(years, unit_costs_from_w2(W2)), '
                         'admissible = domain() minus x = 0; cross-checked on the full grid against the production '
                         'master check fed from the production-read workbook (p515_s55_count_admissible_plans)'),
                'COUNT_ADMISSIBLE_EXCLUDING_X0': len(adm), 'split_by_year': dict(sorted(by_year.items())),
                's55_reference': {'path': S55_JSON_REL, 'sha256': _sha(S55_JSON_REL),
                                  'count': s55['COUNT_ADMISSIBLE_EXCLUDING_X0'], 'split_by_year': s55['split_by_year'],
                                  'csv': S55_CSV_REL, 'csv_sha256': _sha(S55_CSV_REL)},
                'admissible_label_list_sha256': hashlib.sha256('\n'.join(sorted(mine)).encode()).hexdigest(),
                'unit_costs': unit, 'n_grid_points_classified': n_grid, 'master_expression_checks': mchk,
                'params': {k: list(v) for k, v in prm.items()},
                'guards': {'s55_module': {'counts': dict(C.GUARD.counts), 'verify_0': C.GUARD.verify(0)},
                           'pb_parent': {'counts': dict(PB.PARENT_GUARD.counts), 'verify_0': PB.PARENT_GUARD.verify(0)},
                           's55_pickle_counts': dict(C.PICKLE_COUNTS)}})
    return res


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    started = datetime.now(timezone.utc).isoformat()
    others = _other_copy_running()
    if others:
        _log(f'REFUSE: another copy of {SCRIPT} is running: {others}')
        return 2
    if not os.path.isdir(OUT_DIR):
        _log(f'REFUSE: {OUT_REL} does not exist; the launch command creates it (mkdir) to hold launch.log')
        return 2
    present = set(os.listdir(OUT_DIR))
    if not present <= ALLOWED_PREEXISTING:
        _log(f'REFUSE: {OUT_REL} holds {sorted(present - ALLOWED_PREEXISTING)}')
        return 2
    sp_case_path = os.path.join(OUT_DIR, SP_CASE_NAME)
    ev_case_path = os.path.join(OUT_DIR, EV_CASE_NAME)
    prof_path = os.path.join(OUT_DIR, EV_PROFILES_NAME)
    srp1_abs = os.path.join(REPO, CASE_SRP1_REL)

    inputs = {k: _input_record(p) for k, p in {
        'case_srp1': CASE_SRP1_REL, 'w89_case': W89_CASE_REL, 'w89_instance_record': W89_RECORD_REL,
        's55_json': S55_JSON_REL, 's55_csv': S55_CSV_REL,
        'market_data': os.path.join('data', 'SRP1', 'MarketData', 'SRP1_market_data.xlsx'),
        'ess_workbook': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
        'ess_params': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
        'planning_params': os.path.join('data', 'SRP1', 'SRP1_params.json'),
        'op_case9': os.path.join('data', 'SRP1', 'case9', 'case9_operational_data.xlsx'),
        'op_case33_1': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_operational_data.xlsx'),
        'op_case33_2': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_operational_data.xlsx'),
        'op_case33_3': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_operational_data.xlsx'),
        'srp_module': 'shared_resources_planning.py', 'network_module': 'network.py',
        'network_data_module': 'network_data.py', 'mch_module': 'model_construction_helpers.py',
        'definitions_module': 'definitions.py', 'essd_module': 'shared_energy_storage_data.py',
        'reader_module': 'p515_s44_scale_measurement.py', 'harness_module': 'p515_s44_campaign_harness.py',
        'w22_module': 'p515_s47_market_spreads.py', 'audit_A_module': 'p515_s56_expected_profiles.py',
        'ev_injection_module': 'p515_s56_ev_injection.py', 's55_module': 'p515_s55_count_admissible_plans.py',
        'pb_module': 'p515_s47_phase_b_record.py', 'guard_module': 'p513_solve_profile_guard.py',
        'w89_script': 'p515_s53_w89_3x3_campaign.py', 'this_script': SCRIPT}.items()}

    # ---------------- 1. derive the SP case (the W89 route) ----------------
    with open(W89_RECORD_REL) as handle:
        w89 = json.load(handle)
    _ck('W89 committed case sha256 == its instance_record', inputs['w89_case']['sha256'] == w89['case_sha256'],
        w89['case_sha256'], inputs['w89_case']['sha256'])
    w89_case, _ws, w89_changes = S.derive_case(w89['derive_base'], dict(w89['derive_overrides']))
    w89_text = json.dumps(w89_case, indent='\t')
    with open(W89_CASE_REL) as handle:
        w89_on_disk = handle.read()
    _ck('route reproduction: derive_case(W89 base, W89 overrides) + json.dumps(indent=tab) == committed W89 case bytes',
        w89_text == w89_on_disk and w89_changes == w89['changes_vs_source'], w89['case_sha256'],
        hashlib.sha256(w89_text.encode()).hexdigest())
    case, spec, changes = S.derive_case(DERIVE_BASE, dict(DERIVE_OVERRIDES))
    text = json.dumps(case, indent='\t')
    with open(srp1_abs) as handle:
        src = json.load(handle)
    exp_changes = ([{'key': 'NumMarketScenarios', 'from': 1, 'to': 3},
                    {'key': f'TransmissionNetwork[{src["TransmissionNetwork"]["name"]}].num_operation_scenarios',
                     'from': 1, 'to': 3}]
                   + [{'key': f'DistributionNetworks[{dn["name"]}].num_operation_scenarios', 'from': 1, 'to': 3}
                      for dn in src['DistributionNetworks']])
    _ck('SP case: changes_vs_source == the five scenario counts 1 -> 3 (nothing else)', changes == exp_changes,
        exp_changes, changes)
    _ck('SP case: every other item == SRP1.json (parsed, scenario counts removed)', strip_counts(case) ==
        strip_counts(src), True, strip_counts(case) == strip_counts(src))
    _ck('SP case: Years unchanged (2025/2030/2035 x 5)', case['Years'] == src['Years'] == {'2025': 5, '2030': 5,
                                                                                            '2035': 5},
        {'2025': 5, '2030': 5, '2035': 5}, case['Years'])
    with open(sp_case_path, 'x') as handle:
        handle.write(text)
    sp_sha = _sha(sp_case_path)
    _log(f'wrote {_rel(sp_case_path)} sha256 {sp_sha}')

    reads = {}
    # ---------------- 2. production reads: SP, SRP1 ----------------
    p_sp, snap_sp, reads['sp'] = production_read(sp_case_path, 'sp')
    post_sp = AA.snapshot(p_sp)
    sp_checksum = reads['sp']['scenario_metadata']['combined_scenario_checksum']
    _log(f'SP read: combined scenario checksum {sp_checksum} ({reads["sp"]["wall_s"]:.1f} s)')
    p_1, snap_1, reads['srp1'] = production_read(srp1_abs, 'srp1')
    post_1 = AA.snapshot(p_1)
    _ck('SRP1.json production read: combined scenario checksum == canonical',
        reads['srp1']['scenario_metadata']['combined_scenario_checksum'] == SRP1_CANONICAL_CHECKSUM,
        SRP1_CANONICAL_CHECKSUM, reads['srp1']['scenario_metadata']['combined_scenario_checksum'])
    _ck('SP checksum differs from SRP1 canonical', sp_checksum != SRP1_CANONICAL_CHECKSUM, '!=', sp_checksum)
    enum_same = all([k for k, *_ in EVI.array_refs(p)] == [k for k, *_ in AA._array_refs(p)] for p in (p_sp, p_1))
    _ck('p515_s56_ev_injection.array_refs == audit A _array_refs (keys, order; SP and SRP1 reads)', enum_same, True,
        enum_same)

    # ---------------- gate (a): prefix property ----------------
    prefix_hook = gate_prefix(snap_sp, snap_1, 'checksum point (entry of _compute_scenario_metadata)')
    prefix_post = gate_prefix(post_sp, post_1, 'after read_planning_problem')
    for pr in (prefix_hook, prefix_post):
        _ck(f'(a) prefix property bitwise, every array, every year -- {pr["point"]}', pr['holds'],
            {'n_equal': pr['n_arrays'], 'n_by_year': pr['n_by_year']},
            {'n_equal': pr['n_equal'], 'n_equal_by_year': pr['n_equal_by_year'], 'mismatches': pr['mismatches_first'][:3]})
    prefix_5x5 = {'exists': False,
                  'searched': ('every *.json <= 5 MB under data/ (working tree, tracked and untracked) carrying '
                               'NumMarketScenarios and Years: no case with Years 2025/2030/2035 and more than one '
                               'scenario exists (multi-scenario cases are on {2025} only or on the five-block grid '
                               '2025/2028/2031/2034/2037)'),
                  'consequence': 'the 5 x 5 first-rows comparison was not made on this grid'}

    # ---------------- gate (b): non-scenario object identity ----------------
    ident_sp_srp1 = AA.walk_compare(p_sp, p_1)
    _ck('(b) SRP1-3x3 vs SRP1.json: every non-scenario item identical (audit A recursive walk; declared exclusions only)',
        ident_sp_srp1['n_diffs'] == 0 and not ident_sp_srp1['uncompared_types'], {'n_diffs': 0, 'uncompared': {}},
        {k: ident_sp_srp1[k] for k in ('n_diffs', 'diffs_first', 'uncompared_types', 'n_leaves_compared')})

    # ---------------- gate (c): blocks and probabilities ----------------
    blocks = gate_blocks_and_probabilities(p_sp)
    _ck('(c) years == 2025/2030/2035 x 5, 4 days', sorted(p_sp.years) == EXPECTED_YEARS
        and all(v == 5 for v in p_sp.years.values()) and len(p_sp.days) == 4,
        {'years': EXPECTED_YEARS, 'days': 4}, {'years': blocks['years'], 'days': blocks['days']})
    _ck('(c) 12 (year, day) blocks per network (TSO and every DSO)', blocks['n_year_day_blocks'] == 12
        and all(v == 12 for v in blocks['blocks_per_holder'].values()), 12,
        {'n': blocks['n_year_day_blocks'], 'per_holder': blocks['blocks_per_holder']})
    ebc = blocks['expected_block_counts']
    _ck('(c) block counts tso 12 / dso 36 / network 48 / esso 3, 51 solves per cycle (= SRP1)',
        (ebc['tso_blocks'], ebc['dso_blocks'], ebc['network_blocks'], ebc['esso_models'], blocks['solves_per_cycle'])
        == (12, 36, 48, 3, 51), [12, 36, 48, 3, 51], [ebc['tso_blocks'], ebc['dso_blocks'], ebc['network_blocks'],
                                                      ebc['esso_models'], blocks['solves_per_cycle']])
    _ck('(c) market probabilities [1/3] x 3 every year; operation and market [1/3] x 3 every network block',
        all(v == [1 / 3] * 3 for v in blocks['market_probabilities'].values())
        and not blocks['blocks_operation_prob_not_third'] and not blocks['blocks_market_prob_not_third'],
        '[1/3]*3', {'market': blocks['market_probabilities'], 'op_bad': blocks['blocks_operation_prob_not_third'],
                    'mk_bad': blocks['blocks_market_prob_not_third']})
    _ck('(c) _scenario_probability(s_m, s_o) == 1/9 for all 9 pairs on every network block',
        blocks['n_pairs_not_one_ninth'] == 0 and blocks['n_pairs_checked'] == 48 * 9, {'pairs': 432, 'value': 1 / 9},
        {'pairs': blocks['n_pairs_checked'], 'not_1/9': blocks['n_pairs_not_one_ninth'],
         'sum_min': blocks['pair_probability_sum_min'], 'sum_max': blocks['pair_probability_sum_max']})

    # ---------------- means (audit A definition) ----------------
    keys = list(snap_sp)
    _ck('same array key set at the checksum point and after the read (SP)', set(post_sp) == set(snap_sp), True,
        set(post_sp) == set(snap_sp))
    means = AA.compute_means(snap_sp)
    joint_diff = AA.joint_mean_check(p_sp, snap_sp, means)
    _ck('joint 9-pair mean (pi_m * pi_o) == factorized mean (<= 1e-12 x largest |mean|)',
        joint_diff <= 1e-12 * max(1.0, max(float(np.max(np.abs(m))) for m in means.values())), '<= 1e-12 rel',
        joint_diff)
    means_post = AA.compute_means(post_sp)
    changed_post = [AA.key_str(k) for k in keys if not np.array_equal(snap_sp[k]['A'], post_sp[k]['A'])]
    _ck('SP: the only arrays production changes after the checksum point are the TSO ADN loads pd / qd',
        all(k.startswith('TSO|') and '|load|' in k and (k.endswith('|pd') or k.endswith('|qd')) for k in changed_post)
        and len(changed_post) > 0, 'TSO|...|load|i|pd/qd only', changed_post[:12])

    # ---------------- gate (d): premium floor ----------------
    floor = gate_premium_floor(p_sp, means)
    _ck('(d) mean energy price > 0 in every hour of every network block (no premium floor needed)',
        floor['n_nonpositive'] == 0 and not floor['premium_floor_needed'], '> 0',
        {'min': floor['min_hourly_mean_price'], 'argmin': floor['argmin'], 'n_nonpositive': floor['n_nonpositive']})
    _ck('(d) production pibar minimum == audit A means minimum (bitwise)', floor['min_hourly_mean_price']
        == floor['min_from_audit_A_means'], floor['min_from_audit_A_means'], floor['min_hourly_mean_price'])

    # ---------------- 3. EV instance ----------------
    a5_keys = [AA.key_str(k) for k in keys if EVI.is_a5(k)]
    a5_mean_vs_const = max(float(np.max(np.abs(means[k] - EVI.production_a5_constant((1, 24))))) for k in keys
                           if EVI.is_a5(k))
    a5_mean_bitwise = sum(1 for k in keys if EVI.is_a5(k) and _bitwise(means[k], EVI.production_a5_constant((1, 24))))
    post_expected = {AA.key_str(k): means_post[k] for k in keys
                     if not EVI.is_a5(k) and not _bitwise(means_post[k], means[k])}
    _ck('EV: post-read expected differs from the injected mean only at TSO ADN pd / qd (production zeroing)',
        set(post_expected) == {k for k in changed_post} and all(not np.any(v) for v in post_expected.values()),
        sorted(changed_post)[:6], sorted(post_expected)[:6])
    ev_case = json.loads(json.dumps(case))
    ev_changes = [{'key': 'NumMarketScenarios', 'from': ev_case['NumMarketScenarios'], 'to': 1}]
    ev_case['NumMarketScenarios'] = 1
    ev_changes.append({'key': f'TransmissionNetwork[{ev_case["TransmissionNetwork"]["name"]}].num_operation_scenarios',
                       'from': ev_case['TransmissionNetwork']['num_operation_scenarios'], 'to': 1})
    ev_case['TransmissionNetwork']['num_operation_scenarios'] = 1
    for dn in ev_case['DistributionNetworks']:
        ev_changes.append({'key': f'DistributionNetworks[{dn["name"]}].num_operation_scenarios',
                           'from': dn['num_operation_scenarios'], 'to': 1})
        dn['num_operation_scenarios'] = 1
    with open(ev_case_path, 'x') as handle:
        handle.write(json.dumps(ev_case, indent='\t'))
    ev_case_equals_srp1 = ev_case == src
    _ck('EV case (counts 3 -> 1) parses equal to SRP1.json (expected: its distinction is the injection alone)',
        ev_case_equals_srp1, True, ev_case_equals_srp1)
    profiles = {
        'schema': EVI.SCHEMA, 'instance_label': EV_LABEL,
        'source_instance': {'label': SP_LABEL, 'case_path': _rel(sp_case_path), 'case_sha256': sp_sha,
                            'scenario_checksum': sp_checksum},
        'case_file': _rel(ev_case_path),
        'formula': AA.FORMULA,
        'a5': ('reactive-flexibility arrays (flex_q_up, flex_q_down) NOT injected: production\'s constant '
               'np.ones((1, 24)) * EQUALITY_TOLERANCE is kept (S56_DESIGN_NOTE_v1 D2 "A5 keeps production\'s '
               'constant"); listed under a5_keys'),
        'a2_a3': 'plain probability-weighted means of the arrays as stored (audit A, "as built")',
        'key_format': {'market': 'market|<energy|flex>|<year>|<day>',
                       'load': '<TSO|DSO<node>>|<year>|<day>|load|<index in network.loads>|<pd|qd|flex_p_up|flex_p_down>',
                       'gen': '<TSO|DSO<node>>|<year>|<day>|gen|<index in network.generators>|<gen_type>|<pg|qg>'},
        'n_arrays': sum(1 for k in keys if not EVI.is_a5(k)), 'n_a5_keys': len(a5_keys),
        'arrays': {AA.key_str(k): means[k].ravel().tolist() for k in keys if not EVI.is_a5(k)},
        'post_read_expected_note': ('after read_planning_problem production replaces the TSO ADN loads\' pd / qd by '
                                    'np.zeros (shared_resources_planning._add_adn_node_to_transmission_network); '
                                    'their expected post-read value = the mean of the SP read\'s post-read arrays'),
        'post_read_expected': {k: v.ravel().tolist() for k, v in post_expected.items()},
        'a5_keys': a5_keys,
        'ev_scenario_checksum': None,
        'ev_scenario_checksum_note': 'computed by the production read with injection; recorded in ev_instance_record.json',
    }
    with open(prof_path, 'x') as handle:
        GRIO.dump(profiles, handle, indent=0)
    prof = EVI.load_profiles(prof_path)
    rt = set(prof['arrays']) == {AA.key_str(k) for k in keys if not EVI.is_a5(k)} and all(
        _bitwise(prof['arrays'][AA.key_str(k)], means[k]) for k in keys if not EVI.is_a5(k))
    _ck('profiles file round trip bitwise (every injected array)', rt, True, rt)

    # EV read through the production reader + the injection hook
    p_ev, snap_ev, reads['ev'] = production_read(ev_case_path, 'ev', profiles=prof)
    ev_checksum = reads['ev']['scenario_metadata']['combined_scenario_checksum']
    _log(f'EV read: combined scenario checksum {ev_checksum}')
    n_inj_exp = len(keys) - len(a5_keys)
    _ck('EV read: hook injected every non-A5 array once and kept every A5 array',
        reads['ev']['injection'] == [{'n_injected': n_inj_exp, 'n_a5_kept': len(a5_keys)}],
        [{'n_injected': n_inj_exp, 'n_a5_kept': len(a5_keys)}], reads['ev']['injection'])
    _ck('EV read: array key list == SP key list', list(snap_ev) == keys, len(keys), len(snap_ev))
    hook_bitwise = all(_bitwise(snap_ev[k]['A'], means[k]) for k in keys if not EVI.is_a5(k))
    a5_const = all(_bitwise(snap_ev[k]['A'], EVI.production_a5_constant((1, 24))) for k in keys if EVI.is_a5(k))
    _ck('EV read, checksum point: every injected array == computed mean (bitwise)', hook_bitwise, True, hook_bitwise)
    _ck('EV read, checksum point: every A5 array == production constant ones((1,24)) * EQUALITY_TOLERANCE (bitwise)',
        a5_const, True, a5_const)
    post_ev = AA.snapshot(p_ev)
    post_bitwise = all(_bitwise(post_ev[k]['A'], means_post[k]) for k in keys if not EVI.is_a5(k))
    _ck('EV read, after the read: every injected array == mean of the SP post-read arrays (bitwise)', post_bitwise,
        True, post_bitwise)
    _ck('EV checksum differs from SP and from SRP1 canonical',
        ev_checksum not in (sp_checksum, SRP1_CANONICAL_CHECKSUM), '!=', ev_checksum)
    pt = reads['ev']['assertion_at_checksum_point']
    _ck('in-child assertion PASSES at the checksum point of the EV read', len(pt) == 1 and pt[0].get('ok') is True,
        'ok', pt[0] if pt else None)
    assertion_tests = {'ev_checksum_point': pt[0] if pt else None}
    try:
        assertion_tests['ev_post_read'] = EVI.assert_ev_arrays(p_ev, prof, point='post_read',
                                                               expected_scenario_checksum=ev_checksum)
        ok_ev = assertion_tests['ev_post_read']['ok']
    except EVI.EvInjectionMismatch as exc:
        assertion_tests['ev_post_read'] = {'ok': False, 'error': str(exc)[:2000], 'summary': exc.summary}
        ok_ev = False
    _ck('in-child assertion PASSES on the EV read (post_read, with the EV checksum)', ok_ev, 'pass',
        assertion_tests['ev_post_read'].get('failures', assertion_tests['ev_post_read'].get('error')))
    ident_ev_sp = AA.walk_compare(p_ev, p_sp)
    _ck('EV vs SRP1-3x3: every non-scenario item identical (recursive walk; declared exclusions only)',
        ident_ev_sp['n_diffs'] == 0 and not ident_ev_sp['uncompared_types'], {'n_diffs': 0},
        {k: ident_ev_sp[k] for k in ('n_diffs', 'diffs_first', 'uncompared_types', 'n_leaves_compared')})

    def expect_raise(name, planning, **kw):
        try:
            s = EVI.assert_ev_arrays(planning, prof, point='post_read', **kw)
            assertion_tests[name] = {'raised': False, 'summary': s}
            return False
        except EVI.EvInjectionMismatch as exc:
            assertion_tests[name] = {'raised': True, 'error': str(exc)[:2000], 'summary': exc.summary}
            return True
    r1 = expect_raise('srp1_plain_read', p_1)
    s1 = assertion_tests['srp1_plain_read'].get('summary') or {}
    _ck('in-child assertion RAISES on a plain SRP1.json read (array values, not dimensions)',
        r1 and s1.get('n_mismatches', 0) > 0 and s1.get('scenario_dimensions', {}).get('num_market_scenarios') == 1,
        'raise with n_mismatches > 0', {'raised': r1, 'n_mismatches': s1.get('n_mismatches'),
                                        'failures': s1.get('failures')})
    # the EV case read WITHOUT the injection hook = what a harness child would run if the hook were not installed
    p_evp, _snap_evp, reads['ev_without_injection'] = production_read(ev_case_path, 'ev_without_injection')
    _ck('EV case read without injection: combined checksum == SRP1 canonical (it IS the published draw 0)',
        reads['ev_without_injection']['scenario_metadata']['combined_scenario_checksum'] == SRP1_CANONICAL_CHECKSUM,
        SRP1_CANONICAL_CHECKSUM, reads['ev_without_injection']['scenario_metadata']['combined_scenario_checksum'])
    r2 = expect_raise('ev_case_without_injection', p_evp)
    s2 = assertion_tests['ev_case_without_injection'].get('summary') or {}
    _ck('in-child assertion RAISES on the EV case read without injection', r2 and s2.get('n_mismatches', 0) > 0,
        'raise with n_mismatches > 0', {'raised': r2, 'n_mismatches': s2.get('n_mismatches')})
    r2b = expect_raise('ev_read_wrong_checksum', p_ev, expected_scenario_checksum=SRP1_CANONICAL_CHECKSUM)
    _ck('in-child assertion RAISES on the EV read when the expected checksum is the SRP1 one', r2b, 'raise', r2b)
    del p_evp
    r3 = expect_raise('sp_read', p_sp)
    _ck('in-child assertion RAISES on the SRP1-3x3 read', r3, 'raise',
        (assertion_tests['sp_read'].get('summary') or {}).get('failures'))

    # ---------------- 4. spreads, 5. uncertainty summary ----------------
    spr = spreads(p_sp, snap_sp, means, snap_1)
    for h in (4, 2):
        r = spr[f'h{h}']['ratios']
        _log(f'spread h={h}: horizon {spr[f"h{h}"]["horizon"]}; ratios {r}')
    unc = uncertainty_summary(p_sp, snap_sp, means)

    # ---------------- gate (e): domain ----------------
    dom = domain_gate(p_sp, p_1)
    for gname, g in dom['guards'].items():
        if gname.endswith('pickle_counts'):
            _ck('(e) s55 pickle block: 0 load / loads', g == {'load': 0, 'loads': 0}, {'load': 0, 'loads': 0}, g)
        else:
            _ck(f'(e) guard {gname} verify(0) == []', g['verify_0'] == [], [], g)

    aa_fail = AA.GUARD.verify(0)
    _ck('audit A module guard verify(0) == []', aa_fail == [], [], aa_fail)
    guard_fail = GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', guard_fail == [], [], guard_fail)
    guards = {'own': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                      'verify_0': guard_fail},
              'audit_A_module': {'label': AA.GUARD.label, 'counts': dict(AA.GUARD.counts), 'verify_0': aa_fail},
              **dom['guards']}
    git_head = _git(['rev-parse', 'HEAD']).strip()

    # ---------------- records ----------------
    sp_decl = H.validate_derived_instance({
        'instance_label': SP_LABEL, 'case_path': _rel(sp_case_path), 'case_sha256': sp_sha,
        'scenario_checksum': sp_checksum, 'source_case_path': CASE_SRP1_REL,
        'source_case_sha256': inputs['case_srp1']['sha256'], 'changes_vs_source': changes})
    facts = {'scenario_checksum': sp_checksum, 'planning_dimensions': S.planning_dimensions(p_sp),
             'expected_block_counts': ebc, 'declared_solve_profile_per_cycle': blocks['solves_per_cycle'],
             'block_weights_tso': blocks['block_weights_tso'],
             'median_block_weight_production': blocks['median_block_weight_production'],
             'min_hourly_mean_price': floor['min_hourly_mean_price'], 'min_hourly_mean_price_at': floor['argmin'],
             'premium_floor_needed': floor['premium_floor_needed'],
             'investment_cost': {m['name']: {'label': m['label'], 'I_x_eur_master_expression': m['I_master_expression'],
                                             'I_x_eur_lattice': m['I_lattice']} for m in dom['master_expression_checks']},
             'admissible_domain': {'count': dom['COUNT_ADMISSIBLE_EXCLUDING_X0'], 'split_by_year': dom['split_by_year']}}
    failed = [c['check'] for c in CHECKS if not c['ok']]
    sp_record = {
        'schema': 'p515_s56_srp1grid_3x3_instance_record_v1',
        'stage': 'P5.15 S56 input preparation (task E): SRP1-3x3 on the common grid 2025/2030/2035 (zero solves)',
        'utc': datetime.now(timezone.utc).isoformat(), 'git_head': git_head, 'script': SCRIPT,
        'script_sha256': _sha(os.path.abspath(__file__)), 'instance_label': SP_LABEL,
        'case_path': _rel(sp_case_path), 'case_sha256': sp_sha, 'derive_base': DERIVE_BASE,
        'derive_overrides': DERIVE_OVERRIDES, 'derive_description': spec.get('description'),
        'derive_route': ('p515_s44_scale_measurement.derive_case(base, overrides) + json.dumps(case, indent="\\t") -- '
                         'the route of p515_s53_w89_3x3_campaign.derive_instance_text; re-checked: it reproduces the '
                         'committed W89 case bytes'),
        'changes_vs_source': changes, 'source_case': {'path': CASE_SRP1_REL, 'sha256': inputs['case_srp1']['sha256']},
        'derived_instance_declaration': sp_decl,
        'prefix_verification': {'vs_srp1_json_checksum_point': prefix_hook, 'vs_srp1_json_after_read': prefix_post,
                                'vs_5x5_on_this_grid': prefix_5x5},
        'object_walk_vs_srp1_json': {k: ident_sp_srp1[k] for k in ('n_leaves_compared', 'n_diffs', 'diffs_first',
                                                                    'uncompared_types', 'excluded_attribute_hits')},
        'facts': facts, 'all_checks_pass': not failed, 'failed_checks': failed,
        'solve_claim': 'ZERO SOLVES, guard-verified (every armed permitted=() guard verify(0))', 'guards': guards,
    }
    ev_decl_draft = H.validate_derived_instance({
        'instance_label': EV_LABEL, 'case_path': _rel(ev_case_path), 'case_sha256': _sha(ev_case_path),
        'scenario_checksum': ev_checksum, 'source_case_path': _rel(sp_case_path), 'source_case_sha256': sp_sha,
        'changes_vs_source': ev_changes + [{
            'key': 'scenario arrays (all but A5)',
            'from': 'production synthesis at one scenario (= the published draw 0)',
            'to': f'probability-weighted means of {SP_LABEL}, injected at shared_resources_planning.'
                  f'_compute_scenario_metadata from {_rel(prof_path)} (sha256 {_sha(prof_path)})'}]})
    ev_record = {
        'schema': 'p515_s56_srp1grid_3x3_ev_instance_record_v1',
        'stage': 'P5.15 S56 input preparation (task E): the EV twin of SRP1-3x3 (zero solves)',
        'utc': datetime.now(timezone.utc).isoformat(), 'git_head': git_head, 'script': SCRIPT,
        'instance_label': EV_LABEL, 'case_path': _rel(ev_case_path), 'case_sha256': _sha(ev_case_path),
        'profiles_path': _rel(prof_path), 'profiles_sha256': _sha(prof_path), 'scenario_checksum': ev_checksum,
        'source_instance': {'label': SP_LABEL, 'case_path': _rel(sp_case_path), 'case_sha256': sp_sha,
                            'scenario_checksum': sp_checksum},
        'case_changes_vs_source_instance': ev_changes,
        'case_parses_equal_to_srp1_json': ev_case_equals_srp1,
        'warning': ('the EV case file IS SRP1.json\'s content: read without the injection hook it yields the published '
                    'draw 0 (checksum 5a02b77c...). The means reach a run only through '
                    'p515_s56_ev_injection.ScenarioPointHook(profiles=load_profiles(<profiles>)); a child must call '
                    'p515_s56_ev_injection.assert_ev_arrays(planning, profiles, point="post_read", '
                    'expected_scenario_checksum=<this record\'s scenario_checksum>) after its read'),
        'formula': AA.FORMULA, 'a5': profiles['a5'], 'a2_a3': profiles['a2_a3'],
        'a5_mean_vs_production_constant': {'max_abs_diff_pu': a5_mean_vs_const, 'n_a5_arrays': len(a5_keys),
                                           'n_a5_means_bitwise_equal_constant': a5_mean_bitwise},
        'loading_route': ('production reader p515_s44_scale_measurement.read_planning_from_derived_case on the EV case '
                          '+ p515_s56_ev_injection.ScenarioPointHook (inject) at shared_resources_planning.'
                          '_compute_scenario_metadata'),
        'assertion': {'module': 'p515_s56_ev_injection.py', 'module_sha256': _sha('p515_s56_ev_injection.py'),
                      'function': 'assert_ev_arrays(planning, profiles, point="post_read", '
                                  'expected_scenario_checksum=None)', 'tests': assertion_tests},
        'derived_instance_declaration_draft': ev_decl_draft,
        'declaration_note': ('format-validated with p515_s44_campaign_harness.validate_derived_instance only; the '
                             'harness child refuses a one-scenario derived instance (S56_DESIGN_NOTE_v1, corrections) '
                             'and has no injection plumbing -- a harness change, not made here'),
        'object_walk_vs_source_instance': {k: ident_ev_sp[k] for k in ('n_leaves_compared', 'n_diffs', 'diffs_first',
                                                                        'uncompared_types', 'excluded_attribute_hits')},
        'all_checks_pass': not failed, 'failed_checks': failed,
        'solve_claim': 'ZERO SOLVES, guard-verified', 'guards': guards,
    }
    payload = {
        'schema': 'p515_s56_build_instances_v1', 'script': SCRIPT, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - STARTED,
        'interpreter': sys.executable, 'git_head': git_head,
        'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) armed for the whole run, verify(0))',
        'instances': {'sp': {'label': SP_LABEL, 'case': _rel(sp_case_path), 'case_sha256': sp_sha,
                             'scenario_checksum': sp_checksum},
                      'ev': {'label': EV_LABEL, 'case': _rel(ev_case_path), 'case_sha256': _sha(ev_case_path),
                             'profiles': _rel(prof_path), 'profiles_sha256': _sha(prof_path),
                             'scenario_checksum': ev_checksum}},
        'gates': {'a_prefix': {'checksum_point': prefix_hook, 'after_read': prefix_post, 'vs_5x5': prefix_5x5},
                  'b_object_walk_sp_vs_srp1': ident_sp_srp1, 'c_blocks_probabilities': blocks, 'd_premium_floor': floor,
                  'e_domain': dom},
        'ev': {'assertion_tests': assertion_tests, 'object_walk_ev_vs_sp': ident_ev_sp,
               'a5_mean_vs_production_constant': ev_record['a5_mean_vs_production_constant'],
               'post_read_expected_keys': sorted(post_expected), 'joint_mean_max_abs_diff': joint_diff},
        'prediction6_spreads': spr, 'scenario_uncertainty': unc, 'reads': reads, 'inputs': inputs,
        'checks': CHECKS, 'failed_checks': failed, 'all_checks_ok': not failed, 'guards': guards,
    }
    with open(os.path.join(OUT_DIR, SP_RECORD_NAME), 'x') as handle:
        GRIO.dump(sp_record, handle, indent=1, default=GRIO.json_default_item)
    with open(os.path.join(OUT_DIR, EV_RECORD_NAME), 'x') as handle:
        GRIO.dump(ev_record, handle, indent=1, default=GRIO.json_default_item)
    with open(os.path.join(OUT_DIR, RESULTS_NAME), 'x') as handle:
        GRIO.dump(payload, handle, indent=1, default=GRIO.json_default_item)
    outputs = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for f in sorted(files):
            pth = os.path.join(root, f)
            rel = os.path.relpath(pth, OUT_DIR)
            if rel in ALLOWED_PREEXISTING:
                continue
            outputs[rel] = _sha(pth)
    manifest = {'outputs': outputs, 'inputs': inputs,
                'note': 'launch.log is written while the script runs; its sha256 is recorded in the Worker report'}
    with open(os.path.join(OUT_DIR, MANIFEST_NAME), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1)
    _log(f'SP {SP_LABEL}: case sha256 {sp_sha}, checksum {sp_checksum}')
    _log(f'EV {EV_LABEL}: case sha256 {_sha(ev_case_path)}, profiles sha256 {_sha(prof_path)}, checksum {ev_checksum}')
    _log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {failed}')
    return 0 if not failed else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
