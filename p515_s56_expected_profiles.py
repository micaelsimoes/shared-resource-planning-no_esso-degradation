"""P5.15 S56 -- input audit, part A: expected-profile vs stochastic planning on the 3 x 3 instance (ZERO SOLVES).

What this script does (Planner task, S56 part A; questions 1, 2, 3 and 5 -- question 4 is code reading, in the report):
  1. Reads, through the PRODUCTION reader the 3 x 3 harness uses
     (`p515_s44_scale_measurement.read_planning_from_derived_case`: production `SharedResourcesPlanning` on data/SRP1
     with the case file given relative to it; plots / results / logs of the read redirected into this output
     directory), four instances:
        s53_3x3   data/SRP1/Results/P515S53/w89_3x3/instance/SRP1__s53_3x3.json   (the 3 x 3 instance; sha and
                  scenario checksum asserted against instance_record.json)
        srp1      data/SRP1/SRP1.json                                              (1 x 1, years 2025/2030/2035)
        srp1_5x3y data/SRP1/SRP1_5x3y_1x1.json                                     (1 x 1 on the 3 x 3 grid, P5.16)
        ev        <out>/instance/SRP1__s56_3x3_ev.json  (written here: the 3 x 3 case with every scenario count = 1)
  2. SCENARIO ARRAYS are captured at ONE declared point of the production read: a pass-through wrapper on
     `shared_resources_planning._compute_scenario_metadata` (called exactly once per read, after the TSO and the
     DSOs are read and BEFORE the ADN loads are zeroed and the shared ESS added). The captured arrays are exactly
     the ones production's scenario checksum hashes: per (year, day) market energy and flexibility prices
     (`cost_energy_p`, `cost_flex`, shape n_market x 24); per network block, per load `pd`, `qd`,
     `flexibility.active_power.upward/downward`, `flexibility.reactive_power.upward/downward`; per generator `pg`,
     `qg` (shape n_operation x 24). The post-read arrays (after production's ADN-load zeroing) are captured too.
  3. EXPECTED PROFILES (operational definition, recorded in the output):
        market array A[s_m, t] (per year, day):    Abar[t] = sum_{s_m} pi_m[year][s_m] * A[s_m, t]
        operation array A[s_o, t] (per network, year, day, device):  Abar[t] = sum_{s_o} pi_o[s_o] * A[s_o, t]
     with pi_m = planning.prob_market_scenarios[year] and pi_o = network.prob_operation_scenarios, as read by
     production; evaluated as numpy `np.sum(w[:, None] * A, axis=0)` in float64. Equivalently, the joint mean over
     the 9 (s_m, s_o) pairs with weight pi_m * pi_o (production's `_scenario_probability`), checked numerically.
     The plain probability-weighted mean of each array AS STORED -- no power-factor, Q/P-ratio, capacity or sign
     re-derivation (the ambiguities are listed in the output, not resolved).
  4. Writes the new 1 x 1 variant: the case JSON (only the five scenario counts changed) and the profiles file
     (every mean, JSON floats, exact round trip). PRODUCTION HAS NO READER FOR EXPLICIT SCENARIO PROFILES (it
     synthesizes them from the season base profiles at read time; see the report), so the variant is loaded by the
     production reader with an INJECTION HOOK -- the same wrapper as in (2), in 'inject' mode: it writes the means
     from the profiles FILE in place into the arrays production has just generated (1 x 24 each), then lets the
     production checksum run. This is a candidate route for the Planner, not a decision.
  5. Verifies: the EV read's arrays == the means (bitwise and <= 1e-12 relative), at the hook and after the read;
     every non-scenario item of the EV planning object == the 3 x 3 one (a full recursive walk over the object
     graph, with the scenario fields and the output directories as the ONLY declared exclusions); the same walk
     3 x 3 vs srp1_5x3y (informational); single-scenario vs ensemble-mean differences per array family (Q2).
  6. Q5: the production search lattice `p515_s47_phase_b_record.Lattice` on the 3 x 3 years with unit costs from the
     production-read cost workbook, its membership cross-checked on the full grid against the production master
     check `shared_resources_planning._check_candidate_first_stage_feasibility` (reusing
     `p515_s55_count_admissible_plans` by import); I(x) cross-checked against the production master problem's
     `investment_cost` expression (BUILD ONLY); salvage terminal discount and remaining calendar-life fractions
     from the production functions.

Guards: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0, plus the
guards of the imported s55 / Phase B modules.

Output: NEW directory data/SRP1/Results/P515S56/input_audit_A/ (the launch command creates it to hold launch.log;
the script refuses if it holds anything else, or if another copy is running).

Launch (attached, both streams captured):
    mkdir -p data/SRP1/Results/P515S56 && mkdir data/SRP1/Results/P515S56/input_audit_A
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_expected_profiles.py \\
        > data/SRP1/Results/P515S56/input_audit_A/launch.log 2>&1
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 input audit A (never solves)').install()

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict, deque  # noqa: E402
from contextlib import contextmanager, redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import shared_resources_planning as SRP  # noqa: E402
import shared_energy_storage_data as ESSD  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402  (production read of a derived case; constants only at import)

SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'input_audit_A')
# DRY RUN ONLY: `--dry-run-out <absolute dir outside the repository>` redirects every output there (used once, into
# the session scratchpad, to test the script before the write-once launch). The real launch passes no argument.
if '--dry-run-out' in sys.argv:
    OUT_REL = os.path.abspath(sys.argv[sys.argv.index('--dry-run-out') + 1])
    if OUT_REL.startswith(REPO + os.sep):
        raise SystemExit('--dry-run-out must be outside the repository')
OUT_DIR = os.path.join(REPO, OUT_REL)
ALLOWED_PREEXISTING = {'launch.log'}
INSTANCE_DIR = os.path.join(OUT_DIR, 'instance')
EV_CASE_NAME = 'SRP1__s56_3x3_ev.json'
EV_PROFILES_NAME = 'SRP1__s56_3x3_ev_profiles.json'
RESULTS_NAME = 'input_audit_A.json'
MANIFEST_NAME = 'manifest_sha256.json'

CASE_3X3_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'SRP1__s53_3x3.json')
RECORD_3X3_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance', 'instance_record.json')
CASE_SRP1_REL = os.path.join('data', 'SRP1', 'SRP1.json')
CASE_5X3Y_REL = os.path.join('data', 'SRP1', 'SRP1_5x3y_1x1.json')
SRP1_CANONICAL_CHECKSUM = '5a02b77ccbbbbbb869de92958a3851d095624711abc2dbfc0157466064410358'

REL_TOL = 1e-12
TOL_EUR = 1e-6
YEARS_3X3 = (2025, 2028, 2031, 2034, 2037)

FORMULA = {
    'market': ('Abar[year][day][t] = sum_{s_m} pi_m[year][s_m] * A[year][day][s_m, t], A in {cost_energy_p (EUR/MWh), '
               'cost_flex (EUR/MWh)}, pi_m = planning.prob_market_scenarios[year] (production '
               '_read_market_data_from_file: [1/NumMarketScenarios] * NumMarketScenarios)'),
    'operation': ('Abar[net][year][day][device][t] = sum_{s_o} pi_o[s_o] * A[s_o, t], A in {load.pd, load.qd, '
                  'load.flexibility.active_power.upward, .downward, load.flexibility.reactive_power.upward, .downward, '
                  'generator.pg, generator.qg} (p.u. on the network baseMVA, as stored), pi_o = '
                  'network.prob_operation_scenarios (production _read_network_data: [1/num_oper_scenarios] * n)'),
    'joint_equivalence': ('equal to the joint mean over the 9 (s_m, s_o) pairs with weight pi_m * pi_o '
                          '(production _scenario_probability) because each array depends on one dimension only and '
                          'the other dimension\'s weights sum to 1; checked numerically'),
    'evaluation': 'numpy float64, np.sum(w[:, None] * A, axis=0), w the production probability vector',
    'capture_point': ('arrays as stored at the entry of shared_resources_planning._compute_scenario_metadata (the '
                      'production scenario-checksum point: TSO and DSOs read, ADN loads not yet zeroed, shared ESS '
                      'not yet added); the post-read arrays are compared separately'),
    'what_it_is_not': ('no re-derivation of power factor / Q-P ratio, no capacity clipping, no abs() of flexibility '
                       'bounds (production applies abs() at model build), no re-sampling'),
}

CHECKS = []
STARTED = time.time()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _ck(name, ok, detail=None):
    CHECKS.append({'check': name, 'ok': bool(ok), 'detail': detail})
    _log(f'{"OK  " if ok else "FAIL"} {name}' + (f' -- {str(detail)[:600]}' if detail is not None and not ok else ''))


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


# ======================================================================================================================
#  scenario-array capture / injection at the production checksum point
# ======================================================================================================================
def _holders(planning):
    out = [('TSO', planning.transmission_network)]
    out += [(f'DSO{node}', planning.distribution_networks[node]) for node in sorted(planning.distribution_networks)]
    return out


def _array_refs(planning):
    """Ordered list of (key, kind, ndarray reference, weight vector, baseMVA). key is a tuple of str/int.
    kind: 'market' or 'operation'. The reference is the live production array (for in-place injection)."""
    refs = []
    for year in sorted(planning.years):
        w = np.asarray(planning.prob_market_scenarios[year], dtype=float)
        for day in planning.days:
            refs.append((('market', 'energy', year, day), 'market', planning.cost_energy_p[year][day], w, None))
            refs.append((('market', 'flex', year, day), 'market', planning.cost_flex[year][day], w, None))
    for hname, holder in _holders(planning):
        for year in sorted(holder.years):
            for day in holder.days:
                net = holder.network[year][day]
                w = np.asarray(net.prob_operation_scenarios, dtype=float)
                for i, load in enumerate(net.loads):
                    base = (hname, year, day, 'load', i)
                    refs.append((base + ('pd',), 'operation', load.pd, w, net.baseMVA))
                    refs.append((base + ('qd',), 'operation', load.qd, w, net.baseMVA))
                    refs.append((base + ('flex_p_up',), 'operation', load.flexibility.active_power.upward, w, net.baseMVA))
                    refs.append((base + ('flex_p_down',), 'operation', load.flexibility.active_power.downward, w, net.baseMVA))
                    refs.append((base + ('flex_q_up',), 'operation', load.flexibility.reactive_power.upward, w, net.baseMVA))
                    refs.append((base + ('flex_q_down',), 'operation', load.flexibility.reactive_power.downward, w, net.baseMVA))
                for i, gen in enumerate(net.generators):
                    base = (hname, year, day, 'gen', i, int(gen.gen_type))
                    refs.append((base + ('pg',), 'operation', gen.pg, w, net.baseMVA))
                    refs.append((base + ('qg',), 'operation', gen.qg, w, net.baseMVA))
    return refs


def key_str(key):
    return '|'.join(str(k) for k in key)


def family_of(key):
    if key[0] == 'market':
        return f'market.{key[1]}'
    net_kind = 'TSO' if key[0] == 'TSO' else 'DSO'
    if key[3] == 'load':
        return f'{net_kind}.load.{key[-1]}'
    return f'{net_kind}.gen{key[5]}.{key[-1]}'


def snapshot(planning):
    snap = {}
    for key, kind, arr, w, base in _array_refs(planning):
        a = np.array(arr, dtype=float, copy=True)
        if a.ndim != 2:
            raise RuntimeError(f'{key}: expected a 2-D scenario array, got shape {a.shape}')
        snap[key] = {'kind': kind, 'A': a, 'w': w.copy(), 'baseMVA': base}
    return snap


class ScenarioHook:
    """Pass-through wrapper on shared_resources_planning._compute_scenario_metadata (the production checksum point).
    mode 'capture': snapshot the arrays, call the original. mode 'inject': write `self.means` IN PLACE into the
    live 1 x 24 arrays (all references -- e.g. network.cost_energy_p, which aliases planning.cost_energy_p[y][d] --
    therefore see the means), snapshot, call the original."""

    def __init__(self):
        self.original = SRP._compute_scenario_metadata
        self.mode = None
        self.means = None
        self.calls = 0
        self.snapshots = []
        self.inject_counts = []

    def install(self):
        hook = self

        def wrapped(planning_problem):
            hook.calls += 1
            if hook.mode == 'inject':
                n = 0
                for key, kind, arr, w, base in _array_refs(planning_problem):
                    m = hook.means[key]
                    if arr.shape != m.shape:
                        raise RuntimeError(f'inject {key}: shape {arr.shape} != mean shape {m.shape}')
                    arr[...] = m
                    n += 1
                if n != len(hook.means):
                    raise RuntimeError(f'inject: {n} arrays in the read, {len(hook.means)} means in the file')
                hook.inject_counts.append(n)
            hook.snapshots.append(snapshot(planning_problem))
            return hook.original(planning_problem)
        SRP._compute_scenario_metadata = wrapped
        return self

    def uninstall(self):
        SRP._compute_scenario_metadata = self.original


class _Stages:
    @contextmanager
    def stage(self, _name):
        yield


def production_read(case_rel, label, hook, mode, means=None):
    """Production read of `case_rel` (relative to the repo) via the 3 x 3 harness's reader, with the hook armed."""
    read_dir = os.path.join(OUT_DIR, 'reads', label)
    if os.path.exists(read_dir):
        raise RuntimeError(f'refusing to overwrite {read_dir}')
    os.makedirs(read_dir)
    hook.mode, hook.means = mode, means
    n_before = hook.calls
    stream = io.StringIO()
    t0 = time.time()
    with redirect_stdout(stream):
        planning = S.read_planning_from_derived_case({'derived_case': {'path': case_rel}}, read_dir, _Stages())
    text = stream.getvalue()
    with open(os.path.join(read_dir, 'production_stdout.log'), 'w') as handle:
        handle.write(text)
    hook.mode, hook.means = None, None
    if hook.calls - n_before != 1:
        raise RuntimeError(f'{label}: checksum hook called {hook.calls - n_before} times (expected 1)')
    return planning, hook.snapshots[-1], {'wall_s': time.time() - t0, 'read_dir': os.path.relpath(read_dir, REPO),
                                          'scenario_metadata': dict(planning.scenario_metadata)}


# ======================================================================================================================
#  means and comparisons
# ======================================================================================================================
def compute_means(snap):
    means = {}
    for key, rec in snap.items():
        A, w = rec['A'], rec['w']
        if A.shape[0] != w.shape[0]:
            raise RuntimeError(f'{key}: {A.shape[0]} scenario rows, {w.shape[0]} probabilities')
        means[key] = np.sum(w[:, None] * A, axis=0).reshape(1, -1)
    return means


def joint_mean_check(planning, snap, means):
    """The joint mean over the (s_m, s_o) pairs with production's weight _scenario_probability(network, s_m, s_o)
    = pi_m * pi_o, the dimension an array does not depend on summed out explicitly. Returns the max abs difference
    to the factorized mean. Market arrays use the TSO block's weights; network arrays use their own block's."""
    worst = 0.0
    for key, rec in snap.items():
        A = rec['A']
        joint = np.zeros(A.shape[1])
        if key[0] == 'market':
            _k, _t, year, day = key
            net = planning.transmission_network.network[year][day]
            for s_m in range(A.shape[0]):
                for s_o in range(len(net.prob_operation_scenarios)):
                    joint = joint + SRP._scenario_probability(net, s_m, s_o) * A[s_m]
        else:
            hname, year, day = key[0], key[1], key[2]
            holder = planning.transmission_network if hname == 'TSO' else planning.distribution_networks[int(hname[3:])]
            net = holder.network[year][day]
            for s_m in range(len(net.prob_market_scenarios)):
                for s_o in range(A.shape[0]):
                    joint = joint + SRP._scenario_probability(net, s_m, s_o) * A[s_o]
        worst = max(worst, float(np.max(np.abs(joint - means[key].ravel()))))
    return worst


def rel_err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = np.abs(a - b)
    scale = np.maximum(np.abs(b), np.abs(a))
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(d == 0.0, 0.0, d / np.where(scale == 0.0, np.inf, scale))
    return float(np.max(r)) if r.size else 0.0


def diff_stats(pairs):
    """pairs: list of (key, x (1 x T), m (1 x T), baseMVA or None). Per family: max/mean abs (stored units and MW /
    MVAr for network arrays), aggregate relative sum|x - m| / sum|m|, max elementwise relative over the elements with
    |m| >= 1e-6 * max|m| of the family (elementwise relative is undefined where the mean is ~0, e.g. PV at night)."""
    fam = defaultdict(list)
    for key, x, m, base in pairs:
        fam[family_of(key)].append((x.ravel(), m.ravel(), base))
    out = {}
    for f, items in sorted(fam.items()):
        X = np.concatenate([x for x, _m, _b in items])
        M = np.concatenate([m for _x, m, _b in items])
        B = np.concatenate([np.full(x.shape, b if b is not None else 1.0) for x, _m, b in items])
        D = np.abs(X - M)
        mmax = float(np.max(np.abs(M))) if M.size else 0.0
        mask = np.abs(M) >= 1e-6 * mmax if mmax > 0 else np.zeros(M.shape, bool)
        out[f] = {
            'n_arrays': len(items), 'n_elements': int(X.size),
            'units_stored': 'EUR/MWh' if f.startswith('market') else 'p.u. (network baseMVA)',
            'max_abs_stored': float(np.max(D)), 'mean_abs_stored': float(np.mean(D)),
            'max_abs_physical': float(np.max(D * B)), 'mean_abs_physical': float(np.mean(D * B)),
            'units_physical': 'EUR/MWh' if f.startswith('market') else 'MW / MVAr',
            'sum_abs_mean_physical': float(np.sum(np.abs(M) * B)),
            'aggregate_relative_sum_abs_diff_over_sum_abs_mean': (float(np.sum(D) / np.sum(np.abs(M)))
                                                                  if np.sum(np.abs(M)) > 0 else None),
            'max_elementwise_relative_where_mean_ge_1e-6_max': (float(np.max(D[mask] / np.abs(M[mask])))
                                                               if mask.any() else None),
            'n_elements_relative_defined': int(mask.sum()),
            'signed_mean_diff_physical_x_minus_mean': float(np.mean((X - M) * B)),
            'n_negative_values_in_x': int(np.sum(X < 0)), 'n_negative_values_in_mean': int(np.sum(M < 0)),
        }
    return out


def ensemble_ranges(snap):
    """Per family: how far the scenarios spread around their mean (context for Q2), and sign/capacity facts."""
    fam = defaultdict(lambda: {'n_neg': 0, 'n_el': 0, 'max_spread_phys': 0.0, 'sum_abs_dev': 0.0, 'sum_abs_mean': 0.0})
    for key, rec in snap.items():
        A, w, base = rec['A'], rec['w'], rec['baseMVA'] or 1.0
        m = np.sum(w[:, None] * A, axis=0)
        f = fam[family_of(key)]
        f['n_neg'] += int(np.sum(A < 0))
        f['n_el'] += int(A.size)
        f['max_spread_phys'] = max(f['max_spread_phys'], float(np.max((A.max(axis=0) - A.min(axis=0)) * base)))
        f['sum_abs_dev'] += float(np.sum(np.abs(A - m[None, :]) * w[:, None]) * base)
        f['sum_abs_mean'] += float(np.sum(np.abs(m)) * base)
    return {k: {**v, 'expected_abs_dev_over_abs_mean': (v['sum_abs_dev'] / v['sum_abs_mean']
                                                        if v['sum_abs_mean'] > 0 else None)}
            for k, v in sorted(fam.items())}


def res_capacity_facts(planning, snap):
    """RES availability vs installed capacity (pmax) per generator, scenarios and mean; and on/off-hour structure
    (production builds the RES rows only where available apparent power > EQUALITY_TOLERANCE)."""
    from definitions import GEN_CURTAILLABLE_TYPES
    from model_construction_helpers import EQUALITY_TOLERANCE
    over_cap_scen, over_cap_mean, n_curt = 0, 0, 0
    max_ratio_scen, max_ratio_mean = 0.0, 0.0
    hours_on_any, hours_on_all, hours_on_mean = 0, 0, 0
    for hname, holder in _holders(planning):
        for year in sorted(holder.years):
            for day in holder.days:
                net = holder.network[year][day]
                for i, gen in enumerate(net.generators):
                    if gen.gen_type not in GEN_CURTAILLABLE_TYPES:
                        continue
                    n_curt += 1
                    rec = snap[(hname, year, day, 'gen', i, int(gen.gen_type), 'pg')]
                    A = rec['A']
                    m = np.sum(rec['w'][:, None] * A, axis=0)
                    pmax = gen.pmax
                    if pmax > 0:
                        max_ratio_scen = max(max_ratio_scen, float(np.max(A) / pmax))
                        max_ratio_mean = max(max_ratio_mean, float(np.max(m) / pmax))
                    over_cap_scen += int(np.sum(A > pmax + 1e-12))
                    over_cap_mean += int(np.sum(m > pmax + 1e-12))
                    on = A > EQUALITY_TOLERANCE
                    hours_on_any += int(np.sum(on.any(axis=0)))
                    hours_on_all += int(np.sum(on.all(axis=0)))
                    hours_on_mean += int(np.sum(m > EQUALITY_TOLERANCE))
    return {'n_curtaillable_generator_blocks': n_curt,
            'max_pg_over_pmax_any_scenario': max_ratio_scen, 'max_pg_over_pmax_mean': max_ratio_mean,
            'n_elements_pg_above_pmax_scenarios': over_cap_scen, 'n_elements_pg_above_pmax_mean': over_cap_mean,
            'generator_hours_available_in_any_scenario': hours_on_any,
            'generator_hours_available_in_all_scenarios': hours_on_all,
            'generator_hours_available_in_mean': hours_on_mean,
            'note': ('production skips the RES capability / power-factor rows and fixes pg to 0 where available '
                     'apparent power <= EQUALITY_TOLERANCE (model_construction_helpers.renewable_generation_is_unavailable); '
                     'the mean is "available" wherever ANY scenario is')}


def load_pf_facts(snap):
    """Q/P ratio of the mean vs the mean of the per-scenario Q/P ratios (DSO and TSO loads with pd > 0)."""
    max_dev, n = 0.0, 0
    for key, rec in snap.items():
        if key[0] == 'market' or key[3] != 'load' or key[-1] != 'pd':
            continue
        q = snap[key[:-1] + ('qd',)]
        P, Q, w = rec['A'], q['A'], rec['w']
        if np.any(P <= 0):
            continue
        ratio_of_mean = np.sum(w[:, None] * Q, axis=0) / np.sum(w[:, None] * P, axis=0)
        mean_of_ratio = np.sum(w[:, None] * (Q / P), axis=0)
        max_dev = max(max_dev, float(np.max(np.abs(ratio_of_mean - mean_of_ratio))))
        n += 1
    return {'n_loads_compared': n, 'max_abs_diff_Q_over_P_of_mean_vs_mean_of_Q_over_P': max_dev}


def flex_bound_facts(snap):
    """Sign of the flexibility-bound arrays: production uses abs(value) as the bound (pc_flex_up_bounds /
    pc_flex_down_bounds), so for a sign-changing array mean(abs) != abs(mean)."""
    out = {}
    for fld in ('flex_p_up', 'flex_p_down', 'flex_q_up', 'flex_q_down'):
        n_neg, n_el, max_gap = 0, 0, 0.0
        for key, rec in snap.items():
            if key[0] == 'market' or key[-1] != fld:
                continue
            A, w = rec['A'], rec['w']
            n_neg += int(np.sum(A < 0))
            n_el += int(A.size)
            gap = np.abs(np.sum(w[:, None] * np.abs(A), axis=0) - np.abs(np.sum(w[:, None] * A, axis=0)))
            max_gap = max(max_gap, float(np.max(gap)))
        out[fld] = {'n_negative_elements': n_neg, 'n_elements': n_el,
                    'max_abs_mean_of_abs_minus_abs_of_mean_pu': max_gap}
    return out


# ======================================================================================================================
#  recursive non-scenario identity walk
# ======================================================================================================================
# The ONLY declared exclusions, by (class name, attribute): the scenario arrays and probability vectors, the scenario
# counts, the scenario checksum, the case filename / name, and the read's output directories.
EXCLUDED_ATTRS = {
    'SharedResourcesPlanning': {'cost_energy_p', 'cost_flex', 'prob_market_scenarios', 'num_market_scenarios',
                                'scenario_metadata', 'filename', 'results_dir', 'diagrams_dir', 'logs_dir'},
    'NetworkData': {'cost_energy_p', 'cost_flex', 'num_oper_scenarios', 'results_dir', 'diagrams_dir', 'logs_dir'},
    'Network': {'cost_energy_p', 'cost_flex', 'prob_market_scenarios', 'prob_operation_scenarios',
                'num_oper_scenarios', 'results_dir', 'diagrams_dir', 'logs_dir'},
    'Load': {'pd', 'qd'},
    'Generator': {'pg', 'qg'},
    'LoadFlexibilityDirection': {'upward', 'downward'},
    # shared_ess_data.cost_energy_p is the planning's market dict (an alias); its prob_market_scenarios is the
    # INVESTMENT-COST scenario vector from the ESS workbook and is NOT excluded.
    'SharedEnergyStorageData': {'cost_energy_p', 'results_dir', 'logs_dir'},
}
BASIC = (type(None), bool, int, float, str, bytes, complex)


def walk_compare(a, b, max_report=200):
    diffs, uncompared = [], Counter()
    excluded_hits = Counter()
    n_leaves = 0
    visited = set()
    stack = [(a, b, 'planning')]
    while stack:
        x, y, path = stack.pop()
        if type(x) is not type(y):
            diffs.append({'path': path, 'what': f'type {type(x).__name__} != {type(y).__name__}'})
            continue
        if isinstance(x, BASIC) or isinstance(x, np.generic):
            n_leaves += 1
            same = (x == y) or (isinstance(x, float) and x != x and y != y)
            if not same:
                diffs.append({'path': path, 'what': f'{x!r} != {y!r}'})
            continue
        pair = (id(x), id(y))
        if pair in visited:
            continue
        visited.add(pair)
        if isinstance(x, np.ndarray):
            n_leaves += 1
            if x.shape != y.shape or x.dtype != y.dtype or not np.array_equal(x, y, equal_nan=x.dtype.kind == 'f'):
                diffs.append({'path': path, 'what': f'ndarray differs (shape {x.shape} vs {y.shape})'})
            continue
        if isinstance(x, dict):
            if set(x.keys()) != set(y.keys()):
                diffs.append({'path': path, 'what': f'dict keys differ: only a {sorted(map(str, set(x) - set(y)))[:10]}, '
                                                    f'only b {sorted(map(str, set(y) - set(x)))[:10]}'})
            for k in x:
                if k in y:
                    stack.append((x[k], y[k], f'{path}[{k!r}]'))
            continue
        if isinstance(x, (list, tuple, deque)):
            if len(x) != len(y):
                diffs.append({'path': path, 'what': f'length {len(x)} != {len(y)}'})
            for i, (u, v) in enumerate(zip(x, y)):
                stack.append((u, v, f'{path}[{i}]'))
            continue
        if isinstance(x, (set, frozenset)):
            n_leaves += 1
            if x != y:
                diffs.append({'path': path, 'what': 'set differs'})
            continue
        if hasattr(x, '__dict__'):
            cls = type(x).__name__
            excl = EXCLUDED_ATTRS.get(cls, set())
            kx, ky = set(vars(x)), set(vars(y))
            if kx != ky:
                diffs.append({'path': path, 'what': f'attribute sets differ: {sorted(kx ^ ky)}'})
            for k in sorted(kx & ky):
                if k in excl:
                    excluded_hits[f'{cls}.{k}'] += 1
                    continue
                stack.append((getattr(x, k), getattr(y, k), f'{path}.{k}'))
            continue
        uncompared[type(x).__name__] += 1
    return {'n_leaves_compared': n_leaves, 'n_diffs': len(diffs), 'diffs_first': diffs[:max_report],
            'uncompared_types': dict(uncompared), 'excluded_attribute_hits': dict(excluded_hits)}


# ======================================================================================================================
#  Q5: admissible domain on the 3 x 3 grid
# ======================================================================================================================
def q5_domain(planning_3x3, planning_srp1):
    import pyomo.environ as pe
    import p515_s55_count_admissible_plans as C   # installs its own permitted=() guard and a pickle block at import
    PB = C.PB
    res = {}
    df = float(planning_3x3.discount_factor)
    ess = C.build_production_ess_data(YEARS_3X3, df)
    res['cost_workbook'] = {'path': C.PB.COST_FILE['path'], 'sha256': _sha(C.PB.COST_FILE['path']),
                            'pinned_sha256_PB': C.PB.COST_FILE['sha256'],
                            'investment_cost_scenario_probabilities': list(ess.prob_market_scenarios)}
    _ck('Q5 cost workbook sha256 == PB.COST_FILE pin', res['cost_workbook']['sha256'] == C.PB.COST_FILE['sha256'])
    unit = {}
    costs = {}
    for y in YEARS_3X3:
        disc = 1.0 / (1.0 + df) ** (y - YEARS_3X3[0])
        raw_p = {c: ess.cost_investment['power'][c][y] for c in range(len(ess.prob_market_scenarios))}
        raw_e = {c: ess.cost_investment['energy'][c][y] for c in range(len(ess.prob_market_scenarios))}
        cp = disc * sum(p * raw_p[c] for c, p in enumerate(ess.prob_market_scenarios))
        ce = disc * sum(p * raw_e[c] for c, p in enumerate(ess.prob_market_scenarios))
        unit[y] = {'discount': disc, 'power_eur_per_mva_by_cost_scenario': raw_p,
                   'energy_eur_per_mwh_by_cost_scenario': raw_e,
                   'power_eur_per_mva_discounted_expected': cp, 'energy_eur_per_mwh_discounted_expected': ce}
        costs[y] = {'power': cp, 'energy': ce}
    res['unit_costs'] = unit
    # identity with the production read of the 3 x 3 instance itself
    e3 = planning_3x3.shared_ess_data
    same = all(e3.cost_investment[k][c][y] == ess.cost_investment[k][c][y]
               for k in ('power', 'energy') for c in range(len(ess.prob_market_scenarios)) for y in YEARS_3X3)
    _ck('Q5 cost tables read standalone == those in the 3 x 3 planning read', same and
        list(e3.prob_market_scenarios) == list(ess.prob_market_scenarios))
    lattice = PB.Lattice(YEARS_3X3, costs)
    domain = lattice.domain()
    adm = [z for z in domain if lattice.has_storage(z)]
    adm_set = set(adm)
    _ck('Q5 domain()[0] is x = 0 and appears once', domain[0] == lattice.x0() and domain.count(lattice.x0()) == 1)
    # full-grid cross-check against the production master check (as s55 (a)+(b))
    check = C.production_check_fn(ess, lattice.nodes, YEARS_3X3)
    per_node_grid = [(zp, ze) for zp in range(C.GRID_MAX + 1) for ze in range(C.GRID_MAX + 1)]
    grid_l, grid_m, n_grid, disagree = set(), set(), 0, []
    for yi, year in enumerate(YEARS_3X3):
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
    _ck('Q5 (a) grid reason-free points == domain() minus x = 0', grid_l == adm_set,
        {'grid_only': len(grid_l - adm_set), 'domain_only': len(adm_set - grid_l)})
    _ck('Q5 (b) production master check feasible set == lattice admissible set', grid_m == adm_set,
        {'master_only': len(grid_m - adm_set), 'lattice_only': len(adm_set - grid_m), 'first': disagree})
    _ck('Q5 no admissible point at the grid edge', all(max(z[:-1]) < C.GRID_MAX for z in grid_l | grid_m))
    max_cf = 0.0
    rows = []
    for z in adm:
        year = lattice.years[z[-1]]
        I = lattice.investment_cost(z)
        cf = C.closed_form_I(ess, [(zp, ze, year) for zp, ze in lattice.node_pairs(z)])
        max_cf = max(max_cf, abs(cf - I))
        rows.append((year, sum(1 for p, e in lattice.node_pairs(z) if p or e), I))
    _ck('Q5 lattice I(x) == closed form from production-read workbook (<= 1e-6)', max_cf <= TOL_EUR, max_cf)
    # production master problem investment_cost expression, BUILD ONLY (no solve), one cheapest and one most
    # expensive admissible plan per year, against the lattice
    master = ESSD._build_master_problem(planning_3x3.shared_ess_data)
    nodes_m = list(planning_3x3.shared_ess_data.active_distribution_network_nodes)
    years_m = list(planning_3x3.shared_ess_data.years)
    master_checks = []
    for yi, year in enumerate(YEARS_3X3):
        cand = sorted((lattice.investment_cost(z), z) for z in adm if z[-1] == yi)
        for I, z in (cand[0], cand[-1]):
            for e in master.energy_storages:
                for y in master.years:
                    master.es_s_investment[e, y].set_value(0.0)
                    master.es_e_investment[e, y].set_value(0.0)
            for n, (zp, ze) in zip(lattice.nodes, lattice.node_pairs(z)):
                e = nodes_m.index(n)
                master.es_s_investment[e, years_m.index(year)].set_value(zp * PB.GRANULE_P_MVA)
                master.es_e_investment[e, years_m.index(year)].set_value(ze * PB.GRANULE_E_MWH)
            Im = float(pe.value(master.investment_cost))
            master_checks.append({'label': lattice.label(z), 'I_lattice': I, 'I_master_expression': Im,
                                  'abs_diff': abs(Im - I)})
    _ck('Q5 production master investment_cost expression == lattice I(x) (10 plans, <= 1e-6)',
        all(m['abs_diff'] <= TOL_EUR for m in master_checks), master_checks)
    # the 3 x 3 instance record's own production-master value for the node-7 unit 0.25/1 @2025
    with open(RECORD_3X3_REL) as handle:
        rec = json.load(handle)
    i_rec = rec['facts']['investment_cost']['n7_4h_e1']['I_x_eur_master_expression']
    z_ref = (0, 0, 1, 2, 0, 0, 0)
    _ck('Q5 n7 0.25/1 @2025 I(x) == instance_record master expression (<= 1e-6)',
        abs(lattice.investment_cost(z_ref) - i_rec) <= TOL_EUR, [lattice.investment_cost(z_ref), i_rec])
    by_year = Counter(r[0] for r in rows)
    by_n = Counter(r[1] for r in rows)
    by_year_n = Counter((r[0], r[1]) for r in rows)
    # salvage / end-of-horizon via production functions, on both reads
    salvage = {}
    for label, pl in (('s53_3x3', planning_3x3), ('srp1', planning_srp1)):
        sd = pl.shared_ess_data
        yrs = list(sd.years)
        cohort = {}
        for i, y in enumerate(yrs):
            sess = sd.shared_energy_storages[y][0]
            age, rem, frac = ESSD._get_remaining_calendar_life(sd, i, sess)
            cohort[y] = {'age_at_terminal_years': age, 'remaining_life_years': rem,
                         'remaining_life_fraction': frac, 't_cal': sess.t_cal,
                         'tcal_norm_blocks': round(sess.t_cal / sd.years[y]),
                         'expected_energy_unit_cost_eur_per_mwh_undiscounted':
                             ESSD._get_expected_energy_investment_cost(sd, y)}
        salvage[label] = {'years': {str(k): v for k, v in sd.years.items()},
                          'horizon_years_sum': sum(sd.years.values()),
                          'y_end': yrs[0] + sum(sd.years.values()),
                          'terminal_discount_factor': ESSD._get_terminal_discount_factor(sd),
                          'cohorts': {str(k): v for k, v in cohort.items()},
                          'salvage_params': {k: getattr(sd.params.salvage_value, k) for k in (
                              'enabled', 'in_objective', 'energy_recovery_fraction', 'recycling_floor_fraction',
                              'cost_basis', 'health_basis', 'calendar_life_basis')},
                          'available_energy_soh_point': sd.available_energy_soh_point}
    res.update({
        'rule': ('production search lattice p515_s47_phase_b_record.Lattice on years (2025, 2028, 2031, 2034, 2037), '
                 'unit costs = 1.02^-(y-2025) * sum_c w_c c_{c,y} from the production-read SRP1_ESS.xlsx, '
                 'w = investment-cost scenario probabilities; common installation year; granules 0.25 MVA / 0.5 MWh; '
                 'E/P in [2, 4] h; E <= 5 MWh; I(x) <= 1e6 EUR'),
        'COUNT_ADMISSIBLE_EXCLUDING_X0': len(adm), 'split_by_year': {str(k): v for k, v in sorted(by_year.items())},
        'split_by_n_nodes': {str(k): v for k, v in sorted(by_n.items())},
        'split_by_year_and_n_nodes': {f'{k[0]}|{k[1]}': v for k, v in sorted(by_year_n.items())},
        'n_grid_points_classified': n_grid, 'master_expression_checks': master_checks,
        'max_I_admissible_eur': max(r[2] for r in rows), 'min_slack_eur': lattice.budget - max(r[2] for r in rows),
        'srp1_reference_count': {'count': 1704, 'split_by_year': {'2025': 149, '2030': 522, '2035': 1033},
                                 'source': 'data/SRP1/Results/P515S55/admissible_plan_count/admissible_plan_count.json'},
        'salvage_end_of_horizon': salvage,
        'guards': {'s55_module': {'counts': dict(C.GUARD.counts), 'verify_0': C.GUARD.verify(0)},
                   'pb_parent': {'counts': dict(PB.PARENT_GUARD.counts), 'verify_0': PB.PARENT_GUARD.verify(0)},
                   's55_pickle_counts': dict(C.PICKLE_COUNTS)},
    })
    with open(os.path.join(OUT_DIR, 'q5_admissible_plans_3x3.csv'), 'w') as handle:
        handle.write('label,year,n_nodes,I_eur\n')
        for z in sorted(adm, key=lambda q: (q[-1], q)):
            nn = sum(1 for p, e in lattice.node_pairs(z) if p or e)
            handle.write(f'{lattice.label(z)},{lattice.years[z[-1]]},{nn},{lattice.investment_cost(z)!r}\n')
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
    if os.path.exists(OUT_DIR):
        present = set(os.listdir(OUT_DIR))
        if not present <= ALLOWED_PREEXISTING:
            _log(f'REFUSE: {OUT_REL} exists and holds {sorted(present - ALLOWED_PREEXISTING)}')
            return 2
    else:
        _log(f'REFUSE: {OUT_REL} does not exist; the launch command creates it (mkdir) to hold launch.log')
        return 2
    os.makedirs(INSTANCE_DIR)

    inputs = {k: _input_record(p) for k, p in {
        'case_3x3': CASE_3X3_REL, 'instance_record_3x3': RECORD_3X3_REL, 'case_srp1': CASE_SRP1_REL,
        'case_srp1_5x3y_1x1': CASE_5X3Y_REL, 'market_data': os.path.join('data', 'SRP1', 'MarketData', 'SRP1_market_data.xlsx'),
        'ess_workbook': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS.xlsx'),
        'ess_params': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json'),
        'ess_params_5x3y': os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_5x3y_ESS_Params.json'),
        'planning_params': os.path.join('data', 'SRP1', 'SRP1_params.json'),
        'op_case9': os.path.join('data', 'SRP1', 'case9', 'case9_operational_data.xlsx'),
        'op_case33_1': os.path.join('data', 'SRP1', 'case33_1', 'case33_1_operational_data.xlsx'),
        'op_case33_2': os.path.join('data', 'SRP1', 'case33_2', 'case33_2_operational_data.xlsx'),
        'op_case33_3': os.path.join('data', 'SRP1', 'case33_3', 'case33_3_operational_data.xlsx'),
        'srp_module': 'shared_resources_planning.py', 'network_module': 'network.py',
        'network_data_module': 'network_data.py', 'mch_module': 'model_construction_helpers.py',
        'essd_module': 'shared_energy_storage_data.py', 'reader_module': 'p515_s44_scale_measurement.py',
        's55_module': 'p515_s55_count_admissible_plans.py', 'pb_module': 'p515_s47_phase_b_record.py',
        'guard_module': 'p513_solve_profile_guard.py', 'this_script': SCRIPT}.items()}
    with open(RECORD_3X3_REL) as handle:
        record_3x3 = json.load(handle)
    _ck('3 x 3 case sha256 == instance_record case_sha256', inputs['case_3x3']['sha256'] == record_3x3['case_sha256'])

    hook = ScenarioHook().install()
    reads = {}
    try:
        p33, snap33, reads['s53_3x3'] = production_read(CASE_3X3_REL, 's53_3x3', hook, 'capture')
        _ck('3 x 3 production read: combined scenario checksum == instance_record',
            reads['s53_3x3']['scenario_metadata']['combined_scenario_checksum']
            == record_3x3['facts']['scenario_checksum'], reads['s53_3x3']['scenario_metadata'])
        post33 = snapshot(p33)
        p1, snap1, reads['srp1'] = production_read(CASE_SRP1_REL, 'srp1', hook, 'capture')
        _ck('SRP1 production read: combined scenario checksum == canonical',
            reads['srp1']['scenario_metadata']['combined_scenario_checksum'] == SRP1_CANONICAL_CHECKSUM,
            reads['srp1']['scenario_metadata'])
        p5, snap5, reads['srp1_5x3y'] = production_read(CASE_5X3Y_REL, 'srp1_5x3y', hook, 'capture')

        # -------- dimensions and probabilities (Q1) --------
        q1 = {'s53_3x3': S.planning_dimensions(p33), 'srp1': S.planning_dimensions(p1),
              'srp1_5x3y': S.planning_dimensions(p5)}
        probs = {'market': {str(y): list(p33.prob_market_scenarios[y]) for y in sorted(p33.years)},
                 'operation': sorted({(h, tuple(holder.network[y][d].prob_operation_scenarios))
                                      for h, holder in _holders(p33) for y in holder.years for d in holder.days}),
                 'investment_cost_scenarios_ess_workbook': list(p33.shared_ess_data.prob_market_scenarios)}
        probs['operation'] = [[h, list(v)] for h, v in probs['operation']]
        block_weights = {f'{y}|{d}': SRP._get_admm_block_weight(p33.transmission_network, y, d)
                         for y in sorted(p33.years) for d in p33.days}
        _ck('3 x 3 market probabilities all 1/3', all(v == [1 / 3] * 3 for v in probs['market'].values()))
        _ck('3 x 3 operation probabilities all 1/3 on every network block',
            all(v == [1 / 3] * 3 for _h, v in probs['operation']))
        _ck('3 x 3 TSO block weights == instance_record',
            all(abs(block_weights[k] - v) == 0.0 for k, v in record_3x3['facts']['block_weights_tso'].items()))

        # -------- means (Q3) --------
        keys = list(snap33)
        _ck('same array key set at the hook and after the read (3 x 3)', set(post33) == set(snap33))
        means = compute_means(snap33)
        joint_diff = joint_mean_check(p33, snap33, means)
        _ck('joint 9-pair mean (weights _scenario_probability = pi_m * pi_o) == factorized mean (max abs diff <= '
            '1e-12 x the largest |mean|)',
            joint_diff <= 1e-12 * max(1.0, max(float(np.max(np.abs(m))) for m in means.values())), joint_diff)
        means_post = compute_means(post33)
        changed_post = [key_str(k) for k in keys if not np.array_equal(snap33[k]['A'], post33[k]['A'])]
        _ck('3 x 3: the only arrays production changes after the checksum point are the TSO ADN loads pd / qd '
            '(zeroed by _add_adn_node_to_transmission_network)',
            all(k.startswith('TSO|') and '|load|' in k and (k.endswith('|pd') or k.endswith('|qd')) for k in changed_post),
            changed_post[:20])

        # -------- write the EV variant --------
        with open(CASE_3X3_REL) as handle:
            case = json.load(handle)
        ev_case = json.loads(json.dumps(case))
        changes = [{'key': 'NumMarketScenarios', 'from': ev_case['NumMarketScenarios'], 'to': 1}]
        ev_case['NumMarketScenarios'] = 1
        changes.append({'key': f'TransmissionNetwork[{ev_case["TransmissionNetwork"]["name"]}].num_operation_scenarios',
                        'from': ev_case['TransmissionNetwork']['num_operation_scenarios'], 'to': 1})
        ev_case['TransmissionNetwork']['num_operation_scenarios'] = 1
        for dn in ev_case['DistributionNetworks']:
            changes.append({'key': f'DistributionNetworks[{dn["name"]}].num_operation_scenarios',
                            'from': dn['num_operation_scenarios'], 'to': 1})
            dn['num_operation_scenarios'] = 1
        ev_case_path = os.path.join(INSTANCE_DIR, EV_CASE_NAME)
        with open(ev_case_path, 'x') as handle:
            json.dump(ev_case, handle, indent='\t')
            handle.write('\n')
        profiles = {
            'schema': 'p515_s56_expected_profiles_v1',
            'instance_label': 's56_3x3_ev',
            'source_instance': {'case_path': CASE_3X3_REL, 'case_sha256': inputs['case_3x3']['sha256'],
                                'scenario_checksum': reads['s53_3x3']['scenario_metadata']['combined_scenario_checksum']},
            'case_file': os.path.relpath(ev_case_path, REPO),
            'formula': FORMULA,
            'n_arrays': len(means),
            'key_format': {'market': 'market|<energy|flex>|<year>|<day>',
                           'load': '<TSO|DSO<node>>|<year>|<day>|load|<index in network.loads>|<pd|qd|flex_p_up|flex_p_down|flex_q_up|flex_q_down>',
                           'gen': '<TSO|DSO<node>>|<year>|<day>|gen|<index in network.generators>|<gen_type>|<pg|qg>'},
            'arrays': {key_str(k): means[k].ravel().tolist() for k in keys},
            'weights': {key_str(k): snap33[k]['w'].tolist() for k in keys},
        }
        prof_path = os.path.join(INSTANCE_DIR, EV_PROFILES_NAME)
        with open(prof_path, 'x') as handle:
            GRIO.dump(profiles, handle, indent=0)
        # round trip from the FILE
        with open(prof_path) as handle:
            back = json.load(handle)['arrays']
        key_of = {key_str(k): k for k in keys}
        means_file = {key_of[s]: np.asarray(v, dtype=float).reshape(1, -1) for s, v in back.items()}
        _ck('profiles file round trip bitwise (every array)', set(means_file) == set(means)
            and all(np.array_equal(means_file[k], means[k]) for k in keys))

        # -------- read the EV variant back through the production reader + injection hook --------
        ev_rel = os.path.relpath(ev_case_path, REPO)
        pev, snapev, reads['ev'] = production_read(ev_rel, 'ev', hook, 'inject', means=means_file)
        postev = snapshot(pev)
        q3 = {'case_changes_vs_3x3': changes}
        _ck('EV read: hook injected every array exactly once', hook.inject_counts == [len(means)], hook.inject_counts)
        _ck('EV read: array key set == 3 x 3 key set', set(snapev) == set(snap33) and set(postev) == set(post33))
        shapes_ok = all(snapev[k]['A'].shape == (1, snap33[k]['A'].shape[1]) for k in keys)
        _ck('EV read: every scenario array is 1 x 24', shapes_ok)
        _ck('EV read: market probability [1.0] every year, operation probability [1.0] every block',
            all(list(pev.prob_market_scenarios[y]) == [1.0] for y in pev.years)
            and all(list(h.network[y][d].prob_operation_scenarios) == [1.0]
                    for _n, h in _holders(pev) for y in h.years for d in h.days))
        hook_bitwise = all(np.array_equal(snapev[k]['A'], means[k]) for k in keys)
        hook_rel = max(rel_err(snapev[k]['A'], means[k]) for k in keys)
        _ck('EV read, at the checksum point: every array == computed mean (bitwise)', hook_bitwise, hook_rel)
        _ck(f'EV read, at the checksum point: every array == computed mean (<= {REL_TOL} relative)', hook_rel <= REL_TOL,
            hook_rel)
        post_bitwise = all(np.array_equal(postev[k]['A'], means_post[k]) for k in keys)
        post_rel = max(rel_err(postev[k]['A'], means_post[k]) for k in keys)
        _ck('EV read, after the read: every array == mean of the 3 x 3 post-read arrays (bitwise)', post_bitwise, post_rel)
        _ck(f'EV read, after the read: every array == mean of the 3 x 3 post-read arrays (<= {REL_TOL} relative)',
            post_rel <= REL_TOL, post_rel)
        # aliasing: networks see the planning's market arrays
        alias_ok = all(np.array_equal(h.network[y][d].cost_energy_p, means[('market', 'energy', y, d)])
                       and np.array_equal(h.network[y][d].cost_flex, means[('market', 'flex', y, d)])
                       for _n, h in _holders(pev) for y in h.years for d in h.days)
        _ck('EV read: every network block sees the mean market arrays (alias of planning.cost_*)', alias_ok)
        ident = walk_compare(pev, p33)
        _ck('EV vs 3 x 3: every non-scenario item identical (recursive walk; declared exclusions only)',
            ident['n_diffs'] == 0 and not ident['uncompared_types'], {k: ident[k] for k in ('n_diffs', 'diffs_first',
                                                                                             'uncompared_types')})
        q3.update({'identity_walk_ev_vs_3x3': ident, 'excluded_attributes': {k: sorted(v) for k, v in EXCLUDED_ATTRS.items()},
                   'ev_scenario_metadata': reads['ev']['scenario_metadata'],
                   'hook_max_relative_error': hook_rel, 'post_read_max_relative_error': post_rel,
                   'files': {'case': {'path': os.path.relpath(ev_case_path, REPO), 'sha256': _sha(ev_case_path)},
                             'profiles': {'path': os.path.relpath(prof_path, REPO), 'sha256': _sha(prof_path)}},
                   'loading_route': ('production reader (p515_s44_scale_measurement.read_planning_from_derived_case on '
                                     'the EV case) + the S56 injection hook (ScenarioHook, mode inject, means read from '
                                     'the profiles file) at shared_resources_planning._compute_scenario_metadata; '
                                     'production alone cannot load explicit profiles (it synthesizes them)')})

        # -------- Q2: single scenario vs ensemble mean --------
        q2 = {}
        # (a) SRP1 2025 single scenario vs the 3 x 3's scenario index 0 in 2025 (prefix property)
        k2025 = [k for k in keys if (k[2] if k[0] == 'market' else k[1]) == 2025]
        pre_a = sum(1 for k in k2025 if k in snap1 and np.array_equal(snap1[k]['A'][0], snap33[k]['A'][0]))
        _ck('SRP1 2025 single-scenario arrays == 3 x 3 2025 scenario index 0 (bitwise, every array)',
            pre_a == len(k2025) and all(k in snap1 for k in k2025), [pre_a, len(k2025)])
        # (b) SRP1_5x3y_1x1 vs 3 x 3 scenario index 0, every year
        pre_b = sum(1 for k in keys if k in snap5 and np.array_equal(snap5[k]['A'][0], snap33[k]['A'][0]))
        _ck('SRP1_5x3y_1x1 arrays == 3 x 3 scenario index 0 (bitwise, every array, every year)',
            pre_b == len(keys) and set(snap5) == set(snap33), [pre_b, len(keys)])
        q2['prefix_identity'] = {'srp1_2025_vs_3x3_s0_2025': {'n_equal': pre_a, 'n': len(k2025)},
                                 'srp1_5x3y_vs_3x3_s0_all_years': {'n_equal': pre_b, 'n': len(keys)}}
        # (c) single scenario (= 3 x 3 scenario 0 = SRP1_5x3y_1x1) vs ensemble mean, 3 x 3 grid, all years
        q2['scenario0_vs_mean_all_years'] = diff_stats(
            [(k, snap33[k]['A'][0:1], means[k], snap33[k]['baseMVA']) for k in keys])
        q2['scenario0_vs_mean_2025_only'] = diff_stats(
            [(k, snap33[k]['A'][0:1], means[k], snap33[k]['baseMVA']) for k in k2025])
        # (d) other scenario indices vs mean (context: is scenario 0 typical?)
        q2['scenario1_vs_mean_all_years'] = diff_stats(
            [(k, snap33[k]['A'][1:2], means[k], snap33[k]['baseMVA']) for k in keys])
        q2['scenario2_vs_mean_all_years'] = diff_stats(
            [(k, snap33[k]['A'][2:3], means[k], snap33[k]['baseMVA']) for k in keys])
        q2['ensemble_spread'] = ensemble_ranges(snap33)
        # system aggregates per block (MWh/day): load, RES; market mean price
        from definitions import GEN_CURTAILLABLE_TYPES
        agg = {}
        for y in sorted(p33.years):
            for d in p33.days:
                row = {'load_MWh_s0': 0.0, 'load_MWh_mean': 0.0, 'res_MWh_s0': 0.0, 'res_MWh_mean': 0.0}
                for k in keys:
                    if k[0] == 'market' or k[1] != y or k[2] != d:
                        continue
                    b = snap33[k]['baseMVA']
                    if k[3] == 'load' and k[-1] == 'pd' and k[0] != 'TSO':   # every TSO load is an ADN load
                        row['load_MWh_s0'] += float(np.sum(snap33[k]['A'][0]) * b)
                        row['load_MWh_mean'] += float(np.sum(means[k]) * b)
                    if k[3] == 'gen' and k[-1] == 'pg' and k[5] in GEN_CURTAILLABLE_TYPES:
                        row['res_MWh_s0'] += float(np.sum(snap33[k]['A'][0]) * b)
                        row['res_MWh_mean'] += float(np.sum(means[k]) * b)
                ke = ('market', 'energy', y, d)
                row['mean_price_s0'] = float(np.mean(snap33[ke]['A'][0]))
                row['mean_price_mean'] = float(np.mean(means[ke]))
                agg[f'{y}|{d}'] = row
        q2['system_aggregates_per_block'] = agg
        # 4 h energy-price spread (W19 Z4 / W22 definition: mean of the 4 highest minus mean of the 4 lowest hourly
        # prices of the day), per block, for each scenario, for the mean profile, and E_s[spread_s]; horizon value
        # day-weighted with w = num_years * num_days (undiscounted) -- context for the arbitrage value the storage
        # sees: spread(mean) <= E[spread] (the spread is convex in the profile)
        def spread4(v):
            v = np.sort(np.asarray(v, float).ravel())
            return float(np.mean(v[-4:]) - np.mean(v[:4]))
        sp, wsum = {'mean_profile': 0.0, 'expected_over_scenarios': 0.0, 's0': 0.0, 's1': 0.0, 's2': 0.0}, 0.0
        sp_blocks = {}
        for y in sorted(p33.years):
            for d in p33.days:
                ke = ('market', 'energy', y, d)
                A, w_m = snap33[ke]['A'], snap33[ke]['w']
                row = {'mean_profile': spread4(means[ke]),
                       'expected_over_scenarios': float(sum(w_m[s] * spread4(A[s]) for s in range(A.shape[0]))),
                       **{f's{s}': spread4(A[s]) for s in range(A.shape[0])}}
                sp_blocks[f'{y}|{d}'] = row
                wt = p33.years[y] * p33.days[d]
                wsum += wt
                for k in sp:
                    sp[k] += wt * row[k]
        q2['energy_price_4h_spread'] = {'definition': 'mean(4 highest) - mean(4 lowest) hourly EUR/MWh; horizon = '
                                                      'sum_b w_b spread_b / sum_b w_b, w_b = num_years * num_days',
                                        'horizon': {k: v / wsum for k, v in sp.items()}, 'per_block': sp_blocks}
        q2['res_capacity'] = res_capacity_facts(p33, snap33)
        q2['load_q_over_p'] = load_pf_facts(snap33)
        q2['flex_bounds_sign'] = flex_bound_facts(snap33)
        q2['years'] = {'srp1': sorted(p1.years), 's53_3x3': sorted(p33.years),
                       'like_for_like_years': sorted(set(p1.years) & set(p33.years)),
                       'note': ('per (network, year, day) the profile is a draw from the SEASON pool with a seed that '
                                'depends on the year (network_data._read_network_data: derive_random_seed(network seed, '
                                '"realization", year, day); _read_market_data_from_file: derive_random_seed(market seed, '
                                '"selection", <type>, year, day)) times the growth factor (1+g)^(year - first year); '
                                'SRP1 2030 / 2035 have no counterpart on the 3 x 3 grid')}
        # (e) non-scenario identity 3 x 3 vs SRP1_5x3y_1x1 (informational: which other items differ)
        q2['identity_walk_5x3y_vs_3x3'] = walk_compare(p5, p33)

        # -------- Q5 --------
        q5 = q5_domain(p33, p1)
        for gname, g in q5['guards'].items():
            if gname.endswith('pickle_counts'):
                _ck('Q5 s55 pickle block: 0 load / loads', g == {'load': 0, 'loads': 0}, g)
            else:
                _ck(f'Q5 guard {gname} verify(0) == []', g['verify_0'] == [], g)
    finally:
        hook.uninstall()

    guard_fail = GUARD.verify(0)
    _ck('own SolveProfileGuard(permitted=()) verify(0) == []', guard_fail == [], guard_fail)
    payload = {
        'schema': 'p515_s56_input_audit_A_v1', 'script': SCRIPT, 'started_utc': started,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'wall_s': time.time() - STARTED,
        'interpreter': sys.executable, 'git_head': _git(['rev-parse', 'HEAD']).strip(),
        'solve_claim': 'ZERO SOLVES, guard-verified (SolveProfileGuard(permitted=()) armed for the whole run, verify(0))',
        'formula': FORMULA,
        'q1_dimensions': q1, 'q1_probabilities': probs, 'q1_block_weights_tso': block_weights,
        'reads': reads, 'q2': q2, 'q3': q3, 'q5': q5,
        'inputs': inputs, 'checks': CHECKS, 'failed_checks': [c['check'] for c in CHECKS if not c['ok']],
        'all_checks_ok': all(c['ok'] for c in CHECKS),
        'guards': {'own': {'label': GUARD.label, 'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                           'verify_0': guard_fail}},
    }
    out_path = os.path.join(OUT_DIR, RESULTS_NAME)
    with open(out_path, 'x') as handle:
        GRIO.dump(payload, handle, indent=1, default=GRIO.json_default_item)
    outputs = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, OUT_DIR)
            if rel in ALLOWED_PREEXISTING:
                continue
            outputs[rel] = _sha(p)
    manifest = {'outputs': outputs, 'inputs': inputs,
                'note': 'launch.log is written while the script runs; its sha256 is recorded in the Worker report'}
    with open(os.path.join(OUT_DIR, MANIFEST_NAME), 'x') as handle:
        GRIO.dump(manifest, handle, indent=1)
    _log(f'all_checks_ok = {payload["all_checks_ok"]}; failed = {payload["failed_checks"]}')
    return 0 if payload['all_checks_ok'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
