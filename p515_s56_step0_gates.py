"""P5.15 S56 plan v2, validation step 0 -- ZERO-SOLVE GATES (read-only; no file outside the output directory is edited).

Authority: `S56_PLAN_v2.md` "Validation sequence" step 0 (existing authorization); TASKS.md "Step 0" (active).

Items (every number is computed here from production code / production reads / the lattice module, nothing typed):
  1  PROBABILITY AND SCENARIO IDENTITY. For the SRP1-grid 3 x 3 (`instances_v1/SRP1__s56_srp1grid_3x3.json`), its EV
     twin (read with the `p515_s56_ev_injection` hook, as a harness child would) and the five-block 3 x 3
     (`P515S53/w89_3x3/instance/SRP1__s53_3x3.json`): in every (year, day) block, every network (TN and the three DNs)
     carries the same market and operation probability vectors and the same scenario counts (block, holder, array
     rows), and the market price arrays (`cost_energy_p`, `cost_flex`) are bitwise identical across networks and equal
     to the planning-level arrays. The ESS cost-trajectory probabilities are recorded beside (A3: independent).
  2  TERMINAL VALUE (build-only ESSO, `salvage_value.in_objective` switched on IN MEMORY, SoH point asserted 'end'), on
     both grids (2025/2030/2035 from the SRP1-grid read; 2025-2037 from the five-block read): per cohort the
     installation year, end of calendar life, horizon end, whether it expires strictly inside the horizon, whether
     it is alive at the terminal block in the BUILT model; rlf; the 2025 cohort's S == 0 on a built model (credit
     expression constant 0, reporting expression 0 at two synthetic SoH points); the TV coefficient per cohort-year
     (EUR per MWh of E per unit normalised health, discounted to 2025) read off the built objective and checked
     against the hand formula; dS/dTP (EUR per MWh of cell throughput) from production's throughput coefficients.
  3  H3 DERATING over the staged domain (SRP1 grid, `p516_lattice` enumeration as audit I; 37,042 incl. x0): per plan
     and node-year the implied aggregate power limit L = min_c s_c / share_c (share_c = E_c / sum E, the H3 rows;
     s_c the per-cohort converter rows), ratio L / sum s, EXACT (fractions on the granule integers); counts by the
     number of mixed-duration nodes. Verified on BUILT ESSO models for >= 5 representative plans by READING THE ROWS
     (generate_standard_repn of the rated, limit and H3 rows; never a solve) and by evaluating the rows at a
     constructed point on the bound and 1e-6 beyond it. Implied per-cohort SoC excursion under H3 with calendar-only
     end-of-block SoH (both grids' age gaps, equal-E and supremum, and the maximum over the staged domain).
  4  CONVERTER AND COST-BASIS REPRICING tables and functions (zero-solve re-ranking of any evaluated plan): converter
     residual disc_T * E_m[c^S(y)] * rlf * P per cohort (node x year x P); S on the installation-year basis (code) and
     on the horizon-end basis (Addendum 88 R2 (i), last representative year's column, withdrawn by its amendment --
     a sensitivity only). The installation-basis function is cross-checked against production's salvage expression
     on a built model.
  5  CAPTURE PATHS for per-scenario dP / dQ / dV (READ-ONLY; the harness is not modified): source anchors in the harness
     and production, and the W91 3 x 3 x = 0 terminal capture (sha-checked against its child manifest, json only)
     re-reduced to per-DN max and omega-RMS and compared with audit D; what is missing for the step-1 gate fields.
  6  GRID-INPUT TWINS SCOPING (no construction): draw keys, growth factors, per-year network-file differences; the
     2025 arrays of the two grids compared bitwise; draw keys of the other years distinct.

Predictions (recorded in `predictions.json` BEFORE any computation, from the Planner's task):
  P1 derating < 1 only at nodes with mixed-duration cohorts; P2 2025 TV = 0; P3 converter residual per 0.25 MVA
  ~13 kEUR (2030) and ~24 kEUR (2035).

GUARDS: SolveProfileGuard(permitted=()) installed before any repository import and verified at exactly 0;
pickle.load / pickle.loads blocked and counted (0 expected; no model is unpickled).

Writes ONLY into data/SRP1/Results/P515S56/step0_gates/ (production read side writes redirected under reads/). The
launch command creates the directory (mkdir fails if it exists) to hold the two launch logs; the script refuses if it
holds anything else, or if another copy is running.

Launch (repository root, canonical interpreter, attached, both streams captured):
  mkdir data/SRP1/Results/P515S56/step0_gates && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s56_step0_gates.py \\
    > data/SRP1/Results/P515S56/step0_gates/launch_stdout.log \\
    2> data/SRP1/Results/P515S56/step0_gates/launch_stderr.log
then   .../python p515_s56_step0_gates.py --manifest
Exit: 0 written and every check holds; 1 written with failed checks or errors listed; 2 refused before computing.
"""
import os
import pickle
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('MPLBACKEND', 'Agg')

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S56 step 0 gates (zero solves)').install()
PICKLE_COUNTS = {'load': 0, 'loads': 0}


def _blocked_load(*_a, **_k):
    PICKLE_COUNTS['load'] += 1
    raise RuntimeError('step 0 gates: pickle.load called -- no model loads are permitted')


def _blocked_loads(*_a, **_k):
    PICKLE_COUNTS['loads'] += 1
    raise RuntimeError('step 0 gates: pickle.loads called -- no model loads are permitted')


pickle.load, pickle.loads = _blocked_load, _blocked_loads

import bisect  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from contextlib import contextmanager, redirect_stdout  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from itertools import product  # noqa: E402

import numpy as np  # noqa: E402

SCHEMA = 'p515_s56_step0_gates_v1'
SCRIPT = os.path.basename(os.path.abspath(__file__))
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'step0_gates')
OUT = os.path.join(REPO, OUT_REL)
LAUNCH_LOGS = ('launch_stdout.log', 'launch_stderr.log')
RESULT_FILE = 'step0_gates.json'
PREDICTIONS_FILE = 'predictions.json'
MANIFEST_FILE = 'manifest_sha256.json'

SRP1_DIR = os.path.join('data', 'SRP1')
SRP1_CASE = os.path.join(SRP1_DIR, 'SRP1.json')
INST = {
    'srp1grid_3x3': {'case': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1',
                                          'SRP1__s56_srp1grid_3x3.json'),
                     'case_sha256': '71d348492a31daa3b203459a52e97209f68fe61ff9d1e2e299f1c576eafc64e1',
                     'checksum': '73d0a230a0d53d24d214e13774206844acce824dfbb84bb2d07e16f8ed6333f7',
                     'n_market': 3, 'n_oper': 3},
    'srp1grid_3x3_ev': {'case': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1',
                                             'SRP1__s56_srp1grid_3x3_ev.json'),
                        'case_sha256': '45e8e700a547de22b2c9cd2f9373c4307da30764461e647424eca7eb2d1594b3',
                        'checksum': '0040b1463046cf09c3bb8c4f95d839f7871ac88b8522a720da8484cbcf6690de',
                        'profiles': os.path.join('data', 'SRP1', 'Results', 'P515S56', 'instances_v1',
                                                 'SRP1__s56_srp1grid_3x3_ev_profiles.json'),
                        'profiles_sha256': 'f48e9be57aa7e44508e703a32168bc1c99c657f77a11b9210aa51c949e881729',
                        'n_market': 1, 'n_oper': 1},
    'five_block_3x3_w89': {'case': os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w89_3x3', 'instance',
                                                'SRP1__s53_3x3.json'),
                           'case_sha256': '2a64e3c5bd06c30c8a015207f41b69257b001094cbd4b185a84394a6765c5871',
                           'checksum': '0248a5c8710bdbdf047523ca036bd2f4639546aebf18fd08391a64b340a3b193',
                           'n_market': 3, 'n_oper': 3},
}
TV_NODE = 5
EXPECTED_STAGED_INCL_X0 = 37042          # audit I (reuse_audit_I.json 0e91eb18), A7 point by point
Z_MAX = 11
BOUNDARY_EPS_EUR = 1e-6
ROW_TOL = 1e-12
BEYOND = 1e-6
# W91 3 x 3 pair, x = 0 (audit D's primary): the terminal capture re-reduced in item 5
W91_X0_EVAL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'w90_3x3', 'campaign_s53_w91_3x3_pair', 'evals',
                           'f6e9cd53fdbb8ee8_x0')
AUDIT_D_JSON = os.path.join('data', 'SRP1', 'Results', 'P515S56', 'interface_consistency', 'interface_consistency.json')
AUDIT_D_SHA256 = 'da5d365e'  # prefix as recorded in TASKS.md; the full hash is recorded at read time
PRODUCTION_FILES = ['shared_resources_planning.py', 'model_construction_helpers.py', 'shared_energy_storage_data.py',
                    'shared_energy_storage_parameters.py', 'network.py', 'network_data.py', 'helper_functions.py',
                    'definitions.py', 'p515_s44_campaign_harness.py', 'interface_dual_capture.py',
                    'p515_s44_scale_measurement.py', 'p515_s56_ev_injection.py', 'p516_lattice.py',
                    'p513_solve_profile_guard.py', 'gate_result_io.py']

PREDICTIONS = {
    'P1_derating_only_at_mixed_duration_nodes': {
        'source': 'Planner task (S56 step 0) / S56_PLAN_v2.md step 0 key prediction',
        'statement': 'derating < 1 only at nodes with mixed-duration cohorts',
        'operationalised': ('(a) node-year level: L/sum s < 1 iff the active cohorts of that node-year have at least two '
                            'distinct E/P durations; (b) plan level: a plan has some node-year ratio < 1 iff it has '
                            '>= 1 mixed-duration node. Exact rational arithmetic on the granule integers.')},
    'P2_2025_TV_zero': {
        'source': 'Planner task / S56_PLAN_v2.md step 0',
        'statement': '2025 TV = 0 (remaining-life fraction 0 hence S = 0)',
        'operationalised': ('on both grids: rlf(2025) == 0 exactly; on a built ESSO with only a 2025 cohort and '
                            'in_objective on, the salvage credit expression has no variable and constant 0.0, and the '
                            'reporting salvage expression evaluates to 0.0 at two synthetic terminal SoH values')},
    'P3_converter_residual_per_0p25_MVA': {
        'source': 'Planner task (Advisor estimate) / S56_PLAN_v2.md "Formulation as planned"',
        'statement': 'converter residual ~13 kEUR for 2030 and ~24 kEUR for 2035 per 0.25 MVA (SRP1 grid)',
        'operationalised': ('disc_T * E_m[c^S(y)] * rlf(y) * 0.25 MVA, pro-rata calendar life, installation-year expected '
                            'converter cost, discounted from the horizon end like S; "~" read as within +-10 % '
                            '(Worker operationalisation, declared here before computing)'),
        'expected_eur': {'2030': 13000.0, '2035': 24000.0}, 'rel_tol': 0.10},
    'expectation_not_prediction': {
        'staged_domain_incl_x0': EXPECTED_STAGED_INCL_X0,
        'advisor_soc_excursion_estimate': 'about 4 % at a 5-year age gap, 8 % at 10 (equal-energy reading, X6)'},
}


def _utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def _f(x):
    return None if x is None else float(x)


def _bitwise(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.dtype == b.dtype and a.shape == b.shape
                and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def _provenance():
    return {'head': _git('rev-parse', 'HEAD'), 'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
            'porcelain_production_files': _git('status', '--porcelain', '--', *PRODUCTION_FILES),
            'diff_sha256_production_files': hashlib.sha256(
                _git('diff', 'HEAD', '--', *PRODUCTION_FILES).encode()).hexdigest(),
            'sources_sha256': {p: sha256_file(os.path.join(REPO, p)) for p in PRODUCTION_FILES}}


# ======================================================================================================================
#  manifest / preflight
# ======================================================================================================================
def manifest():
    import gate_result_io as GRIO
    target = os.path.join(OUT, MANIFEST_FILE)
    if os.path.exists(target):
        raise SystemExit(f'refusing to overwrite {target}')
    files = {}
    for root, _dirs, names in os.walk(OUT):
        for name in sorted(names):
            path = os.path.join(root, name)
            files[os.path.relpath(path, REPO)] = sha256_file(path)
    files[SCRIPT] = sha256_file(os.path.join(REPO, SCRIPT))
    with open(target, 'w') as handle:
        handle.write(GRIO.dumps({'schema': SCHEMA + '_manifest', 'utc': _utc(), 'files': files},
                                indent=1, sort_keys=True))
    print(f'manifest: {len(files)} files -> {target}')
    return 0


def _preflight():
    if not os.path.isdir(OUT):
        print(f'REFUSE: {OUT_REL} must be created by the launch command (mkdir) to hold the launch logs', flush=True)
        return False
    extra = sorted(set(os.listdir(OUT)) - set(LAUNCH_LOGS))
    if extra:
        print(f'REFUSE: {OUT_REL} already holds {extra}', flush=True)
        return False
    ps = subprocess.run(['ps', '-axo', 'pid=,command='], capture_output=True, text=True).stdout.splitlines()
    others = []
    for line in ps:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] == str(os.getpid()):
            continue
        if 'python' in os.path.basename(parts[1]) and any(os.path.basename(p) == SCRIPT for p in parts[2:]):
            if '--manifest' not in parts:
                others.append(line)
    if others:
        print(f'REFUSE: another copy is running: {others}', flush=True)
        return False
    return True


# ======================================================================================================================
#  production reads
# ======================================================================================================================
class _Stages:
    @contextmanager
    def stage(self, _name):
        yield


def production_read(label, check):
    """The 3 x 3 pair's production reader (`p515_s44_scale_measurement.read_planning_from_derived_case`), side writes
    under OUT/reads/<label>/; the EV twin with the `p515_s56_ev_injection` hook (inject) as a harness child would."""
    import p515_s44_scale_measurement as S
    import p515_s56_ev_injection as EVI
    spec = INST[label]
    case_sha = sha256_file(spec['case'])
    check(f'1.read.{label}.case_sha256_pinned', case_sha == spec['case_sha256'],
          {'path': spec['case'], 'sha256': case_sha}, expected=spec['case_sha256'], observed=case_sha)
    read_dir = os.path.join(OUT, 'reads', label)
    os.makedirs(read_dir)
    hook, profiles = None, None
    if 'profiles' in spec:
        profiles = EVI.load_profiles(spec['profiles'], expected_sha256=spec['profiles_sha256'])
        hook = EVI.ScenarioPointHook(profiles=profiles).install()
    stream = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stdout(stream):
            planning = S.read_planning_from_derived_case({'derived_case': {'path': spec['case']}}, read_dir, _Stages())
    finally:
        if hook is not None:
            hook.uninstall()
        with open(os.path.join(read_dir, 'production_stdout.log'), 'w') as handle:
            handle.write(stream.getvalue())
    rec = {'case': spec['case'], 'case_sha256': case_sha, 'wall_s': time.time() - t0,
           'scenario_metadata': dict(planning.scenario_metadata or {}),
           'read_dir': os.path.relpath(read_dir, REPO)}
    checksum = rec['scenario_metadata'].get('combined_scenario_checksum')
    check(f'1.read.{label}.combined_scenario_checksum_pinned', checksum == spec['checksum'], None,
          expected=spec['checksum'], observed=checksum)
    if profiles is not None:
        rec['ev_injection'] = {'hook_calls': hook.calls, 'inject_counts': hook.inject_counts}
        ok, detail = True, None
        try:
            detail = EVI.assert_ev_arrays(planning, profiles, point='post_read',
                                          expected_scenario_checksum=spec['checksum'])
        except EVI.EvInjectionMismatch as exc:
            ok, detail = False, {'error': str(exc)[:2000], 'summary': exc.summary}
        check(f'1.read.{label}.ev_in_child_assertion_post_read_passes', ok and hook.calls == 1,
              {'hook_calls': hook.calls, 'summary': detail if isinstance(detail, dict) else str(detail)})
    return planning, rec


# ======================================================================================================================
#  item 1 -- probability and scenario identity
# ======================================================================================================================
def item1_identity(planning, label, check):
    import p515_s56_ev_injection as EVI
    import shared_resources_planning as SRP
    spec = INST[label]
    n_m, n_o = spec['n_market'], spec['n_oper']
    holders = EVI.holders(planning)
    blocks, fails = {}, []
    for year in sorted(planning.years):
        pm_plan = [float(v) for v in planning.prob_market_scenarios[year]]
        for day in planning.days:
            ref_e, ref_f = planning.cost_energy_p[year][day], planning.cost_flex[year][day]
            rec = {'planning_prob_market': pm_plan, 'planning_market_rows': [int(np.shape(ref_e)[0]),
                                                                              int(np.shape(ref_f)[0])],
                   'networks': {}}
            prob_m_set, prob_o_set, counts_set = set(), set(), set()
            for h, hd in holders:
                net = hd.network[year][day]
                op_rows = set()
                for load in net.loads:
                    for arr in (load.pd, load.qd, load.flexibility.active_power.upward,
                                load.flexibility.active_power.downward, load.flexibility.reactive_power.upward,
                                load.flexibility.reactive_power.downward):
                        op_rows.add(int(np.shape(arr)[0]))
                for gen in net.generators:
                    op_rows.add(int(np.shape(gen.pg)[0]))
                    op_rows.add(int(np.shape(gen.qg)[0]))
                pm = [float(v) for v in net.prob_market_scenarios]
                po = [float(v) for v in net.prob_operation_scenarios]
                joint = sum(SRP._scenario_probability(net, s_m, s_o) for s_m in range(len(pm)) for s_o in range(len(po)))
                r = {'prob_market': pm, 'prob_operation': po, 'num_oper_scenarios_block': int(net.num_oper_scenarios),
                     'num_oper_scenarios_holder': int(hd.num_oper_scenarios), 'operation_array_rows': sorted(op_rows),
                     'market_rows': [int(np.shape(net.cost_energy_p)[0]), int(np.shape(net.cost_flex)[0])],
                     'cost_energy_p_bitwise_eq_planning': _bitwise(net.cost_energy_p, ref_e),
                     'cost_flex_bitwise_eq_planning': _bitwise(net.cost_flex, ref_f),
                     'cost_energy_p_is_planning_object': net.cost_energy_p is ref_e,
                     'joint_probability_sum': float(joint)}
                rec['networks'][h] = r
                prob_m_set.add(tuple(pm))
                prob_o_set.add(tuple(po))
                counts_set.add((r['num_oper_scenarios_block'], r['num_oper_scenarios_holder'], tuple(r['market_rows']),
                                tuple(r['operation_array_rows']), len(pm), len(po)))
            uniform_m = [1.0 / n_m] * n_m
            uniform_o = [1.0 / n_o] * n_o
            ok = (len(prob_m_set) == 1 and list(next(iter(prob_m_set))) == pm_plan == uniform_m
                  and len(prob_o_set) == 1 and list(next(iter(prob_o_set))) == uniform_o
                  and counts_set == {(n_o, n_o, (n_m, n_m), (n_o,), n_m, n_o)}
                  and rec['planning_market_rows'] == [n_m, n_m]
                  and all(r['cost_energy_p_bitwise_eq_planning'] and r['cost_flex_bitwise_eq_planning']
                          for r in rec['networks'].values())
                  and all(abs(r['joint_probability_sum'] - 1.0) <= 1e-15 for r in rec['networks'].values()))
            rec['holds'] = bool(ok)
            if not ok:
                fails.append(f'{year}|{day}')
            blocks[f'{year}|{day}'] = rec
    sed = planning.shared_ess_data
    out = {'n_blocks': len(blocks), 'holders': [h for h, _ in holders], 'expected_n_market': n_m,
           'expected_n_oper': n_o, 'blocks_failing': fails, 'blocks': blocks,
           'aliasing_note': ('cost_energy_p_is_planning_object True means the network block holds the planning-level '
                             'array object itself (identity), so identity across networks is structural'),
           'ess_cost_trajectory_probabilities': [float(v) for v in sed.prob_market_scenarios],
           'ess_cost_trajectory_note': ('shared_ess_data.prob_market_scenarios (ESS workbook Scenarios sheet) weighs '
                                        'the investment-cost trajectories only; independent of the network market '
                                        'scenarios (A3)')}
    check(f'1.identity.{label}.all_blocks_all_networks', not fails and len(blocks) == len(planning.years) * len(planning.days),
          {'n_blocks': len(blocks), 'failing': fails[:10]},
          expected=f'{len(planning.years) * len(planning.days)} blocks, probabilities 1/{n_m} x 1/{n_o} on every network',
          observed=f'{len(blocks) - len(fails)} blocks hold')
    return out


# ======================================================================================================================
#  ESSO build helpers (production; zero solves)
# ======================================================================================================================
def build_esso(sed, node, plan_by_year, in_objective=False):
    """plan_by_year: {year (sed key): (s MVA, e MWh)}; missing years -> (0, 0). Production `_build_subproblem` +
    `_update_model_with_candidate_solution`; `in_objective` switched IN MEMORY and restored."""
    import shared_energy_storage_data as SED
    sv = sed.params.salvage_value
    saved = getattr(sv, 'in_objective', False)
    sv.in_objective = in_objective
    try:
        model = SED._build_subproblem(sed, node)
    finally:
        sv.in_objective = saved
    cand = {node: {y: {'s': float(plan_by_year.get(y, (0.0, 0.0))[0]), 'e': float(plan_by_year.get(y, (0.0, 0.0))[1])}
                   for y in sed.years}}
    SED._update_model_with_candidate_solution(sed, {node: model}, cand)
    return model


def _repn(expr):
    from pyomo.repn import generate_standard_repn
    return generate_standard_repn(expr, compute_values=True)


def _violation(con):
    import pyomo.environ as pe
    body = pe.value(con.body)
    v = 0.0
    if con.has_lb():
        v = max(v, pe.value(con.lower) - body)
    if con.has_ub():
        v = max(v, body - pe.value(con.upper))
    return float(v)


# ======================================================================================================================
#  item 2 -- terminal value on both grids
# ======================================================================================================================
def cohort_facts(sed, node):
    """Per cohort (installation year): calendar life, rlf, alive window, costs; production functions only."""
    import shared_energy_storage_data as SED
    years = list(sed.years)
    widths = [sed.years[y] for y in years]
    T = len(years) - 1
    horizon_end = int(years[0]) + int(sum(widths))
    disc_t = SED._get_terminal_discount_factor(sed)
    idx = sed.get_shared_energy_storage_idx(node)
    sv = sed.params.salvage_value
    out = {}
    for y_inv, year in enumerate(years):
        ess = sed.shared_energy_storages[year][idx]
        age, rem, rlf = SED._get_remaining_calendar_life(sed, y_inv, ess)
        n = sed.years[year]
        tcal_norm = round(ess.t_cal / n)
        last_alive = min(y_inv + tcal_norm, len(years)) - 1
        eol = float(int(year) + ess.t_cal)
        c_e = SED._get_expected_energy_investment_cost(sed, year)
        c_s = sum(p * sed.cost_investment['power'][m][year] for m, p in enumerate(sed.prob_market_scenarios))
        unit = disc_t * sv.energy_recovery_fraction * c_e * rlf
        alive_blocks_to_t = T - y_inv + 1
        soh_cal = ess.phi_cal ** (n * alive_blocks_to_t) if last_alive == T else None
        out[str(year)] = {
            'installation_year': int(year), 'block_width_y': int(n), 't_cal_y': float(ess.t_cal),
            'end_of_calendar_life_year': eol, 'horizon_end_year': horizon_end,
            'expires_strictly_inside_horizon': bool(eol < horizon_end),
            'reaches_end_of_life_exactly_at_horizon_end': bool(eol == horizon_end),
            'model_alive_blocks': [str(years[b]) for b in range(y_inv, last_alive + 1)],
            'alive_at_terminal_block_by_window': bool(last_alive == T),
            'age_at_terminal_y': float(age), 'remaining_life_y': float(rem), 'rlf': float(rlf),
            'soh_min': float(ess.soh_min), 'phi_cal': float(ess.phi_cal), 'k_cl_eff': float(ess.cl_eff),
            'eff_ch': float(ess.eff_ch), 'eff_dch': float(ess.eff_dch),
            'expected_energy_cost_installation_year_eur_per_mwh': float(c_e),
            'expected_power_cost_installation_year_eur_per_mva': float(c_s),
            'terminal_discount_factor': float(disc_t),
            'tv_unit_eur_per_mwh_at_normalised_health_1': float(unit),
            'tv_coef_eur_per_mwh_per_unit_soh_T': float(unit / (1.0 - ess.soh_min)),
            'calendar_only_terminal_soh': None if soh_cal is None else float(soh_cal),
            'tv_eur_per_mwh_at_calendar_only_soh_T': (None if soh_cal is None
                                                       else float(unit * (soh_cal - ess.soh_min) / (1.0 - ess.soh_min))),
            'dS_dTP_eur_per_mwh_throughput_at_calendar_only_soh_T': (
                None if soh_cal is None else float(-unit * soh_cal / ((1.0 - ess.soh_min) * 2.0 * ess.cl_eff))),
            'dS_dTP_per_unit_soh_T_eur_per_mwh': float(-unit / ((1.0 - ess.soh_min) * 2.0 * ess.cl_eff)),
        }
    return out, {'years': [str(y) for y in years], 'widths': widths, 'terminal_index': T,
                 'horizon_end_year': horizon_end, 'terminal_discount_factor': float(disc_t),
                 'discount_rate': float(sed.discount_factor),
                 'ess_cost_trajectory_probabilities': [float(v) for v in sed.prob_market_scenarios]}


def item2_terminal_value(planning, label, check):
    import pyomo.environ as pe
    import shared_energy_storage_data as SED
    from pyomo.environ import Var
    sed = planning.shared_ess_data
    soh_point, ageing_on = SED._esso_ageing_model_settings(sed)
    sv = sed.params.salvage_value
    config = {'available_energy_soh_point': soh_point, 'ageing_enabled': ageing_on, 'salvage_enabled': sv.enabled,
              'salvage_in_objective_in_file': getattr(sv, 'in_objective', None),
              'energy_recovery_fraction': sv.energy_recovery_fraction,
              'recycling_floor_fraction': sv.recycling_floor_fraction, 'cost_basis': sv.cost_basis,
              'health_basis': sv.health_basis, 'calendar_life_basis': sv.calendar_life_basis,
              'params_file': sed.params_file if hasattr(sed, 'params_file') else None}
    check(f'2.{label}.config_end_salvage_enabled', soh_point == 'end' and sv.enabled and ageing_on
          and sv.recycling_floor_fraction == 0.0 and sv.energy_recovery_fraction == 1.0, config,
          expected="soh point 'end', salvage enabled, f = 1, floor 0, ageing on", observed=config)
    facts, grid = cohort_facts(sed, TV_NODE)
    years = list(sed.years)
    T = len(years) - 1
    # (a) the 2025 cohort alone: S == 0 on the built model
    y0 = years[0]
    m0 = build_esso(sed, TV_NODE, {y0: (0.25, 0.5)}, in_objective=True)
    r0 = _repn(m0.salvage_credit_obj.expr)
    e_r = m0.es_e_rated_per_unit[0, T]
    rep_vals = []
    for soh in (facts[str(y0)]['calendar_only_terminal_soh'] or 1.0, 1.0):
        e_r.set_value(0.5)
        m0.es_soh_per_unit_cumul[0, T].set_value(soh)
        m0.es_e_available_per_unit[0, T].set_value(0.5 * soh)
        rep_vals.append(float(pe.value(m0.salvage_value)))
    first = {'rlf_2025': facts[str(y0)]['rlf'], 'credit_linear_vars': [v.name for v in r0.linear_vars],
             'credit_nonlinear': r0.nonlinear_expr is not None, 'credit_constant': float(r0.constant),
             'reporting_salvage_at_synthetic_soh': rep_vals,
             'e_rated_at_terminal_unfixed': not e_r.fixed}
    check(f'2.{label}.P2_2025_cohort_TV_zero_on_built_model',
          facts[str(y0)]['rlf'] == 0.0 and not first['credit_linear_vars'] and not first['credit_nonlinear']
          and first['credit_constant'] == 0.0 and all(v == 0.0 for v in rep_vals), first,
          expected='rlf 0; credit has no variable, constant 0.0; reporting expression 0.0', observed=first)
    # (b) a cohort in every year at TV_NODE: coefficients vs the hand formula; throughput coefficients
    plan = {y: (0.25, 0.5) for y in years}
    m = build_esso(sed, TV_NODE, plan, in_objective=True)
    rep = _repn(m.salvage_credit_obj.expr)
    coefs = {v.name: float(c) for v, c in zip(rep.linear_vars, rep.linear_coefs)}
    hand, const_hand = {}, 0.0
    alive_built = {}
    for y_inv, year in enumerate(years):
        f = facts[str(year)]
        alive_built[str(year)] = not m.es_e_rated_per_unit[y_inv, T].fixed
        if f['rlf'] > 0:
            hand[f'es_soh_per_unit_cumul[{y_inv},{T}]'] = -f['tv_unit_eur_per_mwh_at_normalised_health_1'] * 0.5 / (
                1.0 - f['soh_min'])
            const_hand += f['tv_unit_eur_per_mwh_at_normalised_health_1'] * 0.5 * f['soh_min'] / (1.0 - f['soh_min'])
    coef_ok = (set(coefs) == set(hand) and all(abs(coefs[k] - hand[k]) <= 1e-12 * abs(hand[k]) for k in hand)
               and abs(float(rep.constant) - const_hand) <= 1e-12 * max(abs(const_hand), 1.0)
               and rep.nonlinear_expr is None and not any(k.startswith('es_s_') for k in coefs))
    check(f'2.{label}.tv_objective_coefficients_equal_hand_formula', coef_ok,
          {'coefs': coefs, 'hand': hand, 'constant': float(rep.constant), 'constant_hand': const_hand})
    check(f'2.{label}.every_cohort_alive_at_terminal_block_in_built_model',
          all(alive_built.values()) and all(f['alive_at_terminal_block_by_window'] for f in facts.values()),
          alive_built)
    expired_inside = [y for y, f in facts.items() if f['expires_strictly_inside_horizon']]
    at_end = [y for y, f in facts.items() if f['reaches_end_of_life_exactly_at_horizon_end']]
    check(f'2.{label}.no_cohort_expires_strictly_inside_horizon_2025_at_horizon_end',
          not expired_inside and at_end == [str(years[0])],
          {'expires_strictly_inside': expired_inside, 'reaches_end_of_life_at_horizon_end': at_end,
           'horizon_end': grid['horizon_end_year']})
    # throughput coefficients (production dict) vs dS/dTP per unit SoH_T
    dt = SED.period_duration_hours(m)
    tp_check, tp_rows = True, {}
    days = list(sed.days)
    for (y_inv, y, d), (c_ch, c_dch) in m._salvage_throughput_coefficients.items():
        f = facts[str(years[y_inv])]
        n, nd = f['block_width_y'], sed.days[days[d]]
        per_ch = c_ch / (n * nd * dt * f['eff_ch'])
        per_dch = c_dch * f['eff_dch'] / (n * nd * dt)
        target = -f['dS_dTP_per_unit_soh_T_eur_per_mwh']
        ok = abs(per_ch - target) <= 1e-12 * abs(target) and abs(per_dch - target) <= 1e-12 * abs(target)
        tp_check = tp_check and ok
        tp_rows.setdefault(str(years[y_inv]), {'n_entries': 0, 'per_mwh_cell_throughput_per_unit_soh': per_ch})
        tp_rows[str(years[y_inv])]['n_entries'] += 1
    check(f'2.{label}.throughput_coefficients_consistent_with_dS_dTP', tp_check and str(years[0]) not in tp_rows,
          {'by_cohort': tp_rows, 'cohort_2025_absent': str(years[0]) not in tp_rows})
    # synthetic point: reporting value == -credit == sum of per-cohort formula (cross-check for item 4)
    for y_inv, year in enumerate(years):
        soh = facts[str(year)]['calendar_only_terminal_soh']
        m.es_e_rated_per_unit[y_inv, T].set_value(0.5)
        m.es_soh_per_unit_cumul[y_inv, T].set_value(soh)
        m.es_e_available_per_unit[y_inv, T].set_value(0.5 * soh)
    s_rep, s_obj = float(pe.value(m.salvage_value)), float(pe.value(m.salvage_credit_obj))
    s_formula = sum(0.5 * f['tv_eur_per_mwh_at_calendar_only_soh_T'] for f in facts.values())
    synthetic = {'salvage_value_reporting': s_rep, 'salvage_credit_obj': s_obj, 'per_cohort_formula_sum': s_formula}
    check(f'2.{label}.synthetic_point_reporting_eq_minus_credit_eq_formula',
          abs(s_rep + s_obj) <= 1e-9 * max(abs(s_rep), 1.0) and abs(s_rep - s_formula) <= 1e-9 * max(abs(s_rep), 1.0),
          synthetic)
    n_vars = sum(1 for _ in m.component_data_objects(Var))
    return {'config': config, 'grid': grid, 'cohorts': facts, 'cohort_2025_alone': first,
            'all_cohort_plan_node': TV_NODE, 'all_cohort_plan_mva_mwh': {str(y): list(v) for y, v in plan.items()},
            'objective_coefs': coefs, 'objective_constant': float(rep.constant), 'synthetic_point': synthetic,
            'throughput_coefficients': tp_rows, 'model_n_vars': n_vars,
            'units': {'tv_unit': 'EUR (PV 2025) per MWh of installed E at normalised health h = 1',
                      'tv_coef_per_unit_soh_T': 'EUR (PV 2025) per MWh of E per unit of SoH_T (= unit / (1 - SoH_min))',
                      'tv_at_calendar_only_soh_T': 'EUR (PV 2025) per MWh of E at SoH_T = phi^(age)',
                      'dS_dTP': ('EUR (PV 2025) per MWh of cell energy throughput (eff_ch*pch*dt + pdch*dt/eff_dch, the '
                                 'D-row quantity, summed over the block) at calendar-only SoH_T; = - unit SoH_T / '
                                 '((1 - SoH_min) 2 k)')}}


# ======================================================================================================================
#  item 3 -- H3 derating over the staged domain
# ======================================================================================================================
def enumerate_staged(check):
    """All admissible staged plans on SRP1 (incl. x0), decided point by point by GeneralLattice.admissible."""
    import p516_lattice as L
    inst = L.load_instance(SRP1_CASE)
    lat = L.GeneralLattice(inst)
    years, nodes, budget = inst.years, inst.nodes, inst.budget_eur
    cohort_opts = [(zp, ze) for zp in range(Z_MAX + 1) for ze in range(Z_MAX + 1)
                   if not lat._cohort_reasons(nodes[0], years[0], zp, ze)[0]]

    def node_cost(pairs):
        total = 0.0
        for y, (zp, ze) in zip(years, pairs):
            p, e = L.GRANULE_P_MVA * zp, L.GRANULE_E_MWH * ze
            for c_s, c_e in lat._coef[y]:
                total += c_s * p
                total += c_e * e
        return total

    def cumulative_ok(pairs):
        cum = {y: 0.0 for y in years}
        for y_inv, (_zp, ze) in zip(years, pairs):
            for y in lat.alive_blocks[y_inv]:
                cum[y] += L.GRANULE_E_MWH * ze
        return all(v <= inst.max_capacity_mwh for v in cum.values())

    per_node = {}
    for n in nodes:
        per_node[n] = [(pairs, node_cost(pairs)) for pairs in product(cohort_opts, repeat=len(years))
                       if cumulative_ok(pairs) and node_cost(pairs) <= budget + 1.0]
    n0, n1, n2 = nodes
    third = sorted((c, pairs) for pairs, c in per_node[n2])
    third_costs = [c for c, _ in third]
    plans, n_checked = [], 0
    for (p0, c0), (p1, c1) in product(per_node[n0], per_node[n1]):
        hi = bisect.bisect_right(third_costs, budget + BOUNDARY_EPS_EUR - c0 - c1)
        for c2, p2 in third[:hi]:
            z = tuple(v for pairs in (p0, p1, p2) for pair in pairs for v in pair)
            n_checked += 1
            if lat.admissible(z):
                plans.append(z)
    plans.sort()
    check('3.enumeration_reproduces_audit_I_count_incl_x0', len(plans) == EXPECTED_STAGED_INCL_X0,
          {'n_checked_by_GeneralLattice': n_checked}, expected=EXPECTED_STAGED_INCL_X0, observed=len(plans))
    return lat, inst, plans


def node_year_cohorts(lat, z, node, year):
    """Active cohorts of (node, year): [(y_inv, zP, zE)] alive in `year` with zP > 0."""
    out = []
    for y_inv in lat.years:
        zp, ze = lat.pair(z, node, y_inv)
        if zp > 0 and year in lat.alive_blocks[y_inv]:
            out.append((y_inv, zp, ze))
    return out


def derating_exact(cohorts):
    """(L MVA, sum s, ratio) as Fractions; s = zP/4 MVA, E = zE/2 MWh, share = E/sum E, L = min s/share."""
    s = {c: Fraction(zp, 4) for c, zp, _ in cohorts}
    e = {c: Fraction(ze, 2) for c, _, ze in cohorts}
    tot_s, tot_e = sum(s.values()), sum(e.values())
    lim = min(s[c] * tot_e / e[c] for c in s)
    return lim, tot_s, lim / tot_s


def item3_derating(lat, plans, check):
    x0 = lat.x0()
    stats = {'n_plans_excl_x0': 0, 'n_derated': 0, 'by_n_mixed_nodes': {}, 'by_n_multi_cohort_nodes': {},
             'node_year_multi_cohort': 0, 'node_year_mixed': 0, 'node_year_ratio_lt_1': 0,
             'node_year_ratio_lt_1_not_mixed': 0, 'node_year_mixed_not_derated': 0,
             'plan_derated_without_mixed_node': 0, 'plan_mixed_without_derating': 0,
             'by_year_ratio_lt_1': {str(y): 0 for y in lat.years}, 'ratio_histogram_node_year': {},
             'lost_mva_max': 0.0}
    worst = None
    plan_min_ratios = []
    examples = {}
    for z in plans:
        if z == x0:
            continue
        stats['n_plans_excl_x0'] += 1
        mixed_nodes, multi_nodes, min_ratio = 0, 0, Fraction(1)
        for n in lat.nodes:
            node_mixed, node_multi = False, False
            for y in lat.years:
                co = node_year_cohorts(lat, z, n, y)
                if len(co) < 2:
                    continue
                node_multi = True
                stats['node_year_multi_cohort'] += 1
                durs = {Fraction(2 * ze, zp) for _c, zp, ze in co}
                mixed = len(durs) > 1
                lim, tot_s, ratio = derating_exact(co)
                # the closed form: ratio = sum E / (max dur * sum s)
                assert ratio == sum(Fraction(ze, 2) for _c, _zp, ze in co) / (max(durs) * tot_s)
                if mixed:
                    node_mixed = True
                    stats['node_year_mixed'] += 1
                if ratio < 1:
                    stats['node_year_ratio_lt_1'] += 1
                    stats['by_year_ratio_lt_1'][str(y)] += 1
                    if not mixed:
                        stats['node_year_ratio_lt_1_not_mixed'] += 1
                    key = f'{math.floor(float(ratio) * 20) / 20:.2f}'
                    stats['ratio_histogram_node_year'][key] = stats['ratio_histogram_node_year'].get(key, 0) + 1
                    stats['lost_mva_max'] = max(stats['lost_mva_max'], float(tot_s - lim))
                elif mixed:
                    stats['node_year_mixed_not_derated'] += 1
                if ratio < min_ratio:
                    min_ratio = ratio
                if worst is None or ratio < worst[0]:
                    worst = (ratio, z, n, y, co, lim, tot_s)
            mixed_nodes += node_mixed
            multi_nodes += node_multi
        derated = min_ratio < 1
        stats['n_derated'] += derated
        b = stats['by_n_mixed_nodes'].setdefault(str(mixed_nodes), {'plans': 0, 'derated': 0})
        b['plans'] += 1
        b['derated'] += derated
        c = stats['by_n_multi_cohort_nodes'].setdefault(str(multi_nodes), {'plans': 0, 'derated': 0})
        c['plans'] += 1
        c['derated'] += derated
        if derated and mixed_nodes == 0:
            stats['plan_derated_without_mixed_node'] += 1
        if mixed_nodes > 0 and not derated:
            stats['plan_mixed_without_derating'] += 1
        if derated:
            plan_min_ratios.append(float(min_ratio))
        examples.setdefault(f'mixed_nodes_{mixed_nodes}', z)
    n = stats['n_plans_excl_x0']
    stats['pct_derated'] = 100.0 * stats['n_derated'] / n
    for d in (stats['by_n_mixed_nodes'], stats['by_n_multi_cohort_nodes']):
        for v in d.values():
            v['pct_of_all_plans'] = 100.0 * v['plans'] / n
            v['pct_derated_within'] = 100.0 * v['derated'] / v['plans']
    pm = np.array(sorted(plan_min_ratios))
    stats['plan_min_ratio_among_derated'] = {
        'min': float(pm.min()), 'p05': float(np.quantile(pm, 0.05)), 'median': float(np.median(pm)),
        'p95': float(np.quantile(pm, 0.95)), 'max': float(pm.max())} if len(pm) else None
    stats['ratio_histogram_node_year'] = dict(sorted(stats['ratio_histogram_node_year'].items()))
    stats['worst'] = {'ratio': float(worst[0]), 'ratio_exact': str(worst[0]), 'z': list(worst[1]), 'node': worst[2],
                      'year': worst[3], 'cohorts_yinv_zP_zE': [list(c) for c in worst[4]],
                      'L_mva': float(worst[5]), 'sum_s_mva': float(worst[6])}
    stats['closed_form'] = ('ratio = sum_c E_c / (max_c dur_c * sum_c s_c) = (s-weighted mean duration) / (max '
                            'duration); the longest-duration cohort binds; ratio = 1 iff all active durations equal')
    p1a = stats['node_year_ratio_lt_1_not_mixed'] == 0 and stats['node_year_mixed_not_derated'] == 0
    p1b = stats['plan_derated_without_mixed_node'] == 0 and stats['plan_mixed_without_derating'] == 0
    check('3.P1_derating_lt_1_iff_mixed_duration_node_year_and_plan', p1a and p1b and n == EXPECTED_STAGED_INCL_X0 - 1,
          {k: stats[k] for k in ('node_year_ratio_lt_1_not_mixed', 'node_year_mixed_not_derated',
                                 'plan_derated_without_mixed_node', 'plan_mixed_without_derating')},
          expected='ratio < 1 only (and always) at mixed-duration node-years',
          observed=f"{stats['node_year_ratio_lt_1']} node-years < 1, all mixed; {stats['n_derated']} of {n} plans")
    return stats, worst, examples


def pick_representatives(lat, plans_set, worst, examples):
    def plan_from(spec):
        z = list(lat.x0())
        for (n, y), (zp, ze) in spec.items():
            i = lat.index(n, y)
            z[i], z[i + 1] = zp, ze
        return tuple(z)
    reps = {
        'R1_single_cohort_n5_2025_0.25_1.0': plan_from({(5, 2025): (1, 2)}),
        'R2_equal_duration_n5_2025_2030_0.25_0.5': plan_from({(5, 2025): (1, 1), (5, 2030): (1, 1)}),
        'R3_planner_example_n5_2025_0.25_1.0__2030_0.25_0.5': plan_from({(5, 2025): (1, 2), (5, 2030): (1, 1)}),
        'R4_three_cohorts_n7_2025_0.25_1.0__2030_0.5_1.0__2035_0.25_0.5': plan_from(
            {(7, 2025): (1, 2), (7, 2030): (2, 2), (7, 2035): (1, 1)}),
        'R5_worst_ratio_in_domain': tuple(worst[1]),
    }
    for k in ('mixed_nodes_2', 'mixed_nodes_3'):
        if k in examples:
            reps[f'R6_first_plan_with_{k}'] = tuple(examples[k])
    return {k: z for k, z in reps.items()}, {k: (z in plans_set) for k, z in reps.items()}


def verify_on_built_models(sed, lat, reps, check):
    """Read the rated / limit / H3 / aggregate rows of production-built ESSO models (no solve) and recover L; then
    evaluate the rows at a constructed point on L and (1 + BEYOND) L."""
    import pyomo.environ as pe
    from pyomo.core.expr.visitor import identify_variables
    years_sed = list(sed.years)
    ymap = {int(y): y for y in years_sed}
    out = {}
    all_ok = True
    for name, z in reps.items():
        per_node = {}
        for n in lat.nodes:
            plan = {ymap[y]: (0.25 * lat.pair(z, n, y)[0], 0.5 * lat.pair(z, n, y)[1]) for y in lat.years}
            m = build_esso(sed, n, plan, in_objective=False)
            nd, npd = len(list(m.days)), len(list(m.periods))
            # rated converter per cohort-year from the active rated_s_capacity_unit rows
            s_rows = {}
            for con in m.rated_s_capacity_unit.values():
                if not con.active:
                    continue
                r = _repn(con.body)
                var = [v for v in r.linear_vars if v.parent_component().name == 'es_s_rated_per_unit']
                assert len(var) == 1 and con.equality
                coef = [float(a) for v, a in zip(r.linear_vars, r.linear_coefs) if v is var[0]][0]
                c, y = var[0].index()
                s_rows[(c, y)] = (float(pe.value(con.upper)) - float(r.constant)) / coef
            # H3 rows -> shares (all (d, p))
            shares, h3_active = {}, {}
            for c in m.years:
                for cname, idx, y in m._esso_cohort_constraints[c]:
                    if cname != 'energy_storage_cohort_pnet_share_h3':
                        continue
                    con = m.energy_storage_cohort_pnet_share_h3[idx]
                    if not con.active:
                        continue
                    r = _repn(con.body)
                    cf = {(v.parent_component().name, v.index()): float(a) for v, a in zip(r.linear_vars, r.linear_coefs)}
                    dp = {(i[2], i[3]) for (nm, i) in cf if nm == 'es_pch_per_unit'}
                    assert len(dp) == 1
                    d, p = dp.pop()
                    others = [k for (nm, k) in cf if nm == 'es_pch_per_unit' and k[0] != c]
                    sh = -cf[('es_pch_per_unit', others[0])]
                    own = cf[('es_pch_per_unit', (c, y, d, p))]
                    consistent = abs(own - (1.0 - sh)) <= 1e-14 and all(
                        abs(cf[('es_pch_per_unit', k)] + sh) <= 1e-14 for k in others) and all(
                        abs(cf[('es_pdch_per_unit', k)] - sh) <= 1e-14 for k in others) and abs(
                        cf[('es_pdch_per_unit', (c, y, d, p))] - (sh - 1.0)) <= 1e-14
                    assert consistent, (name, n, c, y, d, p)
                    shares.setdefault((c, y), set()).add(sh)
                    h3_active.setdefault(y, set()).add((c, d, p))
            node_rec = {}
            for yi, year in enumerate(lat.years):
                co = node_year_cohorts(lat, z, n, year)
                if not co:
                    continue
                cidx = [lat.years.index(c) for c, _zp, _ze in co]
                lim_x, tot_x, ratio_x = derating_exact(co)
                s_read = {c: s_rows[(c, yi)] for c in cidx}
                # shares read from rows; the omitted cohort (largest y_inv) from the sum-to-one identity
                sh_read = {}
                for c in cidx:
                    vals = shares.get((c, yi))
                    if vals is not None:
                        assert len(vals) == 1
                        sh_read[c] = vals.pop()
                n_h3_rows = len(h3_active.get(yi, ()))
                if len(cidx) >= 2:
                    omitted = [c for c in cidx if c not in sh_read]
                    assert len(omitted) == 1 and omitted[0] == max(cidx)
                    sh_read[omitted[0]] = 1.0 - sum(sh_read.values())
                    lim_rows = min(s_read[c] / sh_read[c] for c in cidx)
                    binding = min(cidx, key=lambda c: s_read[c] / sh_read[c])
                else:
                    sh_read[cidx[0]] = 1.0
                    lim_rows = s_read[cidx[0]]
                    binding = cidx[0]
                tot_rows = sum(s_read.values())
                # constructed point at (yi, d=0, p=0): discharge at L, every cohort at its share
                d0, p0 = 0, 0
                for c in m.years:
                    for nm in ('es_pch_per_unit', 'es_pdch_per_unit'):
                        v = getattr(m, nm)[c, yi, d0, p0]
                        if not v.fixed:
                            v.set_value(0.0)
                    vr = m.es_s_rated_per_unit[c, yi]
                    if not vr.fixed:
                        vr.set_value(s_read.get(c, 0.0))
                m.es_s_rated[yi].set_value(tot_rows)
                m.es_qnet[yi, d0, p0].set_value(0.0)
                if hasattr(m, 'slack_es_pnet_up'):
                    m.slack_es_pnet_up[yi, d0, p0].set_value(0.0)
                    m.slack_es_pnet_down[yi, d0, p0].set_value(0.0)
                base = 2 * ((yi * nd + d0) * npd + p0)
                agg_def, circle = m.energy_storage_operation_agg[base + 1], m.energy_storage_operation_agg[base + 2]
                assert m.es_pnet[yi, d0, p0].name in {v.name for v in identify_variables(agg_def.body)}
                assert m.es_pnet[yi, d0, p0].name in {v.name for v in identify_variables(circle.body)}
                rows_here = []
                for c in cidx:
                    for cname, idx, y in m._esso_cohort_constraints[c]:
                        if y != yi or cname not in ('energy_storage_limits', 'energy_storage_cohort_pnet_share_h3'):
                            continue
                        con = getattr(m, cname)[idx]
                        if not con.active:
                            continue
                        vs = {(v.parent_component().name, v.index()) for v in
                              identify_variables(con.body, include_fixed=False)}
                        if any(nm in ('es_pch_per_unit', 'es_pdch_per_unit') and i[2:] == (d0, p0) for nm, i in vs):
                            rows_here.append((cname, c, con, vs))

                def evaluate(scale):
                    lval = lim_rows * scale
                    for c in cidx:
                        m.es_pdch_per_unit[c, yi, d0, p0].set_value(sh_read[c] * lval)
                    m.es_pnet[yi, d0, p0].set_value(-lval)
                    viol = {'h3_max': 0.0, 'limits_max': 0.0, 'binding_limit': None, 'agg_def': _violation(agg_def),
                            'circle': _violation(circle)}
                    for cname, c, con, vs in rows_here:
                        v = _violation(con)
                        if cname == 'energy_storage_cohort_pnet_share_h3':
                            viol['h3_max'] = max(viol['h3_max'], v)
                        else:
                            viol['limits_max'] = max(viol['limits_max'], v)
                            if c == binding and ('es_pdch_per_unit', (c, yi, d0, p0)) in vs:
                                viol['binding_limit'] = v
                                viol['binding_limit_body_minus_ub'] = float(pe.value(con.body) - pe.value(con.upper))
                    return viol
                at = evaluate(1.0)
                beyond = evaluate(1.0 + BEYOND)
                ok = (abs(lim_rows - float(lim_x)) <= ROW_TOL and abs(tot_rows - float(tot_x)) <= ROW_TOL
                      and n_h3_rows == (len(cidx) - 1) * nd * npd
                      and at['h3_max'] <= ROW_TOL and at['limits_max'] <= ROW_TOL and at['agg_def'] <= ROW_TOL
                      and at['circle'] <= ROW_TOL and abs(at['binding_limit_body_minus_ub']) <= ROW_TOL
                      and beyond['binding_limit'] > 0.0 and beyond['h3_max'] <= ROW_TOL)
                all_ok = all_ok and ok
                node_rec[str(year)] = {
                    'cohorts_yinv_zP_zE': [list(c) for c in co], 's_read_mva': {str(lat.years[c]): s_read[c] for c in cidx},
                    'share_read': {str(lat.years[c]): sh_read[c] for c in cidx},
                    'n_active_h3_rows': n_h3_rows, 'L_from_rows_mva': lim_rows, 'L_exact_mva': float(lim_x),
                    'sum_s_mva': tot_rows, 'ratio_from_rows': lim_rows / tot_rows, 'ratio_exact': str(ratio_x),
                    'binding_cohort': str(lat.years[binding]), 'point_at_L': at, 'point_beyond_L': beyond, 'ok': ok}
            per_node[str(n)] = node_rec
        out[name] = {'z': list(z), 'nodes': per_node}
    check('3.derivation_verified_on_built_models_by_reading_rows', all_ok and len(reps) >= 5,
          {'n_plans': len(reps)}, expected='L from rows == exact L; rows hold at L; binding limit row violated beyond L',
          observed='all hold' if all_ok else 'see item3.built_model_verification')
    return out


def soc_excursion(lat_years, widths, phi, plans=None, lat=None):
    """Under H3 (pnet_c = share_c * agg, share by INSTALLED E) and aggregate SoC limits [0.1, 0.9] x available energy
    (start / closure 0.5), the aggregate stored-energy excursion from the start is <= 0.4 sum_k E_k SoH_k; cohort c
    moves share_c of it, i.e. <= 0.4 E_c (sum_k E_k SoH_k / sum_k E_k); relative to its OWN available energy E_c SoH_c
    the bound is 0.4 r_c with r_c = SoH_bar_E / SoH_c. The oldest cohort has r > 1: its implied SoC range
    0.5 +- 0.4 r exceeds its own [0.1, 0.9] by (r - 1). SoH calendar-only, END of block (the 'end' available energy)."""
    import definitions as DEFS
    lo, hi, s0 = DEFS.ENERGY_STORAGE_MIN_ENERGY_STORED, DEFS.ENERGY_STORAGE_MAX_ENERGY_STORED, DEFS.ENERGY_STORAGE_RELATIVE_INIT_SOC
    half = min(hi - s0, s0 - lo)
    table = {}
    w = widths[0]
    for gap_blocks in range(1, len(lat_years)):
        gap_y = gap_blocks * w
        # older cohort a blocks older at the end of a block: SoH_old / SoH_new = phi^gap
        ratio = phi ** gap_y
        r_equal = (1.0 + 1.0 / ratio) / 2.0  # equal E: SoH_bar / SoH_old = (SoH_old + SoH_new) / (2 SoH_old)
        r_sup = 1.0 / ratio
        table[f'{gap_y}y'] = {'soh_old_over_soh_new': ratio, 'excess_equal_E_pct': 100.0 * (r_equal - 1.0),
                              'excess_supremum_pct': 100.0 * (r_sup - 1.0),
                              'implied_soc_range_equal_E': [s0 - half * r_equal, s0 + half * r_equal]}
    out = {'soc_limits': [lo, hi], 'start_closure': s0, 'by_age_gap': table}
    if plans is not None:
        worst = (0.0, None)
        for z in plans:
            for n in lat.nodes:
                for yi, y in enumerate(lat.years):
                    co = node_year_cohorts(lat, z, n, y)
                    if len(co) < 2:
                        continue
                    soh = {c: phi ** (w * (yi - lat.years.index(c) + 1)) for c, _zp, _ze in co}
                    e = {c: 0.5 * ze for c, _zp, ze in co}
                    sbar = sum(e[c] * soh[c] for c in soh) / sum(e.values())
                    r = max(sbar / soh[c] for c in soh)
                    if r > worst[0]:
                        worst = (r, {'z': list(z), 'node': n, 'year': y, 'cohorts_yinv_zP_zE': [list(c) for c in co]})
        out['max_over_staged_domain'] = {'excess_pct': 100.0 * (worst[0] - 1.0), 'where': worst[1]}
    return out


# ======================================================================================================================
#  item 4 -- converter and cost-basis repricing
# ======================================================================================================================
def repricing_table(facts, grid, nodes):
    """Per installation year: converter residual and S unit values on both cost bases. Pure; reusable."""
    years = grid['years']
    last = years[-1]
    ce_last = facts[last]['expected_energy_cost_installation_year_eur_per_mwh']
    per_year = {}
    for y in years:
        f = facts[y]
        d, rlf = f['terminal_discount_factor'], f['rlf']
        per_year[y] = {
            'rlf': rlf, 'disc_T': d,
            'expected_power_cost_eur_per_mva': f['expected_power_cost_installation_year_eur_per_mva'],
            'expected_energy_cost_eur_per_mwh': f['expected_energy_cost_installation_year_eur_per_mwh'],
            'converter_residual_eur_per_mva': d * f['expected_power_cost_installation_year_eur_per_mva'] * rlf,
            'converter_residual_eur_per_0p25_mva': 0.25 * d * f['expected_power_cost_installation_year_eur_per_mva'] * rlf,
            'S_installation_basis_eur_per_mwh_per_unit_h': d * f['expected_energy_cost_installation_year_eur_per_mwh'] * rlf,
            'S_horizon_end_basis_eur_per_mwh_per_unit_h': d * ce_last * rlf,
            'horizon_end_over_installation_basis': (ce_last / f['expected_energy_cost_installation_year_eur_per_mwh']),
            'soh_min': f['soh_min'], 'calendar_only_terminal_soh': f['calendar_only_terminal_soh']}
    rows = []
    for n in nodes:
        for y in years:
            for zp in range(1, 11):
                p = 0.25 * zp
                rows.append({'node': int(n), 'year': y, 'P_mva': p,
                             'converter_residual_eur': per_year[y]['converter_residual_eur_per_mva'] * p})
    return {'horizon_end_cost_column': last, 'per_year': per_year, 'per_cohort_node_year_P': rows,
            'formulas': {
                'converter_residual': 'C_conv(x) = disc_T * sum_cohorts E_m[c^S(y_inv)] * rlf(y_inv) * P (EUR, PV 2025)',
                'S_installation': ('S = disc_T * sum_cohorts E_m[c^E(y_inv)] * rlf * E * (SoH_T - SoH_min)/(1 - SoH_min) '
                                   '(the code, Addendum 88 amendment)'),
                'S_horizon_end': ('S_hend = disc_T * sum_cohorts E_m[c^E(last representative year)] * rlf * E * h '
                                  '(Addendum 88 R2 (i), withdrawn by the amendment; sensitivity only)'),
                'E_m': 'ESS workbook cost-trajectory probabilities (shared_ess_data.prob_market_scenarios)'}}


def reprice(plan, table, soh_t=None, basis='installation'):
    """Zero-solve repricing of an evaluated plan. plan: {node: {year_str: (P MVA, E MWh)}}; soh_t: {node: {year_str:
    SoH_T}} (default: calendar-only). Returns (S_basis, C_conv) in EUR PV 2025."""
    s_total, conv = 0.0, 0.0
    key = 'S_installation_basis_eur_per_mwh_per_unit_h' if basis == 'installation' else \
        'S_horizon_end_basis_eur_per_mwh_per_unit_h'
    for n, by_year in plan.items():
        for y, (p, e) in by_year.items():
            row = table['per_year'][y]
            soh = row['calendar_only_terminal_soh'] if soh_t is None else soh_t[n][y]
            h = (soh - row['soh_min']) / (1.0 - row['soh_min'])
            s_total += row[key] * e * h
            conv += row['converter_residual_eur_per_mva'] * p
    return s_total, conv


def item4_repricing(tv, check):
    out = {}
    for label, rec in tv.items():
        out[label] = repricing_table(rec['cohorts'], rec['grid'], (5, 7, 9))
        # cross-check: installation basis at calendar-only SoH == production's salvage at the synthetic point (item 2)
        y = rec['grid']['years']
        s_fn, _ = reprice({TV_NODE: {yy: (0.25, 0.5) for yy in y}}, out[label])
        prod = rec['synthetic_point']['salvage_value_reporting']
        check(f'4.{label}.installation_basis_function_equals_production_salvage',
              abs(s_fn - prod) <= 1e-9 * max(abs(prod), 1.0), {'function': s_fn, 'production': prod})
    srp = out.get('srp1grid')
    if srp is not None:
        exp = PREDICTIONS['P3_converter_residual_per_0p25_MVA']['expected_eur']
        obs = {y: srp['per_year'][y]['converter_residual_eur_per_0p25_mva'] for y in exp}
        rel = {y: (obs[y] - exp[y]) / exp[y] for y in exp}
        check('4.P3_converter_residual_2030_13k_2035_24k_per_0p25_MVA',
              all(abs(r) <= PREDICTIONS['P3_converter_residual_per_0p25_MVA']['rel_tol'] for r in rel.values()),
              {'rel_diff': rel}, expected=exp, observed=obs)
        examples = {
            'reference_unit_n7_2025_0.25_1.0': {7: {'2025': (0.25, 1.0)}},
            'plan_2030_n5_0.25_0.5__n7_1.0_3.5': {5: {'2030': (0.25, 0.5)}, 7: {'2030': (1.0, 3.5)}},
            'single_2035_n5_0.25_1.0': {5: {'2035': (0.25, 1.0)}},
        }
        out['srp1grid_examples_calendar_only_soh'] = {
            k: dict(zip(('S_installation', 'C_conv'), reprice(p, srp)))
            | {'S_horizon_end': reprice(p, srp, basis='horizon_end')[0]} for k, p in examples.items()}
    out['functions'] = {'module': SCRIPT, 'repricing_table': 'repricing_table(facts, grid, nodes)',
                        'reprice': "reprice(plan, table, soh_t=None, basis='installation'|'horizon_end')",
                        'note': ('import-safe only under a solve guard (importing this module installs a permitted=() '
                                 'SolveProfileGuard and blocks pickle loads, as every p515_s56 audit script does); the '
                                 'tables are also stored in step0_gates.json under item4')}
    return out


# ======================================================================================================================
#  item 5 -- capture paths (read-only)
# ======================================================================================================================
CAPTURE_FACTS = [
    ('ms_terminal_file', 'p515_s44_campaign_harness.py', r"^MULTISCENARIO_TERMINAL_FILE = 'multiscenario_terminal.json'"),
    ('ms_terminal_written_only_for_derived_or_premium', 'p515_s44_campaign_harness.py',
     r"holder\['multiscenario_terminal'\] = write_multiscenario_terminal\("),
    ('interface_profiles_per_scenario', 'p515_s44_campaign_harness.py', r'^def _interface_profiles\(model, network, kind\):'),
    ('interface_profiles_process_results_interface', 'p515_s44_campaign_harness.py',
     r'per = network.process_results_interface\(model\)'),
    ('tso_committed_expected_interface', 'p515_s44_campaign_harness.py', r"model.expected_interface_pf_p\[dn, p\]"),
    ('dispersion_pi_by_market_by_hour', 'p515_s44_campaign_harness.py', r"'pi_by_market_by_hour': \[\[float"),
    ('block_weight_in_terminal_entry', 'p515_s44_campaign_harness.py', r"'admm_block_weight': weight,"),
    ('per_cycle_response_fields', 'p515_s44_campaign_harness.py', r'^PER_CYCLE_RESPONSE_FIELDS = \('),
    ('per_cycle_response_dso_dispersion_only', 'p515_s44_campaign_harness.py',
     r"dispersion = srp._get_operational_interface_dispersion\(planning, models\)"),
    ('per_cycle_max_abs_d_is_dso_vs_own_commitment', 'p515_s44_campaign_harness.py',
     r"max_abs = max\(max_abs, detail\['p'\]\['max_abs_mw'\]\)"),
    ('dual_capture_channels_expectation_only', 'interface_dual_capture.py', r'^CHANNEL_FIELDS = '),
    ('production_dispersion_function', 'shared_resources_planning.py', r'^def _get_operational_interface_dispersion'),
    ('production_scenario_voltage_mismatch', 'shared_resources_planning.py',
     r'^def _get_operational_scenario_voltage_mismatch'),
    ('production_tn_interface_delta', 'model_construction_helpers.py', r'interface_delta_p'),
]
REQUIRED_FIELDS = {
    'dP_s per (DN, block, scenario, hour) = P_TSO[s,t] - P_DSO[s,t]': 'terminal',
    'dQ_s likewise': 'terminal',
    'dV_s = V_TSO[s,t] - V_DSO[s,t]': 'terminal',
    'omega_s per block': 'terminal',
    'w_b (admm_block_weight)': 'terminal',
    'pi_s per (block, market scenario, hour)': 'terminal (DSO dispersion pi_by_market_by_hour; also static input)',
}


def _locate(rel, pattern):
    rx = re.compile(pattern)
    hits = []
    with open(os.path.join(REPO, rel)) as handle:
        for i, line in enumerate(handle, start=1):
            if rx.search(line.rstrip('\n')):
                hits.append({'line': i, 'text': line.strip()[:220]})
    return hits


def item5_capture_paths(check):
    facts = {fid: {'file': rel, 'pattern': pat, 'hits': _locate(rel, pat)[:8]} for fid, rel, pat in CAPTURE_FACTS}
    check('5.every_capture_anchor_located', all(f['hits'] for f in facts.values()),
          {k: len(v['hits']) for k, v in facts.items()})
    import ast

    def _literal(rel, name):
        """The literal assigned to a module-level `name`, read with ast (the module is NOT imported: the harness may
        be mid-edit by another Worker)."""
        tree = ast.parse(open(os.path.join(REPO, rel)).read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        return None
    per_cycle_fields = list(_literal('p515_s44_campaign_harness.py', 'PER_CYCLE_RESPONSE_FIELDS') or [])
    dual_fields = [list(x) for x in (_literal('interface_dual_capture.py', 'CHANNEL_FIELDS') or [])]
    check('5.per_cycle_fields_have_no_tn_side_per_scenario_interface_field',
          per_cycle_fields and not any(('tso' in f.lower() and 'interface' in f.lower()) or 'dv' in f.lower()
                                       or 'dq' in f.lower() for f in per_cycle_fields),
          {'per_cycle_response_fields': per_cycle_fields, 'dual_capture_channel_fields': dual_fields})
    # the W91 x = 0 terminal capture, sha-checked against its child manifest, re-reduced
    mpath = os.path.join(W91_X0_EVAL, 'multiscenario_terminal.json')
    man = json.load(open(os.path.join(W91_X0_EVAL, 'child_manifest_sha256.json')))
    sha = sha256_file(mpath)
    pin = man.get(mpath)
    check('5.w91_x0_terminal_capture_sha_pinned_by_child_manifest', pin == sha, None, expected=pin, observed=sha)
    m = json.load(open(mpath))
    per_dn = {}
    present = {'tso_per_scenario_vpq': True, 'dso_per_scenario_vpq': True, 'probabilities': True, 'weights': True,
               'pi_by_market_by_hour': True}
    for key, b in m['blocks'].items():
        if b['kind'] != 'TSO':
            continue
        year, day = b['year'], b['day']
        tso = b['interface_profiles']['per_scenario']
        for node, by_s in tso.items():
            dkey = f'DSO|{node}|{year}|{day}'
            db = m['blocks'].get(dkey)
            if db is None:
                present['dso_per_scenario_vpq'] = False
                continue
            dso = db['interface_profiles']['per_scenario']
            disp = db.get('dispersion') or {}
            probs = disp.get('probabilities')
            present['probabilities'] &= probs is not None
            present['weights'] &= 'admm_block_weight' in db
            present['pi_by_market_by_hour'] &= 'pi_by_market_by_hour' in disp
            acc = per_dn.setdefault(node, {q: {'max_abs': 0.0, 'ss': 0.0, 'ss_w': 0.0} for q in ('dP_mw', 'dQ_mvar',
                                                                                              'dV_pu')}
                                    | {'n_blocks': 0, 'w_sum': 0.0, 'n_hours': None})
            acc['n_blocks'] += 1
            w = db['admm_block_weight']
            acc['w_sum'] += w
            for s, prof in by_s.items():
                present['tso_per_scenario_vpq'] &= all(k in prof for k in ('v', 'p', 'q'))
                dprof = dso[s]
                omega = probs[s]
                n_h = len(prof['p'])
                acc['n_hours'] = n_h
                for q, a, bb in (('dP_mw', prof['p'], dprof['p']), ('dQ_mvar', prof['q'], dprof['q']),
                                 ('dV_pu', prof['v'], dprof['v'])):
                    for t in range(n_h):
                        x = float(a[t]) - float(bb[t])
                        acc[q]['max_abs'] = max(acc[q]['max_abs'], abs(x))
                        acc[q]['ss'] += omega * x * x
                        acc[q]['ss_w'] += w * omega * x * x
    red = {}
    for node, acc in per_dn.items():
        red[node] = {q: {'max_abs': acc[q]['max_abs'],
                         'rms': math.sqrt(acc[q]['ss'] / (acc['n_blocks'] * acc['n_hours'])),
                         'rms_blockweighted': math.sqrt(acc[q]['ss_w'] / (acc['w_sum'] * acc['n_hours']))}
                     for q in ('dP_mw', 'dQ_mvar', 'dV_pu')} | {'n_blocks': acc['n_blocks']}
    # compare with audit D
    dsha = sha256_file(AUDIT_D_JSON)
    ad = json.load(open(AUDIT_D_JSON))['results']['3x3_pair_x0']['per_dn']
    cmp_ok, cmp = True, {}
    for node, r in red.items():
        cmp[node] = {}
        for q in ('dP_mw', 'dQ_mvar', 'dV_pu'):
            for stat in ('max_abs', 'rms', 'rms_blockweighted'):
                a, b = r[q][stat], ad[node][q][stat]
                diff = abs(a - b)
                cmp[node][f'{q}.{stat}'] = {'step0': a, 'audit_D': b, 'abs_diff': diff}
                cmp_ok = cmp_ok and diff <= 1e-9 * max(abs(b), 1.0)
    check('5.terminal_path_reproduces_audit_D_per_DN_max_and_rms', cmp_ok and dsha.startswith(AUDIT_D_SHA256)
          and all(present.values()), {'present': present, 'audit_D_sha256': dsha})
    missing = [
        {'field': 'per-cycle dP_s / dQ_s / dV_s (TN side minus DN side, per scenario) and their per-DN max and '
                  'omega-RMS',
         'status': 'MISSING',
         'evidence': ('PER_CYCLE_RESPONSE_FIELDS carries only DSO-vs-own-commitment dispersion aggregates '
                      '(E|d|, sum omega d^2, max |d|) and no TN-side per-scenario value; the per-cycle sidecar of '
                      'interface_dual_capture carries the expectation-channel lambda (p, q) only')},
        {'field': 'mu_s: per-scenario consensus duals of the deviation channel (for PR = sum_b w_b sum_s omega_s sum_t '
                  '(pi_s + mu_s) dP_s)',
         'status': 'MISSING (does not exist in the current formulation; F1 creates the channel)',
         'evidence': 'interface_dual_capture.CHANNEL_FIELDS = expectation-channel lambda_pf p/q, TSO and DSO copies'},
        {'field': 'terminal max and omega-RMS per DN as written fields',
         'status': 'DERIVABLE, NOT WRITTEN: computed post hoc here from the terminal per-scenario profiles (matches '
                   'audit D); multiscenario_terminal.summary.per_dso holds DSO-vs-own-commitment dispersion, not TN-DN',
         'evidence': 'item5.reduced_from_terminal_capture'},
        {'field': 'the terminal capture for every evaluation',
         'status': ('CONDITIONAL: multiscenario_terminal.json is written only for derived-instance / premium '
                    'evaluations (capture_multiscenario); the step-1 SRP1 identity runs and any evaluation outside '
                    'that path would have no per-scenario interface capture'),
         'evidence': 'harness anchor ms_terminal_written_only_for_derived_or_premium'},
        {'field': 'priced residual PR (signed and absolute) as a field', 'status': 'MISSING (needs mu_s; pi_s, omega_s, '
                                                                                 'w_b and dP_s exist at the terminal)',
         'evidence': 'X5 of S56_PLAN_v2.md'},
    ]
    return {'anchors': facts, 'harness_sha256_at_read': sha256_file(os.path.join(REPO, 'p515_s44_campaign_harness.py')),
            'per_cycle_response_fields': per_cycle_fields, 'dual_capture_channel_fields': dual_fields,
            'terminal_capture': {'path': mpath, 'sha256': sha, 'child_manifest_pin': pin, 'fields_present': present,
                                 'required_fields_and_where': REQUIRED_FIELDS},
            'reduced_from_terminal_capture': red, 'audit_D_comparison': cmp, 'audit_D_json_sha256': dsha,
            'missing_for_step1_gate_fields': missing,
            'definitions': {'dX': 'X_TSO[s,t] - X_DSO[s,t] (TSO block per-scenario interface profile minus the DSO '
                                  'block per-scenario profile, both from production process_results_interface)',
                            'rms': 'sqrt(sum_b sum_s omega_s sum_t x^2 / (N_b * 24))',
                            'rms_blockweighted': 'sqrt(sum_b w_b sum_s omega_s sum_t x^2 / (sum_b w_b * 24))'}}


# ======================================================================================================================
#  item 6 -- grid-input twins scoping
# ======================================================================================================================
TWIN_FACTS = [
    ('network_block_seed_keyed_by_year_day', 'network_data.py',
     r"random_seed = derive_random_seed\(network_planning.random_seed, 'realization', int\(year\), str\(day\)\)"),
    ('network_pool_seed_year_free', 'network_data.py',
     r"random_seed=derive_random_seed\(network_planning.random_seed, 'synthetic_profiles'\)"),
    ('network_seed_dso', 'shared_resources_planning.py', r"^\s+'dso',$"),
    ('network_seed_tso', 'shared_resources_planning.py', r"^\s+'tso',$"),
    ('load_growth_applied', 'network.py', r'load_growth_cumul = \(1 \+ load_growth_factor\) \*\* \(network.year - initial_year\)'),
    ('flex_growth_applied', 'network.py',
     r'flexibility_growth_cumul = \(1 \+ flexibility_growth_factor\) \*\* \(network.year - initial_year\)'),
    ('pd_is_pc_plus_pc_flex', 'network.py', r'load.pd = \(pc \+ pc_flex\) / network.baseMVA'),
    ('res_pg_scaled_by_year_capacity', 'network.py', r'gen_capacity = generator.pmax \* network.baseMVA'),
    ('per_year_network_file', 'network.py', r"filename = os.path.join\(self.data_dir, self.name, f'\{self.name\}_\{self.year\}.json'\)"),
    ('market_pool_seed_year_free', 'shared_resources_planning.py',
     r"random_seed=derive_random_seed\(market_seed, 'synthetic_profiles'\)"),
    ('market_selection_keyed_by_year_day', 'shared_resources_planning.py',
     r"random_state=derive_random_seed\(market_seed, 'selection', 'energy', year, str\(day\)\)"),
    ('market_growth_applied', 'shared_resources_planning.py',
     r'energy_growth_cumul = \(1 \+ energy_growth_factor\) \*\* \(year - initial_year\)'),
]


def item6_twins(p_sp, p_w89, check):
    import p515_s56_ev_injection as EVI
    from helper_functions import derive_random_seed
    import pandas as pd
    facts = {fid: {'file': rel, 'pattern': pat, 'hits': _locate(rel, pat)[:6]} for fid, rel, pat in TWIN_FACTS}
    check('6.every_twin_anchor_located', all(f['hits'] for f in facts.values()), {k: len(v['hits']) for k, v in facts.items()})
    ya, yb = sorted(int(y) for y in p_sp.years), sorted(int(y) for y in p_w89.years)
    common = sorted(set(ya) & set(yb))
    refs_a = {EVI.key_str(k): arr for k, _kind, arr, _w, _b in EVI.array_refs(p_sp)}
    refs_b = {EVI.key_str(k): arr for k, _kind, arr, _w, _b in EVI.array_refs(p_w89)}

    def year_of(key):
        parts = key.split('|')
        return int(parts[2]) if parts[0] == 'market' else int(parts[1])
    ka = {k for k in refs_a if year_of(k) in common}
    kb = {k for k in refs_b if year_of(k) in common}
    eq = sum(1 for k in ka & kb if _bitwise(refs_a[k], refs_b[k]))
    check('6.shared_year_arrays_bitwise_identical_across_grids', common == [2025] and ka == kb and eq == len(ka) > 0,
          {'common_years': common, 'n_arrays_2025': len(ka), 'n_bitwise_equal': eq},
          expected='only 2025 shared; every 2025 array bitwise equal', observed=f'{common}; {eq}/{len(ka)}')
    # block seeds (production-set) of every network: equal for 2025, distinct otherwise
    seeds = {}
    for lab, p in (('srp1grid', p_sp), ('five_block', p_w89)):
        for h, hd in EVI.holders(p):
            for y in hd.years:
                for d in hd.days:
                    seeds[(lab, h, int(y), str(d))] = hd.network[y][d].random_seed
        mseed = derive_random_seed(p.random_seed, 'market')
        for y in p.years:
            for d in p.days:
                for kind in ('energy', 'flexibility'):
                    seeds[(lab, f'market_{kind}', int(y), str(d))] = derive_random_seed(mseed, 'selection', kind, y, str(d))
    same_2025 = all(seeds[('srp1grid', h, 2025, d)] == seeds[('five_block', h, 2025, d)]
                    for (lab, h, y, d) in seeds if lab == 'srp1grid' and y == 2025)
    vals_a = {v for (lab, h, y, d), v in seeds.items() if lab == 'srp1grid' and y != 2025}
    vals_b = {v for (lab, h, y, d), v in seeds.items() if lab == 'five_block' and y != 2025}
    check('6.draw_keys_equal_2025_disjoint_otherwise', same_2025 and not (vals_a & vals_b),
          {'n_keys_srp1grid_non2025': len(vals_a), 'n_keys_five_block_non2025': len(vals_b),
           'n_shared_non2025': len(vals_a & vals_b)})
    # growth factors (as production reads them) and per-year network-file differences
    growth = {}
    for name in ['case9', 'case33_1', 'case33_2', 'case33_3']:
        df = pd.read_excel(os.path.join(SRP1_DIR, name, f'{name}_operational_data.xlsx'), sheet_name='Growth Factors')
        growth[name] = {str(r['Growth Factor']): float(r['Value, [%]']) for _, r in df.iterrows()}
    dfm = pd.read_excel(os.path.join(SRP1_DIR, 'MarketData', 'SRP1_market_data.xlsx'), sheet_name='Growth Factors')
    growth['market'] = {str(r['Growth factors']): float(r['Value, [%]']) for _, r in dfm.iterrows()}
    file_diffs = {}
    for name in ['case9', 'case33_1', 'case33_2', 'case33_3']:
        base = json.load(open(os.path.join(SRP1_DIR, name, f'{name}_2025.json')))
        file_diffs[name] = {}
        for y in sorted(set(ya) | set(yb)):
            if y == 2025:
                continue
            other = json.load(open(os.path.join(SRP1_DIR, name, f'{name}_{y}.json')))
            keys = [k for k in sorted(set(base) | set(other)) if base.get(k) != other.get(k)]
            gens = []
            if 'generators' in keys:
                bg = {g['gen_id']: g for g in base['generators']}
                for g in other['generators']:
                    b = bg.get(g['gen_id'], {})
                    for fld in sorted(set(g) | set(b)):
                        if g.get(fld) != b.get(fld):
                            gens.append({'gen_id': g['gen_id'], 'type': g.get('type'), 'field': fld,
                                         '2025': b.get(fld), str(y): g.get(fld)})
            file_diffs[name][str(y)] = {'top_level_keys_differing': keys, 'generator_field_changes': gens}
    twin = {
        'what_a_common_base_draw_twin_needs': [
            ('per network block, the pool-row selection of the BASE key (2025, day) for every year: the block seed '
             "derive_random_seed(network_seed, 'realization', 2025, day) instead of (year, day), hence every per-load / "
             "per-generator sample (pc, qc, pc_flex, pc_flex_up, pc_flex_down, pg, qg) draws the same pool rows as 2025"),
            ("market: derive_random_seed(market_seed, 'selection', 'energy' | 'flexibility', 2025, day) for every year"),
            ('growth then applied by production with the TRUE year: loads (1 + g_load)^(year - 2025) on pc, qc; '
             '(1 + g_flex_load)^(year - 2025) on pc_flex, pc_flex_up, pc_flex_down; market (1 + g_energy)^(year - 2025) '
             'and (1 + g_flexibility)^(year - 2025); RES pg / qg scaled by the YEAR network file capacity (Pmax), no '
             'growth factor'),
            ('because pd = (pc + pc_flex) / baseMVA mixes two growth rates (g_load != g_flex_load on every network), the '
             'twin cannot be made by rescaling the final pd arrays post read: it needs the re-keyed draw at generation '
             '(a pass-through wrapper on derive_random_seed in network_data / shared_resources_planning mapping the '
             "'realization' and 'selection' labels' year to 2025, or an equivalent pre-checksum injection of arrays "
             'computed from the 2025 draws with separate pc / pc_flex growth)'),
            ('the synthetic pools (KDE, 100 samples per day) are already year-free and shared by both grids: their '
             "seeds derive from (network seed, 'synthetic_profiles') and (market seed, 'synthetic_profiles') only"),
            ('the twin changes the scenario checksum; the prefix / EV / checksum pins would be new artefacts '
             '(a scenario-construction decision, A5)')],
        'shared_draws_between_grids': ('2025 only (draw keys equal and arrays bitwise equal); every other year uses '
                                       'distinct (year, day) keys; the year-free KDE pools are common to both grids'),
    }
    return {'years_srp1grid': ya, 'years_five_block': yb, 'common_years': common, 'anchors': facts,
            'growth_factors': growth, 'network_file_differences_vs_2025': file_diffs, 'twin_scoping': twin,
            'n_2025_arrays_compared': len(ka), 'n_2025_arrays_bitwise_equal': eq}


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    global OUT
    if '--manifest' in sys.argv:
        return manifest()
    if '--rehearsal' in sys.argv:  # debugging only: a directory OUTSIDE the repository; never the evidence run
        OUT = os.path.abspath(sys.argv[sys.argv.index('--rehearsal') + 1])
        if OUT.startswith(REPO + os.sep):
            print('REFUSE: --rehearsal must point outside the repository', flush=True)
            return 2
    if not _preflight():
        return 2
    import gate_result_io as GRIO
    started = _utc()
    prov_start = _provenance()
    with open(os.path.join(OUT, PREDICTIONS_FILE), 'w') as handle:
        handle.write(GRIO.dumps({'schema': SCHEMA + '_predictions', 'utc': started,
                                 'recorded_before_any_computation': True, 'predictions': PREDICTIONS,
                                 'git_head': prov_start['head'], 'script_sha256': sha256_file(__file__)},
                                indent=1))
    print(f'[predictions] written {PREDICTIONS_FILE} at {started}', flush=True)
    checks = []

    def check(cid, ok, detail, expected=None, observed=None):
        ok = bool(ok)
        checks.append({'id': cid, 'pass': ok, 'expected': expected, 'observed': observed, 'detail': detail})
        print(f"[check] {'PASS' if ok else 'FAIL'} {cid}", flush=True)
        return ok

    result = {'schema': SCHEMA, 'script': SCRIPT, 'script_sha256': sha256_file(__file__), 'started_utc': started,
              'interpreter': sys.executable, 'provenance_start': prov_start, 'predictions_file': PREDICTIONS_FILE,
              'solve_claim': 'zero solves: SolveProfileGuard(permitted=()) installed before any repository import, '
                             'verified at 0; pickle.load/loads blocked and counted'}
    errors = {}
    t0 = time.time()
    plannings = {}

    def run(name, fn):
        try:
            t = time.time()
            res = fn()
            print(f'[{name}] done in {time.time() - t:.1f} s', flush=True)
            return res
        except Exception:
            errors[name] = traceback.format_exc()
            print(errors[name], file=sys.stderr, flush=True)
            return None

    reads = {}
    for label in ('srp1grid_3x3', 'srp1grid_3x3_ev', 'five_block_3x3_w89'):
        r = run(f'read_{label}', lambda label=label: production_read(label, check))
        if r is not None:
            plannings[label], reads[label] = r
    result['reads'] = reads
    result['item1_identity'] = {label: run(f'item1_{label}', lambda label=label: item1_identity(p, label, check))
                                for label, p in plannings.items()}
    tv = {}
    for grid_label, read_label in (('srp1grid', 'srp1grid_3x3'), ('five_block', 'five_block_3x3_w89')):
        if read_label in plannings:
            tv[grid_label] = run(f'item2_{grid_label}', lambda gl=grid_label, rl=read_label:
                                 item2_terminal_value(plannings[rl], gl, check))
    result['item2_terminal_value'] = tv

    def _item3():
        lat, inst, plans = enumerate_staged(check)
        stats, worst, examples = item3_derating(lat, plans, check)
        reps, members = pick_representatives(lat, set(plans), worst, examples)
        check('3.representative_plans_are_admissible_members', all(members.values()), members)
        built = verify_on_built_models(plannings['srp1grid_3x3'].shared_ess_data, lat, reps, check)
        sed = plannings['srp1grid_3x3'].shared_ess_data
        phi = sed.shared_energy_storages[list(sed.years)[0]][0].phi_cal
        soc = {'srp1grid': soc_excursion(list(lat.years), [sed.years[y] for y in sed.years], phi, plans=plans,
                                         lat=lat)}
        if 'five_block_3x3_w89' in plannings:
            sb = plannings['five_block_3x3_w89'].shared_ess_data
            soc['five_block'] = soc_excursion([int(y) for y in sb.years], [sb.years[y] for y in sb.years], phi)
        soc['derivation'] = soc_excursion.__doc__
        e5 = soc['srp1grid']['by_age_gap']['5y']['excess_equal_E_pct']
        e10 = soc['srp1grid']['by_age_gap']['10y']['excess_equal_E_pct']
        check('3.soc_excursion_equal_E_matches_advisor_4_and_8_pct', abs(e5 - 4.0) <= 0.5 and abs(e10 - 8.0) <= 0.5,
              None, expected='about 4 % (5 y), 8 % (10 y)', observed=f'{e5:.2f} %, {e10:.2f} %')
        return {'instance': inst.describe(), 'n_plans_incl_x0': len(plans), 'derating': stats,
                'representatives': {k: list(z) for k, z in reps.items()}, 'built_model_verification': built,
                'soc_excursion': soc}
    if 'srp1grid_3x3' in plannings:
        result['item3_h3_derating'] = run('item3', _item3)
    if tv and all(v is not None for v in tv.values()):
        result['item4_repricing'] = run('item4', lambda: item4_repricing(tv, check))
    result['item5_capture_paths'] = run('item5', lambda: item5_capture_paths(check))
    if 'srp1grid_3x3' in plannings and 'five_block_3x3_w89' in plannings:
        result['item6_twins'] = run('item6', lambda: item6_twins(plannings['srp1grid_3x3'],
                                                                 plannings['five_block_3x3_w89'], check))
    prov_end = _provenance()
    check('0.production_sources_unchanged_during_run', prov_end['sources_sha256'] == prov_start['sources_sha256'],
          {k: (prov_start['sources_sha256'][k], prov_end['sources_sha256'][k]) for k in PRODUCTION_FILES
           if prov_start['sources_sha256'][k] != prov_end['sources_sha256'][k]})
    guard_fail = GUARD.verify(0)
    check('0.guard_verify_0', not guard_fail, guard_fail)
    check('0.pickle_loads_0', PICKLE_COUNTS == {'load': 0, 'loads': 0}, dict(PICKLE_COUNTS))
    result.update({'finished_utc': _utc(), 'wall_s': time.time() - t0, 'provenance_end': prov_end, 'errors': errors,
                   'checks': checks, 'failed_checks': [c['id'] for c in checks if not c['pass']],
                   'all_checks_ok': bool(all(c['pass'] for c in checks) and not errors),
                   'guard': {'label': GUARD.label, 'counts': dict(GUARD.counts), 'verify_0_failures': guard_fail},
                   'pickle_counts': dict(PICKLE_COUNTS)})
    with open(os.path.join(OUT, RESULT_FILE), 'w') as handle:
        handle.write(GRIO.dumps(result, indent=1))
    print(f"[done] checks {sum(c['pass'] for c in checks)}/{len(checks)}; errors {sorted(errors)}; "
          f"guard {dict(GUARD.counts)} failures {guard_fail}; pickle {PICKLE_COUNTS}", flush=True)
    return 0 if result['all_checks_ok'] and not guard_fail else 1


if __name__ == '__main__':
    sys.exit(main())
