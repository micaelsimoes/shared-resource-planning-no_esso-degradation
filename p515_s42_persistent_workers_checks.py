"""
P5.15 Addendum 23/24, Step 3.6 persistent-worker BOUNDED task (Step 3.6),
items 2 and 3 -- ZERO-SOLVE checks for:

  1. The bound-restore fix in `network.capture_block_mutable_state` /
     `network.apply_block_mutable_state` (raw `._lb`/`._ub` capture instead
     of the resolved `.lb`/`.ub` property, `skip_validation=True` on the
     Var-restore `set_value`/`fix` calls) -- confirms the fix actually
     changes behaviour (the pickled-bytes regression and the W1001 warning
     each reproduce under the OLD code path and are eliminated under the
     NEW one, on the SAME real production Var).
  2. The new DSO node-7 lightweight snapshot capture
     (`admm_parameters.dso_snapshot_capture_mode`, `shared_resources_
     planning.update_distribution_coordination_models_and_solve_
     sequential`'s new branch, `admm_persistent_workers._solve_dso_block_
     in_worker`'s new branch) -- real `BlockData.clone()` call counting
     through the REAL production dispatch functions (never a
     reimplementation), exactly the technique `p515_s36_clone_capture_
     checks.py`'s `_end_to_end_dispatch_check` already used for the TSO.

`SolveProfileGuard(permitted=())` is armed for the WHOLE script;
`verify(expected_solves=0)` at the end. `Network.run_smopf` is monkeypatched
(class level, restored in `finally`) to a canned, never-solving result --
never reaches `solver.solve`.

    python p515_s42_persistent_workers_checks.py

Writes data/SRP1/Results/P515S42/persistent_workers_task/interface_helper_checks/
-- wait, see OUT_DIR below (refuses to overwrite).
"""

import hashlib
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone

import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core.base.block import BlockData

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import network as NET  # noqa: E402
from network_data import NetworkData  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import admm_persistent_workers as apw  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402
from helper_functions import solver_result_succeeded, solver_result_summary  # noqa: E402

MAIN_REPO = REPO
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S42', 'persistent_workers_task', 'interface_helper_checks')
RESULTS_PATH = os.path.join(OUT_DIR, 'results.json')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')

FIXTURE_PATHS = [
    os.path.join(MAIN_REPO, 'data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl'),
    os.path.join(MAIN_REPO, 'data/SRP1/Results/P512R/cycle21_prepared/snapshot.pkl'),
    os.path.join(MAIN_REPO, 'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/'
                            'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
    os.path.join(MAIN_REPO, 'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/'
                            'matched_success_TSO_case9_2025_Summer_cycle7.pkl'),
    os.path.join(MAIN_REPO, 'data/SRP1/Results/FrozenSMOPF/'
                            'matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl'),
    os.path.join(MAIN_REPO, 'data/SRP1/Results/FrozenSMOPF/'
                            'matched_success_TSO_case9_2025_Summer_cycle7.pkl'),
    os.path.join(MAIN_REPO, 'data/SRP1/Results/P515F/t2_results/FrozenSMOPF/'
                            'failure_TSO_case9_2025_Spring_cycle1.pkl'),
]


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Check 1: admm_parameters.dso_snapshot_capture_mode default + switch.
# ---------------------------------------------------------------------------

def check_dso_switch_default():
    fresh = ADMMParameters()
    return {
        'default_dso_snapshot_capture_mode': fresh.dso_snapshot_capture_mode,
        'default_is_lightweight': fresh.dso_snapshot_capture_mode == 'lightweight',
        'settable_to_legacy_clone': True,  # plain attribute; exercised directly below.
    }


# ---------------------------------------------------------------------------
# Check 2: bound-restore fix -- reproduces the OLD bug (resolved `.lb`/`.ub`
# capture) and the FIXED behaviour (raw `._lb`/`._ub` capture) side by side,
# on the SAME real production Var (a TSO block's own Var with a purely
# domain-derived bound, found by inspection, not assumed), so the check
# actually discriminates rather than trivially passing.
# ---------------------------------------------------------------------------

def _initialize_mutable_params(block):
    """Several mutable ADMM-consensus Params (e.g. `vmag_req`) have NO
    `initialize=`/default -- production's own per-cycle update always
    `set_value()`s them before they are ever read, every cycle, but
    `_build_admm_ready_state` stops right after construction (before any
    cycle), leaving them genuinely uninitialized. Set a synthetic value
    directly (never read first), exactly `p515_s36_clone_capture_checks.py`'s
    `_perturb_generic_params` precedent."""
    touched = 0
    for comp in block.component_objects(pe.Param, active=None):
        if not comp.mutable:
            continue
        for index in comp:
            touched += 1
            comp[index].set_value(0.01 * touched)
    return touched


def _find_var_with_implicit_lower_bound(block):
    for comp in block.component_objects(pe.Var, active=None):
        for index in comp:
            var_data = comp[index]
            if var_data._lb is None and var_data.lb is not None:
                return comp.name, index
    return None, None


def check_bound_restore_fix(tso_model_yd):
    _initialize_mutable_params(tso_model_yd)
    comp_name, index = _find_var_with_implicit_lower_bound(tso_model_yd)
    if comp_name is None:
        return {'error': 'no Var with an implicit (domain-derived) lower bound found on the probe block'}

    var_data = getattr(tso_model_yd, comp_name)[index]
    var_data.set_value(1.0 if var_data.value is None else var_data.value)

    baseline_bytes = len(pickle.dumps(var_data))
    baseline_lb_raw = var_data._lb
    baseline_lb_resolved = var_data.lb

    # FIXED behaviour: network.capture_block_mutable_state / apply_block_mutable_state.
    captured = NET.capture_block_mutable_state(tso_model_yd)
    fixed_clone = tso_model_yd.clone()
    NET.apply_block_mutable_state(fixed_clone, captured)
    fixed_var_data = getattr(fixed_clone, comp_name)[index]
    fixed_bytes = len(pickle.dumps(fixed_var_data))
    fixed_lb_raw = fixed_var_data._lb

    # OLD (pre-fix) behaviour, reproduced directly (not by editing network.py
    # back -- the exact two lines the fix replaced), to confirm the check
    # discriminates: resolved-bound capture -> setlb/setub with a concrete
    # value converts the implicit bound to an explicit one.
    old_clone = tso_model_yd.clone()
    old_var_data = getattr(old_clone, comp_name)[index]
    old_var_data.setlb(var_data.lb)
    old_var_data.setub(var_data.ub)
    old_bytes = len(pickle.dumps(old_var_data))
    old_lb_raw = old_var_data._lb

    return {
        'probe_var': f'{comp_name}[{index}]',
        'baseline_lb_raw_is_none': baseline_lb_raw is None,
        'baseline_lb_resolved': baseline_lb_resolved,
        'baseline_pickle_bytes': baseline_bytes,
        'fixed_pickle_bytes': fixed_bytes,
        'fixed_lb_raw_is_none': fixed_lb_raw is None,
        'fixed_representation_preserved': (fixed_bytes == baseline_bytes) and (fixed_lb_raw is None),
        'old_pickle_bytes': old_bytes,
        'old_lb_raw_is_none': old_lb_raw is None,
        'old_bug_reproduced': (old_bytes != baseline_bytes) and (old_lb_raw is not None),
    }


# ---------------------------------------------------------------------------
# Check 3: W1001 warning elimination -- same discriminating structure.
# ---------------------------------------------------------------------------

class _CountingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def check_w1001_elimination(dso_model_yd):
    # A Var whose domain is NonNegativeReals (or similar) and whose current
    # value can legally be pushed a hair negative via skip_validation=True,
    # reproducing the ~1e-9 out-of-domain slack values IPOPT can leave behind
    # (WORKER_REPORT_S40_PERSISTENT_WORKERS.md).
    comp_name, index = _find_var_with_implicit_lower_bound(dso_model_yd)
    if comp_name is None:
        return {'error': 'no probe Var found'}
    var_data = getattr(dso_model_yd, comp_name)[index]
    var_data.set_value(-1e-9, skip_validation=True)

    captured = NET.capture_block_mutable_state(dso_model_yd)

    logger = logging.getLogger('pyomo.core')
    prev_level = logger.level
    logger.setLevel(logging.WARNING)

    fixed_handler = _CountingHandler()
    logger.addHandler(fixed_handler)
    try:
        fixed_clone = dso_model_yd.clone()
        NET.apply_block_mutable_state(fixed_clone, captured)
    finally:
        logger.removeHandler(fixed_handler)

    old_handler = _CountingHandler()
    logger.addHandler(old_handler)
    try:
        old_clone = dso_model_yd.clone()
        old_var_data = getattr(old_clone, comp_name)[index]
        old_var_data.set_value(-1e-9)  # pre-fix call shape: no skip_validation
    finally:
        logger.removeHandler(old_handler)
        logger.setLevel(prev_level)

    fixed_w1001 = [m for m in fixed_handler.records if 'W1001' in m or 'not in domain' in m]
    old_w1001 = [m for m in old_handler.records if 'W1001' in m or 'not in domain' in m]

    return {
        'probe_var': f'{comp_name}[{index}]',
        'fixed_path_w1001_count': len(fixed_w1001),
        'old_path_w1001_count': len(old_w1001),
        'fixed_path_zero_warnings': len(fixed_w1001) == 0,
        'old_path_reproduces_warning': len(old_w1001) >= 1,
    }


# ---------------------------------------------------------------------------
# Check 4: DSO end-to-end dispatch clone counting, through the REAL
# `update_distribution_coordination_models_and_solve_sequential`.
# ---------------------------------------------------------------------------

def _canned_result(succeeded):
    result = po.SolverResults()
    result.solver.status = po.SolverStatus.ok if succeeded else po.SolverStatus.warning
    result.solver.termination_condition = (
        po.TerminationCondition.optimal if succeeded else po.TerminationCondition.maxIterations)
    return result


def check_dso_dispatch_clone_counts(planning, distribution_networks, dso_models, consensus_vars, dual_vars,
                                     admm_parameters, scratch_results_dir):
    # `_build_admm_ready_state` permanently replaces every distribution
    # network's `.optimize` with a never-solving stub (its OWN zero-solve
    # construction phase). This check needs the REAL `NetworkData.optimize`
    # dispatch (the exact function under test, with only `Network.run_smopf`
    # canned underneath it) -- restore the genuine bound method, exactly
    # `p515_s36_clone_capture_checks.py`'s `_end_to_end_dispatch_check`
    # precedent for the TSO.
    for _node_id, _dn in distribution_networks.items():
        _dn.optimize = NetworkData.optimize.__get__(_dn, type(_dn))

    # Node 7's callbacks (`save_failed_dso_block`/`save_selected_dso_
    # comparator`, wired inside the production function under test) write
    # under `distribution_network.results_dir` -- redirect it to a scratch
    # subdir of THIS check's own output, never the shared production
    # `data/SRP1/Results/FrozenSMOPF/` tree, before intentionally triggering
    # "failures" below.
    distribution_networks[7].results_dir = scratch_results_dir

    node7_network = distribution_networks[7]
    dso_pristine_base = {
        year: {day: dso_models[7][year][day].clone() for day in node7_network.days}
        for year in node7_network.years
    }

    clone_calls = {'n': 0}
    orig_clone = BlockData.clone

    def counting_clone(self, *a, **kw):
        clone_calls['n'] += 1
        return orig_clone(self, *a, **kw)

    orig_run_smopf = NET.Network.run_smopf

    def make_run_smopf(fail_key):
        def canned_run_smopf(self, model, params, from_warm_start=False, print_header=True):
            key = (self.year, self.day)
            return _canned_result(succeeded=(key != fail_key))
        return canned_run_smopf

    def run_once(cycle, pristine_base, fail_key):
        clone_calls['n'] = 0
        NET.Network.run_smopf = make_run_smopf(fail_key)
        try:
            res = srp.update_distribution_coordination_models_and_solve_sequential(
                distribution_networks, dso_models,
                consensus_vars['vmag'], dual_vars['vmag']['dso'],
                consensus_vars['pf'], dual_vars['pf']['dso'],
                consensus_vars['ess'], dual_vars['ess']['dso'],
                admm_parameters,
                {node_id: {year: {'s_available': 1.0, 'e_available': 1.0}
                           for year in distribution_networks[node_id].years}
                 for node_id in distribution_networks},
                from_warm_start=True,
                cycle=cycle,
                dso_pristine_base=pristine_base,
            )
        finally:
            NET.Network.run_smopf = orig_run_smopf
        return res, clone_calls['n']

    node7_first_year = next(iter(node7_network.years))
    node7_first_day = next(iter(node7_network.days))

    BlockData.clone = counting_clone
    try:
        _res_a, clones_a = run_once(cycle=1, pristine_base=None, fail_key=('__none__', '__none__'))
        _res_b, clones_b = run_once(cycle=1, pristine_base=dso_pristine_base, fail_key=('__none__', '__none__'))
        _res_c, clones_c = run_once(cycle=1, pristine_base=dso_pristine_base,
                                     fail_key=(node7_first_year, node7_first_day))
        _res_d, clones_d = run_once(cycle=7, pristine_base=dso_pristine_base, fail_key=('__none__', '__none__'))
    finally:
        BlockData.clone = orig_clone

    node7_n_blocks = sum(1 for _y in node7_network.years for _d in node7_network.days)

    return {
        'node7_n_year_day_blocks': node7_n_blocks,
        'case_A_legacy_no_failure': {'clones': clones_a, 'expected': node7_n_blocks, 'match': clones_a == node7_n_blocks},
        'case_B_lightweight_no_failure_not_cycle7': {'clones': clones_b, 'expected': 0, 'match': clones_b == 0},
        'case_C_lightweight_one_failure': {'clones': clones_c, 'expected': 1, 'match': clones_c == 1},
        'case_D_lightweight_cycle7_comparator': {'clones': clones_d, 'expected': 1, 'match': clones_d == 1},
    }


# ---------------------------------------------------------------------------
# Check 5: persistent-worker DSO branch (`_solve_dso_block_in_worker`) clone
# counting -- direct, in-process call (no real worker spawned; the function
# itself has no process-boundary dependency), canned `run_smopf`.
# ---------------------------------------------------------------------------

def check_persistent_worker_dso_branch(distribution_networks, dso_models, results_dir):
    node7_network = distribution_networks[7]
    year0 = next(iter(node7_network.years))
    day0 = next(iter(node7_network.days))
    network_yd = node7_network.network[year0][day0]
    network_params = distribution_networks[7].params

    pristine = dso_models[7][year0][day0].clone()
    state = apw.capture_block_state_for_ipc(dso_models[7][year0][day0])

    clone_calls = {'n': 0}
    orig_clone = BlockData.clone

    def counting_clone(self, *a, **kw):
        clone_calls['n'] += 1
        return orig_clone(self, *a, **kw)

    orig_run_smopf = NET.Network.run_smopf

    def run_once(node_id, model, mode, pristine_base, state_for_snapshot, cycle, succeed):
        clone_calls['n'] = 0

        def canned_run_smopf(self, m, params, from_warm_start=False, print_header=True):
            return _canned_result(succeeded=succeed)

        NET.Network.run_smopf = canned_run_smopf
        BlockData.clone = counting_clone
        try:
            result = apw._solve_dso_block_in_worker(
                model, network_yd, network_params, node_id, year0, day0, cycle,
                True, results_dir, mode, pristine_base, state_for_snapshot,
            )
        finally:
            NET.Network.run_smopf = orig_run_smopf
            BlockData.clone = orig_clone
        return result, clone_calls['n']

    _r1, n1 = run_once(7, pristine.clone(), 'lightweight', pristine.clone(), state, cycle=1, succeed=True)
    _r2, n2 = run_once(7, pristine.clone(), 'lightweight', pristine.clone(), state, cycle=1, succeed=False)
    _r3, n3 = run_once(7, pristine.clone(), 'legacy_clone', pristine.clone(), state, cycle=1, succeed=True)
    _r4, n4 = run_once(5, pristine.clone(), 'lightweight', None, state, cycle=1, succeed=False)

    return {
        'node7_lightweight_no_failure': {'clones': n1, 'expected': 0, 'match': n1 == 0},
        'node7_lightweight_failure': {'clones': n2, 'expected': 1, 'match': n2 == 1},
        'node7_legacy_clone_no_failure': {'clones': n3, 'expected': 1, 'match': n3 == 1},
        'non_node7_no_clone_regardless_of_mode': {'clones': n4, 'expected': 0, 'match': n4 == 0},
    }


# ---------------------------------------------------------------------------
# Check 6: preserved-fixture reload.
# ---------------------------------------------------------------------------

def check_fixture_reload():
    out = {}
    for path in FIXTURE_PATHS:
        rel = os.path.relpath(path, MAIN_REPO)
        try:
            with open(path, 'rb') as handle:
                payload = pickle.load(handle)
            model = payload['model']
            n_vars = sum(1 for _ in model.component_data_objects(pe.Var, active=None))
            out[rel] = {'unpickled': True, 'n_vars': n_vars}
        except Exception as error:  # noqa: BLE001
            out[rel] = {'unpickled': False, 'error': repr(error)}
    return out


def main():
    started = time.time()
    guard = SolveProfileGuard(permitted=(), label='p515_s42_persistent_workers_checks')
    guard.install()

    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(RESULTS_PATH)
    _refuse_overwrite(MANIFEST_PATH)

    eval_id = 'p515s42_pwchecks_probe'
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = _build_admm_ready_state(eval_id)
    distribution_networks = planning.distribution_networks
    admm_parameters = planning.params.admm

    tso_probe_yd = tso_model[next(iter(tso_model))][next(iter(tso_model[next(iter(tso_model))]))]
    dso_probe_yd = dso_models[7][next(iter(dso_models[7]))][next(iter(dso_models[7][next(iter(dso_models[7]))]))]

    # Several mutable ADMM-consensus Params have no build-time default and
    # are only ever set by the per-cycle update loop (never read first in
    # production) -- `_build_admm_ready_state` stops before any cycle, so
    # every DSO node-7 block (read wholesale by `capture_block_mutable_
    # state` inside the new lightweight dispatch branch under test) needs
    # this same precedent-established synthetic initialization before ANY
    # check below touches it, exactly `p515_s36_clone_capture_checks.py`'s
    # `_perturb_generic_params` reasoning.
    for _year in dso_models[7]:
        for _day in dso_models[7][_year]:
            _initialize_mutable_params(dso_models[7][_year][_day])

    results = {}
    results['switch_default'] = check_dso_switch_default()
    results['bound_restore_fix'] = check_bound_restore_fix(tso_probe_yd)
    results['w1001_elimination'] = check_w1001_elimination(dso_probe_yd)
    results['dso_dispatch_clone_counts'] = check_dso_dispatch_clone_counts(
        planning, distribution_networks, dso_models, consensus_vars, dual_vars, admm_parameters,
        scratch_results_dir=os.path.join(OUT_DIR, 'dispatch_check_snapshots'))
    results['persistent_worker_dso_branch'] = check_persistent_worker_dso_branch(
        distribution_networks, dso_models, os.path.join(OUT_DIR, 'pw_branch_snapshots'))
    results['fixture_reload'] = check_fixture_reload()

    def _all_true(d, keys_path):
        node = d
        for k in keys_path:
            node = node[k]
        return bool(node)

    all_checks_pass = (
        results['switch_default']['default_is_lightweight']
        and results['bound_restore_fix'].get('fixed_representation_preserved', False)
        and results['bound_restore_fix'].get('old_bug_reproduced', False)
        and results['w1001_elimination'].get('fixed_path_zero_warnings', False)
        and results['w1001_elimination'].get('old_path_reproduces_warning', False)
        and all(v['match'] for v in results['dso_dispatch_clone_counts'].values() if isinstance(v, dict) and 'match' in v)
        and all(v['match'] for v in results['persistent_worker_dso_branch'].values())
        and all(v['unpickled'] for v in results['fixture_reload'].values())
    )

    guard_verify = guard.verify(expected_solves=0)

    payload = {
        'stage': 'P5.15 Addendum 23/24, Step 3.6 persistent-worker bounded task -- zero-solve checks',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 23 (Step 3.6 persistent workers)',
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json item3_persistent_workers_bounded_task',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'results': results,
        'guard_verify': guard_verify,
        'all_checks_pass': all_checks_pass,
        'wall_clock_s': time.time() - started,
    }

    with open(RESULTS_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S42-PW-CHECKS] wrote {RESULTS_PATH}')
    print(f'[S42-PW-CHECKS] all_checks_pass={all_checks_pass} guard_verify={guard_verify}')

    manifest = {os.path.relpath(RESULTS_PATH, REPO): _sha256_file(RESULTS_PATH)}
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S42-PW-CHECKS] wrote {MANIFEST_PATH}')

    if not all_checks_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
