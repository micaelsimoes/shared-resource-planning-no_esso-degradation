"""
P5.15 Step 3.6 (worker task, WORKER_REPORT_S36_CLONE_CAPTURE.md) -- zero-solve
equivalence checks for the lightweight TSO/DSO snapshot capture that replaces
the per-cycle `model.clone()` (PLANNER_BRIEF_2026-09-13.md Addendum 21 item 4).

Verifies, on a FRESH, zero-solve, production-constructed ADMM-ready state
(`p515_s32_zero_solve_checks._build_admm_ready_state`, reused verbatim -- not
reimplemented), that:

  1. `network.capture_block_mutable_state` / `network.apply_block_mutable_state`
     reproduce a real `model.clone()` bit-for-bit (every Var value/bounds/fixed
     flag, every mutable Param value, every Suffix entry, active
     constraint/objective sets, expression string representations), on BOTH a
     TSO block and a DSO node-7 block, after a representative ADMM-cycle
     mutation (generic Param/Var perturbation PLUS a real
     `configure_shared_ess_operational_state` active->inactive transition, to
     exercise the Var fix/unfix/bound/deactivate paths, not just `set_value`).
  2. End-to-end: `shared_resources_planning.update_transmission_coordination_
     model_and_solve`'s lightweight path (default,
     `admm_parameters.tso_snapshot_capture_mode == 'lightweight'`) reaches
     `BlockData.clone()` ZERO times when no snapshot needs writing, and
     EXACTLY once per block that actually needs one (one simulated failure);
     the legacy path (`tso_snapshot_capture_mode = 'legacy_clone'`) reaches it
     once per (year, day) block every cycle, unchanged from pre-Step-3.6
     behaviour -- verified by counting real `BlockData.clone` invocations
     (patched at the class level, calls through to the original), not
     inferred from timing.
  3. Timing: per-block medians for the legacy clone, the lightweight capture
     (paid every cycle), and the on-demand rebuild (clone-of-pristine +
     apply, paid only when a snapshot is actually written).
  4. Every preserved FrozenSMOPF/cycle21 fixture pickle this task's report
     lists still unpickles, read from the MAIN checkout's absolute `data/`
     paths (read-only; this worktree's own `data/` is untracked/absent for
     these fixtures).

`SolveProfileGuard(permitted=())` is armed for the WHOLE script; `verify(
expected_solves=0)` at the end. `Network.run_smopf` is monkeypatched (class
level, restored in `finally`) to a canned, never-failing result for check 2's
"no snapshot needed" case and to a canned, always-failing result for its "one
simulated failure" case -- neither ever reaches `solver.solve`, so the guard
never needs to permit anything; it is the backstop, not the mechanism.

    python p515_s36_clone_capture_checks.py

Writes data/SRP1/Results/P515S36/clone_capture_checks/{results.json,
manifest_sha256.json} (both NEW files; refuses to overwrite).
"""

import hashlib
import json
import os
import statistics
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
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402

# Main checkout: where the preserved fixtures actually live (untracked
# data/, absent from this detached worktree). Read-only; never written.
MAIN_REPO = '/Users/micaelsimoes/Projects/share-resource-planning-no_esso-degradation'

# Optional run suffix (argv[1]): redirects OUT_DIR so a re-run (e.g. P5.15
# Addendum 23/24, Step 3.6 persistent-worker bounded task item 2's TSO-
# capture regression re-check after the `network.apply_block_mutable_state`
# bound-restore/W1001 fix) never overwrites the original committed evidence
# (CLAUDE.md: never re-run a harness onto a cited artifact). Default
# behaviour (no suffix) is unchanged.
_RUN_SUFFIX = ''
for _arg in sys.argv[1:]:
    if not _arg.startswith('--'):
        _RUN_SUFFIX = '_' + _arg.strip('_')
        break
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S36', f'clone_capture_checks{_RUN_SUFFIX}')
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

N_TIMING_REPS = 15


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


# ---------------------------------------------------------------------------
# Representative per-cycle mutation, applied directly to an already-built,
# never-solved block -- generic (works for both the TSO and a DSO block: both
# expose the same Pyomo component types `capture_block_mutable_state` reads).
# ---------------------------------------------------------------------------

def _perturb_generic_params(block):
    # Several mutable ADMM-consensus Params (e.g. `vmag_req`, `p_pf_req`) are
    # declared with NO `initialize=`/default -- production's own per-cycle
    # update ALWAYS `set_value()`s them before they are ever read, every
    # cycle, so `_build_admm_ready_state` (which stops right after
    # construction, before any cycle) leaves them genuinely uninitialized;
    # reading `.value` first (as a real cycle update never does) raises. Set
    # a representative synthetic value directly, without reading first.
    touched = 0
    for comp in block.component_objects(pe.Param, active=None):
        if not comp.mutable:
            continue
        for index in comp:
            touched += 1
            comp[index].set_value(0.01 * touched)
    return touched


def _perturb_generic_vars(block, exclude_names):
    touched = 0
    for comp in block.component_objects(pe.Var, active=None):
        if comp.name in exclude_names:
            continue
        for index in comp:
            var_data = comp[index]
            if var_data.fixed:
                continue
            try:
                current = var_data.value or 0.0
            except (ValueError, TypeError):
                current = 0.0
            touched += 1
            new_value = abs(current) * 1.001 + 1e-4 * touched
            lb, ub = var_data.lb, var_data.ub
            if lb is not None and new_value < lb:
                new_value = lb
            if ub is not None and new_value > ub:
                new_value = ub
            var_data.set_value(new_value, skip_validation=True)
    return touched


def _exercise_shared_ess_transition(block):
    """Real production mutation (not a generic perturbation): drives ONE
    shared-ESS index through `configure_shared_ess_operational_state`'s
    active -> inactive branch (fix-to-zero + deactivate), the exact function
    the TSO's per-cycle update calls -- exercises the Var
    fix/unfix/setlb/setub and Constraint activate/deactivate mutations a
    generic value-only perturbation never touches."""
    import model_construction_helpers as mch
    shared_ess_idx = next(iter(block.shared_energy_storages))
    mch.configure_shared_ess_operational_state(block, shared_ess_idx, 0.0, 0.0)
    return shared_ess_idx


def _set_synthetic_warm_start_suffixes(block):
    n_set = {'zL': 0, 'zU': 0, 'dual': 0}
    for comp in block.component_objects(pe.Var, active=None):
        for index in comp:
            if n_set['zL'] >= 5:
                break
            block.ipopt_zL_in[comp[index]] = 0.1234
            block.ipopt_zU_in[comp[index]] = 0.5678
            n_set['zL'] += 1
            n_set['zU'] += 1
        if n_set['zL'] >= 5:
            break
    for comp in block.component_objects(pe.Constraint, active=None):
        for index in comp:
            if n_set['dual'] >= 5:
                break
            block.dual[comp[index]] = -2.5
            n_set['dual'] += 1
        if n_set['dual'] >= 5:
            break
    return n_set


def _apply_representative_cycle_mutation(block):
    exercised_idx = _exercise_shared_ess_transition(block)
    n_params = _perturb_generic_params(block)
    n_vars = _perturb_generic_vars(
        block, exclude_names={'shared_es_s_rated_fixed', 'shared_es_e_rated_fixed'})
    suffix_counts = _set_synthetic_warm_start_suffixes(block)
    return {
        'exercised_shared_ess_idx': exercised_idx,
        'n_params_perturbed': n_params,
        'n_vars_perturbed': n_vars,
        'suffix_counts': suffix_counts,
    }


# ---------------------------------------------------------------------------
# Bit-for-bit comparison of two structurally-identical blocks.
# ---------------------------------------------------------------------------

def _suffix_index(suffix):
    out = {}
    for comp_data, value in suffix.items():
        parent = comp_data.parent_component()
        out[(parent.name, comp_data.index())] = value
    return out


def _diff_blocks(a, b):
    diffs = []

    a_var_names = {c.name for c in a.component_objects(pe.Var, active=None)}
    b_var_names = {c.name for c in b.component_objects(pe.Var, active=None)}
    if a_var_names != b_var_names:
        diffs.append(f'Var component name sets differ: {a_var_names ^ b_var_names}')
    for name in sorted(a_var_names & b_var_names):
        ca, cb = getattr(a, name), getattr(b, name)
        for index in ca:
            da, db = ca[index], cb[index]
            if da.value != db.value:
                diffs.append(f'Var {name}[{index}].value: {da.value!r} != {db.value!r}')
            if da.lb != db.lb:
                diffs.append(f'Var {name}[{index}].lb: {da.lb!r} != {db.lb!r}')
            if da.ub != db.ub:
                diffs.append(f'Var {name}[{index}].ub: {da.ub!r} != {db.ub!r}')
            if bool(da.fixed) != bool(db.fixed):
                diffs.append(f'Var {name}[{index}].fixed: {da.fixed!r} != {db.fixed!r}')

    a_param_names = {c.name for c in a.component_objects(pe.Param, active=None) if c.mutable}
    b_param_names = {c.name for c in b.component_objects(pe.Param, active=None) if c.mutable}
    if a_param_names != b_param_names:
        diffs.append(f'mutable Param component name sets differ: {a_param_names ^ b_param_names}')
    for name in sorted(a_param_names & b_param_names):
        ca, cb = getattr(a, name), getattr(b, name)
        for index in ca:
            if ca[index].value != cb[index].value:
                diffs.append(f'Param {name}[{index}]: {ca[index].value!r} != {cb[index].value!r}')

    a_con_names = {c.name for c in a.component_objects(pe.Constraint, active=None)}
    b_con_names = {c.name for c in b.component_objects(pe.Constraint, active=None)}
    if a_con_names != b_con_names:
        diffs.append(f'Constraint component name sets differ: {a_con_names ^ b_con_names}')
    n_expr_compared = 0
    for name in sorted(a_con_names & b_con_names):
        ca, cb = getattr(a, name), getattr(b, name)
        for index in ca:
            if bool(ca[index].active) != bool(cb[index].active):
                diffs.append(f'Constraint {name}[{index}].active: {ca[index].active!r} != {cb[index].active!r}')
            if str(ca[index].expr) != str(cb[index].expr):
                diffs.append(f'Constraint {name}[{index}].expr strings differ')
            n_expr_compared += 1

    a_obj_names = {c.name for c in a.component_objects(pe.Objective, active=None)}
    b_obj_names = {c.name for c in b.component_objects(pe.Objective, active=None)}
    if a_obj_names != b_obj_names:
        diffs.append(f'Objective component name sets differ: {a_obj_names ^ b_obj_names}')
    for name in sorted(a_obj_names & b_obj_names):
        ca, cb = getattr(a, name), getattr(b, name)
        for index in ca:
            if bool(ca[index].active) != bool(cb[index].active):
                diffs.append(f'Objective {name}[{index}].active: {ca[index].active!r} != {cb[index].active!r}')
            if str(ca[index].expr) != str(cb[index].expr):
                diffs.append(f'Objective {name}[{index}].expr strings differ')

    for suffix_name in ('ipopt_zL_in', 'ipopt_zU_in', 'dual'):
        sa = _suffix_index(getattr(a, suffix_name))
        sb = _suffix_index(getattr(b, suffix_name))
        if sa != sb:
            diffs.append(f'Suffix {suffix_name} differs: keys_a-keys_b={set(sa) - set(sb)} '
                          f'keys_b-keys_a={set(sb) - set(sa)} '
                          f'value_diffs={[k for k in sa.keys() & sb.keys() if sa[k] != sb[k]]}')

    return diffs, n_expr_compared


def _block_equivalence_check(block_label, pristine, mutated_block):
    """`pristine` is a clone taken BEFORE the mutation; `mutated_block` is the
    SAME object `_apply_representative_cycle_mutation` was run on (so it now
    holds this "cycle"'s state, exactly like a live production block right
    before its solve). Returns the check record and the three timed
    quantities' raw sample lists."""
    legacy_clone = mutated_block.clone()
    captured_state = NET.capture_block_mutable_state(mutated_block)
    rebuilt = NET.apply_block_mutable_state(pristine.clone(), captured_state)

    diffs, n_expr_compared = _diff_blocks(legacy_clone, rebuilt)

    clone_samples = []
    capture_samples = []
    rebuild_samples = []
    for _ in range(N_TIMING_REPS):
        t0 = time.perf_counter()
        mutated_block.clone()
        clone_samples.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        NET.capture_block_mutable_state(mutated_block)
        capture_samples.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        NET.apply_block_mutable_state(pristine.clone(), captured_state)
        rebuild_samples.append(time.perf_counter() - t0)

    record = {
        'block_label': block_label,
        'n_diffs': len(diffs),
        'diffs_sample': diffs[:20],
        'equivalent': len(diffs) == 0,
        'n_constraint_expr_strings_compared': n_expr_compared,
        'timing_seconds': {
            'legacy_clone_median': statistics.median(clone_samples),
            'lightweight_capture_median': statistics.median(capture_samples),
            'on_demand_rebuild_median': statistics.median(rebuild_samples),
            'n_reps': N_TIMING_REPS,
        },
    }
    return record


# ---------------------------------------------------------------------------
# End-to-end dispatch check: real `update_transmission_coordination_model_
# and_solve`, `Network.run_smopf` monkeypatched to a canned (never-solving)
# result, `BlockData.clone` patched to COUNT real invocations (still calls
# through -- never a stub).
# ---------------------------------------------------------------------------

def _canned_result(succeeded):
    result = po.SolverResults()
    result.solver.status = po.SolverStatus.ok if succeeded else po.SolverStatus.warning
    result.solver.termination_condition = (
        po.TerminationCondition.optimal if succeeded else po.TerminationCondition.maxIterations)
    return result


def _end_to_end_dispatch_check(planning, tso_model, consensus_vars, dual_vars, admm_parameters):
    transmission_network = planning.transmission_network

    # `_build_admm_ready_state` (reused, unmodified) permanently replaces
    # `transmission_network.optimize` with a stub that never reaches
    # `Network.run_smopf` at all (by design, for ITS zero-solve construction
    # phase). This check needs the REAL `NetworkData.optimize` dispatch
    # (the exact function under test), with only `Network.run_smopf` faked
    # underneath it -- restore the genuine bound method here.
    transmission_network.optimize = NetworkData.optimize.__get__(
        transmission_network, type(transmission_network))

    clone_calls = {'n': 0}
    orig_clone = BlockData.clone

    def counting_clone(self, *a, **kw):
        clone_calls['n'] += 1
        return orig_clone(self, *a, **kw)

    orig_run_smopf = NET.Network.run_smopf
    fail_block_key = next(iter(
        (year, day)
        for year in transmission_network.years
        for day in transmission_network.days
    ))

    def make_run_smopf(fail_key):
        def canned_run_smopf(self, model, params, from_warm_start=False, print_header=True):
            key = (self.year, self.day)
            return _canned_result(succeeded=(key != fail_key))
        return canned_run_smopf

    def run_once(cycle, tso_pristine_base, fail_key):
        clone_calls['n'] = 0
        NET.Network.run_smopf = make_run_smopf(fail_key)
        try:
            res = srp.update_transmission_coordination_model_and_solve(
                transmission_network, tso_model,
                consensus_vars['vmag'], dual_vars['vmag']['tso'],
                consensus_vars['pf'], dual_vars['pf']['tso'],
                consensus_vars['ess'], dual_vars['ess']['tso'],
                admm_parameters,
                {node_id: {year: {'s_available': 1.0, 'e_available': 1.0}
                           for year in transmission_network.years}
                 for node_id in transmission_network.active_distribution_network_nodes},
                from_warm_start=True,
                cycle=cycle,
                tso_pristine_base=tso_pristine_base,
            )
        finally:
            NET.Network.run_smopf = orig_run_smopf
        return res, clone_calls['n']

    n_blocks = sum(1 for year in transmission_network.years for day in transmission_network.days)

    BlockData.clone = counting_clone
    try:
        # Case A: legacy path (tso_pristine_base=None), cycle=1 (not the
        # comparator cycle), no block "fails" -> matches pre-Step-3.6
        # behaviour: exactly one clone per (year, day) block, every call,
        # because `failure_snapshot_callback` is passed unconditionally.
        _res_a, clones_a = run_once(cycle=1, tso_pristine_base=None,
                                     fail_key=('__none__', '__none__'))

        # Case B: lightweight path, cycle=1, no block fails, not cycle 7 ->
        # zero clones anywhere (the core Step 3.6 claim).
        tso_pristine_base = {
            year: {day: tso_model[year][day].clone() for day in transmission_network.days}
            for year in transmission_network.years
        }
        # the clone() calls just above (building the pristine base) are a
        # REAL, expected cost this end-to-end check must not attribute to
        # the per-cycle dispatch being tested -- reset the counter after.
        clone_calls['n'] = 0
        _res_b, clones_b = run_once(cycle=1, tso_pristine_base=tso_pristine_base,
                                     fail_key=('__none__', '__none__'))

        # Case C: lightweight path, ONE block "fails" -> exactly one clone
        # (the on-demand rebuild for that single block), regardless of how
        # many TSO blocks exist in total.
        _res_c, clones_c = run_once(cycle=1, tso_pristine_base=tso_pristine_base,
                                     fail_key=fail_block_key)

        # Case D: lightweight path, cycle=7 (the comparator cycle), no
        # block fails -> exactly one clone if ('2025', 'Summer') is among
        # this state's TSO blocks (the `save_selected_tso_comparator`
        # rebuild), else zero -- both are asserted explicitly below rather
        # than assumed.
        comparator_present = (
            any(str(y) == '2025' for y in transmission_network.years)
            and any(str(d) == 'Summer' for d in transmission_network.days)
        )
        _res_d, clones_d = run_once(cycle=7, tso_pristine_base=tso_pristine_base,
                                     fail_key=('__none__', '__none__'))
    finally:
        BlockData.clone = orig_clone
        NET.Network.run_smopf = orig_run_smopf

    expected_d = 1 if comparator_present else 0

    return {
        'n_tso_blocks': n_blocks,
        'legacy_path_clone_calls': clones_a,
        'legacy_path_expected': n_blocks,
        'legacy_path_matches_expected': clones_a == n_blocks,
        'lightweight_no_failure_clone_calls': clones_b,
        'lightweight_no_failure_expected': 0,
        'lightweight_no_failure_matches_expected': clones_b == 0,
        'lightweight_one_failure_clone_calls': clones_c,
        'lightweight_one_failure_expected': 1,
        'lightweight_one_failure_matches_expected': clones_c == 1,
        'comparator_present': comparator_present,
        'lightweight_cycle7_comparator_clone_calls': clones_d,
        'lightweight_cycle7_comparator_expected': expected_d,
        'lightweight_cycle7_comparator_matches_expected': clones_d == expected_d,
    }


# ---------------------------------------------------------------------------
# Preserved-fixture reload check (read-only, main checkout).
# ---------------------------------------------------------------------------

def _fixture_reload_check():
    import pickle
    records = []
    for path in FIXTURE_PATHS:
        record = {'path': path, 'exists': os.path.exists(path)}
        if record['exists']:
            try:
                with open(path, 'rb') as handle:
                    payload = pickle.load(handle)
                record['unpickled'] = True
                record['has_model_key'] = isinstance(payload, dict) and 'model' in payload
                if record['has_model_key']:
                    model = payload['model']
                    record['model_type'] = type(model).__name__
                    record['n_vars'] = sum(
                        1 for _ in model.component_data_objects(pe.Var, active=None))
            except Exception as error:
                record['unpickled'] = False
                record['error'] = repr(error)
        records.append(record)
    all_ok = all(r['exists'] and r.get('unpickled') for r in records)
    return records, all_ok


# ---------------------------------------------------------------------------

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _refuse_overwrite(RESULTS_PATH)
    _refuse_overwrite(MANIFEST_PATH)

    guard = SolveProfileGuard(permitted=(), label='S36 clone-capture check').install()
    results = {}
    try:
        planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = (
            _build_admm_ready_state('p515s36_clone_capture_check'))

        transmission_network = planning.transmission_network
        admm_parameters = planning.params.admm

        # ---- Block picks -------------------------------------------------
        tso_year = '2025' if '2025' in transmission_network.years else next(iter(transmission_network.years))
        tso_day = 'Summer' if 'Summer' in transmission_network.days else next(iter(transmission_network.days))
        tso_block = tso_model[tso_year][tso_day]

        dso_node = 7 if 7 in dso_models else next(iter(dso_models))
        dso_network = planning.distribution_networks[dso_node]
        dso_year = '2025' if '2025' in dso_network.years else next(iter(dso_network.years))
        dso_day = 'Autumn' if 'Autumn' in dso_network.days else next(iter(dso_network.days))
        dso_block = dso_models[dso_node][dso_year][dso_day]

        results['block_picks'] = {
            'tso': {'year': tso_year, 'day': tso_day},
            'dso': {'node_id': dso_node, 'year': dso_year, 'day': dso_day},
        }

        # ---- Check 1: TSO block equivalence -------------------------------
        tso_pristine = tso_block.clone()
        tso_mutation_record = _apply_representative_cycle_mutation(tso_block)
        results['tso_mutation'] = tso_mutation_record
        results['tso_equivalence'] = _block_equivalence_check('TSO', tso_pristine, tso_block)

        # ---- Check 2: DSO node-7 block equivalence ------------------------
        dso_pristine = dso_block.clone()
        dso_mutation_record = _apply_representative_cycle_mutation(dso_block)
        results['dso_mutation'] = dso_mutation_record
        results['dso_equivalence'] = _block_equivalence_check('DSO node 7', dso_pristine, dso_block)

        # ---- Check 3: end-to-end dispatch / clone-call-count --------------
        results['end_to_end_dispatch'] = _end_to_end_dispatch_check(
            planning, tso_model, consensus_vars, dual_vars, admm_parameters)

        # ---- Check 4: capture-mode switch default/override ----------------
        results['switch'] = {
            'default_mode': admm_parameters.tso_snapshot_capture_mode,
            'default_is_lightweight': admm_parameters.tso_snapshot_capture_mode == 'lightweight',
        }
        admm_parameters.tso_snapshot_capture_mode = 'legacy_clone'
        results['switch']['settable_to_legacy_clone'] = (
            admm_parameters.tso_snapshot_capture_mode == 'legacy_clone')
        admm_parameters.tso_snapshot_capture_mode = 'lightweight'

        # ---- Check 5: preserved fixture reload -----------------------------
        fixture_records, fixtures_all_ok = _fixture_reload_check()
        results['fixture_reload'] = {
            'records': fixture_records,
            'all_ok': fixtures_all_ok,
        }

    finally:
        guard.uninstall()

    verify_failures = guard.verify(expected_solves=0)

    all_pass = (
        results['tso_equivalence']['equivalent']
        and results['dso_equivalence']['equivalent']
        and results['end_to_end_dispatch']['legacy_path_matches_expected']
        and results['end_to_end_dispatch']['lightweight_no_failure_matches_expected']
        and results['end_to_end_dispatch']['lightweight_one_failure_matches_expected']
        and results['end_to_end_dispatch']['lightweight_cycle7_comparator_matches_expected']
        and results['switch']['default_is_lightweight']
        and results['switch']['settable_to_legacy_clone']
        and results['fixture_reload']['all_ok']
        and not verify_failures
    )

    payload = {
        'stage': 'P5.15 Step 3.6 -- clone-capture zero-solve equivalence checks',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 21 item 4',
            'WORKER_REPORT_S36_CLONE_CAPTURE.md',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'solve_profile_guard': {'permitted': [], 'counts': dict(guard.counts),
                                 'verify_failures': verify_failures},
        'checks': results,
        'all_checks_pass': all_pass,
    }

    with open(RESULTS_PATH, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    manifest = {os.path.relpath(RESULTS_PATH, REPO): _sha256_file(RESULTS_PATH)}
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump(manifest, handle, indent=1)

    print(f'[S36-CLONE-CAPTURE] wrote {RESULTS_PATH}')
    print(f'[S36-CLONE-CAPTURE] all_checks_pass={all_pass} solve_guard_failures={verify_failures}')
    if verify_failures or not all_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
