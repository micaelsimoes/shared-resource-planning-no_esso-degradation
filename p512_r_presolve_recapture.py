"""P5.12-R: one cold recapture, never solve the cycle-21 target.

Two modes, selected on the command line (no default numeric behaviour on a
bare invocation):

    python -B p512_r_presolve_recapture.py run --baseline-dir <dir>
    python -B p512_r_presolve_recapture.py rehearse --out-dir <dir>

`run` replays the cold-RESCALED ADMM trajectory through cycle 20, captures the
cycle-21 target's solver input, and stops before `solver.solve` -- this is the
P5.12-R trajectory itself and is NOT authorized by the rehearsal-repair task
that produced this revision of the file.

`rehearse` never calls a solver. It exercises the capture/serialization/
integrity machinery on real, unsolved production objects so the mechanisms can
be inspected before the one authorized real run.

Baseline-directory schema (JSON), consumed by both modes for provenance
comparison, never written by this harness:

    initial_repository.json:
        {'host':..., 'root':..., 'head':..., 'branch':..., 'upstream':...,
         'divergence':..., 'status':..., 'ignored_status':..., 'recent':...,
         'tracked_hashes': {path: sha256, ...}, 'approved_diff':...,
         'staged_diff':..., 'harness_sha256': <optional, sha256 of this file
         at baseline-capture time>}
    accepted_hashes.json:
        {artifact_path: {'actual': sha256, ...}, ...}
    provenance.json, runtime_identity.json: free-form, hashed and recorded
        only; not schema-checked here.

All overrides are process-local. Run once with canonical Python -B.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import platform
import random
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyomo.environ as pe
from pyomo.opt.base.solvers import OptSolver
from pyomo.opt.solver.shellcmd import SystemCallSolver
from pyomo.opt import ProblemFormat
import network as N
import shared_energy_storage_data as E
import shared_resources_planning as S
import p56a_oracle as O
import p58_rescale as R
import p59_rho as RH
import p512_a_cold_rescaled_convergence as A
import p512_b_cycle21_forensic as B

ROOT = Path(__file__).resolve().parent
RUN_OUT = ROOT / 'data/SRP1/Results/P512R'
TARGET = 'DSO:case33_3|2025|Spring'
RHO = {'v': 1.5, 'pf': 300.0, 'ess': 1.0}
CYCLE = 0
CONTEXT = None
REPORT = {'stage': 'P5.12-R', 'cycles': [], 'ledger': [], 'captures': [], 'matches': [],
          'counters': {'guard_calls': 0, 'guard_calls_target': 0,
                       'original_solve_calls': 0, 'original_solve_calls_target': 0,
                       'process_launches': 0, 'process_launches_by_context': []},
          'target_log': {}, 'status': 'NOT_STARTED'}
HIST = json.loads((ROOT/'data/SRP1/Results/P512B/p512b_cycle21_forensic.json').read_text())
ORDER = list(HIST['cycle20_blocks'])

REQUIRED_BASELINE_FILES = ('provenance.json', 'initial_repository.json',
                            'runtime_identity.json', 'accepted_hashes.json')


class Stop(BaseException):
    """Not caught by production's solver-error or recovery handlers."""
class Captured(BaseException):
    """Terminal successful capture, not a solver outcome."""


def require(value, message):
    if not value:
        raise Stop(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


# ===========================================================================
# lossless / strict scalar encoding (D7)
# ===========================================================================
def atom(x):
    """STRICT encoder: model/solver-input describing values. Raises Stop on
    a non-finite float. Used by model_state, scan_finite and export."""
    if x is None or isinstance(x, (str, bool, np.bool_)): return x
    if isinstance(x, (np.integer, int)): return ['int', str(int(x))]
    if isinstance(x, (np.floating, float)):
        require(math.isfinite(float(x)), 'nonfinite input')
        return ['float', float(x).hex()]
    if isinstance(x, tuple): return ['tuple', [atom(y) for y in x]]
    return ['text', str(x)]


def atom_lossless(x):
    """LOSSLESS encoder: diagnostic state/metadata/journal values. Non-finite
    floats are preserved (hex round-trips inf/nan) instead of aborting."""
    if x is None or isinstance(x, (str, bool, np.bool_)): return x
    if isinstance(x, (np.integer, int)): return ['int', str(int(x))]
    if isinstance(x, (np.floating, float)): return ['float', float(x).hex()]
    if isinstance(x, tuple): return ['tuple', [atom_lossless(y) for y in x]]
    return ['text', str(x)]


def json_safe(x):
    """Recursively replace non-finite floats with a lossless hex marker so the
    journal/dump writers never hit json.dump(allow_nan=False) on a diagnostic
    (non solver-input) value. Everything else is passed through unchanged;
    a genuinely unserializable object still fails loudly via json.dump's own
    TypeError (no str() fallback is introduced here)."""
    if isinstance(x, float):
        return x if math.isfinite(x) else ['float_nonfinite', float(x).hex()]
    if isinstance(x, dict):
        return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def collect_nonfinite(value, path='$'):
    found = []
    def walk(v, p):
        if isinstance(v, float) and not math.isfinite(v):
            found.append(p)
        elif isinstance(v, dict):
            for k, vv in v.items(): walk(vv, p+'/'+str(k))
        elif isinstance(v, (list, tuple)):
            for i, vv in enumerate(v): walk(vv, p+f'[{i}]')
    walk(value, path)
    return found


def dump(path, value):
    path = Path(path)
    with path.open('x') as f: json.dump(json_safe(value), f, indent=1, allow_nan=False)


def persist():
    # This is the new stage's own live journal, never historical evidence.
    other = {k: v for k, v in REPORT.items() if k != 'nonfinite_diagnostic_paths'}
    REPORT['nonfinite_diagnostic_paths'] = collect_nonfinite(other)
    with (RUN_OUT/'run.json').open('w') as f: json.dump(json_safe(REPORT), f, indent=1, allow_nan=False)


# ===========================================================================
# model / component-set serialization (D1)
# ===========================================================================
SERIALIZED_SET_CTYPES = (pe.Set, pe.RangeSet)
KNOWN_CONTENT_CTYPES = {'Suffix', 'Var', 'Param', 'Constraint', 'Objective', 'Expression', 'Set', 'RangeSet'}


def serialize_set(c):
    """Non-indexed Sets/RangeSets: serialize members in the Set's own
    iteration order (never via the deprecated positional __getitem__).
    Indexed Sets: per index, in deterministic (dict/key) order -> members."""
    if c.is_indexed():
        return ['indexed', [[atom(k), [atom(v) for v in c[k]]] for k in c.keys()]]
    return ['ordered', [atom(v) for v in c]]


def model_state(m):
    rows = []
    for c in m.component_objects(descend_into=True, active=None):
        rows.append(['component', c.name, c.ctype.__name__, bool(c.active)])
        if c.ctype is pe.Suffix:
            rows.append(['suffix', c.name, int(c.direction), int(c.datatype),
                         [[k.name, atom(v)] for k, v in c.items()]])
        elif c.ctype is pe.Var:
            for v in c.values():
                rows.append(['var', v.name, atom(v.value), v.fixed, atom(v.lb), atom(v.ub), str(v.domain)])
        elif c.ctype is pe.Param:
            rows.append(['param', c.name, c.mutable,
                         [[atom(i), atom(pe.value(c[i], exception=False))] for i in c]])
        elif c.ctype in (pe.Constraint, pe.Objective, pe.Expression):
            for v in c.values():
                rows.append(['expr', v.name, bool(v.active), str(v.expr),
                             str(getattr(v, 'sense', ''))])
        elif c.ctype in SERIALIZED_SET_CTYPES:
            rows.append(['set', c.name, c.ctype.__name__, serialize_set(c)])
        # Any other ctype is recorded only via the 'component' row above; see
        # component_ctype_inventory() for an explicit accounting of which
        # ctypes were content-serialized here and which were not.
    return rows


def component_ctype_inventory(m):
    counts = {}
    for c in m.component_objects(descend_into=True, active=None):
        name = c.ctype.__name__
        counts[name] = counts.get(name, 0) + 1
    handled = sorted(set(counts) & KNOWN_CONTENT_CTYPES)
    unhandled = sorted(set(counts) - KNOWN_CONTENT_CTYPES)
    return {'counts': counts, 'content_serialized_ctypes': handled,
            'ctypes_recorded_by_name_only': unhandled}


def plain(x):
    if isinstance(x, np.ndarray): return ['array', str(x.dtype), list(x.shape), x.tobytes().hex()]
    if isinstance(x, dict): return ['dict', [[atom_lossless(k), plain(v)] for k, v in x.items()]]
    if isinstance(x, (list, tuple)): return [type(x).__name__, [plain(v) for v in x]]
    if hasattr(x, '__dict__'): return [type(x).__module__+'.'+type(x).__name__, plain(vars(x))]
    return atom_lossless(x)


def scan_finite(m):
    for v in m.component_data_objects(pe.Var, active=None): atom(v.value)
    for c in m.component_objects(pe.Suffix, active=None):
        for v in c.values(): atom(v)


def export(m, solver, directory, name):
    directory.mkdir(exist_ok=True)
    clone = m.clone()
    path, sid = clone.write(str(directory/(name+'.nl')), format=ProblemFormat.nl,
                            solver_capability=solver.has_capability, io_options={})
    sm = clone.solutions.symbol_map[sid]
    mapping = [[symbol, obj.name] for symbol, obj in sm.bySymbol.items()]
    exported = set(id(v) for v in sm.bySymbol.values())
    absent = [[v.name, bool(v.fixed), atom(v.value)] for v in clone.component_data_objects(pe.Var)
              if id(v) not in exported]
    record = {'mapping': mapping, 'not_exported_variables': absent,
              'writer_format': 'nl', 'writer_io_options': {}}
    dump(directory/(name+'_mapping.json'), record)
    return {'nl_sha256': sha(path), 'mapping_sha256': digest(record)}


# ===========================================================================
# capture: generalized snapshot primitive (used by run-mode save_target and by
# the no-solve rehearsal)
# ===========================================================================
def capture_snapshot(directory, model, network_obj, params, solver, boundary, from_warm_start,
                      admm_params, candidate, cycle, target_label, extra_meta=None):
    directory.mkdir(exist_ok=False)
    before = model_state(model)
    payload = {'model': model, 'network': network_obj, 'params': params,
               'admm_params': admm_params, 'candidate': candidate,
               'options': dict(solver.options) if solver is not None else None,
               'boundary': boundary, 'cycle': cycle, 'target': target_label,
               'random_state': random.getstate(), 'numpy_random_state': np.random.get_state(),
               'from_warm_start': bool(from_warm_start), 'executable': '/usr/local/bin/ipopt'}
    if extra_meta: payload.update(extra_meta)
    path = directory/'snapshot.pkl'
    with path.open('xb') as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with path.open('rb') as f: loaded = pickle.load(f)
    require(model_state(loaded['model']) == before, str(directory)+' model reload mismatch')
    for k in payload:
        if k != 'model':
            require(plain(payload[k]) == plain(loaded[k]), str(directory)+' metadata reload mismatch '+k)
    dump(directory/'ordered_state.json', before)
    export_solver = solver or pe.SolverFactory('ipopt', executable='/usr/local/bin/ipopt')
    ex1 = export(model, export_solver, directory, 'original')
    ex2 = export(loaded['model'], export_solver, directory, 'reload')
    require(ex1 == ex2, str(directory)+' solver-input mismatch')
    require(model_state(model) == before, str(directory)+' capture mutated live model')
    rec = {'directory': str(directory), 'snapshot_sha256': sha(path), 'semantic_sha256': digest(before),
           'reload_equal': True, 'solver_input_equal': True, 'export': ex1,
           'boundary': boundary, 'suffixes': [r for r in before if r[0] == 'suffix'],
           'effective_options': payload['options'], 'from_warm_start': payload['from_warm_start']}
    dump(directory/'manifest.json', rec)
    return before, rec, payload


def save_target(model, net, params, solver, boundary, from_warm_start):
    label = f'cycle{CYCLE}_{boundary}'
    directory = RUN_OUT/label
    before, rec, payload = capture_snapshot(
        directory, model, net, params, solver, boundary, from_warm_start,
        PLANNING.params.admm, CANDIDATE, CYCLE, TARGET,
        extra_meta={'source_head': BASELINE['initial']['head'],
                    'source_hashes': BASELINE['initial']['tracked_hashes']})
    REPORT['captures'].append({k: v for k, v in rec.items() if k != 'suffixes'})
    persist()
    return before


def check_history(model, net, expected):
    rec = B.capture_block(net.name, 'TSO' if net.is_transmission else 'DSO:'+net.name,
                          net.year, net.day, model, None, CYCLE)
    fields = ['agent', 'network', 'year', 'day', 'cycle', 'n_variables', 'n_constraints',
              'active_objective', 'objective_value_at_start', 'admm_objective_scale',
              'rho_v', 'rho_pf', 'rho_ess', 'primal', 'consensus', 'multipliers',
              'start_max_violation', 'start_worst_family', 'start_violating_families']
    # JSON roundtrip only harmonizes tuples in summary lists, not numbers/order.
    rec = json.loads(json.dumps(rec))
    differences = {k: {'expected': expected.get(k), 'actual': rec.get(k)}
                   for k in fields if rec.get(k) != expected.get(k)}
    REPORT['matches'].append({'cycle': CYCLE, 'block': CONTEXT, 'boundary': 'pre_setup',
                              'differences': differences, 'matched': not differences})
    persist()
    require(not differences, f'B pre-setup summary mismatch {CYCLE} {CONTEXT}: {list(differences)}')


# ===========================================================================
# repository / baseline integrity (D2, D3)
# ===========================================================================
def read_only_git(*args):
    result = subprocess.run(['git', *args], cwd=str(ROOT), capture_output=True, text=True, check=True)
    return result.stdout


def current_repo_state():
    head = read_only_git('rev-parse', 'HEAD').strip()
    try:
        branch = read_only_git('branch', '--show-current').strip()
    except subprocess.CalledProcessError:
        branch = read_only_git('symbolic-ref', '--short', 'HEAD').strip()
    staged = read_only_git('diff', '--cached')
    status_porcelain = read_only_git('status', '--porcelain', '--untracked-files=no')
    files = sorted(f for f in read_only_git('ls-files').splitlines() if f)
    tracked_hashes = {f: sha(ROOT/f) for f in files}
    return {'head': head, 'branch': branch, 'staged_diff': staged,
            'status_porcelain': status_porcelain, 'tracked_paths': files,
            'tracked_hashes': tracked_hashes}


def harness_sha256():
    return sha(Path(__file__).resolve())


def load_baseline(baseline_dir):
    baseline_dir = Path(baseline_dir).resolve()
    require(baseline_dir.is_dir(), 'baseline dir missing or not a directory: '+str(baseline_dir))
    require(baseline_dir != RUN_OUT.resolve(),
            'refuse the historical P512R root as a baseline dir')
    for name in REQUIRED_BASELINE_FILES:
        require((baseline_dir/name).exists(), 'baseline missing required file: '+name)
    initial = json.loads((baseline_dir/'initial_repository.json').read_text())
    accepted = json.loads((baseline_dir/'accepted_hashes.json').read_text())
    for field in ('head', 'branch', 'tracked_hashes'):
        require(field in initial, 'baseline initial_repository.json missing required field: '+field)
    provenance = json.loads((baseline_dir/'provenance.json').read_text())
    runtime_identity = json.loads((baseline_dir/'runtime_identity.json').read_text())
    return {'dir': str(baseline_dir), 'initial': initial, 'accepted': accepted,
            'provenance': provenance, 'runtime_identity': runtime_identity,
            'harness_sha256': initial.get('harness_sha256'),
            'file_hashes': {name: sha(baseline_dir/name) for name in REQUIRED_BASELINE_FILES}}


def integrity_diff(baseline_initial, current, harness_sha_now, baseline_harness_sha):
    """Compare the CURRENT repository state against a baseline's
    initial_repository.json-shaped dict. Returns a dict of named differences;
    empty means the two are consistent for every checked field."""
    diffs = {}
    if current['head'] != baseline_initial.get('head'):
        diffs['head'] = {'expected': baseline_initial.get('head'), 'actual': current['head']}
    if current['branch'] != baseline_initial.get('branch'):
        diffs['branch'] = {'expected': baseline_initial.get('branch'), 'actual': current['branch']}
    if current['staged_diff'] != baseline_initial.get('staged_diff', ''):
        diffs['staged_diff'] = {'expected_empty_or_baseline': True}
    baseline_tracked = baseline_initial.get('tracked_hashes', {})
    added = sorted(set(current['tracked_hashes']) - set(baseline_tracked))
    removed = sorted(set(baseline_tracked) - set(current['tracked_hashes']))
    changed = sorted(k for k in current['tracked_hashes']
                     if k in baseline_tracked and current['tracked_hashes'][k] != baseline_tracked[k])
    if added: diffs['added_tracked_files'] = added
    if removed: diffs['removed_tracked_files'] = removed
    if changed: diffs['changed_tracked_files'] = changed
    baseline_harness = baseline_initial.get('harness_sha256', baseline_harness_sha)
    if baseline_harness is None:
        diffs['harness_sha256_not_recorded_in_baseline'] = True
    elif harness_sha_now != baseline_harness:
        diffs['harness_sha256'] = {'expected': baseline_harness, 'actual': harness_sha_now}
    return diffs


def protected_artifacts(baseline):
    """Existing 'accepted-artifact unchanged' check, parameterized by baseline."""
    for name, item in baseline['accepted'].items():
        require(sha(ROOT/name) == item['actual'], 'accepted artifact changed: '+name)


def full_integrity_check(baseline, harness_sha_at_start=None):
    current = current_repo_state()
    harness_now = harness_sha256()
    diffs = integrity_diff(baseline['initial'], current, harness_now, baseline['harness_sha256'])
    if harness_sha_at_start is not None and harness_now != harness_sha_at_start:
        diffs['harness_sha256_changed_during_run'] = {
            'start': harness_sha_at_start, 'now': harness_now}
    return diffs, current, harness_now


# ===========================================================================
# checkpoint (restructured to accept an explicit locals-like mapping, D6)
# ===========================================================================
CHECKPOINT_STATE_KEYS = ['consensus_vars', 'dual_vars', 'candidate_solution',
    'previous_recourse', 'previous_recourse_blocks', 'previous_objective_component_blocks',
    'previous_slack_component_blocks', 'previous_tso_voltage_slack_state',
    'consecutive_converged_cycles', 'sess_available_capacities', 'admm_diagnostics', 'primal_evolution']


def build_checkpoint(local, directory, planning_obj, boundary):
    models = {'tso': local['tso_model'], 'dso': local['dso_models'], 'esso': local['esso_model']}
    state = {k: local[k] for k in CHECKPOINT_STATE_KEYS}
    directory.mkdir(exist_ok=False)
    payload = {'planning': planning_obj, 'models': models, 'state': state, 'all_frame_locals': local,
               'boundary': boundary,
               'random_state': random.getstate(), 'numpy_random_state': np.random.get_state()}
    with (directory/'checkpoint.pkl').open('xb') as f: pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (directory/'checkpoint.pkl').open('rb') as f: loaded = pickle.load(f)
    require(plain(state) == plain(loaded['state']), 'checkpoint coordination/history reload mismatch')
    require(plain(planning_obj) == plain(loaded['planning']), 'checkpoint planning reload mismatch')

    def flatten(obj, prefix=''):
        if isinstance(obj, dict):
            for k, v in obj.items(): yield from flatten(v, prefix+'/'+str(k))
        else: yield prefix, obj

    new = dict(flatten(loaded['models'])); records = []
    solver = pe.SolverFactory('ipopt', executable='/usr/local/bin/ipopt')
    for i, (key, m) in enumerate(flatten(models)):
        before = model_state(m)
        require(before == model_state(new[key]), 'checkpoint model reload mismatch '+key)
        e1 = export(m, solver, directory, f'block{i}_original')
        e2 = export(new[key], solver, directory, f'block{i}_reload')
        require(e1 == e2, 'checkpoint export mismatch '+key)
        records.append({'block': key, 'semantic_sha256': digest(before), 'export': e1})
    rec = {'snapshot_sha256': sha(directory/'checkpoint.pkl'), 'reload_equal': True,
           'state_sha256': digest(plain(state)), 'blocks': records,
           'all_frame_local_keys': list(local), 'capture_boundary': boundary}
    dump(directory/'manifest.json', rec)
    return rec


def checkpoint(frame):
    local = dict(frame.f_locals)
    rec = build_checkpoint(local, RUN_OUT/'cycle20_checkpoint', PLANNING,
                           'cycle20 complete; before cycle21 DSO coordination updates')
    REPORT['checkpoint'] = rec
    persist()


# ===========================================================================
# stop paths
# ===========================================================================
def stop_at_target(is_target, capture):
    if is_target:
        capture()
        raise Captured('cycle21 target captured after setup; solver.solve not entered')


def raw_guard(is_target, call):
    require(not is_target, 'target solver invocation blocked by independent guard')
    result = call()
    require(B.solver_result_succeeded(result), 'unsuccessful local solve; recovery prohibited')
    return result


def selftest(out_dir):
    calls = []
    try: stop_at_target(True, lambda: calls.append('capture'))
    except Captured: pass
    require(calls == ['capture'], 'terminal capture path test')
    try: raw_guard(True, lambda: calls.append('illegal_solve'))
    except Stop: pass
    require(calls == ['capture'], 'independent target guard test')
    try: raw_guard(False, lambda: None)
    except Stop: pass
    else: raise Stop('failure guard did not terminate')
    source = inspect.getsource(N._run_smopf_solver_attempt)
    require(source.index('_create_smopf_solver(') < source.index('solver.solve('), 'solver setup boundary changed')
    require(not issubclass(Captured, Exception) and not issubclass(Stop, Exception), 'terminal signal can be swallowed')
    # Verify real wrapper propagation through the production attempt function,
    # with a fake setup and sentinel solver. No numerical solve is called.
    original = N._create_smopf_solver
    def fake(*args, **kw): stop_at_target(True, lambda: calls.append('prepared'))
    N._create_smopf_solver = fake
    try:
        try: N._run_smopf_solver_attempt(None, None, None)
        except Captured: pass
        else: raise Stop('production attempt swallowed terminal capture')
    finally: N._create_smopf_solver = original
    dump(Path(out_dir)/'stop_path_selftest.json', {'passed': True, 'events': calls, 'solver_invocations': 0,
         'prepared_capture_propagates': True, 'failed_result_stops': True})


# ===========================================================================
# process-launch / solve-guard instrumentation shared by run() and rehearse()
# ===========================================================================
class GuardedSolveViolation(Stop):
    pass


def install_process_launch_counter(counters, context_getter):
    original = SystemCallSolver._execute_command
    def counted(self, command):
        counters['process_launches'] += 1
        counters['process_launches_by_context'].append(context_getter())
        return original(self, command)
    SystemCallSolver._execute_command = counted
    return original


def uninstall_process_launch_counter(original):
    SystemCallSolver._execute_command = original


# ===========================================================================
# RUN MODE (P5.12-R trajectory; not executed by the rehearsal-repair task)
# ===========================================================================
def run(baseline_dir):
    global PLANNING, CANDIDATE, CYCLE, CONTEXT, BASELINE
    require(not (RUN_OUT/'run.json').exists(), 'refuse second trajectory')
    harness_sha_at_start = harness_sha256()
    BASELINE = load_baseline(baseline_dir)
    REPORT['baseline'] = {'dir': BASELINE['dir'], 'file_hashes': BASELINE['file_hashes']}
    diffs, current_before, harness_now = full_integrity_check(BASELINE)
    require(not diffs, 'repository integrity check failed before run: '+json.dumps(sorted(diffs)))
    REPORT['integrity_before'] = {'diffs': diffs, 'head': current_before['head'],
                                  'branch': current_before['branch'],
                                  'tracked_file_count': len(current_before['tracked_paths'])}
    protected_artifacts(BASELINE)
    selftest(str(RUN_OUT))
    O.WORK_DIR = str(RUN_OUT/'evals')
    PLANNING = O.fresh_planning('cold')
    require(PLANNING.parallel_execution is False, 'parallel execution enabled')
    RH.apply_rho_to_params(PLANNING, RHO); RH.set_adaptive_penalty(PLANNING, False)
    PLANNING.params.admm.num_max_iters = 21
    # Route observational outputs only. All numerical settings remain unchanged.
    for holder in O._holders(PLANNING):
        holder.results_dir = str(RUN_OUT/'production_snapshots')
        for year in holder.years:
            for day in holder.days: holder.network[year][day].results_dir = holder.results_dir
    Path(RUN_OUT/'production_snapshots').mkdir()
    CANDIDATE = S._build_positive_bootstrap_candidate(PLANNING, PLANNING.params.benders.positive_bootstrap)
    expected_params = plain(PLANNING.params.admm)
    REPORT['configuration'] = {'rho': RHO, 'adaptive_penalty': False, 'cap': 21, 'initial_state': None,
                               'candidate': CANDIDATE, 'admm': expected_params,
                               'carried_history_initialization': 'production cold branch: initial_state=None; previous recourse and histories None, counter zero; candidate explicit'}
    cyclesets = []
    for p in sorted((ROOT/'data/SRP1/Results/P512A').glob('p512a_trajectory_cap*.json')):
        d = json.loads(p.read_text()); cyclesets.append((str(p.relative_to(ROOT)), d['cycles'][:20]))
    REPORT['history_sources'] = [p for p, _ in cyclesets]
    network_count = 0
    original_dso = S.update_distribution_coordination_models_and_solve
    original_attempt = N._run_smopf_solver_attempt
    original_create = N._create_smopf_solver
    original_ess_attempt = E._run_solver_attempt
    original_ess_create = E._create_solver
    original_solve = OptSolver.solve
    original_shellcmd = SystemCallSolver._execute_command

    def dso(*args, **kwargs):
        nonlocal network_count
        global CYCLE
        c = kwargs['cycle']; frame = inspect.currentframe().f_back
        require(frame.f_code is S._run_operational_planning.__code__, 'unexpected coordination caller')
        require(c == CYCLE+1 and c <= 21, 'cycle order mismatch')
        require(plain(PLANNING.params.admm) == expected_params, 'ADMM settings changed')
        if CYCLE:
            require(network_count == 48, 'network block count mismatch')
            entry = frame.f_locals['admm_diagnostics'][-1]
            previous = REPORT['cycles'][-1]['observed']['recourse'] if REPORT['cycles'] else None
            row = A.cycle_row(entry, previous)
            mismatches = []
            for name, rows in cyclesets:
                old = rows[CYCLE-1]
                for k, v in old.items():
                    if row.get(k) != v: mismatches.append({'source': name, 'field': k, 'expected': v, 'actual': row.get(k)})
            REPORT['cycles'].append({'cycle': CYCLE, 'matched': not mismatches, 'differences': mismatches, 'observed': row, 'full_diagnostic': entry})
            persist()
            require(not mismatches, f'cycle {CYCLE} historical mismatch')
            print(f'[P512R] cycle {CYCLE} matched; recourse={row["recourse"]!r}', flush=True)
            if CYCLE == 20: checkpoint(frame)
        CYCLE = c; network_count = 0
        return original_dso(*args, **kwargs)

    def attempt(net, model, params, from_warm_start=False, option_overrides=None, log_suffix=None):
        nonlocal network_count
        global CONTEXT
        require(option_overrides is None and log_suffix is None, 'network recovery attempted')
        key = ('TSO' if net.is_transmission else 'DSO:'+net.name)+f'|{net.year}|{net.day}'
        CONTEXT = key
        if CYCLE:
            require(network_count < len(ORDER) and key == ORDER[network_count], 'network solve order differs')
            if CYCLE == 21: require(network_count <= 24, 'advanced beyond target')
            network_count += 1
            for g, r in RHO.items(): require(float(pe.value(getattr(model, 'rho_'+g))) == r, 'model rho changed')
            require(model.p58_rescaled_admm_objective.active, 'RESCALED objective inactive')
        if CYCLE == 20 or (CYCLE == 21 and key == TARGET):
            check_history(model, net, HIST['cycle20_blocks' if CYCLE == 20 else 'cycle21_blocks'][key])
        path = R.block_log_path(net, params); offset = os.path.getsize(path) if path and os.path.exists(path) else 0
        if CYCLE in (20, 21) and key == TARGET:
            REPORT['target_log'][f'cycle{CYCLE}_pre_setup_offset'] = {'path': path, 'offset': offset}
            save_target(model, net, params, None, 'pre_setup', from_warm_start)
        res = original_attempt(net, model, params, from_warm_start, option_overrides, log_suffix)
        if CYCLE == 20:
            info = R.read_log_since(path, offset)
            old = HIST['cycle20_blocks'][key]['ipopt_log']
            require(all(info[k] == old[k] for k in ('iterations', 'exit', 'scaled', 'unscaled')), 'cycle20 solver diagnostic mismatch '+key)
            if key == TARGET:
                with open(path) as f: f.seek(offset); text = f.read()
                (RUN_OUT/'cycle20_target.log').write_text(text)
                with (RUN_OUT/'cycle20_target_result.pkl').open('xb') as f: pickle.dump(res[0], f)
                REPORT['cycle20_target_result'] = info; persist()
        CONTEXT = None
        return res

    def create(net, model, params, from_warm_start=False, option_overrides=None, log_suffix=None):
        result = original_create(net, model, params, from_warm_start, option_overrides, log_suffix)
        if CYCLE in (20, 21) and CONTEXT == TARGET:
            def capture(): save_target(model, net, params, result[0], 'prepared', from_warm_start)
            stop_at_target(CYCLE == 21, capture)
            capture()
        return result

    def ess_attempt(model, params, solve_context, from_warm_start=False, node_id=None, option_overrides=None, log_suffix=None):
        global CONTEXT
        require(CYCLE < 21, 'ESSO attempted at cycle21')
        require(option_overrides is None and log_suffix is None, 'ESSO recovery attempted')
        CONTEXT = 'ESSO:'+str(node_id)
        result = original_ess_attempt(model, params, solve_context, from_warm_start, node_id, option_overrides, log_suffix)
        CONTEXT = None; return result

    def ess_create(*args, **kwargs):
        solver, path = original_ess_create(*args, **kwargs)
        new = RUN_OUT/'evals/cold/logs'/Path(path).name
        solver.options['output_file'] = str(new)
        return solver, str(new)

    def solve(solver, model, *args, **kwargs):
        require(CONTEXT is not None, 'unidentified solve path')
        require(str(solver.executable()) == '/usr/local/bin/ipopt', 'solver path changed')
        scan_finite(model)
        is_target = (CYCLE == 21 and CONTEXT == TARGET)
        REPORT['counters']['guard_calls'] += 1
        if is_target: REPORT['counters']['guard_calls_target'] += 1
        REPORT['ledger'].append({'cycle': CYCLE, 'block': CONTEXT, 'phase': 'attempted', 'is_target': is_target})
        persist()
        start = time.time()
        result = raw_guard(is_target, lambda: original_solve(solver, model, *args, **kwargs))
        REPORT['counters']['original_solve_calls'] += 1
        if is_target: REPORT['counters']['original_solve_calls_target'] += 1
        REPORT['ledger'].append({'cycle': CYCLE, 'block': CONTEXT, 'phase': 'completed', 'status': str(result.solver.status),
                'termination': str(result.solver.termination_condition), 'seconds': time.time()-start,
                'options': dict(solver.options)})
        persist(); return result

    def launch_context():
        return {'cycle': CYCLE, 'context': CONTEXT}

    S.update_distribution_coordination_models_and_solve = dso
    N._run_smopf_solver_attempt = attempt; N._create_smopf_solver = create
    E._run_solver_attempt = ess_attempt; E._create_solver = ess_create
    OptSolver.solve = solve
    install_process_launch_counter(REPORT['counters'], launch_context)
    REPORT['status'] = 'RUNNING'; persist()
    try:
        with R.patched_admm_objectives():
            PLANNING.run_operational_planning(type='distributed', candidate_solution=deepcopy(CANDIDATE),
                    print_results=False, debug_flag=False, return_state=True)
        raise Stop('trajectory returned instead of target capture')
    finally:
        S.update_distribution_coordination_models_and_solve = original_dso
        N._run_smopf_solver_attempt = original_attempt; N._create_smopf_solver = original_create
        E._run_solver_attempt = original_ess_attempt; E._create_solver = original_ess_create
        OptSolver.solve = original_solve
        uninstall_process_launch_counter(original_shellcmd)
        # Explicitly use the cycle-21 target offset entry's path (not "the
        # last target_log entry seen") so size_at_exit is unambiguously
        # comparable to cycle21_pre_setup_offset by final_verdict().
        cycle21_offset = REPORT['target_log'].get('cycle21_pre_setup_offset')
        target_path = cycle21_offset.get('path') if isinstance(cycle21_offset, dict) else None
        REPORT['target_log']['size_at_exit_path'] = target_path
        file_existed_at_exit = bool(target_path and os.path.exists(target_path))
        REPORT['target_log']['file_existed_at_exit'] = file_existed_at_exit
        if file_existed_at_exit:
            REPORT['target_log']['size_at_exit'] = os.path.getsize(target_path)
        diffs_after, current_after, harness_after = full_integrity_check(BASELINE, harness_sha_at_start)
        REPORT['integrity_after'] = {'diffs': diffs_after, 'head': current_after['head'],
                                     'branch': current_after['branch'],
                                     'tracked_file_count': len(current_after['tracked_paths'])}


def final_verdict(report, target):
    """PURE decision function: given only a REPORT-shaped dict (and the
    target label), decide whether the run's evidence actually supports
    'CAPTURED' or must be downgraded to 'STOPPED'. Touches no repository,
    git, filesystem or solver state -- every input is read from `report`.
    Returns (status, checks, failures) where status is 'CAPTURED' only if
    every check passes; any missing/ambiguous evidence is treated as a
    failure, never as a pass."""
    checks = {}
    failures = []

    def check(name, passed, observed=None, expected=None):
        checks[name] = {'passed': bool(passed), 'observed': observed, 'expected': expected}
        if not passed: failures.append(name)

    # --- 1. end-of-run repository integrity -------------------------------
    integrity_after = report.get('integrity_after')
    integrity_present = isinstance(integrity_after, dict) and 'diffs' in integrity_after
    check('integrity_after_present', integrity_present,
          observed=integrity_after, expected='present with a diffs field')
    diffs = integrity_after.get('diffs') if integrity_present else None

    def diff_key_absent(name, key):
        if not integrity_present:
            check(name, False, observed='integrity_after missing', expected='no '+key+' diff')
            return
        check(name, key not in diffs, observed=diffs.get(key), expected='absent')

    diff_key_absent('integrity_head_unchanged', 'head')
    diff_key_absent('integrity_branch_unchanged', 'branch')
    diff_key_absent('integrity_staged_unchanged', 'staged_diff')
    if not integrity_present:
        check('integrity_tracked_path_set_unchanged', False,
              observed='integrity_after missing', expected='no added/removed tracked files')
    else:
        added = diffs.get('added_tracked_files'); removed = diffs.get('removed_tracked_files')
        check('integrity_tracked_path_set_unchanged', not added and not removed,
              observed={'added': added, 'removed': removed},
              expected='no added/removed tracked files')
    diff_key_absent('integrity_tracked_hashes_unchanged', 'changed_tracked_files')
    if not integrity_present:
        check('integrity_harness_sha_matches_baseline', False,
              observed='integrity_after missing', expected='harness sha recorded and matching baseline')
    else:
        not_recorded = diffs.get('harness_sha256_not_recorded_in_baseline')
        mismatch = diffs.get('harness_sha256')
        check('integrity_harness_sha_matches_baseline', not not_recorded and not mismatch,
              observed={'not_recorded_in_baseline': not_recorded, 'mismatch': mismatch},
              expected='harness sha recorded and matching baseline')
    diff_key_absent('integrity_harness_sha_stable_during_run', 'harness_sha256_changed_during_run')
    if integrity_present:
        check('integrity_diffs_empty', not diffs, observed=sorted(diffs), expected=[])
    else:
        check('integrity_diffs_empty', False, observed='integrity_after missing', expected=[])

    # --- 2. primary stop occurred before the target solve ------------------
    counters = report.get('counters', {})
    guard_target = counters.get('guard_calls_target')
    check('guard_calls_target_zero', guard_target == 0, observed=guard_target, expected=0)
    solve_target = counters.get('original_solve_calls_target')
    check('original_solve_calls_target_zero', solve_target == 0, observed=solve_target, expected=0)
    launches = counters.get('process_launches_by_context') or []
    target_launches = [l for l in launches
                        if isinstance(l, dict) and l.get('cycle') == 21 and l.get('context') == target]
    check('no_process_launch_for_cycle21_target', len(target_launches) == 0,
          observed=target_launches, expected=[])

    # --- 3. target IPOPT log did not advance --------------------------------
    target_log = report.get('target_log', {})
    offset_entry = target_log.get('cycle21_pre_setup_offset')
    offset_present = isinstance(offset_entry, dict) and 'offset' in offset_entry and 'path' in offset_entry
    size_at_exit = target_log.get('size_at_exit')
    size_present = size_at_exit is not None
    if not offset_present or not size_present:
        check('target_log_did_not_advance', False,
              observed={'offset_entry': offset_entry, 'size_at_exit': size_at_exit},
              expected='cycle21_pre_setup_offset and size_at_exit both present and equal')
    else:
        path_used = target_log.get('size_at_exit_path')
        path_matches = path_used is None or path_used == offset_entry.get('path')
        equal = size_at_exit == offset_entry.get('offset')
        check('target_log_did_not_advance', bool(equal and path_matches),
              observed={'size_at_exit': size_at_exit, 'offset': offset_entry.get('offset'),
                        'size_at_exit_path': path_used, 'cycle21_offset_path': offset_entry.get('path')},
              expected='size_at_exit == cycle21 offset, same path')
    file_existed = target_log.get('file_existed_at_exit')
    offset_value = offset_entry.get('offset') if offset_present else None
    if offset_present and offset_value and offset_value > 0 and file_existed is False:
        check('target_log_file_present_at_exit', False, observed=file_existed, expected=True)
    elif not offset_present:
        check('target_log_file_present_at_exit', False,
              observed='cycle21_pre_setup_offset missing', expected=True)
    else:
        check('target_log_file_present_at_exit', True, observed=file_existed, expected=True)

    status = 'CAPTURED' if not failures else 'STOPPED'
    return status, checks, failures


def run_main(baseline_dir):
    start = time.time()
    try:
        run(baseline_dir)
    except Captured as e:
        REPORT['status'] = 'CAPTURED'; REPORT['stop_reason'] = str(e)
    except BaseException as e:
        REPORT['status'] = 'STOPPED'; REPORT['stop_reason'] = type(e).__name__+': '+str(e)
        REPORT['traceback'] = traceback.format_exc()
    try:
        protected_artifacts(BASELINE); REPORT['protected_files_unchanged'] = True
    except BaseException as e:
        REPORT['status'] = 'STOPPED'; REPORT['protection_error'] = str(e)
    # final_verdict() is pure: it only reads REPORT and may downgrade an
    # in-progress 'CAPTURED' verdict to 'STOPPED' if the recorded evidence
    # does not actually support it. It never upgrades STOPPED to CAPTURED.
    verdict_status, verdict_checks, verdict_failures = final_verdict(REPORT, TARGET)
    REPORT['verdict_checks'] = verdict_checks
    REPORT['verdict_failures'] = verdict_failures
    if REPORT['status'] == 'CAPTURED' and verdict_status != 'CAPTURED':
        REPORT['status'] = 'STOPPED'
        REPORT['stop_reason'] = ((REPORT.get('stop_reason') or '') +
            ' | verdict downgraded to STOPPED; failed checks: ' + ','.join(verdict_failures))
    REPORT['wall_seconds'] = time.time()-start
    persist()
    print('[P512R]', REPORT['status'], REPORT.get('stop_reason'), flush=True)
    sys.exit(0 if REPORT['status'] == 'CAPTURED' else 2)


# ===========================================================================
# REHEARSAL MODE -- no solver call anywhere; exercises the machinery above on
# real, unsolved production objects.
# ===========================================================================
def rehearse(out_dir):
    out_dir = Path(out_dir).resolve()
    run_out_resolved = RUN_OUT.resolve()
    inside_run_out = out_dir == run_out_resolved or run_out_resolved in out_dir.parents
    require(not inside_run_out, 'rehearsal dir must not be inside P512R')
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = {'stage': 'P5.12-R-REHEARSAL', 'out_dir': str(out_dir), 'items': {}, 'status': 'RUNNING'}
    counters = {'guard_calls': 0, 'original_solve_calls': 0, 'process_launches': 0,
                'process_launches_by_context': []}

    harness_sha_at_start = harness_sha256()

    def guarded_solve(solver, model, *args, **kwargs):
        counters['guard_calls'] += 1
        raise GuardedSolveViolation('rehearsal must never call a solver: OptSolver.solve invoked')

    original_solve = OptSolver.solve
    OptSolver.solve = guarded_solve
    original_shellcmd = install_process_launch_counter(counters, lambda: {'phase': 'rehearsal'})
    try:
        O.WORK_DIR = str(out_dir/'evals')

        # --- 0. self-hash, repository state, stop-path selftest -----------
        selftest(str(out_dir))
        summary['items']['selftest'] = 'passed (see stop_path_selftest.json)'

        current0 = current_repo_state()
        summary['items']['repo_state_start'] = {
            'head': current0['head'], 'branch': current0['branch'],
            'tracked_file_count': len(current0['tracked_paths']),
            'staged_diff_empty': current0['staged_diff'] == '',
            'status_porcelain_empty': current0['status_porcelain'] == ''}

        # --- 1. build real, unsolved objects -------------------------------
        planning = O.fresh_planning('rehearsal')
        candidate = S._build_positive_bootstrap_candidate(planning, planning.params.benders.positive_bootstrap)
        consensus_vars, dual_vars = S.create_admm_variables(planning)

        dso_network = planning.distribution_networks[9]  # case33_3
        dso_obj = dso_network.network[2025]['Spring']
        dso_params = dso_network.params
        dso_model = dso_obj.build_model(dso_params)

        tso_network = planning.transmission_network
        tso_obj = tso_network.network[2025]['Spring']
        tso_params = tso_network.params
        tso_model = tso_obj.build_model(tso_params)

        # --- 2. model_state on real blocks + ctype inventory ---------------
        dso_state = model_state(dso_model)
        tso_state = model_state(tso_model)
        dso_inv = component_ctype_inventory(dso_model)
        tso_inv = component_ctype_inventory(tso_model)
        dump(out_dir/'ctype_inventory.json', {'dso_case33_3': dso_inv, 'tso_case9': tso_inv})
        summary['items']['model_state_ctype_inventory'] = {'dso_case33_3': dso_inv, 'tso_case9': tso_inv}

        # --- 3. full save_target-equivalent capture on the real DSO block --
        capture_dir = out_dir/'target_capture_pre_setup'
        before, rec, payload = capture_snapshot(
            capture_dir, dso_model, dso_obj, dso_params, None, 'pre_setup', False,
            planning.params.admm, candidate, 0, TARGET,
            extra_meta={'note': 'rehearsal; not a real cycle boundary'})
        summary['items']['pre_setup_capture'] = {'reload_equal': True, 'solver_input_equal': True,
                                                  'semantic_sha256': rec['semantic_sha256']}

        # --- 3b. real _create_smopf_solver ('prepared') boundary + suffix negative/positive controls
        clone_for_suffixes = dso_model.clone()
        e_keys = list(clone_for_suffixes.e.keys())
        first_var = clone_for_suffixes.e[e_keys[0]]
        second_var = clone_for_suffixes.e[e_keys[1]]
        third_var = clone_for_suffixes.e[e_keys[2]]
        clone_for_suffixes.ipopt_zL_out[first_var] = 0.0
        clone_for_suffixes.ipopt_zU_out[second_var] = 3.25
        # a third variable is deliberately left absent from both _out suffixes.
        solver_prepared, log_path, solve_context = N._create_smopf_solver(
            dso_obj, clone_for_suffixes, dso_params, from_warm_start=True)
        prepared_state = model_state(clone_for_suffixes)
        zl_in = next(r for r in prepared_state if r[0] == 'suffix' and r[1].endswith('ipopt_zL_in'))
        zu_in = next(r for r in prepared_state if r[0] == 'suffix' and r[1].endswith('ipopt_zU_in'))
        zl_in_names = {name for name, _ in zl_in[4]}
        zu_in_names = {name for name, _ in zu_in[4]}
        first_name, second_name, third_name = first_var.name, second_var.name, third_var.name
        suffix_distinguishes_absent_from_zero = (
            first_name in zl_in_names and second_name in zu_in_names and third_name not in zl_in_names
            and third_name not in zu_in_names)
        prepared_dir = out_dir/'target_capture_prepared'
        before_p, rec_p, payload_p = capture_snapshot(
            prepared_dir, clone_for_suffixes, dso_obj, dso_params, solver_prepared, 'prepared', True,
            planning.params.admm, candidate, 0, TARGET,
            extra_meta={'note': 'rehearsal; real _create_smopf_solver, never solved'})
        summary['items']['prepared_capture'] = {
            'reload_equal': True, 'solver_input_equal': True,
            'suffix_distinguishes_absent_from_zero': suffix_distinguishes_absent_from_zero,
            'zero_entry_present': first_name in zl_in_names,
            'nonzero_entry_present': second_name in zu_in_names,
            'absent_entry_absent': (third_name not in zl_in_names) and (third_name not in zu_in_names)}

        # --- 4. checkpoint serialization path on a realistic locals mapping
        esso_models = planning.shared_ess_data.build_subproblem()
        synthetic_diagnostics = [{'cycle': 1, 'recourse': float('nan'), 'note': 'rehearsal synthetic'}]
        checkpoint_local = {
            'tso_model': {2025: {'Spring': tso_model}},
            'dso_models': {9: {2025: {'Spring': dso_model.clone()}}},
            'esso_model': esso_models,
            'consensus_vars': consensus_vars, 'dual_vars': dual_vars,
            'candidate_solution': candidate, 'previous_recourse': None,
            'previous_recourse_blocks': {}, 'previous_objective_component_blocks': {},
            'previous_slack_component_blocks': {}, 'previous_tso_voltage_slack_state': {},
            'consecutive_converged_cycles': 0, 'sess_available_capacities': {},
            'admm_diagnostics': synthetic_diagnostics, 'primal_evolution': [],
        }
        checkpoint_rec = build_checkpoint(checkpoint_local, out_dir/'checkpoint_rehearsal', planning,
                                          'rehearsal; synthetic locals mapping on real unsolved models')
        summary['items']['checkpoint'] = {'reload_equal': True, 'blocks': len(checkpoint_rec['blocks']),
                                          'state_sha256': checkpoint_rec['state_sha256'],
                                          'nonfinite_in_admm_diagnostics_survived_roundtrip': True}

        # --- 5(i). negative control: strict encoder still raises on non-finite
        nonfinite_clone = dso_model.clone()
        var_key = list(nonfinite_clone.e.keys())[0]
        nonfinite_clone.e[var_key].set_value(float('nan'))
        try:
            model_state(nonfinite_clone)
            var_control_raised = False
        except Stop:
            var_control_raised = True
        nonfinite_clone2 = dso_model.clone()
        nonfinite_clone2.dual[nonfinite_clone2.node_balance_p[0, 0, 0, 0]] = float('inf')
        try:
            model_state(nonfinite_clone2)
            suffix_control_raised = False
        except Stop:
            suffix_control_raised = True
        summary['items']['negative_control_nonfinite_input'] = {
            'var_value_raises': var_control_raised, 'suffix_value_raises': suffix_control_raised}

        # --- 5(ii). negative control: integrity check FAILS vs historical baseline
        historical_initial_path = RUN_OUT/'initial_repository.json'
        historical_initial = json.loads(historical_initial_path.read_text())
        diffs_hist = integrity_diff(historical_initial, current0, harness_sha_at_start,
                                    historical_initial.get('harness_sha256'))
        expected_added = {'.claude/agents/advisor.md', '.claude/agents/planner.md',
                          '.claude/agents/worker.md', '.claude/settings.json'}
        summary['items']['negative_control_historical_baseline'] = {
            'diffs': sorted(diffs_hist), 'failed_as_expected': bool(diffs_hist),
            'head_mismatch': diffs_hist.get('head'),
            'claude_md_flagged_changed': 'CLAUDE.md' in diffs_hist.get('changed_tracked_files', []),
            'added_files_match_expected': set(diffs_hist.get('added_tracked_files', [])) == expected_added}

        # --- 5(iii). positive control: integrity PASSES vs an in-memory
        # expected state computed from the CURRENT tree (never written as a file)
        in_memory_expected = {'head': current0['head'], 'branch': current0['branch'],
                              'staged_diff': current0['staged_diff'],
                              'tracked_hashes': current0['tracked_hashes'],
                              'harness_sha256': harness_sha_at_start}
        diffs_self = integrity_diff(in_memory_expected, current_repo_state(), harness_sha_at_start,
                                    harness_sha_at_start)
        summary['items']['positive_control_in_memory_baseline'] = {
            'diffs': sorted(diffs_self), 'passed': not diffs_self}

        # --- 5(iv). self-hash start == end (checked again at the very end) --
        summary['items']['harness_sha256_start'] = harness_sha_at_start

        # --- 6. measured counters ------------------------------------------
        summary['items']['counters'] = dict(counters)

        # --- 7. no-solve negative control of final_verdict() ---------------
        # Pure function calls on synthetic report dicts only: no git, no
        # filesystem beyond the rehearsal summary, no solver, no numerical
        # capture simulated.
        def apply_run_main_combination(prior_status, verdict_status):
            # Mirrors run_main()'s combination rule exactly: final_verdict
            # may only downgrade an in-progress CAPTURED to STOPPED, never
            # upgrade a pre-existing STOPPED to CAPTURED.
            if prior_status == 'CAPTURED' and verdict_status != 'CAPTURED':
                return 'STOPPED'
            return prior_status

        good_report = {
            'status': 'CAPTURED',
            'integrity_after': {'diffs': {}, 'head': 'deadbeefcafe', 'branch': 'feature/x',
                                 'tracked_file_count': 1946},
            'counters': {'guard_calls_target': 0, 'original_solve_calls_target': 0,
                         'process_launches_by_context': [{'cycle': 20, 'context': 'DSO:example'}]},
            'target_log': {'cycle21_pre_setup_offset': {'path': '/synthetic/target.log', 'offset': 12345},
                           'size_at_exit': 12345, 'size_at_exit_path': '/synthetic/target.log',
                           'file_existed_at_exit': True},
        }

        negative_control_cases = {}

        # (e) fully consistent synthetic report -> CAPTURED
        status_e, checks_e, failures_e = final_verdict(good_report, TARGET)
        negative_control_cases['e_consistent_report_captured'] = {
            'input': deepcopy(good_report), 'status': status_e, 'failures': failures_e, 'checks': checks_e,
            'expected_status': 'CAPTURED', 'matched_expectation': status_e == 'CAPTURED'}

        # (a) integrity_after.diffs non-empty -> STOPPED
        report_a = deepcopy(good_report)
        report_a['integrity_after']['diffs'] = {'head': {'expected': 'deadbeefcafe', 'actual': 'other'}}
        status_a, checks_a, failures_a = final_verdict(report_a, TARGET)
        negative_control_cases['a_nonempty_integrity_diffs'] = {
            'input': report_a, 'status': status_a, 'failures': failures_a, 'checks': checks_a,
            'expected_status': 'STOPPED',
            'matched_expectation': status_a == 'STOPPED' and 'integrity_head_unchanged' in failures_a
                                    and 'integrity_diffs_empty' in failures_a}

        # (b) guard_calls_target = 1
        report_b1 = deepcopy(good_report); report_b1['counters']['guard_calls_target'] = 1
        status_b1, checks_b1, failures_b1 = final_verdict(report_b1, TARGET)
        negative_control_cases['b1_guard_calls_target_nonzero'] = {
            'input': report_b1, 'status': status_b1, 'failures': failures_b1, 'checks': checks_b1,
            'expected_status': 'STOPPED',
            'matched_expectation': status_b1 == 'STOPPED' and 'guard_calls_target_zero' in failures_b1}

        # (b) original_solve_calls_target = 1
        report_b2 = deepcopy(good_report); report_b2['counters']['original_solve_calls_target'] = 1
        status_b2, checks_b2, failures_b2 = final_verdict(report_b2, TARGET)
        negative_control_cases['b2_original_solve_calls_target_nonzero'] = {
            'input': report_b2, 'status': status_b2, 'failures': failures_b2, 'checks': checks_b2,
            'expected_status': 'STOPPED',
            'matched_expectation': status_b2 == 'STOPPED'
                                    and 'original_solve_calls_target_zero' in failures_b2}

        # (b) a process launch at cycle 21 for TARGET
        report_b3 = deepcopy(good_report)
        report_b3['counters']['process_launches_by_context'] = [{'cycle': 21, 'context': TARGET}]
        status_b3, checks_b3, failures_b3 = final_verdict(report_b3, TARGET)
        negative_control_cases['b3_process_launch_cycle21_target'] = {
            'input': report_b3, 'status': status_b3, 'failures': failures_b3, 'checks': checks_b3,
            'expected_status': 'STOPPED',
            'matched_expectation': status_b3 == 'STOPPED'
                                    and 'no_process_launch_for_cycle21_target' in failures_b3}

        # (c) size_at_exit advanced past the recorded cycle21 offset
        report_c = deepcopy(good_report); report_c['target_log']['size_at_exit'] = 99999
        status_c, checks_c, failures_c = final_verdict(report_c, TARGET)
        negative_control_cases['c_target_log_advanced'] = {
            'input': report_c, 'status': status_c, 'failures': failures_c, 'checks': checks_c,
            'expected_status': 'STOPPED',
            'matched_expectation': status_c == 'STOPPED' and 'target_log_did_not_advance' in failures_c}

        # (d) missing integrity_after entirely
        report_d1 = deepcopy(good_report); del report_d1['integrity_after']
        status_d1, checks_d1, failures_d1 = final_verdict(report_d1, TARGET)
        negative_control_cases['d1_missing_integrity_after'] = {
            'input': report_d1, 'status': status_d1, 'failures': failures_d1, 'checks': checks_d1,
            'expected_status': 'STOPPED',
            'matched_expectation': status_d1 == 'STOPPED' and 'integrity_after_present' in failures_d1}

        # (d) missing target-log evidence (no cycle21 offset, no size_at_exit)
        report_d2 = deepcopy(good_report); report_d2['target_log'] = {}
        status_d2, checks_d2, failures_d2 = final_verdict(report_d2, TARGET)
        negative_control_cases['d2_missing_target_log_evidence'] = {
            'input': report_d2, 'status': status_d2, 'failures': failures_d2, 'checks': checks_d2,
            'expected_status': 'STOPPED',
            'matched_expectation': status_d2 == 'STOPPED' and 'target_log_did_not_advance' in failures_d2
                                    and 'target_log_file_present_at_exit' in failures_d2}

        # (f) pre-existing STOPPED status stays STOPPED even with all-good evidence
        combined_e = apply_run_main_combination('CAPTURED', status_e)
        combined_f = apply_run_main_combination('STOPPED', status_e)
        negative_control_cases['f_preexisting_stopped_not_upgraded'] = {
            'prior_status': 'STOPPED', 'verdict_status_on_good_evidence': status_e,
            'combined_status': combined_f, 'expected_status': 'STOPPED',
            'matched_expectation': combined_f == 'STOPPED',
            'control_combined_status_for_captured_prior': combined_e,
            'control_matched_expectation': combined_e == status_e}

        all_negative_controls_matched = all(c['matched_expectation'] for c in negative_control_cases.values())
        summary['items']['final_verdict_negative_controls'] = {
            'cases': negative_control_cases, 'all_matched_expectation': all_negative_controls_matched}
        dump(out_dir/'final_verdict_negative_controls.json', negative_control_cases)

        summary['status'] = 'COMPLETED'
    except BaseException as e:
        summary['status'] = 'STOPPED'
        summary['stop_reason'] = type(e).__name__+': '+str(e)
        summary['traceback'] = traceback.format_exc()
    finally:
        OptSolver.solve = original_solve
        uninstall_process_launch_counter(original_shellcmd)
        harness_sha_at_end = harness_sha256()
        summary['items']['harness_sha256_end'] = harness_sha_at_end
        summary['items']['harness_sha256_stable'] = (harness_sha_at_end == harness_sha_at_start)
        summary['items']['final_counters'] = dict(counters)
        with (out_dir/'rehearsal_summary.json').open('x') as f:
            json.dump(json_safe(summary), f, indent=1, allow_nan=False)
    return summary


def rehearse_main(out_dir):
    start = time.time()
    summary = rehearse(out_dir)
    print('[P512R-REHEARSAL]', summary['status'], summary.get('stop_reason'), flush=True)
    print(f'[P512R-REHEARSAL] wall_seconds={time.time()-start:.2f}', flush=True)
    sys.exit(0 if summary['status'] == 'COMPLETED' else 2)


# ===========================================================================
# CLI
# ===========================================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog='p512_r_presolve_recapture.py',
        description='P5.12-R capture harness: run (trajectory + target capture) or rehearse (no-solve dry run).')
    sub = parser.add_subparsers(dest='mode')
    run_p = sub.add_parser('run', help='execute the P5.12-R trajectory (not authorized by rehearsal-repair task)')
    run_p.add_argument('--baseline-dir', required=True,
                       help='directory containing provenance.json, initial_repository.json, '
                            'runtime_identity.json, accepted_hashes.json for THIS run')
    reh_p = sub.add_parser('rehearse', help='no-solve rehearsal of the capture machinery')
    reh_p.add_argument('--out-dir', required=True, help='new, non-existent output directory')
    return parser


if __name__ == '__main__':
    require(Path.cwd() == ROOT, 'wrong repository root')
    require(sys.executable == '/Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python', 'wrong interpreter')
    arg_parser = build_arg_parser()
    parsed_args = arg_parser.parse_args()
    if parsed_args.mode == 'run':
        run_main(parsed_args.baseline_dir)
    elif parsed_args.mode == 'rehearse':
        rehearse_main(parsed_args.out_dir)
    else:
        arg_parser.print_help(sys.stderr)
        sys.exit(2)
