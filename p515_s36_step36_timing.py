"""
P5.15 Step 3.6 (`PLANNER_BRIEF_2026-09-13.md` Addendum 18/19/20; design
`P5_15_STEP36_TIMING_DESIGN.md`, commit `2d9ebfc1`) -- per-phase, per-block
cycle-time instrumentation, implemented HARNESS-SIDE (Planner deviation from
the design's `timing_recorder=None` threaded-kwarg proposal).

**No production file is edited or imported for the purpose of being changed.**
Every wrap point below is installed by *monkeypatching a module attribute or a
class attribute* at the exact name the real caller resolves (verified, with
citations, in `WORKER_REPORT_S36_TIMING.md` and re-checked mechanically by
`p515_s36_step36_timing_checks.py`), and is removed again in a `finally`
block, restoring the *exact* object (`is`-identity) that was there before
installation -- so composing this module with another monkeypatch already
installed around it (in particular `p513_solve_profile_guard.SolveProfileGuard`,
which the campaign harness `p515_g_g1_g4_admm_gates.py` already installs around
`planning.run_operational_planning(...)`) is safe as long as this module's own
`with recorder_installed(...):` block is opened OUTSIDE the guard's `with`/
`install()`/`uninstall()` pair (LIFO nesting -- see `WORKER_REPORT_S36_TIMING.md`
"Composability with SolveProfileGuard").

======================================================================
1. DATA MODEL
======================================================================

A `PhaseTimingRecorder` accumulates `PhaseRecord` entries in memory (a Python
list under a `threading.Lock` -- defensive; the measured path is single
process / single thread, `parallel_execution=False`, per the s35ref reference
configuration this instrumentation is sized against). Each entry:

    cycle    : int | None   -- 1-based ADMM cycle number. Production's ADMM
                               loop variable `iter` (`shared_resources_planning.py:2501`)
                               is directly available as a bound argument only
                               at the three per-agent dispatch call sites
                               (`update_distribution_coordination_models_and_solve`,
                               `update_transmission_coordination_model_and_solve`,
                               `update_shared_energy_storages_coordination_model_and_solve`,
                               each declares `cycle=None` and is always called
                               with `cycle=iter`) and at
                               `_update_tso_proximal_centres_after_solve`
                               (also `cycle=iter`). Every other wrap point
                               (`update_and_check_convergence`,
                               `get_admm_residual_metrics`,
                               `get_admm_boyd_residual_metrics`,
                               `_update_admm_penalties` -- the last binds
                               `iter=iter`, a different parameter NAME for the
                               same value -- and every block-level wrap) has NO
                               `cycle`/`iter` parameter in its own signature or
                               call chain, so the recorder falls back to its
                               OWN `current_cycle` attribute, updated at the
                               ENTRY of each per-agent dispatch wrap (DSO is
                               always stage 1 of a cycle, so its entry is a
                               reliable per-cycle boundary). This is a
                               HARNESS-SIDE derived cycle count, not a read of
                               production's `iter` variable, for any wrap point
                               other than the four named above -- stated
                               explicitly, not asserted as an exact reproduction
                               of `iter`.
    agent    : str           -- 'dso' | 'tso' | 'esso' | 'admm_global' | 'unattributed'.
    block    : dict | None   -- JSON-serializable block key. For DSO/TSO block-level
                               records: {'kind': 'dso'|'tso', 'name': <Network.name>,
                               'year': <Network.year>, 'day': <Network.day>} (from the
                               `self` argument of `Network.run_smopf`, NOT re-derived).
                               For ESSO block-level records: {'kind': 'esso',
                               'node_id': <int>}. For stage-level records (`stage_total`,
                               and the admm_global phase): {'kind': 'dso'|'tso'|'esso'|'global'}
                               only (no year/day/node -- the call site does not carry one).
    phase    : str            -- one of PHASES (below) plus the two internal-only
                               bookkeeping tags 'stage_total' and 'block_total' used
                               only as derivation inputs for 'param_update' and
                               'bookkeeping' (see `analyze_phase_timing`).
    attempt  : str            -- 'primary' | 'tier1_recovery' | 'tier2_recovery' | 'n/a'.
    start_perf, end_perf, elapsed_s : float -- `time.perf_counter()` pair and their
                               difference. A monotonic clock read has no side effect on
                               any Pyomo object, solver option or file (design §2.4).
    seq      : int             -- global monotonically increasing sequence number
                               (insertion order), used by `analyze_phase_timing` to
                               recover DSO per-node `param_update` from inter-event gaps.

The eight PUBLIC phase categories the design (§1/§2/§5) and the task ask for:

    PHASES = ('param_update', 'clone', 'solve_bundle', 'load_solution',
              'bookkeeping', 'diagnostics_parse', 'admm_global', 'unattributed')

======================================================================
2. WRAP POINTS (see WORKER_REPORT_S36_TIMING.md for the file:line citations
   and the caller-resolution evidence for each)
======================================================================

Stage total (used only to derive TSO/ESSO stage-level `param_update` and as
the DSO per-node gap-derivation's left boundary for the first node):
  - shared_resources_planning.update_distribution_coordination_models_and_solve
  - shared_resources_planning.update_transmission_coordination_model_and_solve
  - shared_resources_planning.update_shared_energy_storages_coordination_model_and_solve

Block total (used to derive `bookkeeping` as a residual):
  - network.Network.run_smopf                          (per DSO/TSO (year,day) block)
  - shared_energy_storage_data._optimize                (per ESSO node)

clone (phase 'clone'; timed ONLY when the immediate caller frame is
network_data.py:NetworkData.optimize -- i.e. exactly the TSO-unconditional /
DSO-node-7 clone the design's §1.1-1.2 identifies; every OTHER `.clone()` call
anywhere in the process, e.g. `_clone_operational_models`
(shared_resources_planning.py) for a warm continuation, is passed through
UNTIMED, not folded into any bucket -- it is out of this instrumentation's
declared scope, not silently absorbed):
  - pyomo.core.base.block.BlockData.clone

solve_bundle (phase 'solve_bundle', = b+c+d1 combined per design §2.5; timed
regardless of ancestry, attributed via a frame walk to whichever of the two
known call sites the timed call descends from; a call reached from NEITHER
known site is still recorded, tagged agent='unattributed', phase='solve_bundle'
so no time is silently dropped):
  - pyomo.opt.base.solvers.OptSolver.solve
    (ancestry: network.py:_run_smopf_solver_attempt -> agent from its `network`
     local (`.is_transmission`/.name/.year/.day), attempt from its `log_suffix`
     local (None -> primary, 'recovery' -> tier1_recovery, 'recovery_tier2' ->
     tier2_recovery); OR shared_energy_storage_data.py:_run_solver_attempt ->
     agent='esso', block={'node_id': <local `node_id`>}, attempt from `log_suffix`.)

load_solution (phase 'load_solution', = d2 alone per design §2.5, NOT covered
by Pyomo's own `report_timing` since production always solves with
`load_solutions=False`):
  - pyomo.core.base.PyomoModel.ModelSolutions.load_from
    (ancestry: network.py:_run_smopf -> agent from its `network` local, attempt
     from its `recovery_attempted`/`tier2_attempted` locals; OR
     shared_energy_storage_data.py:_optimize -> agent='esso', block from its
     `node_id` local, attempt from `recovery_attempted`/`tier2_attempted`.)

diagnostics_parse (phase 'diagnostics_parse', ESSO-only, design §1.3):
  - shared_energy_storage_data._get_esso_complementarity_diagnostics
    (direct args carry `node_id`; `cycle` is read from the immediate caller
     frame's `cycle` local, since the function itself has no `cycle` parameter.)

admm_global (phase 'admm_global', design §1.4):
  - shared_resources_planning.update_and_check_convergence   (x3 per cycle)
  - shared_resources_planning.get_admm_residual_metrics
  - shared_resources_planning.get_admm_boyd_residual_metrics
  - shared_resources_planning._update_tso_proximal_centres_after_solve
  - shared_resources_planning._update_admm_penalties

param_update (phase 'param_update') and bookkeeping (phase 'bookkeeping') are
NOT live-wrapped -- both are DERIVED, in `analyze_phase_timing`, exactly as the
task specifies: "param_update per block/stage = stage total minus the enclosed
optimize time". See that function's docstring for the exact per-agent formula
(DSO resolves to per-NODE param_update via inter-event gaps because its
set_value loop and its `NetworkData.optimize()` call are INTERLEAVED per node;
TSO and ESSO do each in one shot for the WHOLE stage before any block solves,
so their param_update is stage-level only, not resolvable per block without
editing production -- a finding, not a workaround).

unattributed (phase 'unattributed'): computed only in `analyze_phase_timing`,
as the residual between production's own OWN printed per-cycle wall time
(`shared_resources_planning.py:3007`, `"[INFO] \\t - Iteration {iter}: {X:.2f} s"`,
parsed from the run's captured stdout by the CALLER of `analyze_phase_timing`,
not by this module) and this recorder's own reconstructed per-cycle span (first
event start to last event end for that cycle). This captures whatever precedes
the first wrap (nothing, in production's cycle body -- DSO is called right
after `iter_start`) and whatever follows the last wrap (the worst-primal/worst-
pf diagnostic prints and the iteration-time print itself) -- small by
construction, reported so nothing is silently unaccounted for.

======================================================================
3. WHAT COULD NOT BE ATTRIBUTED WITHOUT EDITING PRODUCTION
======================================================================
See "Unattributed" in `WORKER_REPORT_S36_TIMING.md` for the full list; the
short version:
  - The very first (pre-ADMM-loop) initialization solve goes through
    `create_distribution_networks_models` / `create_transmission_network_model`
    / `create_shared_energy_storage_model`, DIFFERENT functions from the
    Planner's candidate wrap list -- not wrapped, not billed to any cycle
    (the design's own two-cycle measurement plan is itself an ADMM-cycle-only
    measurement, consistent with this).
  - TSO/ESSO per-block `param_update` (see above).
  - The exact NL-write (b) vs IPOPT subprocess (c) vs `.sol` parse (d1) split
    WITHIN `solve_bundle` requires Pyomo's own `report_timing=True` print
    output (design §2.5), which this module can OPTIONALLY inject into the
    `solver.solve(...)` call it wraps (see `install_report_timing_crosscheck`
    below) for exactly the one designated cross-check run design §5
    specifies; without it, `solve_bundle` is reported as one combined number
    and `overhead_local`'s NL-write share is reported as a documented
    UPPER BOUND (the full `solve_bundle`), not a measured value -- see
    `analyze_phase_timing`.
"""

import inspect
import json
import statistics
import sys
import threading
import time
from contextlib import contextmanager

PHASES = ('param_update', 'clone', 'solve_bundle', 'load_solution',
          'bookkeeping', 'diagnostics_parse', 'admm_global', 'unattributed')

# Internal-only bookkeeping tags, never reported as a top-level phase category
# (used purely as derivation inputs inside analyze_phase_timing).
_INTERNAL_PHASES = ('stage_total', 'block_total')

_ATTEMPT_PRIMARY = 'primary'
_ATTEMPT_TIER1 = 'tier1_recovery'
_ATTEMPT_TIER2 = 'tier2_recovery'
_ATTEMPT_NA = 'n/a'
_ATTEMPT_UNKNOWN = 'unknown'


# ======================================================================
#  Recorder
# ======================================================================
class PhaseTimingRecorder:
    """In-memory accumulator. `record()` is called from inside try/finally
    wrappers only (see the wrap installers below) so a recording call never
    changes what exception, if any, propagates from the wrapped call."""

    def __init__(self):
        self._lock = threading.Lock()
        self.records = []          # list[dict]
        self.current_cycle = None  # updated at DSO/TSO/ESSO stage-wrap entry
        self._seq = 0

    def record(self, cycle, agent, block, phase, attempt, start, end):
        with self._lock:
            self._seq += 1
            self.records.append({
                'seq': self._seq,
                'cycle': cycle,
                'agent': agent,
                'block': block,
                'phase': phase,
                'attempt': attempt,
                'start_perf': start,
                'end_perf': end,
                'elapsed_s': end - start,
            })

    def set_current_cycle(self, cycle):
        with self._lock:
            if cycle is not None:
                self.current_cycle = cycle

    def get_current_cycle(self):
        with self._lock:
            return self.current_cycle

    def to_jsonl(self, path):
        with self._lock:
            rows = list(self.records)
        with open(path, 'w') as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write('\n')
        return len(rows)


# ======================================================================
#  Frame-walk attribution (same technique as
#  p513_solve_profile_guard.SolveProfileGuard._matching_frame -- walk UP the
#  live call stack looking for a named (file-suffix, function-name) frame,
#  rather than assuming a fixed depth, so this is robust to how many wrapper
#  frames (ours, SolveProfileGuard's, ...) sit between the real production
#  caller and the point where this code runs.)
# ======================================================================
def _find_ancestor_frame(filename_suffix, function_name, start_depth=2, max_depth=80):
    frame = sys._getframe(start_depth)
    depth = 0
    while frame is not None and depth < max_depth:
        code = frame.f_code
        if code.co_name == function_name and code.co_filename.endswith(filename_suffix):
            return frame
        frame = frame.f_back
        depth += 1
    return None


def _attempt_from_log_suffix(log_suffix):
    if log_suffix is None:
        return _ATTEMPT_PRIMARY
    if log_suffix == 'recovery':
        return _ATTEMPT_TIER1
    if log_suffix == 'recovery_tier2':
        return _ATTEMPT_TIER2
    return _ATTEMPT_UNKNOWN


def _attempt_from_flags(tier2_attempted, recovery_attempted):
    if tier2_attempted:
        return _ATTEMPT_TIER2
    if recovery_attempted:
        return _ATTEMPT_TIER1
    return _ATTEMPT_PRIMARY


# ======================================================================
#  Individual wrap installers.
#
#  Each `_install_*` function:
#    1. imports the target module/class lazily (so importing THIS module never
#       imports Pyomo/production eagerly as a side effect beyond what the
#       caller already imported);
#    2. captures `original = <current attribute value>` (NOT assumed to be the
#       pristine production object -- composes correctly with an outer
#       monkeypatch already in place, e.g. SolveProfileGuard, per the module
#       docstring);
#    3. installs a wrapper that times the call in a try/finally and delegates
#       to `original`, preserving return value and exception;
#    4. returns a zero-argument `restore()` callable that sets the attribute
#       back to `original` -- restoring IDENTITY, not a copy.
# ======================================================================

def _install_stage_total(recorder, module, func_name, agent):
    original = getattr(module, func_name)
    signature = inspect.signature(original)

    def wrapper(*args, **kwargs):
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            cycle = bound.arguments.get('cycle')
        except TypeError:
            cycle = None
        if cycle is not None:
            recorder.set_current_cycle(cycle)
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(recorder.get_current_cycle(), agent, {'kind': agent},
                             'stage_total', _ATTEMPT_NA, start, end)

    setattr(module, func_name, wrapper)
    return lambda: setattr(module, func_name, original)


def _install_admm_global(recorder, module, func_name, cycle_kwarg_names=()):
    original = getattr(module, func_name)
    signature = inspect.signature(original)

    def wrapper(*args, **kwargs):
        cycle = None
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in cycle_kwarg_names:
                if bound.arguments.get(name) is not None:
                    cycle = bound.arguments[name]
                    break
        except TypeError:
            cycle = None
        if cycle is None:
            cycle = recorder.get_current_cycle()
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(cycle, 'admm_global', {'kind': 'global', 'call': func_name},
                             'admm_global', _ATTEMPT_NA, start, end)

    setattr(module, func_name, wrapper)
    return lambda: setattr(module, func_name, original)


def _install_network_run_smopf(recorder):
    from network import Network
    original = Network.run_smopf

    def wrapper(self, *args, **kwargs):
        agent = 'tso' if getattr(self, 'is_transmission', False) else 'dso'
        block = {'kind': agent, 'name': self.name, 'year': self.year, 'day': self.day}
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(recorder.get_current_cycle(), agent, block,
                             'block_total', _ATTEMPT_NA, start, end)

    Network.run_smopf = wrapper
    return lambda: setattr(Network, 'run_smopf', original)


def _install_esso_optimize(recorder):
    import shared_energy_storage_data as SED
    original = SED._optimize

    def wrapper(*args, **kwargs):
        try:
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            node_id = bound.arguments.get('node_id')
            cycle = bound.arguments.get('cycle')
        except TypeError:
            node_id, cycle = None, None
        block = {'kind': 'esso', 'node_id': node_id}
        if cycle is not None:
            recorder.set_current_cycle(cycle)
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(cycle if cycle is not None else recorder.get_current_cycle(),
                             'esso', block, 'block_total', _ATTEMPT_NA, start, end)

    SED._optimize = wrapper
    return lambda: setattr(SED, '_optimize', original)


def _install_clone(recorder):
    from pyomo.core.base.block import BlockData
    original = BlockData.clone

    def wrapper(self, *args, **kwargs):
        frame = _find_ancestor_frame('network_data.py', 'optimize')
        block = None
        agent = None
        if frame is not None:
            holder = frame.f_locals.get('self')
            year = frame.f_locals.get('year')
            day = frame.f_locals.get('day')
            if holder is not None:
                agent = 'tso' if getattr(holder, 'is_transmission', False) else 'dso'
                block = {'kind': agent, 'name': getattr(holder, 'name', None),
                         'year': year, 'day': day}
        if frame is None:
            # Out of declared scope (e.g. shared_resources_planning.py's
            # `_clone_operational_models` for a warm continuation) -- pass
            # through UNTIMED, per the module docstring.
            return original(self, *args, **kwargs)
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(recorder.get_current_cycle(), agent, block,
                             'clone', _ATTEMPT_NA, start, end)

    BlockData.clone = wrapper
    return lambda: setattr(BlockData, 'clone', original)


def _install_solve_bundle(recorder, inject_report_timing=False):
    from pyomo.opt.base.solvers import OptSolver
    original = OptSolver.solve

    def wrapper(self, *args, **kwargs):
        agent, block, attempt = 'unattributed', None, _ATTEMPT_UNKNOWN
        net_frame = _find_ancestor_frame('network.py', '_run_smopf_solver_attempt')
        esso_frame = None if net_frame is not None else \
            _find_ancestor_frame('shared_energy_storage_data.py', '_run_solver_attempt')
        if net_frame is not None:
            network_obj = net_frame.f_locals.get('network')
            log_suffix = net_frame.f_locals.get('log_suffix')
            if network_obj is not None:
                agent = 'tso' if getattr(network_obj, 'is_transmission', False) else 'dso'
                block = {'kind': agent, 'name': getattr(network_obj, 'name', None),
                         'year': getattr(network_obj, 'year', None),
                         'day': getattr(network_obj, 'day', None)}
            attempt = _attempt_from_log_suffix(log_suffix)
        elif esso_frame is not None:
            node_id = esso_frame.f_locals.get('node_id')
            log_suffix = esso_frame.f_locals.get('log_suffix')
            agent = 'esso'
            block = {'kind': 'esso', 'node_id': node_id}
            attempt = _attempt_from_log_suffix(log_suffix)
        if inject_report_timing:
            kwargs = dict(kwargs)
            kwargs.setdefault('report_timing', True)
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(recorder.get_current_cycle(), agent, block,
                             'solve_bundle', attempt, start, end)

    OptSolver.solve = wrapper
    return lambda: setattr(OptSolver, 'solve', original)


def _install_load_solution(recorder):
    from pyomo.core.base.PyomoModel import ModelSolutions
    original = ModelSolutions.load_from

    def wrapper(self, *args, **kwargs):
        agent, block, attempt = 'unattributed', None, _ATTEMPT_UNKNOWN
        net_frame = _find_ancestor_frame('network.py', '_run_smopf')
        esso_frame = None if net_frame is not None else \
            _find_ancestor_frame('shared_energy_storage_data.py', '_optimize')
        if net_frame is not None:
            network_obj = net_frame.f_locals.get('network')
            if network_obj is not None:
                agent = 'tso' if getattr(network_obj, 'is_transmission', False) else 'dso'
                block = {'kind': agent, 'name': getattr(network_obj, 'name', None),
                         'year': getattr(network_obj, 'year', None),
                         'day': getattr(network_obj, 'day', None)}
            attempt = _attempt_from_flags(net_frame.f_locals.get('tier2_attempted'),
                                          net_frame.f_locals.get('recovery_attempted'))
        elif esso_frame is not None:
            node_id = esso_frame.f_locals.get('node_id')
            agent = 'esso'
            block = {'kind': 'esso', 'node_id': node_id}
            attempt = _attempt_from_flags(esso_frame.f_locals.get('tier2_attempted'),
                                          esso_frame.f_locals.get('recovery_attempted'))
        start = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(recorder.get_current_cycle(), agent, block,
                             'load_solution', attempt, start, end)

    ModelSolutions.load_from = wrapper
    return lambda: setattr(ModelSolutions, 'load_from', original)


def _install_diagnostics_parse(recorder):
    import shared_energy_storage_data as SED
    original = SED._get_esso_complementarity_diagnostics

    def wrapper(model, node_id, log_path, *args, **kwargs):
        frame = _find_ancestor_frame('shared_energy_storage_data.py', '_optimize')
        cycle = frame.f_locals.get('cycle') if frame is not None else None
        block = {'kind': 'esso', 'node_id': node_id}
        start = time.perf_counter()
        try:
            return original(model, node_id, log_path, *args, **kwargs)
        finally:
            end = time.perf_counter()
            recorder.record(cycle if cycle is not None else recorder.get_current_cycle(),
                             'esso', block, 'diagnostics_parse', _ATTEMPT_NA, start, end)

    SED._get_esso_complementarity_diagnostics = wrapper
    return lambda: setattr(SED, '_get_esso_complementarity_diagnostics', original)


# ======================================================================
#  Installer context manager
# ======================================================================
@contextmanager
def recorder_installed(recorder, inject_report_timing=False):
    """Install every wrap point listed in the module docstring's section 2,
    in a fixed order, and guarantee exact-identity restoration in `finally`
    even if a later installer in the sequence raises (each earlier one is
    still unwound). `inject_report_timing`: forwarded to the `solve_bundle`
    wrap only -- see design §2.5; use for exactly the one designated
    cross-check run (design §5), never for the plain bitwise-identity
    determinism pair.

    Import shared_resources_planning as srp
    """
    import shared_resources_planning as srp

    restores = []
    try:
        restores.append(_install_stage_total(
            recorder, srp, 'update_distribution_coordination_models_and_solve', 'dso'))
        restores.append(_install_stage_total(
            recorder, srp, 'update_transmission_coordination_model_and_solve', 'tso'))
        restores.append(_install_stage_total(
            recorder, srp, 'update_shared_energy_storages_coordination_model_and_solve', 'esso'))
        restores.append(_install_admm_global(recorder, srp, 'update_and_check_convergence'))
        restores.append(_install_admm_global(recorder, srp, 'get_admm_residual_metrics'))
        restores.append(_install_admm_global(recorder, srp, 'get_admm_boyd_residual_metrics'))
        restores.append(_install_admm_global(
            recorder, srp, '_update_tso_proximal_centres_after_solve', cycle_kwarg_names=('cycle',)))
        restores.append(_install_admm_global(
            recorder, srp, '_update_admm_penalties', cycle_kwarg_names=('iter', 'cycle')))
        restores.append(_install_network_run_smopf(recorder))
        restores.append(_install_esso_optimize(recorder))
        restores.append(_install_clone(recorder))
        restores.append(_install_solve_bundle(recorder, inject_report_timing=inject_report_timing))
        restores.append(_install_load_solution(recorder))
        restores.append(_install_diagnostics_parse(recorder))
        yield recorder
    finally:
        for restore in reversed(restores):
            restore()


# ======================================================================
#  Analysis
# ======================================================================
def _stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'count': 0, 'median': None, 'mean': None, 'max': None, 'sum': None}
    return {
        'count': len(values),
        'median': statistics.median(values),
        'mean': statistics.mean(values),
        'max': max(values),
        'sum': sum(values),
    }


def _block_key(record):
    block = record.get('block') or {}
    return (record.get('agent'), record.get('cycle'),
            block.get('kind'), block.get('name'), block.get('year'),
            block.get('day'), block.get('node_id'))


def derive_param_update_and_bookkeeping(records):
    """Derive `param_update` and `bookkeeping` phase records from the raw
    `stage_total` / `block_total` / `clone` / `solve_bundle` / `load_solution`
    / `diagnostics_parse` wrap events, per the task's own instruction:
    "param_update per block/stage = stage total minus the enclosed optimize
    time".

    DSO: `update_distribution_coordination_models_and_solve_sequential`
    (shared_resources_planning.py:5324-5432) INTERLEAVES, per node, its
    `set_value` loop with the single `distribution_network.optimize(...)`
    call for that node (which internally fires one `Network.run_smopf`
    'block_total' event per (year, day) of that node, in file/declaration
    order). So DSO param_update IS resolvable per node: for the first node
    in a cycle's DSO stage, param_update = (start of its first block_total
    event) - (start of the stage_total event); for every subsequent node,
    param_update = (start of its first block_total event) - (end of the
    previous node's LAST block_total event). This is computed here from
    event order (`seq`) within one (cycle, agent='dso') stage_total span,
    grouping consecutive block_total events by their `name` field (a new
    `name` value marks a new node, since DSO always finishes one node's
    (year,day) loop before starting the next -- shared_resources_planning.py
    :5329-5420, `for node_id in distribution_networks:` is the outer loop).

    TSO and ESSO do their ENTIRE `set_value` loop for ALL blocks BEFORE
    calling `.optimize()`/`shared_ess_data.optimize()` (not interleaved),
    so their param_update is NOT separable per block -- reported as ONE
    stage-level number: stage_total - sum(block_total for that stage's
    cycle). This asymmetry is a finding (see WORKER_REPORT_S36_TIMING.md),
    not fixed here.

    `bookkeeping` (phase 'e' minus diagnostics_parse) is, for every
    (cycle, agent, block) with a `block_total` event:
        bookkeeping = block_total
                      - sum(solve_bundle for that block, all attempts)
                      - sum(load_solution for that block, all attempts)
                      - sum(diagnostics_parse for that block, ESSO only)
    `clone` is DELIBERATELY EXCLUDED from this subtraction (P5.15 Step 3.6
    Worker task, defect D1 -- the original version of this function
    subtracted `clone` here too, which produced a negative `bookkeeping`
    total, e.g. -8.37 s aggregate in the v1 measurement). Re-reading the
    actual nesting (`network_data.py:54-68`, `NetworkData.optimize`):

        pre_solve_model = model[year][day].clone()               # :61
        results[year][day] = self.network[year][day].run_smopf(  # :62-63
            model[year][day], self.params, ...)

    `clone()` (:61) is a SIBLING call that runs BEFORE `run_smopf()` (:62-63,
    the `block_total` wrap), not a call nested INSIDE it -- `block_total`'s
    measured span never includes any `clone()` time at all (confirmed by
    reading the source, not assumed from the phase name). Subtracting it from
    `block_total` therefore subtracts time that was never part of that span,
    which is exactly why v1 went negative. `clone` remains its own reported
    phase (already summed separately into `overhead_local_components.clone`
    in `analyze_phase_timing`); it is simply never used to reduce
    `block_total` here.

    A negative bookkeeping value should not occur now that the wrap the
    subtraction is applied against (`block_total`) strictly and only encloses
    `solve_bundle`+`load_solution`+`diagnostics_parse` (verified against
    `network.py:673-777` / `shared_energy_storage_data.py:1177-1330`, both
    read again for this task) -- so, unlike the prior version, a negative
    result here is now treated as a CORRECTNESS DEFECT, not a signal to note
    and move on: `_raise_if_negative_duration` below raises `ValueError`
    (never silently reports a negative `elapsed_s`) whenever a derived
    `param_update` or `bookkeeping` record's `elapsed_s` is more negative
    than `_DERIVATION_NEGATIVE_TOLERANCE` (a small floating-point-noise
    allowance, not a license to hide a real negative value).

    Returns a NEW list of derived records (phase in {'param_update',
    'bookkeeping'}), in the SAME record schema as the live-wrapped ones,
    with `'derived': True` added.
    """
    by_cycle_agent = {}
    for r in records:
        by_cycle_agent.setdefault((r['cycle'], r['agent']), []).append(r)

    derived = []

    # ---- param_update ----
    for (cycle, agent), group in by_cycle_agent.items():
        stage_events = [r for r in group if r['phase'] == 'stage_total']
        block_events = sorted((r for r in group if r['phase'] == 'block_total'),
                               key=lambda r: r['seq'])
        if not stage_events or not block_events:
            continue
        stage = stage_events[0]
        if agent == 'dso':
            prev_end = stage['start_perf']
            prev_name = None
            for be in block_events:
                name = (be.get('block') or {}).get('name')
                if name != prev_name:
                    if prev_name is not None:
                        # boundary is the END of the previous node's LAST
                        # block_total event, i.e. prev_end already holds it
                        pass
                    pu_start = prev_end
                    pu_end = be['start_perf']
                    derived.append({
                        'seq': None, 'cycle': cycle, 'agent': agent,
                        'block': {'kind': 'dso', 'name': name},
                        'phase': 'param_update', 'attempt': _ATTEMPT_NA,
                        'start_perf': pu_start, 'end_perf': pu_end,
                        'elapsed_s': pu_end - pu_start, 'derived': True,
                    })
                    prev_name = name
                prev_end = be['end_perf']
        else:
            optimize_total = sum(be['elapsed_s'] for be in block_events)
            pu_elapsed = stage['elapsed_s'] - optimize_total
            derived.append({
                'seq': None, 'cycle': cycle, 'agent': agent,
                'block': {'kind': agent}, 'phase': 'param_update',
                'attempt': _ATTEMPT_NA,
                'start_perf': stage['start_perf'], 'end_perf': stage['end_perf'],
                'elapsed_s': pu_elapsed, 'derived': True,
            })

    # ---- bookkeeping ----
    # D1 fix: 'clone' is DELIBERATELY excluded from this grouping -- see the
    # docstring above. block_total (Network.run_smopf / SED._optimize) never
    # encloses clone() at all (it is a sibling call in NetworkData.optimize,
    # network_data.py:61 vs :62-63), so it must never be subtracted from it.
    by_block = {}
    for r in records:
        if r['phase'] in ('block_total', 'solve_bundle', 'load_solution', 'diagnostics_parse'):
            by_block.setdefault(_block_key(r), []).append(r)
    for key, group in by_block.items():
        totals = [r for r in group if r['phase'] == 'block_total']
        if not totals:
            continue
        block_total_sum = sum(r['elapsed_s'] for r in totals)
        nested_sum = sum(r['elapsed_s'] for r in group if r['phase'] != 'block_total')
        agent, cycle = key[0], key[1]
        block = totals[0]['block']
        derived.append({
            'seq': None, 'cycle': cycle, 'agent': agent, 'block': block,
            'phase': 'bookkeeping', 'attempt': _ATTEMPT_NA,
            'start_perf': None, 'end_perf': None,
            'elapsed_s': block_total_sum - nested_sum, 'derived': True,
        })

    for record in derived:
        _raise_if_negative_duration(record)

    return derived


# Floating-point-noise allowance ONLY -- perf_counter deltas summed/subtracted
# across several independently-timed events can disagree from true zero by a
# few microseconds; anything more negative than this is a real nesting/nesting-
# assumption defect (D1's own failure mode: -8.37 s, six orders of magnitude
# past this tolerance) and must raise, never be silently reported.
_DERIVATION_NEGATIVE_TOLERANCE = 1e-6


def _raise_if_negative_duration(record):
    """Deliverable requirement (P5.15 Step 3.6 Worker task): a derived phase
    ('param_update' or 'bookkeeping') that computes to a negative duration
    must RAISE, never be silently reported as a negative `elapsed_s` -- the
    prior version of this module (v1) left such values in the output
    unclamped as a mere "correctness signal", which is how the -8.37 s D1
    defect reached a committed analysis artifact undetected."""
    elapsed = record.get('elapsed_s')
    if elapsed is not None and elapsed < -_DERIVATION_NEGATIVE_TOLERANCE:
        raise ValueError(
            f"derive_param_update_and_bookkeeping produced a negative "
            f"'{record['phase']}' duration ({elapsed!r} s) for cycle="
            f"{record.get('cycle')!r}, agent={record.get('agent')!r}, "
            f"block={record.get('block')!r} -- this indicates a nesting/"
            f"derivation defect (e.g. subtracting a phase that is not "
            f"actually enclosed by the parent span), not a value to report. "
            f"Full record: {record!r}"
        )


def lpt_partition(durations, workers):
    """Longest-processing-time-first greedy bin packing. Returns the max load
    over `workers` bins. Same heuristic as
    `WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s "Theoretical parallel speed-up"
    table (not re-derived, reused verbatim as a method)."""
    if workers <= 0:
        raise ValueError('workers must be positive')
    loads = [0.0] * workers
    for d in sorted(durations, reverse=True):
        i = loads.index(min(loads))
        loads[i] += d
    return max(loads) if loads else 0.0


_DEGRADED_VERDICT = 'INDETERMINATE (NL-write share not separated)'


def analyze_phase_timing(records, x_threshold, production_iter_wall_by_cycle=None,
                          projection_workers=8, nl_write_seconds=None,
                          solve_bundle_subtimes=None):
    """Compute the design §5 per-block-type table, overhead_local vs
    overhead_serial, the X-threshold screening verdict, and the Amdahl
    `projection_workers`-worker projected speed-up.

    P5.15 Step 3.6 Worker task (re-analysis of the already-captured
    measurement) fixed THREE defects in this function relative to the v1
    version that produced the committed `phase_timing_analysis.json`:

    D2 (window inconsistency): v1 summed EVERY record regardless of `cycle`,
    which silently included the pre-ADMM-loop INITIALIZATION solves (cycle is
    `None` for those -- `create_distribution_networks_models` /
    `create_transmission_network_model` / `create_shared_energy_storage_model`
    go through the SAME wrapped `Network.run_smopf` / `SED._optimize` /
    `OptSolver.solve` / `ModelSolutions.load_from` call sites as the ADMM
    loop, but outside any `update_..._and_solve(cycle=...)` stage wrap, so
    `recorder.current_cycle` is still `None` when they fire). This function
    now restricts every total/table to records whose `cycle` is one of the
    SAMPLED ADMM cycles (i.e. `cycle is not None`); the excluded `cycle is
    None` records are reported separately under `initialization_totals`
    (never silently dropped, never silently mixed in). Recovery/tier-2
    re-solves (`attempt != 'primary'`, still WITHIN a sampled cycle) are
    also reported separately under `recovery_totals`, in addition to being
    included in the (now cycle-scoped) main totals -- so a cycle with a
    recovery event does not silently look "the same shape" as one without.

    D3 (classification error): v1 put the FULL `solve_bundle - NL-write`
    remainder into `overhead_serial` under the key
    `solve_bundle_remainder_(ipopt_plus_sol_parse)`. That remainder is NOT
    homogeneous: the IPOPT subprocess (c) is real, parallelizable SOLVE work
    (already handled separately by the LPT 8-worker projection below) and is
    not "overhead" in the sense
    `WORKER_REPORT_S36_PARALLEL_AUDIT.md`'s own `overhead = wall - IPOPT`
    table already uses; only the `.sol`-parse remainder (d1) is per-block,
    local overhead. This function now requires `solve_bundle_subtimes` (a
    per-record `{seq: {'nl_write', 'ipopt', 'sol_parse'}}` map, built from
    Pyomo's own `report_timing=True` stdout, one triplet per solve, matched
    to `solve_bundle` records by call order / `seq` -- see
    `p515_s36_step36_timing_reanalyze.py`) to compute a DETERMINATE verdict:

        overhead_total   = wall_total - ipopt_total   (cycle-scoped)
        overhead_local   = param_update + clone (only where it fires)
                          + nl_write (b) + load_solution (d2)
                          + bookkeeping + diagnostics_parse (e)
                          + sol_parse (d1) + solve_bundle_glue
                            (solve_bundle's own measured elapsed_s minus its
                             three measured sub-times -- small Python/Pyomo
                             wrapper overhead inside solve(), still per-block
                             and local, never silently dropped)
        overhead_serial  = admm_global (f) + unattributed
        overhead_local + overhead_serial == overhead_total (by construction)
        verdict_ratio    = overhead_local / overhead_total

    When `solve_bundle_subtimes` does NOT cover every in-scope `solve_bundle`
    record (missing entirely, or partial coverage), this function falls back
    to the OLD (v1) aggregate-only classification -- `nl_write_seconds`
    (optional `{...: seconds}`, summed) as an aggregate NL-write figure, full
    `solve_bundle` counted as the NL-write UPPER BOUND when even that is
    absent -- and `verdict_pass` is a HARD NON-VERDICT (`_DEGRADED_VERDICT`,
    never `True`/`False`), because the IPOPT/`.sol`-parse split within the
    remainder is unknown and D3 showed that guessing its classification is
    exactly the defect being fixed. `nl_write_share_is_upper_bound` reports
    which mode was used.

    `x_threshold`: REQUIRED, no default (Planner decision, P5.15 Step 3.6
    follow-up: Addendum 20's X=70% is an interim screening figure -- "the
    measurement sets the real one" -- so a caller must state the threshold
    it is screening against explicitly; there is no silent fallback value).

    `records`: the recorder's raw records PLUS `derive_param_update_and_bookkeeping`'s
    derived records (caller concatenates; kept as two functions so a caller can
    inspect raw wrap events independently of the derivation).

    `production_iter_wall_by_cycle`: optional {cycle: seconds}, from parsing
    production's own `"[INFO] \\t - Iteration {iter}: {X:.2f} s"` print
    (`shared_resources_planning.py:3007`) out of the run's captured stdout.
    When given, `unattributed[cycle] = production_iter_wall_by_cycle[cycle] -
    recorder_cycle_span[cycle]`, where `recorder_cycle_span` is
    (max end_perf - min start_perf) over every record tagged with that cycle.
    When not given, `unattributed` is reported as None for every cycle (not
    fabricated as 0).

    `solve_bundle_subtimes`: optional `{seq: {'nl_write': s, 'ipopt': s,
    'sol_parse': s}}`, keyed by the RAW `solve_bundle` record's own `seq`
    (never `None` -- only live-wrapped records carry a real `seq`). Produced
    by parsing Pyomo's `report_timing=True` stdout 1:1, in call order,
    against the `solve_bundle` records sorted by `seq` (see
    `p515_s36_step36_timing_reanalyze.py::parse_report_timing_subtimes`).

    `nl_write_seconds`: optional aggregate `{...: seconds}` map (summed),
    ONLY consulted in the degraded fallback path described above (kept for
    backward compatibility with the v1 caller's call signature).

    Amdahl projection (design §5, `speedup(W) = 1 / (f_serial + (1 - f_serial)/W)`):
        f_serial = (overhead_serial_total + sum_over_agents(lpt_partition(
                       that agent's per-block MEASURED IPOPT durations, W)))
                   / wall_total
        speedup(W) = 1 / (f_serial + (1 - f_serial) / W)
      (degraded fallback: LPT runs on the solve_bundle-minus-NL-write
      remainder, exactly as v1 did, since no separate IPOPT figure exists.)

    Returns a dict with 'per_phase_by_agent' (median/mean/max/sum, PER SAMPLED
    CYCLE INDIVIDUALLY -- design §5: "report both cycles individually rather
    than a median, since 2 points do not support a robust median" -- plus an
    aggregate across all sampled cycles for convenience, clearly labeled),
    'overhead_local_total', 'overhead_serial_total', 'overhead_total',
    'x_threshold', 'verdict_pass', 'wall_total', 'ipopt_total', 'f_serial',
    f'speedup_at_{projection_workers}', 'nl_write_share_is_upper_bound',
    'per_cycle_unattributed', 'initialization_totals', 'recovery_totals'.
    """
    nl_write_seconds = nl_write_seconds or {}
    solve_bundle_subtimes = solve_bundle_subtimes or {}
    cycles = sorted({r['cycle'] for r in records if r['cycle'] is not None})

    # D2: restrict every total/table below to records tagged with a SAMPLED
    # cycle; report the excluded (cycle is None, i.e. pre-loop initialization)
    # records separately, never silently.
    in_scope = [r for r in records if r['cycle'] in cycles]
    init_records = [r for r in records if r['cycle'] is None]

    per_phase_by_agent = {}
    for agent in ('dso', 'tso', 'esso', 'admm_global'):
        per_phase_by_agent[agent] = {}
        for phase in PHASES:
            if phase == 'unattributed':
                continue
            values = [r['elapsed_s'] for r in in_scope if r['agent'] == agent and r['phase'] == phase]
            per_phase_by_agent[agent][phase] = _stats(values)
            per_phase_by_agent[agent][phase]['per_cycle'] = {
                cycle: _stats([r['elapsed_s'] for r in in_scope
                               if r['agent'] == agent and r['phase'] == phase and r['cycle'] == cycle])
                for cycle in cycles
            }

    # ---- initialization_totals (cycle is None -- pre-ADMM-loop solves) ----
    initialization_totals = {}
    for agent in ('dso', 'tso', 'esso'):
        initialization_totals[agent] = {
            phase: _stats([r['elapsed_s'] for r in init_records
                           if r['agent'] == agent and r['phase'] == phase])
            for phase in PHASES if phase != 'unattributed'
        }

    # ---- recovery_totals (attempt != primary/n/a, WITHIN a sampled cycle) ----
    recovery_totals = {}
    for agent in ('dso', 'tso', 'esso'):
        recovery_totals[agent] = {
            phase: _stats([r['elapsed_s'] for r in in_scope
                           if r['agent'] == agent and r['phase'] == phase
                           and r['attempt'] in (_ATTEMPT_TIER1, _ATTEMPT_TIER2)])
            for phase in ('solve_bundle', 'load_solution')
        }

    # ---- recorder-reconstructed per-cycle span, and unattributed ----
    per_cycle_span = {}
    for cycle in cycles:
        starts = [r['start_perf'] for r in in_scope if r['cycle'] == cycle and r['start_perf'] is not None]
        ends = [r['end_perf'] for r in in_scope if r['cycle'] == cycle and r['end_perf'] is not None]
        per_cycle_span[cycle] = (max(ends) - min(starts)) if starts and ends else None

    per_cycle_unattributed = {}
    for cycle in cycles:
        prod_wall = (production_iter_wall_by_cycle or {}).get(cycle)
        span = per_cycle_span.get(cycle)
        if prod_wall is not None and span is not None:
            per_cycle_unattributed[cycle] = prod_wall - span
        else:
            per_cycle_unattributed[cycle] = None

    # ---- wall_total (cycle-scoped, computed before overhead so D3's
    #      overhead_total = wall_total - ipopt_total can use it) ----
    wall_total = None
    if production_iter_wall_by_cycle:
        wall_total = sum(production_iter_wall_by_cycle.get(c, 0.0) for c in cycles)
    elif all(v is not None for v in per_cycle_span.values()) and per_cycle_span:
        wall_total = sum(per_cycle_span.values())

    # ---- D3: solve_bundle b/c/d1 split ----
    sb_in_scope = [r for r in in_scope if r['phase'] == 'solve_bundle']
    sb_seqs = {r['seq'] for r in sb_in_scope}
    fully_covered = bool(sb_seqs) and sb_seqs.issubset(solve_bundle_subtimes.keys())

    param_update_total = sum(r['elapsed_s'] for r in in_scope if r['phase'] == 'param_update')
    clone_total = sum(r['elapsed_s'] for r in in_scope if r['phase'] == 'clone')
    load_solution_total = sum(r['elapsed_s'] for r in in_scope if r['phase'] == 'load_solution')
    bookkeeping_total = sum(r['elapsed_s'] for r in in_scope if r['phase'] == 'bookkeeping')
    diagnostics_parse_total = sum(r['elapsed_s'] for r in in_scope if r['phase'] == 'diagnostics_parse')
    admm_global_total = sum(r['elapsed_s'] for r in in_scope if r['phase'] == 'admm_global')
    unattributed_total = sum(v for v in per_cycle_unattributed.values() if v is not None)
    solve_bundle_total = sum(r['elapsed_s'] for r in sb_in_scope)

    if fully_covered:
        nl_write_is_upper_bound = False
        nl_write_total = sum(solve_bundle_subtimes[r['seq']]['nl_write'] for r in sb_in_scope)
        ipopt_total = sum(solve_bundle_subtimes[r['seq']]['ipopt'] for r in sb_in_scope)
        sol_parse_total = sum(solve_bundle_subtimes[r['seq']]['sol_parse'] for r in sb_in_scope)
        glue_total = sum(
            max(r['elapsed_s'] - sum(solve_bundle_subtimes[r['seq']].values()), 0.0)
            for r in sb_in_scope)
        overhead_local_total = (param_update_total + clone_total + nl_write_total
                                 + load_solution_total + bookkeeping_total
                                 + diagnostics_parse_total + sol_parse_total + glue_total)
        overhead_serial_total = admm_global_total + unattributed_total
        overhead_total = overhead_local_total + overhead_serial_total
        overhead_total_cross_check = (wall_total - ipopt_total) if wall_total is not None else None
        solve_bundle_remainder_total = None  # superseded by the sol_parse/ipopt split below
    else:
        # Degraded fallback -- v1's aggregate-only classification, kept for
        # when a per-record report_timing cross-check is unavailable.
        nl_write_total = sum(nl_write_seconds.values()) if nl_write_seconds else solve_bundle_total
        nl_write_is_upper_bound = not bool(nl_write_seconds)
        ipopt_total = None
        sol_parse_total = None
        glue_total = None
        solve_bundle_remainder_total = max(solve_bundle_total - nl_write_total, 0.0)
        overhead_local_total = (param_update_total + clone_total + nl_write_total
                                 + load_solution_total + bookkeeping_total + diagnostics_parse_total)
        overhead_serial_total = admm_global_total + unattributed_total + solve_bundle_remainder_total
        overhead_total = overhead_local_total + overhead_serial_total
        overhead_total_cross_check = None

    denom = overhead_total
    verdict_ratio = (overhead_local_total / denom) if denom else None
    if nl_write_is_upper_bound or not fully_covered:
        # Hard non-verdict (Addendum 20 item 2, extended by D3's own finding:
        # a verdict is only decisive once solve_bundle's b/c/d1 split -- not
        # only its NL-write share -- is actually measured, not guessed).
        verdict = _DEGRADED_VERDICT
    else:
        verdict = (verdict_ratio is not None) and (verdict_ratio >= x_threshold)

    # ---- Amdahl projection ----
    lpt_residual_total = 0.0
    for agent in ('dso', 'tso', 'esso'):
        agent_sb = [r for r in sb_in_scope if r['agent'] == agent]
        if fully_covered:
            durations = [solve_bundle_subtimes[r['seq']]['ipopt'] for r in agent_sb]
        else:
            durations = [r['elapsed_s'] for r in agent_sb]
            if nl_write_seconds:
                durations = [max(d - (sum(nl_write_seconds.values()) / max(len(durations), 1)), 0.0)
                             for d in durations]
        if durations:
            lpt_residual_total += lpt_partition(durations, projection_workers)

    f_serial = None
    speedup = None
    if wall_total:
        f_serial = (overhead_serial_total + lpt_residual_total) / wall_total
        f_serial = min(max(f_serial, 0.0), 1.0)
        speedup = 1.0 / (f_serial + (1.0 - f_serial) / projection_workers)

    return {
        'per_phase_by_agent': per_phase_by_agent,
        'initialization_totals': initialization_totals,
        'recovery_totals': recovery_totals,
        'per_cycle_unattributed': per_cycle_unattributed,
        'per_cycle_span': per_cycle_span,
        'overhead_local_total': overhead_local_total,
        'overhead_serial_total': overhead_serial_total,
        'overhead_total': overhead_total,
        'overhead_total_cross_check_(wall_minus_ipopt)': overhead_total_cross_check,
        'overhead_local_components': {
            'param_update': param_update_total, 'clone': clone_total,
            'nl_write_share_of_solve_bundle': nl_write_total,
            'load_solution': load_solution_total, 'bookkeeping': bookkeeping_total,
            'diagnostics_parse': diagnostics_parse_total,
            'sol_parse_share_of_solve_bundle': sol_parse_total,
            'solve_bundle_glue': glue_total,
        },
        'overhead_serial_components': {
            'admm_global': admm_global_total, 'unattributed': unattributed_total,
            'solve_bundle_remainder_(ipopt_plus_sol_parse)_degraded_only': solve_bundle_remainder_total,
        },
        'ipopt_total': ipopt_total,
        'solve_bundle_total': solve_bundle_total,
        'nl_write_share_is_upper_bound': nl_write_is_upper_bound,
        'solve_bundle_subtimes_fully_covered': fully_covered,
        'x_threshold': x_threshold,
        'verdict_ratio': verdict_ratio,
        'verdict_pass': verdict,
        'wall_total': wall_total,
        'lpt_residual_total_at_projection_workers': lpt_residual_total,
        'projection_workers': projection_workers,
        'f_serial': f_serial,
        f'speedup_at_{projection_workers}_workers': speedup,
    }
