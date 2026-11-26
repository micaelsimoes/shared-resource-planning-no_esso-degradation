"""
P5.15 Addendum 27 item 5(a) -- zero-solve checks for the SWITCHABLE pristine snapshot
clones (`admm_parameters.tso_snapshot_capture_mode` / `dso_snapshot_capture_mode` gain a
third legal value, 'off'; the default stays 'lightweight').

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 item 5(a) ("pristine snapshot clones
switchable, off for campaign and verification runs"); paper-scale bounded task item (a).
Motivation (measured, not assumed): the paper-scale build crossed the 24 GiB watchdog
INSIDE the pristine-clone stage -- `data/SRP1/Results/P515S44/scale_measurement/
paper_build/watchdog_abort_build.json` records `stage_at_abort = "pristine clones kept by
the ADMM loop (tso_pristine_base, dso_pristine_base)"`, the last pre-stage footprint sample
19,645,758,080 B and the abort at 25,824,742,624 B, i.e. >= 5.75 GiB added by a stage that
never even finished.

WHAT IS CHECKED (all with production functions; stubs never reach a solver)
--------------------------------------------------------------------------
On ONE freshly built, never-solved, production-constructed ADMM-ready SRP1 state
(`p515_s32_zero_solve_checks._build_admm_ready_state`, reused verbatim), with
`Network.run_smopf` monkeypatched to a canned result that reports failure for exactly one
TSO block and one node-7 DSO block (the same technique as
`p515_s36_clone_capture_checks.py`), and with real `BlockData.clone` and production's own
`shared_resources_planning.capture_block_mutable_state` wrapped in pass-through counters:

 (i)   mode 'off': `_build_pristine_snapshot_bases` returns (None, None); the TSO dispatch
       and the node-7 DSO dispatch reach `BlockData.clone` ZERO times and
       `capture_block_mutable_state` ZERO times; the warning naming the mode and the block
       is printed; a JSON marker is written in the FrozenSMOPF directory; the cycle-7
       comparator prints its "SKIPPED" line; and
       `p515_g_g1_g4_admm_gates._scan_frozen_snapshots` (which selects `.pkl` only) counts
       ZERO new snapshots for that arm.
 (ii)  default 'lightweight': both pristine bases ARE built, per-cycle captures happen
       (one per block, count > 0) and exactly the on-demand rebuild clones are taken
       (one per written snapshot) -- today's behaviour, unchanged.
 (ii-b) REGRESSION (not required by the task, added because the new branch precedes the two
       existing ones): 'legacy_clone' still builds no pristine base, takes no capture, and
       clones once per block per cycle inside `NetworkData.optimize` for the TSO and for
       node 7 only -- the pre-Step-3.6 behaviour, unchanged.
 (iii) `_validate_snapshot_capture_modes` raises ValueError on an illegal mode, and on
       'off' with `persistent_workers['enabled']` True (either agent); 'legacy_clone' with
       persistent workers still passes.
 (iv)  with mode 'off' nothing else differs from mode 'lightweight' before the solve call:
       the full pre-solve block state captured INSIDE the stubbed `run_smopf`
       (`network.capture_block_mutable_state`, i.e. every mutable Param, every Var
       value/bounds/fixed flag, every constraint/objective active flag, the multiplier
       suffixes) digests identically block by block, as do the network parameters seen by
       `run_smopf` and the ADMM parameter object itself (mode keys excluded).
 (v)   `p515_s44_scale_measurement`'s `--snapshots off` path sets BOTH modes: its
       `apply_snapshot_setting` and its `snapshot_hook_wrapper` (the pre_solve_hook wrap)
       are exercised DIRECTLY on a stub planning object -- no build, no campaign --
       including the failure path (a params object that refuses the assignment must raise,
       so the setting is verified and not asserted).

`SolveProfileGuard(permitted=())` is installed BEFORE any production import and verified at
exactly 0 solves / 0 process launches / 0 blocked calls at the end.

    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
        p515_s45_snapshot_switch_checks.py [run_suffix] \
        > data/SRP1/Results/P515S45/snapshot_switch_checks_launch.log 2>&1

An optional positional run suffix redirects the output directory (same convention as
`p515_s36_clone_capture_checks.py`), so a re-run never overwrites an earlier run's evidence.

Writes data/SRP1/Results/P515S45/snapshot_switch_checks[_suffix]/{results.json,
arm_stdout_*.log, manifest_sha256.json} (all NEW; refuses to overwrite) plus the
FrozenSMOPF artifacts the arms themselves produce, under
.../snapshot_switch_checks/planning_results/FrozenSMOPF/ (redirected there so no arm can
write into the canonical data/SRP1/Results/FrozenSMOPF).
"""

import hashlib
import io
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

# The guard is installed before ANY production import (CLAUDE.md evidence rule 6).
from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 S45 snapshot switch checks (zero solves)').install()

import pyomo.environ as pe  # noqa: E402
import pyomo.opt as po  # noqa: E402
from pyomo.core.base.block import BlockData  # noqa: E402

import network as NET  # noqa: E402
from network_data import NetworkData  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402
import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s44_scale_measurement as S44  # noqa: E402
from p515_s32_zero_solve_checks import _build_admm_ready_state  # noqa: E402

STAGE = 'P5.15 Addendum 27 item 5(a) -- switchable pristine snapshot clones (zero-solve checks)'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 item 5(a)',
    'paper-scale bounded task item (a)',
    'data/SRP1/Results/P515S44/scale_measurement/paper_build/watchdog_abort_build.json (motivation)',
]
SCHEMA = 'p515_s45_snapshot_switch_checks_v1'

# Optional positional run suffix (argv[1]): redirects OUT_DIR so a re-run never overwrites
# an earlier run's evidence (CLAUDE.md: never re-run a harness onto a cited artifact).
_RUN_SUFFIX = ''
for _arg in sys.argv[1:]:
    if not _arg.startswith('--'):
        _RUN_SUFFIX = '_' + _arg.strip('_')
        break

OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S45',
                       f'snapshot_switch_checks{_RUN_SUFFIX}')
RESULTS_PATH = os.path.join(OUT_DIR, 'results.json')
MANIFEST_PATH = os.path.join(OUT_DIR, 'manifest_sha256.json')
PLANNING_RESULTS_DIR = os.path.join(OUT_DIR, 'planning_results')

EVAL_ID = 's45_snapshot_switch_' + datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')

TSO_COMPARATOR = ('2025', 'Summer')   # update_transmission_coordination_model_and_solve
DSO_COMPARATOR = ('2025', 'Autumn')   # update_distribution_coordination_models_and_solve_sequential
COMPARATOR_CYCLE = 7


def _utc():
    return datetime.now(tz=timezone.utc).isoformat()


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _git(args):
    try:
        return subprocess.run(['git'] + args, capture_output=True, text=True, check=True,
                              cwd=REPO).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'unavailable: {error!r}'


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(obj):
    """Order-independent, type-tolerant canonical form for digesting a captured state
    (keys may be ints, strings or tuples; values floats, bools or None)."""
    if isinstance(obj, dict):
        return '{' + ','.join(f'{repr(k)}:{_canonical(v)}'
                              for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_canonical(v) for v in obj) + ']'
    return repr(obj)


def _digest(obj):
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()


def _params_digest(params, exclude=()):
    items = {k: repr(v) for k, v in vars(params).items() if k not in exclude}
    return _digest(items)


def _canned_result(succeeded):
    result = po.SolverResults()
    result.solver.status = po.SolverStatus.ok if succeeded else po.SolverStatus.warning
    result.solver.termination_condition = (
        po.TerminationCondition.optimal if succeeded else po.TerminationCondition.maxIterations)
    return result


# ---------------------------------------------------------------------------
# Counters: real `BlockData.clone` and production's own
# `shared_resources_planning.capture_block_mutable_state` (the module attribute the
# dispatch functions actually resolve), both PASS-THROUGH -- never stubs.
# ---------------------------------------------------------------------------
COUNTS = {'clone': 0, 'capture': 0}
_ORIG_CLONE = BlockData.clone
_ORIG_CAPTURE = srp.capture_block_mutable_state


def _counting_clone(self, *args, **kwargs):
    COUNTS['clone'] += 1
    return _ORIG_CLONE(self, *args, **kwargs)


def _counting_capture(model):
    COUNTS['capture'] += 1
    return _ORIG_CAPTURE(model)


def _install_counters():
    BlockData.clone = _counting_clone
    srp.capture_block_mutable_state = _counting_capture


def _remove_counters():
    BlockData.clone = _ORIG_CLONE
    srp.capture_block_mutable_state = _ORIG_CAPTURE


def _reset_counters():
    COUNTS['clone'] = 0
    COUNTS['capture'] = 0


# ---------------------------------------------------------------------------
# Stubbed `Network.run_smopf`: canned result (never a solver) + the pre-solve state
# probe used by check (iv). Uses `NET.capture_block_mutable_state` directly, so the
# probe is never counted as one of production's per-cycle captures.
# ---------------------------------------------------------------------------
PROBE = {'blocks': {}, 'network_params': {}}
_ORIG_RUN_SMOPF = NET.Network.run_smopf


def _make_run_smopf(fail_keys, probe_enabled):
    def canned_run_smopf(self, model, params, from_warm_start=False, print_header=True):
        key = f'{self.name}|{self.year}|{self.day}'
        if probe_enabled:
            PROBE['blocks'][key] = _digest(NET.capture_block_mutable_state(model))
            PROBE['network_params'][key] = _params_digest(params)
        return _canned_result(succeeded=(key not in fail_keys))
    return canned_run_smopf


def _new_pkl_files(frozen_dir, since):
    if not os.path.isdir(frozen_dir):
        return []
    out = []
    for name in sorted(os.listdir(frozen_dir)):
        path = os.path.join(frozen_dir, name)
        if os.path.getmtime(path) >= since:
            out.append(name)
    return out


# ===========================================================================
#  Check (iii): mode validation (no state needed)
# ===========================================================================
def check_validation():
    record = {'legal_modes': list(srp.SNAPSHOT_CAPTURE_MODES),
              'off_is_legal': 'off' in srp.SNAPSHOT_CAPTURE_MODES,
              'cases': []}

    def case(name, configure, expect_raise):
        params = ADMMParameters()
        configure(params)
        entry = {'case': name, 'expect_raise': expect_raise,
                 'tso_mode': params.tso_snapshot_capture_mode,
                 'dso_mode': params.dso_snapshot_capture_mode,
                 'persistent_workers': dict(params.persistent_workers)}
        try:
            entry['resolved_modes'] = list(srp._validate_snapshot_capture_modes(params))
            entry['raised'] = False
            entry['error'] = None
        except ValueError as error:
            entry['raised'] = True
            entry['error'] = f'{type(error).__name__}: {error}'
        except Exception as error:  # noqa: BLE001 -- any other exception is a failure
            entry['raised'] = True
            entry['error'] = f'UNEXPECTED {type(error).__name__}: {error}'
        entry['as_expected'] = (entry['raised'] == expect_raise) and (
            not entry['raised'] or not entry['error'].startswith('UNEXPECTED'))
        record['cases'].append(entry)
        return entry

    fresh = ADMMParameters()
    record['default_tso_mode'] = fresh.tso_snapshot_capture_mode
    record['default_dso_mode'] = fresh.dso_snapshot_capture_mode
    record['defaults_are_lightweight'] = (fresh.tso_snapshot_capture_mode == 'lightweight'
                                          and fresh.dso_snapshot_capture_mode == 'lightweight')

    case('defaults (lightweight/lightweight)', lambda p: None, expect_raise=False)
    case('legacy_clone/legacy_clone',
         lambda p: (setattr(p, 'tso_snapshot_capture_mode', 'legacy_clone'),
                    setattr(p, 'dso_snapshot_capture_mode', 'legacy_clone')), expect_raise=False)
    case('off/off', lambda p: (setattr(p, 'tso_snapshot_capture_mode', 'off'),
                               setattr(p, 'dso_snapshot_capture_mode', 'off')), expect_raise=False)
    case('invalid TSO mode', lambda p: setattr(p, 'tso_snapshot_capture_mode', 'nonsense'),
         expect_raise=True)
    case('invalid DSO mode', lambda p: setattr(p, 'dso_snapshot_capture_mode', ''),
         expect_raise=True)
    case('TSO off + persistent workers enabled',
         lambda p: (setattr(p, 'tso_snapshot_capture_mode', 'off'),
                    p.persistent_workers.update({'enabled': True})), expect_raise=True)
    case('DSO off + persistent workers enabled',
         lambda p: (setattr(p, 'dso_snapshot_capture_mode', 'off'),
                    p.persistent_workers.update({'enabled': True})), expect_raise=True)
    case('legacy_clone + persistent workers enabled (still legal)',
         lambda p: (setattr(p, 'tso_snapshot_capture_mode', 'legacy_clone'),
                    p.persistent_workers.update({'enabled': True})), expect_raise=False)

    # `_build_pristine_snapshot_bases` validates BEFORE it touches any model: an illegal
    # mode raises even with empty model dicts, and 'off' returns (None, None).
    params_off = ADMMParameters()
    params_off.tso_snapshot_capture_mode = 'off'
    params_off.dso_snapshot_capture_mode = 'off'
    bases = srp._build_pristine_snapshot_bases(params_off, None, {}, {}, {})
    record['build_bases_off_returns_none_none'] = bases == (None, None)

    params_bad = ADMMParameters()
    params_bad.dso_snapshot_capture_mode = 'nope'
    try:
        srp._build_pristine_snapshot_bases(params_bad, None, {}, {}, {})
        record['build_bases_invalid_raises'] = False
    except ValueError as error:
        record['build_bases_invalid_raises'] = True
        record['build_bases_invalid_error'] = f'{type(error).__name__}: {error}'

    record['pass'] = (record['off_is_legal'] and record['defaults_are_lightweight']
                      and all(c['as_expected'] for c in record['cases'])
                      and record['build_bases_off_returns_none_none']
                      and record['build_bases_invalid_raises'])
    return record


# ===========================================================================
#  Check (v): the scale script's --snapshots off path (no build, no campaign)
# ===========================================================================
class _StubPlanning:
    def __init__(self, params):
        self.params = type('P', (), {})()
        self.params.admm = params


class _RefusingParams(ADMMParameters):
    """A params object that silently REFUSES the assignment -- used to show that
    `apply_snapshot_setting` VERIFIES the setting instead of asserting it."""

    def __setattr__(self, name, value):
        if name in ('tso_snapshot_capture_mode', 'dso_snapshot_capture_mode') and value == 'off':
            return
        object.__setattr__(self, name, value)


def check_scale_script_switch():
    record = {}

    rec_off = {}
    applied_off = S44.apply_snapshot_setting(_StubPlanning(ADMMParameters()), 'off', rec_off)
    record['apply_off'] = applied_off
    record['apply_off_sets_both'] = (applied_off['tso_mode_after'] == 'off'
                                     and applied_off['dso_mode_after'] == 'off'
                                     and applied_off['took_effect'])
    record['apply_off_recorded'] = rec_off.get('snapshot_setting') == applied_off

    rec_on = {}
    applied_on = S44.apply_snapshot_setting(_StubPlanning(ADMMParameters()), 'on', rec_on)
    record['apply_on'] = applied_on
    record['apply_on_changes_nothing'] = (applied_on['tso_mode_after'] == 'lightweight'
                                          and applied_on['dso_mode_after'] == 'lightweight'
                                          and applied_on['took_effect'])

    # The pre_solve_hook wrapper: inner hook runs first, then the modes are set on the
    # planning object the arm will run on, and the applied record reaches the report.
    inner_calls = []

    def inner_hook(planning, sed, candidate, report):
        inner_calls.append({'tso_mode_seen_by_inner': planning.params.admm.tso_snapshot_capture_mode})
        report['inner_hook_ran'] = True

    planning = _StubPlanning(ADMMParameters())
    report = {}
    rec_hook = {}
    S44.snapshot_hook_wrapper(inner_hook, 'off', rec_hook)(
        planning=planning, sed=None, candidate=None, report=report)
    record['hook_inner_calls'] = inner_calls
    record['hook_inner_ran_once'] = len(inner_calls) == 1 and report.get('inner_hook_ran') is True
    record['hook_sets_both_modes'] = (planning.params.admm.tso_snapshot_capture_mode == 'off'
                                      and planning.params.admm.dso_snapshot_capture_mode == 'off')
    record['hook_report_entry'] = report.get('rule_eleven_checklist', {}).get('s44_snapshot_setting')
    record['hook_records_into_report'] = bool(
        (record['hook_report_entry'] or {}).get('took_effect'))
    record['hook_records_into_child_record'] = rec_hook.get('snapshot_setting') is not None

    # The verification is ARMED: a params object that refuses the assignment raises.
    try:
        S44.apply_snapshot_setting(_StubPlanning(_RefusingParams()), 'off', {})
        record['refusing_params_raises'] = False
    except RuntimeError as error:
        record['refusing_params_raises'] = True
        record['refusing_params_error'] = f'{type(error).__name__}: {error}'

    # Source-level evidence for the CLI wiring (never launched as a subprocess: the guard
    # blocks process launches, and this check must stay in-process).
    import inspect
    main_src = inspect.getsource(S44.main)
    build_src = inspect.getsource(S44.child_build)
    cycle_src = inspect.getsource(S44.child_cycle)
    record['cli'] = {
        'flag_declared': "'--snapshots'" in main_src or '"--snapshots"' in main_src,
        'choices_on_off': "choices=('on', 'off')" in main_src,
        'default_on': "default='on'" in main_src.split('--snapshots')[1].split('parser.add_argument')[0],
        'build_child_reads_launch': "launch.get('snapshots', 'on')" in build_src,
        'cycle_child_reads_launch': "launch.get('snapshots', 'on')" in cycle_src,
        'build_child_calls_production_helper': '_build_pristine_snapshot_bases' in build_src,
    }
    record['pass'] = (record['apply_off_sets_both'] and record['apply_off_recorded']
                      and record['apply_on_changes_nothing'] and record['hook_inner_ran_once']
                      and record['hook_sets_both_modes'] and record['hook_records_into_report']
                      and record['hook_records_into_child_record']
                      and record['refusing_params_raises']
                      and all(record['cli'].values()))
    return record


# ===========================================================================
#  Checks (i), (ii), (iv): the real dispatch paths on a real ADMM-ready state
# ===========================================================================
def run_dispatch_arms(record):
    state_t0 = time.time()
    planning, tso_model, dso_models, esso_model, consensus_vars, dual_vars = \
        _build_admm_ready_state(EVAL_ID)
    record['state_build_wall_s'] = round(time.time() - state_t0, 2)

    transmission_network = planning.transmission_network
    distribution_networks = planning.distribution_networks

    # Every FrozenSMOPF artifact of this check must land inside OUT_DIR -- never in the
    # canonical data/SRP1/Results/FrozenSMOPF, which holds preserved fixtures.
    os.makedirs(PLANNING_RESULTS_DIR, exist_ok=True)
    record['results_dir_before_redirect'] = {
        'planning': planning.results_dir,
        'tso': transmission_network.results_dir,
        'dso': {str(n): dn.results_dir for n, dn in distribution_networks.items()},
    }
    planning.results_dir = PLANNING_RESULTS_DIR
    transmission_network.results_dir = PLANNING_RESULTS_DIR
    for dn in distribution_networks.values():
        dn.results_dir = PLANNING_RESULTS_DIR
    if not os.path.abspath(transmission_network.results_dir).startswith(os.path.abspath(OUT_DIR)):
        raise RuntimeError('results_dir redirect failed; refusing to run (would write to data/SRP1/Results)')
    frozen_dir = os.path.join(PLANNING_RESULTS_DIR, 'FrozenSMOPF')
    record['frozen_dir'] = os.path.relpath(frozen_dir, REPO)

    # `_build_admm_ready_state` leaves a non-solving `.optimize` stub on every network; the
    # dispatch paths under test need the REAL `NetworkData.optimize` (it is what wires --
    # or does not wire -- the snapshot callbacks), with only `Network.run_smopf` canned.
    transmission_network.optimize = NetworkData.optimize.__get__(
        transmission_network, type(transmission_network))
    for dn in distribution_networks.values():
        dn.optimize = NetworkData.optimize.__get__(dn, type(dn))

    admm_parameters = planning.params.admm
    try:
        capacities = planning.shared_ess_data.get_updated_capacities(esso_model)
        record['sess_capacities_source'] = 'shared_ess_data.get_updated_capacities (production)'
    except Exception as error:  # noqa: BLE001
        capacities = {node_id: {year: {'s_available': 1.0, 'e_available': 1.0}
                                for year in transmission_network.years}
                      for node_id in transmission_network.active_distribution_network_nodes}
        record['sess_capacities_source'] = (f'synthetic 1.0/1.0 (get_updated_capacities raised: '
                                            f'{type(error).__name__}: {error})')

    tso_blocks = [(year, day) for year in transmission_network.years
                  for day in transmission_network.days]
    dso7 = distribution_networks[7]
    dso7_blocks = [(year, day) for year in dso7.years for day in dso7.days]
    record['block_counts'] = {'tso': len(tso_blocks), 'dso_node7': len(dso7_blocks),
                              'dso_nodes': sorted(distribution_networks)}

    tso_fail = next((y, d) for (y, d) in tso_blocks if (str(y), str(d)) != TSO_COMPARATOR)
    dso_fail = next((y, d) for (y, d) in dso7_blocks if (str(y), str(d)) != DSO_COMPARATOR)
    tso_fail_key = f'{transmission_network.network[tso_fail[0]][tso_fail[1]].name}|{tso_fail[0]}|{tso_fail[1]}'
    dso_fail_key = f'{dso7.network[dso_fail[0]][dso_fail[1]].name}|{dso_fail[0]}|{dso_fail[1]}'
    record['simulated_failures'] = {'tso_block': list(map(str, tso_fail)), 'tso_key': tso_fail_key,
                                    'dso_node7_block': list(map(str, dso_fail)), 'dso_key': dso_fail_key}
    record['comparators'] = {'tso': list(TSO_COMPARATOR), 'dso_node7': list(DSO_COMPARATOR),
                             'cycle': COMPARATOR_CYCLE}
    record['tso_comparator_present'] = any((str(y), str(d)) == TSO_COMPARATOR for y, d in tso_blocks)
    record['dso_comparator_present'] = any((str(y), str(d)) == DSO_COMPARATOR for y, d in dso7_blocks)

    fail_keys = {tso_fail_key, dso_fail_key}

    def run_arm(arm, mode):
        """One full TSO dispatch + one full DSO dispatch at the comparator cycle, with the
        capture mode set to `mode` on the SAME models, so the two arms differ in nothing
        else. Returns the arm record."""
        admm_parameters.tso_snapshot_capture_mode = mode
        admm_parameters.dso_snapshot_capture_mode = mode
        arm_rec = {'arm': arm, 'mode': mode, 'cycle': COMPARATOR_CYCLE}
        arm_rec['admm_params_digest_excluding_modes'] = _params_digest(
            admm_parameters, exclude=('tso_snapshot_capture_mode', 'dso_snapshot_capture_mode'))

        _reset_counters()
        base_t0 = time.time()
        tso_base, dso_base = srp._build_pristine_snapshot_bases(
            admm_parameters, transmission_network, tso_model, distribution_networks, dso_models)
        arm_rec['pristine_bases'] = {
            'tso_is_none': tso_base is None, 'dso_is_none': dso_base is None,
            'tso_blocks': (None if tso_base is None else sum(len(v) for v in tso_base.values())),
            'dso_blocks': (None if dso_base is None else sum(len(v) for v in dso_base.values())),
            'clone_calls': COUNTS['clone'], 'wall_s': round(time.time() - base_t0, 2)}

        PROBE['blocks'] = {}
        PROBE['network_params'] = {}
        since = time.time()
        time.sleep(0.01)

        stdout_buffer = io.StringIO()
        NET.Network.run_smopf = _make_run_smopf(fail_keys, probe_enabled=True)
        try:
            _reset_counters()
            with redirect_stdout(stdout_buffer):
                srp.update_transmission_coordination_model_and_solve(
                    transmission_network, tso_model,
                    consensus_vars['vmag'], dual_vars['vmag']['tso'],
                    consensus_vars['pf'], dual_vars['pf']['tso'],
                    consensus_vars['ess'], dual_vars['ess']['tso'],
                    admm_parameters, capacities, from_warm_start=True,
                    cycle=COMPARATOR_CYCLE, tso_pristine_base=tso_base)
            arm_rec['tso'] = {'clone_calls': COUNTS['clone'], 'capture_calls': COUNTS['capture']}

            _reset_counters()
            with redirect_stdout(stdout_buffer):
                srp.update_distribution_coordination_models_and_solve(
                    distribution_networks, dso_models,
                    consensus_vars['vmag'], dual_vars['vmag']['dso'],
                    consensus_vars['pf'], dual_vars['pf']['dso'],
                    consensus_vars['ess'], dual_vars['ess']['dso'],
                    admm_parameters, capacities, from_warm_start=True,
                    parallel_execution=False, cycle=COMPARATOR_CYCLE,
                    dso_pristine_base=dso_base)
            arm_rec['dso'] = {'clone_calls': COUNTS['clone'], 'capture_calls': COUNTS['capture']}
        finally:
            NET.Network.run_smopf = _ORIG_RUN_SMOPF

        text = stdout_buffer.getvalue()
        log_path = os.path.join(OUT_DIR, f'arm_stdout_{arm}.log')
        _refuse_overwrite(log_path)
        with open(log_path, 'w') as handle:
            handle.write(text)
        print(text, end='', flush=True)

        arm_rec['stdout_log'] = os.path.relpath(log_path, REPO)
        arm_rec['stdout_markers'] = {
            'tso_off_warning': '[WARNING][FROZEN SMOPF] snapshot capture is OFF '
                               '(tso_snapshot_capture_mode=off)' in text,
            'dso_off_warning': '[WARNING][FROZEN SMOPF] snapshot capture is OFF '
                               '(dso_snapshot_capture_mode=off)' in text,
            'tso_comparator_skipped': 'cycle-7 TSO comparator snapshot SKIPPED' in text,
            'dso_comparator_skipped': 'comparator snapshot SKIPPED (dso_snapshot_capture_mode=off)' in text,
            'saved_presolve_block_lines': text.count('Saved '),
            'marker_lines': text.count('Wrote snapshot-skipped marker'),
            'tso_fail_block_named': (f'year={tso_fail[0]} | day={tso_fail[1]}' in text),
            'dso_fail_block_named': (f'year={dso_fail[0]} | day={dso_fail[1]}' in text),
        }
        new_files = _new_pkl_files(frozen_dir, since)
        arm_rec['frozen_dir_new_files'] = new_files
        arm_rec['new_pkl'] = [f for f in new_files if f.endswith('.pkl')]
        arm_rec['new_json_markers'] = [f for f in new_files if f.endswith('.json')]
        scanned = G._scan_frozen_snapshots(PLANNING_RESULTS_DIR, since)
        arm_rec['scan_frozen_snapshots_count'] = len(scanned)
        arm_rec['scan_frozen_snapshots_files'] = [e['filename'] for e in scanned]
        arm_rec['probe'] = {'blocks_seen': len(PROBE['blocks'])}
        return arm_rec, dict(PROBE['blocks']), dict(PROBE['network_params'])

    _install_counters()
    try:
        arm_light, probe_light, netparams_light = run_arm('lightweight', 'lightweight')
        arm_off, probe_off, netparams_off = run_arm('off', 'off')
        # Regression arm (ii-b): the pre-Step-3.6 path, which the new 'off' branch now
        # precedes in both dispatch functions. It rewrites the same four snapshot
        # filenames the lightweight arm wrote (same blocks, same labels).
        arm_legacy, probe_legacy, _netparams_legacy = run_arm('legacy_clone', 'legacy_clone')
    finally:
        _remove_counters()

    n_tso, n_dso7 = len(tso_blocks), len(dso7_blocks)
    expected = {
        'lightweight': {
            'pristine_clone_calls': n_tso + n_dso7,
            'tso_capture_calls': n_tso, 'dso_capture_calls': n_dso7,
            'tso_clone_calls': 2, 'dso_clone_calls': 2,
            'note': ('one capture per block per cycle; two on-demand rebuild clones per agent '
                     '= one for the simulated failure + one for the cycle-7 comparator'),
        },
        'off': {'pristine_clone_calls': 0, 'tso_capture_calls': 0, 'dso_capture_calls': 0,
                'tso_clone_calls': 0, 'dso_clone_calls': 0,
                'note': 'no pristine base, no per-cycle capture, no clone anywhere'},
        'legacy_clone': {
            'pristine_clone_calls': 0, 'tso_capture_calls': 0, 'dso_capture_calls': 0,
            'tso_clone_calls': n_tso, 'dso_clone_calls': n_dso7,
            'note': ('pre-Step-3.6: NetworkData.optimize clones every TSO block and every '
                     'node-7 DSO block on every cycle (both callbacks wired); nodes 5 and 9 '
                     'pass None/None and never clone'),
        },
    }
    record['expected'] = expected
    record['arm_lightweight'] = arm_light
    record['arm_off'] = arm_off
    record['arm_legacy_clone'] = arm_legacy

    light_ok = (
        arm_light['pristine_bases']['tso_is_none'] is False
        and arm_light['pristine_bases']['dso_is_none'] is False
        and arm_light['pristine_bases']['clone_calls'] == expected['lightweight']['pristine_clone_calls']
        and arm_light['tso']['capture_calls'] == expected['lightweight']['tso_capture_calls']
        and arm_light['dso']['capture_calls'] == expected['lightweight']['dso_capture_calls']
        and arm_light['tso']['clone_calls'] == expected['lightweight']['tso_clone_calls']
        and arm_light['dso']['clone_calls'] == expected['lightweight']['dso_clone_calls']
        and len(arm_light['new_pkl']) == 4
        and arm_light['scan_frozen_snapshots_count'] == 4
        and arm_light['new_json_markers'] == []
    )
    off_ok = (
        arm_off['pristine_bases']['tso_is_none'] and arm_off['pristine_bases']['dso_is_none']
        and arm_off['pristine_bases']['clone_calls'] == 0
        and arm_off['tso'] == {'clone_calls': 0, 'capture_calls': 0}
        and arm_off['dso'] == {'clone_calls': 0, 'capture_calls': 0}
        and arm_off['stdout_markers']['tso_off_warning']
        and arm_off['stdout_markers']['dso_off_warning']
        and arm_off['stdout_markers']['tso_comparator_skipped']
        and arm_off['stdout_markers']['dso_comparator_skipped']
        and arm_off['stdout_markers']['tso_fail_block_named']
        and arm_off['stdout_markers']['dso_fail_block_named']
        and len(arm_off['new_json_markers']) == 2
        and arm_off['new_pkl'] == []
        and arm_off['scan_frozen_snapshots_count'] == 0
    )
    legacy_ok = (
        arm_legacy['pristine_bases']['tso_is_none'] and arm_legacy['pristine_bases']['dso_is_none']
        and arm_legacy['pristine_bases']['clone_calls'] == 0
        and arm_legacy['tso'] == {'clone_calls': n_tso, 'capture_calls': 0}
        and arm_legacy['dso'] == {'clone_calls': n_dso7, 'capture_calls': 0}
        and len(arm_legacy['new_pkl']) == 4
        and arm_legacy['scan_frozen_snapshots_count'] == 4
        and arm_legacy['new_json_markers'] == []
    )
    record['check_i_off_pass'] = bool(off_ok)
    record['check_ii_lightweight_pass'] = bool(light_ok)
    record['check_iib_legacy_clone_regression_pass'] = bool(legacy_ok)

    # ---- check (iv): nothing else differs before the solve call ----------
    diffs = [k for k in sorted(set(probe_light) | set(probe_off))
             if probe_light.get(k) != probe_off.get(k)]
    legacy_diffs = [k for k in sorted(set(probe_light) | set(probe_legacy))
                    if probe_light.get(k) != probe_legacy.get(k)]
    netparam_diffs = [k for k in sorted(set(netparams_light) | set(netparams_off))
                      if netparams_light.get(k) != netparams_off.get(k)]
    record['check_iv'] = {
        'blocks_compared': len(probe_light),
        'blocks_compared_off_arm': len(probe_off),
        'block_state_digest_diffs': diffs,
        'block_state_digest_diffs_lightweight_vs_legacy_clone': legacy_diffs,
        'network_params_digest_diffs': netparam_diffs,
        'admm_params_digest_lightweight': arm_light['admm_params_digest_excluding_modes'],
        'admm_params_digest_off': arm_off['admm_params_digest_excluding_modes'],
        'admm_params_identical_excluding_modes': (
            arm_light['admm_params_digest_excluding_modes']
            == arm_off['admm_params_digest_excluding_modes']),
        'what_is_digested': ('network.capture_block_mutable_state of the block as Network.run_smopf '
                             'receives it: every mutable Param value, every Var (value, raw lb, raw '
                             'ub, fixed), every Constraint/Objective active flag, the ipopt_zL_in/'
                             'ipopt_zU_in/dual suffixes'),
    }
    record['check_iv_pass'] = bool(
        probe_light and len(probe_light) == len(probe_off) and not diffs and not netparam_diffs
        and record['check_iv']['admm_params_identical_excluding_modes'])

    # Restore the default for anything downstream in this process.
    admm_parameters.tso_snapshot_capture_mode = 'lightweight'
    admm_parameters.dso_snapshot_capture_mode = 'lightweight'
    return record


def write_manifest():
    entries = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for name in sorted(files):
            path = os.path.join(root, name)
            if os.path.abspath(path) == os.path.abspath(MANIFEST_PATH):
                continue
            rel = os.path.relpath(path, REPO)
            entries[rel] = {'sha256': _sha256_file(path), 'bytes': os.path.getsize(path)}
    manifest = {'schema': SCHEMA, 'stage': STAGE, 'generated_utc': _utc(),
                'out_dir': os.path.relpath(OUT_DIR, REPO), 'files': entries}
    _refuse_overwrite(MANIFEST_PATH)
    with open(MANIFEST_PATH, 'w') as handle:
        json.dump(manifest, handle, indent=1)
    return manifest


def main():
    _refuse_overwrite(RESULTS_PATH)
    _refuse_overwrite(MANIFEST_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()
    results = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'started_utc': _utc(),
        'interpreter': sys.executable, 'argv': sys.argv, 'cwd': os.getcwd(),
        'git_head': _git(['rev-parse', 'HEAD']),
        'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
        'eval_id': EVAL_ID,
        'memory_evidence': {
            'source': 'data/SRP1/Results/P515S44/scale_measurement/paper_build/'
                      'watchdog_abort_build.json + rss_samples_build.jsonl',
            'stage_at_abort': 'pristine clones kept by the ADMM loop (tso_pristine_base, dso_pristine_base)',
            'last_pre_stage_footprint_bytes': 19645758080,
            'abort_measure_bytes': 25824742624,
            'delta_bytes': 25824742624 - 19645758080,
            'delta_gib': round((25824742624 - 19645758080) / (1 << 30), 3),
            'note': 'the clone stage did NOT complete, so the delta is a LOWER bound on the saving',
        },
    }
    status = 'complete'
    try:
        print('[S45] check (iii): mode validation', flush=True)
        results['check_iii_validation'] = check_validation()
        print('[S45] check (v): scale script --snapshots switch', flush=True)
        results['check_v_scale_script'] = check_scale_script_switch()
        print('[S45] checks (i)/(ii)/(iv): dispatch arms on a real ADMM-ready state', flush=True)
        results['dispatch'] = run_dispatch_arms({})
    except BaseException as error:  # noqa: BLE001 -- recorded, then re-raised
        status = 'error'
        results['error'] = f'{type(error).__name__}: {error}'
        results['traceback'] = traceback.format_exc()
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    GUARD.uninstall()
    results['guard'] = {'permitted': list(GUARD.permitted), 'counts': dict(GUARD.counts),
                        'verify_failures': GUARD.verify(expected_solves=0, expected_execs=0)}
    dispatch = results.get('dispatch', {})
    gates = {
        'check_i_off': dispatch.get('check_i_off_pass'),
        'check_ii_lightweight': dispatch.get('check_ii_lightweight_pass'),
        'check_iib_legacy_clone_regression': dispatch.get('check_iib_legacy_clone_regression_pass'),
        'check_iii_validation': (results.get('check_iii_validation') or {}).get('pass'),
        'check_iv_configuration_identical': dispatch.get('check_iv_pass'),
        'check_v_scale_script': (results.get('check_v_scale_script') or {}).get('pass'),
        'zero_solves': not results['guard']['verify_failures'],
    }
    results['gates'] = gates
    results['status'] = status if status == 'error' else ('PASS' if all(gates.values()) else 'FAIL')
    results['wall_s'] = round(time.time() - started, 2)
    results['ended_utc'] = _utc()
    _refuse_overwrite(RESULTS_PATH)
    with open(RESULTS_PATH, 'w') as handle:
        json.dump(results, handle, indent=1, default=str)
    manifest = write_manifest()
    print(json.dumps({'status': results['status'], 'gates': gates,
                      'guard_counts': results['guard']['counts'],
                      'results': os.path.relpath(RESULTS_PATH, REPO),
                      'manifest_files': len(manifest['files'])}, indent=1), flush=True)
    return 0 if results['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
