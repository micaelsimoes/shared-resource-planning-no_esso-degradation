"""
P5.15 Addendum 83 ruling 3 (iv) (task W183c) -- ZERO-SOLVE gates of the no-reverse-flow (NRF) production option.
Model BUILDS are permitted (Addendum 83 names them); nothing is solved. Pickle loads are permitted ONLY in gate 1, on
a list declared before any load and counted exactly.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 83 ruling 3 (iv) and the W183 order item W183c ("production change on
both copies and both DSO builders; fixture-loading check of every preserved pickle; gate: option off -> NL files
byte-identical to the pre-change build for x = 0 and the reference unit; option on at x = 0 -> exactly 12 x 24 x 3 new
rows per copy, none active elsewhere"); P5_15_W182E_CODE_PATHS.md (Z6 c); P5_15_W183_0_NL_REFERENCE.md;
P5_15_W183A_OPTIONS.md.

THE CHANGE UNDER TEST (committed before this script runs). `network_parameters.NetworkParameters.no_reverse_flow`
(default False; params-file key 'no_reverse_flow', a JSON boolean). `shared_resources_planning`: the row
`INTERFACE_NO_REVERSE_FLOW_ROW` = 'interface_no_reverse_flow_p', CONSTRUCTED ONLY when the network's option is on --
TSO `pc_adn[dn, s_m, s_o, p] >= 0` in `create_transmission_network_model`, DSO `pg_adn[s_m, s_o, p] >= 0` in
`create_distribution_networks_models_sequential` and `create_distribution_network_model` -- with module-level rule
functions. The harness (`p515_s44_campaign_harness.py`) asserts the NRF capture path before the run and records the
NRF rows and binding hours at terminal; the NL-digest harness (`p515_s53_w183_nl_digest.py`) gains `--nrf`.

GATES (all zero solves; `SolveProfileGuard(permitted=())` installed before any project import, verified at exactly 0;
each digest subprocess has its own guard, read back from its JSON)
  1  Fixture loading: every preserved pickle the repo's fixture checks name -- the union of every .pkl tracked in git
     (the W39 / W51 checks' scope) and the untracked anchors named by p515_s49_memory_fix_checks.py (FIXTURES),
     p515_s31c_zero_solve_checks.py, p515_s36_clone_capture_checks.py / p515_s40 / p515_s42 (FIXTURE_PATHS) and
     p515_s53_init_fix_zero_solve_checks.py (cycle-21 anchor + FrozenSMOPF/**) -- unpickles under the new code. The
     list and its length are fixed before the first load; loads are counted and must equal it exactly; a load of any
     other file, or a `pickle.loads`, raises and is counted.
  2  Option off -> identity: the NL-digest harness for all 8 configurations (x0/unit x A/N x 1x/2x; its default
     determinism child) into a fresh `--off-out`; every block of both builders and every ESSO node: NL plain, labelled
     .nl, .row, .col sha256, block digest and every component digest equal to the committed W183-0 reference
     (`w183_nl_reference_pre/w183_nl_digest.json.gz`); eval keys equal. Expert prediction: the NRF-off NL identity holds.
  3  Option on: `--nrf` for x0_A_1x into a fresh `--on-x0-out` and for unit_A_1x into a fresh `--on-unit-out`; per
     block against the off run's same configuration: exactly ONE new Constraint component (the NRF row), absent from
     the off build; 72 indices per TSO block (3 DNs x 24 h; 864 over 12 blocks) and 24 per DSO block (864 over 36
     blocks) in BOTH DSO builders; every index active; no other component differs; ESSO NL and digests identical; the
     NL row names of the on build = the off build's + exactly the NRF rows. And the NRF component is absent from every
     block of every configuration of the off run.
  4  In-process, x0_A_1x, fresh planning objects, production's own builders with every `optimize` stubbed (as the
     NL-digest harness does): without the option `hasattr(block, row)` is False on every TSO and DSO block (sequential
     and parallel builders); with it (through the hook) True, with the counts of gate 3; `NetworkParameters()` default
     False; the params-file reader (a scratch copy of case9_params.json: key absent / true / false / a non-bool);
     `H.nrf_option_production_support()` returns True. Supplementary, REPORTED not gated: `H.capture_nrf_terminal` and
     `H.assert_nrf_capture_path` exercised on the unsolved on-build models (values are initial values, not a solution).
  5  W183a's T4 inverted: with `nrf_option=True` the hook sets the flag on the TSO and every DSO and reads it back
     (record checks pass); undeclared with a DSO flag forced on raises naming 'every_network_flag_is_off'; undeclared
     with every flag off passes; `nrf_option=False` explicit passes.

OUTPUT (write-once; `--out` may pre-exist holding only launch.log; the three digest dirs must not exist):
  <out>/w183c_nrf_gates.json (gate_result_io)   <out>/manifest_sha256.json
  each digest dir: the harness's own output + w183_nl_digest.json.gz (gzip mtime 0) + manifest_post_run_sha256.json
LAUNCH (attached, alone, both streams captured; this script, the harnesses and production committed first):
  mkdir -p data/SRP1/Results/P515S55/w183c_nrf_gates
  NLP_SOLVER_PATH=/usr/local/bin/ipopt /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
      p515_s53_w183c_nrf_gates.py --out data/SRP1/Results/P515S55/w183c_nrf_gates \\
      --off-out data/SRP1/Results/P515S55/w183_nl_post_w183c_off \\
      --on-x0-out data/SRP1/Results/P515S55/w183_nl_post_w183c_on_x0 \\
      --on-unit-out data/SRP1/Results/P515S55/w183_nl_post_w183c_on_unit \\
      --scratch <scratchpad>/w183c_gates --expect-production-commit <the W183c commit> \\
      > data/SRP1/Results/P515S55/w183c_nrf_gates/launch.log 2>&1
Exit 0 iff every gate passes.
"""

import argparse
import gc
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183c NRF gates (zero solves, builds only)').install()

import pickle  # noqa: E402

_REAL_PICKLE_LOAD = pickle.load
PICKLE = {'permitted_paths': None, 'load': 0, 'loads': 0, 'refused_load': 0, 'loaded_paths': []}


def _guarded_load(handle, *args, **kwargs):
    """Gate 1 only: a load is let through iff its file is on the declared list (set for gate 1 alone)."""
    name = os.path.abspath(getattr(handle, 'name', '') or '')
    if PICKLE['permitted_paths'] is None or name not in PICKLE['permitted_paths']:
        PICKLE['refused_load'] += 1
        raise RuntimeError(f'W183c: pickle.load of {name!r} refused -- loads are permitted only in gate 1 on the '
                           f'declared fixture list')
    PICKLE['load'] += 1
    PICKLE['loaded_paths'].append(name)
    return _REAL_PICKLE_LOAD(handle, *args, **kwargs)


def _blocked_loads(*_a, **_k):
    PICKLE['loads'] += 1
    raise RuntimeError('W183c: pickle.loads called -- not permitted')


pickle.load, pickle.loads = _guarded_load, _blocked_loads

import gate_result_io as GRIO  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402

SCRIPT_REL = os.path.basename(__file__)
HARNESS_REL = 'p515_s44_campaign_harness.py'
NL_DIGEST_REL = 'p515_s53_w183_nl_digest.py'
STAGE = 'P5.15 Addendum 83 W183c -- no-reverse-flow production option: gates 1-5 (zero solves, builds only)'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 83 ruling 3 (iv); W183 order (W183c)',
             'P5_15_W182E_CODE_PATHS.md Z6 (c)', 'P5_15_W183_0_NL_REFERENCE.md', 'P5_15_W183A_OPTIONS.md']
REFERENCE_DIR_REL = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183_nl_reference_pre')
ALL_CONFIGS = ('x0_A_1x', 'x0_A_2x', 'x0_N_1x', 'x0_N_2x', 'unit_A_1x', 'unit_A_2x', 'unit_N_1x', 'unit_N_2x')
ON_CONFIGS = {'on_x0': 'x0_A_1x', 'on_unit': 'unit_A_1x'}
BUILDERS = ('sequential', 'parallel')
CAP = 300
REQUIRED_CONSECUTIVE = 10
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CONVERGENCE_DEPTH_TAIL = {'compl_inf_tol': 1e-06, 'enabled': True}
INVESTMENT_YEAR = 2025
X0 = {5: (0.0, 0.0), 7: (0.0, 0.0), 9: (0.0, 0.0)}
UNIT = {5: (0.0, 0.0), 7: (0.25, 1.0), 9: (0.0, 0.0)}
# Declared before the run: the SRP1 counts the order states (12 blocks = 3 years x 4 days; 3 DNs; 24 h; 1 x 1
# scenarios) -- per TSO block 3 x 24 = 72, per DSO block 24; totals 864 per copy.
EXPECTED_ROWS_PER_TSO_BLOCK = 72
EXPECTED_ROWS_PER_DSO_BLOCK = 24
EXPECTED_N_TSO_BLOCKS = 12
EXPECTED_N_DSO_BLOCKS = 36
EXPECTED_TOTAL_PER_COPY = 864
# Gate 1: the untracked anchors the repo's fixture checks name (tracked .pkl files are added from git at run time).
NAMED_FIXTURES = {
    'data/SRP1/Results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl': ['p515_s49_memory_fix_checks.py'],
    'data/SRP1/Results/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl': [
        'p515_s49_memory_fix_checks.py', 'p515_s36_clone_capture_checks.py', 'p515_s33e2_zero_solve_checks.py'],
    'data/SRP1/Results/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl': [
        'p515_s49_memory_fix_checks.py', 'p515_s36_clone_capture_checks.py', 'p515_s33e2_zero_solve_checks.py'],
    'data/SRP1/Results/P512R/cycle21_pre_setup/snapshot.pkl': [
        'p515_s49_memory_fix_checks.py', 'p515_s31c_zero_solve_checks.py', 'p515_s36_clone_capture_checks.py',
        'p515_s53_init_fix_zero_solve_checks.py'],
    'data/SRP1/Results/P512R/cycle21_prepared/snapshot.pkl': ['p515_s36_clone_capture_checks.py'],
    'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_DSO_node7_case33_2_2025_Autumn_cycle7.pkl':
        ['p515_s31c_zero_solve_checks.py', 'p515_s36_clone_capture_checks.py'],
    'data/SRP1/Results/P512R/production_snapshots/FrozenSMOPF/matched_success_TSO_case9_2025_Summer_cycle7.pkl':
        ['p515_s36_clone_capture_checks.py'],
    'data/SRP1/Results/P515G3F_r2/results/FrozenSMOPF/frozen_DSO_node7_case33_2_2035_Autumn_cycle38.pkl':
        ['p515_s31c_zero_solve_checks.py'],
    'data/SRP1/Results/P515F/t2_results/FrozenSMOPF/failure_TSO_case9_2025_Spring_cycle1.pkl':
        ['p515_s36_clone_capture_checks.py'],
}
FROZEN_SMOPF_GLOB_DIR = os.path.join('data', 'SRP1', 'Results', 'FrozenSMOPF')  # p515_s53_init_fix: FrozenSMOPF/**


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'[W183c {datetime.now(timezone.utc).strftime("%H:%M:%S")}] {msg}', flush=True)


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git(args):
    return subprocess.run(['git'] + args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def _git_status(paths):
    text = _git(['status', '--porcelain', '--'] + list(paths))
    return {line[3:]: line[:2] for line in text.splitlines()}


def _row_name():
    import shared_resources_planning as srp
    return srp.INTERFACE_NO_REVERSE_FLOW_ROW


# ======================================================================================================================
#  gate 1 -- fixtures
# ======================================================================================================================
def declared_fixture_list():
    tracked = [p for p in _git(['ls-files', '--', '*.pkl']).splitlines() if p]
    globbed = []
    for root, _dirs, files in os.walk(os.path.join(REPO, FROZEN_SMOPF_GLOB_DIR)):
        for f in sorted(files):
            if f.endswith('.pkl'):
                globbed.append(os.path.relpath(os.path.join(root, f), REPO))
    named_by = {}
    for rel in tracked:
        named_by.setdefault(rel, []).append('git ls-files *.pkl (p515_s51_row18_zero_solve_checks.py F scope)')
    for rel, checks in NAMED_FIXTURES.items():
        named_by.setdefault(rel, []).extend(checks)
    for rel in globbed:
        named_by.setdefault(rel, []).append('p515_s53_init_fix_zero_solve_checks.py (FrozenSMOPF/**)')
    return dict(sorted(named_by.items())), set(tracked)


def gate1_fixtures():
    declared, tracked = declared_fixture_list()
    absent = [rel for rel in declared if not os.path.isfile(os.path.join(REPO, rel))]
    n_declared = len(declared)
    _log(f'gate 1: {n_declared} fixtures declared ({len(tracked)} tracked + named untracked anchors); '
         f'absent {len(absent)}')
    PICKLE['permitted_paths'] = {os.path.abspath(os.path.join(REPO, rel)) for rel in declared}
    loaded, failed = [], []
    load_before = PICKLE['load']
    try:
        for rel in declared:
            path = os.path.join(REPO, rel)
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'rb') as handle:
                    payload = pickle.load(handle)
                loaded.append({'path': rel, 'sha256': _sha_file(path), 'bytes': os.path.getsize(path),
                               'type': type(payload).__name__, 'tracked': rel in tracked,
                               'named_by': declared[rel]})
                del payload
            except Exception as error:  # noqa: BLE001
                failed.append({'path': rel, 'sha256': _sha_file(path), 'tracked': rel in tracked,
                               'named_by': declared[rel], 'error': f'{type(error).__name__}: {error}'})
            gc.collect()
    finally:
        PICKLE['permitted_paths'] = None
    n_loads = PICKLE['load'] - load_before
    checks = {'no_declared_fixture_absent': not absent,
              'every_declared_fixture_unpickles': not failed and len(loaded) == n_declared,
              'pickle_load_count_equals_declared': n_loads == n_declared,
              'four_s49_fixtures_included': all(r in declared for r in list(NAMED_FIXTURES)[:4]),
              'no_pickle_loads_call': PICKLE['loads'] == 0}
    return {'pass': all(checks.values()), 'checks': checks, 'n_declared': n_declared, 'n_tracked': len(tracked),
            'n_loaded': len(loaded), 'n_failed': len(failed), 'n_pickle_load_calls': n_loads, 'absent': absent,
            'failed': failed, 'loaded': loaded}


# ======================================================================================================================
#  gates 4 / 5 -- in-process builds and hook cases
# ======================================================================================================================
def spec_like(declared, pin):
    return {'configuration': {'overrides': {}, 'arm_label': 's39_D',
                              'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                              'ess_ageing_baseline': declared, 'ess_params_file': pin,
                              'ess_ageing_baseline_label': 'BASELINE (C2 + phi_cal 0.985 + soh_min 0.70)',
                              'convergence_depth_tail': dict(CONVERGENCE_DEPTH_TAIL)},
            'cap': CAP, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE}


def construct(G, ctx, case, investment_map, nrf_option, mutate=None):
    """`_construct_arm_planning` + the campaign child's configuration hook (exactly as the NL-digest harness does).
    Returns (planning, sed, candidate, outcome record)."""
    report, holder = {}, {}
    eval_id = f"p515s55_w183c_{ctx['run_tag']}_{case}"
    planning, sed, candidate = G._construct_arm_planning(
        's39_D', os.path.join(ctx['scratch'], 'builds', case), report, investment_map=investment_map,
        eval_id=eval_id, num_max_iters_override=CAP, apply_rho=False, investment_year=INVESTMENT_YEAR)
    mutation = mutate(planning) if mutate is not None else None
    hook = H._config_hook_factory(ctx['spec'], holder, overrides={}, model_variant=None,
                                  investment_year=INVESTMENT_YEAR, expected_floor_rows=None,
                                  flex_price_multiplier=1.0, interface_deviation_premium=None,
                                  release_solution_bookkeeping=None, nrf_option=nrf_option)
    try:
        hook(planning=planning, sed=sed, candidate=candidate, report=report)
        outcome, error = 'passed', None
    except Exception as exc:  # noqa: BLE001
        outcome, error = 'raised', f'{type(exc).__name__}: {exc}'
    nets = [('TSO', planning.transmission_network.params)] + [
        (f'DSO{n}', dn.params) for n, dn in sorted(planning.distribution_networks.items())]
    rec = {'case': case, 'eval_id': eval_id, 'nrf_option': nrf_option, 'mutation': mutation, 'outcome': outcome,
           'error': error, 'w183a_nrf_option': (report.get('rule_eleven_checklist') or {}).get('w183a_nrf_option'),
           'no_reverse_flow_read_back': {n: getattr(p, 'no_reverse_flow', '<absent>') for n, p in nets}}
    return planning, sed, candidate, rec


def _block_rows(block, row):
    comp = block.component(row)
    return {'hasattr': hasattr(block, row), 'n_indices': 0 if comp is None else len(comp),
            'n_active': 0 if comp is None else sum(1 for c in comp.values() if c.active)}


def build_in_process(G, srp, ctx, case, nrf_option):
    """Sequential: production's `run_operational_planning` with every `optimize` stubbed (TSO, each DSO, the ESSO);
    parallel: `create_distribution_network_model` in-process, stubbed. Fresh planning object for each builder."""
    row = srp.INTERFACE_NO_REVERSE_FLOW_ROW
    out = {}
    # sequential
    planning, sed, candidate, hook_rec = construct(G, ctx, f'{case}_seq', X0, nrf_option)
    if hook_rec['outcome'] != 'passed':
        raise RuntimeError(f'gate 4 build {case}: hook {hook_rec}')
    calls, captured = {}, {}

    def net_stub(tag):
        def stub(model, *args, **kwargs):
            calls[tag] = calls.get(tag, 0) + 1
            captured[tag] = model
            return {y: {d: None for d in model[y]} for y in model}
        return stub

    def esso_stub(models, *args, **kwargs):
        calls['ESSO'] = calls.get('ESSO', 0) + 1
        return {n: None for n in models}
    planning.transmission_network.optimize = net_stub('TSO')
    for node_id, dn in planning.distribution_networks.items():
        dn.optimize = net_stub(f'DSO{node_id}')
    sed.optimize = esso_stub
    _conv, _res, _models, _sens, _prim, run_state = planning.run_operational_planning(
        type='distributed', candidate_solution=deepcopy(candidate), print_results=False, debug_flag=False,
        return_state=True)
    expected_calls = {'TSO': 1, 'ESSO': 1, **{f'DSO{n}': 1 for n in planning.distribution_networks}}
    seq = {'hook': hook_rec, 'stub_calls': dict(calls), 'stub_calls_exact': calls == expected_calls,
           'initialization_failed': run_state.get('initialization_failed'), 'blocks': {}}
    for tag, model in sorted(captured.items()):
        for y in model:
            for d in model[y]:
                seq['blocks'][f'{tag}|{y}|{d}'] = _block_rows(model[y][d], row)
    supplementary = None
    if nrf_option is True:
        supplementary = {}
        models = {'tso': captured['TSO'],
                  'dso': {n: captured[f'DSO{n}'] for n in planning.distribution_networks}}
        try:
            supplementary['assert_nrf_capture_path'] = H.assert_nrf_capture_path()
        except Exception as error:  # noqa: BLE001
            supplementary['assert_nrf_capture_path'] = {'error': f'{type(error).__name__}: {error}'}
        try:
            cap = H.capture_nrf_terminal(planning, models)
            supplementary['capture_nrf_terminal'] = {
                'status': cap.get('status'), 'row_counts_per_network': cap.get('row_counts_per_network'),
                'n_binding_hours': cap.get('n_binding_hours'), 'per_node': cap.get('per_node'),
                'binding_tolerance_mw': cap.get('binding_tolerance_mw'),
                'binding_hours_first_3': (cap.get('binding_hours') or [])[:3],
                'note': ('unsolved initialisation models: the values are the Vars\' initial values (expected interface '
                         'P initialised at 0.0), not a solution -- this exercises the code path only')}
        except Exception as error:  # noqa: BLE001
            supplementary['capture_nrf_terminal'] = {'status': 'error', 'error': f'{type(error).__name__}: {error}',
                                                     'traceback': traceback.format_exc()}
    out['sequential'] = seq
    del planning, sed, candidate, captured, _res, _models, run_state
    gc.collect()
    # parallel
    planning, sed, candidate, hook_rec = construct(G, ctx, f'{case}_par', X0, nrf_option)
    if hook_rec['outcome'] != 'passed':
        raise RuntimeError(f'gate 4 build {case} (parallel): hook {hook_rec}')
    calls, captured = {}, {}
    premium = planning.params.admm.interface_deviation_premium
    for node_id in sorted(planning.distribution_networks):
        dn = planning.distribution_networks[node_id]
        dn.optimize = net_stub(f'DSO{node_id}')
        srp.create_distribution_network_model(node_id, dn, candidate['total_capacity'], premium['alpha'],
                                              premium['floor'])
    par = {'hook': hook_rec, 'stub_calls': dict(calls),
           'stub_calls_exact': calls == {f'DSO{n}': 1 for n in planning.distribution_networks}, 'blocks': {}}
    for tag, model in sorted(captured.items()):
        for y in model:
            for d in model[y]:
                par['blocks'][f'{tag}|{y}|{d}'] = _block_rows(model[y][d], row)
    out['parallel'] = par
    del planning, sed, candidate, captured
    gc.collect()
    return out, supplementary


def _count_ok(blocks, on):
    tso = {k: v for k, v in blocks.items() if k.startswith('TSO|')}
    dso = {k: v for k, v in blocks.items() if k.startswith('DSO')}
    if not on:
        return {'no_block_has_the_row': not any(v['hasattr'] for v in blocks.values()),
                'n_tso_blocks': len(tso), 'n_dso_blocks': len(dso)}
    return {'every_block_has_the_row': all(v['hasattr'] for v in blocks.values()),
            'tso_rows_per_block_72': all(v['n_indices'] == EXPECTED_ROWS_PER_TSO_BLOCK for v in tso.values()),
            'dso_rows_per_block_24': all(v['n_indices'] == EXPECTED_ROWS_PER_DSO_BLOCK for v in dso.values()),
            'every_index_active': all(v['n_active'] == v['n_indices'] for v in blocks.values()),
            'tso_total': sum(v['n_indices'] for v in tso.values()),
            'dso_total': sum(v['n_indices'] for v in dso.values()),
            'n_tso_blocks': len(tso), 'n_dso_blocks': len(dso)}


def gate4_builds(G, srp, ctx):
    import network_parameters as NP
    res = {}
    supported, description = H.nrf_option_production_support()
    res['production_support'] = {'supported': supported, 'description': description}
    res['default_attribute'] = getattr(NP.NetworkParameters(), 'no_reverse_flow', '<absent>')
    # the params-file reader, on scratch copies of the TSO params file
    src = os.path.join(REPO, 'data', 'SRP1', 'case9', 'case9_params.json')
    with open(src) as handle:
        base = json.load(handle)
    reader = {}
    for tag, value in (('absent', None), ('true', True), ('false', False), ('non_bool_1', 1), ('non_bool_str', 'yes')):
        data = dict(base)
        if value is not None:
            data['no_reverse_flow'] = value
        path = os.path.join(ctx['scratch'], 'reader', f'case9_params_{tag}.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(data, handle)
        p = NP.NetworkParameters()
        try:
            p.read_parameters_from_file(path)
            reader[tag] = {'outcome': 'read', 'no_reverse_flow': p.no_reverse_flow}
        except SystemExit as exc:
            reader[tag] = {'outcome': 'exit', 'code': exc.code}
    res['reader'] = reader
    res['source_params_file_has_key'] = 'no_reverse_flow' in base
    res['builds'] = {}
    sup = None
    for case, nrf in (('x0_off', None), ('x0_on', True)):
        t0 = time.time()
        res['builds'][case], s = build_in_process(G, srp, ctx, case, nrf)
        if s is not None:
            sup = s
        _log(f"gate 4 build {case}: {time.time() - t0:.0f}s; stub calls seq "
             f"{res['builds'][case]['sequential']['stub_calls']} par {res['builds'][case]['parallel']['stub_calls']}")
    counts = {case: {b: _count_ok(res['builds'][case][b]['blocks'], case == 'x0_on') for b in BUILDERS}
              for case in res['builds']}
    res['counts'] = counts
    res['supplementary_capture_on_unsolved_models'] = sup
    off, on = counts['x0_off'], counts['x0_on']
    checks = {
        'nrf_option_production_support_true': supported is True,
        'network_parameters_default_false': res['default_attribute'] is False,
        'reader_absent_keeps_false': reader['absent'] == {'outcome': 'read', 'no_reverse_flow': False},
        'reader_true_sets_true': reader['true'] == {'outcome': 'read', 'no_reverse_flow': True},
        'reader_false_sets_false': reader['false'] == {'outcome': 'read', 'no_reverse_flow': False},
        'reader_non_bool_refused': all(reader[t]['outcome'] == 'exit' for t in ('non_bool_1', 'non_bool_str')),
        'case9_params_file_does_not_declare_the_key': not res['source_params_file_has_key'],
        'stub_calls_exact_every_build': all(res['builds'][c][b]['stub_calls_exact'] for c in res['builds']
                                            for b in BUILDERS),
        'off_hasattr_false_every_tso_and_dso_block_both_builders': all(off[b]['no_block_has_the_row']
                                                                       for b in BUILDERS),
        'off_block_counts_12_tso_36_dso_seq_36_dso_par': (
            off['sequential']['n_tso_blocks'] == EXPECTED_N_TSO_BLOCKS
            and off['sequential']['n_dso_blocks'] == EXPECTED_N_DSO_BLOCKS
            and off['parallel']['n_tso_blocks'] == 0 and off['parallel']['n_dso_blocks'] == EXPECTED_N_DSO_BLOCKS),
        'on_hasattr_true_every_block_both_builders': all(on[b]['every_block_has_the_row'] for b in BUILDERS),
        'on_tso_72_per_block_total_864': (on['sequential']['tso_rows_per_block_72']
                                          and on['sequential']['tso_total'] == EXPECTED_TOTAL_PER_COPY
                                          and on['sequential']['n_tso_blocks'] == EXPECTED_N_TSO_BLOCKS),
        'on_dso_24_per_block_total_864_both_builders': all(
            on[b]['dso_rows_per_block_24'] and on[b]['dso_total'] == EXPECTED_TOTAL_PER_COPY
            and on[b]['n_dso_blocks'] == EXPECTED_N_DSO_BLOCKS for b in BUILDERS),
        'on_every_index_active': all(on[b]['every_index_active'] for b in BUILDERS),
    }
    res['checks'] = checks
    res['pass'] = all(checks.values())
    return res


def gate5_hook(G, ctx):
    cases = {}

    def flag_on_first_dso(planning):
        node = sorted(planning.distribution_networks)[0]
        planning.distribution_networks[node].params.no_reverse_flow = True
        return {'dso_flag_forced_on_before_hook': f'DSO{node}'}
    for case, nrf, mutate in (('nrf_true', True, None), ('undeclared_flag_forced_on', None, flag_on_first_dso),
                              ('undeclared_off', None, None), ('nrf_false_explicit', False, None)):
        planning, sed, candidate, rec = construct(G, ctx, f'hook_{case}', UNIT, nrf, mutate=mutate)
        cases[case] = rec
        del planning, sed, candidate
        gc.collect()
    expected_nets = {'TSO', 'DSO5', 'DSO7', 'DSO9'}
    t = cases['nrf_true']
    trec = t['w183a_nrf_option'] or {}
    u = cases['undeclared_flag_forced_on']
    o = cases['undeclared_off']
    f = cases['nrf_false_explicit']
    checks = {
        'nrf_true_passes': t['outcome'] == 'passed',
        'nrf_true_flag_read_back_true_on_tso_and_every_dso': (
            set(t['no_reverse_flow_read_back']) == expected_nets
            and all(v is True for v in t['no_reverse_flow_read_back'].values())),
        'nrf_true_record_in_force_true_and_checks_pass': (
            set((trec.get('in_force') or {})) == expected_nets
            and all(v is True for v in (trec.get('in_force') or {}).values())
            and bool(trec.get('checks')) and all(trec['checks'].values())
            and trec.get('label') == H.NRF_OPTION_LABEL),
        'undeclared_forced_on_raises_naming_every_network_flag_is_off': (
            u['outcome'] == 'raised' and (u['error'] or '').startswith('RuntimeError')
            and 'every_network_flag_is_off' in (u['error'] or '')),
        'undeclared_off_passes_with_every_flag_off': (
            o['outcome'] == 'passed' and all(v is False for v in o['no_reverse_flow_read_back'].values())
            and all(((o['w183a_nrf_option'] or {}).get('checks') or {'missing': False}).values())),
        'nrf_false_explicit_passes_with_every_flag_off': (
            f['outcome'] == 'passed' and all(v is False for v in f['no_reverse_flow_read_back'].values())),
    }
    return {'pass': all(checks.values()), 'checks': checks, 'cases': cases}


# ======================================================================================================================
#  gates 2 / 3 -- the NL-digest harness
# ======================================================================================================================
def run_digest(args, out_dir, label, configs, nrf, determinism):
    if os.path.exists(out_dir):
        raise RuntimeError(f'digest output directory exists (never re-run onto an existing dir): {out_dir}')
    os.makedirs(out_dir)
    cmd = [sys.executable, '-u', os.path.join(REPO, NL_DIGEST_REL), '--out', out_dir, '--label', label,
           '--scratch', os.path.join(args.scratch, f'digest_{label}'), '--configs', ','.join(configs)]
    if not determinism:
        cmd.append('--no-determinism')
    if nrf:
        cmd.append('--nrf')
    if args.expect_production_commit:
        cmd += ['--expect-production-commit', args.expect_production_commit]
    if args.allow_uncommitted:
        cmd.append('--allow-uncommitted-harness')
    log_path = os.path.join(out_dir, 'launch.log')
    _log(f'digest {label}: {cmd} (stdout + stderr -> {os.path.relpath(log_path, REPO)})')
    t0 = time.time()
    with open(log_path, 'w') as log:
        rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)).returncode
    info = {'command': cmd, 'returncode': rc, 'wall_s': time.time() - t0, 'launch_log': os.path.relpath(log_path, REPO)}
    post_json = os.path.join(out_dir, 'w183_nl_digest.json')
    if not os.path.isfile(post_json):
        info['error'] = 'no result JSON'
        return info, None
    raw = open(post_json, 'rb').read()
    gz_path = post_json + '.gz'
    with open(gz_path, 'wb') as handle:
        with gzip.GzipFile(filename='', mode='wb', fileobj=handle, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    info.update({'json_sha256': _sha_bytes(raw), 'json_bytes': len(raw), 'json_gz': os.path.relpath(gz_path, REPO),
                 'json_gz_sha256': _sha_file(gz_path),
                 'json_gz_roundtrip_equal': _sha_bytes(gzip.open(gz_path).read()) == _sha_bytes(raw)})
    post = json.loads(raw)
    del raw
    guard = post.get('solve_profile_guard') or {}
    info.update({'all_ok': post.get('all_ok'), 'checks': post.get('checks'), 'git_head_at_run': post.get(
        'git_head_at_run'), 'solve_profile_guard': guard,
        'guard_zero': (not guard.get('verify_0_failures')
                       and all(v == 0 for v in (guard.get('counts') or {'missing': 1}).values())),
        'harness_sha256_imported': (post.get('repo_modules_imported_sha256_start') or {}).get(HARNESS_REL),
        'production_modules_sha256': {k: v for k, v in ((post.get('production') or {}).get('modules_sha256')
                                                         or {}).items()
                                      if k in ('shared_resources_planning.py', 'network_parameters.py',
                                               'model_construction_helpers.py')}})
    manifest = {}
    for fname in ('w183_nl_digest.json', 'w183_nl_digest.json.gz', 'launch.log', 'manifest_sha256.json'):
        p = os.path.join(out_dir, fname)
        if os.path.isfile(p):
            manifest[os.path.relpath(p, REPO)] = _sha_file(p)
    child_dir = os.path.join(out_dir, 'determinism_child')
    for fname in ('w183_nl_digest.json', 'manifest_sha256.json'):
        p = os.path.join(child_dir, fname)
        if os.path.isfile(p):
            manifest[os.path.relpath(p, REPO)] = _sha_file(p)
    with open(os.path.join(out_dir, 'manifest_post_run_sha256.json'), 'w') as handle:
        GRIO.dump(dict(sorted(manifest.items())), handle, indent=1, sort_keys=True)
    info['manifest_post_run'] = manifest
    return info, post


def _block_files_equal(a, b):
    files = {'plain': (a['nl']['plain'] or {}).get('sha256') == (b['nl']['plain'] or {}).get('sha256')}
    for f in ('nl', 'row', 'col'):
        files[f'symbolic_{f}'] = ((a['nl']['symbolic'] or {}).get(f, {}).get('sha256')
                                  == (b['nl']['symbolic'] or {}).get(f, {}).get('sha256'))
    files['plain_sha256_not_null'] = (a['nl']['plain'] or {}).get('sha256') is not None
    files['block_digest'] = a['sha256'] == b['sha256']
    files['component_digests'] = a['components'] == b['components']
    return files


def gate2_off_identity(ref, post, off_info):
    per, n_blocks, n_identical = {}, 0, 0
    keys_equal = {}
    for c in ALL_CONFIGS:
        per[c] = {}
        keys_equal[c] = ref['records'][c]['eval_key_base'] == post['records'][c]['eval_key_base']
        for b in BUILDERS:
            ra, rb = ref['records'][c][b]['blocks'], post['records'][c][b]['blocks']
            diffs = []
            for key in sorted(set(ra) | set(rb)):
                if key not in ra or key not in rb:
                    diffs.append({'block': key, 'missing_in': 'reference' if key not in ra else 'post'})
                    continue
                files = _block_files_equal(ra[key], rb[key])
                n_blocks += 1
                if all(files.values()):
                    n_identical += 1
                else:
                    diffs.append({'block': key, 'files': files})
            per[c][b] = {'n_blocks_reference': len(ra), 'n_blocks_post': len(rb), 'byte_identical': not diffs,
                         'n_esso_blocks': sum(1 for k in rb if k.startswith('ESSO')), 'differences': diffs[:20],
                         'n_differences': len(diffs)}
    expected_blocks = len(ALL_CONFIGS) * (51 + 36)
    checks = {
        'digest_run_exit_0_all_ok': off_info['returncode'] == 0 and off_info.get('all_ok') is True,
        'digest_run_guard_zero': off_info.get('guard_zero') is True,
        'digest_run_used_changed_production': (
            off_info['production_modules_sha256'].get('shared_resources_planning.py')
            != (ref.get('production') or {}).get('modules_sha256', {}).get('shared_resources_planning.py')),
        'every_block_compared_696': n_blocks == expected_blocks,
        'every_block_byte_identical_nl_and_digests': n_blocks == n_identical and all(
            v['byte_identical'] for c in per.values() for v in c.values()),
        'eval_keys_equal_every_config': all(keys_equal.values()),
        'gz_roundtrip_equal': off_info.get('json_gz_roundtrip_equal') is True,
    }
    return {'pass': all(checks.values()), 'checks': checks, 'n_blocks_compared': n_blocks,
            'n_blocks_identical': n_identical, 'expected_blocks': expected_blocks, 'eval_keys_equal': keys_equal,
            'per_config': per,
            'reference_git_head_at_run': ref.get('git_head_at_run'), 'post_git_head_at_run': post.get('git_head_at_run')}


def _load_items(rel):
    with gzip.open(os.path.join(REPO, rel), 'rt') as handle:
        return json.load(handle)


def _row_names(rec):
    with gzip.open(os.path.join(REPO, rec['nl']['symbolic']['row']['gz']), 'rt') as handle:
        return handle.read().splitlines()


def _col_names(rec):
    with gzip.open(os.path.join(REPO, rec['nl']['symbolic']['col']['gz']), 'rt') as handle:
        return handle.read().splitlines()


def gate3_on(off, on, config, on_info, row):
    comp_name = f'Constraint:{row}'
    per_builder, totals = {}, {}
    all_ok = True
    for b in BUILDERS:
        ba, bb = off['records'][config][b]['blocks'], on['records'][config][b]['blocks']
        blocks = {}
        tso_total = dso_total = 0
        n_tso = n_dso = 0
        for key in sorted(set(ba) | set(bb)):
            if key not in ba or key not in bb:
                blocks[key] = {'missing_in': 'off' if key not in ba else 'on'}
                all_ok = False
                continue
            a, o = ba[key], bb[key]
            comps_diff = sorted(n for n in set(a['components']) | set(o['components'])
                                if a['components'].get(n) != o['components'].get(n))
            entry = {'components_differing': comps_diff}
            if key.startswith('ESSO'):
                files = _block_files_equal(a, o)
                entry['files_equal'] = files
                entry['ok'] = all(files.values())
            else:
                is_tso = key.startswith('TSO|')
                expected = EXPECTED_ROWS_PER_TSO_BLOCK if is_tso else EXPECTED_ROWS_PER_DSO_BLOCK
                items = _load_items(o['items_gz'])
                idx = items['items'].get(comp_name) or {}
                active = items['constraint_active'].get(comp_name) or {}
                n_idx, n_act = len(idx), sum(1 for v in active.values() if v is True)
                rows_off, rows_on = _row_names(a), _row_names(o)
                added = sorted(set(rows_on) - set(rows_off))
                entry.update({
                    'n_indices': n_idx, 'n_active': n_act, 'expected': expected,
                    'component_absent_in_off': comp_name not in a['components'],
                    'n_rows_off': len(rows_off), 'n_rows_on': len(rows_on), 'n_rows_added': len(added),
                    'off_rows_subset_of_on': set(rows_off) <= set(rows_on),
                    'added_rows_are_all_nrf': all(r.startswith(f'{row}[') for r in added),
                    'col_identical': _col_names(a) == _col_names(o),
                    'nl_plain_differs': (a['nl']['plain'] or {}).get('sha256') != (o['nl']['plain'] or {}).get(
                        'sha256'),
                })
                entry['ok'] = (comps_diff == [comp_name] and entry['component_absent_in_off'] and n_idx == expected
                               and n_act == n_idx and len(added) == expected and entry['off_rows_subset_of_on']
                               and entry['added_rows_are_all_nrf'] and len(rows_on) == len(rows_off) + expected)
                if is_tso:
                    tso_total += n_idx
                    n_tso += 1
                else:
                    dso_total += n_idx
                    n_dso += 1
            all_ok = all_ok and entry['ok']
            blocks[key] = entry
        per_builder[b] = blocks
        totals[b] = {'tso_total': tso_total, 'dso_total': dso_total, 'n_tso_blocks': n_tso, 'n_dso_blocks': n_dso,
                     'n_esso_blocks': sum(1 for k in blocks if k.startswith('ESSO')),
                     'n_blocks_ok': sum(1 for v in blocks.values() if v.get('ok')), 'n_blocks': len(blocks),
                     'n_blocks_col_identical': sum(1 for v in blocks.values() if v.get('col_identical') is True)}
    checks = {
        'digest_run_exit_0_all_ok': on_info['returncode'] == 0 and on_info.get('all_ok') is True,
        'digest_run_guard_zero': on_info.get('guard_zero') is True,
        'nrf_flags_in_force_as_declared': all(
            on['records'][config][b]['readback_checks'].get('nrf_flags_as_declared') is True for b in BUILDERS),
        'every_network_block_one_new_component_with_expected_rows_all_active_and_esso_identical': all_ok,
        'tso_12_blocks_total_864': (totals['sequential']['n_tso_blocks'] == EXPECTED_N_TSO_BLOCKS
                                    and totals['sequential']['tso_total'] == EXPECTED_TOTAL_PER_COPY),
        'dso_36_blocks_total_864_sequential': (totals['sequential']['n_dso_blocks'] == EXPECTED_N_DSO_BLOCKS
                                               and totals['sequential']['dso_total'] == EXPECTED_TOTAL_PER_COPY),
        'dso_36_blocks_total_864_parallel': (totals['parallel']['n_dso_blocks'] == EXPECTED_N_DSO_BLOCKS
                                             and totals['parallel']['dso_total'] == EXPECTED_TOTAL_PER_COPY),
        'esso_3_nodes_identical': totals['sequential']['n_esso_blocks'] == 3 and all(
            v.get('ok') for k, v in per_builder['sequential'].items() if k.startswith('ESSO')),
        'eval_key_carries_the_option': (on['records'][config]['eval_key_base']
                                        != off['records'][config]['eval_key_base']
                                        and on['records'][config].get('nrf_option') is True),
        'gz_roundtrip_equal': on_info.get('json_gz_roundtrip_equal') is True,
    }
    return {'pass': all(checks.values()), 'checks': checks, 'config': config, 'totals': totals,
            'eval_key_on': on['records'][config]['eval_key_base'], 'eval_key_off': off['records'][config]['eval_key_base'],
            'per_builder': per_builder}


def off_absence(off, row):
    comp_name = f'Constraint:{row}'
    hits = [f'{c}|{b}|{k}' for c in ALL_CONFIGS for b in BUILDERS
            for k, v in off['records'][c][b]['blocks'].items() if comp_name in v['components']]
    n = sum(len(off['records'][c][b]['blocks']) for c in ALL_CONFIGS for b in BUILDERS)
    return {'pass': not hits and n == len(ALL_CONFIGS) * 87, 'n_blocks_scanned': n, 'blocks_with_component': hits}


# ======================================================================================================================
#  main
# ======================================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--off-out', required=True)
    parser.add_argument('--on-x0-out', required=True)
    parser.add_argument('--on-unit-out', required=True)
    parser.add_argument('--scratch', required=True, help='scratch dir outside the repo; must not exist')
    parser.add_argument('--expect-production-commit', default=None)
    parser.add_argument('--skip-digest', action='store_true', help='smoke runs only: gates 1, 4, 5')
    parser.add_argument('--skip-fixtures', action='store_true', help='smoke runs only')
    parser.add_argument('--allow-uncommitted', action='store_true', help='smoke runs only')
    args = parser.parse_args()
    started = time.time()

    def _abs(p):
        return p if os.path.isabs(p) else os.path.join(REPO, p)
    out, off_out, on_x0_out, on_unit_out = (_abs(args.out), _abs(args.off_out), _abs(args.on_x0_out),
                                            _abs(args.on_unit_out))
    failures = []
    if os.path.exists(out) and [p for p in os.listdir(out) if p != 'launch.log']:
        failures.append(f'output directory exists and is not empty (write-once): {out}')
    if not args.skip_digest:
        for d in (off_out, on_x0_out, on_unit_out):
            if os.path.exists(d):
                failures.append(f'digest output directory exists: {d}')
    if os.path.exists(args.scratch):
        failures.append(f'scratch dir exists: {args.scratch}')
    if os.path.abspath(args.scratch).startswith(REPO + os.sep):
        failures.append(f'scratch dir must be outside the repo: {args.scratch}')
    if os.path.exists(H.CAMPAIGN_LOCK_PATH):
        failures.append(f'campaign lock present: {H.CAMPAIGN_LOCK_PATH}')
    procs = {}
    for line in subprocess.run(['ps', '-axo', 'pid=,ppid=,command='], capture_output=True,
                               text=True).stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3:
            procs[int(parts[0])] = (int(parts[1]), parts[2])
    ancestors, pid = set(), os.getpid()
    while pid in procs and pid not in ancestors and pid > 1:
        ancestors.add(pid)
        pid = procs[pid][0]
    others = [f'{p} {cmd}' for p, (_pp, cmd) in procs.items()
              if (SCRIPT_REL in cmd or NL_DIGEST_REL in cmd) and p not in ancestors]
    if others:
        failures.append(f'another copy is running (refusing to run concurrently): {others}')
    pinned = [SCRIPT_REL, HARNESS_REL, NL_DIGEST_REL, 'shared_resources_planning.py', 'network_parameters.py']
    status = _git_status(pinned)
    tracked = {f: bool(_git(['ls-files', f]).strip()) for f in pinned}
    if not args.allow_uncommitted and (status or not all(tracked.values())):
        failures.append(f'{pinned} not committed and clean: {status} {tracked}')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    os.makedirs(out, exist_ok=True)
    os.makedirs(args.scratch)
    run_tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    head = _git(['rev-parse', 'HEAD']).strip()
    hashes = {f: _sha_file(os.path.join(REPO, f)) for f in pinned + ['model_construction_helpers.py']}
    _log(f'{STAGE}; HEAD {head}; hashes {hashes}')
    result = {'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(), 'run_tag': run_tag, 'argv': sys.argv,
              'git_head_at_run': head, 'script': SCRIPT_REL, 'file_sha256_at_start': hashes,
              'git_status_at_start': status or 'clean', 'interpreter': sys.executable,
              'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
              'declared_constants': {'EXPECTED_ROWS_PER_TSO_BLOCK': EXPECTED_ROWS_PER_TSO_BLOCK,
                                     'EXPECTED_ROWS_PER_DSO_BLOCK': EXPECTED_ROWS_PER_DSO_BLOCK,
                                     'EXPECTED_N_TSO_BLOCKS': EXPECTED_N_TSO_BLOCKS,
                                     'EXPECTED_N_DSO_BLOCKS': EXPECTED_N_DSO_BLOCKS,
                                     'EXPECTED_TOTAL_PER_COPY': EXPECTED_TOTAL_PER_COPY,
                                     'NRF_BINDING_TOLERANCE_MW (harness)': H.NRF_BINDING_TOLERANCE_MW},
              'expert_prediction': 'the NRF-off NL identity holds (Addendum 83, W183 predictions)'}
    gates = {}
    try:
        if not args.skip_fixtures:
            gates['G1_fixtures'] = gate1_fixtures()
            _log(f"gate 1: PASS {gates['G1_fixtures']['pass']} {gates['G1_fixtures']['checks']} "
                 f"(loaded {gates['G1_fixtures']['n_loaded']} / {gates['G1_fixtures']['n_declared']})")
        import p515_g_g1_g4_admm_gates as G
        import shared_resources_planning as srp
        row = srp.INTERFACE_NO_REVERSE_FLOW_ROW
        ess_path = os.path.join(REPO, H.ESS_PARAMS_FILE_REL)
        declared = H.load_ess_ageing_parameters(ess_path)
        pin = {'path': H.ESS_PARAMS_FILE_REL, 'sha256': _sha_file(ess_path)}
        ctx = {'scratch': args.scratch, 'run_tag': run_tag, 'spec': spec_like(declared, pin)}
        result['spec_like'] = ctx['spec']
        result['row'] = row
        gates['G4_builds_hasattr'] = gate4_builds(G, srp, ctx)
        _log(f"gate 4: PASS {gates['G4_builds_hasattr']['pass']} {gates['G4_builds_hasattr']['checks']}")
        sup = gates['G4_builds_hasattr'].get('supplementary_capture_on_unsolved_models') or {}
        _log(f"gate 4 supplementary (not gated): capture "
             f"{(sup.get('capture_nrf_terminal') or {}).get('status')}; row counts "
             f"{(sup.get('capture_nrf_terminal') or {}).get('row_counts_per_network')}")
        gates['G5_hook_inverted_t4'] = gate5_hook(G, ctx)
        _log(f"gate 5: PASS {gates['G5_hook_inverted_t4']['pass']} {gates['G5_hook_inverted_t4']['checks']}")
        if not args.skip_digest:
            ref = json.load(gzip.open(os.path.join(REPO, REFERENCE_DIR_REL, 'w183_nl_digest.json.gz')))
            result['reference'] = {
                'gz': os.path.join(REFERENCE_DIR_REL, 'w183_nl_digest.json.gz'),
                'gz_sha256': _sha_file(os.path.join(REPO, REFERENCE_DIR_REL, 'w183_nl_digest.json.gz')),
                'gz_git_status': _git_status([os.path.join(REFERENCE_DIR_REL, 'w183_nl_digest.json.gz')]) or 'clean',
                'git_head_at_run': ref.get('git_head_at_run')}
            off_info, off = run_digest(args, off_out, 'post_w183c_off', ALL_CONFIGS, nrf=False, determinism=True)
            result['digest_off'] = off_info
            if off is None:
                raise RuntimeError(f'off digest run produced no JSON: {off_info}')
            gates['G2_off_identity'] = gate2_off_identity(ref, off, off_info)
            _log(f"gate 2: PASS {gates['G2_off_identity']['pass']} {gates['G2_off_identity']['checks']} "
                 f"({gates['G2_off_identity']['n_blocks_identical']} / {gates['G2_off_identity']['n_blocks_compared']})")
            del ref
            gc.collect()
            g3 = {'off_build_has_no_nrf_component': off_absence(off, row)}
            for tag, d in (('on_x0', on_x0_out), ('on_unit', on_unit_out)):
                config = ON_CONFIGS[tag]
                info, on = run_digest(args, d, f'post_w183c_{tag}', [config], nrf=True, determinism=False)
                result[f'digest_{tag}'] = info
                if on is None:
                    g3[tag] = {'pass': False, 'error': f'no JSON: {info}'}
                    continue
                g3[tag] = gate3_on(off, on, config, info, row)
                _log(f"gate 3 {tag}: PASS {g3[tag]['pass']} {g3[tag]['checks']} totals {g3[tag]['totals']}")
                del on
                gc.collect()
            g3['pass'] = all(v.get('pass') is True for k, v in g3.items() if k != 'pass')
            gates['G3_on_rows'] = g3
            _log(f"gate 3: PASS {g3['pass']}; off absence {g3['off_build_has_no_nrf_component']['pass']}")
    except Exception as error:  # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, file=sys.stderr, flush=True)
        result['error'] = {'error': f'{type(error).__name__}: {error}', 'traceback': tb}
    guard_failures = GUARD.verify(0)
    result['gates'] = gates
    result['solve_profile_guard'] = {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures}
    result['pickle'] = {'load_calls': PICKLE['load'], 'loads_calls': PICKLE['loads'],
                        'refused_load_calls': PICKLE['refused_load'],
                        'declared_in_gate1': (gates.get('G1_fixtures') or {}).get('n_declared'),
                        'load_count_equals_declared': PICKLE['load'] == (gates.get('G1_fixtures') or {}).get(
                            'n_declared', 0)}
    expected = {'G4_builds_hasattr', 'G5_hook_inverted_t4'}
    if not args.skip_fixtures:
        expected.add('G1_fixtures')
    if not args.skip_digest:
        expected |= {'G2_off_identity', 'G3_on_rows'}
    result['summary'] = {name: g.get('pass') for name, g in gates.items()}
    result['all_pass'] = (set(gates) == expected and all(g.get('pass') is True for g in gates.values())
                          and not guard_failures and result['pickle']['load_count_equals_declared']
                          and PICKLE['loads'] == 0 and PICKLE['refused_load'] == 0 and 'error' not in result)
    result['file_sha256_at_end'] = {f: _sha_file(os.path.join(REPO, f)) for f in hashes}
    result['files_unchanged_during_run'] = result['file_sha256_at_end'] == hashes
    result['all_pass'] = result['all_pass'] and result['files_unchanged_during_run']
    result['wall_clock_s'] = time.time() - started
    json_path = os.path.join(out, 'w183c_nrf_gates.json')
    with open(json_path, 'w') as handle:
        GRIO.dump(result, handle, indent=1, sort_keys=True)
    manifest = {os.path.relpath(json_path, REPO): _sha_file(json_path)}
    manifest.update({f: h for f, h in hashes.items()})
    for tag in ('digest_off', 'digest_on_x0', 'digest_on_unit'):
        info = result.get(tag) or {}
        manifest.update(info.get('manifest_post_run') or {})
    for d in (off_out, on_x0_out, on_unit_out):
        mp = os.path.join(d, 'manifest_post_run_sha256.json')
        if os.path.isfile(mp):
            manifest[os.path.relpath(mp, REPO)] = _sha_file(mp)
    with open(os.path.join(out, 'manifest_sha256.json'), 'w') as handle:
        GRIO.dump(dict(sorted(manifest.items())), handle, indent=1, sort_keys=True)
    GUARD.uninstall()
    _log(f"ALL_PASS={result['all_pass']} summary={result['summary']} guard={dict(GUARD.counts)} "
         f"pickle={result['pickle']} wall={result['wall_clock_s']:.1f}s")
    sys.exit(0 if result['all_pass'] else 1)


if __name__ == '__main__':
    main()
