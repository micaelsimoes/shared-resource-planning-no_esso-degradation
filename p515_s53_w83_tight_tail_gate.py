"""
P5.15 Addendum 46 ruling 7, Planner task W83 -- the SRP1 TWO-CYCLE BITWISE GATE for the convergence-depth TIGHT
TAIL: an arm with the tail ENABLED must be bit-identical to the committed C* reference and to the committed r2 gate
arm, because at cap 2 the AA-off predicate never holds, so the tail must never act ("everything before the tail
bitwise unchanged").

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 46 (ruling 7); Planner task W83 ("an SRP1 two-cycle bitwise gate,
run attached and alone, both streams captured, with a declared solve profile verified EXACTLY by GUARD.verify ...
bit-identical to the committed C* reference -- 0 diffs, exactly as the r2 gate achieved (153/153 solves). Use a
distinct arm name ... and a fresh launch-log name"); frozen spec v28 (found by glob, hash-checked, below).

HOW IT IS BUILT. The committed r2 gate `p515_s53_srp1_bitwise_gate.py` (sha256 b63b1fb9..., run at 2167c0d9,
committed d5a00bd9) BY IMPORT -- which imports W48 -> W39 -> W35 -> W32 -> W10, whose armed `SolveProfileGuard`
(W10.GUARD, bounded to `p514_n_instrumented_cstar.PERMITTED`) is installed at import and verified EXACTLY against
the per-event reconciled count by the W35 layer. Everything is the committed gates', with exactly these declared
substitutions and no others:
  1. identifiers STAGE / SCHEMA / ARM ('s53w83tailgate') / OUT_REL (write-once, P515S53/tight_tail_w83/
     srp1_bitwise_gate) -- the arm-named eval dir under P56A/evals is new;
  2. EXTRA_CLEAN_FILES += admm_anderson_acceleration.py, the W83 checks script, this file, the frozen spec v28 and
     the committed W83 checks artifact;
  3. the whole-arm reference W48_ARM -> the COMMITTED r2 gate arm (P515S53/srp1_bitwise_gate/arm, same configuration,
     run at 2167c0d9 without the W83 code), manifest sha256 pinned below;
  4. the pre-run presence assertion = the r2 gate's combined presence (row 18 + W47 + W51) + `w83_code_presence`;
  5. at the run-lock acquisition (AFTER every precondition and presence check, so they read the live functions --
     the r2 CAVEAT lesson), `W10.run_arm` is replaced ONE-SHOT by `_run_arm_with_tail`, which for the duration of the
     arm only: wraps the hook factory so the child hook, AFTER the committed hook, sets
     `planning.params.admm.convergence_depth_tail = {'enabled': True, 'compl_inf_tol': 1e-6}` and reads it back
     (recorded under rule_eleven_checklist, which the comparators classify as provenance); installs pass-through
     counters on `srp._apply_convergence_depth_tail` and `srp._convergence_depth_tail_next_state`; captures the
     returned state; and removes all of it when the arm returns.

DECLARED BEFORE THE RUN (from the case file and W10.CAP = 2): 153 solves base (51 x 3), reconciled per event;
tail apply calls EXACTLY cap + 1 = 3 (cycles 1, 2 and the exit restore), every one active False and acted False;
next-state calls EXACTLY cap = 2, both returning False; network floor records EXACTLY observed solves - 9 ESSO
solves (= 144 + attempted network retries), every one parsed, with compl_inf_tol in force = production.

GATE = the r2 gate verdict (W35 + W39 + W51 items and 0 genuine diffs vs the reference arm) AND every W83 item.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w83_tight_tail_gate.py \\
        > data/SRP1/Results/P515S53/tight_tail_w83/srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/tight_tail_w83/srp1_bitwise_gate/{gate.json, gate.md,
manifest_sha256.json, row18_gate_addendum.json, w51_gate_addendum.json, w51_manifest_sha256.json, arm/,
w83_tail_state.json, w83_network_ipopt_solve_records.jsonl, w83_gate_addendum.json, w83_manifest_sha256.json}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import copy
import glob
import hashlib
import inspect
import json
import os
import sys
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s53_srp1_bitwise_gate as S53G  # noqa: E402 -- imports W48 -> W39 -> W35 -> W32 -> W10 (armed guard)

S52G = S53G.S52G
S51G = S53G.S51G
W35G = S53G.W35G
W10 = S53G.W10
GUARD = W10.GUARD
CP = W10.CP
H = W10.H

import network as NET  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from admm_parameters import ADMMParameters  # noqa: E402

STAGE = ('P5.15 Addendum 46 ruling 7 W83 -- SRP1 two-cycle bitwise identity with the convergence-depth tight tail '
         'ENABLED (inert while the AA-off predicate has not held), vs committed baseline C* and the committed r2 arm')
SCHEMA = 'p515_s53_w83_tight_tail_gate_v1'
ARM = 's53w83tailgate'
_S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_REL = os.path.join(_S53, 'tight_tail_w83', 'srp1_bitwise_gate')
CHECKS_JSON = os.path.join(_S53, 'tight_tail_w83', 'checks', 'checks_w83.json')
CHECKS_SCRIPT = 'p515_s53_w83_tight_tail_checks.py'
DECLARED_TAIL = {'enabled': True, 'compl_inf_tol': 1e-6}
PRODUCTION_FILES = ('network.py', 'shared_resources_planning.py', 'admm_parameters.py',
                    'admm_anderson_acceleration.py', 'solver_parameters.py')
ESSO_SOLVES_PER_ROUND = 3

# The committed r2 gate arm: same configuration (C*, baseline, cap 2, snapshots on), pre-W83 code.
R2_ARM = {
    'gate_dir': os.path.join(_S53, 'srp1_bitwise_gate'),
    'arm_dir': os.path.join(_S53, 'srp1_bitwise_gate', 'arm'),
    'manifest': os.path.join(_S53, 'srp1_bitwise_gate', 'manifest_sha256.json'),
    'manifest_sha256': '8dfda779b3546c66a7818cfd38b74f284a92d219eb3e29e440dddfd45cd4f7c3',
    'gate_json': os.path.join(_S53, 'srp1_bitwise_gate', 'gate.json'),
    'arm_name': 's53w51gate',
    'commit': 'd5a00bd9 (run at 2167c0d9)',
}
R2_GATE_SCRIPT_SHA256 = 'b63b1fb94ec305a782cb139d6e9c42be8b7a49b2ef78cb0d174f09e3d53f6766'


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W83-gate] {msg}', flush=True)


def _sha(rel):
    return CP._sha256_file(os.path.join(REPO, rel))


def _find_spec():
    paths = sorted(glob.glob(os.path.join(REPO, _S53, 'frozen_s53_spec_v28_*.json')))
    if len(paths) != 1:
        raise SystemExit(f'expected exactly one frozen spec v28, found {len(paths)}')
    rel = W10._rel(paths[0])
    sha = _sha(rel)
    if not os.path.basename(rel).endswith(f'_{sha[:8]}.json'):
        raise SystemExit(f'{rel}: sha256 {sha} does not match its name')
    return rel, sha


def w83_code_presence():
    """The W83 production code must be present in the LIVE modules (read before the counters are installed and
    after they are removed)."""
    run_src = inspect.getsource(srp._run_operational_planning)
    attempt_src = inspect.getsource(NET._run_smopf_solver_attempt)
    return {
        'tail_functions_present': all(callable(getattr(srp, n, None)) for n in (
            'convergence_depth_tail_enabled', '_capture_convergence_depth_tail_baseline',
            '_apply_convergence_depth_tail', '_convergence_depth_tail_next_state', '_drain_network_ipopt_solve_records')),
        'tail_reads_the_aa_off_predicate': (
            'convergence_depth_tail_active_next = _convergence_depth_tail_next_state(\n'
            '                cycle_convergence, aa_enabled, aa_record)' in run_src
            and 'cycle_convergence = boyd_all_pass and local_solves_ok' in run_src),
        'tail_applied_at_the_top_of_each_cycle': (
            '_apply_convergence_depth_tail(\n                planning_problem, admm_parameters, '
            'convergence_depth_tail_active_next,' in run_src),
        'tail_restored_at_loop_exit': "convergence_depth_tail_state['restore_at_exit'] = _apply_convergence_depth_tail(" in run_src,
        'tail_starts_off': 'convergence_depth_tail_active_next = False' in run_src,
        'records_drained_each_cycle': (
            'network_ipopt_solve_records.extend(_drain_network_ipopt_solve_records(planning_problem, iter))' in run_src),
        'capture_after_the_solve_in_the_guarded_site': (
            attempt_src.find('result = solver.solve(model') >= 0
            and attempt_src.find('result = solver.solve(model') < attempt_src.find('_append_ipopt_solve_record(')),
        'tail_default_off': ADMMParameters().convergence_depth_tail == {'enabled': False, 'compl_inf_tol': 1e-6},
        'aa_off_constant': srp.CONVERGENCE_DEPTH_TAIL_AA_OFF_ACTION == 'off (all channels within Boyd tolerance)',
    }


R2_COMBINED_PRESENCE = S53G.combined_code_presence   # the committed r2 checks, captured before substitution


def combined_code_presence():
    prior = R2_COMBINED_PRESENCE()
    return {**prior, **{f'w83:{k}': v for k, v in w83_code_presence().items()}}


# ======================================================================================================================
#  substitution 5 -- the one-shot arm wrapper (installed at the run-lock acquisition, removed when the arm returns)
# ======================================================================================================================
TAIL = {'hook_calls': 0, 'apply_calls': [], 'next_calls': [], 'state': None, 'run_arm_wrapped_calls': 0}
TRUE_ACQUIRE = W10.G._acquire_exclusive_run_lock
ORIG_RUN_ARM = W10.run_arm


def _holders_snapshot(planning):
    snap = {}
    for label, nd in srp._convergence_depth_tail_holders(planning):
        opts = nd.params.solver_params.options
        snap[label] = {'id': id(opts), 'items': copy.deepcopy(list(opts.items())) if opts is not None else None}
    return snap


def _esso_snapshot(sed):
    opts = sed.params.solver_params.options
    return {'id': id(opts), 'items': copy.deepcopy(list(opts.items())) if opts is not None else None,
            'recovery_items': copy.deepcopy(list((sed.params.solver_params.recovery_options or {}).items()))}


def _run_arm_with_tail(arm, snapshots, out_dir, counters, expected_block_counts_holder):
    W10.run_arm = ORIG_RUN_ARM                     # one-shot
    TAIL['run_arm_wrapped_calls'] += 1
    factory_now = H._config_hook_factory          # the W35 layer's factory (it injects the recert declaration)
    orig_apply = srp._apply_convergence_depth_tail
    orig_next = srp._convergence_depth_tail_next_state
    orig_run = srp._run_operational_planning

    def factory_enabling_tail(spec_like, holder, overrides=None, **kwargs):
        inner = factory_now(spec_like, holder, overrides=overrides, **kwargs)

        def hook(planning, sed, candidate, report):
            inner(planning=planning, sed=sed, candidate=candidate, report=report)   # the committed hook FIRST
            TAIL['hook_calls'] += 1
            admm = planning.params.admm
            before = copy.deepcopy(getattr(admm, 'convergence_depth_tail', None))
            admm.convergence_depth_tail = dict(DECLARED_TAIL)
            TAIL['planning'], TAIL['sed'] = planning, sed
            TAIL['holders_before'] = _holders_snapshot(planning)
            TAIL['esso_before'] = _esso_snapshot(sed)
            readback = {'before': before, 'declared': dict(DECLARED_TAIL),
                        'after': copy.deepcopy(admm.convergence_depth_tail),
                        'enabled_helper': srp.convergence_depth_tail_enabled(admm),
                        'persistent_workers_enabled': bool(admm.persistent_workers.get('enabled')),
                        'parallel_execution': bool(planning.parallel_execution)}
            readback['ok'] = (before == {'enabled': False, 'compl_inf_tol': 1e-6}
                              and readback['after'] == DECLARED_TAIL and readback['enabled_helper'] is True
                              and not readback['persistent_workers_enabled'] and not readback['parallel_execution'])
            TAIL['readback'] = readback
            report.setdefault('rule_eleven_checklist', {})['w83_convergence_depth_tail'] = readback
            if not readback['ok']:
                raise RuntimeError(f'W83: tail enable did not take effect as declared: {readback}')
        return hook

    def counting_apply(planning_problem, admm_parameters, active, baseline, cycle):
        record = orig_apply(planning_problem, admm_parameters, active, baseline, cycle)
        TAIL['apply_calls'].append({'cycle': cycle, 'active': active, 'acted': record['acted'],
                                    'stack_tail': ''.join(traceback.format_stack(limit=3)[:-1])})
        return record

    def counting_next(cycle_convergence, aa_enabled, aa_record):
        result = orig_next(cycle_convergence, aa_enabled, aa_record)
        TAIL['next_calls'].append({'cycle_convergence': cycle_convergence, 'aa_enabled': aa_enabled,
                                   'aa_action': (aa_record or {}).get('action'), 'returned': result})
        return result

    def capturing_run(*args, **kwargs):
        result = orig_run(*args, **kwargs)
        TAIL['state'] = result[-1]
        return result

    H._config_hook_factory = factory_enabling_tail
    srp._apply_convergence_depth_tail = counting_apply
    srp._convergence_depth_tail_next_state = counting_next
    srp._run_operational_planning = capturing_run
    try:
        return ORIG_RUN_ARM(arm, snapshots, out_dir, counters, expected_block_counts_holder)
    finally:
        srp._run_operational_planning = orig_run
        srp._convergence_depth_tail_next_state = orig_next
        srp._apply_convergence_depth_tail = orig_apply
        H._config_hook_factory = factory_now
        if TAIL.get('planning') is not None:
            TAIL['holders_after'] = _holders_snapshot(TAIL['planning'])
            TAIL['esso_after'] = _esso_snapshot(TAIL['sed'])


def _acquire_then_wrap_run_arm(*args, **kwargs):
    W10.G._acquire_exclusive_run_lock = TRUE_ACQUIRE    # one-shot
    result = TRUE_ACQUIRE(*args, **kwargs)
    W10.run_arm = _run_arm_with_tail
    return result


# ======================================================================================================================
#  post-run evaluation
# ======================================================================================================================
def evaluate(declared, observed_solves):
    cap = W10.CAP
    state = TAIL['state'] or {}
    tail_state = state.get('convergence_depth_tail') or {}
    records = state.get('network_ipopt_solve_records') or []
    per_cycle = tail_state.get('per_cycle') or []
    baseline = tail_state.get('baseline') or {}
    labels = sorted(baseline)
    expected_records = (observed_solves - ESSO_SOLVES_PER_ROUND * (cap + 1)) if observed_solves is not None else None
    prod_cit = {'TSO': 5e-4}
    rounds = {}
    for r in records:
        rounds[r.get('round')] = rounds.get(r.get('round'), 0) + 1
    bad_records = []
    for r in records:
        want_passed = prod_cit.get(r.get('agent'))
        want_in_force = want_passed if want_passed is not None else NET.IPOPT_DEFAULT_COMPL_INF_TOL
        if not (r.get('parse_reason') is None and r.get('options_list_agrees') is True
                and r.get('compl_inf_tol_passed') == want_passed and r.get('compl_inf_tol_logged') == want_passed
                and r.get('compl_inf_tol_in_force') == want_in_force):
            bad_records.append({k: r.get(k) for k in ('agent', 'network', 'year', 'day', 'round', 'attempt',
                                                      'compl_inf_tol_passed', 'compl_inf_tol_logged',
                                                      'compl_inf_tol_in_force', 'options_list_agrees', 'parse_reason')})
    floor_tally = {}
    for r in records:
        key = f"{'TSO' if r.get('agent') == 'TSO' else 'DSO'}|round{r.get('round')}|{r.get('floor_status')}"
        floor_tally[key] = floor_tally.get(key, 0) + 1
    exit_tally = {}
    for r in records:
        exit_tally[str(r.get('exit'))] = exit_tally.get(str(r.get('exit')), 0) + 1
    stdout_path = os.path.join(REPO, OUT_REL, 'arm', f'stdout_{W10.ARM_LABEL}.log')
    stdout_text = open(stdout_path, errors='replace').read() if os.path.isfile(stdout_path) else None
    items = {
        'tail_enabled_and_read_back_in_the_arm': TAIL['hook_calls'] == 1 and bool((TAIL.get('readback') or {}).get('ok')),
        'run_arm_wrapped_exactly_once': TAIL['run_arm_wrapped_calls'] == 1,
        'apply_calls_exactly_cap_plus_1_all_inactive_none_acted': (
            len(TAIL['apply_calls']) == cap + 1
            and [c['cycle'] for c in TAIL['apply_calls']] == list(range(1, cap + 1)) + [None]
            and all(c['active'] is False and c['acted'] is False for c in TAIL['apply_calls'])),
        'next_state_calls_exactly_cap_all_false': (
            len(TAIL['next_calls']) == cap
            and all(c['returned'] is False and c['cycle_convergence'] is False for c in TAIL['next_calls'])),
        'state_tail_record_as_declared': (
            tail_state.get('enabled') is True and tail_state.get('compl_inf_tol_tail') == 1e-6
            and len(per_cycle) == cap
            and all(p.get('active') is False and p.get('acted') is False
                    and p.get('aa_off_predicate_end_of_cycle') is False for p in per_cycle)
            and (tail_state.get('restore_at_exit') or {}).get('acted') is False),
        'baseline_is_production': (
            labels == ['DSO5', 'DSO7', 'DSO9', 'TSO']
            and baseline['TSO']['has_key'] and baseline['TSO']['value'] == 5e-4
            and all(not baseline[k]['has_key'] for k in labels if k != 'TSO')),
        'holders_identical_before_and_after_the_arm': (
            TAIL.get('holders_before') is not None and TAIL.get('holders_before') == TAIL.get('holders_after')),
        'esso_options_unchanged': (TAIL.get('esso_before') is not None
                                   and TAIL.get('esso_before') == TAIL.get('esso_after')),
        'floor_capture_complete': (
            expected_records is not None and len(records) == expected_records
            and all(rounds.get(k, 0) >= declared['network_solves_per_cycle'] for k in range(cap + 1))
            and sorted(rounds) == list(range(cap + 1))),
        'floor_capture_every_record_parsed_at_production_compl_inf_tol': not bad_records and bool(records),
        'no_tail_on_line_in_the_arm_stdout': stdout_text is not None and 'Convergence-depth tail ON' not in stdout_text,
    }
    return {
        'items': items,
        'declared_before_the_run': {
            'apply_calls': cap + 1, 'next_state_calls': cap,
            'network_records': 'observed solves - 9 ESSO solves (= 144 + attempted network retries)'},
        'observed': {
            'apply_calls': TAIL['apply_calls'], 'next_calls': TAIL['next_calls'],
            'hook_calls': TAIL['hook_calls'], 'readback': TAIL.get('readback'),
            'n_records': len(records), 'expected_records': expected_records, 'records_per_round': rounds,
            'floor_status_tally_REPORTED_not_gated': floor_tally, 'exit_tally': exit_tally,
            'bad_records_first': bad_records[:20], 'n_bad_records': len(bad_records),
            'holders_before': TAIL.get('holders_before'), 'holders_after': TAIL.get('holders_after'),
            'esso_before': TAIL.get('esso_before'), 'esso_after': TAIL.get('esso_after'),
            'stale_records_discarded': state.get('stale_network_ipopt_solve_records_discarded')},
        'tail_state': tail_state,
        'records': records,
    }


def main():
    out_root = os.path.join(REPO, OUT_REL)
    _log(STAGE)
    _log(f'git HEAD {W10._git(["rev-parse", "HEAD"])}; output root {OUT_REL}')

    # ---- W83 preconditions (before anything the committed gates do) ----
    problems = []
    try:
        spec_rel, spec_sha = _find_spec()
        spec = json.load(open(os.path.join(REPO, spec_rel)))
    except SystemExit as error:
        _log(f'[PRECONDITION FAILED] {error}')
        return 1
    if spec.get('predecessor', {}).get('sha256') != '654046ecc97c485616abffcbd258d4a7c85b7d18531dc0cbd83f424f4e3e846b':
        problems.append('spec v28 predecessor is not v27 654046ec')
    if (spec.get('harness_sha256_at_freeze') or {}).get(os.path.basename(__file__)) != _sha(os.path.basename(__file__)):
        problems.append('this gate script is not the one frozen in spec v28')
    if spec.get('production_sha256_at_freeze') != {f: _sha(f) for f in PRODUCTION_FILES}:
        problems.append('production files differ from those frozen in spec v28')
    if _sha('p515_s53_srp1_bitwise_gate.py') != R2_GATE_SCRIPT_SHA256:
        problems.append('the imported r2 gate script is not the one r2 ran')
    checks = json.load(open(os.path.join(REPO, CHECKS_JSON))) if os.path.isfile(os.path.join(REPO, CHECKS_JSON)) else {}
    if not checks.get('all_hold'):
        problems.append(f'W83 zero-solve checks absent or not all holding: {CHECKS_JSON}')
    elif checks.get('production_sha256') != {f: _sha(f) for f in PRODUCTION_FILES}:
        problems.append('W83 checks ran against different production files')
    eval_dir = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals',
                            f'p515s44_s45_w10_snapoff_{ARM}_{W10.C_STAR_KEY_PIN[:16]}_run')
    if os.path.exists(eval_dir):
        problems.append(f'arm-named eval dir already exists: {W10._rel(eval_dir)}')
    presence = w83_code_presence()
    problems += [f'W83 code not present in the live module: {k}' for k, v in presence.items() if not v]
    if problems:
        for p in problems:
            _log(f'[PRECONDITION FAILED] {p}')
        return 1
    _log(f'spec v28 {spec_rel} (sha256 {spec_sha}); checks {CHECKS_JSON} all_hold; W83 presence '
         f'{sum(presence.values())}/{len(presence)}; arm eval dir fresh')

    declared = W10.derive_solves_from_case_file()
    _log(f"DECLARED BEFORE THE RUN: {declared['declared_base_solves_per_arm']} solves base (per-event reconciled); "
         f'tail apply calls exactly {W10.CAP + 1}, all inactive and not acting; next-state calls exactly {W10.CAP}, '
         'all False; network floor records exactly observed - 9 ESSO solves')

    # ---- the declared substitutions, and no others ----
    S53G.STAGE = STAGE
    S53G.SCHEMA = SCHEMA
    S53G.ARM = ARM
    S53G.OUT_REL = OUT_REL
    S53G.EXTRA_CLEAN_FILES = tuple(S53G.EXTRA_CLEAN_FILES) + (
        'admm_anderson_acceleration.py', CHECKS_SCRIPT, os.path.basename(__file__), spec_rel, CHECKS_JSON)
    S53G.W48_ARM = R2_ARM
    S53G.combined_code_presence = combined_code_presence
    W10.G._acquire_exclusive_run_lock = _acquire_then_wrap_run_arm   # substitution 5 is installed AT the lock
    try:
        status = S53G.main()
    finally:
        W10.G._acquire_exclusive_run_lock = TRUE_ACQUIRE
        W10.run_arm = ORIG_RUN_ARM
    _log(f'r2 gate layers verdict (exit status): {status}')

    gate_json = os.path.join(out_root, 'gate.json')
    observed = None
    if os.path.isfile(gate_json):
        observed = json.load(open(gate_json))['solve_profile']['observed_in_arm']
    ev = evaluate(declared, observed)
    presence_after = w83_code_presence()
    ev['items']['w83_code_present_before_and_after_the_arm'] = all(presence.values()) and all(presence_after.values())
    ev['items']['r2_gate_layers_pass'] = status == 0
    ev['items']['lock_hook_and_run_arm_restored'] = (W10.G._acquire_exclusive_run_lock is TRUE_ACQUIRE
                                                     and W10.run_arm is ORIG_RUN_ARM)
    gate_pass = all(ev['items'].values())

    if not os.path.isdir(out_root):
        _log('output root absent (the committed layers refused before the arm); nothing more to write')
        for k, v in ev['items'].items():
            _log(f'   {k}: {v}')
        _log(f'GATE_PASS={gate_pass}')
        return 1

    state_path = os.path.join(out_root, 'w83_tail_state.json')
    records_path = os.path.join(out_root, 'w83_network_ipopt_solve_records.jsonl')
    addendum_path = os.path.join(out_root, 'w83_gate_addendum.json')
    manifest_path = os.path.join(out_root, 'w83_manifest_sha256.json')
    for p in (state_path, records_path, addendum_path, manifest_path):
        W10._refuse_overwrite(p)
    with open(state_path, 'w') as handle:
        json.dump(ev['tail_state'], handle, indent=1, default=str)
    with open(records_path, 'w') as handle:
        for r in ev['records']:
            handle.write(json.dumps(r, default=str) + '\n')
    payload = {
        'schema': SCHEMA + '_addendum', 'stage': STAGE,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7', 'Planner task W83', spec_rel],
        'spec': {'path': spec_rel, 'sha256': spec_sha},
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': _sha(os.path.basename(__file__)),
        'imported_r2_gate_sha256': _sha('p515_s53_srp1_bitwise_gate.py'),
        'production_sha256': {f: _sha(f) for f in PRODUCTION_FILES},
        'instance': {'candidate': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025',
                     'candidate_key': W10.C_STAR_KEY_PIN},
        'objective_convention': 'gross_operational_cost (settlement-excluded gross cost), as in gate.json',
        'declared_substitutions': {
            '1_identifiers': {'STAGE': STAGE, 'SCHEMA': SCHEMA, 'ARM': ARM, 'OUT_REL': OUT_REL},
            '2_extra_clean_files': list(S53G.EXTRA_CLEAN_FILES),
            '3_reference_arm': R2_ARM,
            '4_presence': 'r2 combined presence (row 18 + W47 + W51) + w83_code_presence',
            '5_arm_wrapper': ('one-shot W10.run_arm wrapper installed at the run-lock acquisition: tail enabled after '
                              'the committed hook and read back; pass-through counters on the two tail functions; '
                              'returned-state capture; all removed when the arm returns')},
        'w83_code_presence_before_run': presence, 'w83_code_presence_after_run': presence_after,
        'w83_evaluation': {k: v for k, v in ev.items() if k not in ('records', 'tail_state')},
        'sidecars': {'tail_state': W10._rel(state_path), 'network_ipopt_solve_records': W10._rel(records_path)},
        'guard_counts_at_end': dict(GUARD.counts),
        'gate_items': ev['items'], 'gate_pass': gate_pass,
    }
    with open(addendum_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f"floor records {ev['observed']['n_records']} (expected {ev['observed']['expected_records']}); per round "
         f"{ev['observed']['records_per_round']}; floor tally (reported) {ev['observed']['floor_status_tally_REPORTED_not_gated']}")
    for key, value in ev['items'].items():
        _log(f'   {key}: {value}')
    _log(f'wrote {W10._rel(addendum_path)}, {W10._rel(manifest_path)} ({len(manifest)} files)')
    _log(f'GATE_PASS={gate_pass}')
    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())
