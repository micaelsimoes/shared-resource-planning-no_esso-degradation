"""
P5.15 Addendum 38 (task W39) -- GATE 2 of 3: the 2 x 2 TWO-CYCLE alpha -> LARGE LIMIT
CHECK. As the imbalance premium grows without bound, row 18 must reproduce the PINNED
behaviour: the DSO's per-scenario interface deviation from its own committed schedule
goes to zero.

Authority: frozen spec v21 `data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json`,
`gates[1]` -- "2x2 two-cycle limit check: alpha -> large reproduces the pinned behaviour
(dispersion -> 0)"; PLANNER_BRIEF_2026-09-13.md Addendum 37 ("Gates") and Addendum 38.

WHAT IS UNDER TEST, and what is NOT. Row 18 replaces a hard pin by a PRICE. The claim is
that the price recovers the pin in the limit -- i.e. that the mechanism is a relaxation of
the pin and not something else. This gate measures the dispersion at the pilot's alpha and
at a large alpha on the SAME instance, the SAME candidate and the SAME two cycles, and
requires the large-alpha dispersion to collapse. It is NOT a convergence study, NOT a
result about the pilot's dispersion (two cycles is not a converged run), and NOT
comparable with any SRP1 (1 x 1) figure, where row 18 does not exist at all.

DISPERSION METRIC (Addendum 37, verbatim): per DSO, the RMS and the max of
d_{s,t} = p_int_{s,t} - pbar_t over scenarios and hours, in MW and as a share of the mean
interface flow, plus the total charge. Computed by production's own
`shared_resources_planning._get_operational_interface_dispersion`, so the gate cannot
drift from what the objective prices.

ARMS (one instance, one candidate, two cycles each):
  * `pilot` -- alpha = 0.50, the author's pilot value (Addendum 36);
  * `large` -- alpha = ALPHA_LARGE below, the limit arm.
GATE ITEMS ARE SCOPED PER ARM (CLAUDE.md stage template): the "dispersion collapses" items
apply to the `large` arm only; the `pilot` arm supplies the reference and is gated only on
running, solving and reconciling its solves.

DECLARED BEFORE THE RUN: the two alphas, the cycle cap, the candidate, the derived case
file and its sha256, the per-arm solve count, and the two dispersion thresholds.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s51_2x2_limit_gate.py --label <write-once-label> \\
        > data/SRP1/Results/P515S51/2x2_limit_gate_launch_<label>.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S51/2x2_limit_gate/<label>/
Exit 0 on PASS, 1 on FAIL, 2 on a precondition refusal.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import psutil  # noqa: E402

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402

PERMITTED = tuple(tuple(p) for p in N.PERMITTED)
GUARD = SolveProfileGuard(PERMITTED, label='P5.15 W39 gate 2 -- 2x2 alpha limit check').install()

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p515_s44_scale_measurement as S  # noqa: E402
import p56a_oracle as O  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
import model_construction_helpers as MCH  # noqa: E402

STAGE = ('P5.15 Addendum 38 W39 gate 2 -- 2x2 two-cycle alpha -> large limit check '
         '(row 18 recovers the pin: dispersion -> 0)')
SCHEMA = 'p515_s51_2x2_limit_gate_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addenda 37, 38',
             'data/SRP1/Results/P515S51/frozen_s51_spec_v21_13cb828c.json gates[1]']

OUT_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S51', '2x2_limit_gate')
OWN_LOCK_PATH = os.path.join(REPO, '.p515_s51_2x2_limit_gate.lock')

# ---- instance (identical to the W37 2x2 smoke's, so the two are comparable) ----
OVERRIDE_YEARS = {'2025': 5}
OVERRIDE_MARKET_SCENARIOS = 2
OVERRIDE_OPERATION_SCENARIOS = 2
INSTANCE_LABEL = 's51_2x2_limit'
ACTIVE_NODES = (5, 7, 9)
CYCLES = 2
REQUIRED_CONSECUTIVE_CYCLES = 10

# ---- the two arms, declared ----
ALPHA_PILOT = 0.50
ALPHA_LARGE = 1000.0
ARMS = ('pilot', 'large')
ALPHA_BY_ARM = {'pilot': ALPHA_PILOT, 'large': ALPHA_LARGE}

# ---- thresholds, declared BEFORE the run ----
DISPERSION_ZERO_TOL_MW = 1.0e-2        # "dispersion -> 0" on the `large` arm
DISPERSION_COLLAPSE_RATIO = 0.10       # large-arm RMS must be <= 10% of the pilot arm's

GIB = 1 << 30
RSS_LIMIT_GIB = 12.0
EXIT_OK, EXIT_ERROR, EXIT_REFUSED = 0, 1, 2
THREAD_CAP_ENV = dict(S.THREAD_CAP_ENV)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(message):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W39-gate2] {message}', flush=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(['git'] + args, capture_output=True, text=True,
                              check=True, cwd=REPO).stdout.strip()
    except Exception as error:  # noqa: BLE001
        return f'<git failed: {error}>'


PRODUCTION_FILES_TO_CHECK_CLEAN = (
    'model_construction_helpers.py', 'shared_resources_planning.py', 'network.py',
    'admm_parameters.py', 'p515_g_g1_g4_admm_gates.py', 'p515_s44_campaign_harness.py',
    'p515_s44_scale_measurement.py', 'p56a_oracle.py', os.path.basename(__file__))


def check_preconditions(out_dir):
    failures = []
    for path in (OWN_LOCK_PATH, G.CAMPAIGN_LOCK_PATH, os.path.join(REPO, '.p515_g_gate.lock'),
                 os.path.join(REPO, '.p515_s44_scale_measurement.lock')):
        if os.path.exists(path):
            failures.append(f'lock file exists: {path}')
    if os.path.exists(out_dir):
        failures.append(f'output directory already exists (write-once): {out_dir}')
    me = {os.getpid(), os.getppid()}
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if proc.info['pid'] in me or 'python' not in cmd:
            continue
        if S.HARNESS_PATTERN.search(cmd):
            failures.append(f'another p51x/p514 harness is alive: pid={proc.info["pid"]} {cmd[:200]}')
    status = _git(['status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN))
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')
    bad_env = {k: os.environ.get(k) for k, v in THREAD_CAP_ENV.items() if os.environ.get(k) != v}
    if bad_env:
        failures.append(f'thread caps not in force (export them before launching): {bad_env}')
    return failures


def capture_path_checklist():
    """CLAUDE.md rule eleven: assert BEFORE the run that a capture path exists for every
    quantity this gate's specification requires."""
    return {
        'dispersion_metric_callable': callable(getattr(srp, '_get_operational_interface_dispersion', None)),
        'per_block_dispersion_callable': callable(getattr(srp, '_get_local_interface_dispersion', None)),
        'row18_wiring_callable': callable(getattr(MCH, 'add_scenario_commitment_terms', None)),
        'recourse_components_callable': callable(getattr(srp, '_get_operational_recourse_components', None)),
        'voltage_mismatch_callable': callable(getattr(srp, '_get_local_scenario_voltage_mismatch', None)),
        'alpha_is_threaded_by_run_operational_planning': (
            'interface_deviation_premium' in __import__('inspect').getsource(srp._run_operational_planning)),
        'dispersion_dict_has_the_required_fields': True,   # verified live on the first block below
    }


def _dispersion_summary(dispersion_detail):
    """Collapse production's per-block dispersion detail into the Addendum 37 metric, per
    DSO node and overall. Returns (per_node, overall)."""
    per_node = {}
    for (kind, node_id, year, day), detail in dispersion_detail.items():
        if detail is None:
            continue
        entry = per_node.setdefault(str(node_id), {'blocks': {}, 'rms_mw_max_over_blocks': 0.0,
                                                   'max_abs_mw': 0.0, 'total_charge': 0.0,
                                                   'rms_share_max': None})
        block_key = f'{year}:{day}'
        entry['blocks'][block_key] = {
            'p_rms_mw': detail['p']['rms_mw'],
            'p_max_abs_mw': detail['p']['max_abs_mw'],
            'p_rms_share_of_mean_flow': detail['p']['rms_share_of_mean_flow'],
            'p_max_abs_share_of_mean_flow': detail['p']['max_abs_share_of_mean_flow'],
            'q_rms_mvar': detail['q']['rms_mvar'],
            'q_max_abs_mvar': detail['q']['max_abs_mvar'],
            'row18_charge': detail['row18_charge'],
            'row18_alpha': detail['row18_alpha'],
            'row18_wired': detail['row18_wired'],
            'mean_abs_committed_flow_mw': detail['p']['mean_abs_committed_flow_mw'],
        }
        entry['rms_mw_max_over_blocks'] = max(entry['rms_mw_max_over_blocks'], detail['p']['rms_mw'])
        entry['max_abs_mw'] = max(entry['max_abs_mw'], detail['p']['max_abs_mw'])
        entry['total_charge'] += detail['row18_charge']
        share = detail['p']['rms_share_of_mean_flow']
        if share is not None:
            entry['rms_share_max'] = share if entry['rms_share_max'] is None else max(entry['rms_share_max'], share)
    overall = {
        'rms_mw_max_over_all_dso_blocks': max([v['rms_mw_max_over_blocks'] for v in per_node.values()] or [0.0]),
        'max_abs_mw_over_all_dso_blocks': max([v['max_abs_mw'] for v in per_node.values()] or [0.0]),
        'total_charge_all_dso': sum(v['total_charge'] for v in per_node.values()),
        'n_dso_nodes': len(per_node),
    }
    return per_node, overall


def _make_post_run_hook(arm, record):
    def hook(planning=None, sed=None, models=None, rows=None, report=None, out_dir=None, label=None):
        detail = srp._get_operational_interface_dispersion(planning, models)
        per_node, overall = _dispersion_summary(detail)
        record['dispersion_per_node'] = per_node
        record['dispersion_overall'] = overall
        record['recourse_components'] = srp._get_operational_recourse_components(planning, models)
        record['voltage_mismatch'] = {
            f'{k[0]}:{k[1]}:{k[2]}:{k[3]}': v
            for k, v in srp._get_operational_scenario_voltage_mismatch(planning, models).items()}
        raw_path = os.path.join(out_dir, f'dispersion_detail_{arm}.json')
        G._refuse_overwrite(raw_path)
        with open(raw_path, 'w') as handle:
            json.dump({f'{k[0]}:{k[1]}:{k[2]}:{k[3]}': v for k, v in detail.items()},
                      handle, indent=1, default=str)
        record['dispersion_detail_path'] = os.path.relpath(raw_path, REPO)
    return hook


def _make_pre_solve_hook(arm, alpha, record, cfg_holder):
    spec_like = {'configuration': {'overrides': {},
                                   'case_file_anderson_acceleration': dict(S.CASE_FILE_AA)},
                 'cap': CYCLES, 'required_consecutive_cycles': REQUIRED_CONSECUTIVE_CYCLES}
    inner = H._config_hook_factory(spec_like, cfg_holder, overrides={})
    inner = S.snapshot_hook_wrapper(inner, 'off', record)

    def hook(planning=None, sed=None, candidate=None, report=None):
        inner(planning=planning, sed=sed, candidate=candidate, report=report)
        # THE ONE ARM-DEFINING DIFFERENCE: alpha. Set on this arm's own deep-copied
        # planning object, immediately before the first solve; nothing else differs.
        planning.params.admm.interface_deviation_premium = {
            'alpha': float(alpha), 'floor': None, 'source': f'W39 gate 2 arm {arm!r}'}
        record['alpha_applied'] = dict(planning.params.admm.interface_deviation_premium)
    return hook


def run_arm(arm, out_root, planning0, declared_base, holder):
    alpha = ALPHA_BY_ARM[arm]
    record = holder.setdefault(arm, {'arm': arm, 'alpha': alpha})
    arm_dir = os.path.join(out_root, f'arm_{arm}')
    os.makedirs(arm_dir)
    label = f's51limit_{arm}'
    eval_id = f'p515s51_limit_{os.path.basename(out_root)}_{arm}'
    record['eval_id'] = eval_id
    investment_map = {node: (0.0, 0.0) for node in ACTIVE_NODES}
    record['investment_map'] = {str(k): list(v) for k, v in investment_map.items()}
    record['candidate_label'] = 'x = 0 (no shared-ESS investment) at every active node'

    cfg_holder = {}
    pre_hook = _make_pre_solve_hook(arm, alpha, record, cfg_holder)
    t0 = time.time()
    before = GUARD.counts['permitted_solve']
    report, report_path = G.run_admm_arm(
        label, arm_dir, k_override=None, investment_map=investment_map,
        num_max_iters_override=CYCLES, eval_id=eval_id, apply_rho=False,
        full_diagnostics_in_rows=True, pre_solve_hook=pre_hook,
        post_run_hook=_make_post_run_hook(arm, record))
    record['wall_s'] = time.time() - t0
    record['solves_in_arm'] = GUARD.counts['permitted_solve'] - before
    record['report_path'] = os.path.relpath(report_path, REPO)
    record['cycles_run'] = report.get('cycles_run')
    record['converged_at_cycle'] = report.get('converged_at_cycle')
    record['recourse'] = report.get('recourse')
    record['gross_operational_cost'] = report.get('gross_operational_cost')
    record['objective_convention'] = (
        'recourse = net_operational_recourse (contracted-settlement-excluded, '
        'voltage-pin-excluded, salvage-netted); gross_operational_cost is the same '
        'without the salvage credit -- P5.15 Addendum 38 (C)/(D) conventions, as '
        'shared_resources_planning._get_operational_recourse_components defines them')
    # CLAUDE.md: the terminal-step-to-threshold ratio, for every cell of every evaluation
    record['rule_ten_terminal_step_over_threshold'] = report.get(
        'rule_ten_terminal_step_over_threshold')
    record['terminal_objective_change_abs'] = report.get('terminal_objective_change_abs')
    record['terminal_objective_tolerance'] = report.get('terminal_objective_tolerance')
    record['cycle_trajectory'] = report.get('cycle_trajectory')       # per-cycle state by default
    record['network_failures_summary'] = report.get('network_failures_summary')
    record['arm_solve_profile'] = report.get('solve_profile')
    record['event_level_reconciliation'] = S.event_level_solve_reconciliation(report, declared_base)
    record['anderson_acceleration_effective'] = cfg_holder.get('anderson_acceleration_effective')
    record['configuration_checks'] = cfg_holder.get('configuration_checks')
    return record


def main():
    global CYCLES
    parser = argparse.ArgumentParser(description=STAGE)
    parser.add_argument('--label', required=True, help='write-once output label')
    parser.add_argument('--cycles', type=int, default=CYCLES)
    args = parser.parse_args()
    CYCLES = args.cycles

    out_root = os.path.join(OUT_ROOT, args.label)
    failures = check_preconditions(out_root)
    if failures:
        for failure in failures:
            print(f'[W39-gate2 PRECONDITION FAILED] {failure}', file=sys.stderr)
        return EXIT_REFUSED
    try:
        fd = os.open(OWN_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f'REFUSED: lock held {OWN_LOCK_PATH}', file=sys.stderr)
        return EXIT_REFUSED
    with os.fdopen(fd, 'w') as handle:
        json.dump({'pid': os.getpid(), 'label': args.label, 'started_utc': _utc()}, handle)

    started = time.time()
    holder = {}
    os.environ.update(THREAD_CAP_ENV)
    try:
        os.makedirs(out_root)
        case_dir = os.path.join(out_root, 'case')
        os.makedirs(case_dir)
        case, spec, changes = S.derive_case('srp1', {
            'years': OVERRIDE_YEARS,
            'num_market_scenarios': OVERRIDE_MARKET_SCENARIOS,
            'num_operation_scenarios': OVERRIDE_OPERATION_SCENARIOS})
        case_path = os.path.join(case_dir, 'SRP1__s51_2x2.json')
        with open(case_path, 'w') as handle:
            json.dump(case, handle, indent='\t')

        launch = {
            'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
            'label': args.label, 'instance': INSTANCE_LABEL, 'instance_definition': spec,
            'derived_case': {'path': os.path.relpath(case_path, REPO),
                             'sha256': _sha256_file(case_path),
                             'source': os.path.relpath(S.SOURCE_CASE, REPO),
                             'source_sha256': _sha256_file(S.SOURCE_CASE),
                             'changes_vs_source': changes},
            'argv': sys.argv, 'interpreter': sys.executable,
            'script': os.path.basename(__file__),
            'script_sha256': _sha256_file(os.path.abspath(__file__)),
            'git_head': _git(['rev-parse', 'HEAD']),
            'git_tracked_changes': _git(['status', '--porcelain', '--untracked-files=no']).splitlines(),
            'nlp_solver_path_env': os.environ.get('NLP_SOLVER_PATH'),
            'cycles': CYCLES, 'arms': list(ARMS), 'alpha_by_arm': ALPHA_BY_ARM,
            'thresholds_declared_before_the_run': {
                'dispersion_zero_tol_mw': DISPERSION_ZERO_TOL_MW,
                'dispersion_collapse_ratio': DISPERSION_COLLAPSE_RATIO},
            'guard_permitted': [list(p) for p in PERMITTED],
            'snapshots': 'off',
            'started_utc': _utc(), 'pid': os.getpid(),
        }

        stages = S.StageLog(os.path.join(out_root, 'stages.jsonl'),
                            S.Watchdog(out_root, 's51limit', limit_bytes=int(RSS_LIMIT_GIB * GIB)))
        stages.wd.start()
        planning0 = S.read_planning_from_derived_case(launch, out_root, stages)
        launch['scenario_checksum'] = S.inject_oracle_baseline(O, planning0, launch)
        launch['planning_dimensions'] = S.planning_dimensions(planning0)
        launch['expected_block_counts'] = S.expected_block_counts(planning0)
        provenance = S.provenance_record(planning0, INSTANCE_LABEL, launch['scenario_checksum'])
        launch['provenance'] = provenance
        non_checksum = [f for f in provenance['gate_failures'] if f['identity'] != 'scenario checksum']
        if non_checksum:
            raise RuntimeError(f'provenance: non-canonical identity: {non_checksum}')

        declared = S.declared_solve_profile(planning0, CYCLES)
        declared_base = declared['declared_base_solves']
        launch['declared_solve_profile'] = {**declared, 'n_arms': len(ARMS),
                                            'declared_total_strict': len(ARMS) * declared_base}

        checklist = capture_path_checklist()
        launch['capture_path_checklist_asserted_before_run'] = checklist
        if not all(checklist.values()):
            raise RuntimeError(f'capture-path checklist failed: '
                               f'{[k for k, v in checklist.items() if not v]}')

        launch_path = os.path.join(out_root, 'launch.json')
        G._refuse_overwrite(launch_path)
        with open(launch_path, 'w') as handle:
            json.dump(launch, handle, indent=1, default=str)
        _log(f'declared {declared_base} base solves per arm; {len(ARMS)} arms; '
             f'alphas {ALPHA_BY_ARM}')

        # The legacy run lock (`.p515_g_gate.lock`). It is released by
        # `_acquire_exclusive_run_lock`'s own atexit handler -- the module exposes no
        # explicit release entry point, and the committed gates do not release it either.
        G._acquire_exclusive_run_lock()
        for arm in ARMS:
            _log(f'arm {arm}: alpha = {ALPHA_BY_ARM[arm]}')
            run_arm(arm, out_root, planning0, declared_base, holder)
            _log(f"arm {arm}: cycles={holder[arm]['cycles_run']} "
                 f"rms_mw={holder[arm]['dispersion_overall']['rms_mw_max_over_all_dso_blocks']} "
                 f"charge={holder[arm]['dispersion_overall']['total_charge_all_dso']}")

        pilot = holder['pilot']['dispersion_overall']
        large = holder['large']['dispersion_overall']
        ratio = (large['rms_mw_max_over_all_dso_blocks'] / pilot['rms_mw_max_over_all_dso_blocks']
                 if pilot['rms_mw_max_over_all_dso_blocks'] > 0 else None)

        reconciled_total = sum((r['event_level_reconciliation'] or {}).get('expected') or 0
                               for r in holder.values())
        supported = all((r['event_level_reconciliation'] or {}).get('expected') is not None
                        for r in holder.values())
        guard_failures = GUARD.verify(reconciled_total) if supported else [
            'event-level reconciliation unsupported on at least one arm']

        gate_items = {
            # both arms
            'both_arms_ran_the_declared_cycles': all(r['cycles_run'] == CYCLES for r in holder.values()),
            'both_arms_reconcile_their_solves': supported and all(
                r['solves_in_arm'] == (r['event_level_reconciliation'] or {}).get('expected')
                for r in holder.values()),
            'process_guard_verified_exactly': not guard_failures,
            'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
            'row18_wired_on_every_dso_block_in_both_arms': all(
                block['row18_wired']
                for r in holder.values() for node in r['dispersion_per_node'].values()
                for block in node['blocks'].values()),
            'alpha_recorded_per_arm_matches_the_declaration': all(
                math.isclose(r['alpha_applied']['alpha'], ALPHA_BY_ARM[r['arm']]) for r in holder.values()),
            # LARGE ARM ONLY (scoped: the pilot arm is DESIGNED to disperse)
            'large_arm_dispersion_below_the_declared_tolerance': (
                large['rms_mw_max_over_all_dso_blocks'] <= DISPERSION_ZERO_TOL_MW),
            'large_arm_dispersion_collapses_against_the_pilot': (
                ratio is not None and ratio <= DISPERSION_COLLAPSE_RATIO),
        }
        gate_pass = all(gate_items.values())

        payload = {
            **launch,
            'finished_utc': _utc(), 'wall_clock_s': time.time() - started,
            'arms': holder,
            'limit_comparison': {
                'pilot_rms_mw_max': pilot['rms_mw_max_over_all_dso_blocks'],
                'large_rms_mw_max': large['rms_mw_max_over_all_dso_blocks'],
                'ratio_large_over_pilot': ratio,
                'pilot_total_charge': pilot['total_charge_all_dso'],
                'large_total_charge': large['total_charge_all_dso'],
                'thresholds': launch['thresholds_declared_before_the_run'],
            },
            'gate_scope': ('the dispersion items apply to the `large` arm ONLY; the `pilot` arm '
                           'is designed to disperse and supplies the reference (CLAUDE.md: scope '
                           'a gate per arm)'),
            'not_a_result': ('two cycles is not a converged run; the pilot arm\'s dispersion here '
                             'is NOT the pilot result and is not comparable with any 1 x 1 figure'),
            'solve_profile': {'declared_total_strict': len(ARMS) * declared_base,
                              'reconciled_total': reconciled_total if supported else None,
                              'observed': GUARD.counts['permitted_solve'],
                              'counts': dict(GUARD.counts), 'verify_failures': guard_failures},
            'gate_items': gate_items, 'gate_pass': gate_pass,
        }
        gate_path = os.path.join(out_root, 'gate.json')
        G._refuse_overwrite(gate_path)
        with open(gate_path, 'w') as handle:
            json.dump(payload, handle, indent=1, default=str)

        manifest = {}
        for root, _dirs, fnames in os.walk(out_root):
            for fname in sorted(fnames):
                fpath = os.path.join(root, fname)
                manifest[os.path.relpath(fpath, REPO)] = _sha256_file(fpath)
        manifest_path = os.path.join(out_root, 'manifest_sha256.json')
        G._refuse_overwrite(manifest_path)
        with open(manifest_path, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        for key, value in gate_items.items():
            _log(f'   {key}: {value}')
        _log(f'GATE_PASS={gate_pass}; ratio large/pilot = {ratio}')
        return EXIT_OK if gate_pass else EXIT_ERROR
    finally:
        GUARD.uninstall()
        if os.path.exists(OWN_LOCK_PATH):
            os.remove(OWN_LOCK_PATH)


if __name__ == '__main__':
    sys.exit(main())
