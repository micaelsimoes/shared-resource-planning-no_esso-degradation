"""
P5.15 Addenda 28-29 (task W20, item 4(b)) -- THE ONE SOLVE-BEARING GATE: defaults
unchanged. A TWO-CYCLE run at C* with the AA-on case file and the DEFAULT settings
(no model variant, no override), compared BITWISE with the committed AA C*
trajectory `cycle_trajectory[:2]` of
`data/SRP1/Results/P515S45/reverify_aa_c_star/evals/3e741dac72c9e1bc_c_star_aa_case_file/g_s39_D.json`.
Run AFTER the case-file edit (commit cb165d4e) and the production switches (commit
e5721b6d) are committed, so it covers both: the reference was produced before
either existed.

EVERYTHING IS W10's, BY IMPORT (`p515_s45_snapshot_off_two_cycle_gate` as W10):
  * the armed guard: `W10.GUARD`, a `SolveProfileGuard(p514_n_instrumented_cstar.
    PERMITTED)` installed when W10 is imported -- the first import of this script
    after the guard module, before any model code -- and armed for the whole run;
  * the preconditions (`W10.check_preconditions`: locks, live processes, write-once
    output, production files clean in git, the reference's pinned sha256, the case
    file's AA dict == the declaration, the pinned C* key) and the capture-path
    checklist (`W10.capture_path_checklist`);
  * the solve profile declared BEFORE the run from the case file
    (`W10.derive_solves_from_case_file`: 51 solves/cycle, base 51 x (2 + 1) = 153)
    and the reconciliation identity observed == 153 + tier1 + 2 x tier2;
  * the arm itself: `W10.run_arm(arm='s46default', snapshots='on', ...)` -- the
    campaign child's own wiring (`run_admm_arm` cap 2, `apply_rho=False`,
    `full_diagnostics_in_rows=True`, the child's configuration hook with the
    case-file AA declaration and NO model variant, the capture wrappers); 'on'
    performs no assignment at all (default snapshot modes);
  * the comparator: `W10.compare_on_against_committed_reference` (CP._diff over
    cycle_trajectory[:2] + the rows-derived top-level fields; only the
    rule_eleven_checklist subtree is provenance) and `W10.trajectory_field_table`
    (every `H.RECORD_TRAJECTORY_FIELDS` field, cycle by cycle, bitwise).
Here the reference comparison is GATING (W10 reported it non-gating, and it
reproduced there with 0 diffs at cap 2).

THE GATE (every item must hold): cycles_run == 2; zero diffs against the reference
(trajectory rows and derived top-level fields); the per-field table has zero
mismatches; the solve identity holds; `GUARD.verify(reconciled total)` passes
EXACTLY; no blocked solver call; the shared FrozenSMOPF tree untouched; the run's
baseline carries the edited factor (4.0) and the default ageing settings ('end',
True).

Working-dir ids are W10's form with this arm's name:
`p515s44_s45_w10_snapoff_s46default_<key16>_{run,precheck}`.

EXACT LAUNCH COMMAND (repo root; attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s46_default_two_cycle_gate.py \\
        > data/SRP1/Results/P515S46/default_two_cycle_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S46/default_two_cycle_gate/
    gate.json, manifest_sha256.json, arm/ (the arm's artifacts)
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s45_snapshot_off_two_cycle_gate as W10  # noqa: E402 -- installs W10.GUARD (armed) at import

GUARD = W10.GUARD
H = W10.H
G = W10.G
CP = W10.CP

STAGE = 'P5.15 Addenda 28-29 W20 item 4(b) -- two-cycle bitwise gate: defaults unchanged (C*, AA-on case file)'
SCHEMA = 'p515_s46_default_two_cycle_gate_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addenda 28-29', 'Planner task W20 item 4(b) (the ONE solve-bearing gate)',
             'commits e5721b6d (ageing switches, model_variant) and cb165d4e (case-file edit)']
ARM = 's46default'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S46', 'default_two_cycle_gate')
ESS_PARAMS_REL = os.path.join('data', 'SRP1', 'SharedESS', 'SRP1_ESS_Params.json')
REQUIRED_COMMITS = ('e5721b6d', 'cb165d4e')
EDITED_FACTOR = 4.0


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W20-2cyc] {msg}', flush=True)


def _extra_preconditions():
    failures, info = [], {}
    for commit in REQUIRED_COMMITS:
        ok = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=REPO,
                            capture_output=True).returncode == 0
        info[f'commit_{commit}_in_HEAD'] = ok
        if not ok:
            failures.append(f'commit {commit} is not an ancestor of HEAD')
    # this script's own sha256 is recorded in gate.json (it is committed WITH its evidence)
    status = W10._git(['status', '--porcelain', '--', ESS_PARAMS_REL, 'shared_energy_storage_data.py',
                       'p515_s44_campaign_harness.py', 'p515_s45_snapshot_off_two_cycle_gate.py'])
    info['files_clean'] = not status.strip()
    if status.strip():
        failures.append(f'files not clean in git:\n{status}')
    with open(os.path.join(REPO, ESS_PARAMS_REL)) as handle:
        factor = json.load(handle)['max_energy_to_power_factor']
    info['ess_params_factor_in_file'] = factor
    if factor != EDITED_FACTOR:
        failures.append(f'{ESS_PARAMS_REL} max_energy_to_power_factor {factor} != {EDITED_FACTOR}')
    return failures, info


def _baseline_state():
    """After the run: the in-process baseline every arm deep-copies (p56a_oracle.load_baseline, cached)."""
    import shared_energy_storage_data as SED
    sed = G.O.load_baseline()['planning'].shared_ess_data
    return {'max_energy_to_power_ratio_loaded': sed.params.max_energy_to_power_ratio,
            'ageing_model_settings': list(SED._esso_ageing_model_settings(sed)),
            'cl_eff_per_year': {str(y): sed.shared_energy_storages[y][0].cl_eff for y in sed.years},
            'phi_cal_per_year': {str(y): sed.shared_energy_storages[y][0].phi_cal for y in sed.years}}


def main():
    out_root = os.path.join(REPO, OUT_REL)
    started = time.time()
    _log(STAGE)
    _log(f'git HEAD {W10._git(["rev-parse", "HEAD"])}; output root {W10._rel(out_root)}')
    env_before = {k: os.environ.get(k) for k in H.THREAD_CAP_ENV}
    os.environ.update(H.THREAD_CAP_ENV)
    _log(f'thread caps: before={env_before} after={ {k: os.environ.get(k) for k in H.THREAD_CAP_ENV} }')

    failures, instance = W10.check_preconditions(out_root)
    more, extra_info = _extra_preconditions()
    failures += more
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    checklist = W10.capture_path_checklist()
    import inspect
    run_arm_src = inspect.getsource(W10.run_arm)
    checklist['w20_run_arm_hook_call_has_no_model_variant'] = ('model_variant' not in run_arm_src
                                                               and 'overrides={})' in run_arm_src)
    if not all(checklist.values()):
        raise SystemExit(f'capture-path checklist failed: {[k for k, v in checklist.items() if not v]}')
    _log(f'preconditions passed ({extra_info}); capture-path checklist {len(checklist)} items, all true')
    declared = W10.derive_solves_from_case_file()
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle; base "
         f"{declared['declared_base_solves_per_arm']} for this ONE arm (cap {W10.CAP} + initialization); gate on "
         f"{declared['identity']} and GUARD.verify(reconciled) exactly")

    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')
    counters = W10.CloneCaptureCounters()
    expected_counts = {}
    arm_dir = os.path.join(out_root, 'arm')
    with counters.installed():
        report, summary = W10.run_arm(ARM, 'on', arm_dir, counters, expected_counts)
        counters.phase = 'post'
    _log(f"arm: cycles={summary['cycles_run']} recourse={summary['recourse']} gross={summary['gross_operational_cost']} "
         f"solves={(summary['solve_profile'] or {}).get('observed')} failures={summary['network_failures_summary']}")

    observed = ((summary['solve_profile'] or {}).get('observed') or {}).get('permitted_solve')
    classes = (summary['network_failures_summary'] or {}).get('classes') or {}
    tier1, tier2 = classes.get('recovered_tier1', 0), classes.get('recovered_tier2', 0)
    base = declared['declared_base_solves_per_arm']
    reconciled = base + tier1 + 2 * tier2
    guard_failures = GUARD.verify(reconciled)
    solve = {'declared_before_the_run': declared, 'base_declared': base, 'recovered_tier1': tier1,
             'recovered_tier2': tier2, 'expected_after_reconciliation': reconciled, 'observed_in_arm': observed,
             'identity_holds': observed == reconciled,
             'expected_block_counts_from_planning': expected_counts.get(ARM),
             'planning_derivation_agrees_with_case_file': ((expected_counts.get(ARM) or {}).get('solves_per_cycle')
                                                          == declared['solves_per_cycle']),
             'process_guard': {'permitted_call_sites': [list(p) for p in W10.N.PERMITTED],
                               'counts': dict(GUARD.counts), 'verify_exactly': reconciled,
                               'verify_failures': guard_failures,
                               'no_solves_outside_the_arm': GUARD.counts['permitted_solve'] == observed}}

    reference = W10.compare_on_against_committed_reference(report)
    reference['gating'] = True
    reference['caveat'] = ('GATING here (W20 item 4(b)); cap 2 vs the reference cap 500 -- W10 (committed) '
                           'reproduced the same two rows with 0 diffs at cap 2')
    with open(os.path.join(REPO, W10.AA_REFERENCE['report'])) as handle:
        ref_rows = (json.load(handle).get('cycle_trajectory') or [])[:W10.CAP]
    table = W10.trajectory_field_table(ref_rows, report.get('cycle_trajectory') or [])
    baseline = _baseline_state()

    gate_items = {
        'ran_two_cycles': summary['cycles_run'] == W10.CAP,
        'reproduces_committed_reference_zero_diffs': reference['reproduces'] and reference['n_diffs'] == 0,
        'trajectory_fields_identical_to_reference': (table['total_mismatches'] == 0 and table['rows_length_equal']),
        'solve_reconciliation_identity_holds': solve['identity_holds'],
        'planning_derivation_agrees_with_case_file': solve['planning_derivation_agrees_with_case_file'],
        'process_guard_verified_exactly': not guard_failures,
        'no_solves_outside_the_arm': solve['process_guard']['no_solves_outside_the_arm'],
        'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
        'shared_frozen_smopf_untouched': not (report.get('shared_frozen_smopf_modified')
                                              or report.get('shared_frozen_smopf_new_files')),
        'baseline_carries_edited_factor_4': baseline['max_energy_to_power_ratio_loaded'] == EDITED_FACTOR,
        'baseline_default_ageing_settings': baseline['ageing_model_settings'] == ['end', True],
        'no_model_variant_applied': 'w20_model_variant' not in (report.get('rule_eleven_checklist') or {}),
    }
    gate_pass = all(gate_items.values())
    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'w10_module_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s45_snapshot_off_two_cycle_gate.py')),
        'harness_sha256': CP._sha256_file(H.HARNESS_PATH),
        'shared_energy_storage_data_sha256': CP._sha256_file(os.path.join(REPO, 'shared_energy_storage_data.py')),
        'ess_params_file': {'path': ESS_PARAMS_REL, 'sha256': CP._sha256_file(os.path.join(REPO, ESS_PARAMS_REL))},
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': CP._sha256_file(H.CASE_FILE),
                      'anderson_acceleration_loaded': instance['case_file_anderson_acceleration_loaded'],
                      'declaration': W10.CASE_FILE_AA},
        'instance': {'candidate_label': 'C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025)',
                     'candidate_canonical': instance['candidate_canonical'],
                     'candidate_key': instance['candidate_key'],
                     'candidate_key_pin_matches': instance['candidate_key'] == W10.C_STAR_KEY_PIN},
        'configuration': {'arm': ARM, 'cap': W10.CAP, 'arm_label': W10.ARM_LABEL, 'snapshots': 'on (no assignment)',
                          'model_variant': None, 'overrides': {}, 'apply_rho': False,
                          'full_diagnostics_in_rows': True, 'thread_caps': dict(H.THREAD_CAP_ENV),
                          'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
                          'nlp_solver_path': H._resolve_solver_path_from_dotenv()},
        'objective_convention': ('gross_operational_cost is the settlement-excluded gross cost; '
                                 'net_operational_recourse differs by the terminal salvage credit'),
        'preconditions_extra': extra_info,
        'capture_path_checklist_asserted_before_run': checklist,
        'arm_summary': summary,
        'clone_capture_counters_informational': counters.buckets,
        'baseline_state_after_run': baseline,
        'solve_profile': solve,
        'reference_comparison_GATING': reference,
        'trajectory_field_table_vs_reference': table,
        'gate_items': gate_items, 'gate_pass': gate_pass,
        'wall_clock_s': time.time() - started,
    }
    gate_path = os.path.join(out_root, 'gate.json')
    W10._refuse_overwrite(gate_path)
    with open(gate_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_root, 'manifest_sha256.json')
    W10._refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f'wrote {W10._rel(gate_path)} and {W10._rel(manifest_path)} ({len(manifest)} files)')
    for name, value in gate_items.items():
        _log(f'  {"OK  " if value else "FAIL"} {name}')
    _log(f"solves: base {base} + tier1 {tier1} + 2 x tier2 {tier2} = {reconciled}; observed {observed}; "
         f"guard {dict(GUARD.counts)} verify_failures={guard_failures}")
    _log(f"reference: reproduces={reference['reproduces']} n_diffs={reference['n_diffs']} "
         f"first={reference['first_differing_field']}; field-table mismatches={table['total_mismatches']}")
    _log(f'GATE_PASS={gate_pass} wall={time.time() - started:.1f}s')
    if not gate_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
