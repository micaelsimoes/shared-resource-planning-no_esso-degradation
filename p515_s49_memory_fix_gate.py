"""
P5.15 Addendum 29 (task W32, STEP M2 gate) -- the SRP1 TWO-CYCLE BITWISE GATE for the solution-
bookkeeping release switch (`SolverParameters.release_solution_bookkeeping`, network.py
`_release_solution_bookkeeping`), switched ON, against the COMMITTED baseline C* trajectory.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 29 ("Fix what is bookkeeping; gate: bitwise identity on
two SRP1 cycles"); Planner task W32 STEP M2 ("bitwise identity on two SRP1 cycles at C* under the
baseline configuration, fix on vs the committed reference trajectory -- reuse the comparator conventions
of p515_s45_snapshot_off_two_cycle_gate.py / p515_s47 gates by import; declared solve count 51 x 3 = 153
+ recoveries, reconciled, armed bounded SolveProfileGuard").

REFERENCE (committed): the baseline re-certification of C*,
    data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json
(campaign s47_recert, spec campaign_spec_s47_recert_902f93aa.json; BASELINE (C2 + phi_cal 0.985 +
soh_min 0.70); AA keep_memory from the case file; cap 500), `cycle_trajectory[:2]` plus the rows-derived
top-level fields. Its sha256 is pinned below and checked before anything runs.

EVERYTHING IS W10's / W20's, BY IMPORT (`p515_s45_snapshot_off_two_cycle_gate` as W10), exactly as the
committed W20 gate `p515_s46_default_two_cycle_gate.py` uses it:
  * the armed guard `W10.GUARD` = SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED), installed at
    W10's import (the first import below), armed for the whole run, verified EXACTLY;
  * `W10.check_preconditions` (locks, live processes, write-once output, production files clean in git,
    the reference's pinned sha256, case-file AA == declaration, the pinned C* key) and
    `W10.capture_path_checklist`;
  * `W10.derive_solves_from_case_file` (51 per cycle; base 51 x (2 + 1) = 153) and the identity
    observed == 153 + tier1 + 2 x tier2;
  * `W10.run_arm(arm, 'on', ...)` -- the campaign child's own wiring (run_admm_arm cap 2, apply_rho False,
    full diagnostics, the child's configuration hook, the capture wrappers), snapshots at their defaults;
  * `W10.compare_on_against_committed_reference` (GATING here) and `W10.trajectory_field_table`.
TWO DECLARED SUBSTITUTIONS (the only departures from W20's gate; each recorded in gate.json):
  (1) `W10.AA_REFERENCE` is set to the baseline C* reference above (W10's module constant points at the
      pre-baseline S45 C* run, which the edited ESS parameters file no longer reproduces).
  (2) `p515_s44_campaign_harness._config_hook_factory`, as W10.run_arm resolves it, is wrapped so that
      the spec it receives ALSO declares the recert campaign's `ess_ageing_baseline` (+ label + the ESS
      parameters file pin), read from the committed recert spec -- so the child hook VERIFIES the baseline
      parameters loaded and reads them back from probe ESSO models, as the reference's child did -- and
      so that, after that hook, the switch under test is applied to the arm's planning object
      (`p515_s44_scale_measurement.set_release_solution_bookkeeping`, which reads it back and raises if it
      did not take effect).
THE SWITCH MUST DEMONSTRABLY RUN: pass-through counters around `network._run_smopf` (successful network
solves) and `network._release_solution_bookkeeping` (calls) -- the two counts must be EQUAL and > 0.

GATE (every item must hold): cycles_run == 2; zero diffs against the reference (trajectory rows and
derived top-level fields); zero per-field mismatches; solve identity; GUARD.verify(reconciled) exactly;
no blocked solver call; shared FrozenSMOPF tree untouched; the switch read back ON for the TSO and every
DSO; release calls == successful network solves > 0; the ESS-ageing declaration verified in the child.

EXACT LAUNCH COMMAND (repo root; attached, alone, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_memory_fix_gate.py \\
        > data/SRP1/Results/P515S49/memory_fix_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S49/memory_fix_gate/{gate.json, gate.md, manifest_sha256.json, arm/}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import copy
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

import network as NET  # noqa: E402
import p515_s44_scale_measurement as S44  # noqa: E402

GUARD = W10.GUARD
H = W10.H
G = W10.G
CP = W10.CP

STAGE = 'P5.15 Addendum 29 W32 M2 -- two-cycle bitwise gate: release_solution_bookkeeping ON vs committed baseline C*'
SCHEMA = 'p515_s49_memory_fix_gate_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 29', 'Planner task W32 STEP M2',
             'data/SRP1/Results/P515S48/frozen_s48_spec_v18_8bda2a0a.json memory_task']
ARM = 's49memfix'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'memory_fix_gate')
RECERT_DIR = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert')
RECERT_SPEC = {'path': os.path.join(RECERT_DIR, 'campaign_spec_s47_recert_902f93aa.json'),
               'sha256_prefix': '902f93aa'}
BASELINE_REFERENCE = {
    'report': os.path.join(RECERT_DIR, 'evals', '070f833e1e318f85_c_star', 'g_s39_D.json'),
    'sha256': 'ced90e47873925b5bb87360984d348558d11ea11fd004e93f3a2668fdc389971',
    'campaign': ('s47_recert (spec campaign_spec_s47_recert_902f93aa.json), BASELINE (C2 + phi_cal 0.985 + '
                 'soh_min 0.70), AA keep_memory from the case file, cap 500; cycles_run 87, '
                 'gross_operational_cost 650912327.1956586'),
}
EXTRA_CLEAN_FILES = ('network.py', 'solver_parameters.py', 'p515_s44_scale_measurement.py',
                     'p515_s45_snapshot_off_two_cycle_gate.py', 'p515_s44_campaign_harness.py',
                     'shared_energy_storage_data.py', 'data/SRP1/SharedESS/SRP1_ESS_Params.json')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W32-gate] {msg}', flush=True)


def _load_recert_declaration():
    path = os.path.join(REPO, RECERT_SPEC['path'])
    sha = CP._sha256_file(path)
    if not sha.startswith(RECERT_SPEC['sha256_prefix']):
        raise SystemExit(f'recert spec sha256 {sha} does not carry the pinned prefix {RECERT_SPEC["sha256_prefix"]}')
    with open(path) as handle:
        conf = json.load(handle)['configuration']
    decl = {k: conf[k] for k in ('ess_ageing_baseline', 'ess_ageing_baseline_label', 'ess_params_file')}
    return decl, {'path': RECERT_SPEC['path'], 'sha256': sha}


class SwitchCounters:
    """Pass-through counters (never behaviour changes) for the path under test."""

    def __init__(self):
        self.run_smopf_calls = 0
        self.run_smopf_succeeded = 0
        self.release_calls = 0
        self.release_after_success = 0
        self._last_succeeded = None

    def install(self):
        import helper_functions as HF
        self._orig_run = NET._run_smopf
        self._orig_release = NET._release_solution_bookkeeping
        counters = self

        def run_smopf(network, model, params, from_warm_start=False):
            counters.run_smopf_calls += 1
            result = counters._orig_run(network, model, params, from_warm_start=from_warm_start)
            if HF.solver_result_succeeded(result):
                counters.run_smopf_succeeded += 1
            return result

        def release(model, result):
            counters.release_calls += 1
            return counters._orig_release(model, result)

        NET._run_smopf = run_smopf
        NET._release_solution_bookkeeping = release
        return self

    def uninstall(self):
        NET._run_smopf = self._orig_run
        NET._release_solution_bookkeeping = self._orig_release


def main():
    out_root = os.path.join(REPO, OUT_REL)
    started = time.time()
    _log(STAGE)
    _log(f'git HEAD {W10._git(["rev-parse", "HEAD"])}; output root {W10._rel(out_root)}')
    os.environ.update(H.THREAD_CAP_ENV)

    # declared substitution (1): the baseline C* reference, pinned
    W10.AA_REFERENCE = dict(BASELINE_REFERENCE)
    failures, instance = W10.check_preconditions(out_root)
    status = W10._git(['status', '--porcelain', '--'] + list(EXTRA_CLEAN_FILES))
    if status.strip():
        failures.append(f'files not clean in git:\n{status}')
    declaration, recert_spec = _load_recert_declaration()
    ess_pin = declaration['ess_params_file']
    ess_sha_now = CP._sha256_file(os.path.join(REPO, ess_pin['path']))
    if ess_sha_now != ess_pin['sha256']:
        failures.append(f'ESS parameters file sha256 {ess_sha_now} != recert pin {ess_pin["sha256"]}')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    checklist = W10.capture_path_checklist()
    for module, name in ((NET, '_release_solution_bookkeeping'), (NET, '_run_smopf'),
                         (S44, 'set_release_solution_bookkeeping'), (H, '_config_hook_factory')):
        checklist[f'callable_{module.__name__}.{name}'] = callable(getattr(module, name, None))
    import inspect
    checklist['network_run_smopf_calls_release_under_switch'] = (
        'release_solution_bookkeeping' in inspect.getsource(NET._run_smopf)
        and '_release_solution_bookkeeping(model, result)' in inspect.getsource(NET._run_smopf))
    checklist['w10_run_arm_resolves_hook_factory_via_H'] = 'H._config_hook_factory(' in inspect.getsource(W10.run_arm)
    if not all(checklist.values()):
        raise SystemExit(f'capture-path checklist failed: {[k for k, v in checklist.items() if not v]}')
    _log(f'preconditions passed; capture-path checklist {len(checklist)} items, all true')
    declared = W10.derive_solves_from_case_file()
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle; base "
         f"{declared['declared_base_solves_per_arm']} (cap {W10.CAP} + initialization); gate on "
         f"{declared['identity']} and GUARD.verify(reconciled) exactly")

    # declared substitution (2): the recert declaration + the switch, around the child's own hook
    orig_factory = H._config_hook_factory
    switch_record = {}

    def factory_with_declaration_and_switch(spec_like, holder, overrides=None, **kwargs):
        spec2 = copy.deepcopy(spec_like)
        spec2['configuration'].update(copy.deepcopy(declaration))
        inner = orig_factory(spec2, holder, overrides=overrides, **kwargs)

        def hook(planning, sed, candidate, report):
            inner(planning=planning, sed=sed, candidate=candidate, report=report)
            applied = S44.set_release_solution_bookkeeping(planning, True)
            switch_record.update(applied)
            report.setdefault('rule_eleven_checklist', {})['w32_release_solution_bookkeeping'] = applied
        return hook

    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')
    counters = W10.CloneCaptureCounters()
    switch_counters = SwitchCounters().install()
    H._config_hook_factory = factory_with_declaration_and_switch
    expected_counts = {}
    arm_dir = os.path.join(out_root, 'arm')
    try:
        with counters.installed():
            report, summary = W10.run_arm(ARM, 'on', arm_dir, counters, expected_counts)
            counters.phase = 'post'
    finally:
        H._config_hook_factory = orig_factory
        switch_counters.uninstall()
    _log(f"arm: cycles={summary['cycles_run']} recourse={summary['recourse']} gross={summary['gross_operational_cost']} "
         f"solves={(summary['solve_profile'] or {}).get('observed')} failures={summary['network_failures_summary']}")
    _log(f'switch: {switch_record}; run_smopf calls {switch_counters.run_smopf_calls} succeeded '
         f'{switch_counters.run_smopf_succeeded}; release calls {switch_counters.release_calls}')

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
    reference['caveat'] = ('GATING here (W32 M2); cap 2 vs the reference cap 500 -- W10/W20 (committed) '
                           'reproduced committed C* rows with 0 diffs at cap 2')
    with open(os.path.join(REPO, W10.AA_REFERENCE['report'])) as handle:
        ref_rows = (json.load(handle).get('cycle_trajectory') or [])[:W10.CAP]
    table = W10.trajectory_field_table(ref_rows, report.get('cycle_trajectory') or [])
    ess_ageing_checklist = (report.get('rule_eleven_checklist') or {}).get('w21_ess_ageing_baseline') or {}
    network_solves_declared = declared['network_solves_per_cycle'] * (W10.CAP + 1)

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
        'switch_read_back_on_for_tso_and_every_dso': bool(switch_record.get('took_effect')),
        'release_called_once_per_successful_network_solve': (
            switch_counters.release_calls == switch_counters.run_smopf_succeeded > 0),
        'network_run_smopf_calls_equal_declared_network_solves': (
            switch_counters.run_smopf_calls == network_solves_declared),
        'ess_ageing_declaration_verified_in_child': (all((ess_ageing_checklist.get('checks') or {}).values())
                                                     and bool(ess_ageing_checklist.get('checks'))
                                                     and ess_ageing_checklist.get('readback_all_match') is True),
    }
    gate_pass = all(gate_items.values())
    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'w10_module_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s45_snapshot_off_two_cycle_gate.py')),
        'network_py_sha256': CP._sha256_file(os.path.join(REPO, 'network.py')),
        'solver_parameters_py_sha256': CP._sha256_file(os.path.join(REPO, 'solver_parameters.py')),
        'scale_script_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s44_scale_measurement.py')),
        'harness_sha256': CP._sha256_file(H.HARNESS_PATH),
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': CP._sha256_file(H.CASE_FILE),
                      'anderson_acceleration_loaded': instance['case_file_anderson_acceleration_loaded']},
        'ess_params_file': {'path': ess_pin['path'], 'sha256': ess_sha_now, 'recert_pin': ess_pin},
        'declared_substitutions': {
            '1_reference': BASELINE_REFERENCE,
            '2_hook_declaration_and_switch': {'recert_spec': recert_spec, 'declaration_injected': declaration,
                                              'switch': 'p515_s44_scale_measurement.set_release_solution_bookkeeping(planning, True) after the child hook'}},
        'instance': {'candidate_label': 'C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025)',
                     'candidate_canonical': instance['candidate_canonical'],
                     'candidate_key': instance['candidate_key'],
                     'candidate_key_pin_matches': instance['candidate_key'] == W10.C_STAR_KEY_PIN},
        'configuration': {'arm': ARM, 'cap': W10.CAP, 'arm_label': W10.ARM_LABEL, 'snapshots': 'on (no assignment)',
                          'release_solution_bookkeeping': True, 'model_variant': None, 'overrides': {},
                          'apply_rho': False, 'full_diagnostics_in_rows': True, 'thread_caps': dict(H.THREAD_CAP_ENV),
                          'nlp_solver_path': H._resolve_solver_path_from_dotenv()},
        'objective_convention': ('gross_operational_cost is the settlement-excluded gross cost; '
                                 'net_operational_recourse differs by the terminal salvage credit'),
        'capture_path_checklist_asserted_before_run': checklist,
        'switch_record': switch_record,
        'switch_counters': {'run_smopf_calls': switch_counters.run_smopf_calls,
                            'run_smopf_succeeded': switch_counters.run_smopf_succeeded,
                            'release_calls': switch_counters.release_calls,
                            'network_solves_declared': network_solves_declared},
        'arm_summary': summary,
        'clone_capture_counters_informational': counters.buckets,
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
    lines = [f'# {STAGE}', '', f"git HEAD at run {payload['git_head_at_run']}; GATE_PASS = **{gate_pass}**", '',
             f"Instance: C* key {instance['candidate_key']} (pin matches: {payload['instance']['candidate_key_pin_matches']}).",
             f"Reference: {BASELINE_REFERENCE['report']} (sha256 {BASELINE_REFERENCE['sha256']}); cycle_trajectory[:2].",
             '', '| gate item | holds |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in gate_items.items()]
    lines += ['', f"Solves: base {base} + tier1 {tier1} + 2 x tier2 {tier2} = {reconciled}; observed {observed}; "
                  f"guard {dict(GUARD.counts)}; verify failures {guard_failures}.",
              f"Switch: run_smopf calls {switch_counters.run_smopf_calls} (declared network solves {network_solves_declared}), "
              f"succeeded {switch_counters.run_smopf_succeeded}, release calls {switch_counters.release_calls}.",
              f"Reference comparison: n_diffs {reference['n_diffs']}, first {reference['first_differing_field']}; "
              f"field-table mismatches {table['total_mismatches']} over {table['n_fields']} fields.",
              f"Arm: cycles {summary['cycles_run']}, gross_operational_cost {summary['gross_operational_cost']} "
              f"(gross convention), recourse {summary['recourse']}, terminal step / threshold "
              f"{summary['rule_ten_terminal_step_over_threshold']}.",
              f"Peak RSS (ru_maxrss, bytes): {summary['peak_rss']}"]
    md_path = os.path.join(out_root, 'gate.md')
    W10._refuse_overwrite(md_path)
    with open(md_path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_root, 'manifest_sha256.json')
    W10._refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f'wrote {W10._rel(gate_path)}, gate.md and {W10._rel(manifest_path)} ({len(manifest)} files)')
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
