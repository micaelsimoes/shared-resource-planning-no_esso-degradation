"""
P5.15 Addendum 34 (task W33, item 3) -- the SRP1 TWO-CYCLE BITWISE GATE for the flexibility-price override
(`p515_s44_campaign_harness`, evaluation option `flex_price_multiplier`, commit 2e82a23f), PRESENT at m = 1.0
explicitly, against the COMMITTED baseline C* trajectory.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 34; frozen spec v19
`data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json` `flex_price_ladder`; Planner task W33 item 3
("with the override present at m = 1.0 explicitly, two SRP1 cycles at C* under the baseline reproduce the
committed baseline re-certification trajectory cycle_trajectory[:2] bitwise -- reuse the W32 gate
(p515_s49_memory_fix_gate.py) conventions by import; declared solves 153 + recoveries reconciled; armed bounded
guard").

REFERENCE (committed): the baseline re-certification of C*, `W32.BASELINE_REFERENCE` (BY IMPORT):
    data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json
`cycle_trajectory[:2]` plus the rows-derived top-level fields; its sha256 is pinned and checked before anything
runs (W10.check_preconditions).

EVERYTHING IS W10's / W32's, BY IMPORT, exactly as the committed W32 gate uses it:
  * the armed guard `W10.GUARD` = SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED), installed at W10's
    import (the first import below, via W32), armed for the whole run, verified EXACTLY;
  * `W10.check_preconditions`, `W10.capture_path_checklist`, `W10.derive_solves_from_case_file` (51 per cycle;
    base 51 x (2 + 1) = 153), `W10.run_arm(arm, 'on', ...)` (the campaign child's own wiring: run_admm_arm cap 2,
    apply_rho False, full diagnostics, the child's configuration hook, the capture wrappers),
    `W10.compare_on_against_committed_reference` (GATING here) and `W10.trajectory_field_table`;
  * `W32.BASELINE_REFERENCE` and `W32._load_recert_declaration` (the recert campaign's ESS-ageing declaration,
    label and ESS-parameters pin, read from the committed recert spec).
TWO DECLARED SUBSTITUTIONS (each recorded in gate.json):
  (1) `W10.AA_REFERENCE` := `W32.BASELINE_REFERENCE` (as W32).
  (2) `p515_s44_campaign_harness._config_hook_factory`, as W10.run_arm resolves it, is wrapped so that the spec it
      receives ALSO declares the recert `ess_ageing_baseline` (as W32), and the factory is called with
      `flex_price_multiplier=1.0` -- the override under test, PRESENT at m = 1.0 -- so the child hook replaces every
      DSO block's cost_flex by a new array 1.0 x cost_flex and reads it back from 36 probe DSO blocks before any
      solve (refusing on mismatch). The wrapper keeps a reference to the hook's `holder` to record what was applied.
No other departure: W32's switch (release_solution_bookkeeping) is NOT applied here (default off, as the
reference ran).

SOLVE RECONCILIATION (declared before the run, gating). W10's identity credits retries only to RECOVERED blocks
(observed == 153 + tier1 + 2 x tier2), so a block whose retries all failed ('unrecovered') would leave its retry
solves uncredited -- the accounting gap noted at 78a9b230. This gate therefore gates on an EVENT-LEVEL identity
that credits every retry actually attempted, recovered or not:
    observed == 153 + sum over network-failure events of [recovery_attempted] + [tier2_attempted]
read from the arm's own `network_failures_s39_D.jsonl` (the per-event flags set by the production-log parser
`p515_g_g1_g4_admm_gates._scan_and_write_network_failures`); ESSO recovery events carry no attempt count in that
record, so ANY ESSO recovery event, and any 'indeterminate' network event, makes the reconciliation
unsupported and the gate FAIL (never silently credited). W10's recovered-only identity is reported beside it; the
two agree whenever no unrecovered block occurs. `GUARD.verify` is called with the event-level count, EXACTLY.

GATE (every item must hold): cycles_run == 2; zero diffs against the reference (trajectory rows and derived
top-level fields); zero per-field mismatches; the event-level solve identity; GUARD.verify exactly; no blocked
solver call; shared FrozenSMOPF tree untouched; the ESS-ageing declaration verified in the child; the override
applied at m = 1.0 (every DSO array a NEW object equal bitwise to 1.0 x the original, TSO arrays untouched) and read
back from all 36 probe DSO blocks with every flex coefficient bitwise equal.

EXACT LAUNCH COMMAND (repo root; attached, alone, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s49_flex_price_gate.py \\
        > data/SRP1/Results/P515S49/flex_price_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S49/flex_price_gate/{gate.json, gate.md, manifest_sha256.json, arm/}
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

import p515_s49_memory_fix_gate as W32  # noqa: E402 -- imports W10 first (installs W10.GUARD, armed)

W10 = W32.W10
GUARD = W10.GUARD
H = W10.H
G = W10.G
CP = W10.CP

STAGE = ('P5.15 Addendum 34 W33 item 3 -- two-cycle bitwise gate: flexibility-price override PRESENT at m = 1.0 vs '
         'committed baseline C*')
SCHEMA = 'p515_s49_flex_price_gate_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 34', 'Planner task W33 item 3',
             'data/SRP1/Results/P515S49/frozen_s49_spec_v19_f8adcc97.json flex_price_ladder']
ARM = 's49flexgate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'flex_price_gate')
FLEX_PRICE_MULTIPLIER = 1.0
HARNESS_COMMIT = '2e82a23f'
EXTRA_CLEAN_FILES = ('network.py', 'solver_parameters.py', 'model_construction_helpers.py',
                     'p515_s44_scale_measurement.py', 'p515_s45_snapshot_off_two_cycle_gate.py',
                     'p515_s49_memory_fix_gate.py', 'p515_s44_campaign_harness.py', 'shared_energy_storage_data.py',
                     'data/SRP1/SharedESS/SRP1_ESS_Params.json')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W33-gate] {msg}', flush=True)


def event_level_reconciliation(report, base):
    """observed == base + sum over network-failure events of [recovery_attempted] + [tier2_attempted]; unsupported
    (None) when any ESSO recovery event or 'indeterminate' network event exists."""
    summary = report.get('network_failures_summary') or {}
    path = summary.get('path')
    events = []
    if path and os.path.isfile(os.path.join(REPO, path)):
        with open(os.path.join(REPO, path)) as handle:
            events = [json.loads(line) for line in handle if line.strip()]
    network_events = [e for e in events if e.get('record_type', 'network_block') == 'network_block'
                      and e.get('class') is not None]
    retries = sum(int(bool(e.get('recovery_attempted'))) + int(bool(e.get('tier2_attempted'))) for e in network_events)
    n_esso = int(summary.get('n_esso_recovery_events') or 0)
    n_indet = sum(1 for e in network_events if e.get('class') == 'indeterminate')
    supported = (n_esso == 0 and n_indet == 0 and len(network_events) == int(summary.get('n_blocks') or 0))
    return {'definition': ('observed == base + sum over network-failure events of [recovery_attempted] + '
                           '[tier2_attempted] (every retry actually attempted, recovered or not); unsupported when '
                           'an ESSO recovery event or an indeterminate network event exists, or the event file does '
                           'not hold summary.n_blocks events'),
            'events_file': path, 'n_network_events': len(network_events),
            'n_events_by_class': {c: sum(1 for e in network_events if e.get('class') == c)
                                  for c in ('recovered_tier1', 'recovered_tier2', 'unrecovered', 'not_attempted',
                                            'indeterminate')},
            'retry_solves_credited': retries, 'n_esso_recovery_events': n_esso, 'n_indeterminate': n_indet,
            'supported': supported, 'expected': (base + retries) if supported else None}


def main():
    out_root = os.path.join(REPO, OUT_REL)
    started = time.time()
    _log(STAGE)
    _log(f'git HEAD {W10._git(["rev-parse", "HEAD"])}; output root {W10._rel(out_root)}')
    os.environ.update(H.THREAD_CAP_ENV)

    # declared substitution (1): the baseline C* reference, pinned (W32's constant, by import)
    W10.AA_REFERENCE = dict(W32.BASELINE_REFERENCE)
    failures, instance = W10.check_preconditions(out_root)
    status = W10._git(['status', '--porcelain', '--'] + list(EXTRA_CLEAN_FILES))
    if status.strip():
        failures.append(f'files not clean in git:\n{status}')
    in_head = subprocess.run(['git', 'merge-base', '--is-ancestor', HARNESS_COMMIT, 'HEAD'], cwd=REPO,
                             capture_output=True).returncode == 0
    declaration, recert_spec = W32._load_recert_declaration()
    ess_pin = declaration['ess_params_file']
    ess_sha_now = CP._sha256_file(os.path.join(REPO, ess_pin['path']))
    if ess_sha_now != ess_pin['sha256']:
        failures.append(f'ESS parameters file sha256 {ess_sha_now} != recert pin {ess_pin["sha256"]}')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    checklist = W10.capture_path_checklist()
    import inspect
    for name in ('apply_flex_price_multiplier', 'validate_flex_price_multiplier', '_config_hook_factory',
                 'flex_price_readback_run_models'):
        checklist[f'callable_H.{name}'] = callable(getattr(H, name, None))
    checklist['hook_factory_accepts_flex_price_multiplier'] = (
        'flex_price_multiplier' in inspect.signature(H._config_hook_factory).parameters)
    checklist['hook_applies_the_multiplier'] = ('apply_flex_price_multiplier(planning, flex_price_multiplier)'
                                                in inspect.getsource(H._config_hook_factory))
    checklist['w10_run_arm_resolves_hook_factory_via_H'] = 'H._config_hook_factory(' in inspect.getsource(W10.run_arm)
    checklist['network_failure_events_carry_attempt_flags'] = (
        "'recovery_attempted'" in inspect.getsource(G._new_network_event)
        and "'tier2_attempted'" in inspect.getsource(G._new_network_event))
    if not all(checklist.values()):
        raise SystemExit(f'capture-path checklist failed: {[k for k, v in checklist.items() if not v]}')
    _log(f'preconditions passed; capture-path checklist {len(checklist)} items, all true')
    declared = W10.derive_solves_from_case_file()
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle; base "
         f"{declared['declared_base_solves_per_arm']} (cap {W10.CAP} + initialization); gate on the event-level "
         f"identity observed == base + every attempted retry (recovered or not) and GUARD.verify(that) exactly")

    # declared substitution (2): the recert declaration + the override at m = 1.0, around the child's own hook
    orig_factory = H._config_hook_factory
    captured = {}

    def factory_with_declaration_and_override(spec_like, holder, overrides=None, **kwargs):
        if 'flex_price_multiplier' in kwargs:
            raise RuntimeError('unexpected flex_price_multiplier from W10.run_arm')
        spec2 = copy.deepcopy(spec_like)
        spec2['configuration'].update(copy.deepcopy(declaration))
        captured['holder'] = holder
        return orig_factory(spec2, holder, overrides=overrides, flex_price_multiplier=FLEX_PRICE_MULTIPLIER, **kwargs)

    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')
    counters = W10.CloneCaptureCounters()
    H._config_hook_factory = factory_with_declaration_and_override
    expected_counts = {}
    arm_dir = os.path.join(out_root, 'arm')
    try:
        with counters.installed():
            report, summary = W10.run_arm(ARM, 'on', arm_dir, counters, expected_counts)
            counters.phase = 'post'
    finally:
        H._config_hook_factory = orig_factory
    holder = captured.get('holder') or {}
    applied = holder.get('flex_price_applied') or {}
    readback = applied.get('readback_pre_run') or {}
    _log(f"arm: cycles={summary['cycles_run']} recourse={summary['recourse']} gross={summary['gross_operational_cost']} "
         f"solves={(summary['solve_profile'] or {}).get('observed')} failures={summary['network_failures_summary']}")
    _log(f"override: m={applied.get('flex_price_multiplier')} checks={applied.get('checks')} readback "
         f"{ {k: readback.get(k) for k in ('n_blocks', 'all_match', 'n_flex_coefficients', 'n_bitwise_exact', 'max_rel_dev')} }")

    observed = ((summary['solve_profile'] or {}).get('observed') or {}).get('permitted_solve')
    classes = (summary['network_failures_summary'] or {}).get('classes') or {}
    tier1, tier2 = classes.get('recovered_tier1', 0), classes.get('recovered_tier2', 0)
    base = declared['declared_base_solves_per_arm']
    w10_identity = base + tier1 + 2 * tier2
    event_level = event_level_reconciliation(report, base)
    reconciled = event_level['expected']
    guard_failures = GUARD.verify(reconciled) if reconciled is not None else ['event-level reconciliation unsupported']
    solve = {'declared_before_the_run': declared, 'base_declared': base,
             'event_level_reconciliation_GATING': event_level,
             'w10_recovered_only_identity_reported': {'recovered_tier1': tier1, 'recovered_tier2': tier2,
                                                      'expected': w10_identity, 'holds': observed == w10_identity},
             'observed_in_arm': observed, 'identity_holds': reconciled is not None and observed == reconciled,
             'expected_block_counts_from_planning': expected_counts.get(ARM),
             'planning_derivation_agrees_with_case_file': ((expected_counts.get(ARM) or {}).get('solves_per_cycle')
                                                          == declared['solves_per_cycle']),
             'process_guard': {'permitted_call_sites': [list(p) for p in W10.N.PERMITTED],
                               'counts': dict(GUARD.counts), 'verify_exactly': reconciled,
                               'verify_failures': guard_failures,
                               'no_solves_outside_the_arm': GUARD.counts['permitted_solve'] == observed}}

    reference = W10.compare_on_against_committed_reference(report)
    reference['gating'] = True
    reference['caveat'] = ('GATING here (W33 item 3); cap 2 vs the reference cap 500 -- W10/W20/W32 (committed) '
                           'reproduced committed C* rows with 0 diffs at cap 2')
    with open(os.path.join(REPO, W10.AA_REFERENCE['report'])) as handle:
        ref_rows = (json.load(handle).get('cycle_trajectory') or [])[:W10.CAP]
    table = W10.trajectory_field_table(ref_rows, report.get('cycle_trajectory') or [])
    ess_ageing_checklist = (report.get('rule_eleven_checklist') or {}).get('w21_ess_ageing_baseline') or {}
    rule_eleven_flex = (report.get('rule_eleven_checklist') or {}).get('w33_flex_price_multiplier') or {}

    gate_items = {
        'ran_two_cycles': summary['cycles_run'] == W10.CAP,
        'reproduces_committed_reference_zero_diffs': reference['reproduces'] and reference['n_diffs'] == 0,
        'trajectory_fields_identical_to_reference': (table['total_mismatches'] == 0 and table['rows_length_equal']),
        'solve_reconciliation_event_level_identity_holds': solve['identity_holds'],
        'planning_derivation_agrees_with_case_file': solve['planning_derivation_agrees_with_case_file'],
        'process_guard_verified_exactly': not guard_failures,
        'no_solves_outside_the_arm': solve['process_guard']['no_solves_outside_the_arm'],
        'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
        'shared_frozen_smopf_untouched': not (report.get('shared_frozen_smopf_modified')
                                              or report.get('shared_frozen_smopf_new_files')),
        'ess_ageing_declaration_verified_in_child': (all((ess_ageing_checklist.get('checks') or {}).values())
                                                     and bool(ess_ageing_checklist.get('checks'))
                                                     and ess_ageing_checklist.get('readback_all_match') is True),
        'override_present_at_m_1_0': (applied.get('flex_price_multiplier') == FLEX_PRICE_MULTIPLIER
                                      and rule_eleven_flex.get('flex_price_multiplier') == FLEX_PRICE_MULTIPLIER),
        'override_applied_checks_all_true': bool(applied.get('checks')) and all(applied['checks'].values()),
        'override_read_back_36_blocks_bitwise': (readback.get('all_match') is True and readback.get('n_blocks') == 36
                                                 and readback.get('n_flex_coefficients', 0) > 0
                                                 and readback.get('n_bitwise_exact') == readback.get('n_flex_coefficients')),
        'harness_commit_in_head': in_head,
    }
    gate_pass = all(gate_items.values())
    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'w10_module_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s45_snapshot_off_two_cycle_gate.py')),
        'w32_module_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s49_memory_fix_gate.py')),
        'harness_sha256': CP._sha256_file(H.HARNESS_PATH), 'harness_commit': HARNESS_COMMIT,
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': CP._sha256_file(H.CASE_FILE),
                      'anderson_acceleration_loaded': instance['case_file_anderson_acceleration_loaded']},
        'ess_params_file': {'path': ess_pin['path'], 'sha256': ess_sha_now, 'recert_pin': ess_pin},
        'declared_substitutions': {
            '1_reference': W32.BASELINE_REFERENCE,
            '2_hook_declaration_and_override': {
                'recert_spec': recert_spec, 'declaration_injected': declaration,
                'override': (f'p515_s44_campaign_harness._config_hook_factory(..., flex_price_multiplier='
                             f'{FLEX_PRICE_MULTIPLIER}) -- present explicitly')}},
        'instance': {'candidate_label': 'C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025)',
                     'candidate_canonical': instance['candidate_canonical'],
                     'candidate_key': instance['candidate_key'],
                     'candidate_key_pin_matches': instance['candidate_key'] == W10.C_STAR_KEY_PIN},
        'configuration': {'arm': ARM, 'cap': W10.CAP, 'arm_label': W10.ARM_LABEL, 'snapshots': 'on (no assignment)',
                          'flex_price_multiplier': FLEX_PRICE_MULTIPLIER, 'release_solution_bookkeeping': 'default (off)',
                          'model_variant': None, 'overrides': {}, 'apply_rho': False,
                          'full_diagnostics_in_rows': True, 'thread_caps': dict(H.THREAD_CAP_ENV),
                          'nlp_solver_path': H._resolve_solver_path_from_dotenv()},
        'objective_convention': ('gross_operational_cost is the settlement-excluded gross cost; '
                                 'net_operational_recourse differs by the terminal salvage credit'),
        'capture_path_checklist_asserted_before_run': checklist,
        'flex_price_applied_in_child': applied,
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
             f"Reference: {W32.BASELINE_REFERENCE['report']} (sha256 {W32.BASELINE_REFERENCE['sha256']}); "
             'cycle_trajectory[:2].',
             f'Override: flex_price_multiplier = {FLEX_PRICE_MULTIPLIER} PRESENT; read-back {readback.get("n_blocks")} '
             f'probe DSO blocks, {readback.get("n_bitwise_exact")}/{readback.get("n_flex_coefficients")} flex '
             'coefficients bitwise equal.', '', '| gate item | holds |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in gate_items.items()]
    lines += ['', f"Solves (event-level, gating): base {base} + retries credited {event_level['retry_solves_credited']} "
                  f"= {reconciled}; observed {observed}; events by class {event_level['n_events_by_class']}; "
                  f"W10 recovered-only identity {w10_identity} (reported); guard {dict(GUARD.counts)}; verify "
                  f'failures {guard_failures}.',
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
    _log(f"solves: base {base} + retries {event_level['retry_solves_credited']} = {reconciled}; observed {observed}; "
         f"W10 identity {w10_identity}; guard {dict(GUARD.counts)} verify_failures={guard_failures}")
    _log(f"reference: reproduces={reference['reproduces']} n_diffs={reference['n_diffs']} "
         f"first={reference['first_differing_field']}; field-table mismatches={table['total_mismatches']}")
    _log(f'GATE_PASS={gate_pass} wall={time.time() - started:.1f}s')
    if not gate_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
