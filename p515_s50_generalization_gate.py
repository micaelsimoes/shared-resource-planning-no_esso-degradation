"""
P5.15 Addendum 36 (task W35, item 3) -- the SRP1 TWO-CYCLE BITWISE GATE for the single-scenario
generalization, under the BASELINE configuration, against the COMMITTED baseline C* trajectory.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 36 ("Prerequisite: generalize the four
single-scenario code paths ...; gate: bitwise two-cycle reproduction on SRP1 after the change");
frozen spec v20 `data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json` item
`3_generalization`; Planner task W35 item 3.

WHAT IS UNDER TEST. Commit "W35 (ii)" changed four code paths that the SRP1 ADMM run executes:
  * `p56a_oracle.common_coordinated_values` / `p515_s41_hull_polish.apply_hull_bounds` /
    `p56a_oracle.per_block_base_objectives` (hull-polish helpers),
  * `shared_resources_planning._get_interface_reporting_detail` and its S31C writer,
  * `p56a_oracle.install_baseline` (new; `load_baseline` untouched),
  * `p515_g_g1_g4_admm_gates.run_admm_arm`'s solve count.
Each is claimed to leave SRP1 behaviour byte-identical. This gate is the solve-bearing test of that
claim on the ADMM path: two cycles at C* under the baseline configuration must reproduce the
committed baseline re-certification's `cycle_trajectory[:2]` and the rows-derived top-level fields
with ZERO diffs.

WHAT THIS GATE DOES NOT COVER, stated rather than implied: the hull polish and the S31C settlement
writer are POST-CERTIFICATION steps and do not run inside a two-cycle arm. Their SRP1 invariance is
established by construction (the legacy single-scenario branch runs verbatim below 2 scenarios) and
by the zero-solve checks `p515_s50_generalization_checks.py` -- in particular the S31C priced
residual reproduced bit for bit on all 864 committed C* period entries. This gate covers
`run_admm_arm`'s report and every production path the ADMM cycle itself touches.

REFERENCE (committed): `W32.BASELINE_REFERENCE` (BY IMPORT),
    data/SRP1/Results/P515S47/campaign_s47_recert/evals/070f833e1e318f85_c_star/g_s39_D.json
`cycle_trajectory[:2]` plus the rows-derived top-level fields; its sha256 is pinned and checked
before anything runs (`W10.check_preconditions`).

EVERYTHING IS W10's / W32's, BY IMPORT, exactly as the committed W32 and W33 gates use it:
  * the armed guard `W10.GUARD` = SolveProfileGuard(p514_n_instrumented_cstar.PERMITTED), installed
    at W10's import and verified EXACTLY;
  * `W10.check_preconditions` (which REFUSES while `.p515_s44_campaign.lock` or `.p515_g_gate.lock`
    exists, and while any forbidden harness process is alive), `W10.capture_path_checklist`,
    `W10.derive_solves_from_case_file` (51 per cycle; base 51 x (2 + 1) = 153),
    `W10.run_arm(arm, 'on', ...)`, `W10.compare_on_against_committed_reference` (GATING here) and
    `W10.trajectory_field_table`;
  * `W32.BASELINE_REFERENCE` and `W32._load_recert_declaration`.
ONE DECLARED SUBSTITUTION, the same W32 and W33 made and no other: `W10.AA_REFERENCE` :=
`W32.BASELINE_REFERENCE`, and `p515_s44_campaign_harness._config_hook_factory` (as `W10.run_arm`
resolves it) is wrapped so the spec it receives ALSO declares the recert `ess_ageing_baseline`.
NO flex-price override (W33's arm had one at m = 1.0); NO `release_solution_bookkeeping` (W32's
switch); snapshots 'on' with no assignment. This is the baseline configuration the reference ran.

SOLVE RECONCILIATION (declared before the run, gating) -- the per-EVENT identity of W35 item 1:
    observed == 153 + sum over network-failure events of [recovery_attempted] + [tier2_attempted]
read from the arm's own `network_failures_s39_D.jsonl`. ANY ESSO recovery event or 'indeterminate'
network event makes it unsupported and the gate FAIL. `GUARD.verify` is called with that count,
EXACTLY. `run_admm_arm`'s own (now instance-derived) `solve_profile` is reported beside it and must
agree.

GATE (every item must hold): cycles_run == 2; zero diffs against the reference (trajectory rows and
derived top-level fields); zero per-field mismatches; the event-level solve identity; the arm's own
instance-derived solve profile agreeing with the gate's; GUARD.verify exactly; no blocked solver
call; shared FrozenSMOPF tree untouched; the ESS-ageing declaration verified in the child; and the
W35 code actually present in the modules that ran.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s50_generalization_gate.py \\
        > data/SRP1/Results/P515S50/generalization_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S50/generalization_gate/{gate.json, gate.md,
manifest_sha256.json, arm/}
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
"""

import copy
import inspect
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

import p515_s44_scale_measurement as S44  # noqa: E402 -- the per-EVENT rule, BY IMPORT
import p56a_oracle as O  # noqa: E402 -- presence checks only
import p515_s41_hull_polish as HP  # noqa: E402 -- presence checks only
import shared_resources_planning as srp  # noqa: E402

STAGE = ('P5.15 Addendum 36 W35 item 3 -- two-cycle bitwise gate: the single-scenario '
         'generalization, baseline configuration, vs committed baseline C*')
SCHEMA = 'p515_s50_generalization_gate_v1'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 36', 'Planner task W35 item 3',
             'data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json item 3_generalization']
ARM = 's50gengate'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S50', 'generalization_gate')
# The four W35-touched modules, plus the files W33's gate required clean.
EXTRA_CLEAN_FILES = ('network.py', 'solver_parameters.py', 'model_construction_helpers.py',
                     'shared_resources_planning.py', 'p56a_oracle.py', 'p515_s41_hull_polish.py',
                     'p515_g_g1_g4_admm_gates.py', 'p515_s44_scale_measurement.py',
                     'p515_s45_snapshot_off_two_cycle_gate.py', 'p515_s49_memory_fix_gate.py',
                     'p515_s44_campaign_harness.py', 'shared_energy_storage_data.py',
                     'data/SRP1/SharedESS/SRP1_ESS_Params.json')
# W10's list stops at p515_s45_; the campaigns that have run since must be caught too.
EXTRA_FORBIDDEN = ('p515_s46_', 'p515_s47_', 'p515_s48_', 'p515_s49_', 'p515_s50_')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W35-gate] {msg}', flush=True)


def w35_code_presence():
    """The W35 changes must be present in the modules that actually ran -- asserted from the LIVE
    module objects, not from the repository text, so a stale import cannot pass this gate."""
    arm_src = inspect.getsource(G.run_admm_arm)
    ccv_src = inspect.getsource(O.common_coordinated_values)
    hull_src = inspect.getsource(HP.apply_hull_bounds)
    detail_src = inspect.getsource(srp._get_interface_reporting_detail)
    return {
        'run_admm_arm_derives_solves_from_the_instance': (
            "guard.counts['permitted_solve'] == 51 * len(rows) + 51" not in arm_src
            and '_solves_per_cycle = (1 + _n_dso)' in arm_src),
        'run_admm_arm_credits_every_attempted_retry': (
            "int(bool(b.get('recovery_attempted'))) + int(bool(b.get('tier2_attempted')))" in arm_src),
        'oracle_install_baseline_present': callable(getattr(O, 'install_baseline', None)),
        'oracle_load_baseline_srp1_path_unchanged': (
            "SharedResourcesPlanning('data/SRP1', 'SRP1.json')" in inspect.getsource(O.load_baseline)
            and 'checksum != CANONICAL_CHECKSUM' in inspect.getsource(O.load_baseline)),
        'common_coordinated_values_branches_on_scenarios': 'expectation_mode' in ccv_src,
        'common_coordinated_values_legacy_reads_intact': (
            't_p = float(pe.value(t_model.pc_adn[dn, 0, 0, p]))' in ccv_src),
        'apply_hull_bounds_branches_on_scenarios': 'expectation_mode' in hull_src,
        'apply_hull_bounds_legacy_bounds_intact': (
            'tv = t_model.vmag_sqr[adn_idx, 0, 0, p]' in hull_src),
        'per_block_base_objectives_uses_the_expectation': (
            '_scenario_expectation(model, network, term)' in inspect.getsource(O.per_block_base_objectives)),
        'interface_detail_uses_the_expected_price': '_expected_market_price(' in detail_src,
        'interface_detail_supplies_the_priced_residual': (
            "'priced_interface_residual_expected_mu'" in detail_src),
        's31c_writer_reads_the_supplied_priced_residual': (
            "period_detail['priced_interface_residual_expected_mu']" in inspect.getsource(G._s31c_interface_detail)),
        'scale_harness_per_event_rule_present': callable(getattr(S44, 'event_level_solve_reconciliation', None)),
    }


def main():
    out_root = os.path.join(REPO, OUT_REL)
    started = time.time()
    _log(STAGE)
    _log(f'git HEAD {W10._git(["rev-parse", "HEAD"])}; output root {W10._rel(out_root)}')
    os.environ.update(H.THREAD_CAP_ENV)

    # declared substitution: the baseline C* reference, pinned (W32's constant, by import)
    W10.AA_REFERENCE = dict(W32.BASELINE_REFERENCE)
    W10.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(W10.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + EXTRA_FORBIDDEN
    failures, instance = W10.check_preconditions(out_root)
    status = W10._git(['status', '--porcelain', '--'] + list(EXTRA_CLEAN_FILES))
    if status.strip():
        failures.append(f'files not clean in git:\n{status}')
    declaration, recert_spec = W32._load_recert_declaration()
    ess_pin = declaration['ess_params_file']
    ess_sha_now = CP._sha256_file(os.path.join(REPO, ess_pin['path']))
    if ess_sha_now != ess_pin['sha256']:
        failures.append(f'ESS parameters file sha256 {ess_sha_now} != recert pin {ess_pin["sha256"]}')
    presence = w35_code_presence()
    for name, ok in presence.items():
        if not ok:
            failures.append(f'W35 code not present in the live module: {name}')
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)

    checklist = W10.capture_path_checklist()
    checklist['network_failure_events_carry_attempt_flags'] = (
        "'recovery_attempted'" in inspect.getsource(G._new_network_event)
        and "'tier2_attempted'" in inspect.getsource(G._new_network_event))
    checklist['w10_run_arm_resolves_hook_factory_via_H'] = 'H._config_hook_factory(' in inspect.getsource(W10.run_arm)
    checklist['scale_harness_event_rule_callable'] = callable(S44.event_level_solve_reconciliation)
    if not all(checklist.values()):
        raise SystemExit(f'capture-path checklist failed: {[k for k, v in checklist.items() if not v]}')
    _log(f'preconditions passed; capture-path checklist {len(checklist)} items, all true; '
         f'W35 presence checks {len(presence)} items, all true')

    declared = W10.derive_solves_from_case_file()
    base = declared['declared_base_solves_per_arm']
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle; base {base} "
         f'(cap {W10.CAP} + initialization); gate on observed == base + every attempted retry '
         f'(recovered or not) and GUARD.verify(that) exactly')

    # declared substitution: the recert ESS-ageing declaration, around the child's own hook
    orig_factory = H._config_hook_factory
    captured = {}

    def factory_with_declaration(spec_like, holder, overrides=None, **kwargs):
        spec2 = copy.deepcopy(spec_like)
        spec2['configuration'].update(copy.deepcopy(declaration))
        captured['holder'] = holder
        return orig_factory(spec2, holder, overrides=overrides, **kwargs)

    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')
    counters = W10.CloneCaptureCounters()
    H._config_hook_factory = factory_with_declaration
    expected_counts = {}
    arm_dir = os.path.join(out_root, 'arm')
    try:
        with counters.installed():
            report, summary = W10.run_arm(ARM, 'on', arm_dir, counters, expected_counts)
            counters.phase = 'post'
    finally:
        H._config_hook_factory = orig_factory

    _log(f"arm: cycles={summary['cycles_run']} recourse={summary['recourse']} "
         f"gross={summary['gross_operational_cost']} solves={(summary['solve_profile'] or {}).get('observed')} "
         f"failures={summary['network_failures_summary']}")

    observed = ((summary['solve_profile'] or {}).get('observed') or {}).get('permitted_solve')
    classes = (summary['network_failures_summary'] or {}).get('classes') or {}
    tier1, tier2 = classes.get('recovered_tier1', 0), classes.get('recovered_tier2', 0)
    recovered_only = base + tier1 + 2 * tier2
    event_level = S44.event_level_solve_reconciliation(report, base)
    reconciled = event_level['expected']
    guard_failures = GUARD.verify(reconciled) if reconciled is not None else ['event-level reconciliation unsupported']
    arm_profile = summary.get('solve_profile') or {}
    solve = {
        'declared_before_the_run': declared, 'base_declared': base,
        'event_level_reconciliation_GATING': event_level,
        'recovered_only_identity_reported': {'recovered_tier1': tier1, 'recovered_tier2': tier2,
                                             'expected': recovered_only, 'holds': observed == recovered_only},
        'observed_in_arm': observed, 'identity_holds': reconciled is not None and observed == reconciled,
        'expected_block_counts_from_planning': expected_counts.get(ARM),
        'planning_derivation_agrees_with_case_file': ((expected_counts.get(ARM) or {}).get('solves_per_cycle')
                                                      == declared['solves_per_cycle']),
        'arm_own_instance_derived_profile': {k: arm_profile.get(k) for k in (
            'solves_per_cycle', 'derivation', 'n_dso', 'n_years', 'n_days', 'n_esso_nodes', 'rounds',
            'base_solves', 'retry_solves_credited', 'reconciliation_supported', 'expected_solves',
            'identity', 'identity_holds', 'recovered_only_identity')},
        'arm_profile_agrees_with_gate': (arm_profile.get('solves_per_cycle') == declared['solves_per_cycle']
                                         and arm_profile.get('base_solves') == base
                                         and arm_profile.get('expected_solves') == reconciled
                                         and arm_profile.get('identity_holds') is True),
        'process_guard': {'permitted_call_sites': [list(p) for p in W10.N.PERMITTED],
                          'counts': dict(GUARD.counts), 'verify_exactly': reconciled,
                          'verify_failures': guard_failures,
                          'no_solves_outside_the_arm': GUARD.counts['permitted_solve'] == observed},
    }

    reference = W10.compare_on_against_committed_reference(report)
    reference['gating'] = True
    reference['caveat'] = ('GATING here (W35 item 3); cap 2 vs the reference cap 500 -- W10/W20/W32/W33 '
                           '(committed) reproduced committed C* rows with 0 diffs at cap 2')
    with open(os.path.join(REPO, W10.AA_REFERENCE['report'])) as handle:
        ref_rows = (json.load(handle).get('cycle_trajectory') or [])[:W10.CAP]
    table = W10.trajectory_field_table(ref_rows, report.get('cycle_trajectory') or [])
    ess_ageing_checklist = (report.get('rule_eleven_checklist') or {}).get('w21_ess_ageing_baseline') or {}

    gate_items = {
        'ran_two_cycles': summary['cycles_run'] == W10.CAP,
        'reproduces_committed_reference_zero_diffs': reference['reproduces'] and reference['n_diffs'] == 0,
        'trajectory_fields_identical_to_reference': (table['total_mismatches'] == 0 and table['rows_length_equal']),
        'solve_reconciliation_event_level_identity_holds': solve['identity_holds'],
        'arm_instance_derived_solve_profile_agrees': solve['arm_profile_agrees_with_gate'],
        'planning_derivation_agrees_with_case_file': solve['planning_derivation_agrees_with_case_file'],
        'process_guard_verified_exactly': not guard_failures,
        'no_solves_outside_the_arm': solve['process_guard']['no_solves_outside_the_arm'],
        'no_blocked_solver_calls': GUARD.counts['blocked_solve'] == 0 and GUARD.counts['blocked_exec'] == 0,
        'shared_frozen_smopf_untouched': not (report.get('shared_frozen_smopf_modified')
                                              or report.get('shared_frozen_smopf_new_files')),
        'ess_ageing_declaration_verified_in_child': (all((ess_ageing_checklist.get('checks') or {}).values())
                                                     and bool(ess_ageing_checklist.get('checks'))
                                                     and ess_ageing_checklist.get('readback_all_match') is True),
        'w35_code_present_in_the_modules_that_ran': all(presence.values()),
    }
    gate_pass = all(gate_items.values())

    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'w10_module_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s45_snapshot_off_two_cycle_gate.py')),
        'w32_module_sha256': CP._sha256_file(os.path.join(REPO, 'p515_s49_memory_fix_gate.py')),
        'w35_module_sha256': {name: CP._sha256_file(os.path.join(REPO, name)) for name in (
            'p56a_oracle.py', 'p515_s41_hull_polish.py', 'p515_g_g1_g4_admm_gates.py',
            'shared_resources_planning.py', 'p515_s44_scale_measurement.py')},
        'w35_code_presence_asserted_before_run': presence,
        'harness_sha256': CP._sha256_file(H.HARNESS_PATH),
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': CP._sha256_file(H.CASE_FILE),
                      'anderson_acceleration_loaded': instance['case_file_anderson_acceleration_loaded']},
        'ess_params_file': {'path': ess_pin['path'], 'sha256': ess_sha_now, 'recert_pin': ess_pin},
        'declared_substitutions': {
            '1_reference': W32.BASELINE_REFERENCE,
            '2_hook_declaration': {'recert_spec': recert_spec, 'declaration_injected': declaration,
                                   'note': 'no flex-price override, no release_solution_bookkeeping'}},
        'instance': {'candidate_label': 'C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025)',
                     'candidate_canonical': instance['candidate_canonical'],
                     'candidate_key': instance['candidate_key'],
                     'candidate_key_pin_matches': instance['candidate_key'] == W10.C_STAR_KEY_PIN},
        'configuration': {'arm': ARM, 'cap': W10.CAP, 'arm_label': W10.ARM_LABEL,
                          'snapshots': 'on (no assignment)', 'flex_price_multiplier': None,
                          'release_solution_bookkeeping': 'default (off)', 'model_variant': None,
                          'overrides': {}, 'apply_rho': False, 'full_diagnostics_in_rows': True,
                          'thread_caps': dict(H.THREAD_CAP_ENV),
                          'nlp_solver_path': H._resolve_solver_path_from_dotenv()},
        'objective_convention': ('gross_operational_cost is the settlement-excluded gross cost; '
                                 'net_operational_recourse differs by the terminal salvage credit'),
        'capture_path_checklist_asserted_before_run': checklist,
        'arm_summary': summary,
        'clone_capture_counters_informational': counters.buckets,
        'solve_profile': solve,
        'reference_comparison_GATING': reference,
        'trajectory_field_table_vs_reference': table,
        'not_covered_by_this_gate': (
            'the hull polish and the S31C settlement writer are post-certification steps and do not '
            'run inside a two-cycle arm; their SRP1 invariance rests on the legacy branch running '
            'verbatim below 2 scenarios and on p515_s50_generalization_checks.py (864 committed C* '
            'period entries reproduced bit for bit)'),
        'gate_items': gate_items, 'gate_pass': gate_pass,
        'wall_clock_s': time.time() - started,
    }
    gate_path = os.path.join(out_root, 'gate.json')
    W10._refuse_overwrite(gate_path)
    with open(gate_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)

    lines = [f'# {STAGE}', '', f"git HEAD at run {payload['git_head_at_run']}; GATE_PASS = **{gate_pass}**", '',
             f"Instance: C* key {instance['candidate_key']} (pin matches: "
             f"{payload['instance']['candidate_key_pin_matches']}).",
             f"Reference: {W32.BASELINE_REFERENCE['report']} (sha256 {W32.BASELINE_REFERENCE['sha256']}); "
             'cycle_trajectory[:2].',
             'Configuration: baseline -- no flex-price override, no release_solution_bookkeeping, '
             'snapshots on, apply_rho False, cap 2.', '', '| gate item | holds |', '|---|---|']
    lines += [f'| {k} | {v} |' for k, v in gate_items.items()]
    lines += ['', f"Solves (event-level, gating): base {base} + retries credited "
                  f"{event_level['retry_solves_credited']} = {reconciled}; observed {observed}; events by class "
                  f"{event_level['n_events_by_class']}; recovered-only identity {recovered_only} (reported); "
                  f"guard {dict(GUARD.counts)}; verify failures {guard_failures}.",
              f"Arm's own instance-derived profile: {solve['arm_own_instance_derived_profile']}",
              f"Reference comparison: n_diffs {reference['n_diffs']}, first {reference['first_differing_field']}; "
              f"field-table mismatches {table['total_mismatches']} over {table['n_fields']} fields.",
              f"Arm: cycles {summary['cycles_run']}, gross_operational_cost "
              f"{summary['gross_operational_cost']} (gross convention), recourse {summary['recourse']}, "
              f"terminal step / threshold {summary['rule_ten_terminal_step_over_threshold']}.",
              f"Peak RSS (ru_maxrss, bytes): {summary['peak_rss']}", '',
              f"NOT covered by this gate: {payload['not_covered_by_this_gate']}"]
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

    for key, value in gate_items.items():
        _log(f'   {key}: {value}')
    _log(f'wrote {W10._rel(gate_path)}, {W10._rel(md_path)}, {W10._rel(manifest_path)}')
    _log(f'GATE_PASS={gate_pass} in {time.time() - started:.0f} s')
    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())
