"""
P5.15 Addendum 27 item 1 -- the case-file-alone AA re-verification campaign at
C*, through the campaign harness (`p515_s44_campaign_harness.py`), NO overrides.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 item 1; frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`
(`configuration.oracle`, `configuration.reverification_gate`).

Campaign id `s45_reverify_aa_c_star`, root
`data/SRP1/Results/P515S45/reverify_aa_c_star/`: ONE evaluation,
  c_star_aa_case_file : C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9,
                        investment year 2025), configuration = the case file
                        alone (`configuration.case_file_anderson_acceleration`
                        declared = {enabled: True, memory: 5, regularization:
                        1e-10, reject_policy: 'keep_memory'}), no overrides,
                        no post-certification; cap 500, 10 consecutive
                        all-pass cycles, concurrency 1.
The reference it is to be compared with (NOT compared here; the comparison is
the Planner's separate gate): the committed AA keep_memory evaluation at C*,
`campaign_s44_aa_variant/evals/837fc982565dbba3_c_star_aa_keep_memory` (107
cycles, 650,982,939.9389359), pinned below by sha256.

Two modes, both attached, both streams captured, never detached:
  --freeze          ZERO SOLVES. Campaign preconditions (locks, live processes,
                    clean git for the production files AND this script, root
                    absent); the production loader reads the case file's AA
                    dict == the declaration; the reference record hash-verified;
                    the spec frozen by the harness's `freeze_campaign_spec` into
                    the campaign root and validated. No lock, no evaluation.
  --run --spec-sha256 <sha>
                    loads THAT frozen spec (the root must hold only it),
                    re-checks the preconditions, the spec's harness / case-file
                    sha256 against the files on disk and the spec contents;
                    takes the campaign lock; evaluates; writes
                    campaign_results.json and campaign_manifest_sha256.json.
The parent never solves: SolveProfileGuard(permitted=()) is installed before
any model import and verified at exactly 0 (both modes).

EXACT COMMANDS (repo root):
  freeze (zero solves):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_reverify_aa_c_star.py --freeze \\
        > data/SRP1/Results/P515S45/reverify_aa_c_star_freeze_launch.log 2>&1
  run (Planner):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s45_reverify_aa_c_star.py --run \\
        --spec-sha256 <sha256 printed by --freeze> \\
        > data/SRP1/Results/P515S45/reverify_aa_c_star_launch.log 2>&1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

PARENT_GUARD = SolveProfileGuard(permitted=(), label='P5.15-S45 AA C* re-verification parent (never solves)').install()

import p515_s44_campaign_harness as H  # noqa: E402

CAMPAIGN_ID = 's45_reverify_aa_c_star'
CAMPAIGN_ROOT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S45', 'reverify_aa_c_star')
LABEL = 'c_star_aa_case_file'
C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
CASE_FILE_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
CAP = 500
CONCURRENCY = 1
REQUIRED_CONSECUTIVE_CYCLES = 10
D_C_STAR_KEY = '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c'
AA_OVERRIDE_C_STAR_EVAL_KEY = '837fc982565dbba32241ed21a46da14984395006e3dbb4b03d5f72b9b72c50a7'
_P44 = os.path.join('data', 'SRP1', 'Results', 'P515S44')
# The committed AA keep_memory evaluation at C* (same pin as p515_s44_selection_aa_campaign.AA_C_STAR).
AA_C_STAR_REFERENCE = {
    'eval_dir': os.path.join(_P44, 'campaign_s44_aa_variant', 'evals', '837fc982565dbba3_c_star_aa_keep_memory'),
    'evaluation_record_sha256': '6e90d4424df3405b0cc066bba1f57c28666a1524e5cdcc62890c6e6a74e69a4f',
    'component_levels_terminal_sha256': '8452f6261b1bf1fe84d4dfd8a177fa8f4145ade4c042a4904a29d4483a745782',
    'certification_cycle': 107, 'certified_cost': 650982939.9389359,
    'campaign_spec': os.path.join(_P44, 'campaign_s44_aa_variant', 'campaign_spec_s44_aa_variant_f90618e8.json'),
}
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 item 1 (AA-on into the case file; re-verification)',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json configuration.oracle, '
    'configuration.reverification_gate',
]
REVERIFICATION_GATE = ('spec v15 5feefd7b configuration.reverification_gate: a case-file-alone C* evaluation through '
                       'the campaign harness (no overrides) reproduces the committed AA C* evaluation '
                       '(campaign_s44_aa_variant, 107 cycles, 650,982,939.9389359) bitwise on every numeric '
                       'trajectory field and cost; tie-order in the recourse-jump top-10 classified with '
                       'p515_s44_tie_classifier; provenance non-gating. NOT evaluated by this script.')
EXTRA_CLEAN_FILES = (os.path.basename(__file__), 'p515_s45_case_file_aa_check.py')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _check_case_file_loads_to_declaration():
    """The PRODUCTION loader (no model import) reads the case file's AA dict == the declaration."""
    from planning_parameters import PlanningParameters
    params = PlanningParameters()
    params.read_parameters_from_file(H.CASE_FILE)
    loaded = params.admm.anderson_acceleration
    return {'loaded': loaded, 'equals_declaration': loaded == CASE_FILE_AA}


def _check_reference():
    ref_dir = os.path.join(REPO, AA_C_STAR_REFERENCE['eval_dir'])
    got = {k: H.sha256_file(os.path.join(ref_dir, f)) for k, f in (
        ('evaluation_record_sha256', 'evaluation_record.json'),
        ('component_levels_terminal_sha256', 'component_levels_terminal.json'))}
    with open(os.path.join(ref_dir, 'evaluation_record.json')) as handle:
        rec = json.load(handle)
    return {'hashes_match_pin': all(got[k] == AA_C_STAR_REFERENCE[k] for k in got),
            'certification_cycle': rec.get('certification_cycle'), 'certified_cost': rec.get('certified_cost'),
            'record_matches_pin': (rec.get('certification_cycle') == AA_C_STAR_REFERENCE['certification_cycle']
                                   and rec.get('certified_cost') == AA_C_STAR_REFERENCE['certified_cost']),
            'eval_key': rec.get('eval_key')}


def _validate_spec(spec):
    """The frozen spec is exactly this campaign: one C* entry, declaration, no overrides, no post-cert."""
    entries = spec['candidates']
    e = entries[0] if len(entries) == 1 else {}
    checks = {
        'campaign_id': spec.get('campaign_id') == CAMPAIGN_ID,
        'one_entry': len(entries) == 1,
        'label': e.get('label') == LABEL,
        'canonical_is_c_star': e.get('canonical') == H.canonical_candidate(C_STAR),
        'candidate_key_is_c_star': e.get('key') == D_C_STAR_KEY,
        'declaration': spec['configuration'].get('case_file_anderson_acceleration') == CASE_FILE_AA,
        'no_campaign_overrides': spec['configuration'].get('overrides') == {},
        'no_entry_overrides': e.get('overrides') == {},
        'no_post_certification': e.get('post_certification') is None,
        'effective_aa_is_declaration': e.get('effective_anderson_acceleration') == CASE_FILE_AA,
        'eval_key_recomputes': e.get('eval_key') == H.evaluation_key(D_C_STAR_KEY, {}, case_file_aa=CASE_FILE_AA),
        'eval_key_not_D': e.get('eval_key') != D_C_STAR_KEY,
        'eval_key_not_aa_override': e.get('eval_key') != AA_OVERRIDE_C_STAR_EVAL_KEY,
        'cap': spec.get('cap') == CAP,
        'concurrency': spec.get('concurrency') == CONCURRENCY,
        'required_consecutive_cycles': spec.get('required_consecutive_cycles') == REQUIRED_CONSECUTIVE_CYCLES,
        'arm_label_s39_D': spec['configuration'].get('arm_label') == 's39_D',
        'not_a_stub_spec': not spec.get('extra', {}).get('test_only_stub'),
    }
    return checks


def freeze(started):
    failures = H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
    case_file = _check_case_file_loads_to_declaration()
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    ref = _check_reference()
    if not (ref['hashes_match_pin'] and ref['record_matches_pin']):
        failures.append(f'AA C* reference does not match its pin: {ref}')
    if failures:
        for f in failures:
            _log(f'[S45-REVERIFY FREEZE PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    spec_path, spec_sha, spec = H.freeze_campaign_spec(
        CAMPAIGN_ROOT, CAMPAIGN_ID, [(LABEL, C_STAR)],
        configuration={'name': 'case file alone: AA keep_memory adopted in data/SRP1/SRP1_params.json (Addendum 27)',
                       'arm_label': 's39_D', 'overrides': {}, 'case_file_anderson_acceleration': dict(CASE_FILE_AA),
                       'note': ('no overrides; the case file carries AA (declared, checked by the child hook); '
                                'num_max_iters := cap (run_admm_arm always sets it)')},
        cap=CAP, concurrency=CONCURRENCY, authority=AUTHORITY,
        required_consecutive_cycles=REQUIRED_CONSECUTIVE_CYCLES,
        extra={'campaign_script': os.path.basename(__file__),
               'campaign_script_sha256': H.sha256_file(os.path.abspath(__file__)),
               'mode': 'full',
               'reference_aa_c_star_evaluation': dict(AA_C_STAR_REFERENCE),
               'reverification_gate': REVERIFICATION_GATE})
    checks = _validate_spec(spec)
    guard_failures = PARENT_GUARD.verify(0)
    _log(f'[S45-REVERIFY] frozen campaign spec: {os.path.relpath(spec_path, REPO)} sha256={spec_sha}')
    _log(f"[S45-REVERIFY] eval_key={spec['candidates'][0]['eval_key']} eval_dir={spec['candidates'][0]['eval_dir']}")
    _log(f'[S45-REVERIFY] spec checks: {checks}')
    _log(f'[S45-REVERIFY] case file AA (production loader): {case_file}; reference: {ref}')
    _log(f'[S45-REVERIFY] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures} '
         f'wall={time.time() - started:.1f}s')
    ok = all(checks.values()) and not guard_failures
    _log(f'[S45-REVERIFY] freeze {"OK" if ok else "NOT OK"}')
    PARENT_GUARD.uninstall()
    if not ok:
        sys.exit(1)


def run(started, spec_sha256):
    spec_path, spec = H.load_frozen_spec(CAMPAIGN_ROOT, spec_sha256)
    failures = [f for f in H.check_campaign_preconditions(CAMPAIGN_ROOT, extra_clean_files=EXTRA_CLEAN_FILES)
                if f != f'campaign root already exists (write-once): {CAMPAIGN_ROOT}']
    root_contents = sorted(os.listdir(CAMPAIGN_ROOT))
    if root_contents != [os.path.basename(spec_path)]:
        failures.append(f'campaign root must hold only the frozen spec; holds {root_contents}')
    checks = _validate_spec(spec)
    failures += [f'spec check failed: {k}' for k, v in checks.items() if not v]
    if spec['harness']['sha256'] != H.sha256_file(H.HARNESS_PATH):
        failures.append('harness sha256 differs from the frozen spec')
    if spec['configuration']['case_file_sha256'] != H.sha256_file(H.CASE_FILE):
        failures.append('case file sha256 differs from the frozen spec')
    if spec['extra'].get('campaign_script_sha256') != H.sha256_file(os.path.abspath(__file__)):
        failures.append('this script sha256 differs from the frozen spec')
    case_file = _check_case_file_loads_to_declaration()
    if not case_file['equals_declaration']:
        failures.append(f"case file AA {case_file['loaded']} != declaration {CASE_FILE_AA}")
    if failures:
        for f in failures:
            _log(f'[S45-REVERIFY PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    head = H._git(['rev-parse', 'HEAD'])
    _log(f'[S45-REVERIFY] preconditions passed; spec {os.path.relpath(spec_path, REPO)} sha256={spec_sha256}; '
         f'git HEAD {head} (spec frozen at {spec["git_head"]})')
    lock = H.acquire_campaign_lock(CAMPAIGN_ID, spec_sha256)
    _log(f'[S45-REVERIFY] campaign lock acquired: {lock}')
    try:
        ctx = H.CampaignContext(CAMPAIGN_ROOT, spec_path, spec_sha256, spec, log=_log)
        records = H.evaluate([LABEL], ctx)
        batch_info = getattr(H.evaluate, 'last_batch_info', {})
    finally:
        H.release_campaign_lock(expected_pid=os.getpid())
        _log('[S45-REVERIFY] campaign lock released')
    rec = records[0] or {}
    guard_failures = PARENT_GUARD.verify(0)
    summary = {k: rec.get(k) for k in (
        'candidate_label', 'candidate_key', 'eval_key', 'eval_dir', 'status', 'barrier', 'barrier_cause',
        'certification_cycle', 'cycles_run', 'certified_cost', 'terminal_gross_operational_cost',
        'terminal_net_operational_recourse', 'first_pass_cycle_per_channel', 'terminal_ratios_per_channel',
        'rule_ten', 'evaluation_overrides_effective', 'overrides_applied_in_child',
        'anderson_acceleration_effective_in_child', 'case_file_sha256_in_child', 'configuration_checks_in_child',
        'aa_per_cycle', 'local_solve_failures', 'peak_rss', 'wall_time_s', 'parent_view', 'campaign_spec_sha256')}
    summary['bar'] = (rec.get('bar') or {}).get('value')
    summary['objective_convention'] = 'gross_operational_cost, settlement-excluded (certified_cost)'
    results = {
        'stage': 'P5.15 Addendum 27 item 1 -- case-file-alone AA re-verification at C* (campaign harness)',
        'authority': AUTHORITY, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head, 'campaign_spec_path': os.path.relpath(spec_path, REPO),
        'campaign_spec_sha256': spec_sha256, 'evaluation': summary,
        'reference_aa_c_star_evaluation': AA_C_STAR_REFERENCE, 'reverification_gate': REVERIFICATION_GATE,
        'batch_info': batch_info,
        'parent_solve_profile_guard': {'counts': dict(PARENT_GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_results.json'), results)
    manifest = {}
    for r_, _dirs, files in os.walk(CAMPAIGN_ROOT):
        for fname in sorted(files):
            fpath = os.path.join(r_, fname)
            manifest[os.path.relpath(fpath, REPO)] = H.sha256_file(fpath)
    H._write_once_json(os.path.join(CAMPAIGN_ROOT, 'campaign_manifest_sha256.json'), manifest)
    PARENT_GUARD.uninstall()
    _log(f"[S45-REVERIFY] {LABEL}: status={summary['status']} cycles={summary['cycles_run']} "
         f"cert_cycle={summary['certification_cycle']} cost={summary['certified_cost']} "
         f"aa_actions={(summary['aa_per_cycle'] or {}).get('action_counts')}")
    _log(f'[S45-REVERIFY] parent guard {PARENT_GUARD.counts} verify0_failures={guard_failures}')
    ok = (not guard_failures and summary['status'] in ('certified', 'not_certified')
          and (summary['parent_view'] or {}).get('exit_code') == 0)
    _log(f'[S45-REVERIFY] {"OK" if ok else "NOT OK"}')
    if not ok:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true', help='zero solves: freeze + validate the spec only')
    mode.add_argument('--run', action='store_true', help='evaluate the frozen spec named by --spec-sha256')
    parser.add_argument('--spec-sha256', default=None)
    args = parser.parse_args()
    started = time.time()
    if args.freeze:
        if args.spec_sha256:
            parser.error('--spec-sha256 is for --run only')
        freeze(started)
    else:
        if not args.spec_sha256:
            parser.error('--run requires --spec-sha256')
        run(started, args.spec_sha256)


if __name__ == '__main__':
    main()
