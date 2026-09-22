"""
P5.15 Addendum 35 (task W35, item 1) -- ZERO-SOLVE checks for the scale harness's solve identity
and for the unrecovered-failure policy.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 35; frozen spec v20
`data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json`, item `1_harness`
("scale script's solve identity aligned with the per-event rule (as the ladder harness does);
explicit unrecovered-failure policy for long runs recorded and verified against production
behaviour: continue with the last iterate, count the event, and NO unrecovered failure inside the
10 certifying cycles").

WHAT IS CHECKED (all from COMMITTED artifacts; nothing is re-run, no model is built, no solve):

(1) REPLICA == LADDER. `p515_s44_scale_measurement.event_level_solve_reconciliation` is a replica
    of `p515_s49_flex_ladder_campaign.solve_reconciliation`'s rule (replicated because the ladder
    module installs `SolveProfileGuard(permitted=())` at import, and the W33 gate module installs
    W10's armed guard the same way -- either import inside the scale harness's SOLVING cycle child
    would install a foreign guard). Both are run on the SAME inputs and the credited retry count
    and the expected total must agree exactly.

(2) THE NEW IDENTITY HOLDS WHERE THE OLD ONE DID. On every committed run whose recovered-only
    identity `observed == base + tier1 + 2 x tier2` holds, the per-event identity
    `observed == base + sum[recovery_attempted] + [tier2_attempted]` must hold too and give the
    same number.

(3) AND ALSO ON THE UNRECOVERED CASE. On the paper-scale re-measure of 78a9b230
    (`paper_cycle_snapoff_memfix_r1`), where one DSO block stayed unrecovered after two attempted
    retries, the recovered-only identity must FAIL (166 vs 168 observed) and the per-event identity
    must HOLD (168).

(4) UNRECOVERED-FAILURE POLICY vs PRODUCTION. The three policy clauses are checked against the
    production SOURCE (read with `inspect.getsource`, never executed): continue with the last
    iterate; count the event; no unrecovered failure inside the certifying cycles. Each clause
    records the file:line it was verified at. The policy note is
    `data/SRP1/Results/P515S50/unrecovered_failure_policy.md`.

ZERO SOLVES: `SolveProfileGuard(permitted=())` is installed BEFORE any production import and
`verify(0)`-ed; the ladder module's own import-time `PARENT_GUARD` is verified at 0 as well.

EXACT COMMAND (repo root, canonical interpreter, attached, both streams captured):
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s50_harness_identity_checks.py \
      > data/SRP1/Results/P515S50/harness_identity_checks_launch.log 2>&1
`--suffix <s>` redirects the output to `harness_identity_checks_<s>/` -- used to RE-VERIFY on a later
tree without overwriting an artifact an earlier report cites (the policy citations are line numbers,
read from the working tree by `inspect` at run time, so they move when production moves).
OUTPUT (write-once): data/SRP1/Results/P515S50/harness_identity_checks[_<suffix>]/harness_identity_checks.json
Exit 0 when every check passes, 1 otherwise.
"""

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W35 item 1 harness identity checks (never solves)').install()

import p515_s44_scale_measurement as S44  # noqa: E402 -- no import-time side effects
import p515_s49_flex_ladder_campaign as LADDER  # noqa: E402 -- installs its own PARENT_GUARD at import

STAGE = 'P5.15 Addendum 35 W35 item 1 -- scale-harness per-event solve identity + unrecovered-failure policy'
SCHEMA = 'p515_s50_harness_identity_checks_v1'
OUT_REL = os.path.join('data', 'SRP1', 'Results', 'P515S50', 'harness_identity_checks')

_S44M = os.path.join('data', 'SRP1', 'Results', 'P515S44', 'scale_measurement')
_S47 = os.path.join('data', 'SRP1', 'Results', 'P515S47', 'campaign_s47_recert', 'evals')
_S49 = os.path.join('data', 'SRP1', 'Results', 'P515S49', 'campaign_s49_flex_ladder', 'evals')

# Scale-harness cycle records: `cycle_record.json` carries its own declared base and observed count.
SCALE_RECORDS = (
    ('paper_cycle_snapoff_memfix_r1 (78a9b230, ONE UNRECOVERED BLOCK)',
     os.path.join(_S44M, 'paper_cycle_snapoff_memfix_r1', 'cycle_record.json')),
    ('paper_cycle_snapoff_r2', os.path.join(_S44M, 'paper_cycle_snapoff_r2', 'cycle_record.json')),
    ('paper_cycle_snapoff_r1', os.path.join(_S44M, 'paper_cycle_snapoff_r1', 'cycle_record.json')),
    ('srp1_cycle_snapoff_r1', os.path.join(_S44M, 'srp1_cycle_snapoff_r1', 'cycle_record.json')),
    ('srp1_calibration_r2', os.path.join(_S44M, 'srp1_calibration_r2', 'cycle_record.json')),
    ('srp1_calibration', os.path.join(_S44M, 'srp1_calibration', 'cycle_record.json')),
)

# Campaign arm reports: `g_s39_D.json` carries observed, cycles_run and the failure summary.
# SRP1: 51 solves per cycle, base = 51 x (cycles_run + 1) -- the initialization round plus the cycles.
ARM_REPORTS = (
    ('s47_recert 070f833e1e318f85_c_star', os.path.join(_S47, '070f833e1e318f85_c_star', 'g_s39_D.json'), 51),
    ('s47_recert bd504ecf5a288d44_n7_4h_e1', os.path.join(_S47, 'bd504ecf5a288d44_n7_4h_e1', 'g_s39_D.json'), 51),
)
for _name in sorted(os.listdir(os.path.join(REPO, _S49))) if os.path.isdir(os.path.join(REPO, _S49)) else []:
    ARM_REPORTS = ARM_REPORTS + ((f's49_flex_ladder {_name}', os.path.join(_S49, _name, 'g_s39_D.json'), 51),)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W35-item1] {msg}', flush=True)


def _recovered_only(base, summary):
    classes = (summary or {}).get('classes') or {}
    return base + classes.get('recovered_tier1', 0) + 2 * classes.get('recovered_tier2', 0)


def _ladder_on(summary, observed, base, per_cycle, cycles):
    """The LADDER's own function on an equivalent record (`p515_s49_flex_ladder_campaign.
    solve_reconciliation(rec, per_cycle)`; it derives base = per_cycle x (cycles + 1))."""
    rec = {'solve_profile': {'observed': {'permitted_solve': observed}},
           'network_failures_summary': summary, 'cycles_run': cycles}
    out = LADDER.solve_reconciliation(rec, per_cycle)
    out['base_agrees_with_declared'] = (out['base'] == base)
    return out


def check_one(label, summary, observed, base, per_cycle, cycles):
    """Runs the replica and the ladder's own function on the same inputs and compares."""
    replica = S44.event_level_solve_reconciliation({'network_failures_summary': summary}, base)
    ladder = _ladder_on(summary, observed, base, per_cycle, cycles)
    recovered_only = _recovered_only(base, summary)
    return {
        'label': label, 'observed': observed, 'declared_base': base,
        'per_event_expected': replica['expected'], 'per_event_supported': replica['supported'],
        'per_event_retries_credited': replica['retry_solves_credited'],
        'per_event_events_by_class': replica['n_events_by_class'],
        'per_event_identity_holds': (replica['expected'] is not None and observed == replica['expected']),
        'recovered_only_expected': recovered_only,
        'recovered_only_identity_holds': observed == recovered_only,
        'ladder_expected': ladder['expected'], 'ladder_retries_credited': ladder['retries_credited'],
        'ladder_supported': ladder['supported'], 'ladder_base': ladder['base'],
        'ladder_base_agrees_with_declared': ladder['base_agrees_with_declared'],
        'replica_agrees_with_ladder': (replica['retry_solves_credited'] == ladder['retries_credited']
                                       and replica['supported'] == ladder['supported']
                                       and replica['expected'] == ladder['expected']),
        'events_file': replica['events_file'],
    }


def unrecovered_failure_policy_verification():
    """Clause-by-clause verification against the production SOURCE. Zero solves, zero execution of
    the cited code: `inspect.getsource` + `inspect.getsourcelines` only."""
    import network as network_module
    import shared_resources_planning as srp

    def where(module, obj_name, needle, start_hint=None):
        obj = getattr(module, obj_name)
        src, first = inspect.getsourcelines(obj)
        for offset, line in enumerate(src):
            if needle in line:
                return {'file': os.path.basename(inspect.getsourcefile(obj)),
                        'line': first + offset, 'text': line.strip()}
        return {'file': os.path.basename(inspect.getsourcefile(obj)), 'line': None,
                'text': None, 'needle_not_found': needle}

    run_smopf_src = inspect.getsource(network_module._run_smopf)
    attempt_src = inspect.getsource(network_module._run_smopf_solver_attempt)
    admm_src = inspect.getsource(srp._run_operational_planning)
    ok_src = inspect.getsource(srp._admm_local_solves_succeeded)

    clauses = {
        'clause_1_continue_with_the_last_iterate': {
            'statement': ('an unrecovered local network solve neither stops the run nor overwrites the '
                          "block's values; the model keeps the previous cycle's iterate"),
            'solver_called_with_load_solutions_false': {
                'holds': 'load_solutions=False' in attempt_src,
                'at': where(network_module, '_run_smopf_solver_attempt', 'load_solutions=False')},
            'solution_loaded_only_on_success': {
                'holds': ('if solver_result_succeeded(result):' in run_smopf_src
                          and 'model.solutions.load_from(result)' in run_smopf_src),
                'at': where(network_module, '_run_smopf', 'model.solutions.load_from(result)'),
                'guarded_by': where(network_module, '_run_smopf', 'if solver_result_succeeded(result):')},
            'multiplier_suffixes_restored_on_failed_tier2': {
                'holds': '_restore_multiplier_suffixes(model, multiplier_snapshot)' in run_smopf_src,
                'at': where(network_module, '_run_smopf', 'if not solver_result_succeeded(tier2_result):')},
            'admm_loop_breaks_only_on_convergence': {
                'holds': admm_src.count('            break') == 1 and 'if convergence:' in admm_src,
                'at': where(srp, '_run_operational_planning', 'if convergence:'),
                'n_break_statements_in_the_cycle_loop': admm_src.count('            break')},
        },
        'clause_2_count_the_event': {
            'statement': 'the failure is written as ONE event with its attempt flags and its retries are credited',
            'event_carries_attempt_flags': None,     # filled below from the gates module
            'per_event_rule_is_what_the_harness_gates_on': {
                'holds': ('event_level_solve_reconciliation' in inspect.getsource(S44.child_cycle)
                          and 'guard.verify(reconciled_solves)' in inspect.getsource(S44.child_cycle)),
                'at': where(S44, 'child_cycle', '_event_level = event_level_solve_reconciliation(')},
        },
        'clause_3_no_unrecovered_failure_inside_the_certifying_cycles': {
            'statement': ('an unrecovered failure resets the consecutive-converged counter, so a certified '
                          'point cannot contain one in its last `minimum_consecutive_converged_cycles` cycles'),
            'local_solves_ok_is_false_for_the_cycle': {
                'holds': 'def _admm_local_solves_succeeded' in ok_src and 'return False' in ok_src,
                'at': where(srp, '_admm_local_solves_succeeded', 'def _admm_local_solves_succeeded')},
            'cycle_convergence_requires_local_solves_ok': {
                'holds': 'cycle_convergence = boyd_all_pass and local_solves_ok' in admm_src,
                'at': where(srp, '_run_operational_planning',
                            'cycle_convergence = boyd_all_pass and local_solves_ok')},
            'counter_reset_to_zero': {
                'holds': 'consecutive_converged_cycles = 0' in admm_src,
                'at': where(srp, '_run_operational_planning', 'consecutive_converged_cycles = 0')},
            'certification_needs_the_streak': {
                'holds': ('convergence = (consecutive_converged_cycles >= '
                          'admm_parameters.minimum_consecutive_converged_cycles)') in admm_src,
                'at': where(srp, '_run_operational_planning',
                            'convergence = (consecutive_converged_cycles >=')},
            'aa_step_skipped': {'holds': 'aa_state.skip_on_failure(iter)' in admm_src,
                                'at': where(srp, '_run_operational_planning', 'aa_state.skip_on_failure(iter)')},
            'rho_not_updated': {'holds': 'allow_update=local_solves_ok' in admm_src,
                                'at': where(srp, '_run_operational_planning', 'allow_update=local_solves_ok')},
            'residual_convergence_forced_false': {
                'holds': 'residual_convergence = False' in admm_src,
                'at': where(srp, '_run_operational_planning', 'residual_convergence = False')},
        },
    }

    import p515_g_g1_g4_admm_gates as G
    new_event_src = inspect.getsource(G._new_network_event)
    clauses['clause_2_count_the_event']['event_carries_attempt_flags'] = {
        'holds': "'recovery_attempted'" in new_event_src and "'tier2_attempted'" in new_event_src,
        'at': where(G, '_new_network_event', "'recovery_attempted'")}

    def _all_hold(node):
        out = []
        for key, value in node.items():
            if isinstance(value, dict) and 'holds' in value:
                out.append(bool(value['holds']))
            elif isinstance(value, dict):
                out.extend(_all_hold(value))
        return out

    holds = _all_hold(clauses)
    return {'production_already_complies': all(holds), 'n_sub_checks': len(holds), 'clauses': clauses}


def main():
    parser = argparse.ArgumentParser()
    # The line:column citations below are produced by `inspect` from the WORKING TREE at run time,
    # so a re-run on a later tree yields different (correct) line numbers. `--suffix` sends that
    # re-run to its OWN output directory instead of overwriting the earlier one (CLAUDE.md: never
    # re-run a harness onto an artifact a committed report cites).
    parser.add_argument('--suffix', default='', help='output directory suffix for a re-run on a later tree')
    args = parser.parse_args()
    suffix = ('_' + args.suffix.strip('_')) if args.suffix else ''
    out_dir = os.path.join(REPO, OUT_REL + suffix)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'harness_identity_checks.json')
    if os.path.exists(path):
        raise SystemExit(f'refusing to overwrite {path}')
    _log(STAGE)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()

    rows, evidence = [], {}
    for label, rel in SCALE_RECORDS:
        full = os.path.join(REPO, rel)
        if not os.path.isfile(full):
            # No cycle record: either the run was BUILD ONLY (no `cycle_exit_code.txt`) or the
            # cycle child was aborted by the memory watchdog (exit 97, `watchdog_abort_cycle.json`),
            # which writes no cycle record by design. Recorded, not silently skipped.
            run_dir = os.path.dirname(full)
            exit_path = os.path.join(run_dir, 'cycle_exit_code.txt')
            exit_code = open(exit_path).read().strip() if os.path.isfile(exit_path) else None
            rows.append({'label': label, 'missing': rel, 'cycle_exit_code': exit_code,
                         'watchdog_abort': os.path.isfile(os.path.join(run_dir, 'watchdog_abort_cycle.json')),
                         'reason': ('memory-watchdog abort of the cycle child (exit 97): no cycle record by design'
                                    if exit_code == '97' else
                                    ('build-only run: no cycle child was launched' if exit_code is None
                                     else f'no cycle record; cycle child exit {exit_code}'))})
            continue
        rec = json.load(open(full))
        guard = rec.get('guard') or {}
        summary = rec.get('network_failures_summary')
        observed = guard.get('observed_solves', (guard.get('counts') or {}).get('permitted_solve'))
        base = guard.get('declared_base_solves')
        prof = rec.get('declared_solve_profile') or {}
        rows.append(check_one(label, summary, observed, base,
                              prof.get('solves_per_cycle'), prof.get('cap')))
        evidence[rel] = _sha256(full)
        if summary and summary.get('path') and os.path.isfile(os.path.join(REPO, summary['path'])):
            evidence[summary['path']] = _sha256(os.path.join(REPO, summary['path']))

    for label, rel, per_cycle in ARM_REPORTS:
        full = os.path.join(REPO, rel)
        if not os.path.isfile(full):
            rows.append({'label': label, 'missing': rel})
            continue
        rep = json.load(open(full))
        observed = ((rep.get('solve_profile') or {}).get('observed') or {}).get('permitted_solve')
        cycles = rep.get('cycles_run')
        base = per_cycle * (cycles + 1)
        rows.append(check_one(label, rep.get('network_failures_summary'), observed, base, per_cycle, cycles))
        evidence[rel] = _sha256(full)
        nfs = rep.get('network_failures_summary') or {}
        if nfs.get('path') and os.path.isfile(os.path.join(REPO, nfs['path'])):
            evidence[nfs['path']] = _sha256(os.path.join(REPO, nfs['path']))

    checked = [r for r in rows if 'missing' not in r]
    unrecovered_cases = [r for r in checked
                         if (r['per_event_events_by_class'].get('unrecovered', 0)
                             + r['per_event_events_by_class'].get('not_attempted', 0)) > 0]
    checks = {
        'replica_agrees_with_ladder_everywhere': all(r['replica_agrees_with_ladder'] for r in checked),
        'per_event_identity_holds_everywhere': all(r['per_event_identity_holds'] for r in checked),
        'per_event_holds_wherever_recovered_only_did': all(
            r['per_event_identity_holds'] for r in checked if r['recovered_only_identity_holds']),
        'the_two_agree_when_no_unrecovered_block': all(
            r['per_event_expected'] == r['recovered_only_expected'] for r in checked
            if (r['per_event_events_by_class'].get('unrecovered', 0)
                + r['per_event_events_by_class'].get('not_attempted', 0)) == 0),
        'unrecovered_case_present_in_the_evidence': len(unrecovered_cases) >= 1,
        'unrecovered_case_old_rule_fails_new_rule_holds': all(
            (not r['recovered_only_identity_holds']) and r['per_event_identity_holds']
            for r in unrecovered_cases),
        'n_records_checked': len(checked),
    }
    policy = unrecovered_failure_policy_verification()
    checks['unrecovered_failure_policy_verified_against_production'] = policy['production_already_complies']

    guard_failures = GUARD.verify(0)
    ladder_guard_failures = LADDER.PARENT_GUARD.verify(0)
    checks['zero_solves'] = not guard_failures and not ladder_guard_failures

    all_ok = all(v for k, v in checks.items() if isinstance(v, bool))
    payload = {
        'schema': SCHEMA, 'stage': STAGE,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 35',
                      'data/SRP1/Results/P515S50/frozen_s50_spec_v20_69bccc62.json item 1_harness',
                      'Planner task W35 item 1'],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head': head,
        'rule_gated_on': S44.EVENT_LEVEL_IDENTITY,
        'replicated_from': ('p515_s49_flex_ladder_campaign.solve_reconciliation / '
                            'p515_s49_flex_price_gate.event_level_reconciliation; replicated because both '
                            'modules install a SolveProfileGuard at module import'),
        'records': rows, 'checks': checks, 'all_ok': all_ok,
        'unrecovered_failure_policy': policy,
        'guard': {'permitted': [], 'counts': dict(GUARD.counts), 'declared_solves': 0,
                  'verify_failures': guard_failures,
                  'ladder_parent_guard_counts': dict(LADDER.PARENT_GUARD.counts),
                  'ladder_parent_guard_verify_failures': ladder_guard_failures},
        'evidence_sha256': evidence,
    }
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    for row in rows:
        if 'missing' in row:
            _log(f"NO CYCLE RECORD {row['label']}: {row['missing']} -- {row.get('reason')}")
            continue
        _log(f"{row['label']}: observed {row['observed']} | per-event {row['per_event_expected']} "
             f"({'HOLDS' if row['per_event_identity_holds'] else 'FAILS'}) | recovered-only "
             f"{row['recovered_only_expected']} "
             f"({'holds' if row['recovered_only_identity_holds'] else 'FAILS'}) | "
             f"classes {row['per_event_events_by_class']}")
    for key, value in checks.items():
        _log(f'  {key}: {value}')
    _log(f'wrote {os.path.relpath(path, REPO)} (sha256 {_sha256(path)})')
    _log(f'ALL_OK={all_ok}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
