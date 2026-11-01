"""
P5.15 Addendum 27 item 5(a), task W11 -- the SRP1 SNAPSHOT-OFF **FAILURE-BRANCH**
GATE: the same two-arm (`'off'` vs `'on'`) bitwise gate as W10, run at a cap that
reaches REAL local-solve failures and the cycle at which the `'on'` arm actually
WRITES a FrozenSMOPF snapshot.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 item 5(a); Planner task W11
("snapshots-off failure branch, under a bitwise gate"; THIS TASK SOLVES); W9
(commit b1272593, the `'off'` mode) and W10 (commit 31f1ba6e, the two-cycle gate
this file reuses BY IMPORT).

WHAT W10 LEFT OPEN
------------------
W10 proved `'off'` bitwise identical to `'on'` over TWO cycles (24/48 clones/
captures ON, 0/0 OFF, zero genuine diffs). But over those two cycles NO block
failed and NO snapshot was written, so neither the `'off'` warning+JSON-marker
branch nor the `'lightweight'` ON-DEMAND snapshot-write branch (clone the pristine
base, replay the capture, pickle) ever executed with real solves. This gate runs
long enough for both of those code paths to be reachable.

THE CAP, AND WHY -- READ FROM THE COMMITTED AA-ON C* REFERENCE **BEFORE** RUNNING
---------------------------------------------------------------------------------
Reference: `data/SRP1/Results/P515S45/reverify_aa_c_star/evals/
3e741dac72c9e1bc_c_star_aa_case_file/` (cap 500, certified at cycle 107).
  * `network_failures_s39_D.jsonl`: 43 network blocks, FIRST at **cycle 3**
    (DSO node 5, case33_1, 2030 Winter, primary termination `maxIterations`,
    class `recovered_tier1`). Classes over the whole run: 41 `recovered_tier1`,
    2 `recovered_tier2`, **0 `unrecovered`**, 0 `not_attempted`, 0 `indeterminate`.
  * `frozen_snapshots_s39_D.jsonl` / `results/FrozenSMOPF/`: exactly **2** `.pkl`,
    both at **cycle 7** and both `matched_success` comparators
    (`DSO_node7_case33_2_2025_Autumn_cycle7`, `TSO_case9_2025_Summer_cycle7`).

WHAT THE PRODUCTION CODE SAYS ABOUT THOSE TWO FACTS (read, not assumed)
-----------------------------------------------------------------------
`network._run_smopf` performs tier-1 (cold) and tier-2 (cold, mu_strategy=adaptive)
recovery INTERNALLY and returns the FINAL result. Every snapshot decision in
`shared_resources_planning` -- the legacy callback, the lightweight branch's
`needs_failure_snapshot`, and the `'off'` branch's warning+marker loop -- tests
`_solver_result_succeeded(res[...])` on that FINAL result. Therefore:
  (i)  a *recovered* failure writes NO `.pkl` in `'on'` and NO marker in `'off'`;
       only a block that fails through every tier does;
  (ii) DSO failure snapshots exist for **node 7 only** (`snapshot_callback` is
       `None` for every other node, and `dso_pristine_base` is built for node 7
       only); TSO failure snapshots exist for every TSO block;
  (iii) at C* AA-on the reference has ZERO finally-failed blocks over 107 cycles,
       so the marker/warning sub-branch is **not reachable at this instance**,
       while the cycle-7 comparator path IS.
This is a statement about the code and about the committed reference, and it is
declared here BEFORE the run so that the gate items below can be read as
falsifiable predictions rather than as post-hoc rationalization.

CAP = 8. The Planner asked for "the smallest cap that includes the first failure
with one cycle of margin"; that alone would be cap 4. Cap 4 cannot satisfy the
Planner's own gate item (2) -- "in the ON arm at least one snapshot .pkl is written
and n_frozen_snapshots > 0" -- because the only snapshot-writing cycle at C* is
cycle 7. Cap 8 is the smallest cap that includes BOTH the first failure (cycle 3,
five cycles of margin) AND the snapshot-writing cycle (cycle 7, one cycle of
margin). The conflict between the two instructions, and its resolution, is
recorded in `gate.json` under `cap_choice`.

THE ARMS
--------
Identical to W10 in every respect except the cap, the campaign id and the output
root: both arms are `p515_g_g1_g4_admm_gates.run_admm_arm` (BY IMPORT) at candidate
C*, arm label `s39_D`, `apply_rho=False`, `full_diagnostics_in_rows=True`, with the
campaign child's own configuration hook (`p515_s44_campaign_harness.
_config_hook_factory`, case-file AA), wrapped by `p515_s44_scale_measurement.
snapshot_hook_wrapper` with `'on'` (no assignment at all) / `'off'` (both modes set
and read back). W10's own `run_arm`, `CloneCaptureCounters`, `check_preconditions`,
`capture_path_checklist`, `classify`, `trajectory_field_table` and committed-
reference comparison are used BY IMPORT -- this file changes the cap, the campaign
id, the output root and the gate items, and NOTHING else. W10's module-level
`SolveProfileGuard` is armed at import of that module, i.e. before any call here.

THE GATE (every item must hold; 'vacuous' items are flagged, never hidden)
--------------------------------------------------------------------------
(1) BITWISE IDENTITY, same classification as W10
    - all compared files present; zero genuine diffs over `CP.ARTIFACT_FILES`,
      `CP.SIDECAR_JSONL_FILES`, `aa_per_cycle.jsonl`, `per_cycle_record.jsonl`;
    - every `H.RECORD_TRAJECTORY_FIELDS` field bitwise equal cycle by cycle;
    - both arms ran the full cap.
(2) ON ARM REALLY WROTE A SNAPSHOT
    - `.pkl` count > 0 under the ON arm's own `results/FrozenSMOPF/`;
    - `network_failures_summary.n_frozen_snapshots` > 0 and equal to that count;
    - the number of `[DEBUG][FROZEN SMOPF] Saved ... pre-solve block` lines in the
      ON arm's captured stdout equals it (the write is evidenced from two
      independent capture paths).
(3) OFF ARM TOOK THE OFF BRANCH AND WROTE NOTHING
    - exactly 0 clones and 0 captures (pass-through counters, whole run);
    - exactly 0 `.pkl` and `n_frozen_snapshots == 0`;
    - markers == the number of blocks that failed through EVERY tier and are
      snapshot-eligible (TSO, or DSO node 7), computed from the OFF arm's own
      failure rows; warning lines in the OFF stdout == markers. At C* this
      identity is expected to be 0 == 0: it is GATED but flagged `vacuous`, and
      the marker branch is then reported as NOT EXERCISED;
    - the NON-vacuous proof that the off branch ran where `'on'` wrote snapshots:
      the OFF stdout contains the cycle-7 `comparator snapshot SKIPPED` line for
      the TSO and for DSO node=7, and the ON stdout contains none.
(4) FAILURE / RECOVERY ACCOUNTING IDENTICAL
    - `n_blocks`, every class count (tier-1, tier-2, unrecovered, not_attempted,
      indeterminate, recovered) and the ESSO recovery-event count equal between
      arms; the per-block failure rows equal after stripping path-like keys;
    - at least one REAL local-solve failure in EACH arm (else the run did not
      reach what it was launched to reach -- FAIL, and the Planner is told);
    - zero `indeterminate` classifications, so the marker identity above is
      well defined.
(5) SOLVES -- declared in advance, gated on the reconciliation identity
    51 solves/cycle at SRP1 (12 (year, day) blocks x 4 networks + 3 ESSO), base =
    51 x (cap + 1) = 459 per arm (initialization counts as one round), 918 both
    arms; gate on `observed_arm == base + 1*tier1 + 2*tier2` from each arm's OWN
    failure summary, with W10's process-wide `SolveProfileGuard(N.PERMITTED)`
    armed for the whole run and `verify()`-ed EXACTLY against the reconciled
    total (too few fails as loudly as too many).

NOT GATED, REPORTED: ESSO model pickles. W10 established that they are NOT
byte-reproducible (a `ComponentSet` hashes on `id()`), so their sha256 is recorded
as informational only and never gated. Arm ON vs the committed AA-on reference over
the first 8 cycles is likewise reported and never gated (the reference ran at cap
500; cap-independence with AA on is UNVERIFIED).

PRECONDITIONS: W10's `check_preconditions` verbatim (no lock, no forbidden live
process, fresh output root, production files clean in git, case-file AA == the
declaration, C* candidate key == the pinned key, committed reference sha256 ==
pin), then W10's rule-eleven capture-path checklist plus this file's own additions
(marker writer, snapshot writer, stdout capture path, frozen-snapshot scanner).
Then the legacy run lock.

EXACT LAUNCH COMMAND (repo root; attached, alone, both streams captured)
------------------------------------------------------------------------
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \
        p515_s45_snapshot_off_failure_gate.py \
        > data/SRP1/Results/P515S45/snapshot_off_failure_gate_launch.log 2>&1

OUTPUT (all NEW; refuses to overwrite)
    data/SRP1/Results/P515S45/snapshot_off_failure_gate/gate.json
    data/SRP1/Results/P515S45/snapshot_off_failure_gate/manifest_sha256.json
    data/SRP1/Results/P515S45/snapshot_off_failure_gate/{on,off}/   (arm artifacts)
Exit 0 on PASS, 1 on FAIL or a precondition refusal.
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

# Importing W10 installs ITS module-level SolveProfileGuard (p513_solve_profile_guard),
# i.e. the guard is armed before any call this file makes. It is reused rather than
# installed a second time: two nested installs would double-count every solve.
import p515_s45_snapshot_off_two_cycle_gate as W10  # noqa: E402

GUARD = W10.GUARD
GUARD.label = ('P5.15 Addendum 27 W11 snapshot off/on failure-branch gate '
               '(guard object installed at import of p515_s45_snapshot_off_two_cycle_gate)')

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402
import p514_n_instrumented_cstar as N  # noqa: E402
import shared_resources_planning as srp  # noqa: E402

STAGE = ('P5.15 Addendum 27 item 5(a), W11 -- SRP1 snapshot-off FAILURE-BRANCH bitwise '
         'gate: pristine snapshot clones off vs on, at a cap that reaches real local-solve '
         'failures and the cycle at which a snapshot is written')
SCHEMA = 'p515_s45_snapshot_off_failure_gate_v1'
AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 item 5(a)',
    'Planner task W11 (snapshots-off failure branch, under a bitwise gate; THIS TASK SOLVES)',
    'commit b1272593 (W9: switchable pristine snapshot clones)',
    'commit 31f1ba6e (W10: two-cycle bitwise gate; its harness is reused BY IMPORT)',
]

CAP = 8
CAMPAIGN_ID = 's45_w11_snapoff_fail'
OUT_BASENAME = 'snapshot_off_failure_gate'

# Declared BEFORE the run, from the committed AA-on C* reference (see the docstring).
REFERENCE_FACTS = {
    'reference_eval_dir': os.path.join('data', 'SRP1', 'Results', 'P515S45', 'reverify_aa_c_star',
                                       'evals', '3e741dac72c9e1bc_c_star_aa_case_file'),
    'reference_report_sha256_pin': W10.AA_REFERENCE['sha256'],
    'first_network_failure': {'cycle': 3, 'agent': 'DSO', 'node_id': 5, 'network_name': 'case33_1',
                              'year': '2030', 'day': 'Winter',
                              'primary_termination': 'maxIterations', 'class': 'recovered_tier1'},
    'failure_classes_over_107_cycles': {'recovered_tier1': 41, 'recovered_tier2': 2,
                                        'unrecovered': 0, 'not_attempted': 0, 'indeterminate': 0},
    'first_failure_cycle_by_class': {'recovered_tier1': 3, 'recovered_tier2': 13},
    'frozen_snapshots': [{'cycle': 7, 'label': 'matched_success', 'agent': 'DSO', 'node_id': 7,
                          'network_name': 'case33_2', 'year': 2025, 'day': 'Autumn'},
                         {'cycle': 7, 'label': 'matched_success', 'agent': 'TSO',
                          'network_name': 'case9', 'year': 2025, 'day': 'Summer'}],
}
CAP_CHOICE = {
    'cap': CAP,
    'first_failure_cycle_in_reference': 3,
    'first_snapshot_writing_cycle_in_reference': 7,
    'rule_from_the_task': ('"the smallest cap that includes the first failure with one cycle of '
                           'margin" -> cap 4'),
    'why_not_cap_4': ('gate item (2) requires the ON arm to WRITE a snapshot; production writes a '
                      'snapshot only for a block that fails through EVERY recovery tier (none at C*) '
                      'or for the cycle-7 matched_success comparators -- so no .pkl exists before '
                      'cycle 7 and cap 4 would fail item (2)'),
    'chosen': ('cap 8 = the smallest cap that includes BOTH the first failure (cycle 3, five cycles '
               'of margin) AND the snapshot-writing cycle (cycle 7, one cycle of margin)'),
}

# Declared IN ADVANCE: which failure classes mean "this block failed through every tier",
# i.e. exactly the condition production tests before writing a failure snapshot / marker.
FINAL_FAILURE_CLASSES = ('unrecovered', 'not_attempted')
RECOVERED_CLASSES = ('recovered_tier1', 'recovered_tier2')

# Production's own log lines, as they appear AT RUNTIME in the captured stdout.
LINE_OFF_WARNING = '[WARNING][FROZEN SMOPF] snapshot capture is OFF'
LINE_OFF_TSO_SKIP = '[INFO][FROZEN SMOPF] cycle-7 TSO comparator snapshot SKIPPED'
LINE_OFF_DSO_SKIP = '[INFO][FROZEN SMOPF] cycle-7 DSO node=7 comparator snapshot SKIPPED'
LINE_ON_SAVED = '[DEBUG][FROZEN SMOPF] Saved'
LINE_MARKER_WRITTEN = '[DEBUG][FROZEN SMOPF] Wrote snapshot-skipped marker to'
LINE_WRITER_FAILED = '[WARNING][FROZEN SMOPF] Could not'
# The same lines as they appear in the production SOURCE: some are f-strings, so the
# runtime form ('node=7') is not a substring of the source ('node={node_id}'). The
# capture-path checklist greps these fragments; the counters above match the runtime form.
SOURCE_FRAGMENTS = {
    'off_warning': '[WARNING][FROZEN SMOPF] snapshot capture is OFF ',
    'off_tso_skip': '[INFO][FROZEN SMOPF] cycle-7 TSO comparator snapshot SKIPPED ',
    'off_dso_skip': '[INFO][FROZEN SMOPF] cycle-7 DSO node={node_id} comparator snapshot SKIPPED ',
    'on_saved': '[DEBUG][FROZEN SMOPF] Saved {label} pre-solve block to {filepath}',
    'marker_written': '[DEBUG][FROZEN SMOPF] Wrote snapshot-skipped marker to {filepath}',
    'writer_failed': '[WARNING][FROZEN SMOPF] Could not ',
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W11] {msg}', flush=True)


def _rel(path):
    return os.path.relpath(path, REPO)


def w11_capture_path_checklist():
    """Rule eleven, this stage's ADDITIONS on top of W10's checklist: every quantity
    THIS gate reports must have a capture path, asserted BEFORE the run."""
    import inspect
    checks = dict(W10.capture_path_checklist())
    src_srp = inspect.getsource(srp)
    src_arm = inspect.getsource(G.run_admm_arm)
    checks['w11_marker_writer_exists'] = callable(getattr(srp, '_write_snapshot_skipped_marker', None))
    checks['w11_snapshot_writer_exists'] = callable(getattr(srp, '_save_frozen_network_block', None))
    checks['w11_frozen_snapshot_scanner_exists'] = callable(getattr(G, '_scan_frozen_snapshots', None))
    checks['w11_failure_scanner_exists'] = callable(getattr(G, '_scan_network_failures', None))
    checks['w11_stdout_capture_path_in_report'] = "report['stdout_path']" in src_arm
    checks['w11_failure_summary_in_report'] = "report['network_failures_summary']" in src_arm
    for name, fragment in SOURCE_FRAGMENTS.items():
        checks[f'w11_production_prints_{name}'] = fragment in src_srp
    checks['w11_marker_filename_prefix'] = "snapshot_skipped_" in src_srp
    checks['w11_failure_classes_exist_in_scanner'] = all(
        f"'{c}'" in inspect.getsource(G._scan_network_failures)
        for c in FINAL_FAILURE_CLASSES + RECOVERED_CLASSES)
    missing = sorted(k for k, v in checks.items() if not v)
    if missing:
        raise AssertionError(f'RULE ELEVEN: capture paths missing for this gate: {missing}')
    return checks


def _read_lines(path):
    if not path or not os.path.isfile(path):
        return None
    with open(path, 'r', errors='replace') as handle:
        return handle.read().splitlines()


def _count(lines, needle):
    return sum(1 for line in (lines or []) if needle in line)


def _snapshot_eligible(row):
    """Exactly the blocks for which production would write a failure snapshot (`'on'`)
    or a skipped marker (`'off'`): every TSO block, and DSO node 7 only."""
    agent = row.get('agent')
    if agent == 'TSO':
        return True
    return agent == 'DSO' and row.get('node_id') == 7


def failure_accounting(rows, summary):
    classes = (summary or {}).get('classes') or {}
    final_failed = [r for r in rows if r.get('class') in FINAL_FAILURE_CLASSES]
    eligible_final_failed = [r for r in final_failed if _snapshot_eligible(r)]
    by_cycle = {}
    for r in rows:
        key = str(r.get('cycle'))
        by_cycle.setdefault(key, []).append(
            {'agent': r.get('agent'), 'node_id': r.get('node_id'),
             'network_name': r.get('network_name'), 'year': r.get('year'), 'day': r.get('day'),
             'class': r.get('class'), 'primary_termination': r.get('primary_termination'),
             'recovery_termination': r.get('recovery_termination'),
             'tier2_attempted': r.get('tier2_attempted'), 'termination': r.get('termination')})
    return {
        'n_blocks_rows': len(rows),
        'n_blocks_summary': (summary or {}).get('n_blocks'),
        'classes': classes,
        'n_esso_recovery_events': (summary or {}).get('n_esso_recovery_events'),
        'n_frozen_snapshots_summary': (summary or {}).get('n_frozen_snapshots'),
        'first_failure_cycle': min((r.get('cycle') for r in rows), default=None),
        'cycles_with_failures': sorted({r.get('cycle') for r in rows}),
        'n_final_failed_blocks': len(final_failed),
        'n_final_failed_snapshot_eligible': len(eligible_final_failed),
        'final_failed_snapshot_eligible': eligible_final_failed,
        'final_failure_classes_declared': list(FINAL_FAILURE_CLASSES),
        'blocks_by_cycle': by_cycle,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='', help='redirects the output root and the working-dir ids')
    args = parser.parse_args()
    suffix = ('_' + args.suffix.strip('_')) if args.suffix else ''
    out_root = os.path.join(REPO, W10._P45, f'{OUT_BASENAME}{suffix}')
    started = time.time()

    _log(STAGE)
    _log(f'git HEAD {W10._git(["rev-parse", "HEAD"])}; output root {_rel(out_root)}')
    _log(f'CAP={CAP}: {CAP_CHOICE["chosen"]}')
    _log(f'reference first failure: {REFERENCE_FACTS["first_network_failure"]}')
    _log(f'reference snapshots: {REFERENCE_FACTS["frozen_snapshots"]}')

    # W10's cap / campaign id are module globals read at call time by its `run_arm` and
    # `derive_solves_from_case_file`; this is the ONLY way this file differs from W10's
    # run configuration. Recorded in gate.json under `reused_w10_module`.
    W10.CAP = CAP
    W10.CAMPAIGN_ID = CAMPAIGN_ID

    env_before = {k: os.environ.get(k) for k in H.THREAD_CAP_ENV}
    os.environ.update(H.THREAD_CAP_ENV)
    env_after = {k: os.environ.get(k) for k in H.THREAD_CAP_ENV}
    _log(f'thread caps: before={env_before} after={env_after}')

    failures, instance = W10.check_preconditions(out_root)
    if failures:
        for f in failures:
            _log(f'[PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    checklist = w11_capture_path_checklist()
    _log(f'preconditions passed; capture-path checklist: {len(checklist)} items, all true')
    _log(f"instance: candidate_key={instance['candidate_key']} canonical={instance['candidate_canonical']}")

    declared = W10.derive_solves_from_case_file()
    assert declared['cap'] == CAP, declared
    _log(f"DECLARED BEFORE THE RUN: {declared['solves_per_cycle']} solves/cycle; base = "
         f"{declared['declared_base_solves_per_arm']} per arm, "
         f"{declared['declared_base_solves_both_arms']} both arms; identity {declared['identity']}")

    G._acquire_exclusive_run_lock()
    _log('legacy run lock acquired (.p515_g_gate.lock)')

    out_dirs = {'on': os.path.join(out_root, 'on'), 'off': os.path.join(out_root, 'off')}
    counters = W10.CloneCaptureCounters()
    expected_block_counts_holder = {}
    reports, summaries = {}, {}
    with counters.installed():
        for arm, snapshots in (('on', 'on'), ('off', 'off')):
            _log(f'--- ARM {arm.upper()}: run_admm_arm cap {CAP}, snapshots={snapshots!r} ---')
            report, summary = W10.run_arm(arm, snapshots, out_dirs[arm], counters,
                                          expected_block_counts_holder)
            reports[arm], summaries[arm] = report, summary
            bucket = counters.buckets.get(f'arm_{arm}', {})
            _log(f"arm {arm}: cycles={summary['cycles_run']} recourse={summary['recourse']} "
                 f"gross={summary['gross_operational_cost']} "
                 f"clones={bucket.get('clone')} captures={bucket.get('capture')} "
                 f"solves={(summary['solve_profile'] or {}).get('observed', {}).get('permitted_solve')} "
                 f"failures={summary['network_failures_summary']} "
                 f"snapshots_dir={summary['frozen_smopf_inventory']} "
                 f"wall={summary['wall_clock_s']['measured_here']:.1f}s")
        counters.phase = 'post'

    # ---------------- clone / capture ----------------
    on_bucket = counters.buckets.get('arm_on', {'clone': 0, 'capture': 0})
    off_bucket = counters.buckets.get('arm_off', {'clone': 0, 'capture': 0})
    clone_capture = {
        'buckets': counters.buckets,
        'arm_on': {'clone': on_bucket.get('clone', 0), 'capture': on_bucket.get('capture', 0)},
        'arm_off': {'clone': off_bucket.get('clone', 0), 'capture': off_bucket.get('capture', 0)},
        'gate_on_positive': on_bucket.get('clone', 0) > 0 and on_bucket.get('capture', 0) > 0,
        'gate_off_zero': off_bucket.get('clone', 0) == 0 and off_bucket.get('capture', 0) == 0,
    }
    n_yd = declared['n_year_day_blocks']
    n_pkl_on = summaries['on']['frozen_smopf_n_pkl']
    clone_capture['reconciliation_arm_on'] = {
        'expected_pristine_base_clones': 2 * n_yd,
        'note': (f'{n_yd} TSO blocks + {n_yd} node-7 DSO blocks cloned once per '
                 '_run_operational_planning call (_build_pristine_snapshot_bases), plus ONE '
                 'on-demand rebuild clone per snapshot actually written -- the branch W10 could '
                 'not reach'),
        'snapshots_written': n_pkl_on,
        'expected_total_clones': 2 * n_yd + n_pkl_on,
        'observed_clones': on_bucket.get('clone', 0),
        'identity_holds': on_bucket.get('clone', 0) == 2 * n_yd + n_pkl_on,
        'expected_captures': 2 * n_yd * (summaries['on']['cycles_run'] or 0),
        'observed_captures': on_bucket.get('capture', 0),
        'capture_identity_holds': (on_bucket.get('capture', 0)
                                   == 2 * n_yd * (summaries['on']['cycles_run'] or 0)),
        'gating': True,
    }

    # ---------------- solve profile ----------------
    solve = {'declared_before_the_run': declared, 'per_arm': {}, 'method':
             ('base declared in advance; gate on the reconciliation identity with each arm\'s own '
              'recovery counts; the process-wide guard is verify()-ed EXACTLY against the total')}
    reconciled_total = 0
    for arm in ('on', 'off'):
        observed = ((summaries[arm]['solve_profile'] or {}).get('observed') or {}).get('permitted_solve')
        classes = (summaries[arm]['network_failures_summary'] or {}).get('classes') or {}
        tier1, tier2 = classes.get('recovered_tier1', 0), classes.get('recovered_tier2', 0)
        base = declared['declared_base_solves_per_arm']
        expected = base + tier1 + 2 * tier2
        reconciled_total += expected
        solve['per_arm'][arm] = {
            'base_declared': base, 'recovered_tier1': tier1, 'recovered_tier2': tier2,
            'expected_after_reconciliation': expected, 'observed': observed,
            'identity_holds': observed == expected,
            'expected_block_counts_from_planning': expected_block_counts_holder.get(arm),
            'planning_derivation_agrees_with_case_file': (
                (expected_block_counts_holder.get(arm) or {}).get('solves_per_cycle')
                == declared['solves_per_cycle']),
            'arm_inner_guard_counts': (summaries[arm]['solve_profile'] or {}).get('observed'),
        }
    guard_failures = GUARD.verify(reconciled_total)
    solve['process_guard'] = {
        'guard_label': GUARD.label,
        'permitted_call_sites': [list(p) for p in N.PERMITTED],
        'counts': dict(GUARD.counts),
        'reconciled_total_expected': reconciled_total,
        'verify_failures': guard_failures,
        'sum_of_arm_counts': sum(((summaries[a]['solve_profile'] or {}).get('observed') or {}).get(
            'permitted_solve', 0) for a in ('on', 'off')),
    }
    solve['process_guard']['no_solves_outside_the_arms'] = (
        GUARD.counts['permitted_solve'] == solve['process_guard']['sum_of_arm_counts'])
    solve['identity_holds_both_arms'] = all(v['identity_holds'] for v in solve['per_arm'].values())

    # ---------------- bitwise comparison (W10's classification, verbatim) ----------------
    ids_by_arm = {arm: summaries[arm]['working_dir_ids'] for arm in ('on', 'off')}
    token_groups = W10._arm_tokens(out_dirs, ids_by_arm)
    _log('comparing the two arms bitwise ...')
    files = {}
    for fname in W10.GATING_JSON:
        a = CP._load_json(os.path.join(out_dirs['on'], fname))
        b = CP._load_json(os.path.join(out_dirs['off'], fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: on={a is not None} off={b is not None}'}
            _log(f'  {fname}: {files[fname]["error"]}')
            continue
        files[fname] = W10.classify(fname, a, b, token_groups)
        _log(f'  {fname}: {files[fname]["n"]}')
    for fname in W10.GATING_JSONL:
        a = CP._load_jsonl(os.path.join(out_dirs['on'], fname))
        b = CP._load_jsonl(os.path.join(out_dirs['off'], fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: on={a is not None} off={b is not None}'}
            _log(f'  {fname}: {files[fname]["error"]}')
            continue
        entry = W10.classify(fname, a, b, token_groups)
        entry['n_rows'] = {'on': len(a), 'off': len(b)}
        files[fname] = entry
        _log(f'  {fname}: rows={entry["n_rows"]} {entry["n"]}')
        del a, b

    supplementary = {}
    failure_rows = {}
    for fname in W10.SUPPLEMENTARY_JSONL:
        a = CP._load_jsonl(os.path.join(out_dirs['on'], fname))
        b = CP._load_jsonl(os.path.join(out_dirs['off'], fname))
        if a is None or b is None:
            supplementary[fname] = {'error': f'missing: on={a is not None} off={b is not None}'}
            continue
        raw = CP._diff(a, b, fname)
        stripped = CP._diff(W10._strip_keys(a, W10.SUPPLEMENTARY_PATH_LIKE_KEYS),
                            W10._strip_keys(b, W10.SUPPLEMENTARY_PATH_LIKE_KEYS), fname)
        supplementary[fname] = {'n_rows': {'on': len(a), 'off': len(b)}, 'n_raw_diffs': len(raw),
                                'n_diffs_ignoring_path_like_keys': len(stripped),
                                'path_like_keys_ignored_in_that_count': list(
                                    W10.SUPPLEMENTARY_PATH_LIKE_KEYS),
                                'first_diffs_ignoring_path_like_keys': stripped[:20]}
        if fname.startswith('network_failures_'):
            failure_rows = {'on': a, 'off': b}

    pickles = {}
    for arm in ('on', 'off'):
        p = os.path.join(out_dirs[arm], f'esso_models_{W10.ARM_LABEL}.pkl')
        pickles[arm] = CP._sha256_file(p) if os.path.isfile(p) else None
    pickles['sha256_equal'] = pickles['on'] is not None and pickles['on'] == pickles['off']
    pickles['not_gated_because'] = ('W10 established that the ESSO model pickle is not '
                                    'byte-reproducible (a ComponentSet hashes on id()); recorded '
                                    'as informational only')

    totals = {'provenance': 0, 'tie_order': 0, 'genuine': 0}
    for v in files.values():
        for k, n in (v.get('n') or {}).items():
            totals[k] += n
    all_files_present = all('error' not in v for v in files.values())

    trajectory = W10.trajectory_field_table(reports['on'].get('cycle_trajectory') or [],
                                            reports['off'].get('cycle_trajectory') or [])
    reference_comparison = W10.compare_on_against_committed_reference(reports['on'])
    reference_comparison['caveat'] = (
        f'the reference ran at cap 500 and this arm at cap {CAP}; cap-independence with AA on is '
        'UNVERIFIED, so this comparison is reported and not gated')

    # ---------------- failure / snapshot / marker accounting ----------------
    stdout_lines, stdout_counts = {}, {}
    for arm in ('on', 'off'):
        path = reports[arm].get('stdout_path')
        path = os.path.join(REPO, path) if path else None
        lines = _read_lines(path)
        stdout_lines[arm] = lines
        stdout_counts[arm] = {
            'stdout_path': reports[arm].get('stdout_path'),
            'stdout_readable': lines is not None,
            'n_lines': len(lines or []),
            'off_warning_lines': _count(lines, LINE_OFF_WARNING),
            'off_cycle7_tso_skip_lines': _count(lines, LINE_OFF_TSO_SKIP),
            'off_cycle7_dso_node7_skip_lines': _count(lines, LINE_OFF_DSO_SKIP),
            'saved_snapshot_lines': _count(lines, LINE_ON_SAVED),
            'marker_written_lines': _count(lines, LINE_MARKER_WRITTEN),
            'snapshot_writer_failure_lines': _count(lines, LINE_WRITER_FAILED),
            'snapshot_writer_failure_examples': [l for l in (lines or [])
                                                 if LINE_WRITER_FAILED in l][:10],
        }

    accounting = {arm: failure_accounting(failure_rows.get(arm) or [],
                                          summaries[arm]['network_failures_summary'])
                  for arm in ('on', 'off')}
    expected_markers = accounting['off']['n_final_failed_snapshot_eligible']
    markers_off = summaries['off']['frozen_smopf_n_skipped_markers']
    marker_files_off = [f for f in summaries['off']['frozen_smopf_inventory']
                        if f.startswith('snapshot_skipped_')]
    marker_payloads = []
    for fname in marker_files_off:
        fpath = os.path.join(out_dirs['off'], 'results', 'FrozenSMOPF', fname)
        try:
            with open(fpath) as handle:
                marker_payloads.append(json.load(handle))
        except Exception as error:  # noqa: BLE001
            marker_payloads.append({'file': fname, 'read_error': f'{type(error).__name__}: {error}'})

    snapshot_metadata_on = [entry.get('metadata') for entry in
                            (CP._load_jsonl(os.path.join(out_dirs['on'],
                                                         f'frozen_snapshots_{W10.ARM_LABEL}.jsonl')) or [])]

    off_branch = {
        'expected_marker_count_identity': ('markers == blocks that failed through EVERY recovery '
                                           'tier AND are snapshot-eligible (TSO, or DSO node 7); '
                                           f'final-failure classes declared in advance as '
                                           f'{list(FINAL_FAILURE_CLASSES)}'),
        'expected_markers': expected_markers,
        'observed_markers': markers_off,
        'marker_identity_holds': markers_off == expected_markers,
        'marker_branch_exercised': expected_markers > 0,
        'marker_identity_is_vacuous': expected_markers == 0,
        'vacuity_note': ('at C* AA-on every failure recovers (the committed 107-cycle reference has '
                         '0 unrecovered of 43), so no block reaches the marker/warning branch; the '
                         'identity is GATED but 0 == 0 and is therefore NOT evidence that the '
                         'marker writer works'),
        'marker_files': marker_files_off,
        'marker_payloads': marker_payloads,
        'warning_lines_in_off_stdout': stdout_counts['off']['off_warning_lines'],
        'warning_lines_match_markers': (stdout_counts['off']['off_warning_lines'] == expected_markers),
        'marker_written_lines_in_off_stdout': stdout_counts['off']['marker_written_lines'],
        'cycle7_skip_lines': {'tso': stdout_counts['off']['off_cycle7_tso_skip_lines'],
                              'dso_node7': stdout_counts['off']['off_cycle7_dso_node7_skip_lines']},
        'cycle7_skip_lines_in_on_arm': {'tso': stdout_counts['on']['off_cycle7_tso_skip_lines'],
                                        'dso_node7': stdout_counts['on']['off_cycle7_dso_node7_skip_lines']},
        'non_vacuous_proof': ('the OFF arm printed the cycle-7 comparator-SKIPPED line for the TSO '
                              'and for DSO node=7 -- the exact two blocks for which the ON arm wrote '
                              'a .pkl -- and the ON arm printed neither'),
    }
    on_branch = {
        'pkl_count': n_pkl_on,
        'n_frozen_snapshots_reported': (summaries['on']['network_failures_summary'] or {}).get(
            'n_frozen_snapshots'),
        'inventory': summaries['on']['frozen_smopf_inventory'],
        'snapshot_metadata': snapshot_metadata_on,
        'saved_lines_in_on_stdout': stdout_counts['on']['saved_snapshot_lines'],
        'three_capture_paths_agree': (
            n_pkl_on > 0
            and n_pkl_on == (summaries['on']['network_failures_summary'] or {}).get('n_frozen_snapshots')
            and n_pkl_on == stdout_counts['on']['saved_snapshot_lines']),
    }

    cls_on = accounting['on']['classes']
    cls_off = accounting['off']['classes']
    failure_rows_identical = (
        supplementary.get(f'network_failures_{W10.ARM_LABEL}.jsonl', {}).get(
            'n_diffs_ignoring_path_like_keys') == 0)
    accounting_identical = {
        'n_blocks_equal': accounting['on']['n_blocks_summary'] == accounting['off']['n_blocks_summary'],
        'classes_equal': cls_on == cls_off,
        'esso_recovery_events_equal': (accounting['on']['n_esso_recovery_events']
                                       == accounting['off']['n_esso_recovery_events']),
        'failure_rows_equal_ignoring_paths': failure_rows_identical,
        'cycles_with_failures_equal': (accounting['on']['cycles_with_failures']
                                       == accounting['off']['cycles_with_failures']),
        'on': cls_on, 'off': cls_off,
    }

    # ---------------- the gate ----------------
    snapshot_setting_ok = (
        (summaries['on']['snapshot_setting'] or {}).get('took_effect') is True
        and (summaries['on']['snapshot_setting'] or {}).get('tso_mode_after') == 'lightweight'
        and (summaries['on']['snapshot_setting'] or {}).get('dso_mode_after') == 'lightweight'
        and (summaries['off']['snapshot_setting'] or {}).get('took_effect') is True
        and (summaries['off']['snapshot_setting'] or {}).get('tso_mode_after') == 'off'
        and (summaries['off']['snapshot_setting'] or {}).get('dso_mode_after') == 'off')
    n_indeterminate = (cls_on.get('indeterminate', 0), cls_off.get('indeterminate', 0))

    gate_items = {
        # (1) bitwise identity -- same classification as W10
        'both_arms_ran_the_full_cap': (summaries['on']['cycles_run'] == CAP
                                       and summaries['off']['cycles_run'] == CAP),
        'snapshot_setting_verified_both_arms': snapshot_setting_ok,
        'all_compared_files_present': all_files_present,
        'zero_genuine_diffs': totals['genuine'] == 0,
        'trajectory_fields_identical': (trajectory['total_mismatches'] == 0
                                        and trajectory['rows_length_equal']),
        # (2) the ON arm really wrote a snapshot
        'arm_on_wrote_at_least_one_pkl': n_pkl_on > 0,
        'arm_on_n_frozen_snapshots_positive': (
            ((summaries['on']['network_failures_summary'] or {}).get('n_frozen_snapshots') or 0) > 0),
        'arm_on_snapshot_capture_paths_agree': on_branch['three_capture_paths_agree'],
        'arm_on_clones_and_captures_positive': clone_capture['gate_on_positive'],
        'arm_on_clone_identity_holds': clone_capture['reconciliation_arm_on']['identity_holds'],
        'arm_on_capture_identity_holds': clone_capture['reconciliation_arm_on']['capture_identity_holds'],
        # (3) the OFF arm took the off branch and wrote nothing
        'arm_off_clones_and_captures_exactly_zero': clone_capture['gate_off_zero'],
        'arm_off_wrote_no_pkl_snapshot': summaries['off']['frozen_smopf_n_pkl'] == 0,
        'arm_off_n_frozen_snapshots_zero': (
            (summaries['off']['network_failures_summary'] or {}).get('n_frozen_snapshots') == 0),
        'arm_off_marker_identity_holds': off_branch['marker_identity_holds'],
        'arm_off_warning_lines_match_markers': off_branch['warning_lines_match_markers'],
        'arm_off_printed_both_cycle7_skips': (
            off_branch['cycle7_skip_lines']['tso'] == 1
            and off_branch['cycle7_skip_lines']['dso_node7'] == 1),
        'arm_on_printed_no_cycle7_skips': (
            off_branch['cycle7_skip_lines_in_on_arm']['tso'] == 0
            and off_branch['cycle7_skip_lines_in_on_arm']['dso_node7'] == 0),
        'no_snapshot_writer_failure_lines': (stdout_counts['on']['snapshot_writer_failure_lines'] == 0
                                             and stdout_counts['off']['snapshot_writer_failure_lines'] == 0),
        # (4) failure / recovery accounting
        'at_least_one_real_failure_in_each_arm': (accounting['on']['n_blocks_summary'] or 0) > 0
                                                 and (accounting['off']['n_blocks_summary'] or 0) > 0,
        'failure_accounting_identical': all(v for k, v in accounting_identical.items()
                                            if k not in ('on', 'off')),
        'no_indeterminate_failure_classes': n_indeterminate == (0, 0),
        # (5) solves
        'solve_reconciliation_identity_holds': solve['identity_holds_both_arms'],
        'process_guard_verified_exactly': not guard_failures,
        'no_solves_outside_the_arms': solve['process_guard']['no_solves_outside_the_arms'],
        'no_blocked_solver_calls': (GUARD.counts['blocked_solve'] == 0
                                    and GUARD.counts['blocked_exec'] == 0),
        'shared_frozen_smopf_untouched': not any(
            reports[a].get('shared_frozen_smopf_modified') or reports[a].get('shared_frozen_smopf_new_files')
            for a in ('on', 'off')),
    }
    gate_pass = all(gate_items.values())

    payload = {
        'schema': SCHEMA, 'stage': STAGE, 'authority': AUTHORITY, 'timestamp_utc': _utc(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__),
        'script_sha256': CP._sha256_file(os.path.abspath(__file__)),
        'reused_w10_module': {
            'module': 'p515_s45_snapshot_off_two_cycle_gate',
            'sha256': CP._sha256_file(os.path.join(REPO, 'p515_s45_snapshot_off_two_cycle_gate.py')),
            'reused_by_import': ['CloneCaptureCounters', 'check_preconditions',
                                 'capture_path_checklist', 'derive_solves_from_case_file', 'run_arm',
                                 'classify', '_arm_tokens', '_strip_keys', 'trajectory_field_table',
                                 'compare_on_against_committed_reference', 'GATING_JSON',
                                 'GATING_JSONL', 'SUPPLEMENTARY_JSONL',
                                 'PERMITTED_PROVENANCE_SUFFIXES', 'ARM_LABEL', 'GUARD'],
            'overridden_module_globals': {'CAP': CAP, 'CAMPAIGN_ID': CAMPAIGN_ID},
            'w10_file_unmodified': True,
        },
        'harness_sha256': CP._sha256_file(H.HARNESS_PATH),
        'case_file': {'path': H.CASE_FILE_REL, 'sha256': CP._sha256_file(H.CASE_FILE),
                      'anderson_acceleration_loaded': instance['case_file_anderson_acceleration_loaded'],
                      'declaration': W10.CASE_FILE_AA},
        'instance': {'candidate_label': 'C* (0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025)',
                     'candidate_canonical': instance['candidate_canonical'],
                     'candidate_key': instance['candidate_key'],
                     'candidate_key_pin_matches': instance['candidate_key'] == W10.C_STAR_KEY_PIN},
        'cap_choice': CAP_CHOICE,
        'reference_facts_declared_before_the_run': REFERENCE_FACTS,
        'production_reading_declared_before_the_run': {
            'recovery_is_internal_to_network._run_smopf': (
                'tier-1 and tier-2 run inside network._run_smopf, which returns the FINAL result'),
            'snapshot_and_marker_condition': (
                'every snapshot/marker decision tests _solver_result_succeeded on that FINAL result, '
                'so a RECOVERED failure writes neither a .pkl (on) nor a marker (off)'),
            'dso_snapshot_eligibility': 'node 7 only; TSO: every block',
            'consequence_for_this_gate': (
                'at C* the marker/warning sub-branch is not reachable (0 unrecovered in 107 '
                'reference cycles); the reachable snapshot path is the cycle-7 matched_success '
                'comparator, which is why the cap is 8'),
        },
        'configuration': {
            'arm_label': W10.ARM_LABEL, 'cap': CAP,
            'required_consecutive_cycles': W10.REQUIRED_CONSECUTIVE_CYCLES,
            'campaign_id': CAMPAIGN_ID,
            'apply_rho': False, 'full_diagnostics_in_rows': True,
            'configuration_verified_by': ('p515_s44_campaign_harness._config_hook_factory with the '
                                          "case-file AA declaration (the campaign child's own hook)"),
            'the_only_difference_between_arms': ("p515_s44_scale_measurement.snapshot_hook_wrapper's "
                                                 "`snapshots` argument: 'on' (no assignment at all) vs "
                                                 "'off' (both modes set and read back)"),
            'thread_caps': {'before': env_before, 'after': env_after,
                            'source': 'p515_s44_campaign_harness.THREAD_CAP_ENV'},
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
            'nlp_solver_path': H._resolve_solver_path_from_dotenv(),
            'both_arms_in_one_process': True,
        },
        'objective_convention': ('gross_operational_cost is the settlement-excluded gross cost; '
                                 'net_operational_recourse differs by the terminal salvage credit'),
        'capture_path_checklist_asserted_before_run': checklist,
        'preconditions': {'forbidden_live_process_substrings': list(W10.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS),
                          'production_files_checked_clean': list(W10.PRODUCTION_FILES_TO_CHECK_CLEAN)},
        'arms': summaries,
        'clone_capture_counters': clone_capture,
        'solve_profile': solve,
        'comparison_on_vs_off': {
            'comparator': ('p515_s40_clone_capture_preflight._diff + p515_s43_aa_flagoff_gate.'
                           '_classify_diffs + p515_s44_tie_classifier.reclassify_sidecar_diffs, all '
                           'BY IMPORT via p515_s45_snapshot_off_two_cycle_gate.classify'),
            'diff_value_naming': "CP._diff names arm ON's value 'legacy' and arm OFF's value 'lightweight'",
            'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
            'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
            'intentional_difference_suffixes_excluded': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
            'permitted_provenance_suffixes': W10.PERMITTED_PROVENANCE_SUFFIXES,
            'arm_path_normalization_tokens': token_groups,
            'gating_files': list(W10.GATING_JSON + W10.GATING_JSONL),
            'files': files,
            'totals_by_class': totals,
            'all_files_present': all_files_present,
            'supplementary_not_gating': supplementary,
            'not_compared': W10.NOT_COMPARED,
            'esso_models_pickle_sha256_informational': pickles,
        },
        'trajectory_field_table': trajectory,
        'failure_accounting': accounting,
        'failure_accounting_identical': accounting_identical,
        'stdout_line_counts': stdout_counts,
        'arm_on_snapshot_branch': on_branch,
        'arm_off_branch': off_branch,
        'arm_on_vs_committed_aa_reference_NON_GATING': reference_comparison,
        'gate_items': gate_items,
        'gate_pass': gate_pass,
        'wall_clock_s': time.time() - started,
    }

    os.makedirs(out_root, exist_ok=True)
    gate_path = os.path.join(out_root, 'gate.json')
    W10._refuse_overwrite(gate_path)
    with open(gate_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    _log(f'wrote {_rel(gate_path)}')

    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            fpath = os.path.join(root, fname)
            manifest[_rel(fpath)] = CP._sha256_file(fpath)
    manifest_path = os.path.join(out_root, 'manifest_sha256.json')
    W10._refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f'wrote {_rel(manifest_path)} ({len(manifest)} files)')

    _log(f'GATE_PASS={gate_pass}')
    for name, value in gate_items.items():
        _log(f'  {"OK " if value else "FAIL"} {name}')
    _log(f"diff classification totals: {totals}; trajectory mismatches: {trajectory['total_mismatches']}")
    _log(f"clone/capture: on={clone_capture['arm_on']} off={clone_capture['arm_off']}")
    _log(f"arm ON snapshots: {on_branch['pkl_count']} pkl {on_branch['inventory']}")
    _log(f"arm OFF: markers={off_branch['observed_markers']} expected={off_branch['expected_markers']} "
         f"vacuous={off_branch['marker_identity_is_vacuous']} "
         f"cycle7_skips={off_branch['cycle7_skip_lines']}")
    _log(f"failure accounting on={cls_on}")
    _log(f"failure accounting off={cls_off}")
    _log(f"solves: {solve['per_arm']}")
    _log(f"NON-GATING vs the committed AA-on reference: reproduces={reference_comparison['reproduces']} "
         f"n_diffs={reference_comparison['n_diffs']} "
         f"first={reference_comparison['first_differing_field']}")
    if off_branch['marker_identity_is_vacuous']:
        _log('NOTE: the marker/warning branch was NOT exercised -- every failure recovered, so no '
             'block reached it. The marker identity above holds at 0 == 0 and is NOT evidence that '
             'the marker writer works.')
    if not gate_pass:
        _log('*** GATE FAILED *** first genuine diffs, reported as-is:')
        for fname, entry in files.items():
            for d in (entry.get('genuine_diffs_first') or [])[:10]:
                _log(f'  [GENUINE] {fname}: {d}')
        sys.exit(1)


if __name__ == '__main__':
    main()
