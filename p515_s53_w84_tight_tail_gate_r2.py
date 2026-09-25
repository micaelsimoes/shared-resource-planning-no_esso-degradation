"""
P5.15 Addendum 46 ruling 7, Planner task W84 item 1 -- the SRP1 TWO-CYCLE BITWISE GATE for the convergence-depth tight
tail, RE-RUN with the whole-arm comparison fixed per the Planner's ruling Q1. Everything else is the committed W83 gate
(`p515_s53_w83_tight_tail_gate.py`, frozen in spec v28 c8b5a715), BY IMPORT, unchanged.

WHY. The committed W83 run (eb8cf81f, arm s53w83tailgate) held every item except one: 44 "genuine" diffs against the
committed r2 arm, all in `per_cycle_record.jsonl` = the 22 `PER_CYCLE_RESPONSE_FIELDS` keys W64 (70950e8c) added to
`p515_s44_campaign_harness.PER_CYCLE_RECORD_FIELDS`, which `W10.run_arm` writes, x 2 rows, value None in the new arm and
absent in r2 (run before W64). PLANNER RULING Q1 (verbatim): "re-run the gate under a new arm name, but do NOT declare
the 22 fields as 'expected differences.' Declaring them expected would mask a real change in any of those fields later.
Instead: compare only fields present in both arms, strictly, as now; separately assert that the extra keys are exactly
the 22 known PER_CYCLE_RESPONSE_FIELDS added by W64 (70950e8c), and that all of them are None in this arm."

THE COMPARISON RULE (declared before the run; this is the ONLY change to what is compared):
  * every file other than `per_cycle_record.jsonl`: the committed comparator (`S52G.compare_against_w39_arm` ->
    `W10.classify`), unchanged, results taken as they come;
  * `per_cycle_record.jsonl`: row i of this arm is restricted to the keys present in row i of the reference arm and
    compared with the reference row by the SAME `W10.classify`, strictly (a key of the reference missing here stays in
    and is a genuine diff);
  * SEPARATE ASSERTION, gating (folded into the comparison's `zero_genuine_diffs`, so the committed layers' verdict
    fails with it): same row count; on EVERY row the keys of this arm minus the keys of the reference are EXACTLY the 22
    W64 fields (the literal below, taken from 70950e8c and checked equal to the live harness tuple), each of them None;
    the reference minus this arm is empty; and the reference keys are exactly the pre-W64 `PER_CYCLE_TRAJECTORY_FIELDS`;
  * the committed comparator's own raw result for `per_cycle_record.jsonl` is kept beside it (reported, not gated).

DECLARED SUBSTITUTIONS (and no others), on the committed W83 gate module (T83) and the r2 layers it imports:
  1. identifiers: T83.STAGE / SCHEMA / ARM ('s53w84tailgate', so a fresh arm-named eval dir and fresh working-dir ids)
     / OUT_REL (write-once, P515S53/tight_tail_w84/srp1_bitwise_gate); fresh launch log (below);
  2. S53G.EXTRA_CLEAN_FILES += this file, the W84 harness checks script and its committed artifact (T83 then adds its own);
  3. S52G.compare_against_w39_arm -> `compare_both_arms_strict` (the rule above), installed before T83.main and restored
     after; called exactly once.
The tail is enabled exactly as in W83 (T83's one-shot arm wrapper, after the committed hook). The campaign harness now
carries the W84 hook checklist (committed 84e45402, checked 6b9a0984): with the W10 configuration it declares nothing,
so it records the tail as off before T83's wrapper enables it -- a new `rule_eleven_checklist` entry, provenance.

DECLARED SOLVE PROFILE (as W83, from the case file and W10.CAP = 2): 153 base solves (51 x 3), per-event reconciled;
W10.GUARD.verify(observed) EXACTLY, inside the W35 layer.

PREDICTIONS (recorded before the run, in this committed file):
  P1 153/153 solves, GUARD.verify(153) == [], 0 retries;
  P2 0 diffs vs the committed C* reference rows[:2] + derived fields; 0 of 29 trajectory-field mismatches;
  P3 strict both-arms comparison: 0 genuine diffs on every one of the 12 compared files; per_cycle_record: 2 rows,
     29 common keys each, 0 diffs; extra keys exactly the 22 W64 fields, all None, on both rows; none missing;
  P4 the committed comparator's raw result on per_cycle_record reproduces W83's 44 genuine diffs (22 x 2);
  P5 tail inert: 3 apply calls (cycles 1, 2, exit) all inactive and not acting, 2 next-state calls both False, 144
     floor records all parsed at the production compl_inf_tol (TSO 5e-4, DSO 1e-4), no "tail ON" line;
  P6 g_s39_D.json vs the r2 arm: provenance diffs only (under rule_eleven_checklist), count 3 = W83's 2 + the W84
     hook entry (low confidence on the count: it depends on how CP._diff counts an absent subtree).
GATE = T83's verdict (every W35/W39/W51/W83 item, with the fixed comparison) AND every W84 item below.

EXACT LAUNCH COMMAND (repo root; attached, ALONE, both streams captured; never detached):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s53_w84_tight_tail_gate_r2.py \\
        > data/SRP1/Results/P515S53/tight_tail_w84/srp1_bitwise_gate_launch.log 2>&1
OUTPUT (write-once): data/SRP1/Results/P515S53/tight_tail_w84/srp1_bitwise_gate/ (everything T83 writes, under its
W83 file names) + w84_gate_addendum.json + w84_manifest_sha256.json. Exit 0 on PASS, 1 on FAIL or a refusal.
"""

import copy
import json
import os
import sys
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

import p515_s53_w83_tight_tail_gate as T83  # noqa: E402 -- imports the r2 gate -> W48 -> W39 -> W35 -> W32 -> W10 (armed guard)

S53G = T83.S53G
S52G = T83.S52G
W10 = T83.W10
GUARD = T83.GUARD
CP = T83.CP
H = T83.H

STAGE = ('P5.15 Addendum 46 ruling 7 W84 -- SRP1 two-cycle bitwise identity with the convergence-depth tight tail '
         'ENABLED, whole-arm comparison per Planner ruling Q1 (keys present in both arms, strictly; the 22 W64 keys '
         'asserted separately)')
SCHEMA = 'p515_s53_w84_tight_tail_gate_r2_v1'
ARM = 's53w84tailgate'
_S53 = os.path.join('data', 'SRP1', 'Results', 'P515S53')
OUT_REL = os.path.join(_S53, 'tight_tail_w84', 'srp1_bitwise_gate')
W84_CHECKS_SCRIPT = 'p515_s53_w84_harness_capture_checks.py'
W84_CHECKS_JSON = os.path.join(_S53, 'tight_tail_w84', 'harness_checks', 'checks_w84.json')
W83_GATE_SCRIPT_SHA256 = '2895a8f215fca6e9a65561dc941052bdc79d59eea5a610107f1b841089dd9dfc'   # frozen in spec v28
PER_CYCLE_FILE = 'per_cycle_record.jsonl'
# `PER_CYCLE_RESPONSE_FIELDS` as W64 added it (git show 70950e8c:p515_s44_campaign_harness.py), verbatim.
W64_RESPONSE_FIELDS = (
    'response_cycle', 'response_captured', 'response_capture_error',
    'E_abs_d_p_mwh_weighted', 'E_abs_d_p_mwh_unweighted',
    'sum_omega_d2_p_mw2h_weighted', 'sum_omega_d2_p_mw2h_unweighted',
    'market_part_mw2h_weighted', 'operation_part_mw2h_weighted',
    'market_part_mw2h_unweighted', 'operation_part_mw2h_unweighted',
    'row18_charge_weighted', 'covariance_dso_weighted',
    'curtailed_res_dso_mwh_weighted', 'curtailed_res_tso_mwh_weighted',
    'max_abs_d_p_mw', 'n_non_optimal_block_terminations', 'non_optimal_blocks',
    'cycle_wall_s', 'rss_bytes', 'ru_maxrss_bytes', 'response_capture_s',
)
W64_COMMIT = '70950e8c'

ORIG_COMPARE = S52G.compare_against_w39_arm
COMPARE = {'calls': 0, 'result': None, 'assertion': None}


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W84-gate] {msg}', flush=True)


def _sha(rel):
    return CP._sha256_file(os.path.join(REPO, rel))


def extra_keys_assertion(ref_rows, mine_rows):
    """Planner ruling Q1's separate assertion, per row."""
    w64 = set(W64_RESPONSE_FIELDS)
    trajectory = set(H.PER_CYCLE_TRAJECTORY_FIELDS)
    rows = []
    for i, (ref, mine) in enumerate(zip(ref_rows, mine_rows)):
        extra = set(mine) - set(ref)
        missing = set(ref) - set(mine)
        not_none = sorted(k for k in extra if mine[k] is not None)
        rows.append({'row': i, 'n_ref_keys': len(ref), 'n_arm_keys': len(mine), 'n_common_keys': len(set(ref) & set(mine)),
                     'extra_equals_the_22_w64_fields': extra == w64, 'extra_not_in_w64': sorted(extra - w64),
                     'w64_not_extra': sorted(w64 - extra), 'extra_not_none': not_none,
                     'missing_from_this_arm': sorted(missing), 'ref_keys_are_pre_w64_trajectory_fields':
                         set(ref) == trajectory})
    holds = (len(ref_rows) == len(mine_rows) and len(rows) > 0
             and all(r['extra_equals_the_22_w64_fields'] and not r['extra_not_none'] and not r['missing_from_this_arm']
                     and r['ref_keys_are_pre_w64_trajectory_fields'] for r in rows)
             and len(W64_RESPONSE_FIELDS) == 22 and tuple(W64_RESPONSE_FIELDS) == tuple(H.PER_CYCLE_RESPONSE_FIELDS))
    return {'n_rows': {'reference': len(ref_rows), 'this_arm': len(mine_rows)}, 'per_row': rows,
            'w64_literal_equals_live_harness_tuple': tuple(W64_RESPONSE_FIELDS) == tuple(H.PER_CYCLE_RESPONSE_FIELDS),
            'w64_commit': W64_COMMIT, 'holds': holds}


def compare_both_arms_strict(out_root):
    """Substitution 3: the committed comparator on every file, with `per_cycle_record.jsonl` re-compared on the keys
    present in both arms (strictly, by the same W10.classify) and the separate extra-key assertion."""
    COMPARE['calls'] += 1
    result = ORIG_COMPARE(out_root)          # committed comparator, unchanged
    raw_entry = result['per_file'].get(PER_CYCLE_FILE)
    ref_rows = CP._load_jsonl(os.path.join(REPO, S52G.W39_ARM['arm_dir'], PER_CYCLE_FILE))
    mine_rows = CP._load_jsonl(os.path.join(out_root, 'arm', PER_CYCLE_FILE))
    if ref_rows is None or mine_rows is None:
        assertion = {'holds': False, 'error': f'missing: reference={ref_rows is not None} arm={mine_rows is not None}'}
        strict_entry = {'error': assertion['error']}
    else:
        assertion = extra_keys_assertion(ref_rows, mine_rows)
        restricted = [{k: v for k, v in mine.items() if k in ref} for ref, mine in zip(ref_rows, mine_rows)]
        restricted += mine_rows[len(ref_rows):]          # any surplus row stays in, whole
        strict_entry = W10.classify(PER_CYCLE_FILE, ref_rows, restricted, result['token_groups'])
        strict_entry['n_rows'] = {'w39': len(ref_rows), 'w48': len(mine_rows)}
        strict_entry['rule'] = ('W84 / Planner ruling Q1: this arm restricted to the keys present in the reference '
                                'row, compared strictly by W10.classify; extra keys asserted separately')
    per_file = dict(result['per_file'])
    per_file[PER_CYCLE_FILE] = strict_entry
    totals = {'provenance': 0, 'tie_order': 0, 'genuine': 0}
    for v in per_file.values():
        for k, n in (v.get('n') or {}).items():
            totals[k] += n
    out = dict(result)
    out.update({
        'per_file': per_file, 'totals': totals,
        'all_files_present': all('error' not in v for v in per_file.values()),
        'zero_genuine_diffs': totals['genuine'] == 0 and bool(assertion.get('holds')),
        'w84_rule': ('Planner ruling Q1: keys present in both arms, strictly; the extra keys asserted separately to be '
                     'exactly the 22 W64 PER_CYCLE_RESPONSE_FIELDS, all None; zero_genuine_diffs requires both'),
        'w84_extra_keys_assertion': assertion,
        'w84_per_cycle_record_committed_comparator_raw_REPORTED': raw_entry,
        'w84_committed_comparator_totals_raw_REPORTED': result['totals'],
    })
    COMPARE['result'] = out
    COMPARE['assertion'] = assertion
    return out


def _raw_diffs_all_w64(raw_entry, n_rows):
    if not raw_entry or 'n' not in raw_entry:
        return False
    listed = raw_entry.get('genuine_diffs_first') or []
    ok_listed = all(d.get('legacy') == '<MISSING>' and d.get('lightweight') is None
                    and str(d.get('field', '')).rsplit('.', 1)[-1] in W64_RESPONSE_FIELDS for d in listed)
    return ok_listed and raw_entry['n']['genuine'] == 22 * n_rows


def main():
    out_root = os.path.join(REPO, OUT_REL)
    _log(STAGE)
    _log(f"git HEAD {W10._git(['rev-parse', 'HEAD'])}; output root {OUT_REL}")
    problems = []
    if _sha('p515_s53_w83_tight_tail_gate.py') != W83_GATE_SCRIPT_SHA256:
        problems.append('the imported W83 gate script is not the one frozen in spec v28')
    checks = json.load(open(os.path.join(REPO, W84_CHECKS_JSON))) if os.path.isfile(
        os.path.join(REPO, W84_CHECKS_JSON)) else {}
    if not checks.get('all_hold'):
        problems.append(f'W84 harness checks absent or not all holding: {W84_CHECKS_JSON}')
    elif checks.get('harness_sha256') != _sha('p515_s44_campaign_harness.py'):
        problems.append('the campaign harness differs from the one the W84 checks verified')
    if os.path.exists(out_root):
        problems.append(f'output root already exists (write-once): {OUT_REL}')
    if tuple(W64_RESPONSE_FIELDS) != tuple(H.PER_CYCLE_RESPONSE_FIELDS):
        problems.append('the W64 literal differs from the live harness PER_CYCLE_RESPONSE_FIELDS')
    if problems:
        for p in problems:
            _log(f'[PRECONDITION FAILED] {p}')
        return 1
    _log('DECLARED BEFORE THE RUN: 153 solves base (51 x 3, per-event reconciled, GUARD.verify exact); comparison '
         'per Planner ruling Q1 (per_cycle_record on keys present in both arms + the 22-key assertion); predictions '
         'P1-P6 in this file (committed before the run)')

    # ---- the declared substitutions, and no others ----
    T83.STAGE = STAGE
    T83.SCHEMA = SCHEMA
    T83.ARM = ARM
    T83.OUT_REL = OUT_REL
    S53G.EXTRA_CLEAN_FILES = tuple(S53G.EXTRA_CLEAN_FILES) + (
        os.path.basename(__file__), W84_CHECKS_SCRIPT, W84_CHECKS_JSON)
    S52G.compare_against_w39_arm = compare_both_arms_strict
    try:
        status = T83.main()
    finally:
        S52G.compare_against_w39_arm = ORIG_COMPARE
    _log(f'W83 gate layers verdict (exit status, with the Q1 comparison): {status}')

    comparison = COMPARE['result'] or {}
    assertion = COMPARE['assertion'] or {}
    raw_entry = comparison.get('w84_per_cycle_record_committed_comparator_raw_REPORTED')
    strict_entry = (comparison.get('per_file') or {}).get(PER_CYCLE_FILE) or {}
    n_rows = ((assertion.get('n_rows') or {}).get('this_arm')) or 0
    items = {
        'w83_gate_layers_pass_with_the_q1_comparison': status == 0,
        'strict_comparator_called_exactly_once': COMPARE['calls'] == 1,
        'comparator_restored': S52G.compare_against_w39_arm is ORIG_COMPARE,
        'extra_keys_exactly_the_22_w64_fields_all_none_every_row': bool(assertion.get('holds')),
        'per_cycle_record_zero_diffs_on_common_keys': (strict_entry.get('n') or {}).get('genuine') == 0
                                                        and (strict_entry.get('n_raw_diffs') == 0),
        'every_compared_file_zero_genuine_diffs': (comparison.get('totals') or {}).get('genuine') == 0,
    }
    reported = {
        'P4_committed_comparator_raw_reproduces_w83_44_diffs': _raw_diffs_all_w64(raw_entry, n_rows),
        'raw_per_cycle_genuine': (raw_entry or {}).get('n'),
        'per_file_n': {f: v.get('n', v.get('error')) for f, v in (comparison.get('per_file') or {}).items()},
    }
    gate_pass = all(items.values())
    if not os.path.isdir(out_root):
        _log('output root absent (the committed layers refused before the arm); nothing more to write')
        for k, v in items.items():
            _log(f'   {k}: {v}')
        _log(f'GATE_PASS={gate_pass}')
        return 1
    addendum_path = os.path.join(out_root, 'w84_gate_addendum.json')
    manifest_path = os.path.join(out_root, 'w84_manifest_sha256.json')
    for p in (addendum_path, manifest_path):
        W10._refuse_overwrite(p)
    gate_json = os.path.join(out_root, 'gate.json')
    solve_profile = json.load(open(gate_json)).get('solve_profile') if os.path.isfile(gate_json) else None
    payload = {
        'schema': SCHEMA + '_addendum', 'stage': STAGE,
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 46 ruling 7', 'Planner task W84 item 1 and ruling Q1',
                      'data/SRP1/Results/P515S53/frozen_s53_spec_v28_c8b5a715.json (the W83 gate, by import)'],
        'planner_ruling_q1_verbatim': ('"re-run the gate under a new arm name, but do NOT declare the 22 fields as '
                                       '"expected differences." ... Instead: compare only fields present in both '
                                       'arms, strictly, as now; separately assert that the extra keys are exactly the '
                                       '22 known PER_CYCLE_RESPONSE_FIELDS added by W64 (70950e8c), and that all of '
                                       'them are None in this arm."'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': W10._git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': _sha(os.path.basename(__file__)),
        'imported_w83_gate_sha256': _sha('p515_s53_w83_tight_tail_gate.py'),
        'campaign_harness_sha256': _sha('p515_s44_campaign_harness.py'),
        'w84_harness_checks': {'path': W84_CHECKS_JSON, 'sha256': _sha(W84_CHECKS_JSON)},
        'instance': {'candidate': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025',
                     'candidate_key': W10.C_STAR_KEY_PIN},
        'objective_convention': 'gross_operational_cost (settlement-excluded gross cost), as in gate.json',
        'declared_substitutions': {
            '1_identifiers': {'STAGE': STAGE, 'SCHEMA': SCHEMA, 'ARM': ARM, 'OUT_REL': OUT_REL},
            '2_extra_clean_files': [os.path.basename(__file__), W84_CHECKS_SCRIPT, W84_CHECKS_JSON],
            '3_comparator': 'S52G.compare_against_w39_arm -> compare_both_arms_strict (Planner ruling Q1)'},
        'solve_profile_from_gate_json': solve_profile,
        'comparison_GATING': comparison,
        'w84_items': items, 'reported_not_gated': reported,
        'guard_counts_at_end': dict(GUARD.counts),
        'gate_pass': gate_pass,
    }
    with open(addendum_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            if os.path.join(root, fname) == manifest_path:
                continue
            fpath = os.path.join(root, fname)
            manifest[W10._rel(fpath)] = CP._sha256_file(fpath)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _log(f"extra-key assertion: {assertion.get('holds')} (rows {assertion.get('n_rows')}); per_cycle_record strict "
         f"{strict_entry.get('n')}; totals {comparison.get('totals')}; raw committed comparator totals "
         f"{comparison.get('w84_committed_comparator_totals_raw_REPORTED')}")
    for key, value in items.items():
        _log(f'   {key}: {value}')
    for key, value in reported.items():
        _log(f'   (reported) {key}: {value}')
    _log(f'wrote {W10._rel(addendum_path)}, {W10._rel(manifest_path)} ({len(manifest)} files)')
    _log(f'GATE_PASS={gate_pass}')
    return 0 if gate_pass else 1


if __name__ == '__main__':
    sys.exit(main())
