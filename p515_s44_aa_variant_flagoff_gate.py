"""
P5.15 Addendum 25 item 2 -- Task 2: the FLAG-OFF two-cycle bitwise gate after
the AA reject-policy sub-option was added.

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 25 (the flag-off path stays
bitwise-verified); frozen spec v14
`data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json`,
`item2_aa_variant.flag` ("the flag-off path stays bitwise-verified").

METHOD = `p515_s43_aa_flagoff_gate.py`'s (commit 7f675637 evidence), reused BY
IMPORT -- its `_force_aa_flag_off_hook`, `_aa_call_counters`,
`_capture_returned_state`, `_classify_diffs`, `AA_NEW_DIAGNOSTIC_FIELD_NAMES`,
`_find_peak_rss_keys` -- with only the output location and mode label new
(its own `main()` writes to a fixed `P515S43/flagoff_gate*` root, which a
committed report cites; this script never writes there):
  * `run_s39_arm('s39_D', num_max_iters_override=2, output_root_override=OUT_DIR,
    mode_label_override=MODE_LABEL)`, `anderson_acceleration.enabled` set to
    False EXPLICITLY, every AA-module function/method that reads or writes the
    iterate call-counted (gate: exactly 0 calls);
  * (1) truncated comparison against D's committed trajectory
    (`p515_s40_polish_gap._reproduction_check`, BY IMPORT);
  * (2) full comparison of the report, every terminal artifact and every
    sidecar against `data/SRP1/Results/P515S40/clone_preflight_v2/lightweight/`
    (comparator `p515_s40_clone_capture_preflight._diff`, 8f5cff48, BY IMPORT);
  * every `aa_*` row field at its documented flag-off value; no `peak_rss` key
    in the written report.
ONE addition, from the gate-ruling follow-up (c): both references predate the
recourse-jump top-k sort-key changes (8682cfdd for
`objective_component_block_deltas`, Task 1 (a) of this task for
`block_deltas`), so any GENUINE diff inside those two lists of the recourse-
jump sidecar is passed to `p515_s44_tie_classifier.reclassify_sidecar_diffs`
(committed 631d4183, verified on the s44_gate evidence): a diff moves to the
non-gating `tie_order_diffs` bucket only if its whole row is explained by the
re-sort-and-straddle test; every other diff stays genuine and gates.

Launch (attached, alone, both streams captured):
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s44_aa_variant_flagoff_gate.py \\
        > data/SRP1/Results/P515S44/aa_variant_flagoff_gate_launch.log 2>&1

Preconditions (refuses loudly otherwise): no legacy or campaign lock; no other
live `p515_g_g1_g4_admm_gates.py` / `p515_s39_` / `p515_s4*` process; fresh
output root; production + harness files clean in git; both references present.
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

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator, BY IMPORT
import p515_s43_aa_flagoff_gate as FG  # noqa: E402 -- the flag-off method, BY IMPORT
import p515_s44_tie_classifier as TC  # noqa: E402 -- follow-up (c), BY IMPORT
from p515_s40_polish_gap import ARM_LABEL, D_REFERENCE_PATH, _reproduction_check  # noqa: E402

ARM_KEY = 's39_D'
NUM_CYCLES = 2
_SUFFIX = next(('_' + a.strip('_') for a in sys.argv[1:] if not a.startswith('--')), '')
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S44', f'aa_variant_flagoff_gate{_SUFFIX}')
MODE_LABEL = f'preflight_s44aavflagoff{_SUFFIX}'  # must start with 'preflight_' (assert_s39_capture_paths)
LIGHTWEIGHT_REFERENCE_DIR = FG.LIGHTWEIGHT_REFERENCE_DIR
RECOURSE_JUMP = 'recourse_jump_sidecar_baseline.jsonl'
FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = ('p515_g_g1_g4_admm_gates.py', 'p515_s39_', 'p515_s4')
PRODUCTION_FILES_TO_CHECK_CLEAN = tuple(FG.PRODUCTION_FILES_TO_CHECK_CLEAN) + (
    'p515_s43_aa_flagoff_gate.py', 'p515_s44_tie_classifier.py', 'p515_s44_aa_variant_flagoff_gate.py',
    'p515_s40_polish_gap.py', 'p515_s40_clone_capture_preflight.py')


def _check_preconditions():
    failures = []
    for lock in (os.path.join(REPO, '.p515_g_gate.lock'), G.CAMPAIGN_LOCK_PATH):
        if os.path.exists(lock):
            failures.append(f'lock file exists: {lock}')
    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded = {str(p) for p in CP._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        if len(fields) > 1 and fields[1] in excluded:
            continue
        if any(s in line for s in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')
    if os.path.exists(OUT_DIR):
        failures.append(f'output directory already exists (write-once): {OUT_DIR}')
    status = subprocess.run(['git', 'status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN),
                            capture_output=True, text=True, cwd=REPO).stdout
    if status.strip():
        failures.append(f'production/harness files not clean in git:\n{status}')
    if not os.path.isfile(D_REFERENCE_PATH):
        failures.append(f"D's committed reference report missing: {D_REFERENCE_PATH}")
    if not os.path.isdir(LIGHTWEIGHT_REFERENCE_DIR):
        failures.append(f'lightweight reference dir missing: {LIGHTWEIGHT_REFERENCE_DIR}')
    return failures


def _bucket(a_json, b_json, label, ref_rows=None, new_rows=None):
    if a_json is None or b_json is None:
        return {'error': f'missing artifact: reference_exists={a_json is not None} mine_exists={b_json is not None}',
                'n_genuine_diffs': None}
    raw = CP._diff(a_json, b_json, label)
    prov, aa_new, tie, genuine = FG._classify_diffs(raw)
    tie_order, row_classes = [], {}
    if label == RECOURSE_JUMP and genuine:
        tie_order, genuine, row_classes = TC.reclassify_sidecar_diffs(genuine, ref_rows, new_rows, RECOURSE_JUMP)
    return {'n_raw_diffs': len(raw), 'provenance_diffs': prov, 'aa_new_field_diffs': aa_new,
            'known_tie_break_diffs': tie, 'tie_order_diffs_resort_straddle': tie_order,
            'tie_order_row_classes': row_classes, 'genuine_diffs': genuine, 'n_genuine_diffs': len(genuine)}


def main():
    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[S44-AAV-FLAGOFF PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S44-AAV-FLAGOFF] preconditions passed.', flush=True)
    G._acquire_exclusive_run_lock()
    started = time.time()
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip()

    with FG._force_aa_flag_off_hook(), FG._aa_call_counters() as aa_call_counts, \
            FG._capture_returned_state() as state_holder:
        report, report_path = G.run_s39_arm(ARM_KEY, num_max_iters_override=NUM_CYCLES,
                                            output_root_override=OUT_DIR, mode_label_override=MODE_LABEL)
    total_aa_calls = sum(v for k, v in aa_call_counts.items() if 'informational' not in k)
    print(f"[S44-AAV-FLAGOFF] cycles_run={report['cycles_run']} aa_call_counts={aa_call_counts}", flush=True)

    d_repro = _reproduction_check(report)
    _dp, d_aa_new, d_tie, d_genuine = FG._classify_diffs(d_repro['diffs'])

    lw_report = FG._load_json(os.path.join(LIGHTWEIGHT_REFERENCE_DIR, f'g_{ARM_LABEL}.json'))
    lw_report_bucket = _bucket(lw_report, report, 'report')
    artifacts = {f: _bucket(FG._load_json(os.path.join(LIGHTWEIGHT_REFERENCE_DIR, f)),
                            FG._load_json(os.path.join(OUT_DIR, f)), f) for f in CP.ARTIFACT_FILES}
    sidecars = {}
    for f in CP.SIDECAR_JSONL_FILES:
        a_rows = FG._load_jsonl(os.path.join(LIGHTWEIGHT_REFERENCE_DIR, f))
        b_rows = FG._load_jsonl(os.path.join(OUT_DIR, f))
        sidecars[f] = _bucket(a_rows, b_rows, f, a_rows, b_rows)
    buckets = [lw_report_bucket] + list(artifacts.values()) + list(sidecars.values())
    lw_all_present = all('error' not in v for v in buckets)
    lw_genuine = sum((v.get('n_genuine_diffs') or 0) for v in buckets)

    aa_pattern_violations = []
    for row in report.get('cycle_trajectory') or []:
        if row.get('aa_enabled') is not False:
            aa_pattern_violations.append({'cycle': row.get('cycle'), 'field': 'aa_enabled', 'value': row.get('aa_enabled')})
        for field in FG.AA_NEW_DIAGNOSTIC_FIELD_NAMES - {'aa_enabled'}:
            if row.get(field) is not None:
                aa_pattern_violations.append({'cycle': row.get('cycle'), 'field': field, 'value': row.get(field)})
    peak_rss_hits = FG._find_peak_rss_keys(report)
    state = state_holder['state'] or {}

    gate = {
        'zero_aa_calls': total_aa_calls == 0,
        'vs_D_committed_first_2_cycles_zero_genuine': len(d_genuine) == 0,
        'vs_D_committed_n_cycles_compared_is_2': d_repro.get('n_cycles_compared') == NUM_CYCLES,
        'vs_lightweight_all_artifacts_present': lw_all_present,
        'vs_lightweight_zero_genuine': lw_genuine == 0,
        'aa_fields_at_flag_off_values': not aa_pattern_violations,
        'no_peak_rss_in_written_report': not peak_rss_hits,
    }
    gate_pass = all(gate.values())
    payload = {
        'stage': 'P5.15 Addendum 25 item 2 -- Task 2: flag-off two-cycle bitwise gate (AA reject-policy variant added)',
        'authority': ['PLANNER_BRIEF_2026-09-13.md Addendum 25',
                      'data/SRP1/Results/P515S44/frozen_s44_selection_spec_v14_e4500e27.json item2_aa_variant.flag'],
        'method': 'p515_s43_aa_flagoff_gate.py components BY IMPORT + p515_s44_tie_classifier for the recourse-jump top-k lists',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head': head,
        'arm': ARM_KEY, 'num_cycles': NUM_CYCLES, 'mode_label': MODE_LABEL,
        'out_dir': os.path.relpath(OUT_DIR, REPO), 'report_path': os.path.relpath(report_path, REPO),
        'anderson_acceleration_settings_in_force': report.get('rule_eleven_checklist', {}).get(
            's43_anderson_acceleration_settings_in_force'),
        'aa_call_counts': aa_call_counts, 'total_gated_aa_calls': total_aa_calls,
        'state_peak_rss_ru_maxrss': state.get('peak_rss_ru_maxrss'),
        'peak_rss_keys_in_written_report': peak_rss_hits,
        'aa_field_pattern_violations': aa_pattern_violations,
        'vs_d_committed_truncated': {**{k: v for k, v in d_repro.items() if k != 'diffs'},
                                     'n_aa_new_field_diffs': len(d_aa_new), 'n_known_tie_break_diffs': len(d_tie),
                                     'genuine_diffs': d_genuine, 'n_genuine_diffs': len(d_genuine)},
        'vs_clone_preflight_v2_lightweight': {
            'reference_dir': os.path.relpath(LIGHTWEIGHT_REFERENCE_DIR, REPO),
            'report': lw_report_bucket, 'artifacts': artifacts, 'sidecars': sidecars,
            'total_genuine_diffs': lw_genuine, 'all_artifacts_present': lw_all_present},
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'gate_items': gate, 'gate_pass': gate_pass, 'wall_clock_s': time.time() - started,
    }
    results_path = os.path.join(OUT_DIR, 'flagoff_gate_results.json')
    if os.path.exists(results_path):
        raise RuntimeError(f'refusing to overwrite {results_path}')
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    manifest = {}
    for root, _dirs, files in os.walk(OUT_DIR):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            manifest[os.path.relpath(fpath, REPO)] = CP._sha256_file(fpath)
    with open(os.path.join(OUT_DIR, 'manifest_sha256.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S44-AAV-FLAGOFF] gate items: {gate}')
    for name, v in [('report', lw_report_bucket)] + list(artifacts.items()) + list(sidecars.items()):
        print(f"[S44-AAV-FLAGOFF]   vs lightweight {name}: raw={v.get('n_raw_diffs')} "
              f"aa_new={len(v.get('aa_new_field_diffs', []))} alias={len(v.get('known_tie_break_diffs', []))} "
              f"tie_order={len(v.get('tie_order_diffs_resort_straddle', []))} genuine={v.get('n_genuine_diffs')}")
    print(f'[S44-AAV-FLAGOFF] GATE_PASS={gate_pass}')
    if not gate_pass:
        for d in (d_genuine + [x for v in buckets for x in v.get('genuine_diffs', [])])[:20]:
            print(f'  [FIRST GENUINE DIFFS] {d}')
        sys.exit(1)


if __name__ == '__main__':
    main()
