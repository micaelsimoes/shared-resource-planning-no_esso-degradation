"""
P5.15 Step 3.7 integration -- Task 1: the flag-off two-cycle bitwise gate for
Anderson acceleration (AA).

Authority: `PLANNER_BRIEF_2026-09-13.md` Addendum 23 item 3.7 amendment (v)
("Gate: two-cycle bitwise identity with the flag off") and Addendum 24
("Step 3.7 code choices accepted ... with the flag off the AA code must not
touch the iterate (the bitwise gate verifies it)"); frozen spec v12
`data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json`,
`item4_step_3_7_anderson.gate` ("flag off: two-cycle bitwise identity against
the oracle configuration"); frozen spec v13
`data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json`,
`items_2_3`.

================================================================================
WHAT THIS SCRIPT DOES
================================================================================
Runs `p515_g_g1_g4_admm_gates.run_s39_arm('s39_D', num_max_iters_override=2,
output_root_override=..., mode_label_override=...)` BY IMPORT (never through
that module's own `__main__`) ONCE, with `admm_parameters.anderson_
acceleration['enabled']` set to `False` EXPLICITLY (not merely relying on the
constructor default) via the same `pre_solve_hook`-wrapping technique
`p515_s40_clone_capture_preflight._capture_mode_hook_override` already uses on
`p515_g_g1_g4_admm_gates._s39_configure_hook` for `tso_snapshot_capture_mode`.

Compares the resulting `g_s39_D.json` (and every terminal artifact/sidecar)
BITWISE against TWO independent references, using
`p515_s40_clone_capture_preflight.py`'s comparator (`_diff`, `EXCLUDE_KEY_
NAMES`, `EXCLUDE_DOTTED_SUFFIXES`, `INTENTIONAL_DIFF_SUFFIXES`) BY IMPORT, as
fixed in commit `8f5cff48` (rule_eleven_checklist provenance non-gating):

  (1) D's own committed trajectory, `data/SRP1/Results/P515S39_D_run/
      g_s39_D.json` -- TRUNCATED comparison (`p515_s40_polish_gap.
      _reproduction_check`, BY IMPORT, unchanged): `cycle_trajectory[:2]` plus
      the rows-derived top-level fields, vs D's own first two committed rows.
  (2) `data/SRP1/Results/P515S40/clone_preflight_v2/lightweight/` -- a FULL
      2-cycle run of the SAME oracle configuration on code committed AFTER
      the Step 3.6 lightweight-capture integration (bitwise-gate-passed,
      `8214be0d`) but BEFORE the bound-restore fix (`16a19456`) and BEFORE
      AA integration (`9a965494`) -- every terminal artifact and sidecar
      compared, not just the trajectory.

Because AA integration added TWELVE new `aa_*` fields, unconditionally, to
every `admm_diagnostics`/`cycle_trajectory` row (module docstring of
`admm_anderson_acceleration.py`; `shared_resources_planning.py` ~3147-3158),
BOTH references necessarily lack these keys (they predate the AA commit).
This is an EXPECTED, DOCUMENTED difference distinct from the bitwise-identity
claim the gate exists to verify (the claim is "AA code does not touch the
existing iterate/behaviour with the flag off", not "no new field was ever
added") -- `AA_NEW_DIAGNOSTIC_FIELD_NAMES` below lists the exact twelve field
names (read directly off `shared_resources_planning.py`'s own dict literal),
and `_classify_diffs` splits every raw diff into:
  - `aa_new_field_diffs`: the reference side is `'<MISSING>'` (key did not
    exist in the pre-AA reference at all) AND the new run's value matches the
    documented flag-off constant (`aa_enabled` -> `False`; every other
    `aa_*` field -> `None`) -- reported, NOT gated.
  - `genuine_diffs`: everything else -- GATES. A single `aa_*` field whose
    value does NOT match the documented flag-off constant (e.g. `aa_enabled`
    reading `True`, or any other `aa_*` field being non-`None`) is NOT
    filtered out here and fails the gate, exactly as any other unexplained
    diff would.

Also INSTRUMENTS, harness-side (never modifying production), every function
in `admm_anderson_acceleration.py` that reads or writes the ADMM iterate
(`build_iterate_layout`, `collect_w`, `write_back_w`, `combined_scaled_
residual`, `_antisymmetry_check`, every `AndersonAccelerationState` method)
plus the two orchestration helpers in `shared_resources_planning.py`
(`_get_admm_rho_channel_scalars_for_aa`, `_anderson_acceleration_cycle_step`),
by wrapping each with a call counter that calls straight through (never
changes behaviour) -- and asserts the total is exactly 0 for the whole run.
`admm_anderson_acceleration.anderson_acceleration_enabled` (the flag read
itself, which touches nothing) is counted SEPARATELY, informationally, and is
NOT part of the zero-call gate.

Also captures the `state` dict `shared_resources_planning._run_operational_
planning` returns (by wrapping that module-level function, the same
unqualified-call-interception technique `s39_exempt_until_capture_hooks`/
`s38_pf_capture_hooks` already use) to read `state['peak_rss_ru_maxrss']`/
`state['peak_rss_platform_units']` directly, and recursively scans every
compared JSON artifact for a `peak_rss` key to report whether these NEW
`state`-only fields ever reach a compared artifact (answer stated in the
results payload, not assumed).

================================================================================
EXACT LAUNCH COMMAND
================================================================================
    /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s43_aa_flagoff_gate.py \\
        > data/SRP1/Results/P515S43/flagoff_gate_launch.log 2>&1

Run attached, alone (no `screen`/`nohup`/backgrounding), both streams
captured via the shell redirection above.

================================================================================
PRECONDITIONS (checked before anything is written; refuses loudly otherwise)
================================================================================
  1. `.p515_g_gate.lock` does not already exist.
  2. No OTHER process (excluding this process's own ancestor chain) matches
     `p515_g_g1_g4_admm_gates.py`, `p515_s39_`, `p515_s40_`, `p515_s41_`,
     `p515_s42_`, or `p515_s43_` in `ps aux`.
  3. The output root does not exist yet (write-once; `--suffix` redirects it).
  4. The production files this task's diff depends on are clean in git.
  5. D's committed reference report and the `clone_preflight_v2/lightweight`
     reference directory both exist.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_g_g1_g4_admm_gates as G  # noqa: E402
import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator, BY IMPORT
import admm_anderson_acceleration as AA  # noqa: E402
import shared_resources_planning as srp  # noqa: E402
from p515_s40_polish_gap import (  # noqa: E402
    ARM_LABEL, D_REFERENCE_PATH, D_CERTIFICATION_CYCLE, D_CERTIFIED_COST,
    _reproduction_check,
)

ARM_KEY = 's39_D'
NUM_CYCLES = 2
LIGHTWEIGHT_REFERENCE_DIR = os.path.join(
    REPO, 'data', 'SRP1', 'Results', 'P515S40', 'clone_preflight_v2', 'lightweight')

_RUN_SUFFIX = ''
for _arg in sys.argv[1:]:
    if not _arg.startswith('--'):
        _RUN_SUFFIX = '_' + _arg.strip('_')
        break
OUT_DIR = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S43', f'flagoff_gate{_RUN_SUFFIX}')
# Must start with 'preflight_' -- `assert_s39_capture_paths`'s own
# `this_call_mode_is_valid` check hard-requires it for a non-'real' mode.
MODE_LABEL = f'preflight_s43flagoff{_RUN_SUFFIX}'

FORBIDDEN_LIVE_PROCESS_SUBSTRINGS = tuple(CP.FORBIDDEN_LIVE_PROCESS_SUBSTRINGS) + (
    'p515_s40_', 'p515_s41_', 'p515_s42_', 'p515_s43_')

PRODUCTION_FILES_TO_CHECK_CLEAN = tuple(CP.PRODUCTION_FILES_TO_CHECK_CLEAN) + (
    'admm_anderson_acceleration.py',)

# The twelve `aa_*` fields added UNCONDITIONALLY to every admm_diagnostics /
# cycle_trajectory row by the AA integration commit (read directly off
# `shared_resources_planning.py`'s own dict literal, ~3147-3158). With the
# flag off, `aa_enabled` must read `False` and every other field must read
# `None` -- anything else is a genuine (gating) diff, not filtered here.
AA_NEW_DIAGNOSTIC_FIELD_NAMES = {
    'aa_enabled', 'aa_action', 'aa_accepted', 'aa_combined_residual',
    'aa_baseline_residual_before', 'aa_memory_size_before', 'aa_memory_size_after',
    'aa_gamma_columns', 'aa_reset', 'aa_reset_reason', 'aa_rho_changed_channels',
    'aa_boyd_all_pass',
}
AA_NEW_FIELD_EXPECTED_FLAG_OFF_VALUE = {'aa_enabled': False}  # every other -> None


def _refuse_overwrite(path):
    if os.path.exists(path):
        raise RuntimeError(f'refusing to overwrite existing artifact: {path}')


def _check_preconditions():
    failures = []
    lock_path = os.path.join(REPO, '.p515_g_gate.lock')
    if os.path.exists(lock_path):
        failures.append(f'lock file already exists: {lock_path}')

    try:
        ps_output = subprocess.run(['ps', 'aux'], capture_output=True, text=True, check=True).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not scan process table: {error}')
        ps_output = ''
    excluded_pids = {str(p) for p in CP._ancestor_pids()}
    for line in ps_output.splitlines():
        fields = line.split()
        pid = fields[1] if len(fields) > 1 else None
        if pid in excluded_pids:
            continue
        if any(substring in line for substring in FORBIDDEN_LIVE_PROCESS_SUBSTRINGS):
            failures.append(f'a forbidden process appears to be alive: {line.strip()}')

    if os.path.exists(OUT_DIR):
        failures.append(f'output directory already exists (write-once): {OUT_DIR}')

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--'] + list(PRODUCTION_FILES_TO_CHECK_CLEAN),
            capture_output=True, text=True, check=True, cwd=REPO).stdout
    except Exception as error:  # noqa: BLE001
        failures.append(f'could not run git status: {error}')
        status = ''
    if status.strip():
        failures.append(f'production files are not clean in git:\n{status}')

    if not os.path.isfile(D_REFERENCE_PATH):
        failures.append(f"D's committed reference report missing: {D_REFERENCE_PATH}")
    if not os.path.isdir(LIGHTWEIGHT_REFERENCE_DIR):
        failures.append(f'clone_preflight_v2/lightweight reference dir missing: {LIGHTWEIGHT_REFERENCE_DIR}')

    return failures


# ---------------------------------------------------------------------------
# The ONE configuration action: force `anderson_acceleration.enabled = False`
# EXPLICITLY (not merely relying on the ADMMParameters constructor default),
# via the SAME `pre_solve_hook`-wrapping technique
# `p515_s40_clone_capture_preflight._capture_mode_hook_override` already uses.
# ---------------------------------------------------------------------------
@contextmanager
def _force_aa_flag_off_hook():
    inner_factory = G._s39_configure_hook

    def wrapped_factory(arm_key):
        inner_hook = inner_factory(arm_key)

        def hook(planning, sed, candidate, report):
            inner_hook(planning=planning, sed=sed, candidate=candidate, report=report)
            admm_params = planning.params.admm
            admm_params.anderson_acceleration = dict(admm_params.anderson_acceleration)
            admm_params.anderson_acceleration['enabled'] = False
            applied = (admm_params.anderson_acceleration['enabled'] is False)
            report.setdefault('rule_eleven_checklist', {})['s43_anderson_acceleration_explicitly_off'] = applied
            report['rule_eleven_checklist']['s43_anderson_acceleration_settings_in_force'] = dict(
                admm_params.anderson_acceleration)
            if not applied:
                raise RuntimeError('S43 flag-off gate: anderson_acceleration.enabled did not read False')
        return hook

    G._s39_configure_hook = wrapped_factory
    try:
        yield
    finally:
        G._s39_configure_hook = inner_factory


# ---------------------------------------------------------------------------
# Zero-AA-call instrumentation: wraps every AA-module function/method that
# reads or writes the iterate, plus the two orchestration helpers in
# `shared_resources_planning.py`, with a pass-through call counter.
# `anderson_acceleration_enabled` (the flag read) is counted SEPARATELY,
# informationally -- it does not touch the iterate and is not part of the
# zero-call gate.
# ---------------------------------------------------------------------------
_MODULE_LEVEL_TARGETS = (
    (AA, 'build_iterate_layout'),
    (AA, 'collect_w'),
    (AA, 'write_back_w'),
    (AA, 'combined_scaled_residual'),
    (AA, '_antisymmetry_check'),
    (srp, '_get_admm_rho_channel_scalars_for_aa'),
    (srp, '_anderson_acceleration_cycle_step'),
)
_STATE_METHOD_NAMES = ('__init__', 'step', 'clear_memory', 'clear_for_rho_change',
                       'skip_on_failure', 'memory_size')


@contextmanager
def _aa_call_counters():
    counts = {}
    originals = []

    for mod, attr in _MODULE_LEVEL_TARGETS:
        orig = getattr(mod, attr)
        counts[attr] = 0

        def make_wrapper(name=attr, orig=orig):
            def wrapper(*args, **kwargs):
                counts[name] += 1
                return orig(*args, **kwargs)
            return wrapper

        setattr(mod, attr, make_wrapper())
        originals.append((mod, attr, orig))

    state_cls = AA.AndersonAccelerationState
    for m in _STATE_METHOD_NAMES:
        orig = getattr(state_cls, m)
        key = f'AndersonAccelerationState.{m}'
        counts[key] = 0

        def make_method_wrapper(name=key, orig=orig):
            def wrapper(self, *args, **kwargs):
                counts[name] += 1
                return orig(self, *args, **kwargs)
            return wrapper

        setattr(state_cls, m, make_method_wrapper())
        originals.append((state_cls, m, orig))

    # Informational only, NOT part of the zero-call gate (does not touch the iterate).
    orig_flag_fn = AA.anderson_acceleration_enabled
    counts['anderson_acceleration_enabled (informational, not gated)'] = 0

    def flag_wrapper(*args, **kwargs):
        counts['anderson_acceleration_enabled (informational, not gated)'] += 1
        return orig_flag_fn(*args, **kwargs)

    AA.anderson_acceleration_enabled = flag_wrapper
    originals.append((AA, 'anderson_acceleration_enabled', orig_flag_fn))

    try:
        yield counts
    finally:
        for mod, attr, orig in originals:
            setattr(mod, attr, orig)


@contextmanager
def _capture_returned_state():
    """Wraps the unqualified module-level call
    `shared_resources_planning._run_operational_planning` (the SAME
    interception technique `s39_exempt_until_capture_hooks`/`s38_pf_capture_
    hooks` already use on other module-level functions) purely to grab the
    `state` dict it returns (last tuple element) -- calls straight through,
    changes nothing."""
    inner = srp._run_operational_planning
    holder = {'state': None, 'calls': 0}

    def wrapper(*args, **kwargs):
        result = inner(*args, **kwargs)
        holder['state'] = result[-1]
        holder['calls'] += 1
        return result

    srp._run_operational_planning = wrapper
    try:
        yield holder
    finally:
        srp._run_operational_planning = inner


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _load_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _classify_diffs(diffs):
    """Split raw `_diff`/`_reproduction_check` output into `aa_new_field_
    diffs` (the reference lacks the key entirely AND the new run's value
    matches the documented flag-off constant) and `genuine_diffs` (gates).
    A diff is classified as `aa_new_field_diffs` ONLY when BOTH conditions
    hold -- a field that happens to share a name in `AA_NEW_DIAGNOSTIC_
    FIELD_NAMES` but differs for any other reason (reference not `<MISSING>`,
    or the new value is not the documented flag-off constant) is a genuine
    diff and gates, exactly like any unexplained difference would."""
    aa_new_field_diffs, genuine_diffs = [], []
    for d in diffs:
        field = str(d.get('field', ''))
        leaf = field.rsplit('.', 1)[-1]
        # `CP._diff` ALWAYS names the two compared values 'legacy' (its first
        # positional argument) and 'lightweight' (its second) regardless of
        # what the caller's own variables are named -- confirmed by reading
        # `_diff`'s three `diffs.append({...})` sites, all fixed key names.
        # Every call site in this script passes the OLD/reference value
        # first, so 'legacy' is always the reference side here.
        if leaf in AA_NEW_DIAGNOSTIC_FIELD_NAMES:
            ref_val = d.get('legacy')
            new_val = d.get('lightweight')
            expected_new = AA_NEW_FIELD_EXPECTED_FLAG_OFF_VALUE.get(leaf, None)
            is_expected = (ref_val == '<MISSING>' and new_val == expected_new)
            if is_expected:
                aa_new_field_diffs.append(d)
                continue
        genuine_diffs.append(d)
    return aa_new_field_diffs, genuine_diffs


def _find_peak_rss_keys(obj, path=''):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{path}.{k}' if path else str(k)
            if 'peak_rss' in str(k):
                hits.append(p)
            hits.extend(_find_peak_rss_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_find_peak_rss_keys(v, f'{path}[{i}]'))
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    failures = _check_preconditions()
    if failures:
        for f in failures:
            print(f'[S43-FLAGOFF-GATE PRECONDITION FAILED] {f}')
        raise SystemExit(1)
    print('[S43-FLAGOFF-GATE] preconditions passed (no lock, no forbidden process, '
          'fresh output dir, production files clean, both references present).')

    G._acquire_exclusive_run_lock()
    os.makedirs(OUT_DIR, exist_ok=True)
    started = time.time()

    print(f'[S43-FLAGOFF-GATE] run {ARM_KEY} for {NUM_CYCLES} cycles, '
          'anderson_acceleration.enabled EXPLICITLY set to False, instrumented '
          'for zero AA calls.')
    with _force_aa_flag_off_hook(), _aa_call_counters() as aa_call_counts, \
         _capture_returned_state() as state_holder:
        report, report_path = G.run_s39_arm(
            ARM_KEY, num_max_iters_override=NUM_CYCLES, output_root_override=OUT_DIR,
            mode_label_override=MODE_LABEL)

    total_aa_calls = sum(v for k, v in aa_call_counts.items() if 'informational' not in k)
    zero_aa_calls = (total_aa_calls == 0)
    print(f"[S43-FLAGOFF-GATE] cycles_run={report['cycles_run']} recourse={report['recourse']} "
          f'aa_call_counts={aa_call_counts} total_gated_aa_calls={total_aa_calls}')

    state = state_holder['state'] or {}
    peak_rss_ru_maxrss = state.get('peak_rss_ru_maxrss')
    peak_rss_platform_units = state.get('peak_rss_platform_units')
    print(f"[S43-FLAGOFF-GATE] state['peak_rss_ru_maxrss']={peak_rss_ru_maxrss!r} "
          f"platform_units={peak_rss_platform_units!r} (from the returned state dict, "
          "NOT necessarily present in the written report -- checked below).")

    # ---- (1) truncated comparison vs D's own committed trajectory ---------
    d_repro = _reproduction_check(report)
    d_aa_new_field_diffs, d_genuine_diffs = _classify_diffs(d_repro['diffs'])
    print(f"[S43-FLAGOFF-GATE] vs D committed (truncated, mode={d_repro['mode']}): "
          f"n_diffs_raw={d_repro['n_diffs']} aa_new_field_diffs={len(d_aa_new_field_diffs)} "
          f"genuine_diffs={len(d_genuine_diffs)} "
          f"n_provenance_diffs_reported_not_gating={d_repro['n_provenance_diffs']}")

    # ---- (2) full comparison vs clone_preflight_v2/lightweight ------------
    lw_report_path = os.path.join(LIGHTWEIGHT_REFERENCE_DIR, f'g_{ARM_LABEL}.json')
    lw_report = _load_json(lw_report_path)
    lw_report_diff_raw = CP._diff(lw_report, report, 'report') if lw_report is not None else None
    lw_aa_new_field_diffs, lw_genuine_diffs = (
        _classify_diffs(lw_report_diff_raw) if lw_report_diff_raw is not None else ([], []))
    print(f"[S43-FLAGOFF-GATE] vs clone_preflight_v2/lightweight report: "
          f"n_diffs_raw={len(lw_report_diff_raw) if lw_report_diff_raw is not None else None} "
          f"aa_new_field_diffs={len(lw_aa_new_field_diffs)} genuine_diffs={len(lw_genuine_diffs)}")

    lw_artifact_diffs = {}
    for fname in CP.ARTIFACT_FILES:
        a_path = os.path.join(LIGHTWEIGHT_REFERENCE_DIR, fname)
        b_path = os.path.join(OUT_DIR, fname)
        a_json, b_json = _load_json(a_path), _load_json(b_path)
        if a_json is None or b_json is None:
            lw_artifact_diffs[fname] = {
                'error': f'missing artifact: reference_exists={a_json is not None} mine_exists={b_json is not None}',
                'aa_new_field_diffs': [], 'genuine_diffs': [], 'n_genuine_diffs': None,
            }
            continue
        raw = CP._diff(a_json, b_json, fname)
        aa_new, genuine = _classify_diffs(raw)
        lw_artifact_diffs[fname] = {
            'aa_new_field_diffs': aa_new, 'genuine_diffs': genuine, 'n_genuine_diffs': len(genuine),
        }

    lw_sidecar_diffs = {}
    for fname in CP.SIDECAR_JSONL_FILES:
        a_path = os.path.join(LIGHTWEIGHT_REFERENCE_DIR, fname)
        b_path = os.path.join(OUT_DIR, fname)
        a_rows, b_rows = _load_jsonl(a_path), _load_jsonl(b_path)
        if a_rows is None or b_rows is None:
            lw_sidecar_diffs[fname] = {
                'error': f'missing sidecar: reference_exists={a_rows is not None} mine_exists={b_rows is not None}',
                'aa_new_field_diffs': [], 'genuine_diffs': [], 'n_genuine_diffs': None,
            }
            continue
        raw = CP._diff(a_rows, b_rows, fname)
        aa_new, genuine = _classify_diffs(raw)
        lw_sidecar_diffs[fname] = {
            'aa_new_field_diffs': aa_new, 'genuine_diffs': genuine, 'n_genuine_diffs': len(genuine),
        }

    lw_all_genuine_n = (
        len(lw_genuine_diffs)
        + sum((v['n_genuine_diffs'] or 0) for v in lw_artifact_diffs.values())
        + sum((v['n_genuine_diffs'] or 0) for v in lw_sidecar_diffs.values())
    )
    lw_all_artifacts_present = (
        all('error' not in v for v in lw_artifact_diffs.values())
        and all('error' not in v for v in lw_sidecar_diffs.values())
    )

    # ---- peak_rss reach check: does the NEW state-only field ever land in a
    #      compared, WRITTEN artifact? ----------------------------------------
    peak_rss_hits_in_report = _find_peak_rss_keys(report)
    peak_rss_hits_in_reference = _find_peak_rss_keys(lw_report) if lw_report is not None else []

    # ---- aa_* field pattern validation (every row) -------------------------
    aa_pattern_ok = True
    aa_pattern_violations = []
    for row in report.get('cycle_trajectory') or []:
        if row.get('aa_enabled') is not False:
            aa_pattern_ok = False
            aa_pattern_violations.append({'cycle': row.get('cycle'), 'field': 'aa_enabled', 'value': row.get('aa_enabled')})
        for field in AA_NEW_DIAGNOSTIC_FIELD_NAMES - {'aa_enabled'}:
            if row.get(field) is not None:
                aa_pattern_ok = False
                aa_pattern_violations.append({'cycle': row.get('cycle'), 'field': field, 'value': row.get(field)})

    gate_pass = (
        zero_aa_calls
        and len(d_genuine_diffs) == 0
        and (lw_report is not None and lw_all_genuine_n == 0 and lw_all_artifacts_present)
        and aa_pattern_ok
        and len(peak_rss_hits_in_report) == 0
    )

    payload = {
        'stage': 'P5.15 Step 3.7 integration -- Task 1: flag-off two-cycle bitwise gate',
        'authority': [
            'PLANNER_BRIEF_2026-09-13.md Addendum 23/24',
            'data/SRP1/Results/P515S41/frozen_s41_hull_aa_spec_v12_6e5a546f.json',
            'data/SRP1/Results/P515S42/frozen_s42_helper_aa_spec_v13_2cab76e8.json',
        ],
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'arm': ARM_KEY, 'num_cycles': NUM_CYCLES, 'mode_label': MODE_LABEL,
        'out_dir': os.path.relpath(OUT_DIR, REPO),
        'report_path': os.path.relpath(report_path, REPO),
        'anderson_acceleration_explicitly_off': report.get('rule_eleven_checklist', {}).get(
            's43_anderson_acceleration_explicitly_off'),
        'anderson_acceleration_settings_in_force': report.get('rule_eleven_checklist', {}).get(
            's43_anderson_acceleration_settings_in_force'),
        'aa_call_counts': aa_call_counts,
        'total_gated_aa_calls': total_aa_calls,
        'zero_aa_calls': zero_aa_calls,
        'peak_rss': {
            'state_peak_rss_ru_maxrss': peak_rss_ru_maxrss,
            'state_peak_rss_platform_units': peak_rss_platform_units,
            'peak_rss_keys_found_in_this_run_report': peak_rss_hits_in_report,
            'peak_rss_keys_found_in_lightweight_reference_report': peak_rss_hits_in_reference,
            'appears_in_any_compared_artifact': bool(peak_rss_hits_in_report or peak_rss_hits_in_reference),
            'reading': (
                'peak_rss_ru_maxrss/peak_rss_platform_units are added to the `state` dict '
                'shared_resources_planning._run_operational_planning returns, NOT copied into '
                'the report run_admm_arm writes to g_<label>.json -- confirmed empty above by '
                'a recursive key scan of the actual written/compared JSON, not by static reading '
                'of the source alone.'),
        },
        'aa_new_diagnostic_field_names': sorted(AA_NEW_DIAGNOSTIC_FIELD_NAMES),
        'aa_field_pattern_ok': aa_pattern_ok,
        'aa_field_pattern_violations': aa_pattern_violations,
        'vs_d_committed_truncated': {
            **{k: v for k, v in d_repro.items() if k != 'diffs'},
            'aa_new_field_diffs': d_aa_new_field_diffs,
            'n_aa_new_field_diffs': len(d_aa_new_field_diffs),
            'genuine_diffs': d_genuine_diffs,
            'n_genuine_diffs': len(d_genuine_diffs),
        },
        'vs_clone_preflight_v2_lightweight': {
            'reference_dir': os.path.relpath(LIGHTWEIGHT_REFERENCE_DIR, REPO),
            'reference_report_exists': lw_report is not None,
            'report_diff': {
                'aa_new_field_diffs': lw_aa_new_field_diffs, 'n_aa_new_field_diffs': len(lw_aa_new_field_diffs),
                'genuine_diffs': lw_genuine_diffs, 'n_genuine_diffs': len(lw_genuine_diffs),
            },
            'artifact_diffs': lw_artifact_diffs,
            'sidecar_diffs': lw_sidecar_diffs,
            'total_genuine_diffs': lw_all_genuine_n,
            'all_artifacts_present': lw_all_artifacts_present,
            'note_on_intervening_production_changes': (
                'This reference (commit 8214be0d) predates BOTH the Step 3.6 bound-restore fix '
                '(16a19456) and AA integration (9a965494); any genuine (non-aa_*) diff found here '
                'is reported as-is, with its likely attribution stated, never rationalized away.'),
        },
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'intentional_difference_suffixes': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
        'gate_pass': gate_pass,
        'wall_clock_s': time.time() - started,
    }

    results_path = os.path.join(OUT_DIR, 'flagoff_gate_results.json')
    _refuse_overwrite(results_path)
    with open(results_path, 'w') as handle:
        json.dump(payload, handle, indent=1, default=str)
    print(f'[S43-FLAGOFF-GATE] wrote {results_path}')
    print(f'[S43-FLAGOFF-GATE] GATE_PASS={gate_pass} zero_aa_calls={zero_aa_calls} '
          f'd_genuine_diffs={len(d_genuine_diffs)} lw_genuine_diffs={lw_all_genuine_n} '
          f'aa_field_pattern_ok={aa_pattern_ok} '
          f'peak_rss_appears_anywhere={bool(peak_rss_hits_in_report or peak_rss_hits_in_reference)}')
    if not gate_pass:
        print('[S43-FLAGOFF-GATE] *** GATE FAILED *** -- first genuine diffs:')
        for d in (d_genuine_diffs + lw_genuine_diffs)[:20]:
            print(f'  [FIRST DIFFS] {d}')

    manifest = {os.path.relpath(results_path, REPO): CP._sha256_file(results_path)}
    manifest_path = os.path.join(OUT_DIR, 'manifest_sha256.json')
    _refuse_overwrite(manifest_path)
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f'[S43-FLAGOFF-GATE] wrote {manifest_path}')

    if not gate_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
