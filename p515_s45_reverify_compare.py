"""
P5.15 Addendum 27 item 1 (W4) -- the RE-VERIFICATION COMPARATOR: ZERO SOLVES.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 27 item 1; frozen spec v15
`data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json`,
`configuration.reverification_gate`:
    "a case-file-alone C* evaluation through the campaign harness (no
    overrides) reproduces the committed AA C* evaluation (campaign_s44_aa_variant,
    107 cycles, 650,982,939.9389359) bitwise on every numeric trajectory field
    and cost; tie-order in the recourse-jump top-10 classified with
    p515_s44_tie_classifier; provenance non-gating".

Compares
    CANDIDATE  data/SRP1/Results/P515S45/reverify_aa_c_star/evals/3e741dac72c9e1bc_c_star_aa_case_file
               (campaign s45_reverify_aa_c_star, spec 4a214b99; the case file alone)
    REFERENCE  data/SRP1/Results/P515S44/campaign_s44_aa_variant/evals/837fc982565dbba3_c_star_aa_keep_memory
               (AA keep_memory as an override; certified at 107, 650,982,939.9389359)
Modelled on `p515_s44_gate.compare_c_star_to_d`. Comparator conventions BY
IMPORT: `p515_s40_clone_capture_preflight` (`_diff`, `EXCLUDE_KEY_NAMES`,
`EXCLUDE_DOTTED_SUFFIXES`, `INTENTIONAL_DIFF_SUFFIXES`, `ARTIFACT_FILES`,
`SIDECAR_JSONL_FILES`, `_load_json`, `_load_jsonl`), `p515_s43_aa_flagoff_gate.
_classify_diffs`, `p515_s44_tie_classifier.reclassify_sidecar_diffs`.

Files compared (GATING), in addition to `CP.ARTIFACT_FILES` and
`CP.SIDECAR_JSONL_FILES`: `aa_per_cycle.jsonl` (both runs have AA on),
`per_cycle_record.jsonl`, `evaluation_record.json`.
SUPPLEMENTARY (reported, NOT gating -- the `p515_s44_gate` convention; these
rows carry absolute log / snapshot paths and mtimes): leak_classification /
network_failures / esso_recovery_events / frozen_snapshots JSONL.

CLASSIFICATION of every raw `_diff` difference:
  provenance (reported, listed, NON-GATING):
    - the `rule_eleven_checklist` subtree (`FG._classify_diffs`);
    - in evaluation_record.json, the top-level keys of `RECORD_PROVENANCE_KEYS`
      below (campaign / eval identity, spec hashes, override-vs-case-file
      configuration bookkeeping incl. the two Addendum 27 fields
      `anderson_acceleration_effective_in_child` / `case_file_sha256_in_child`,
      case-file sha256 and last commit, run environment, wall time / peak RSS),
      and `aa_per_cycle.path`;
    - in evaluation_record.json, the post-certification bookkeeping
      (`post_certification`, `post_certification_path`,
      `post_certification_capture_checklist_asserted_before_run`) ONLY IF the
      candidate record holds None for all three (the reference campaign
      requested a post-certification step; this campaign's frozen spec has
      `post_certification: null`); otherwise genuine.
  tie_order (reported, NON-GATING): a diff inside the recourse-jump sidecar's
    `objective_component_block_deltas` / `block_deltas` whose whole row is
    explained by `p515_s44_tie_classifier` (identical / resort / straddle).
    `FG`'s alias-pair bucket is NOT trusted on its own: its diffs are passed
    through the classifier like any other and stay genuine unless explained.
  genuine (GATING): everything else, including `FG`'s `aa_new_field` bucket
    (both runs have AA on; a reference lacking an aa_* field is a real
    difference here).
GATE (all must hold): every required file present on both sides; 0 genuine
diffs; candidate certified, cycles_run == certification_cycle == 107 (record
and report), certified_cost == report gross_operational_cost ==
650982939.9389359 EXACTLY; candidate record `anderson_acceleration_effective_in_child`
== EXPECTED_AA and reference record `overrides_applied_in_child.anderson_acceleration`
== EXPECTED_AA; reference files match their pinned sha256; the armed
SolveProfileGuard(permitted=()) verified at exactly 0.

CAPTURE-PATH RULE: before any comparison, `capture_path_checklist` asserts that
every file and record field the gate needs exists on both sides (and that the
comparator functions resolve); real mode FAILS FAST if any item is False.

WAIT-FREE GUARD (real mode): refuses if the campaign lock
`.p515_s44_campaign.lock` exists or the candidate `evaluation_record.json` is
missing -- it never waits and never touches the running campaign.

OUTPUT (write-once; refuses if the directory exists):
    data/SRP1/Results/P515S45/reverify_aa_c_star_compare/compare.json
    data/SRP1/Results/P515S45/reverify_aa_c_star_compare/manifest_sha256.json
and the launch log (sibling, created by the shell under noclobber).

EXACT COMMAND (real comparison; repo root; attached, both streams captured):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u \\
        p515_s45_reverify_compare.py \\
        > data/SRP1/Results/P515S45/reverify_aa_c_star_compare_launch.log 2>&1

SELF-TEST MODE (`--self-test NAME --candidate-dir DIR --expect pass|fail`):
writes to data/SRP1/Results/P515S45/reverify_compare_selftest/NAME/ (write-once),
never to the real output. Declared differences from real mode, recorded in the
output: (1) capture-path failures are recorded and the comparison proceeds
over the files present on both sides (a missing file is a gating error); (2)
the candidate's effective AA dict falls back to
`overrides_applied_in_child.anderson_acceleration` when the candidate record
predates `anderson_acceleration_effective_in_child`; (3) the wait-free guard
is not applied, and the candidate must NOT lie inside the running campaign's
root. `--expect fail` is met only if the gate fails WITH genuine diffs > 0.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15-S45 re-verification comparator (zero solves)').install()

import p515_s40_clone_capture_preflight as CP  # noqa: E402 -- comparator conventions (8f5cff48), BY IMPORT
import p515_s43_aa_flagoff_gate as FG  # noqa: E402 -- diff classification, BY IMPORT
import p515_s44_tie_classifier as TC  # noqa: E402 -- re-sort-and-straddle tie classifier, BY IMPORT

_RESULTS = os.path.join(REPO, 'data', 'SRP1', 'Results')
REFERENCE_DIR = os.path.join(_RESULTS, 'P515S44', 'campaign_s44_aa_variant', 'evals',
                             '837fc982565dbba3_c_star_aa_keep_memory')
REFERENCE_PINS = {  # = p515_s45_reverify_aa_c_star.AA_C_STAR_REFERENCE (also checked against the frozen campaign spec)
    'evaluation_record.json': '6e90d4424df3405b0cc066bba1f57c28666a1524e5cdcc62890c6e6a74e69a4f',
    'component_levels_terminal.json': '8452f6261b1bf1fe84d4dfd8a177fa8f4145ade4c042a4904a29d4483a745782',
}
CAMPAIGN_ROOT = os.path.join(_RESULTS, 'P515S45', 'reverify_aa_c_star')
CAMPAIGN_SPEC = os.path.join(CAMPAIGN_ROOT, 'campaign_spec_s45_reverify_aa_c_star_4a214b99.json')
CAMPAIGN_SPEC_SHA256 = '4a214b99df34e58791d31d842f2ad6c5293f83e82064d52320934b6d32b6e58d'
CANDIDATE_DIR = os.path.join(CAMPAIGN_ROOT, 'evals', '3e741dac72c9e1bc_c_star_aa_case_file')
CAMPAIGN_LOCK_PATH = os.path.join(REPO, '.p515_s44_campaign.lock')
SPEC_V15 = os.path.join(_RESULTS, 'P515S45', 'frozen_s45_phaseA_spec_v15_5feefd7b.json')
SPEC_V15_SHA256 = '5feefd7b642fc3d480156ad5e52ed6e1cf9d6698cfd40dbb33389bab6e6229fe'
OUT_DIR = os.path.join(_RESULTS, 'P515S45', 'reverify_aa_c_star_compare')
SELFTEST_ROOT = os.path.join(_RESULTS, 'P515S45', 'reverify_compare_selftest')

C_STAR_KEY = '578636daa6d6360d6701764c73ddf795e53c2c37e511e21be1400024f8f6350c'
EXPECTED_CYCLES = 107
EXPECTED_COST = 650982939.9389359
EXPECTED_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}

RECORD_FILE = 'evaluation_record.json'
REPORT_FILE = 'g_s39_D.json'
RECOURSE_JUMP = 'recourse_jump_sidecar_baseline.jsonl'
EXTRA_GATING_JSONL = ('aa_per_cycle.jsonl', 'per_cycle_record.jsonl')
GATING_JSON = tuple(CP.ARTIFACT_FILES) + (RECORD_FILE,)
GATING_JSONL = tuple(CP.SIDECAR_JSONL_FILES) + EXTRA_GATING_JSONL
SUPPLEMENTARY_JSONL = ('leak_classification_s39_D.jsonl', 'network_failures_s39_D.jsonl',
                       'esso_recovery_events_s39_D.jsonl', 'frozen_snapshots_s39_D.jsonl')
SUPPLEMENTARY_PATH_LIKE_KEYS = ('log_path', 'primary_log', 'recovery_log', 'tier2_log', 'path', 'mtime_utc')
NOT_COMPARED = {
    'esso_models_s39_D.pkl': 'binary pickle; its size is compared via the report field esso_models_pickle.bytes; '
                             'sha256 equality recorded (informational)',
    'post_certification.json': 'reference only (post-certification requested there, not in this campaign)',
    'hull_bound_detail.json': 'reference only (post-certification)',
    'certified_models.pkl': 'reference only (post-certification)',
    'child_manifest_sha256.json': 'hashes of files under each run\'s own eval dir (paths differ by construction)',
    'launch.json / heartbeat_s39_D.json / child_stdout.log / child_stderr.log / stdout_s39_D.log / exit_code.txt / '
    'wait4_rusage.json / esso_capture/ / results/': 'process bookkeeping, logs and working files',
}

# evaluation_record.json top-level keys whose whole subtree is provenance, with the reason.
RECORD_PROVENANCE_KEYS = {
    'campaign_id': 'campaign identity',
    'campaign_spec_path': 'campaign identity (spec path)',
    'campaign_spec_sha256': 'campaign identity (spec hash)',
    'candidate_label': 'campaign identity (label)',
    'working_dir_ids': 'campaign identity (working-dir ids derive from campaign id + eval key)',
    'eval_dir': 'campaign identity (eval dir)',
    'eval_key': 'campaign identity (eval key = candidate key + effective overrides + case-file AA declaration)',
    'report_path': 'path under the run\'s own eval dir',
    'per_cycle_record_path': 'path under the run\'s own eval dir',
    'campaign_lock_seen_by_child': 'run provenance (lock content: pid, campaign id, spec hash, start time)',
    'child_pid': 'run provenance (pid)',
    'parent_pid': 'run provenance (pid)',
    'configuration': 'configuration bookkeeping (name, note, case_file_sha256, case_file_last_commit, '
                     'case_file_anderson_acceleration declaration)',
    'evaluation_overrides_effective': 'configuration bookkeeping (override vs case file)',
    'overrides_applied_in_child': 'configuration bookkeeping (override vs case file); AA dict gated separately',
    'anderson_acceleration_effective_in_child': 'configuration bookkeeping (Addendum 27 field); AA dict gated '
                                                'separately',
    'case_file_sha256_in_child': 'configuration bookkeeping (Addendum 27 field; case-file sha256)',
    'configuration_checks_in_child': 'configuration bookkeeping (check names differ: AA-off-before-overrides vs '
                                     'case-file-AA-matches-declaration)',
    'record_capture_checklist_asserted_before_run': 'run provenance (rule-eleven checklist)',
    'thread_caps_seen_by_child': 'run provenance (environment)',
    'PYTHONHASHSEED_in_child': 'run provenance (environment)',
    'nlp_solver_path_in_child': 'run provenance (environment)',
    'schema': 'record schema version',
    'wall_time_s': 'run measurement (wall time; not reproducible)',
    'peak_rss': 'run measurement (peak RSS; not reproducible)',
}
RECORD_PROVENANCE_DOTTED = {'aa_per_cycle.path': 'path under the run\'s own eval dir'}
POST_CERTIFICATION_KEYS = ('post_certification', 'post_certification_path',
                           'post_certification_capture_checklist_asserted_before_run')

AUTHORITY = [
    'PLANNER_BRIEF_2026-09-13.md Addendum 27 item 1 (re-verification; W4 comparator)',
    'data/SRP1/Results/P515S45/frozen_s45_phaseA_spec_v15_5feefd7b.json configuration.reverification_gate',
    'p515_s44_gate.compare_c_star_to_d (model); p515_s44_tie_classifier (tie order)',
]
MAX_LISTED_GENUINE_PER_FILE = 200


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _rel(path):
    return os.path.relpath(path, REPO)


def _sha(path):
    return CP._sha256_file(path)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _pattern(field):
    return re.sub(r'\[\d+\]', '[*]', field)


def _record_subfield(field):
    """'evaluation_record.json.configuration.case_file_sha256' -> ('configuration', 'configuration.case_file_sha256')."""
    prefix = RECORD_FILE + '.'
    if not field.startswith(prefix):
        return None, None
    rest = field[len(prefix):]
    top = rest.split('[', 1)[0].split('.', 1)[0]
    return top, rest


def _classify(fname, a, b, candidate_record):
    raw = CP._diff(a, b, fname)
    fg_prov, fg_aa_new, fg_alias, fg_genuine = FG._classify_diffs(raw)
    provenance = [dict(d, reason='rule_eleven_checklist subtree') for d in fg_prov]
    genuine = list(fg_aa_new) + list(fg_alias) + list(fg_genuine)
    if fname == RECORD_FILE:
        post_cert_absent = all((candidate_record or {}).get(k) is None for k in POST_CERTIFICATION_KEYS)
        kept = []
        for d in genuine:
            top, rest = _record_subfield(str(d.get('field', '')))
            if top in RECORD_PROVENANCE_KEYS:
                provenance.append(dict(d, reason=RECORD_PROVENANCE_KEYS[top]))
            elif rest in RECORD_PROVENANCE_DOTTED:
                provenance.append(dict(d, reason=RECORD_PROVENANCE_DOTTED[rest]))
            elif top in POST_CERTIFICATION_KEYS and post_cert_absent:
                provenance.append(dict(d, reason='post-certification requested in the reference campaign only '
                                                 '(candidate record holds None for all three keys)'))
            else:
                kept.append(d)
        genuine = kept
    tie_order, row_classes = [], {}
    if fname == RECOURSE_JUMP and genuine:
        tie_order, genuine, row_classes = TC.reclassify_sidecar_diffs(genuine, a, b, fname)
    patterns = {}
    for d in genuine:
        p = _pattern(str(d.get('field', '')))
        patterns[p] = patterns.get(p, 0) + 1
    return {
        'n_raw_diffs': len(raw),
        'fg_buckets_raw_counts': {'provenance': len(fg_prov), 'aa_new_field': len(fg_aa_new),
                                  'known_tie_break_alias': len(fg_alias), 'genuine': len(fg_genuine)},
        'n': {'provenance': len(provenance), 'tie_order': len(tie_order), 'genuine': len(genuine)},
        'provenance_diffs': provenance,
        'tie_order_diffs': tie_order,
        'tie_order_row_classes': row_classes,
        'genuine_field_patterns': dict(sorted(patterns.items())),
        'genuine_diffs_first': genuine[:MAX_LISTED_GENUINE_PER_FILE],
        'genuine_diffs_listed_truncated': len(genuine) > MAX_LISTED_GENUINE_PER_FILE,
    }


def _strip_keys(obj, keys):
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(v, keys) for v in obj]
    return obj


def _candidate_aa(candidate_record, self_test):
    rec = candidate_record or {}
    if 'anderson_acceleration_effective_in_child' in rec:
        return rec.get('anderson_acceleration_effective_in_child'), 'anderson_acceleration_effective_in_child'
    if self_test:
        return ((rec.get('overrides_applied_in_child') or {}).get('anderson_acceleration'),
                'SELF-TEST FALLBACK: overrides_applied_in_child.anderson_acceleration')
    return None, 'anderson_acceleration_effective_in_child (absent)'


def capture_path_checklist(candidate_dir, self_test):
    """Rule eleven: every file and field the gate needs, on both sides, BEFORE comparing."""
    items = {}
    for side, d in (('reference', REFERENCE_DIR), ('candidate', candidate_dir)):
        for f in GATING_JSON + GATING_JSONL:
            items[f'{side}_file_{f}'] = os.path.isfile(os.path.join(d, f))
    ref_rec = CP._load_json(os.path.join(REFERENCE_DIR, RECORD_FILE)) or {}
    cand_rec = CP._load_json(os.path.join(candidate_dir, RECORD_FILE)) or {}
    cand_rep = CP._load_json(os.path.join(candidate_dir, REPORT_FILE)) or {}
    for side, rec in (('reference', ref_rec), ('candidate', cand_rec)):
        for k in ('status', 'certification', 'cycles_run', 'certification_cycle', 'certified_cost', 'candidate_key',
                  'rule_ten'):
            items[f'{side}_record_field_{k}'] = k in rec
    items['reference_record_field_overrides_applied_in_child.anderson_acceleration'] = (
        'anderson_acceleration' in (ref_rec.get('overrides_applied_in_child') or {}))
    _aa, aa_source = _candidate_aa(cand_rec, self_test)
    items['candidate_record_field_effective_anderson_acceleration'] = (
        'anderson_acceleration_effective_in_child' in cand_rec
        or (self_test and 'anderson_acceleration' in (cand_rec.get('overrides_applied_in_child') or {})))
    items['candidate_record_field_case_file_sha256_in_child'] = ('case_file_sha256_in_child' in cand_rec
                                                                 or self_test)
    for k in ('cycles_run', 'gross_operational_cost', 'cycle_trajectory'):
        items[f'candidate_report_field_{k}'] = k in cand_rep
    items['comparator_CP__diff'] = callable(getattr(CP, '_diff', None))
    items['comparator_FG__classify_diffs'] = callable(getattr(FG, '_classify_diffs', None))
    items['tie_classifier_TC_reclassify_sidecar_diffs'] = callable(getattr(TC, 'reclassify_sidecar_diffs', None))
    return items, aa_source


def compare(candidate_dir, self_test):
    ref_rec = CP._load_json(os.path.join(REFERENCE_DIR, RECORD_FILE))
    cand_rec = CP._load_json(os.path.join(candidate_dir, RECORD_FILE))
    ref_rep = CP._load_json(os.path.join(REFERENCE_DIR, REPORT_FILE))
    cand_rep = CP._load_json(os.path.join(candidate_dir, REPORT_FILE))
    files = {}
    for fname in GATING_JSON:
        a = ref_rec if fname == RECORD_FILE else (ref_rep if fname == REPORT_FILE else
                                                  CP._load_json(os.path.join(REFERENCE_DIR, fname)))
        b = cand_rec if fname == RECORD_FILE else (cand_rep if fname == REPORT_FILE else
                                                   CP._load_json(os.path.join(candidate_dir, fname)))
        if a is None or b is None:
            files[fname] = {'error': f'missing: reference={a is not None} candidate={b is not None}'}
            continue
        files[fname] = _classify(fname, a, b, cand_rec)
        _log(f'[S45-CMP]   {fname}: {files[fname]["n"]}')
    for fname in GATING_JSONL:
        a = CP._load_jsonl(os.path.join(REFERENCE_DIR, fname))
        b = CP._load_jsonl(os.path.join(candidate_dir, fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: reference={a is not None} candidate={b is not None}'}
            _log(f'[S45-CMP]   {fname}: {files[fname]["error"]}')
            continue
        c = _classify(fname, a, b, cand_rec)
        c['n_rows'] = {'reference': len(a), 'candidate': len(b)}
        files[fname] = c
        _log(f'[S45-CMP]   {fname}: rows={c["n_rows"]} {c["n"]}')
        del a, b
    supplementary = {}
    for fname in SUPPLEMENTARY_JSONL:
        a = CP._load_jsonl(os.path.join(REFERENCE_DIR, fname))
        b = CP._load_jsonl(os.path.join(candidate_dir, fname))
        if a is None or b is None:
            supplementary[fname] = {'error': f'missing: reference={a is not None} candidate={b is not None}'}
            continue
        raw = CP._diff(a, b, fname)
        stripped = CP._diff(_strip_keys(a, SUPPLEMENTARY_PATH_LIKE_KEYS), _strip_keys(b, SUPPLEMENTARY_PATH_LIKE_KEYS),
                            fname)
        supplementary[fname] = {'n_rows': {'reference': len(a), 'candidate': len(b)}, 'n_raw_diffs': len(raw),
                                'n_diffs_ignoring_path_like_keys': len(stripped),
                                'path_like_keys_ignored_in_that_count': list(SUPPLEMENTARY_PATH_LIKE_KEYS),
                                'first_raw_diffs_ignoring_path_like_keys': stripped[:20]}
    pk_ref, pk_cand = os.path.join(REFERENCE_DIR, 'esso_models_s39_D.pkl'), os.path.join(candidate_dir,
                                                                                         'esso_models_s39_D.pkl')
    esso_pickle = {'reference_sha256': _sha(pk_ref) if os.path.isfile(pk_ref) else None,
                   'candidate_sha256': _sha(pk_cand) if os.path.isfile(pk_cand) else None}
    esso_pickle['sha256_equal'] = (esso_pickle['reference_sha256'] is not None
                                   and esso_pickle['reference_sha256'] == esso_pickle['candidate_sha256'])

    totals = {'provenance': 0, 'tie_order': 0, 'genuine': 0}
    for v in files.values():
        for k, n in (v.get('n') or {}).items():
            totals[k] += n
    all_present = all('error' not in v for v in files.values())

    rec, rep = cand_rec or {}, cand_rep or {}
    scalar = {
        'candidate_status_certified': {'value': rec.get('status'), 'expected': 'certified'},
        'candidate_certification.certified': {'value': (rec.get('certification') or {}).get('certified'),
                                              'expected': True},
        'candidate_record_cycles_run': {'value': rec.get('cycles_run'), 'expected': EXPECTED_CYCLES},
        'candidate_record_certification_cycle': {'value': rec.get('certification_cycle'), 'expected': EXPECTED_CYCLES},
        'candidate_report_cycles_run': {'value': rep.get('cycles_run'), 'expected': EXPECTED_CYCLES},
        'candidate_record_certified_cost': {'value': rec.get('certified_cost'), 'expected': EXPECTED_COST},
        'candidate_report_gross_operational_cost': {'value': rep.get('gross_operational_cost'),
                                                    'expected': EXPECTED_COST},
        'candidate_record_candidate_key_is_c_star': {'value': rec.get('candidate_key'), 'expected': C_STAR_KEY},
        'reference_record_certified_cost': {'value': (ref_rec or {}).get('certified_cost'), 'expected': EXPECTED_COST},
        'reference_record_certification_cycle': {'value': (ref_rec or {}).get('certification_cycle'),
                                                 'expected': EXPECTED_CYCLES},
    }
    for v in scalar.values():
        v['match'] = (type(v['value']) is type(v['expected'])) and v['value'] == v['expected']
    cand_aa, cand_aa_source = _candidate_aa(cand_rec, self_test)
    ref_aa = ((ref_rec or {}).get('overrides_applied_in_child') or {}).get('anderson_acceleration')
    aa_checks = {
        'candidate_effective_aa': {'source': cand_aa_source, 'value': cand_aa, 'expected': EXPECTED_AA,
                                   'match': cand_aa == EXPECTED_AA},
        'reference_applied_override_aa': {'source': 'overrides_applied_in_child.anderson_acceleration',
                                          'value': ref_aa, 'expected': EXPECTED_AA, 'match': ref_aa == EXPECTED_AA},
    }
    provenance_fields = sorted({(_pattern(str(d.get('field'))), d['reason'])
                                for v in files.values() for d in v.get('provenance_diffs', [])})
    return {
        'reference_dir': _rel(REFERENCE_DIR), 'candidate_dir': _rel(candidate_dir),
        'instance': {'reference_candidate_key': (ref_rec or {}).get('candidate_key'),
                     'candidate_candidate_key': rec.get('candidate_key'),
                     'reference_candidate_canonical': (ref_rec or {}).get('candidate_canonical'),
                     'candidate_candidate_canonical': rec.get('candidate_canonical')},
        'comparator': ('p515_s40_clone_capture_preflight._diff (8f5cff48) + p515_s43_aa_flagoff_gate._classify_diffs '
                       '+ record provenance keys (this script) + p515_s44_tie_classifier.reclassify_sidecar_diffs'),
        'diff_value_naming': "CP._diff names the reference value 'legacy' and the candidate value 'lightweight'",
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'intentional_difference_suffixes_excluded': sorted(CP.INTENTIONAL_DIFF_SUFFIXES),
        'record_provenance_keys': RECORD_PROVENANCE_KEYS,
        'record_provenance_dotted': RECORD_PROVENANCE_DOTTED,
        'record_post_certification_keys_conditional': list(POST_CERTIFICATION_KEYS),
        'gating_files': list(GATING_JSON + GATING_JSONL),
        'files': files,
        'supplementary_not_gating': supplementary,
        'not_compared': NOT_COMPARED,
        'esso_models_pickle_informational': esso_pickle,
        'totals_by_class': totals,
        'provenance_field_patterns_with_reason': [list(p) for p in provenance_fields],
        'all_gating_files_present': all_present,
        'n_genuine_diffs': totals['genuine'],
        'scalar_checks': scalar,
        'aa_checks': aa_checks,
        'rule_ten_terminal_step_over_threshold': {
            'reference': ((ref_rec or {}).get('rule_ten') or {}).get('terminal_step_over_threshold'),
            'candidate': (rec.get('rule_ten') or {}).get('terminal_step_over_threshold')},
        'objective_convention': 'gross_operational_cost (settlement-excluded; = certified_cost)',
    }


def main():
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', dest='self_test', default=None, metavar='NAME')
    parser.add_argument('--candidate-dir', default=None)
    parser.add_argument('--expect', choices=('pass', 'fail'), default=None)
    args = parser.parse_args()
    self_test = args.self_test is not None
    failures = []
    if self_test:
        if not re.fullmatch(r'[A-Za-z0-9_]+', args.self_test) or not args.candidate_dir or not args.expect:
            parser.error('--self-test NAME (alnum/_) requires --candidate-dir and --expect')
        candidate_dir = os.path.abspath(os.path.join(REPO, args.candidate_dir))
        out_dir = os.path.join(SELFTEST_ROOT, args.self_test)
        if os.path.commonpath([candidate_dir, CAMPAIGN_ROOT]) == CAMPAIGN_ROOT:
            failures.append(f'self-test candidate lies inside the running campaign root: {candidate_dir}')
        if not os.path.isdir(candidate_dir):
            failures.append(f'self-test candidate dir missing: {candidate_dir}')
    else:
        if args.candidate_dir or args.expect:
            parser.error('--candidate-dir/--expect are self-test options')
        candidate_dir, out_dir = CANDIDATE_DIR, OUT_DIR
        # wait-free guard: never wait, never touch the running campaign
        if os.path.exists(CAMPAIGN_LOCK_PATH):
            failures.append(f'campaign lock still exists (evaluation not finished): {CAMPAIGN_LOCK_PATH}')
        if not os.path.isfile(os.path.join(candidate_dir, RECORD_FILE)):
            failures.append(f'candidate evaluation_record.json missing: {_rel(os.path.join(candidate_dir, RECORD_FILE))}')
        dirty = _git(['status', '--porcelain', '--', os.path.basename(__file__), 'p515_s40_clone_capture_preflight.py',
                      'p515_s43_aa_flagoff_gate.py', 'p515_s44_tie_classifier.py'])
        if dirty:
            failures.append(f'comparator scripts not clean in git:\n{dirty}')
    if os.path.exists(out_dir):
        failures.append(f'output directory exists (write-once): {out_dir}')
    ref_hashes = {f: _sha(os.path.join(REFERENCE_DIR, f)) for f in REFERENCE_PINS}
    if ref_hashes != REFERENCE_PINS:
        failures.append(f'reference files do not match their pins: {ref_hashes}')
    with open(CAMPAIGN_SPEC) as handle:
        spec_pin = (json.load(handle).get('extra') or {}).get('reference_aa_c_star_evaluation') or {}
    if (spec_pin.get('evaluation_record_sha256') != REFERENCE_PINS[RECORD_FILE]
            or spec_pin.get('component_levels_terminal_sha256') != REFERENCE_PINS['component_levels_terminal.json']
            or spec_pin.get('certified_cost') != EXPECTED_COST
            or spec_pin.get('certification_cycle') != EXPECTED_CYCLES
            or os.path.join(REPO, spec_pin.get('eval_dir', '')) != REFERENCE_DIR):
        failures.append(f'this script\'s reference pins differ from the frozen campaign spec: {spec_pin}')
    spec_hashes = {'campaign_spec': _sha(CAMPAIGN_SPEC), 'spec_v15': _sha(SPEC_V15)}
    if spec_hashes != {'campaign_spec': CAMPAIGN_SPEC_SHA256, 'spec_v15': SPEC_V15_SHA256}:
        failures.append(f'spec hashes differ from their pins: {spec_hashes}')
    if failures:
        for f in failures:
            _log(f'[S45-CMP REFUSED] {f}')
        raise SystemExit(1)
    _log(f'[S45-CMP] mode={"self-test " + args.self_test if self_test else "REAL"} candidate={_rel(candidate_dir)} '
         f'reference={_rel(REFERENCE_DIR)} out={_rel(out_dir)}')

    checklist, aa_source = capture_path_checklist(candidate_dir, self_test)
    missing = sorted(k for k, v in checklist.items() if not v)
    _log(f'[S45-CMP] capture-path checklist: {len(checklist)} items, {len(missing)} False: {missing}')
    if missing and not self_test:
        _log('[S45-CMP] CAPTURE-PATH CHECK FAILED -- refusing to compare (fail fast)')
        raise SystemExit(1)

    comparison = compare(candidate_dir, self_test)
    guard_failures = GUARD.verify(0)
    gate_items = {
        'capture_path_checklist_all_true': not missing,
        'all_gating_files_present': comparison['all_gating_files_present'],
        'zero_genuine_diffs': comparison['n_genuine_diffs'] == 0,
        'scalar_checks_all_match': all(v['match'] for v in comparison['scalar_checks'].values()),
        'aa_checks_all_match': all(v['match'] for v in comparison['aa_checks'].values()),
        'reference_pins_verified': True,  # refused above otherwise
        'solve_profile_guard_verified_0': not guard_failures,
    }
    gate_pass = all(gate_items.values())
    input_hashes = {}
    for d in (REFERENCE_DIR, candidate_dir):
        for f in GATING_JSON + GATING_JSONL + SUPPLEMENTARY_JSONL:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                input_hashes[_rel(p)] = _sha(p)
    candidate_campaign = {}
    if not self_test:
        for f in ('campaign_results.json', 'campaign_manifest_sha256.json'):
            p = os.path.join(CAMPAIGN_ROOT, f)
            candidate_campaign[f] = _sha(p) if os.path.isfile(p) else None
    result = {
        'stage': 'P5.15 Addendum 27 item 1 (W4) -- re-verification comparator (zero solves)',
        'authority': AUTHORITY,
        'gate_text_spec_v15': ('a case-file-alone C* evaluation through the campaign harness (no overrides) reproduces '
                               'the committed AA C* evaluation (campaign_s44_aa_variant, 107 cycles, '
                               '650,982,939.9389359) bitwise on every numeric trajectory field and cost; tie-order in '
                               'the recourse-jump top-10 classified with p515_s44_tie_classifier; provenance '
                               'non-gating'),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': _git(['rev-parse', 'HEAD']),
        'script': os.path.basename(__file__), 'script_sha256': _sha(os.path.abspath(__file__)),
        'script_git_status': _git(['status', '--porcelain', '--', os.path.basename(__file__)]) or 'clean',
        'mode': 'self_test' if self_test else 'real',
        'self_test': ({'name': args.self_test, 'expect': args.expect, 'declared_differences_from_real_mode': [
            'capture-path failures recorded, comparison proceeds over files present on both sides (missing file = '
            'gating error)',
            'candidate effective AA falls back to overrides_applied_in_child.anderson_acceleration when the record '
            'predates anderson_acceleration_effective_in_child; case_file_sha256_in_child presence not required',
            'wait-free guard (lock / evaluation_record) not applied; candidate must lie outside the campaign root']}
            if self_test else None),
        'inputs': {'reference_dir': _rel(REFERENCE_DIR), 'candidate_dir': _rel(candidate_dir),
                   'reference_pins': REFERENCE_PINS, 'reference_pins_verified': ref_hashes == REFERENCE_PINS,
                   'campaign_spec': _rel(CAMPAIGN_SPEC), 'campaign_spec_sha256': spec_hashes['campaign_spec'],
                   'spec_v15': _rel(SPEC_V15), 'spec_v15_sha256': spec_hashes['spec_v15'],
                   'candidate_campaign_outputs_sha256': candidate_campaign or None,
                   'input_file_sha256': input_hashes},
        'capture_path_checklist_asserted_before_compare': checklist,
        'capture_path_checklist_false_items': missing,
        'candidate_aa_source': aa_source,
        'comparison': comparison,
        'gate_items': gate_items,
        'gate_pass': gate_pass,
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    if self_test:
        if args.expect == 'pass':
            met = gate_pass and comparison['n_genuine_diffs'] == 0
        else:
            met = (not gate_pass) and comparison['n_genuine_diffs'] > 0
        result['self_test']['expectation_met'] = met
    if os.path.exists(out_dir):
        raise RuntimeError(f'refusing to overwrite: {out_dir}')
    os.makedirs(out_dir)
    out = os.path.join(out_dir, 'compare.json')
    with open(out, 'x') as handle:
        json.dump(result, handle, indent=1, default=str)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        json.dump({_rel(out): _sha(out)}, handle, indent=2)
    GUARD.uninstall()

    for name, v in comparison['files'].items():
        _log(f'[S45-CMP] {name}: ' + (v['error'] if 'error' in v else
                                      f"raw={v['n_raw_diffs']} {v['n']} genuine_patterns="
                                      f"{list(v['genuine_field_patterns'].items())[:6]}"))
    for name, v in comparison['supplementary_not_gating'].items():
        _log(f'[S45-CMP] supplementary (not gating) {name}: {v.get("error") or {k: v[k] for k in ("n_rows", "n_raw_diffs", "n_diffs_ignoring_path_like_keys")}}')
    _log(f"[S45-CMP] totals_by_class={comparison['totals_by_class']}")
    _log(f"[S45-CMP] provenance fields: {comparison['provenance_field_patterns_with_reason']}")
    _log(f"[S45-CMP] scalar checks: { {k: (v['value'], v['match']) for k, v in comparison['scalar_checks'].items()} }")
    _log(f"[S45-CMP] AA checks: {comparison['aa_checks']}")
    _log(f'[S45-CMP] gate items: {gate_items}')
    _log(f'[S45-CMP] guard {GUARD.counts} verify0_failures={guard_failures}')
    _log(f'[S45-CMP] wrote {_rel(out)}')
    _log(f'[S45-CMP] GATE {"PASS" if gate_pass else "FAIL"}')
    if self_test:
        _log(f"[S45-CMP] self-test {args.self_test}: expect={args.expect} "
             f"expectation_met={result['self_test']['expectation_met']}")
        sys.exit(0 if result['self_test']['expectation_met'] else 1)
    if not gate_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
