"""
P5.15 Addendum 32 (task W27) -- the Q4 REPRODUCTION GATE: ZERO SOLVES.

Gate (frozen spec v18 `Q4_terminal_tso_capture.gate`, gating): "the x = 0 evaluation reproduces the committed A0 x0
evaluation bitwise on every trajectory field and cost (132 cycles, 653,859,461.2279255) - Q(0) is
ageing-independent (W21 NL identity) and the oracle is deterministic; the persistence step must not alter the
trajectory".

Compares
    CANDIDATE  the single evaluation of campaign s48_x0_capture (data/SRP1/Results/P515S48/x0_capture/evals/<key>_x0;
               baseline ageing declaration; post-certification persist_certified_models only)
    REFERENCE  data/SRP1/Results/P515S45/campaign_s45_a0_c7/evals/7aa017f09989b56d_x0 (A0 x0; certified at 132,
               653,859,461.2279255; no declaration; no post-certification)

Comparator conventions BY IMPORT from `p515_s45_reverify_compare` (itself importing
`p515_s40_clone_capture_preflight` `_diff` + exclusions, `p515_s43_aa_flagoff_gate._classify_diffs`,
`p515_s44_tie_classifier.reclassify_sidecar_diffs`): the same gating files (`GATING_JSON`, `GATING_JSONL`), the
same supplementary (non-gating) files, the same record provenance keys (`RECORD_PROVENANCE_KEYS`,
`RECORD_PROVENANCE_DOTTED`). The armed guard is that module's `GUARD` (SolveProfileGuard(permitted=()), installed
at its import, before any model import), verified at exactly 0.

CLASSIFICATION (fixed before the run; the only departures from `p515_s45_reverify_compare._classify` are the two
roles reversed / added by W27, stated here):
  provenance (NON-GATING, each listed with its reason):
    - `FG._classify_diffs` rule_eleven_checklist subtree (incl. the w21_ess_ageing_baseline checklist in g_s39_D);
    - evaluation_record.json top-level keys in `RC.RECORD_PROVENANCE_KEYS` / dotted `RC.RECORD_PROVENANCE_DOTTED`;
    - the post-certification bookkeeping (`RC.POST_CERTIFICATION_KEYS`) ONLY IF the REFERENCE record holds None for
      all three AND the candidate's post-certification is the persistence-only step (requested exactly
      {persist_certified_models: true, hull_polish: false, reference: null}; status evaluated; gate_b/c/d None).
      (RC applies the mirror condition -- candidate None -- because there the REFERENCE had the step.)
    - BASELINE_DECLARATION_KEYS: the six evaluation_record.json top-level keys a spec declaring
      `ess_ageing_baseline` adds (W21) -- ONLY as keys ABSENT from the reference (a key present on both sides with
      different values is genuine).
  tie_order (NON-GATING): recourse-jump sidecar rows explained by `p515_s44_tie_classifier`.
  genuine (GATING): everything else.
GATE (all must hold): capture-path checklist all true (fail fast before comparing); reference files match their
pins (A0 x0 committed files) and the A0 child manifest (for the untracked large sidecars); every gating file
present on both sides; 0 genuine diffs; candidate certified with cycles_run == certification_cycle == 132 (record
and report), certified_cost == report gross_operational_cost == 653859461.2279255 EXACTLY (type float); candidate
key == A0's; effective AA == the declaration; per_cycle_record.jsonl has 132 rows on both sides; the persisted
certified_models.pkl hashes to what the record recorded; the guard verified at 0.

OUTPUT (write-once): data/SRP1/Results/P515S48/x0_capture_gate/{gate.json, manifest_sha256.json}; launch log
sibling data/SRP1/Results/P515S48/x0_capture_gate_launch.log (shell, noclobber).

EXACT COMMAND (repo root; attached; both streams):
    set -o noclobber && /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s48_x0_capture_gate.py \\
        > data/SRP1/Results/P515S48/x0_capture_gate_launch.log 2>&1
SELF-TEST (`--self-test NAME --candidate-dir DIR --out-dir DIR --expect pass|fail`): classification only; the
expectation is on the genuine-diff count (pass: 0; fail: > 0); writes only to --out-dir (must be outside the repo).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import p515_s45_reverify_compare as RC  # noqa: E402 -- installs its armed GUARD (permitted=()) at import

CP, FG, TC, GUARD = RC.CP, RC.FG, RC.TC, RC.GUARD

_RESULTS = os.path.join(REPO, 'data', 'SRP1', 'Results')
REFERENCE_DIR = os.path.join(_RESULTS, 'P515S45', 'campaign_s45_a0_c7', 'evals', '7aa017f09989b56d_x0')
REFERENCE_PINS = {
    'evaluation_record.json': '34868965df76a9f1dc62d93d6ce4e2a50ceb4e12f8d3f2aeb4f8c96742bb4835',
    'g_s39_D.json': '9b4d86760fd920b40c5c93400428dc566a5bdab01c27eb9eefcd5cbfd1dbc0d0',
    'per_cycle_record.jsonl': '8663b96213c87dbb69d90790fc4fe771106f2bd838e17d9809f752cdf7a2f01e',
    'component_levels_terminal.json': '32696d9c3afc7f4e3e6624d59f21910509e6e6fe1f5e75a43b33f1413acd7279',
}
CAMPAIGN_ROOT = os.path.join(_RESULTS, 'P515S48', 'x0_capture')
OUT_DIR = os.path.join(_RESULTS, 'P515S48', 'x0_capture_gate')
X0_KEY = '8435c71859ddde68e7ae5818b4ff91c03b4171791bcfaa70edcc3ddb52bacb57'
EXPECTED_CYCLES = 132
EXPECTED_COST = 653859461.2279255
EXPECTED_AA = {'enabled': True, 'memory': 5, 'regularization': 1e-10, 'reject_policy': 'keep_memory'}
EXPECTED_POST_REQUEST = {'persist_certified_models': True, 'hull_polish': False, 'reference': None}
BASELINE_DECLARATION_KEYS = {
    'ageing_trajectory_terminal': 'baseline declaration (W21): ageing trajectory captured only for declared specs',
    'ess_ageing_baseline': 'baseline declaration (W21): the declared ageing dict',
    'ess_ageing_baseline_label': 'baseline declaration (W21): the declaration label',
    'ess_ageing_readback_terminal': 'baseline declaration (W21): post-run read-back on ESSO clones',
    'ess_ageing_verified_pre_run': 'baseline declaration (W21): pre-run read-back on probe ESSO models',
    'ess_params_sha256_in_child': 'baseline declaration (W21): ESS params file sha256 seen by the child',
}
RECORD_FILE, REPORT_FILE, RECOURSE_JUMP = RC.RECORD_FILE, RC.REPORT_FILE, RC.RECOURSE_JUMP
GATE_TEXT = ('the x = 0 evaluation reproduces the committed A0 x0 evaluation bitwise on every trajectory field and '
             'cost (132 cycles, 653,859,461.2279255) - Q(0) is ageing-independent (W21 NL identity) and the oracle '
             'is deterministic; the persistence step must not alter the trajectory')


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {msg}', flush=True)


def _rel(path):
    return os.path.relpath(path, REPO)


def _git(args):
    return subprocess.run(['git'] + list(args), cwd=REPO, capture_output=True, text=True).stdout.strip()


def _persistence_only(candidate_record):
    pc = (candidate_record or {}).get('post_certification') or {}
    return (pc.get('requested') == EXPECTED_POST_REQUEST and pc.get('status') == 'evaluated'
            and pc.get('gate_b') is None and pc.get('gate_c') is None and pc.get('gate_d') is None
            and bool((pc.get('persisted_models') or {}).get('sha256')))


def classify(fname, a, b, reference_record, candidate_record):
    """RC._classify with the W27 roles (see the module docstring)."""
    raw = CP._diff(a, b, fname)
    fg_prov, fg_aa_new, fg_alias, fg_genuine = FG._classify_diffs(raw)
    provenance = [dict(d, reason='rule_eleven_checklist subtree') for d in fg_prov]
    genuine = list(fg_aa_new) + list(fg_alias) + list(fg_genuine)
    if fname == RECORD_FILE:
        post_cert_reference_absent = all((reference_record or {}).get(k) is None for k in RC.POST_CERTIFICATION_KEYS)
        persist_only = _persistence_only(candidate_record)
        kept = []
        for d in genuine:
            top, rest = RC._record_subfield(str(d.get('field', '')))
            if top in RC.RECORD_PROVENANCE_KEYS:
                provenance.append(dict(d, reason=RC.RECORD_PROVENANCE_KEYS[top]))
            elif rest in RC.RECORD_PROVENANCE_DOTTED:
                provenance.append(dict(d, reason=RC.RECORD_PROVENANCE_DOTTED[rest]))
            elif top in RC.POST_CERTIFICATION_KEYS and post_cert_reference_absent and persist_only:
                provenance.append(dict(d, reason='post-certification (persistence only) requested in this campaign '
                                                 'only (reference record holds None for all three keys)'))
            elif top in BASELINE_DECLARATION_KEYS and top not in (reference_record or {}):
                provenance.append(dict(d, reason=BASELINE_DECLARATION_KEYS[top] + ' (absent from the reference)'))
            else:
                kept.append(d)
        genuine = kept
    tie_order, row_classes = [], {}
    if fname == RECOURSE_JUMP and genuine:
        tie_order, genuine, row_classes = TC.reclassify_sidecar_diffs(genuine, a, b, fname)
    patterns = {}
    for d in genuine:
        p = RC._pattern(str(d.get('field', '')))
        patterns[p] = patterns.get(p, 0) + 1
    return {'n_raw_diffs': len(raw),
            'fg_buckets_raw_counts': {'provenance': len(fg_prov), 'aa_new_field': len(fg_aa_new),
                                      'known_tie_break_alias': len(fg_alias), 'genuine': len(fg_genuine)},
            'n': {'provenance': len(provenance), 'tie_order': len(tie_order), 'genuine': len(genuine)},
            'provenance_diffs': provenance, 'tie_order_diffs': tie_order, 'tie_order_row_classes': row_classes,
            'genuine_field_patterns': dict(sorted(patterns.items())),
            'genuine_diffs_first': genuine[:RC.MAX_LISTED_GENUINE_PER_FILE],
            'genuine_diffs_listed_truncated': len(genuine) > RC.MAX_LISTED_GENUINE_PER_FILE}


def _reference_manifest_check():
    """Every gating / supplementary reference file against A0's committed child manifest (covers the large,
    untracked sidecars pf_entry_stride / ess_entry_stride) and the four pinned files against their pins."""
    with open(os.path.join(REFERENCE_DIR, 'child_manifest_sha256.json')) as handle:
        manifest = json.load(handle)
    by_name = {}
    for k, v in manifest.items():
        if os.path.dirname(os.path.join(REPO, k)) == REFERENCE_DIR:
            by_name[os.path.basename(k)] = v
    out = {}
    for f in RC.GATING_JSON + RC.GATING_JSONL + RC.SUPPLEMENTARY_JSONL:
        p = os.path.join(REFERENCE_DIR, f)
        got = RC._sha(p) if os.path.isfile(p) else None
        exp = by_name.get(f)
        out[f] = {'sha256': got, 'child_manifest': exp, 'pin': REFERENCE_PINS.get(f),
                  'match_manifest': (exp is None) or (got == exp),
                  'in_manifest': exp is not None,
                  'match_pin': (f not in REFERENCE_PINS) or (got == REFERENCE_PINS[f])}
    return out


def _candidate_dir_from_spec():
    specs = [f for f in os.listdir(CAMPAIGN_ROOT) if f.startswith('campaign_spec_') and f.endswith('.json')]
    if len(specs) != 1:
        raise RuntimeError(f'expected exactly one frozen spec in {CAMPAIGN_ROOT}: {specs}')
    with open(os.path.join(CAMPAIGN_ROOT, specs[0])) as handle:
        spec = json.load(handle)
    entry = spec['candidates'][0]
    return os.path.join(CAMPAIGN_ROOT, 'evals', entry['eval_dir']), os.path.join(CAMPAIGN_ROOT, specs[0]), spec


def capture_path_checklist(candidate_dir):
    items = {}
    for side, d in (('reference', REFERENCE_DIR), ('candidate', candidate_dir)):
        for f in RC.GATING_JSON + RC.GATING_JSONL:
            items[f'{side}_file_{f}'] = os.path.isfile(os.path.join(d, f))
    ref_rec = CP._load_json(os.path.join(REFERENCE_DIR, RECORD_FILE)) or {}
    cand_rec = CP._load_json(os.path.join(candidate_dir, RECORD_FILE)) or {}
    cand_rep = CP._load_json(os.path.join(candidate_dir, REPORT_FILE)) or {}
    for side, rec in (('reference', ref_rec), ('candidate', cand_rec)):
        for k in ('status', 'certification', 'cycles_run', 'certification_cycle', 'certified_cost', 'candidate_key',
                  'rule_ten', 'anderson_acceleration_effective_in_child'):
            items[f'{side}_record_field_{k}'] = k in rec
    items['candidate_record_field_post_certification.persisted_models'] = bool(
        (cand_rec.get('post_certification') or {}).get('persisted_models'))
    for k in ('cycles_run', 'gross_operational_cost', 'cycle_trajectory'):
        items[f'candidate_report_field_{k}'] = k in cand_rep
    items['comparator_CP__diff'] = callable(getattr(CP, '_diff', None))
    items['comparator_FG__classify_diffs'] = callable(getattr(FG, '_classify_diffs', None))
    items['tie_classifier_TC_reclassify_sidecar_diffs'] = callable(getattr(TC, 'reclassify_sidecar_diffs', None))
    return items


def compare(candidate_dir):
    ref_rec = CP._load_json(os.path.join(REFERENCE_DIR, RECORD_FILE))
    cand_rec = CP._load_json(os.path.join(candidate_dir, RECORD_FILE))
    ref_rep = CP._load_json(os.path.join(REFERENCE_DIR, REPORT_FILE))
    cand_rep = CP._load_json(os.path.join(candidate_dir, REPORT_FILE))
    files, n_rows = {}, {}
    for fname in RC.GATING_JSON:
        a = ref_rec if fname == RECORD_FILE else (ref_rep if fname == REPORT_FILE else
                                                  CP._load_json(os.path.join(REFERENCE_DIR, fname)))
        b = cand_rec if fname == RECORD_FILE else (cand_rep if fname == REPORT_FILE else
                                                   CP._load_json(os.path.join(candidate_dir, fname)))
        if a is None or b is None:
            files[fname] = {'error': f'missing: reference={a is not None} candidate={b is not None}'}
            continue
        files[fname] = classify(fname, a, b, ref_rec, cand_rec)
        _log(f'[S48-GATE]   {fname}: {files[fname]["n"]}')
    for fname in RC.GATING_JSONL:
        a = CP._load_jsonl(os.path.join(REFERENCE_DIR, fname))
        b = CP._load_jsonl(os.path.join(candidate_dir, fname))
        if a is None or b is None:
            files[fname] = {'error': f'missing: reference={a is not None} candidate={b is not None}'}
            _log(f'[S48-GATE]   {fname}: {files[fname]["error"]}')
            continue
        c = classify(fname, a, b, ref_rec, cand_rec)
        c['n_rows'] = {'reference': len(a), 'candidate': len(b)}
        n_rows[fname] = c['n_rows']
        files[fname] = c
        _log(f'[S48-GATE]   {fname}: rows={c["n_rows"]} {c["n"]}')
        del a, b
    supplementary = {}
    for fname in RC.SUPPLEMENTARY_JSONL:
        a = CP._load_jsonl(os.path.join(REFERENCE_DIR, fname))
        b = CP._load_jsonl(os.path.join(candidate_dir, fname))
        if a is None or b is None:
            supplementary[fname] = {'error': f'missing: reference={a is not None} candidate={b is not None}'}
            continue
        raw = CP._diff(a, b, fname)
        stripped = CP._diff(RC._strip_keys(a, RC.SUPPLEMENTARY_PATH_LIKE_KEYS),
                            RC._strip_keys(b, RC.SUPPLEMENTARY_PATH_LIKE_KEYS), fname)
        supplementary[fname] = {'n_rows': {'reference': len(a), 'candidate': len(b)}, 'n_raw_diffs': len(raw),
                                'n_diffs_ignoring_path_like_keys': len(stripped),
                                'first_raw_diffs_ignoring_path_like_keys': stripped[:20]}
    pk_ref, pk_cand = (os.path.join(REFERENCE_DIR, 'esso_models_s39_D.pkl'),
                       os.path.join(candidate_dir, 'esso_models_s39_D.pkl'))
    esso_pickle = {'reference_sha256': RC._sha(pk_ref) if os.path.isfile(pk_ref) else None,
                   'candidate_sha256': RC._sha(pk_cand) if os.path.isfile(pk_cand) else None}
    esso_pickle['sha256_equal'] = (esso_pickle['reference_sha256'] is not None
                                   and esso_pickle['reference_sha256'] == esso_pickle['candidate_sha256'])
    totals = {'provenance': 0, 'tie_order': 0, 'genuine': 0}
    for v in files.values():
        for k, n in (v.get('n') or {}).items():
            totals[k] += n
    rec, rep = cand_rec or {}, cand_rep or {}
    persisted = ((rec.get('post_certification') or {}).get('persisted_models')) or {}
    ppath = os.path.join(REPO, persisted.get('path') or '__missing__')
    persisted_sha = RC._sha(ppath) if os.path.isfile(ppath) else None
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
        'candidate_record_candidate_key_is_x0': {'value': rec.get('candidate_key'), 'expected': X0_KEY},
        'reference_record_candidate_key_is_x0': {'value': (ref_rec or {}).get('candidate_key'), 'expected': X0_KEY},
        'reference_record_certified_cost': {'value': (ref_rec or {}).get('certified_cost'), 'expected': EXPECTED_COST},
        'reference_record_certification_cycle': {'value': (ref_rec or {}).get('certification_cycle'),
                                                 'expected': EXPECTED_CYCLES},
        'candidate_effective_aa': {'value': rec.get('anderson_acceleration_effective_in_child'),
                                   'expected': EXPECTED_AA},
        'reference_effective_aa': {'value': (ref_rec or {}).get('anderson_acceleration_effective_in_child'),
                                   'expected': EXPECTED_AA},
        'per_cycle_rows_reference': {'value': (n_rows.get('per_cycle_record.jsonl') or {}).get('reference'),
                                     'expected': EXPECTED_CYCLES},
        'per_cycle_rows_candidate': {'value': (n_rows.get('per_cycle_record.jsonl') or {}).get('candidate'),
                                     'expected': EXPECTED_CYCLES},
        'candidate_post_certification_persistence_only': {'value': _persistence_only(rec), 'expected': True},
        'candidate_persisted_models_sha256_verified': {'value': persisted_sha, 'expected': persisted.get('sha256')},
    }
    for v in scalar.values():
        v['match'] = (type(v['value']) is type(v['expected'])) and v['value'] == v['expected'] and v['value'] is not None
    provenance_fields = sorted({(RC._pattern(str(d.get('field'))), d['reason'])
                                for v in files.values() for d in v.get('provenance_diffs', [])})
    return {
        'reference_dir': _rel(REFERENCE_DIR), 'candidate_dir': _rel(candidate_dir),
        'instance': {'reference_candidate_key': (ref_rec or {}).get('candidate_key'),
                     'candidate_candidate_key': rec.get('candidate_key'),
                     'reference_eval_key': (ref_rec or {}).get('eval_key'), 'candidate_eval_key': rec.get('eval_key')},
        'comparator': ('p515_s45_reverify_compare conventions by import (CP._diff + exclusions, FG._classify_diffs, '
                       'RC record provenance keys, TC tie classifier) + the W27 roles (module docstring)'),
        'diff_value_naming': "CP._diff names the reference value 'legacy' and the candidate value 'lightweight'",
        'excluded_field_names': sorted(CP.EXCLUDE_KEY_NAMES),
        'excluded_dotted_suffixes': sorted(CP.EXCLUDE_DOTTED_SUFFIXES),
        'record_provenance_keys': RC.RECORD_PROVENANCE_KEYS, 'record_provenance_dotted': RC.RECORD_PROVENANCE_DOTTED,
        'record_post_certification_keys_conditional': list(RC.POST_CERTIFICATION_KEYS),
        'baseline_declaration_keys_conditional': BASELINE_DECLARATION_KEYS,
        'gating_files': list(RC.GATING_JSON + RC.GATING_JSONL), 'files': files,
        'supplementary_not_gating': supplementary, 'esso_models_pickle_informational': esso_pickle,
        'totals_by_class': totals, 'provenance_field_patterns_with_reason': [list(p) for p in provenance_fields],
        'all_gating_files_present': all('error' not in v for v in files.values()),
        'n_genuine_diffs': totals['genuine'], 'scalar_checks': scalar,
        'persisted_models': dict(persisted, sha256_on_disk=persisted_sha),
        'rule_ten_terminal_step_over_threshold': {
            'reference': ((ref_rec or {}).get('rule_ten') or {}).get('terminal_step_over_threshold'),
            'candidate': (rec.get('rule_ten') or {}).get('terminal_step_over_threshold')},
        'objective_convention': 'gross_operational_cost (settlement-excluded; = certified_cost)',
    }


def main():
    started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', dest='self_test', default=None)
    parser.add_argument('--candidate-dir', default=None)
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--expect', choices=('pass', 'fail'), default=None)
    args = parser.parse_args()
    self_test = args.self_test is not None
    failures = []
    spec_path, spec = None, None
    if self_test:
        if not (args.candidate_dir and args.out_dir and args.expect):
            parser.error('--self-test requires --candidate-dir, --out-dir and --expect')
        candidate_dir = os.path.abspath(args.candidate_dir)
        out_dir = os.path.abspath(args.out_dir)
        if os.path.commonpath([out_dir, REPO]) == REPO:
            failures.append(f'self-test out dir must be outside the repository: {out_dir}')
    else:
        if args.candidate_dir or args.out_dir or args.expect:
            parser.error('--candidate-dir/--out-dir/--expect are self-test options')
        candidate_dir, spec_path, spec = _candidate_dir_from_spec()
        out_dir = OUT_DIR
        if os.path.exists(RC.CAMPAIGN_LOCK_PATH):
            failures.append(f'campaign lock exists (evaluation not finished): {RC.CAMPAIGN_LOCK_PATH}')
        for f in ('campaign_results.json', 'campaign_manifest_sha256.json'):
            if not os.path.isfile(os.path.join(CAMPAIGN_ROOT, f)):
                failures.append(f'campaign output missing: {f}')
        dirty = _git(['status', '--porcelain', '--', os.path.basename(__file__), 'p515_s45_reverify_compare.py',
                      'p515_s40_clone_capture_preflight.py', 'p515_s43_aa_flagoff_gate.py',
                      'p515_s44_tie_classifier.py'])
        if dirty:
            failures.append(f'comparator scripts not clean in git:\n{dirty}')
        if (spec.get('extra') or {}).get('gate_script_sha256') != RC._sha(os.path.abspath(__file__)):
            failures.append('this gate script differs from the sha256 frozen in the campaign spec')
    if not os.path.isfile(os.path.join(candidate_dir, RECORD_FILE)):
        failures.append(f'candidate evaluation_record.json missing: {candidate_dir}')
    if os.path.exists(out_dir):
        failures.append(f'output directory exists (write-once): {out_dir}')
    ref_check = _reference_manifest_check()
    bad_ref = {f: v for f, v in ref_check.items() if not (v['match_manifest'] and v['match_pin'])}
    if bad_ref:
        failures.append(f'reference files do not match the A0 child manifest / pins: {bad_ref}')
    if failures:
        for f in failures:
            _log(f'[S48-GATE REFUSED] {f}')
        raise SystemExit(1)
    _log(f'[S48-GATE] mode={"self-test " + args.self_test if self_test else "REAL"} candidate={_rel(candidate_dir)} '
         f'reference={_rel(REFERENCE_DIR)} out={out_dir}')
    checklist = capture_path_checklist(candidate_dir)
    missing = sorted(k for k, v in checklist.items() if not v)
    _log(f'[S48-GATE] capture-path checklist: {len(checklist)} items, {len(missing)} False: {missing}')
    if missing and not self_test:
        _log('[S48-GATE] CAPTURE-PATH CHECK FAILED -- refusing to compare (fail fast)')
        raise SystemExit(1)
    comparison = compare(candidate_dir)
    guard_failures = GUARD.verify(0)
    gate_items = {
        'capture_path_checklist_all_true': not missing,
        'reference_files_match_manifest_and_pins': not bad_ref,
        'all_gating_files_present': comparison['all_gating_files_present'],
        'zero_genuine_diffs': comparison['n_genuine_diffs'] == 0,
        'scalar_checks_all_match': all(v['match'] for v in comparison['scalar_checks'].values()),
        'solve_profile_guard_verified_0': not guard_failures,
    }
    gate_pass = all(gate_items.values())
    input_hashes = {}
    for d in (REFERENCE_DIR, candidate_dir):
        for f in RC.GATING_JSON + RC.GATING_JSONL + RC.SUPPLEMENTARY_JSONL:
            p = os.path.join(d, f)
            if os.path.isfile(p):
                input_hashes[_rel(p)] = RC._sha(p)
    result = {
        'stage': 'P5.15 Addendum 32 W27 -- Q4 reproduction gate (zero solves)',
        'gate_text_spec_v18': GATE_TEXT, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head': _git(['rev-parse', 'HEAD']), 'script': os.path.basename(__file__),
        'script_sha256': RC._sha(os.path.abspath(__file__)),
        'mode': 'self_test' if self_test else 'real',
        'self_test': {'name': args.self_test, 'expect': args.expect} if self_test else None,
        'inputs': {'reference_dir': _rel(REFERENCE_DIR), 'candidate_dir': _rel(candidate_dir),
                   'reference_pins': REFERENCE_PINS, 'reference_manifest_check': ref_check,
                   'campaign_spec': _rel(spec_path) if spec_path else None,
                   'campaign_spec_sha256': RC._sha(spec_path) if spec_path else None,
                   'input_file_sha256': input_hashes},
        'capture_path_checklist_asserted_before_compare': checklist,
        'capture_path_checklist_false_items': missing,
        'comparison': comparison, 'gate_items': gate_items, 'gate_pass': gate_pass,
        'solve_profile_guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_clock_s': time.time() - started,
    }
    if self_test:
        met = (comparison['n_genuine_diffs'] == 0) if args.expect == 'pass' else (comparison['n_genuine_diffs'] > 0)
        result['self_test']['expectation_met'] = met
    os.makedirs(out_dir)
    out = os.path.join(out_dir, 'gate.json')
    with open(out, 'x') as handle:
        json.dump(result, handle, indent=1, default=str)
    with open(os.path.join(out_dir, 'manifest_sha256.json'), 'x') as handle:
        json.dump({(_rel(out) if not self_test else out): RC._sha(out)}, handle, indent=2)
    GUARD.uninstall()
    for name, v in comparison['files'].items():
        _log(f'[S48-GATE] {name}: ' + (v['error'] if 'error' in v else
                                       f"raw={v['n_raw_diffs']} {v['n']} genuine_patterns="
                                       f"{list(v['genuine_field_patterns'].items())[:6]}"))
    for name, v in comparison['supplementary_not_gating'].items():
        _log(f'[S48-GATE] supplementary (not gating) {name}: '
             f'{v.get("error") or {k: v[k] for k in ("n_rows", "n_raw_diffs", "n_diffs_ignoring_path_like_keys")}}')
    _log(f"[S48-GATE] totals_by_class={comparison['totals_by_class']}")
    _log(f"[S48-GATE] provenance fields: {comparison['provenance_field_patterns_with_reason']}")
    _log(f"[S48-GATE] scalar checks: { {k: (v['value'], v['match']) for k, v in comparison['scalar_checks'].items()} }")
    _log(f'[S48-GATE] gate items: {gate_items}')
    _log(f'[S48-GATE] guard {GUARD.counts} verify0_failures={guard_failures}')
    _log(f'[S48-GATE] wrote {out}')
    _log(f'[S48-GATE] GATE {"PASS" if gate_pass else "FAIL"}')
    if self_test:
        _log(f"[S48-GATE] self-test {args.self_test}: expect={args.expect} "
             f"expectation_met={result['self_test']['expectation_met']}")
        sys.exit(0 if result['self_test']['expectation_met'] else 1)
    if not gate_pass:
        sys.exit(1)


if __name__ == '__main__':
    main()
