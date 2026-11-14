"""
P5.15 Addendum 85 ruling 1 (c), W183g (a) -- WRAPPER v2 of the SRP1 two-cycle bitwise gate: the STANDING COMPARATOR
RULE for field-list drift, recorded beside the frozen verdict. Successor of `p515_s55_w183f_bitwise_gate.py` (v1,
committed 17b4c1cf, NOT edited); the committed gate chain (`p515_s53_srp1_bitwise_gate.py` -> W48 -> W39 -> W35 ->
W32 -> W10) is imported, NOT edited, and its frozen verdict `GATE_PASS` is never rewritten.

Authority: PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 1 (c) and W183g order item (a); Planner task W183g (a).

RULE (c), as ruled (Addendum 85 ruling 1 (c)), applied ON TOP OF the gate's own comparison (`W10.classify` via
`S52G.compare_against_w39_arm`, the gate's compared-file set `W10.GATING_JSON + W10.GATING_JSONL` and its declared
exclusions -- imported, not reimplemented):
  A diff that the gate's comparator classifies `genuine` is reclassified `field_list_drift` iff ALL of:
    1. the reference ('legacy') side is `<MISSING>` and the new-arm ('lightweight') side is JSON null (None);
    2. it resolves STRUCTURALLY to one key k of one record R -- R is the innermost pair of dicts at the same dotted
       path in the two compared objects (dotted paths named exactly as `CP._diff` names them), k is in R_new with
       value None and k is NOT in R_ref (the `<MISSING>` sentinel is not trusted on its own); an ambiguous dotted
       path (two record pairs, or two keys, rendering to the same text) does not resolve;
    3. EVERY KEY SHARED BY THE TWO RECORDS IS BITWISE EQUAL under the gate's comparator: no raw `CP._diff` diff lies
       in R's subtree other than direct-child diffs on keys NOT shared by R_ref and R_new (absent on one side --
       the drift candidates themselves, and any other one-sided key, which stays genuine on its own). A raw diff of ANY class
       (genuine, provenance or tie-order) on a shared key of R, at any depth, disqualifies EVERY drift candidate of
       R, which then stays `genuine` (reason recorded). Keys the gate's comparator excludes by declaration
       (`CP.EXCLUDE_KEY_NAMES` / `EXCLUDE_DOTTED_SUFFIXES` / `INTENTIONAL_DIFF_SUFFIXES`) are not compared, as in the
       gate; a strict type-exact equality over ALL shared keys including those is recorded per record as
       information.
  A key present in both records with different values stays `genuine`. A key absent in the reference and NON-null
  in the new arm (including 0, False, "", [], {}) stays `genuine` (reason 'absent in reference, non-null in new
  arm'); per the rule's text it is not a SHARED key, so it does not by itself disqualify the record's other drift
  candidates. A key present in the reference and absent in the new arm stays `genuine`; it is not a shared key
  either, so it does not by itself disqualify R (rule's text). Either keeps GATE_PASS_RULED False on its own.
  A key absent in both arms produces no diff.
  If any file's genuine list is truncated by the gate (`genuine_diffs_listed_truncated`), the rule cannot see every
  genuine diff and the ruled verdict is False for that reason.

VERDICTS. `GATE_PASS` = the frozen verdict, read as recorded (never recomputed, never edited).
`GATE_PASS_RULED` = every frozen gate item other than the zero-genuine item, as recorded, AND every compared file
present AND the recomputed gate comparison reproducing the frozen one exactly (per-file entries and totals) AND zero
genuine diffs after rule (c). The drift keys are listed by key and by occurrence (file, record, key).
The W48 arm (`data/SRP1/Results/P515S52/srp1_bitwise_gate/arm`, `S53G.W48_ARM`, manifest-pinned in the gate) stays
the whole-arm reference.

MODES
  --ruled-from-committed  ZERO SOLVES. Applies rule (c) to the COMMITTED W183f gate output
                          (`data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate`, committed c5b292dd) against the W48 arm,
                          after hash-checking every file of both against their committed manifests (manifest sha256
                          pinned below) and a clean git status; writes `gate_ruled.json` + `manifest_sha256.json` into
                          the fresh write-once directory `data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_ruled/`
                          (refused if it holds anything but `launch.log`). Includes the negative controls below.
                          Guards: `SolveProfileGuard(permitted=())` installed BEFORE any project import and verified at
                          exactly 0 (and the imported chain's own W10.GUARD, installed at import, also 0);
                          `pickle.load` / `pickle.loads` blocked and counted (0); `pyomo ConcreteModel.__init__`
                          blocked and counted (0). JSON through `gate_result_io`.
  --post-run              ZERO SOLVES. Writes `manifest_post_run_sha256.json` over the output dir incl. launch.log.
  --run-gate --arm A --out-rel R
                          SOLVES (153 + retries, armed by the chain's W10.GUARD). Runs the committed gate exactly as v1
                          does (the same three declared substitutions: identifiers, EXTRA_CLEAN_FILES + this file,
                          EXTRA_FORBIDDEN), then records GATE_PASS (frozen) and GATE_PASS_RULED (rule (c), vs the W48
                          arm) in `w183g_v2_wrapper_record.json`. WRITTEN, NOT RUN at W183g (a) (Addendum 85: nothing is
                          re-run); its post-gate step is the same `ruled_verdict` the zero-solve mode exercises.

NEGATIVE CONTROLS (zero-solve mode; planted on in-memory deep copies of the committed W183f `per_cycle_record.jsonl`
(NC0-NC4) or `g_s39_D.json` (NC5a/b), each classified by the gate's own `W10.classify` against the committed W48 rows, then by rule (c); the verdict is
recomputed with the planted file in place of the real one, every other file and frozen item as committed):
  NC0 positive control, nothing planted: 44 drift (22 keys x 2 rows), 0 genuine, GATE_PASS_RULED True.
  NC1 (i)  a SHARED key with a different value (row 0 `recourse` + 1.0): that diff stays genuine; GATE_PASS_RULED
           False.
  NC2 (ii) a key absent in the reference but NON-null in the new arm: row 0 `rss_bytes` = 123456789, row 1
           `n_non_optimal_block_terminations` = 0 (falsy, not null), and a new key `planted_new_key` = 'x' in row 1:
           each stays genuine ('absent in reference, non-null in new arm'); GATE_PASS_RULED False.
  NC3 (iii) a shared key differing by ONE ULP (row 1 `boyd_v_primal_ratio` -> math.nextafter(v, inf)): that diff is
           genuine AND all 22 drift candidates of row 1 are disqualified (stay genuine); row 0 keeps its 22 drift
           keys; GATE_PASS_RULED False.
  NC4 a key present in the reference and absent in the new arm (row 0 `cycle_convergence` deleted): genuine; not a
           shared key, so row 0 keeps its 22 drift keys (rule's text); GATE_PASS_RULED False.
  NC5a positive, nested record: `g_s39_D.json` `instance.planted_drift_key` = None -> 1 drift key in record
           `g_s39_D.json.instance`, 0 genuine; GATE_PASS_RULED True (the per-cycle file as committed).
  NC5b (iii) at depth and of any class: `g_s39_D.json` top-level `planted_drift_key` = None -> stays genuine, because
           the top-level record's shared key `rule_eleven_checklist` differs at depth 2 (the gate's 3 PROVENANCE
           diffs); GATE_PASS_RULED False.

EXACT COMMANDS (repo root; canonical interpreter; attached, alone, both streams captured; never detached):
  mkdir -p data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_ruled && \\
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183f_bitwise_gate_v2.py \\
      --ruled-from-committed > data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate_ruled/launch.log 2>&1
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s55_w183f_bitwise_gate_v2.py --post-run
Exit 0 when the mode completed and (ruled mode) every negative control behaved as declared; 1 otherwise. The exit
status does NOT encode GATE_PASS_RULED (it is a recorded outcome, read from gate_ruled.json).
"""
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

RUN_GATE = '--run-gate' in sys.argv

if not RUN_GATE:
    # ---- zero-solve modes: guards BEFORE any project import ----
    import pickle

    from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

    ZERO_GUARD = SolveProfileGuard(permitted=(), label='P5.15 W183g (a) wrapper v2 ruled (never solves)').install()
    PICKLE_COUNTS = {'load': 0, 'loads': 0}

    def _blocked_load(*_a, **_k):
        PICKLE_COUNTS['load'] += 1
        raise RuntimeError('W183g (a) v2: pickle.load called -- nothing is unpickled')

    def _blocked_loads(*_a, **_k):
        PICKLE_COUNTS['loads'] += 1
        raise RuntimeError('W183g (a) v2: pickle.loads called -- nothing is unpickled')

    pickle.load, pickle.loads = _blocked_load, _blocked_loads

    import pyomo.environ as pe  # noqa: E402  (model-build tripwire only)

    MODEL_BUILD_COUNTS = {'ConcreteModel.__init__': 0}

    def _blocked_concrete_init(self, *_a, **_k):
        MODEL_BUILD_COUNTS['ConcreteModel.__init__'] += 1
        raise RuntimeError('W183g (a) v2: a pyomo ConcreteModel was constructed -- no model build')

    pe.ConcreteModel.__init__ = _blocked_concrete_init
else:
    ZERO_GUARD = None
    PICKLE_COUNTS = None
    MODEL_BUILD_COUNTS = None

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import gate_result_io as GRIO  # noqa: E402
import p515_s55_w183f_bitwise_gate as V1  # noqa: E402 -- v1 wrapper (imports S53G -> ... -> W10, installs W10.GUARD)

if not RUN_GATE:
    pickle.load, pickle.loads = _blocked_load, _blocked_loads   # re-assert after the imports
os.chdir(REPO)

S53G = V1.S53G
S52G = S53G.S52G
W10 = S53G.W10
CP = W10.CP
H = W10.H
GUARD = W10.GUARD

SCRIPT = os.path.basename(__file__)
SCHEMA = 'p515_s55_w183f_bitwise_gate_v2'
AUTHORITY = ['PLANNER_BRIEF_2026-09-13.md Addendum 85 ruling 1 (c); W183g order item (a)',
             'Planner task W183g (a)']
RULE_TEXT = ('Addendum 85 ruling 1 (c): a key absent in the reference record and null or absent in the new arm is '
             'field_list_drift, listed by key, never genuine, provided every key shared by the two records is bitwise '
             'equal; a key present in both with different values stays genuine.')

# ---- the committed W183f gate output (read-only input of the zero-solve mode) ----
W183F = {
    'out_rel': V1.OUT_REL,                     # data/SRP1/Results/P515S55/w183f_srp1_bitwise_gate
    'arm_name': V1.ARM,                        # s55w183fgate
    'commit': 'c5b292dd',
    # sha256 of each committed manifest (git blob at c5b292dd == working tree; checked again at run time)
    'manifests': {'manifest_sha256.json': '140c33e1a5cfc73cd05c121f10a0f8d68837dafa002f08f542cadef0dd2f17a1',
                  'w51_manifest_sha256.json': 'a6c783394501190525ca2a91d0f281e5b3dd7deef91b9aeef8d2a3ffcaa63638',
                  'w183f_manifest_sha256.json': 'e7cd233e812998949def55b8f175389e5a546eb1d92783271c50515837f26943'},
}
REFERENCE_ARM = S53G.W48_ARM                   # the W48 arm stays the whole-arm reference (Addendum 85 ruling 1)
REFERENCE_EXTRA_MANIFESTS = {                  # every W48 gate-dir manifest, beyond the gate's own pin
    'w48_manifest_sha256.json': 'fad4d941dadae2f2c76e0602552bd471fa4789fafaf6d52c588aff58d9d95659'}

OUT_REL_RULED = os.path.join('data', 'SRP1', 'Results', 'P515S55', 'w183f_srp1_bitwise_gate_ruled')
COMPARED_FILES = tuple(W10.GATING_JSON) + tuple(W10.GATING_JSONL)
ZERO_GENUINE_ITEM_SUFFIX = '_zero_genuine_diffs'
MISSING = '<MISSING>'                          # CP._diff's sentinel (read from its three append sites)


def _log(msg):
    print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} [W183g-a v2] {msg}', flush=True)


def _sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _canon(obj):
    """The frozen payloads were written with json.dump(default=str); compare on the same text form."""
    return json.dumps(obj, sort_keys=True, default=str)


# ======================================================================================================================
#  hash checks of committed gate directories
# ======================================================================================================================
def check_gate_dir(gate_rel, manifests):
    """Every pinned manifest hashes to its pin, every file listed in every manifest hashes to its entry, every file
    on disk is listed by some manifest or is a manifest, and the directory is clean in git."""
    problems, checked = [], 0
    listed = set()
    for mname, pin in manifests.items():
        mpath = os.path.join(REPO, gate_rel, mname)
        got = _sha(mpath) if os.path.isfile(mpath) else None
        if got != pin:
            problems.append(f'{gate_rel}/{mname}: sha256 {got} != pin {pin}')
            continue
        with open(mpath) as handle:
            manifest = json.load(handle)
        for rel, sha in manifest.items():
            listed.add(rel)
            path = os.path.join(REPO, rel)
            checked += 1
            if not os.path.isfile(path) or _sha(path) != sha:
                problems.append(f'{rel} does not hash to {mname}')
    on_disk = {os.path.relpath(os.path.join(r, f), REPO) for r, _d, fs in os.walk(os.path.join(REPO, gate_rel))
               for f in fs}
    manifests_rel = {os.path.join(gate_rel, m) for m in manifests}
    unlisted = sorted(on_disk - listed - manifests_rel)
    if unlisted:
        problems.append(f'{gate_rel}: files on disk in no manifest: {unlisted}')
    status = W10._git(['status', '--porcelain', '--', gate_rel])
    if status.strip():
        problems.append(f'{gate_rel} not clean in git:\n{status}')
    return problems, {'gate_dir': gate_rel, 'manifests_pinned': manifests, 'file_entries_checked': checked,
                      'files_on_disk': len(on_disk), 'unlisted_files': unlisted, 'git_clean': not status.strip()}


# ======================================================================================================================
#  rule (c)
# ======================================================================================================================
def _record_index(a, b, prefix, index, ambiguous):
    """Every pair of dicts at the same dotted path in a (reference) and b (new), named exactly as CP._diff names
    paths (dict: f'{prefix}.{key}' or str(key) at the root; list of equal length: f'{prefix}[{i}]')."""
    if isinstance(a, dict) and isinstance(b, dict):
        if prefix in index:
            ambiguous.add(prefix)
        index[prefix] = (a, b)
        for key in set(a) & set(b):
            dotted = f'{prefix}.{key}' if prefix else str(key)
            _record_index(a[key], b[key], dotted, index, ambiguous)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _record_index(x, y, f'{prefix}[{i}]', index, ambiguous)


def _resolve_candidate(diff, index, ambiguous):
    """(record_path, key) if `diff` is <MISSING> (reference) vs None (new) and resolves to exactly one key that is
    absent in the reference record and None in the new record; else (None, reason)."""
    if not (diff.get('legacy') == MISSING and isinstance(diff.get('legacy'), str)):
        return None, 'reference side is not <MISSING>'
    if diff.get('lightweight') is not None:
        return None, 'absent in reference, non-null in new arm'
    field = str(diff.get('field', ''))
    matches = []
    for pos, ch in enumerate(field):
        if ch != '.':
            continue
        path, kstr = field[:pos], field[pos + 1:]
        if path not in index:
            continue
        rec_a, rec_b = index[path]
        for k in rec_b:
            if str(k) == kstr and k not in rec_a and rec_b[k] is None:
                matches.append((path, k))
    if len(matches) != 1:
        return None, f'does not resolve to exactly one record key ({len(matches)} matches)'
    path, key = matches[0]
    if path in ambiguous:
        return None, 'record path ambiguous'
    return (path, key), None


def _strict_equal(x, y):
    if type(x) is not type(y):
        return False
    if isinstance(x, dict):
        return set(x) == set(y) and all(_strict_equal(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return len(x) == len(y) and all(_strict_equal(p, q) for p, q in zip(x, y))
    return x == y


def apply_rule_c(fname, ref, new, classified):
    """Rule (c) on one compared file. `classified` = W10.classify(fname, ref, new, token_groups) (the gate's own
    output for this file). Returns the ruled split of its genuine diffs."""
    if classified.get('genuine_diffs_listed_truncated'):
        return {'error': 'genuine list truncated by the gate; rule (c) cannot see every genuine diff',
                'n': {'genuine_frozen': classified['n']['genuine'], 'field_list_drift': 0,
                      'genuine_ruled': classified['n']['genuine']}}
    raw = CP._diff(ref, new, fname)            # the comparator's raw diff list (the call classify makes)
    index, ambiguous = {}, set()
    _record_index(ref, new, fname, index, ambiguous)
    genuine = list(classified['genuine_diffs_first'])
    candidates, non_candidates = {}, []
    for d in genuine:
        resolved, reason = _resolve_candidate(d, index, ambiguous)
        if resolved is None:
            non_candidates.append(dict(d, rule_c_reason=reason))
        else:
            candidates.setdefault(resolved[0], []).append((resolved[1], d))
    drift, disqualified, records = [], [], {}
    for path, items in candidates.items():
        rec_a, rec_b = index[path]
        shared = sorted((set(rec_a) & set(rec_b)), key=str)
        # direct-child diffs on NON-shared keys (absent on one side) are not diffs of a shared key: they never block
        # (each is either a drift candidate of this record or stays genuine on its own)
        non_shared_fields = {f'{path}.{k}' for k in (set(rec_a) ^ set(rec_b))}
        sub = path + '.'
        sub_list = path + '['
        blocking = [r for r in raw if (str(r.get('field', '')).startswith(sub)
                                       or str(r.get('field', '')).startswith(sub_list))
                    and str(r.get('field', '')) not in non_shared_fields]
        strict_all_shared_equal = all(_strict_equal(rec_a[k], rec_b[k]) for k in shared)
        records[path] = {'n_candidates': len(items), 'n_shared_keys': len(shared),
                         'shared_keys_equal_under_gate_comparator': not blocking,
                         'shared_keys_strict_type_exact_equal_incl_comparator_exclusions': strict_all_shared_equal,
                         'blocking_diffs': blocking}
        for key, d in items:
            if blocking:
                disqualified.append(dict(d, rule_c_reason=(
                    f'record {path}: {len(blocking)} raw diff(s) on keys shared by the two records '
                    '(not every shared key bitwise equal) -- drift classification disqualified')))
            else:
                drift.append({'field': d['field'], 'record': path, 'key': str(key), 'reference': MISSING,
                              'new_arm': None})
    genuine_ruled = non_candidates + disqualified
    return {
        'n': {'genuine_frozen': len(genuine), 'field_list_drift': len(drift), 'genuine_ruled': len(genuine_ruled)},
        'drift_keys': sorted({x['key'] for x in drift}),
        'field_list_drift': drift,
        'records_with_candidates': records,
        'genuine_ruled': genuine_ruled,
    }


def ruled_comparison(gate_out_root, gate_arm_name, reference_arm):
    """The gate's own comparison (S52G.compare_against_w39_arm with the reference arm and arm name substituted, as
    the gate itself substitutes them), then rule (c) per file on the same loaded objects."""
    S52G.W39_ARM = reference_arm
    S52G.ARM = gate_arm_name
    comparison = S52G.compare_against_w39_arm(gate_out_root)
    ref_dir = os.path.join(REPO, reference_arm['arm_dir'])
    new_dir = os.path.join(gate_out_root, 'arm')
    ruled, loaded = {}, {}
    for fname in COMPARED_FILES:
        loader = CP._load_json if fname in W10.GATING_JSON else CP._load_jsonl
        ref, new = loader(os.path.join(ref_dir, fname)), loader(os.path.join(new_dir, fname))
        loaded[fname] = (ref, new)
        entry = comparison['per_file'][fname]
        if 'error' in entry:
            ruled[fname] = {'error': entry['error']}
            continue
        ruled[fname] = apply_rule_c(fname, ref, new, entry)
    return comparison, ruled, loaded


def ruled_verdict(frozen_gate_items, frozen_gate_pass, comparison, ruled, frozen_comparison):
    zero_items = [k for k in frozen_gate_items if k.endswith(ZERO_GENUINE_ITEM_SUFFIX)]
    other_items = {k: v for k, v in frozen_gate_items.items() if k not in zero_items}
    reproduces = (frozen_comparison is not None
                  and _canon(comparison['per_file']) == _canon(frozen_comparison['per_file'])
                  and _canon(comparison['totals']) == _canon(frozen_comparison['totals']))
    totals = {'genuine_frozen': 0, 'field_list_drift': 0, 'genuine_ruled': 0}
    errors = {}
    for fname, r in ruled.items():
        if 'error' in r:
            errors[fname] = r['error']
        for k in totals:
            totals[k] += (r.get('n') or {}).get(k, 0)
    drift_keys = sorted({k for r in ruled.values() for k in r.get('drift_keys', [])})
    items = {
        'exactly_one_frozen_zero_genuine_item': len(zero_items) == 1,
        'frozen_items_other_than_zero_genuine_all_true_as_recorded': (
            bool(other_items) and all(v is True for v in other_items.values())),
        'all_compared_files_present': comparison['all_files_present'] and not any(
            'error' in comparison['per_file'][f] for f in COMPARED_FILES),
        'recomputed_comparison_reproduces_the_frozen_one': reproduces,
        'no_file_left_undetermined_by_rule_c': not errors,
        'zero_genuine_diffs_after_rule_c': totals['genuine_ruled'] == 0 and not errors,
    }
    return {
        'GATE_PASS': frozen_gate_pass,
        'GATE_PASS_RULED': all(items.values()),
        'ruled_items': items,
        'frozen_items_carried_as_recorded': other_items,
        'frozen_zero_genuine_item_replaced': {k: frozen_gate_items[k] for k in zero_items},
        'totals': totals,
        'drift_keys': drift_keys,
        'n_drift_keys': len(drift_keys),
        'per_file_errors': errors,
    }


# ======================================================================================================================
#  negative controls
# ======================================================================================================================
def negative_controls(token_groups, ruled, loaded, comparison, frozen_items, frozen_gate_pass, frozen_comparison):
    fname = 'per_cycle_record.jsonl'

    def run(label, plant, expect, fname=fname):
        ref_rows, new_rows = loaded[fname]
        rows = copy.deepcopy(new_rows)
        planted = plant(rows)
        entry = W10.classify(fname, ref_rows, rows, token_groups)
        rc = apply_rule_c(fname, ref_rows, rows, entry)
        comp = copy.deepcopy(comparison)
        comp['per_file'][fname] = entry
        comp['totals'] = {k: sum((v.get('n') or {}).get(k, 0) for v in comp['per_file'].values())
                          for k in ('provenance', 'tie_order', 'genuine')}
        rl = dict(ruled)
        rl[fname] = rc
        # the reproduction item compares against the frozen comparison WITH the planted file in place, so that only
        # rule (c) decides the planted verdict (the real reproduction is checked on the unplanted data).
        frozen = copy.deepcopy(frozen_comparison)
        frozen['per_file'][fname] = entry
        frozen['totals'] = comp['totals']
        verdict = ruled_verdict(frozen_items, frozen_gate_pass, comp, rl, frozen)
        genuine_fields = sorted(str(d['field']) for d in rc['genuine_ruled'])
        drift_by_row = {}
        for x in rc['field_list_drift']:
            drift_by_row[x['record']] = drift_by_row.get(x['record'], 0) + 1
        observed = {'gate_classify_n': entry['n'], 'rule_c_n': rc['n'], 'drift_by_record': drift_by_row,
                    'genuine_ruled_fields': genuine_fields,
                    'genuine_ruled_reasons': sorted({d.get('rule_c_reason', '') for d in rc['genuine_ruled']}),
                    'GATE_PASS_RULED': verdict['GATE_PASS_RULED']}
        checks = expect(observed)
        ok = all(checks.values())
        _log(f'{label}: {"OK" if ok else "FAILED"} rule_c {rc["n"]} drift_by_record {drift_by_row} '
             f'GATE_PASS_RULED {verdict["GATE_PASS_RULED"]}')
        return {'planted': planted, 'observed': observed, 'checks': checks, 'ok': ok}

    r0, r1 = f'{fname}[0]', f'{fname}[1]'
    n_keys = len(H.PER_CYCLE_RESPONSE_FIELDS)

    def nc0(rows):
        return 'nothing planted (positive control)'

    def nc1(rows):
        rows[0]['recourse'] = rows[0]['recourse'] + 1.0
        return 'row 0 recourse += 1.0 (shared key, different value)'

    def nc2(rows):
        rows[0]['rss_bytes'] = 123456789
        rows[1]['n_non_optimal_block_terminations'] = 0
        rows[1]['planted_new_key'] = 'x'
        return ("row 0 rss_bytes = 123456789; row 1 n_non_optimal_block_terminations = 0 (falsy, non-null); "
                "row 1 planted_new_key = 'x' (keys absent in the reference, non-null in the new arm)")

    def nc3(rows):
        rows[1]['boyd_v_primal_ratio'] = math.nextafter(rows[1]['boyd_v_primal_ratio'], math.inf)
        return 'row 1 boyd_v_primal_ratio -> math.nextafter(v, inf) (shared key, one ulp)'

    def nc4(rows):
        del rows[0]['cycle_convergence']
        return 'row 0 cycle_convergence deleted (present in the reference, absent in the new arm)'

    g = 'g_s39_D.json'

    def nc5a(obj):
        obj['instance']['planted_drift_key'] = None
        return "g_s39_D.json instance.planted_drift_key = None (nested record; its shared keys all equal)"

    def nc5b(obj):
        obj['planted_drift_key'] = None
        return ("g_s39_D.json top-level planted_drift_key = None (the top-level record's shared key "
                "rule_eleven_checklist differs at depth 2 by provenance diffs)")

    controls = {
        'NC0_positive_unplanted': run('NC0', nc0, lambda o: {
            'drift_44_as_22_per_row': o['drift_by_record'] == {r0: n_keys, r1: n_keys},
            'genuine_ruled_0': o['rule_c_n']['genuine_ruled'] == 0,
            'gate_pass_ruled_true': o['GATE_PASS_RULED'] is True}),
        'NC1_shared_key_different_value': run('NC1', nc1, lambda o: {
            'planted_diff_genuine': f'{r0}.recourse' in o['genuine_ruled_fields'],
            'row0_drift_disqualified': r0 not in o['drift_by_record'],
            'row1_drift_kept': o['drift_by_record'].get(r1) == n_keys,
            'gate_pass_ruled_false': o['GATE_PASS_RULED'] is False}),
        'NC2_absent_in_reference_non_null_in_new': run('NC2', nc2, lambda o: {
            'rss_bytes_genuine': f'{r0}.rss_bytes' in o['genuine_ruled_fields'],
            'falsy_zero_genuine': f'{r1}.n_non_optimal_block_terminations' in o['genuine_ruled_fields'],
            'new_key_genuine': f'{r1}.planted_new_key' in o['genuine_ruled_fields'],
            'reason_recorded': 'absent in reference, non-null in new arm' in o['genuine_ruled_reasons'],
            'other_drift_kept_rule_literal': o['drift_by_record'] == {r0: n_keys - 1, r1: n_keys - 1},
            'gate_pass_ruled_false': o['GATE_PASS_RULED'] is False}),
        'NC3_shared_key_one_ulp_disqualifies_record': run('NC3', nc3, lambda o: {
            'planted_diff_genuine': f'{r1}.boyd_v_primal_ratio' in o['genuine_ruled_fields'],
            'row1_all_candidates_genuine': sum(f.startswith(r1 + '.') for f in o['genuine_ruled_fields']) == n_keys + 1,
            'row1_drift_disqualified': r1 not in o['drift_by_record'],
            'row0_drift_kept': o['drift_by_record'].get(r0) == n_keys,
            'gate_pass_ruled_false': o['GATE_PASS_RULED'] is False}),
        'NC4_reference_key_missing_in_new': run('NC4', nc4, lambda o: {
            'planted_diff_genuine': f'{r0}.cycle_convergence' in o['genuine_ruled_fields'],
            'genuine_ruled_is_only_the_planted_diff': o['genuine_ruled_fields'] == [f'{r0}.cycle_convergence'],
            'drift_kept_rule_literal': o['drift_by_record'] == {r0: n_keys, r1: n_keys},
            'gate_pass_ruled_false': o['GATE_PASS_RULED'] is False}),
        'NC5a_nested_record_drift_positive': run('NC5a', nc5a, lambda o: {
            'nested_drift': o['drift_by_record'] == {f'{g}.instance': 1},
            'genuine_ruled_0': o['rule_c_n']['genuine_ruled'] == 0,
            'gate_pass_ruled_true': o['GATE_PASS_RULED'] is True}, fname=g),
        'NC5b_shared_key_differs_at_depth_any_class_disqualifies': run('NC5b', nc5b, lambda o: {
            'candidate_genuine': f'{g}.planted_drift_key' in o['genuine_ruled_fields'],
            'no_drift': o['drift_by_record'] == {},
            'gate_pass_ruled_false': o['GATE_PASS_RULED'] is False}, fname=g),
    }
    return controls, all(c['ok'] for c in controls.values())


# ======================================================================================================================
#  modes
# ======================================================================================================================
def _guards_record():
    zero = ZERO_GUARD.verify(0)
    w10 = GUARD.verify(0)
    return {'zero_solve_guard': {'counts': dict(ZERO_GUARD.counts), 'verify_0_failures': zero},
            'imported_chain_W10_GUARD': {'counts': dict(GUARD.counts), 'verify_0_failures': w10},
            'pickle': dict(PICKLE_COUNTS), 'model_build': dict(MODEL_BUILD_COUNTS),
            'all_zero': (not zero and not w10 and PICKLE_COUNTS == {'load': 0, 'loads': 0}
                         and MODEL_BUILD_COUNTS['ConcreteModel.__init__'] == 0)}


def _out_dir_precondition(out_dir):
    if not os.path.isdir(out_dir):
        return f'output directory {out_dir} does not exist (create it with the launch-log redirect)'
    extra = sorted(set(os.listdir(out_dir)) - {'launch.log'})
    if extra:
        return f'output directory {out_dir} is not fresh (holds {extra}); write-once'
    return None


def _write_json(path, payload):
    W10._refuse_overwrite(path)
    text = GRIO.dumps(payload, indent=1)
    with open(path, 'w') as handle:
        handle.write(text)


def mode_ruled_from_committed(out_rel):
    started = time.time()
    out_dir = os.path.join(REPO, out_rel)
    _log(f'{SCHEMA}: rule (c) on the committed W183f gate output vs the W48 arm (zero solves)')
    problem = _out_dir_precondition(out_dir)
    if problem:
        _log(f'[REFUSED] {problem}')
        return 1
    lock = os.path.join(out_dir, '.w183g_a_v2.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        _log('[REFUSED] another copy holds the lock')
        return 1
    try:
        return _ruled_body(out_dir, started)
    finally:
        os.close(fd)
        os.remove(lock)


def _ruled_body(out_dir, started):
    head = W10._git(['rev-parse', 'HEAD'])
    gate_root = os.path.join(REPO, W183F['out_rel'])
    # ---- hash checks first; refuse on any mismatch ----
    problems_new, chk_new = check_gate_dir(W183F['out_rel'], W183F['manifests'])
    ref_manifests = {os.path.basename(REFERENCE_ARM['manifest']): REFERENCE_ARM['manifest_sha256'],
                     **REFERENCE_EXTRA_MANIFESTS}
    problems_ref, chk_ref = check_gate_dir(REFERENCE_ARM['gate_dir'], ref_manifests)
    S52G.W39_ARM = REFERENCE_ARM
    problems_pin, _m = S52G._check_w39_arm_pins()      # the gate's own pin check of the reference arm
    problems = problems_new + problems_ref + problems_pin
    for p in problems:
        _log(f'[REFUSED] {p}')
    _log(f'hash checks: W183f {chk_new["file_entries_checked"]} entries, reference {chk_ref["file_entries_checked"]} '
         f'entries, gate pin check {len(problems_pin)} problems')
    if problems:
        return 1

    with open(os.path.join(gate_root, 'w51_gate_addendum.json')) as handle:
        addendum = json.load(handle)
    with open(os.path.join(gate_root, 'w183f_wrapper_record.json')) as handle:
        wrapper = json.load(handle)
    frozen_items = addendum['gate_items']
    frozen_gate_pass = wrapper['gate_pass']
    frozen_comparison = addendum['comparison_vs_committed_w48_arm_GATING']
    consistent = (frozen_gate_pass is addendum['gate_pass']
                  and wrapper['declared_substitutions']['1_identifiers']['ARM'] == W183F['arm_name'])
    _log(f'frozen GATE_PASS (wrapper record) = {frozen_gate_pass}; w51 addendum gate_pass = {addendum["gate_pass"]}; '
         f'consistent {consistent}')
    if not consistent:
        _log('[REFUSED] frozen records inconsistent')
        return 1

    comparison, ruled, loaded = ruled_comparison(gate_root, W183F['arm_name'], REFERENCE_ARM)
    verdict = ruled_verdict(frozen_items, frozen_gate_pass, comparison, ruled, frozen_comparison)
    for fname in COMPARED_FILES:
        _log(f'   {fname}: gate {comparison["per_file"][fname].get("n")} -> rule (c) {ruled[fname].get("n")}')
    _log(f'totals {verdict["totals"]}; drift keys {verdict["n_drift_keys"]}: {verdict["drift_keys"]}')
    drift_equals_harness = verdict['drift_keys'] == sorted(H.PER_CYCLE_RESPONSE_FIELDS)
    _log(f'drift keys == sorted(p515_s44_campaign_harness.PER_CYCLE_RESPONSE_FIELDS): {drift_equals_harness}')
    _log(f'GATE_PASS={verdict["GATE_PASS"]} (as frozen); GATE_PASS_RULED={verdict["GATE_PASS_RULED"]}; '
         f'items {verdict["ruled_items"]}')

    controls, controls_ok = negative_controls(comparison['token_groups'], ruled, loaded, comparison, frozen_items,
                                              frozen_gate_pass, frozen_comparison)
    guards = _guards_record()
    _log(f'negative controls all as declared: {controls_ok}; guards {guards}')

    payload = {
        'schema': SCHEMA + '_ruled_from_committed', 'authority': AUTHORITY, 'rule': RULE_TEXT,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_head_at_run': head,
        'script': SCRIPT, 'script_sha256': _sha(os.path.abspath(__file__)),
        'imported_sha256': {n: _sha(os.path.join(REPO, n)) for n in (
            'p515_s55_w183f_bitwise_gate.py', 'p515_s53_srp1_bitwise_gate.py', 'p515_s52_srp1_bitwise_gate.py',
            'p515_s45_snapshot_off_two_cycle_gate.py', 'p515_s40_clone_capture_preflight.py',
            'p515_s43_aa_flagoff_gate.py', 'p515_s44_tie_classifier.py', 'p515_s44_campaign_harness.py',
            'gate_result_io.py', 'p513_solve_profile_guard.py')},
        'nothing_re_run': 'no solve; the gate is not re-run; the committed W183f outputs are read only',
        'instance': {'candidate': 'C* 0.96875 MVA / 3.875 MWh at nodes 5, 7, 9, investment year 2025',
                     'candidate_key': W10.C_STAR_KEY_PIN},
        'objective_convention': 'gross_operational_cost (settlement-excluded gross cost), as in the gate',
        'new_arm': {**W183F, 'gate_dir_check': chk_new},
        'reference_arm': {**REFERENCE_ARM, 'gate_dir_check': chk_ref, 'gate_pin_check_problems': problems_pin},
        'comparator': ('the gate\'s own: S52G.compare_against_w39_arm -> W10.classify (CP._diff bitwise with the '
                       'declared exclusions; rule_eleven_checklist subtree = provenance; arm-path tokens normalized; '
                       'recourse-jump tie order by TC); compared files W10.GATING_JSON + W10.GATING_JSONL. '
                       "'legacy' = reference (W48) side, 'lightweight' = new (W183f) side"),
        'files_compared': list(COMPARED_FILES),
        'GATE_PASS': verdict['GATE_PASS'],
        'GATE_PASS_RULED': verdict['GATE_PASS_RULED'],
        'drift_keys': verdict['drift_keys'],
        'drift_keys_equal_harness_PER_CYCLE_RESPONSE_FIELDS': drift_equals_harness,
        'verdict': verdict,
        'per_file_gate': {f: comparison['per_file'][f].get('n', comparison['per_file'][f].get('error'))
                          for f in COMPARED_FILES},
        'per_file_rule_c': ruled,
        'gate_comparison_recomputed': comparison,
        'negative_controls': controls, 'negative_controls_all_as_declared': controls_ok,
        'guards': guards,
        'wall_clock_s': time.time() - started,
    }
    _write_json(os.path.join(out_dir, 'gate_ruled.json'), payload)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_dir):
        for fname in sorted(fnames):
            if fname in ('launch.log', '.w183g_a_v2.lock'):
                continue
            manifest[W10._rel(os.path.join(root, fname))] = _sha(os.path.join(root, fname))
    _write_json(os.path.join(out_dir, 'manifest_sha256.json'), manifest)
    _log(f'wrote {W10._rel(out_dir)}/gate_ruled.json, manifest_sha256.json ({len(manifest)} files)')
    _log(f'GATE_PASS={verdict["GATE_PASS"]} GATE_PASS_RULED={verdict["GATE_PASS_RULED"]}')
    return 0 if (controls_ok and guards['all_zero']) else 1


def mode_post_run(out_rel):
    out_dir = os.path.join(REPO, out_rel)
    path = os.path.join(out_dir, 'manifest_post_run_sha256.json')
    manifest = {}
    for root, _dirs, fnames in os.walk(out_dir):
        for fname in sorted(fnames):
            manifest[W10._rel(os.path.join(root, fname))] = _sha(os.path.join(root, fname))
    _write_json(path, manifest)
    guards = _guards_record()
    _log(f'wrote {W10._rel(path)} ({len(manifest)} files); guards all zero {guards["all_zero"]}')
    return 0 if guards['all_zero'] else 1


def mode_run_gate(arm, out_rel):
    """WRITTEN, NOT RUN at W183g (a). V1's flow with v2 identifiers, then the ruled verdict vs the W48 arm."""
    out_root = os.path.join(REPO, out_rel)
    if os.path.exists(out_root):
        _log(f'[PRECONDITION FAILED] output root exists: {out_rel}')
        return 1
    stage = f'P5.15 SRP1 two-cycle bitwise gate via wrapper v2 (Addendum 85 ruling 1 (c)), arm {arm}'
    head = W10._git(['rev-parse', 'HEAD'])
    code_files = tuple(V1.CODE_FILES) + (SCRIPT,)
    code_before = {n: CP._sha256_file(os.path.join(REPO, n)) for n in code_files}
    eval_root = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P56A', 'evals')
    stub = f'p515s44_s45_w10_snapoff_{arm}_{W10.C_STAR_KEY_PIN[:16]}'
    existing = [d for d in (f'{stub}_run', f'{stub}_precheck') if os.path.exists(os.path.join(eval_root, d))]
    if existing:
        for d in existing:
            _log(f'[PRECONDITION FAILED] arm-named eval dir already exists: {d}')
        return 1
    extra_clean = tuple(V1.EXTRA_CLEAN_FILES) + (SCRIPT,)
    S53G.STAGE, S53G.SCHEMA, S53G.ARM, S53G.OUT_REL = stage, SCHEMA, arm, out_rel
    S53G.EXTRA_CLEAN_FILES = extra_clean
    S53G.EXTRA_FORBIDDEN = tuple(V1.EXTRA_FORBIDDEN)
    status = S53G.main()
    code_after = {n: CP._sha256_file(os.path.join(REPO, n)) for n in code_files}
    w51_path = os.path.join(out_root, 'w51_gate_addendum.json')
    if not os.path.isfile(w51_path):
        _log('the committed layers refused before the arm; GATE_PASS=False; nothing ruled')
        return 1
    with open(w51_path) as handle:
        addendum = json.load(handle)
    gate_pass = status == 0 and addendum.get('gate_pass') is True
    comparison, ruled, _loaded = ruled_comparison(out_root, arm, REFERENCE_ARM)
    verdict = ruled_verdict(addendum['gate_items'], gate_pass, comparison, ruled,
                            addendum['comparison_vs_committed_w48_arm_GATING'])
    record = {
        'schema': SCHEMA + '_wrapper_record', 'stage': stage, 'authority': AUTHORITY, 'rule': RULE_TEXT,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'git_head_at_run': head,
        'code_sha256_before_run': code_before, 'code_sha256_after_run': code_after,
        'code_unchanged_during_run': code_before == code_after,
        'instance': {'candidate_key': W10.C_STAR_KEY_PIN},
        'declared_substitutions': {'1_identifiers': {'STAGE': stage, 'SCHEMA': SCHEMA, 'ARM': arm, 'OUT_REL': out_rel},
                                   '2_extra_clean_files': list(extra_clean),
                                   '3_extra_forbidden': list(V1.EXTRA_FORBIDDEN)},
        'committed_gate_exit_status': status,
        'GATE_PASS': gate_pass, 'GATE_PASS_RULED': verdict['GATE_PASS_RULED'],
        'drift_keys': verdict['drift_keys'], 'verdict': verdict, 'per_file_rule_c': ruled,
        'guard_counts_at_end': dict(GUARD.counts),
    }
    _write_json(os.path.join(out_root, 'w183g_v2_wrapper_record.json'), record)
    manifest = {}
    for root, _dirs, fnames in os.walk(out_root):
        for fname in sorted(fnames):
            manifest[W10._rel(os.path.join(root, fname))] = CP._sha256_file(os.path.join(root, fname))
    _write_json(os.path.join(out_root, 'w183g_v2_manifest_sha256.json'), manifest)
    _log(f'GATE_PASS={gate_pass} GATE_PASS_RULED={verdict["GATE_PASS_RULED"]}')
    return 0 if verdict['GATE_PASS_RULED'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--ruled-from-committed', action='store_true')
    mode.add_argument('--post-run', action='store_true')
    mode.add_argument('--run-gate', action='store_true')
    parser.add_argument('--arm')
    parser.add_argument('--out-rel')
    args = parser.parse_args(argv)
    if args.run_gate:
        if not (args.arm and args.out_rel):
            parser.error('--run-gate requires --arm and --out-rel')
        return mode_run_gate(args.arm, args.out_rel)
    if args.arm or args.out_rel:
        parser.error('--arm / --out-rel apply to --run-gate only')
    if args.post_run:
        return mode_post_run(OUT_REL_RULED)
    return mode_ruled_from_committed(OUT_REL_RULED)


if __name__ == '__main__':
    sys.exit(main())
