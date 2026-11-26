"""P5.15 Addendum 44 ("Harness"), task W74 item 1 -- ZERO-SOLVE checks of the two harness fixes in
`p515_s44_campaign_harness.py`:

  (a) per-pair / per-wave campaign heartbeat: `evaluate(batch, ctx, heartbeat_tag=None)`; untagged writes the shared
      `campaign_heartbeat.json` exactly as before, a tag writes `campaign_heartbeat_<tag>.json`, write-once per tag;
  (b) boolean flags: every artifact writer of the harness serializes a numpy boolean as a JSON boolean
      (`_json_default`), instead of the string "True"/"False" that `default=str` produced
      (`hull_polish_full.gate.pass` on five of the six alpha-row cells, P5_15_ADDENDUM40_42_REPORT.md section 8).

Checks (all read-only on committed artifacts; every write goes under the write-once root OUT):
  V1 `evaluation_key` unchanged: its source text is identical to the pre-W74 harness (git blob of PRE_W74_COMMIT,
     sha256-pinned), every top-level function / assignment the diff did not declare is textually identical, and the
     eval keys recomputed with the NEW harness for the six alpha-row cells equal the committed spec entries, the
     committed evaluation_record.json eval keys (each file verified against its pair manifest first) and the eval dir
     prefixes.
  V2 `PER_CYCLE_TRAJECTORY_FIELDS`, `PER_CYCLE_RESPONSE_FIELDS`, `PER_CYCLE_RECORD_FIELDS`, `RECORD_TRAJECTORY_FIELDS`
     equal the pre-W74 values (the four assignments executed from the pre-W74 source) and the key order of every line of
     the six committed per_cycle_record.jsonl files.
  V3 heartbeat collision: two sequential tagged launches ('pair_1', 'pair_2') into ONE stub campaign root through the
     REAL `evaluate()` with stub children (the harness's own `--stub-mode`: no model import, no solve); the first
     launch's end-state file survives byte-identical; a re-used tag is refused before anything launches. Negative
     control: two untagged launches into another root overwrite the single shared file (the pre-W74 behaviour, kept
     for untagged callers), and its payload keys are exactly the pre-W74 set. The periodic writer is exercised
     (HEARTBEAT_EVERY_S = 0, POLL_S = 0.5 in this process only).
  V4 boolean flags: a numpy boolean produced by the hull-polish gate's own expression (with the committed x0_a0p50
     gate numbers as numpy floats) is written by each harness writer and read back as a real bool (strict `is True` /
     `is False`); the pre-W74 hook (`default=str`) reproduces the string, as a negative control.
  V5 search scope: every `default=` hook in the harness, literal "True"/"False" strings in it, and the stringified
     booleans present in the committed alpha-row artifacts (by file and JSON path), with the writer of each.

A `SolveProfileGuard(permitted=())` is installed BEFORE any harness import and verified at exactly 0 at the end.

COMMAND (repo root, canonical interpreter, attached, both streams, noclobber):
  set -o noclobber
  /Users/micaelsimoes/miniconda3/envs/opf_env_py311/bin/python -u p515_s53_w74_harness_fix_checks.py \
      > data/SRP1/Results/P515S53/harness_fix_w74_launch.log 2>&1
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from p513_solve_profile_guard import SolveProfileGuard  # noqa: E402

GUARD = SolveProfileGuard(permitted=(), label='P5.15 W74 harness fix checks').install()

import numpy as np  # noqa: E402
import p515_s44_campaign_harness as H  # noqa: E402

OUT = os.path.join(REPO, 'data', 'SRP1', 'Results', 'P515S53', 'harness_fix_w74')
OUT_JSON = os.path.join(OUT, 'harness_fix_w74_checks.json')
OUT_MANIFEST = os.path.join(OUT, 'harness_fix_w74_manifest_sha256.json')
LAUNCH_LOG_REL = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'harness_fix_w74_launch.log')

PRE_W74_COMMIT = '5fc92a8d'
PRE_W74_HARNESS_SHA256 = '5c67acc27d2e56c8783fc93705cd5de7201cf0800a969281ae8ebbb5c8a50589'
# Top-level names the W74 diff declares as changed (edited) or added; everything else must be textually identical.
DECLARED_CHANGED = {'_atomic_write_json', '_write_once_json', 'evaluate', 'alpha_row_run_hooks',
                    '_write_once_json_compact', 'write_response_terminal', '_child_real'}
DECLARED_ADDED = {'_json_default', 'heartbeat_file_name', 'HEARTBEAT_FILE'}
FIELD_TUPLES = ('PER_CYCLE_TRAJECTORY_FIELDS', 'PER_CYCLE_RESPONSE_FIELDS', 'PER_CYCLE_RECORD_FIELDS',
                'RECORD_TRAJECTORY_FIELDS')

ALPHA_ROOT = os.path.join('data', 'SRP1', 'Results', 'P515S53', 'alpha_row', 'campaign_s53_alpha_row_v25')
SPEC_REL = os.path.join(ALPHA_ROOT, 'campaign_spec_s53_alpha_row_v25_70965374.json')
CELLS = {'x0_a0p00': ('b123cd978794d690_x0_a0p00', 2), 'x0_a0p10': ('7516903c91153a29_x0_a0p10', 3),
         'x0_a0p25': ('62b46280a65f7744_x0_a0p25', 3), 'x0_a0p50': ('7d53b6f21b686a44_x0_a0p50', 1),
         'x0_a1p00': ('1bb2d63a07273887_x0_a1p00', 2), 'n7_4h_e1_a0p50': ('711fce9aa74d6878_n7_4h_e1_a0p50', 1)}

C_STAR = {5: (0.96875, 3.875), 7: (0.96875, 3.875), 9: (0.96875, 3.875)}
PAPER = {5: (0.0, 0.0), 7: (1.62, 3.24), 9: (0.0, 0.0)}
N7E = {5: (0.96875, 3.875), 7: (0.0, 0.0), 9: (0.96875, 3.875)}
LEGACY_HEARTBEAT_KEYS = ['utc', 'parent_pid', 'running', 'pending', 'done']


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    return H.sha256_file(path if os.path.isabs(path) else os.path.join(REPO, path))


def _expect_raise(fn, exc_types):
    try:
        fn()
    except exc_types as error:
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)[:300]}
    except BaseException as error:  # noqa: BLE001
        return {'raised': True, 'type': type(error).__name__, 'message': str(error)[:300], 'unexpected_type': True}
    return {'raised': False}


# ----------------------------------------------------------------------------------------------------------------------
def pre_w74_source():
    text = subprocess.run(['git', 'show', f'{PRE_W74_COMMIT}:p515_s44_campaign_harness.py'], cwd=REPO,
                          capture_output=True, check=True).stdout
    got = hashlib.sha256(text).hexdigest()
    if got != PRE_W74_HARNESS_SHA256:
        raise RuntimeError(f'pre-W74 harness blob sha256 {got} != pinned {PRE_W74_HARNESS_SHA256}')
    return text.decode()


def _top_level(src):
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.get_source_segment(src, node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = ast.get_source_segment(src, node)
    return out


def v1_evaluation_key(old_src, new_src):
    old, new = _top_level(old_src), _top_level(new_src)
    changed = sorted(n for n in set(old) & set(new) if old[n] != new[n])
    added, removed = sorted(set(new) - set(old)), sorted(set(old) - set(new))
    spec = json.load(open(os.path.join(REPO, SPEC_REL)))
    cfg = spec['configuration']
    by_label = {c['label']: c for c in spec['candidates']}
    cells = {}
    for label, (eval_dir, pair) in CELLS.items():
        entry = by_label[label]
        manifest = json.load(open(os.path.join(REPO, ALPHA_ROOT, f'pair_{pair}_manifest_sha256.json')))
        rec_rel = os.path.join(ALPHA_ROOT, 'evals', eval_dir, 'evaluation_record.json')
        rec_ok = manifest.get(rec_rel) == _sha(rec_rel)
        rec = json.load(open(os.path.join(REPO, rec_rel)))
        canon = entry['canonical']
        ckey = H.candidate_key(H.canonical_candidate(canon['nodes'], investment_year=canon['investment_year']))
        ekey = H.evaluation_key(ckey, entry.get('overrides') or {},
                                case_file_aa=cfg.get('case_file_anderson_acceleration'),
                                model_variant=entry.get('model_variant'),
                                ess_ageing_baseline=cfg.get('ess_ageing_baseline'),
                                flex_price_multiplier=entry.get('flex_price_multiplier'),
                                derived_instance=cfg.get('derived_instance'),
                                interface_deviation_premium=entry.get('interface_deviation_premium'))
        cells[label] = {'eval_dir': eval_dir, 'record_verified_against_pair_manifest': rec_ok,
                        'candidate_key_recomputed': ckey, 'candidate_key_spec': entry['key'],
                        'eval_key_recomputed': ekey, 'eval_key_spec': entry['eval_key'],
                        'eval_key_record': rec.get('eval_key'),
                        'match': (ckey == entry['key'] and ekey == entry['eval_key'] == rec.get('eval_key')
                                  and eval_dir == entry['eval_dir'] and eval_dir.startswith(ekey[:16]) and rec_ok)}
    checks = {
        'evaluation_key_source_identical': old.get('evaluation_key') == new.get('evaluation_key'),
        'changed_top_level_names_exactly_declared': set(changed) == DECLARED_CHANGED,
        'added_top_level_names_exactly_declared': set(added) == DECLARED_ADDED,
        'no_top_level_name_removed': removed == [],
        'spec_file_sha8_matches_name': _sha(SPEC_REL)[:8] == '70965374',
        'six_cells_eval_keys_equal_committed': all(c['match'] for c in cells.values()) and len(cells) == 6,
    }
    return {'checks': checks, 'changed_top_level': changed, 'added_top_level': added, 'removed_top_level': removed,
            'spec': SPEC_REL, 'spec_sha256': _sha(SPEC_REL), 'cells': cells}


def v2_field_tuples(old_src):
    tree = ast.parse(old_src)
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in FIELD_TUPLES for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), '<pre-W74>', 'exec'), ns)  # noqa: S102
    equal = {name: (ns.get(name) == getattr(H, name)) for name in FIELD_TUPLES}
    lines = {}
    for label, (eval_dir, _pair) in CELLS.items():
        path = os.path.join(REPO, ALPHA_ROOT, 'evals', eval_dir, 'per_cycle_record.jsonl')
        orders = set()
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    orders.add(tuple(json.loads(line).keys()))
        lines[label] = {'n_distinct_key_orders': len(orders),
                        'equals_PER_CYCLE_RECORD_FIELDS': orders == {tuple(H.PER_CYCLE_RECORD_FIELDS)}}
    checks = {f'{n}_unchanged': v for n, v in equal.items()}
    checks['committed_per_cycle_record_key_order_equals_fields'] = all(
        v['equals_PER_CYCLE_RECORD_FIELDS'] for v in lines.values())
    return {'checks': checks, 'lengths': {n: len(getattr(H, n)) for n in FIELD_TUPLES}, 'per_cycle_record': lines}


# ----------------------------------------------------------------------------------------------------------------------
def _stub_spec(root, campaign_id, candidates):
    return H.freeze_campaign_spec(
        root, campaign_id, candidates,
        configuration={'name': 'STUB (W74 checks only)', 'arm_label': 's39_D', 'overrides': {}},
        cap=500, concurrency=1, authority=['W74 checks only'], extra={'test_only_stub': True})


def _read(path):
    with open(path, 'rb') as handle:
        raw = handle.read()
    return raw, hashlib.sha256(raw).hexdigest(), json.loads(raw)


def v3_heartbeat():
    H.HEARTBEAT_EVERY_S = 0.0   # this process only: exercise the periodic writer on every poll
    H.POLL_S = 0.5
    log_lines = []

    def log(message):
        log_lines.append(message)
        print(message, flush=True)

    legacy_absent = os.path.join(OUT, 'no_legacy.lock')
    out = {'monkeypatched_in_this_process': {'HEARTBEAT_EVERY_S': 0.0, 'POLL_S': 0.5}}

    # --- tagged: two sequential pair launches into ONE root --------------------------------------------------------
    root = os.path.join(OUT, 'v3_tagged_root')
    lock = os.path.join(OUT, 'v3_tagged.lock')
    path, sha, spec = _stub_spec(root, 'w74v3t', [('p1', C_STAR), ('p2', PAPER), ('p3', N7E)])
    extra = ['--lock-path', lock, '--stub-mode', 'ok', '--stub-sleep-s', '2', '--stub-alloc-mb', '16']
    hb1 = os.path.join(root, H.heartbeat_file_name('pair_1'))
    hb2 = os.path.join(root, H.heartbeat_file_name('pair_2'))
    H.acquire_campaign_lock('w74v3t', sha, lock_path=lock, legacy_lock_path=legacy_absent)
    try:
        n0 = len(log_lines)
        r1 = H.evaluate(['p1'], H.CampaignContext(root, path, sha, spec, log=log, child_extra_args=extra),
                        heartbeat_tag='pair_1')
        info1 = dict(H.evaluate.last_batch_info)
        periodic_1 = sum(1 for m in log_lines[n0:] if m.startswith('[S44-HARNESS] heartbeat '))
        raw1, sha1, hb1_after_pair1 = _read(hb1)
        n0 = len(log_lines)
        r2 = H.evaluate(['p2'], H.CampaignContext(root, path, sha, spec, log=log, child_extra_args=extra),
                        heartbeat_tag='pair_2')
        info2 = dict(H.evaluate.last_batch_info)
        periodic_2 = sum(1 for m in log_lines[n0:] if m.startswith('[S44-HARNESS] heartbeat '))
        raw1b, sha1b, hb1_after_pair2 = _read(hb1)
        _raw2, sha2, hb2_end = _read(hb2)
        reuse = _expect_raise(lambda: H.evaluate(
            ['p3'], H.CampaignContext(root, path, sha, spec, log=log, child_extra_args=extra),
            heartbeat_tag='pair_1'), RuntimeError)
        p3_dir = os.path.join(root, 'evals', next(c['eval_dir'] for c in spec['candidates'] if c['label'] == 'p3'))
        _raw1c, sha1c, _ = _read(hb1)
    finally:
        H.release_campaign_lock(lock_path=lock, expected_pid=os.getpid())
    root_listing = sorted(os.listdir(root))
    tagged_checks = {
        'pair_1_stub_ok': [x.get('status') for x in r1] == ['stub'],
        'pair_2_stub_ok': [x.get('status') for x in r2] == ['stub'],
        'periodic_writer_ran_in_both_pairs': periodic_1 >= 1 and periodic_2 >= 1,
        'pair_1_end_state_survives_pair_2_byte_identical': raw1 == raw1b and sha1 == sha1b,
        'pair_1_end_state_content': (hb1_after_pair2.get('done') == ['p1'] and hb1_after_pair2.get('running') == []
                                     and hb1_after_pair2.get('pending') == []
                                     and hb1_after_pair2.get('heartbeat_tag') == 'pair_1'),
        'pair_2_end_state_content': (hb2_end.get('done') == ['p2'] and hb2_end.get('running') == []
                                     and hb2_end.get('heartbeat_tag') == 'pair_2'),
        'distinct_files': hb1 != hb2 and sha1 != sha2,
        'no_shared_legacy_file_written_by_tagged_calls': H.HEARTBEAT_FILE not in root_listing,
        'last_batch_info_names_the_file': (info1.get('heartbeat_file') == os.path.relpath(hb1, REPO)
                                           and info2.get('heartbeat_file') == os.path.relpath(hb2, REPO)),
        'reused_tag_refused': reuse['raised'] and reuse['type'] == 'RuntimeError' and not reuse.get('unexpected_type'),
        'reused_tag_refused_before_launch': not os.path.exists(p3_dir),
        'pair_1_file_unchanged_after_refusal': sha1c == sha1,
    }
    out['tagged'] = {'checks': tagged_checks, 'root_listing': root_listing, 'pair_1_file': os.path.relpath(hb1, REPO),
                     'pair_1_sha256_after_pair_1': sha1, 'pair_1_sha256_after_pair_2': sha1b,
                     'pair_1_content_after_pair_2': hb1_after_pair2, 'pair_2_content': hb2_end,
                     'periodic_heartbeats_logged': {'pair_1': periodic_1, 'pair_2': periodic_2},
                     'reuse_refusal': reuse, 'batch_info': {'pair_1': info1, 'pair_2': info2}}

    # --- untagged negative control: the pre-W74 behaviour, kept for untagged callers ------------------------------
    root_u = os.path.join(OUT, 'v3_untagged_root')
    lock_u = os.path.join(OUT, 'v3_untagged.lock')
    path_u, sha_u, spec_u = _stub_spec(root_u, 'w74v3u', [('u1', C_STAR), ('u2', PAPER)])
    extra_u = ['--lock-path', lock_u, '--stub-mode', 'ok', '--stub-sleep-s', '2', '--stub-alloc-mb', '16']
    hbu = os.path.join(root_u, H.HEARTBEAT_FILE)
    H.acquire_campaign_lock('w74v3u', sha_u, lock_path=lock_u, legacy_lock_path=legacy_absent)
    try:
        H.evaluate(['u1'], H.CampaignContext(root_u, path_u, sha_u, spec_u, log=log, child_extra_args=extra_u))
        info_u1 = dict(H.evaluate.last_batch_info)
        _r, shau1, hbu1 = _read(hbu)
        H.evaluate(['u2'], H.CampaignContext(root_u, path_u, sha_u, spec_u, log=log, child_extra_args=extra_u))
        _r, shau2, hbu2 = _read(hbu)
    finally:
        H.release_campaign_lock(lock_path=lock_u, expected_pid=os.getpid())
    untagged_checks = {
        'untagged_file_name_is_pre_w74': H.heartbeat_file_name(None) == 'campaign_heartbeat.json',
        'untagged_payload_keys_exactly_pre_w74': list(hbu1) == LEGACY_HEARTBEAT_KEYS and list(hbu2) == LEGACY_HEARTBEAT_KEYS,
        'untagged_batch_info_keys_pre_w74': sorted(info_u1) == ['max_concurrent_observed', 'timeline'],
        'negative_control_second_untagged_launch_overwrites_first': (hbu1.get('done') == ['u1']
                                                                      and hbu2.get('done') == ['u2'] and shau1 != shau2),
    }
    out['untagged'] = {'checks': untagged_checks, 'root_listing': sorted(os.listdir(root_u)),
                       'after_u1': hbu1, 'after_u2': hbu2}
    out['checks'] = {**{f'tagged_{k}': v for k, v in tagged_checks.items()},
                     **{f'untagged_{k}': v for k, v in untagged_checks.items()}}
    return out


# ----------------------------------------------------------------------------------------------------------------------
def _hp_gate_threshold_pct():
    src = open(os.path.join(REPO, 'p515_s41_hull_polish.py')).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'GATE_THRESHOLD_PCT'
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError('GATE_THRESHOLD_PCT')


def _write_line(path, text):
    with open(path, 'x') as handle:
        handle.write(text + '\n')


def v4_booleans():
    pc_rel = os.path.join(ALPHA_ROOT, 'evals', CELLS['x0_a0p50'][0], 'post_certification.json')
    manifest = json.load(open(os.path.join(REPO, ALPHA_ROOT, 'pair_1_manifest_sha256.json')))
    pc_ok = manifest.get(pc_rel) == _sha(pc_rel)
    gate = json.load(open(os.path.join(REPO, pc_rel)))['hull_polish_full']['gate']
    threshold_pct = _hp_gate_threshold_pct()
    # the gate's own expression (p515_s41_hull_polish._polish_all_blocks_hull), on numpy floats as at alpha > 0
    delta = np.float64(gate['delta_sum_blocks'])
    certified = np.float64(gate['certified_cost'])
    relative = abs(delta) / abs(certified) if certified else None
    gate_pass = (relative is not None and relative < threshold_pct / 100.0)
    gate_fail = (relative is not None and relative < np.float64(0.0))
    payload = {'hull_polish_full': {'gate': {'pass': gate_pass, 'fail_case': gate_fail}}, 'plain_float': relative}
    results = {}
    writers = {
        '_write_once_json': lambda p: H._write_once_json(p, payload),
        '_atomic_write_json': lambda p: H._atomic_write_json(p, payload),
        '_write_once_json_compact': lambda p: H._write_once_json_compact(p, payload),
        'jsonl_line_with__json_default': lambda p: _write_line(p, json.dumps(payload, default=H._json_default)),
    }
    for name, write in writers.items():
        p = os.path.join(OUT, f'v4_{name}.json')
        write(p)
        back = json.loads(open(p).read())
        g = back['hull_polish_full']['gate']
        results[name] = {'pass_type': type(g['pass']).__name__, 'pass_is_True': g['pass'] is True,
                         'fail_case_is_False': g['fail_case'] is False,
                         'float_unchanged': back['plain_float'] == float(relative)}
    pre_w74 = json.loads(json.dumps(payload, default=str))['hull_polish_full']['gate']
    checks = {
        'input_verified_against_pair_1_manifest': pc_ok,
        'committed_value_is_the_string': gate['pass'] == 'True',
        'gate_expression_yields_numpy_bool': type(gate_pass).__module__ == 'numpy',
        'numpy_bool_is_not_a_python_bool': not isinstance(gate_pass, bool),
        'negative_control_pre_w74_hook_writes_string': pre_w74['pass'] == 'True' and pre_w74['fail_case'] == 'False',
        'every_writer_writes_real_bool_strict_is_True': all(r['pass_is_True'] and r['fail_case_is_False']
                                                            for r in results.values()),
        'floats_unaffected': all(r['float_unchanged'] for r in results.values()),
        'python_bool_unchanged': json.dumps({'a': True}, default=H._json_default) == '{"a": true}',
        'other_types_still_str': H._json_default(np.int64(5)) == '5' and H._json_default(object) == str(object),
    }
    return {'checks': checks, 'input': pc_rel, 'input_sha256': _sha(pc_rel), 'threshold_pct': threshold_pct,
            'relative_type': type(relative).__name__, 'gate_pass_type': f'{type(gate_pass).__module__}.'
                                                                        f'{type(gate_pass).__name__}',
            'per_writer': results, 'pre_w74_serialization': pre_w74}


# ----------------------------------------------------------------------------------------------------------------------
def _walk(obj, path, hits):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f'{path}.{k}', hits)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, f'{path}[]', hits)
    elif isinstance(obj, str) and obj in ('True', 'False'):
        hits[path] = hits.get(path, 0) + 1


HARNESS_WRITTEN = {'launch.json', 'wait4_rusage.json', 'evaluation_record.json', 'parent_barrier_record.json',
                   'post_certification.json', 'hull_bound_detail.json', 'per_cycle_record.jsonl', 'aa_per_cycle.jsonl',
                   'activation_readback.json', 'initialisation_identity.json', 'per_cycle_response.jsonl',
                   'response_terminal.json', 'multiscenario_terminal.json', 'child_manifest_sha256.json',
                   'campaign_heartbeat.json'}


def v5_scope(new_src):
    tree = ast.parse(new_src)
    hooks = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('dump', 'dumps')
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'json'):
            for kw in node.keywords:
                if kw.arg == 'default':
                    hooks.append({'line': node.lineno, 'hook': ast.get_source_segment(new_src, kw.value)})
    literal_bool_strings = [n.lineno for n in ast.walk(tree)
                            if isinstance(n, ast.Constant) and n.value in ('True', 'False')]
    hits_by_file = {}
    for dirpath, _dirs, files in os.walk(os.path.join(REPO, ALPHA_ROOT)):
        if f'{os.sep}esso_capture' in dirpath or f'{os.sep}planning_read' in dirpath:
            continue
        for fname in sorted(files):
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, REPO)
            hits = {}
            if fname.endswith('.json'):
                _walk(json.load(open(fpath)), '', hits)
            elif fname.endswith('.jsonl'):
                with open(fpath) as handle:
                    for line in handle:
                        if '"True"' in line or '"False"' in line:
                            _walk(json.loads(line), '', hits)
            else:
                continue
            for p, n in hits.items():
                key = (fname, p)
                hits_by_file.setdefault(key, {'n': 0, 'files': 0})
                hits_by_file[key]['n'] += n
                hits_by_file[key]['files'] += 1
    found = [{'file': f, 'json_path': p, 'n_values': v['n'], 'n_files': v['files'],
              'writer': 'p515_s44_campaign_harness.py' if f in HARNESS_WRITTEN else
              'p515_g_g1_g4_admm_gates.py (default=str writers; not the campaign harness)'}
             for (f, p), v in sorted(hits_by_file.items())]
    harness_found = [x for x in found if x['writer'] == 'p515_s44_campaign_harness.py']
    checks = {
        'every_artifact_hook_is__json_default_except_hash_inputs': all(
            h['hook'] == '_json_default' for h in hooks if h['line'] not in _hash_hook_lines(new_src)),
        'no_literal_bool_strings_outside_docstrings': literal_bool_strings == [],
        'harness_written_stringified_flags_found_only_gate_pass': [x['json_path'] for x in harness_found] == [
            '.hull_polish_full.gate.pass'],
    }
    return {'checks': checks, 'default_hooks': hooks, 'hash_input_hook_lines_left_as_str': _hash_hook_lines(new_src),
            'literal_bool_string_lines': literal_bool_strings, 'committed_alpha_row_stringified_booleans': found,
            'scanned': f'{ALPHA_ROOT} recursively (all .json / .jsonl), excluding esso_capture/ and planning_read/'}


def _hash_hook_lines(src):
    """The `default=str` hooks deliberately left unchanged: serializations that feed a digest / identity (the frozen
    spec text and the constant-signature), not artifact writers."""
    lines = []
    for i, line in enumerate(src.splitlines(), 1):
        if 'default=str' in line and ('text = json.dumps(spec' in line or 'sig = json.dumps(' in line):
            lines.append(i)
    return lines


# ----------------------------------------------------------------------------------------------------------------------
def main():
    started = time.time()
    if os.path.exists(OUT):
        raise SystemExit(f'REFUSED: output root exists (write-once): {OUT}')
    os.makedirs(OUT)
    new_src = open(H.HARNESS_PATH).read()
    old_src = pre_w74_source()
    sections, errors = {}, {}
    for name, fn in (('V1_evaluation_key', lambda: v1_evaluation_key(old_src, new_src)),
                     ('V2_field_tuples', lambda: v2_field_tuples(old_src)),
                     ('V3_heartbeat_collision', v3_heartbeat),
                     ('V4_boolean_flags', v4_booleans),
                     ('V5_search_scope', lambda: v5_scope(new_src))):
        try:
            sections[name] = fn()
        except Exception as error:  # noqa: BLE001
            errors[name] = f'{type(error).__name__}: {error}\n{traceback.format_exc()}'
            print(f'[W74] {name} ERROR {errors[name]}', flush=True)
    failing = sorted(f'{s}.{k}' for s, v in sections.items() for k, ok in v['checks'].items() if ok is not True)
    guard_failures = GUARD.verify(0)
    payload = {
        'task': 'P5.15 Addendum 44 W74 item 1 -- harness fixes, zero-solve checks', 'utc': _utc(),
        'git_head': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        'harness_sha256': _sha(H.HARNESS_PATH), 'pre_w74_harness_sha256': PRE_W74_HARNESS_SHA256,
        'script_sha256': _sha(os.path.abspath(__file__)), 'sections': sections, 'errors': errors,
        'failing': failing, 'pass': (not failing and not errors and not guard_failures),
        'guard': {'counts': dict(GUARD.counts), 'verify_0_failures': guard_failures},
        'wall_s': time.time() - started}
    H._write_once_json(OUT_JSON, payload)
    for s, v in sections.items():
        for k, ok in v['checks'].items():
            print(f'[W74] {s}: {k}: {"PASS" if ok is True else "FAIL"}', flush=True)
    print(f'[W74] failing={failing} errors={sorted(errors)} guard {dict(GUARD.counts)} verify(0) {guard_failures}',
          flush=True)
    print(f"[W74] {'PASS' if payload['pass'] else 'FAIL'} -> {os.path.relpath(OUT_JSON, REPO)}", flush=True)
    GUARD.uninstall()
    sys.exit(0 if payload['pass'] else 1)


def manifest():
    if os.path.exists(OUT_MANIFEST):
        raise SystemExit(f'REFUSED: {OUT_MANIFEST} exists (write-once)')
    m = {}
    for dirpath, _dirs, files in os.walk(OUT):
        for fname in sorted(files):
            fpath = os.path.join(dirpath, fname)
            m[os.path.relpath(fpath, REPO)] = _sha(fpath)
    for rel in (LAUNCH_LOG_REL, os.path.basename(os.path.abspath(__file__)), 'p515_s44_campaign_harness.py'):
        m[rel] = _sha(rel)
    with open(OUT_MANIFEST, 'x') as handle:
        json.dump(m, handle, indent=1)
    failures = GUARD.verify(0)
    print(f'[W74] wrote {os.path.relpath(OUT_MANIFEST, REPO)}: {len(m)} entries; guard verify(0) {failures}', flush=True)
    GUARD.uninstall()
    sys.exit(0 if not failures else 1)


if __name__ == '__main__':
    if sys.argv[1:] == ['--manifest']:
        manifest()
    else:
        main()
